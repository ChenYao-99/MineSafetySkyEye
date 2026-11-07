"""Decode only required native TIFF tiles from length-validated small HTTP ranges."""
from pathlib import Path
import hashlib,io,json,math,re
import numpy as np,requests,tifffile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class VerifiedRangeFile(io.RawIOBase):
    def __init__(self,url,cache_root,block_size=131072,max_transfer=64*1024*1024):
        self.url=url;self.block_size=block_size;self.pos=0;self.transferred=0;self.max_transfer=max_transfer
        root=Path(cache_root).resolve()
        if root.drive.upper()!='D:':raise ValueError('Range cache must stay D:')
        self.folder=root/hashlib.sha256(url.encode()).hexdigest();self.folder.mkdir(parents=True,exist_ok=True)
        self.session=requests.Session();self.session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
        r=self.session.get(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},timeout=(10,30));r.raise_for_status()
        match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''))
        if r.status_code!=206 or not match or len(r.content)!=1:raise ValueError('Server did not return requested one-byte metadata range')
        self.size=int(match[1]);self.etag=r.headers.get('ETag');self.transferred=1
        if self.size>1024**3:raise ValueError('Unexpected single-band TIFF size')
        p=self.folder/'manifest.json'
        self.manifest=json.loads(p.read_text()) if p.exists() else {'url':url,'size':self.size,'etag':self.etag,'blocks':{}}
        if self.manifest['url']!=url or self.manifest['size']!=self.size or self.manifest['etag']!=self.etag:raise ValueError('Remote source changed after caching')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if pos<0:raise ValueError('Negative source seek')
        self.pos=pos;return pos
    def _block(self,index):
        start=index*self.block_size;end=min(self.size-1,start+self.block_size-1);path=self.folder/f'{index:08d}.bin';key=str(index)
        if path.exists():
            data=path.read_bytes();record=self.manifest['blocks'].get(key)
            if record is None or len(data)!=end-start+1 or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Cached source range damaged')
            return data
        if self.transferred+end-start+1>self.max_transfer:raise ValueError('Window source transfer budget exceeded')
        for attempt in range(3):
            href=self.url if attempt==0 else self.url+('?range_retry=' if '?' not in self.url else '&range_retry=')+str(attempt)
            r=self.session.get(href,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(10,30));r.raise_for_status()
            data=r.content
            valid=r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}' and len(data)==end-start+1 and r.headers.get('ETag')==self.etag
            if valid:break
            if attempt==2:raise ValueError('Truncated/mismatched source range rejected')
        self.transferred+=len(data);path.write_bytes(data)
        self.manifest['blocks'][key]={'start':start,'end':end,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        (self.folder/'manifest.json').write_text(json.dumps(self.manifest,indent=2)+'\n')
        return data
    def read(self,n=-1):
        if n<0:raise ValueError('Implicit complete scene download is forbidden')
        n=min(n,max(0,self.size-self.pos));pieces=[]
        while n:
            index=self.pos//self.block_size;offset=self.pos%self.block_size;data=self._block(index);take=min(n,len(data)-offset)
            if take<=0:raise ValueError('Invalid source range progress')
            pieces.append(data[offset:offset+take]);self.pos+=take;n-=take
        return b''.join(pieces)
    def close(self):
        if not self.closed:self.session.close()
        super().close()

def decode_window(page,read_at,row_off,col_off,height,width):
    if not page.is_tiled or page.samplesperpixel!=1 or page.dtype!=np.dtype('uint16'):raise ValueError('Only native tiled uint16 single-band inputs supported')
    if min(row_off,col_off,height,width)<0 or row_off+height>page.imagelength or col_off+width>page.imagewidth:raise ValueError('Source window exceeds native image')
    result=np.zeros((height,width),np.uint16);tw=page.tilewidth;th=page.tilelength;cols=math.ceil(page.imagewidth/tw)
    for row in range(row_off//th,(row_off+height-1)//th+1):
        for col in range(col_off//tw,(col_off+width-1)//tw+1):
            index=row*cols+col;count=page.databytecounts[index];data=read_at(page.dataoffsets[index],count)
            if len(data)!=count:raise ValueError('Encoded tile source truncated')
            decoded,_,_=page.decode(data,index)
            if decoded is None:raise ValueError('Missing decoded native tile')
            tile=decoded[0,:,:,0];y0=max(row_off,row*th);y1=min(row_off+height,(row+1)*th);x0=max(col_off,col*tw);x1=min(col_off+width,(col+1)*tw)
            result[y0-row_off:y1-row_off,x0-col_off:x1-col_off]=tile[y0-row*th:y1-row*th,x0-col*tw:x1-col*tw]
    return result

def read_native_cog_window(url,row_off,col_off,height,width,cache_root,expected_shape=None,expected_transform=None):
    with VerifiedRangeFile(url,cache_root) as source:
        with tifffile.TiffFile(source) as tf:
            page=tf.pages[0]
            if expected_shape and (page.imagelength,page.imagewidth)!=tuple(expected_shape):raise ValueError('Native TIFF dimensions differ from provider metadata')
            tags=page.geotiff_tags
            if expected_transform is not None:
                scale=tags.get('ModelPixelScale');tie=tags.get('ModelTiepoint')
                if not scale or not tie:raise ValueError('Native georeferencing tags missing')
                actual=[scale[0],0,tie[3]-tie[0]*scale[0],0,-scale[1],tie[4]+tie[1]*scale[1]]
                if not np.allclose(actual,np.asarray(expected_transform)[:6],rtol=0,atol=1e-8):raise ValueError('TIFF georeferencing differs from provider metadata')
            def read_at(offset,n):source.seek(offset);return source.read(n)
            result=decode_window(page,read_at,row_off,col_off,height,width)
            evidence={'unsigned_source':url,'source_etag':source.etag,'source_total_bytes':source.size,'verified_range_bytes_transferred_this_call':source.transferred,'cache_manifest':str(source.folder/'manifest.json'),'source_window':[row_off,col_off,height,width],
                      'native_shape':[page.imagelength,page.imagewidth],'compression':str(page.compression),'predictor':str(page.predictor),'resized':False,'unrequested_scene_tiles_downloaded':False}
            return result,evidence
