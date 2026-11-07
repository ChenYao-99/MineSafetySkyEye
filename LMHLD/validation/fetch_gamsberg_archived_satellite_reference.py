"""Archive the originally published dated SkySat reference; does not grant raw imagery licence."""
from pathlib import Path
from datetime import datetime
import concurrent.futures,hashlib,json,re,sys,urllib.request
from PIL import Image
from html import unescape
OUT=Path('D:/LMHLD/datasets/arid_accident_operation_20261004/source_review')
revision=json.loads((OUT/'dated_archive_revision.json').read_text(encoding='utf-8'))
if not revision.get('date_verified'):raise ValueError('Dated archive revision required')
text=(OUT/'dated_archived_gamsberg_article.html').read_text(encoding='utf-8')
clean=unescape(re.sub(r'<[^>]+>','',text));clean=' '.join(clean.split())
if '3rd December' not in clean or 'Planet Labs' not in clean:raise ValueError('Archived article date/provider not verified')
SOURCES={'gamsberg_skysat_full_20201203.jpg':'https://blogs.agu.org/landslideblog/files/2020/12/20_12-gamsberg-3.jpg',
         'gamsberg_skysat_detail_20201203.jpg':'https://blogs.agu.org/landslideblog/files/2020/12/20_12-gamsberg-4.jpg'}


def fetch(item):
    name,url=item;row={'original_url':url,'date_from_original_article':'2020-12-03'}
    dest=OUT/name
    for requested in (url,'https://web.archive.org/web/20201215095110id_/'+url):
        try:
            if not dest.exists():
                with urllib.request.urlopen(urllib.request.Request(requested,headers={'User-Agent':'Mozilla/5.0'}),timeout=25) as r:data=r.read(12*1024*1024+1)
                if len(data)>12*1024*1024 or not data.startswith(b'\xff\xd8'):raise ValueError('Not actual original JPEG')
                dest.write_bytes(data)
            with Image.open(dest) as image:shape=list(image.size);image.verify()
            row.update(retrieved_from=requested,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),pixels_xy=shape,status='downloaded');break
        except Exception as exc:row.setdefault('failed_routes',[]).append({'url':requested,'error':str(exc)})
    return name,row


with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=dict(pool.map(fetch,SOURCES.items()))
p={'downloaded_at':datetime.now().isoformat(timespec='seconds'),'source_article':'https://blogs.agu.org/landslideblog/2020/12/15/the-large-landslide-at-gamsberg-mine-high-resolution-satellite-images/',
   'article_archive_snapshot':revision['archive_timestamp'],'image_archive_route':'20201215095110','capture_date_verified_from_archived_article':'2020-12-03','sources':rows,
   'licence_limit':'Planet Labs copyright, blog used with permission; public reference review is not a licence to redistribute original imagery',
   'independent_field_GIS_received':False,'source_image_never_a_model_input':True,'source_relation':'enlargement and full image are same acquisition, not two independent surveys'}
(OUT/'dated_satellite_reference.json').write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(rows,ensure_ascii=False))
