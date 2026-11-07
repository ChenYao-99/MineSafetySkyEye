"""Bounded recent pre-event Sentinel-2 context; not model evaluation."""
import sys
import json
from urllib.parse import urlencode
from screen_noto import DATA, HERE, ACQ, now, read, write, sha
sys.path.insert(0, 'D:/LMHLD')
sys.path.insert(0, str(HERE.parent / 'phase7_20261015/vendor'))
from shapely.geometry import box, shape
from verified_cog_window import read_native_cog_window
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import rasterio
import numpy as np
from PIL import Image


def read_uint8_scl(asset, window, cache):
    """Use the same verified transport; keep native uint8 SCL unsupported by BGR reader."""
    from verified_cog_window import VerifiedRangeFile
    import math
    import tifffile
    row0,col0,height,width = map(int,[window.row_off,window.col_off,window.height,window.width])
    with VerifiedRangeFile(asset['href'],cache,max_transfer=8*1024**2) as source:
        with tifffile.TiffFile(source) as tf:
            page = tf.pages[0]
            if (not page.is_tiled or page.samplesperpixel != 1 or page.dtype != np.dtype('uint8')
                or [page.imagelength,page.imagewidth] != asset['proj:shape']):
                raise ValueError('Unexpected native SCL layout')
            tags = page.geotiff_tags
            scale,tie = tags['ModelPixelScale'],tags['ModelTiepoint']
            actual = [scale[0],0,tie[3]-tie[0]*scale[0],0,-scale[1],tie[4]+tie[1]*scale[1]]
            if not np.allclose(actual,asset['proj:transform'][:6],rtol=0,atol=1e-8):
                raise ValueError('SCL source transform differs from catalogue')
            if min(row0,col0) < 0 or row0+height>page.imagelength or col0+width>page.imagewidth:
                raise ValueError('SCL window outside image')
            result = np.zeros((height,width),dtype='uint8')
            th,tw = page.tilelength,page.tilewidth
            columns = math.ceil(page.imagewidth/tw)
            for row in range(row0//th,(row0+height-1)//th+1):
                for col in range(col0//tw,(col0+width-1)//tw+1):
                    index = row*columns+col
                    count = page.databytecounts[index]
                    source.seek(page.dataoffsets[index])
                    body = source.read(count)
                    if len(body) != count:
                        raise ValueError('Encoded SCL tile is truncated')
                    decoded,_,_ = page.decode(body,index)
                    if decoded is None:
                        raise ValueError('Missing SCL tile')
                    tile = decoded[0,:,:,0]
                    y0,y1 = max(row0,row*th),min(row0+height,(row+1)*th)
                    x0,x1 = max(col0,col*tw),min(col0+width,(col+1)*tw)
                    result[y0-row0:y1-row0,x0-col0:x1-col0] = tile[y0-row*th:y1-row*th,x0-col*tw:x1-col*tw]
            return result, {'unsigned_source':asset['href'],'source_etag':source.etag,
                    'source_total_bytes':source.size,'verified_range_bytes_transferred_this_call':source.transferred,
                    'cache_manifest':str(source.folder/'manifest.json'),'source_window':[row0,col0,height,width],
                    'dtype':'uint8','resized':False,'unrequested_scene_tiles_downloaded':False}


def main():
    path = HERE / 'recent_pre_satellite_protocol.json'
    if not path.exists():
        write(path, {'registered_at': now(), 'bbox_wgs84': [136.9583,37.3330,136.9721,37.3440],
              'interval': ['2023-10-01','2023-12-31'], 'maximum_search_requests': 1,
              'maximum_search_bytes': 8 * 1024**2, 'maximum_selected_scenes': 1,
              'selection_rule': 'latest complete-coverage scene with tile eo:cloud_cover <=20; tie by id; no model scores',
              'maximum_native_window_pixels': 180 * 180,
              'bands': ['blue','green','red','scl'], 'purpose': 'pre-existing excavation screening only',
              'training_allowed': False, 'model_predictions': 0,
              'range_reader': 'existing verified COG reader; bounded cache, ETag and length checked'})
    plan = read(path)
    name = 'niigyo_recent_pre_search.json'
    receipt_path = DATA / 'receipts' / (name + '.json')
    if receipt_path.exists():
        record = read(receipt_path)
    else:
        params = {'collections':'sentinel-2-l2a','bbox':','.join(map(str,plan['bbox_wgs84'])),
                  'datetime':plan['interval'][0]+'T00:00:00Z/'+plan['interval'][1]+'T23:59:59Z','limit':100}
        record = ACQ.acquire(name,'https://earth-search.aws.element84.com/v1/search?'+urlencode(params),
                             plan['maximum_search_bytes'],direct=True)
    if not record['ok']:
        write(DATA / 'qualification/recent_pre_satellite_result.json',
              {'status':'catalogue_unavailable','receipt':str(receipt_path),'model_predictions':0})
        print('Catalogue unavailable; no unchanged retry', flush=True)
        return
    if sha(record['path']) != record['sha256']:
        raise ValueError('Original recent-pre catalogue changed')
    source = read(record['path'])
    matched = source.get('numberMatched',source.get('context',{}).get('matched'))
    if matched != len(source['features']):
        raise ValueError('Full catalogue page not obtained')
    candidates = [r for r in source['features'] if shape(r['geometry']).covers(box(*plan['bbox_wgs84']))
                  and r['properties']['eo:cloud_cover'] <= 20]
    candidates.sort(key=lambda r:(r['properties']['datetime'],r['id']),reverse=True)
    if not candidates:
        write(DATA / 'qualification/recent_pre_satellite_result.json',
              {'status':'no_eligible_recent_pre_scene','matched':matched,'model_predictions':0})
        print('No eligible recent-pre scene',flush=True)
        return
    scene = candidates[0]
    root = DATA / 'recent_pre_satellite'
    root.mkdir(exist_ok=True)
    selection = root / 'selection.json'
    if selection.exists() and read(selection)['scene_id'] != scene['id']:
        raise ValueError('Frozen scene selection changed')
    if not selection.exists():
        write(selection,{'selected_at':now(),'scene_id':scene['id'],'datetime':scene['properties']['datetime'],
              'tile_cloud_cover':scene['properties']['eo:cloud_cover'],'catalogue_sha256':record['sha256'],
              'rule':plan['selection_rule'],'scene':scene})
    ledger = []
    images = []
    crs = f"EPSG:{scene['properties']['proj:epsg']}"
    for band in plan['bands']:
        asset = scene['assets'][band]
        grid = Affine(*asset['proj:transform'][:6])
        window = from_bounds(*transform_bounds('EPSG:4326',crs,*plan['bbox_wgs84']),transform=grid).round_offsets().round_lengths()
        height,width = int(window.height),int(window.width)
        if height * width > plan['maximum_native_window_pixels']:
            raise ValueError('Pre-event native window exceeds protocol')
        if band == 'scl':
            raw,evidence = read_uint8_scl(asset,window,root/'_range_cache')
        else:
            raw,evidence = read_native_cog_window(asset['href'],int(window.row_off),int(window.col_off),
                  height,width,root/'_range_cache',asset['proj:shape'],asset['proj:transform'])
        target = root / (band+'_provider_native.tif')
        native_grid = grid * Affine.translation(window.col_off,window.row_off)
        with rasterio.open(target,'w',driver='GTiff',height=height,width=width,count=1,
                           crs=crs,transform=native_grid,dtype=raw.dtype,nodata=0,compress='deflate') as dst:
            dst.write(raw,1)
            dst.update_tags(scene_id=scene['id'],capture_datetime=scene['properties']['datetime'],
                            physical_reflectance_interpretation='not used',model_predictions='0')
        ledger.append({'band':band,'path':str(target),'sha256':sha(target),
                       'native_shape':[height,width],'transport':evidence})
        if band != 'scl':
            images.append(raw)
        else:
            scl_counts = {str(k):int(v) for k,v in zip(*np.unique(raw,return_counts=True))}
            clear_fraction = float((~np.isin(raw,[0,1,3,7,8,9,10,11])).mean())
    rgb = np.stack(images[::-1],axis=2).astype(float)
    # Source pixel histogram stretch only, no indices, classification or band warping.
    valid = np.all(rgb>0,axis=2)
    lo,hi = np.percentile(rgb[valid],[2,98])
    preview = np.clip((rgb-lo)/max(hi-lo,1)*255,0,255).astype('uint8')
    Image.fromarray(preview).resize((preview.shape[1]*5,preview.shape[0]*5),Image.Resampling.NEAREST).save(root/'preview_5x_nearest.png')
    write(DATA / 'qualification/recent_pre_satellite_result.json',
          {'status':'native_recent_pre_context_acquired','scene_id':scene['id'],'datetime':scene['properties']['datetime'],
           'source_scl_counts':scl_counts,'native20m_scl_clear_fraction':clear_fraction,
           'images':ledger,'source_pixels_unaltered':True,'model_predictions':0,
           'physical_indices_computed':False,'pre_event_is_not_verified_safe_background':True})
    print(json.dumps({'scene_id':scene['id'],'datetime':scene['properties']['datetime'],
                      'scl_clear_fraction':clear_fraction,'scl_counts':scl_counts}),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
