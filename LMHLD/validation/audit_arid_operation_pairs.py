"""Unannotated source review views and actual weather/calibration statistics."""
from pathlib import Path
import json,sys
import numpy as np
import rasterio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from radiometry import parse_xml
OUT=ROOT/'datasets/arid_operation_pairs_20261004'


def run():
    audit={'sites':{},'model_predictions':0,'normal_or_stability_labels_created':0}
    for site in ('ZhundongSouth','ZhundongMine1'):
        folder=OUT/site;record=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'))
        frames=[];rows={};grids=[]
        for period,item in record['images'].items():
            path=folder/f'{period}_BGRN_10m.tif';sclpath=folder/f'{period}_SCL_10m.tif'
            if file_sha256(path)!=item['sha256'] or file_sha256(sclpath)!=item['scl_sha256']:raise ValueError('Original pair changed')
            coefficients=parse_xml((folder/f'{period}_product.xml').read_bytes())
            with rasterio.open(path) as f:
                raw=f.read().transpose(1,2,0).astype(np.float32);valid=f.dataset_mask()>0
                if not np.allclose(f.scales,coefficients['scales']) or not np.allclose(f.offsets,coefficients['reflectance_offsets']):raise ValueError('XML/header calibration mismatch')
                grids.append((str(f.crs),tuple(f.transform),f.shape))
                reflectance=raw*np.asarray(f.scales)+np.asarray(f.offsets)
            with rasterio.open(sclpath) as f:scl=f.read(1)
            rgb=np.maximum(reflectance[:,:,[2,1,0]],0);gain=np.percentile(rgb[valid],98)
            picture=Image.fromarray((np.clip(rgb/max(gain,1e-6),0,1)*255).astype(np.uint8))
            frame=Image.new('RGB',(picture.width,picture.height+36),'white');frame.paste(picture,(0,36))
            ImageDraw.Draw(frame).text((7,7),period+' '+item['datetime'][:10]+' | no model or labels',fill='black');frames.append(frame)
            rows[period]={'datetime':item['datetime'],'shape':list(raw.shape[:2]),'DN_offset':coefficients['DN_offsets'],
                'SCL_obscured_fraction':float(np.isin(scl,[0,1,3,8,9,10,11]).mean()),
                'SCL_snow_fraction':float((scl==11).mean()),'SCL_counts':{str(i):int((scl==i).sum()) for i in range(12)}}
        canvas=Image.new('RGB',(sum(f.width for f in frames),max(f.height for f in frames)),'white');left=0
        for f in frames:canvas.paste(f,(left,0));left+=f.width
        canvas.save(folder/'unannotated_pre_post_review.png')
        audit['sites'][site]={'periods':rows,'matching_native_grid':grids[0]==grids[1],
                            'constructed_surface_polygons_review_pending':True,'not_new_accident_case':True}
    (OUT/'quality_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
