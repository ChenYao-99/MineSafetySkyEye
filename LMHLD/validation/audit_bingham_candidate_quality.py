"""No-model observation/calibration audit; absence of an accident boundary remains unresolved."""
from pathlib import Path
import json,sys,urllib.parse
import requests
import numpy as np
import rasterio
from PIL import Image,ImageDraw
from mine_data_protocol import DATA,registry,verify_assets,sha
from audit_development_radiometry import parse_xml

ID='US_BC2021';FOLDER=DATA/'reserved_candidates'/ID
OUT=FOLDER/'quality_audit'


def run():
    verify_assets();value=registry();case=value['cases'][ID]
    if case['model_predictions_run']!=0 or case['role']!='reserved_final_test':raise ValueError('Candidate role changed')
    OUT.mkdir(exist_ok=True);session=requests.Session()
    acquisition=json.loads((FOLDER/'acquisition.json').read_text(encoding='utf-8'))
    catalogue=json.loads((FOLDER/'catalogue.json').read_text(encoding='utf-8'))
    result={'case_id':ID,'model_predictions':0,'independent_boundary_available':False,
            'final_quality_accepted':False,'periods':{},'no_independent_ground_reflectance_validation':True}
    panels=[];grids=[]
    for period,item in acquisition['images'].items():
        scene=next(s for s in catalogue['features'] if s['id']==item['scene_id'])
        href=scene['assets']['product-metadata']['href'];xml=OUT/(period+'_product.xml')
        if not xml.exists():
            response=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign?href='+urllib.parse.quote(href,safe=''),timeout=(10,30));response.raise_for_status()
            response=session.get(response.json()['href'],timeout=(10,30));response.raise_for_status()
            if len(response.content)>5*1024*1024:raise ValueError('Oversized product XML')
            parse_xml(response.content);xml.write_bytes(response.content)
        calibration=parse_xml(xml.read_bytes())
        with rasterio.open(FOLDER/item['file']) as f:
            raw=f.read();valid=f.dataset_mask()>0;profile=f.profile;descs=f.descriptions;tags=f.tags()
            grids.append((str(f.crs),tuple(f.transform),f.shape))
        with rasterio.open(FOLDER/item['scl_file']) as f:scl=f.read(1)
        valid &= np.all(raw>0,axis=0)
        obscured=np.isin(scl,[0,1,3,8,9,10,11])
        derivative=OUT/(period+'_BGRN_calibrated_tags.tif')
        if not derivative.exists():
            with rasterio.open(derivative,'w',**profile) as f:
                f.write(raw);f.descriptions=descs
                f.scales=tuple(calibration['scales']);f.offsets=tuple(calibration['reflectance_offsets'])
                f.update_tags(**{**tags,'radiometry_source_xml_sha':sha(xml),'original_sha':sha(FOLDER/item['file']),
                                 'radiometry_status':'actual_product_XML_coefficients','role':'reserved_quality_derivative_no_predictions'})
        with rasterio.open(derivative) as f:
            if not np.array_equal(f.read(),raw):raise ValueError('Calibrated derivative changed raw samples')
        reflectance=raw.transpose(1,2,0)*np.asarray(calibration['scales'])+np.asarray(calibration['reflectance_offsets'])
        rgb=np.maximum(reflectance[:,:,[2,1,0]],0)
        gain=np.percentile(rgb[valid],98);rgb=np.clip(rgb/max(float(gain),1e-6),0,1)
        rendered=Image.fromarray((rgb*255).astype(np.uint8)).convert('RGB')
        header=Image.new('RGB',(rendered.width,rendered.height+38),'white');header.paste(rendered,(0,38))
        ImageDraw.Draw(header).text((8,8),f'{period} {item["datetime"][:10]} | RGB display only, no model',fill='black')
        panels.append(header)
        result['periods'][period]={'acquisition':item['datetime'],'original_sha256':sha(FOLDER/item['file']),
            'XML_sha256':sha(xml),'calibration':calibration,'calibrated_derivative_sha256':sha(derivative),
            'valid_fraction':float(valid.mean()),'SCL_obscured_fraction':float((obscured&valid).sum()/max(valid.sum(),1)),
            'SCL_counts':{str(i):int((scl==i).sum()) for i in range(12)},
            'local_accident_visibility_not_verified_without_boundary':True}
    result['matching_native_crs_affine_shape']=grids[0]==grids[1]
    canvas=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white')
    left=0
    for p in panels:canvas.paste(p,(left,0));left+=p.width
    canvas.save(OUT/'pre_post_unannotated.png')
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    verify_assets();print(json.dumps(result))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
