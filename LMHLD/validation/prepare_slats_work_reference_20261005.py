"""Source polygons + actual observation concurrence, before any detector outputs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from PIL import Image,ImageDraw
from model_registry import file_sha256
from operation_work_reference import source_mine_clearing_geometries,derive_work_core
from prepare_winter_constructed_normals import alignment
DATA=ROOT/'datasets/slats_operation_development_20261005'
SOURCE={'AU_AMRUN_SLATS2324':ROOT/'datasets/actual_operation_evidence_20261005/Amrun/SLATS_official_spatial',
        'AU_CAVAL_REGION_SLATS2324':DATA/'Caval_region_official_SLATS'}

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    parent=load(DATA/'site_plan.json');rows=[]
    for case,source in SOURCE.items():
        folder=DATA/case;out=folder/'work_reference_v1_SLATS';out.mkdir(exist_ok=True)
        acquisition=load(folder/'acquisition.json');ref=source/'255_original_polygons.geojson';collection=load(ref)
        receipt=load(source/'acquisition.json');expected=next(r['original_polygons_sha256'] for r in receipt['records'] if r['layer_id']==255)
        assert file_sha256(ref)==expected
        geometries=source_mine_clearing_geometries(collection);images=[];rgb=[];supports=[];grids=[]
        for period in ('pre','post'):
            row=acquisition['images'][period]
            for name,key in ((period+'_BGRN_10m.tif','sha256'),(period+'_SCL_10m.tif','scl_sha256'),(period+'_product.xml','product_XML_sha256')):assert file_sha256(folder/name)==row[key]
            with rasterio.open(folder/(period+'_BGRN_10m.tif')) as src:
                raw=src.read().transpose(1,2,0).astype(np.float32);valid=np.all(src.read_masks()>0,axis=0)&np.all(raw>0,axis=2)
                image=raw*np.asarray(src.scales,np.float32)+np.asarray(src.offsets,np.float32)
                grids.append((src.crs,src.transform,src.width,src.height));profile=src.profile.copy()
            with rasterio.open(folder/(period+'_SCL_10m.tif')) as src:valid&=~np.isin(src.read(1),[0,1,3,6,7,8,9,10,11])
            images.append(image);supports.append(valid)
            display=np.uint8(255*np.power(np.clip(image[:,:,[2,1,0]]/.45,0,1),.65));display[~valid]=[80,0,100];rgb.append(display)
        assert grids[0]==grids[1]
        common=supports[0]&supports[1]
        area,interior,core,counts=derive_work_core(geometries,grids[0][0],grids[0][1],images[0],images[1],common)
        try:registration=alignment(rgb[0],rgb[1],~common|area)
        except ValueError as exc:registration={'raw_pair_gate_pass':False,'reason':str(exc),'raw_images_warped':False}
        usable=registration['raw_pair_gate_pass'] and float(common.mean())>=.6
        proposed=int(core.sum())
        if not usable:core[:]=False
        y=np.full(core.shape,255,np.uint8);y[core]=0;w=np.zeros(core.shape,np.float32);w[core]=.25
        for name,array,dtype in (('labels.tif',y,'uint8'),('confidence.tif',w,'float32'),('expected_work_mask.tif',core.astype(np.uint8),'uint8')):
            with rasterio.open(out/name,'w',**{**profile,'count':1,'dtype':dtype,'nodata':None}) as dst:dst.write(array,1)
        h,width=core.shape;canvas=Image.new('RGB',(2*width,h+30),'white');draw=ImageDraw.Draw(canvas)
        for i,period in enumerate(('pre','post')):
            im=rgb[i].copy();im[core]=np.uint8(.6*im[core]+.4*np.array([0,240,100]))
            contours,_=cv2.findContours(area.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(255,200,0),1)
            canvas.paste(Image.fromarray(im),(i*width,30));draw.text((i*width+5,5),case+' '+period+' '+acquisition['images'][period]['datetime'][:10]+' | govt annual Mine polygons;green=weakwork core',fill='black')
        canvas.save(out/'source_and_observation_review.png')
        entry={'prepared_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'case_id':case,'mine_group':parent['sites'][case]['mine_group'],
            'status':'draft_pending_original_image_visual_acceptance','source_original_geojson':str(ref),'source_sha256':file_sha256(ref),
            'source_original_mine_object_IDs':[identifier for identifier,_ in geometries],
            'source_nominal_period':parent['source_nominal_period'],'actual_dates':{p:acquisition['images'][p]['datetime'] for p in ('pre','post')},
            'same_native_grid':True,'raw_images_changed':False,'common_visible_fraction':float(common.mean()),
            'stable_ground_QA_not_ground_displacement':registration,'counts':counts,'proposed_before_observation_gate':proposed,
            'expected_work_pixels':int(core.sum()),'unknown_pixels':int((~core).sum()),'confidence':.25,
            'geotechnical_safety_or_no_accident_ground_truth':False,'exact_daily_cause_verified':False,
            'independent_government_annual_mining_class_with_observed_concurrence':True,'field_trial_or_project_expert_endorsement':False,
            'classifier_predictions_used_for_annotation':0,'source_plan_sha256':file_sha256(DATA/'site_plan.json'),
            'frozen_assets':{name:file_sha256(out/name) for name in ('labels.tif','confidence.tif','expected_work_mask.tif','source_and_observation_review.png')}}
        path=out/'reference.json'
        if path.exists():
            saved=load(path)
            if any(saved.get(k)!=v for k,v in entry.items() if k!='prepared_at'):raise ValueError('Frozen expected-work proposal changed')
        else:write(path,entry)
        rows.append(entry);print(case,'workcore',int(core.sum()),'common',round(float(common.mean()),3),'alignment',registration,flush=True)
    write(DATA/'work_reference_drafts.json',{'cases':rows,'classifier_predictions':0,'all_drafts_require_visual_acceptance':True})

if __name__=='__main__':run()
