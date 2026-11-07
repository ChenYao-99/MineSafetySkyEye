"""Conservative source-supported mining-work references, before any predictions."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, json, sys
ROOT = Path('D:/LMHLD')
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/'validation'))
sys.path.insert(0, str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2, numpy as np, rasterio
from PIL import Image, ImageDraw
from operation_work_reference import source_mine_clearing_geometries, derive_work_core
from prepare_winter_constructed_normals import alignment
from model_registry import file_sha256
DATA = ROOT/'datasets/tropical_work_pairs_20261005'

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run(data=DATA):
    plan=load(data/'site_plan.json');rows=[]
    for case,site in plan['sites'].items():
        folder=data/case;acqpath=folder/'acquisition.json'
        if not acqpath.exists() or len(load(acqpath)['images'])!=2:
            print(case,'acquisition pending',flush=True);continue
        out=folder/'work_reference_v1_SLATS';out.mkdir(exist_ok=True)
        acquisition=load(acqpath);source=folder/'official_mining_change_subset.geojson'
        assert file_sha256(source)==site['source_subset_sha256']
        geometries=source_mine_clearing_geometries(load(source));images=[];supports=[];displays=[];grids=[]
        for period in ('pre','post'):
            entry=acquisition['images'][period]
            for name,field in [(period+'_BGRN_10m.tif','sha256'),(period+'_SCL_10m.tif','scl_sha256'),(period+'_product.xml','product_XML_sha256')]:
                assert file_sha256(folder/name)==entry[field]
            with rasterio.open(folder/(period+'_BGRN_10m.tif')) as src:
                raw=src.read().transpose(1,2,0).astype(np.float32)
                observed=np.all(src.read_masks()>0,axis=0)&np.all(raw>0,axis=2)
                image=raw*np.asarray(src.scales,np.float32)+np.asarray(src.offsets,np.float32)
                grids.append((src.crs,src.transform,src.height,src.width));profile=src.profile.copy()
            with rasterio.open(folder/(period+'_SCL_10m.tif')) as src:
                observed&=~np.isin(src.read(1),[0,1,3,6,7,8,9,10,11])
            rgb=np.uint8(255*np.power(np.clip(image[:,:,[2,1,0]]/.45,0,1),.65));rgb[~observed]=[80,0,100]
            images.append(image);supports.append(observed);displays.append(rgb)
        assert grids[0]==grids[1]
        common=supports[0]&supports[1]
        area,interior,core,counts=derive_work_core(geometries,grids[0][0],grids[0][1],images[0],images[1],common)
        try:pair=alignment(displays[0],displays[1],~common|area)
        except ValueError as exc:pair={'raw_pair_gate_pass':False,'reason':str(exc),'raw_images_warped':False}
        proposed=int(core.sum());usable=pair['raw_pair_gate_pass'] and float(common.mean())>=.6
        if not usable:core[:]=False
        labels=np.full(core.shape,255,np.uint8);labels[core]=0
        weight=np.zeros(core.shape,np.float32);weight[core]=.25
        for name,values,dtype in [('labels.tif',labels,'uint8'),('confidence.tif',weight,'float32'),('expected_work_mask.tif',core.astype(np.uint8),'uint8'),('government_area_mask.tif',area.astype(np.uint8),'uint8')]:
            path=out/name
            if path.exists():
                with rasterio.open(path) as src:assert np.array_equal(src.read(1),values)
            else:
                with rasterio.open(path,'w',**{**profile,'count':1,'dtype':dtype,'nodata':None}) as dst:dst.write(values,1)
        height,width=core.shape;canvas=Image.new('RGB',(width*2,height+36),'white');original=canvas.copy()
        for index,period in enumerate(('pre','post')):
            title=case+' '+period+' '+acquisition['images'][period]['datetime'][:10]+' | yellow=govt mine geometry;green=post weakwork'
            rgb=displays[index].copy();original.paste(Image.fromarray(rgb),(index*width,36))
            ImageDraw.Draw(original).text((index*width+5,5),title,fill='black')
            rgb[core]=np.uint8(.6*rgb[core]+.4*np.array([0,240,100]))
            contours,_=cv2.findContours(area.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,(255,200,0),1)
            for x in range(0,width,100):cv2.putText(rgb,str(x),(x+2,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            for y in range(100,height,100):cv2.putText(rgb,str(y),(2,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            canvas.paste(Image.fromarray(rgb),(index*width,36));ImageDraw.Draw(canvas).text((index*width+5,5),title,fill='black')
        # Pre overlay only locates the future post reference; it does not label pre.
        canvas.save(out/'source_and_observation_review.png');original.save(out/'original_native_pair_preview.png')
        entry={'prepared_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'case_id':case,'mine_group':site['mine_group'],'role':site['role'],
               'status':'draft_pending_original_image_visual_acceptance','source_original_geojson':str(source),'source_sha256':file_sha256(source),
               'source_original_mine_object_IDs':[identifier for identifier,_ in geometries],'source_nominal_period':plan['source_nominal_period'],
               'actual_dates':{period:acquisition['images'][period]['datetime'] for period in ('pre','post')},'same_native_grid':True,'raw_images_changed':False,
               'common_visible_fraction':float(common.mean()),'stable_ground_QA_not_ground_displacement':pair,'counts':counts,
               'proposed_before_observation_gate':proposed,'expected_work_pixels':int(core.sum()),'unknown_pixels':int((~core).sum()),'confidence':.25,
               'geotechnical_safety_or_no_accident_ground_truth':False,'exact_daily_cause_verified':False,'field_trial_or_project_expert_endorsement':False,
               'classifier_predictions_used_for_annotation':0,'source_plan_sha256':file_sha256(data/'site_plan.json'),
               'frozen_assets':{name:file_sha256(out/name) for name in ['labels.tif','confidence.tif','expected_work_mask.tif','government_area_mask.tif','source_and_observation_review.png','original_native_pair_preview.png']}}
        path=out/'reference.json'
        if path.exists():
            old=load(path);assert all(old.get(key)==value for key,value in entry.items() if key!='prepared_at')
        else:write(path,entry)
        rows.append(entry);print(case,'core',int(core.sum()),'common',round(float(common.mean()),3),'pairpass',pair['raw_pair_gate_pass'],flush=True)
    write(data/'work_reference_drafts.json',{'cases':rows,'classifier_predictions':0})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DATA);run(parser.parse_args().data)
