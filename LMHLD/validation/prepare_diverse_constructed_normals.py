"""Project weak appearance reference, selected on original pairs before candidate predictions."""
from pathlib import Path
from datetime import datetime
import json,sys
import cv2
import numpy as np
import rasterio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from prepare_winter_constructed_normals import display,alignment
DATA=ROOT/'datasets/diverse_operation_pairs_20261004';OUT=DATA/'review_v1_20261004'
REGIONS={
 'Morenci':[
  ('western_haul_bench',[[50,88],[57,84],[67,124],[70,160],[79,190],[90,203],[89,213],[72,200],[61,171],[58,137]]),
  ('central_processing_access',[[298,231],[310,223],[329,236],[345,253],[338,260],[321,247],[307,235]]),
  ('southern_constructed_access',[[215,371],[233,374],[248,388],[247,398],[232,389],[213,382]])],
 'Amrun':[
  ('diagonal_haul_route',[[71,127],[82,129],[105,155],[131,180],[170,203],[211,219],[271,227],[281,234],[265,240],[212,230],[171,214],[133,193],[102,168],[74,140]]),
  ('north_haul_strip',[[124,45],[175,29],[229,15],[232,23],[175,39],[127,55]]),
  ('eastern_flat_access',[[271,31],[287,34],[289,68],[285,96],[277,96],[279,66]])],
 'Carajas':[
  ('central_main_haul_bench',[[180,185],[194,173],[201,184],[196,218],[189,241],[181,248],[177,235],[186,213]]),
  ('upper_platform_access',[[149,71],[159,66],[189,72],[216,65],[235,63],[247,63],[249,69],[232,74],[207,80],[178,82],[155,78]]),
  ('eastern_haul_platform',[[499,188],[510,184],[522,196],[529,219],[519,223],[511,205],[501,202]])],
 'Faxe':[
  ('southern_constructed_bench_floor',[[240,205],[267,216],[289,238],[278,247],[253,234],[237,215]]),
  ('western_bench_access',[[223,185],[230,184],[242,200],[241,215],[235,219],[222,204]]),
  ('eastern_constructed_platform',[[300,159],[313,157],[316,175],[303,181],[296,174]])]}


def write(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def prepare():
    OUT.mkdir(exist_ok=True);path=OUT/'review_manifest.json'
    if path.exists():
        p=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in p['frozen_assets'].items():
            if file_sha256(OUT/name)!=digest:raise ValueError('Frozen appearance reference changed')
        return p
    plan=json.loads((DATA/'site_plan.json').read_text(encoding='utf-8'))
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'status':'draft_pending_visual_acceptance',
       'site_plan_sha256':file_sha256(DATA/'site_plan.json'),'label_semantics':'visibly constructed/intact appearance on both dates, confidence0.25; no hidden stability or accident-absence claim',
       'model_predictions_used_for_selection':False,'expert_feedback_received':0,'new_failure_positives':0,
       'outside_selected_surfaces_unknown':True,'training_sites':plan['training_sites'],'normal_holdout_sites':plan['normal_holdout_sites'],
       'sites':{},'frozen_assets':{}}
    for site,regions in REGIONS.items():
        folder=DATA/site;dest=OUT/site;dest.mkdir(exist_ok=True);images=[];valids=[];scls=[]
        acq=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'))
        for period in ('pre','post'):
            original=folder/f'{period}_BGRN_10m.tif'
            if file_sha256(original)!=acq['images'][period]['sha256']:raise ValueError('Original acquisition changed')
            image,valid=display(original);images.append(image);valids.append(valid)
            with rasterio.open(folder/f'{period}_SCL_10m.tif') as f:scls.append(f.read(1))
        shape=images[0].shape[:2];proposal=np.zeros(shape,np.uint8)
        for name,points in regions:cv2.fillPoly(proposal,[np.asarray(points,np.int32)],1)
        visible=valids[0]&valids[1]
        for scl in scls:visible&=~np.isin(scl,[0,1,3,6,8,9,10,11])
        # Ground-registration evidence excludes both proposed answers and unavailable sensor pixels.
        try:pair=alignment(images[0],images[1],proposal.astype(bool)|~visible)
        except ValueError as exc:pair={'raw_pair_gate_pass':False,'reason':str(exc),'raw_images_warped':False}
        core=cv2.erode(proposal,np.ones((3,3),np.uint8)).astype(bool)&visible
        proposed=int(core.sum())
        if not pair['raw_pair_gate_pass']:core[:]=False
        y=np.full(shape,255,np.uint8);y[core]=0;w=np.zeros(shape,np.float32);w[core]=.25
        with rasterio.open(folder/'post_BGRN_10m.tif') as f:profile=f.profile.copy()
        for name,a,dtype in (('labels.tif',y,'uint8'),('confidence.tif',w,'float32'),('normal_mask.tif',core.astype(np.uint8),'uint8')):
            with rasterio.open(dest/name,'w',**{**profile,'count':1,'dtype':dtype,'nodata':None}) as f:f.write(a,1)
        canvas=Image.new('RGB',(shape[1]*2,shape[0]+34),'white');draw=ImageDraw.Draw(canvas)
        for j,(period,image) in enumerate(zip(('pre','post'),images)):
            overlay=image.astype(float);overlay[core]=.65*overlay[core]+.35*np.array([30,210,80]);overlay=np.uint8(np.clip(overlay,0,255))
            for i,(name,points) in enumerate(regions):
                pts=np.asarray(points,np.int32);cv2.polylines(overlay,[pts],True,(255,60,20),1);cv2.putText(overlay,str(i+1),tuple(pts.mean(0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
            canvas.paste(Image.fromarray(overlay),(j*shape[1],34));draw.text((j*shape[1]+5,5),site+' '+period+' | selected surfaces, not safety labels',fill='black')
        canvas.save(dest/'review_overlay.png')
        p['sites'][site]={'role':plan['sites'][site]['role'],'commodity':plan['sites'][site]['commodity'],
           'polygons':[{'id':name,'pixel_polygon':pts} for name,pts in regions],
           'normal_pixels':int(core.sum()),'unknown_pixels':int((y==255).sum()),'confidence':.25,'pair_alignment':pair,
           'proposed_pixels_before_pair_gate':proposed,'SCL_conservatively_excluded_classes':[0,1,3,6,8,9,10,11],
           'annotated_original_sha256':{period:acq['images'][period]['sha256'] for period in ('pre','post')},
           'actual_dates':{period:acq['images'][period]['datetime'] for period in ('pre','post')}}
        for asset in (dest/'labels.tif',dest/'confidence.tif',dest/'normal_mask.tif',dest/'review_overlay.png'):
            p['frozen_assets'][asset.relative_to(OUT).as_posix()]=file_sha256(asset)
    write(path,p);return p


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');p=prepare();print(json.dumps({s:{'pixels':r['normal_pixels'],'pair':r['pair_alignment']} for s,r in p['sites'].items()}))
