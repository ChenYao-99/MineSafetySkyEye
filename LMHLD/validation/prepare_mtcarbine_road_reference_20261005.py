"""Original-pair review of constructed roads; weak appearance, never safety truth."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from PIL import Image,ImageDraw
from model_registry import file_sha256
DATA=ROOT/'datasets/tropical_work_pairs_20261005/AU_MTCARBINE_SLATS2324'
OUT=DATA/'constructed_road_reference_v1'
# Pixel-edge polygons selected from the original calibrated pair, with no
# detector output opened for this mine. Water and uncertain vegetation excluded.
REGIONS={
 'pit_to_lower_working_area_access':[[175,166],[183,170],[182,190],[183,205],[201,215],[198,222],[178,210],[174,190]],
 'eastern_constructed_haul_curve':[[218,215],[235,218],[250,231],[261,250],[263,257],[257,261],[254,242],[243,230],[231,225],[218,222]],
 'central_constructed_access_strip':[[148,231],[157,235],[157,255],[151,267],[144,266],[149,251]],
 'western_pit_rim_road':[[131,134],[138,131],[137,147],[141,159],[150,169],[148,177],[139,169],[131,153]],
}

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    OUT.mkdir(exist_ok=True);annual=load(DATA/'work_reference_v1_SLATS/reference.json')
    assert annual['stable_ground_QA_not_ground_displacement']['raw_pair_gate_pass'] and annual['common_visible_fraction']>=.6
    acquisition=load(DATA/'acquisition.json');images=[];visible=[];display=[]
    for period in ('pre','post'):
        p=DATA/(period+'_BGRN_10m.tif');assert file_sha256(p)==acquisition['images'][period]['sha256']
        with rasterio.open(p) as src:
            raw=src.read().transpose(1,2,0).astype(np.float32);valid=np.all(src.read_masks()>0,axis=0)&np.all(raw>0,axis=2)
            image=raw*np.asarray(src.scales,np.float32)+np.asarray(src.offsets,np.float32);profile=src.profile.copy()
        with rasterio.open(DATA/(period+'_SCL_10m.tif')) as src:valid&=np.isin(src.read(1),[2,4,5])
        images.append(image);visible.append(valid)
        rgb=np.uint8(255*np.power(np.clip(image[:,:,[2,1,0]]/.45,0,1),.65));rgb[~valid]=[80,0,100];display.append(rgb)
    def ndvi(image):
        denominator=image[:,:,3]+image[:,:,2]
        return np.divide(image[:,:,3]-image[:,:,2],denominator,out=np.zeros_like(denominator),where=np.abs(denominator)>1e-6)
    a,b=map(ndvi,images);usable=visible[0]&visible[1]&(a<.25)&(b<.25)&(np.abs(a-b)<=.15)
    mask=np.zeros(a.shape,bool);rows=[]
    for name,points in REGIONS.items():
        area=np.zeros(a.shape,np.uint8);cv2.fillPoly(area,[np.asarray(points,np.int32)],1)
        core=(cv2.erode(area,np.ones((3,3),np.uint8))>0)&usable
        if core.sum()<20:core[:]=False
        mask|=core;rows.append({'id':name,'pixel_edge_polygon':points,'accepted_pixels':int(core.sum())})
    labels=np.full(a.shape,255,np.uint8);labels[mask]=0;confidence=np.zeros(a.shape,np.float32);confidence[mask]=.25
    for name,values,dtype in [('labels.tif',labels,'uint8'),('confidence.tif',confidence,'float32'),('normal_mask.tif',mask.astype(np.uint8),'uint8')]:
        path=OUT/name
        if path.exists():
            with rasterio.open(path) as src:assert np.array_equal(src.read(1),values)
        else:
            with rasterio.open(path,'w',**{**profile,'count':1,'dtype':dtype,'nodata':None}) as dst:dst.write(values,1)
    height,width=a.shape;canvas=Image.new('RGB',(width*2,height+32),'white')
    for index,period in enumerate(('pre','post')):
        rgb=display[index].copy();rgb[mask]=np.uint8(.6*rgb[mask]+.4*np.array([0,240,100]))
        for number,row in enumerate(rows):
            points=np.asarray(row['pixel_edge_polygon'],np.int32);cv2.polylines(rgb,[points],True,(255,200,0),1)
            cv2.putText(rgb,str(number+1),tuple(points.mean(0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        canvas.paste(Image.fromarray(rgb),(index*width,32));ImageDraw.Draw(canvas).text((index*width+5,5),period+' | green=weak persistent constructed appearance',fill='black')
    canvas.save(OUT/'native_road_review.png')
    receipt={'prepared_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'status':'draft_pending_original_image_visual_acceptance',
             'case_id':'AU_MTCARBINE_SLATS2324','mine_group':'MtCarbine','role':'auxiliary_development_train','regions':rows,
             'weak_constructed_appearance_pixels':int(mask.sum()),'confidence':.25,'outside_selected_observed_cores_unknown':True,
             'source_rule':'project review of persistent constructed road/bench appearance on original pair;official mining location and operator July2024 mine layout only corroborate mine context',
             'source_context':'https://www.eqresources.com.au/annualreport/2024/22/',
             'period_contains_cyclone_and_flooding':True,'certified_normal_year_or_accident_absence_truth':False,
             'actual_dates':annual['actual_dates'],'classifier_predictions_used_for_annotation':0,'independent_expert_review':False,
             'original_inputs_sha256':{period:acquisition['images'][period]['sha256'] for period in ('pre','post')},
             'frozen_assets':{name:file_sha256(OUT/name) for name in ['labels.tif','confidence.tif','normal_mask.tif','native_road_review.png']}}
    path=OUT/'reference.json'
    if path.exists():
        old=load(path);assert all(old.get(k)==v for k,v in receipt.items() if k!='prepared_at')
    else:write(path,receipt)
    print('MtCarbine constructed weak reference',int(mask.sum()),rows,flush=True)

if __name__=='__main__':run()
