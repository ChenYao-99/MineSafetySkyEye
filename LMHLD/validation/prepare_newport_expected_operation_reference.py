"""Source-backed expected-work appearance, never a site-wide safety negative."""
from pathlib import Path
from datetime import datetime
import json,sys
import cv2,numpy as np,rasterio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from prepare_winter_constructed_normals import display,alignment
DATA=ROOT/'datasets/arid_accident_operation_20261004';SITE=DATA/'UK_NEWPORT2025';OUT=SITE/'weak_expected_operation_v1'
REGIONS=[('central_constructed_work_floor',[[144,165],[157,163],[168,168],[165,178],[149,181],[140,176]]),
         ('eastern_working_platform',[[170,159],[179,160],[182,173],[176,179],[168,174]]),
         ('western_constructed_access',[[130,167],[137,165],[141,175],[135,180],[128,177]])]


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def prepare():
    OUT.mkdir(exist_ok=True);path=OUT/'reference.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    acquisition=json.loads((SITE/'acquisition.json').read_text(encoding='utf-8'));images=[];visible=None
    for period in ('pre','post'):
        for name,digest in acquisition['images'][period]['assets_sha256'].items():
            if file_sha256(SITE/name)!=digest:raise ValueError('Original operation data changed')
        image,valid=display(SITE/f'{period}_BGRN_10m.tif');images.append(image)
        with rasterio.open(SITE/f'{period}_SCL_10m.tif') as f:scl=f.read(1)
        seen=valid&~np.isin(scl,[0,1,3,6,8,9,10,11]);visible=seen if visible is None else visible&seen
    proposal=np.zeros(visible.shape,np.uint8)
    for name,points in REGIONS:cv2.fillPoly(proposal,[np.asarray(points,np.int32)],1)
    try:pair=alignment(images[0],images[1],proposal.astype(bool)|~visible)
    except ValueError as exc:pair={'raw_pair_gate_pass':False,'reason':str(exc)}
    core=cv2.erode(proposal,np.ones((3,3),np.uint8)).astype(bool)&visible
    if not pair['raw_pair_gate_pass']:core[:]=False
    y=np.full(core.shape,255,np.uint8);y[core]=0;w=np.zeros(core.shape,np.float32);w[core]=.25
    with rasterio.open(SITE/'post_BGRN_10m.tif') as f:profile=f.profile.copy()
    for name,array,dtype in (('labels.tif',y,'uint8'),('confidence.tif',w,'float32'),('normal_mask.tif',core.astype(np.uint8),'uint8')):
        with rasterio.open(OUT/name,'w',**{**profile,'count':1,'dtype':dtype,'nodata':None}) as f:f.write(array,1)
    h,width=core.shape;canvas=Image.new('RGB',(width*2,h+30),'white');draw=ImageDraw.Draw(canvas)
    for i,image in enumerate(images):
        overlay=image.astype(float);overlay[core]=.7*overlay[core]+.3*np.array([40,210,80]);overlay=np.uint8(np.clip(overlay,0,255))
        for j,(name,points) in enumerate(REGIONS):
            pts=np.asarray(points,np.int32);cv2.polylines(overlay,[pts],True,(255,60,20),1);cv2.putText(overlay,str(j+1),tuple(pts.mean(0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        canvas.paste(Image.fromarray(overlay),(width*i,30));draw.text((width*i+5,5),('pre' if i==0 else 'post')+' | visible expected-work surface, safety unknown',fill='black')
    canvas.save(OUT/'review_overlay.png')
    source=DATA/'source_review/newport_independent_inspection_20250307.pdf'
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'status':'draft_pending_visual_acceptance','case_id':'UK_NEWPORT2025','mine_group':'UK_NEWPORT_CHALK',
       'role':'auxiliary_development','independent_source_type':'physical regulator site inspection, actual operations and phasing described',
       'independent_source_sha256':file_sha256(source),'inspection_date':'2025-03-07','source_to_post_date_gap_days':1,
       'source_confirms_actual_execution_not_only_permission':True,'source_geotechnical_safety_certificate':False,
       'label_type':'project-reviewed expected constructed-work appearance confidence0.25; documented activity supports cause at site level',
       'exact_phases_or_working_faces_not_claimed_from_point_source':True,'outside_selected_surfaces_unknown':True,'no_hidden_instability_absence_claim':True,
       'model_predictions_used_for_selection':False,'known_normal_pixels':int(core.sum()),'confidence':.25,'pair_alignment':pair,
       'polygons':[{'id':n,'pixel_polygon':points} for n,points in REGIONS],
       'original_acquisition_sha256':file_sha256(SITE/'acquisition.json'),
       'assets_sha256':{a.name:file_sha256(a) for a in (OUT/'labels.tif',OUT/'confidence.tif',OUT/'normal_mask.tif',OUT/'review_overlay.png')}}
    write(path,p);print(json.dumps({'pixels':p['known_normal_pixels'],'pair':pair}));return p


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');prepare()
