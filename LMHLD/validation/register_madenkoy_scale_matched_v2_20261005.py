"""Common physical resolution on an unannotated original aerial background."""
from pathlib import Path
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,cv2
from PIL import Image
from model_registry import file_sha256

def prepare():
    folder=ROOT/'datasets/madenkoy_development_20261005';p=folder/'carla2018_original_public_fig3.jpg'
    source=np.asarray(Image.open(p).convert('RGB'))[:719].copy();target=folder/'carla2018_original_pre_aerial_panel.png'
    if target.exists():assert np.array_equal(np.asarray(Image.open(target)),source)
    else:Image.fromarray(source).save(target)
    red=((source[:,:,0]>190)&(source[:,:,1]<80)&(source[:,:,2]<100)).astype(np.uint8)
    contours,_=cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);contour=max(contours,key=cv2.contourArea)
    assert cv2.contourArea(contour)>100000,'Original red field boundary must be a closed large perimeter'
    polygon=cv2.approxPolyDP(contour,2,True).reshape(-1,2).astype(float)
    base=ROOT/'validation/register_madenkoy_field_outline_20261005.py';text=base.read_text(encoding='utf-8')
    start=text.index('SOURCE=SOURCE_ROOT/');end=text.index('ROI=[',start)
    text=text[:start]+"SOURCE=SOURCE_ROOT/'carla2018_original_pre_aerial_panel.png';OUT=CASE/'field_registration_v2_scale_matched'\nPOLYGON=np.asarray("+repr(polygon.tolist())+",np.float64)\n"+text[end:]
    text=text.replace('1/24.2,0,148','1/14.2,0,149').replace('0,1/24.2,274','0,1/14.2,278')
    text=text.replace("'matcher_only_source_width':1024,'matcher_only_target_width':640", "'matcher_only_common_resolution_m':10,'matcher_only_interpolation_factor':4,'actual_source_scalebar':{'original_x_0m':783,'original_x_100m':925},'only_source_aerial_texture_coarsened_to_actualS2resolution':True")
    text=text.replace("'source_duplicate_radius_original_px':48.4","'source_duplicate_radius_original_px':28.4")
    text=text.replace('source_mask[:410,2200:]=0;source_mask[1380:,2550:]=0;source_mask[1780:,2000:]=0',
                      'source_mask[:85,880:]=0;source_mask[510:,930:]=0;source_mask[650:,750:]=0')
    text=text.replace('np.ones((161,161),np.uint8)','np.ones((93,93),np.uint8)')
    text=text.replace('source_scale=1024/width;source_small=cv2.resize(source,None,fx=source_scale,fy=source_scale,interpolation=cv2.INTER_AREA)',
        'coarse_width=int(round(width/14.2));source_scale=coarse_width*4/width;coarse=cv2.resize(source,(coarse_width,int(round(height*coarse_width/width))),interpolation=cv2.INTER_AREA);source_small=cv2.resize(coarse,(coarse_width*4,coarse.shape[0]*4),interpolation=cv2.INTER_LINEAR)')
    text=text.replace('target_scale=640/(right-left)','target_scale=4')
    text=text.replace("<48.4", "<28.4").replace('point[0]//400','point[0]//250').replace('point[1]//400','point[1]//250')
    text=text.replace('p[0]//400','p[0]//250').replace('p[1]//400','p[1]//250')
    text=text.replace('.025<singular.min() and singular.max()<.07','.05<singular.min() and singular.max()<.1')
    text=text.replace('[(source,p,194),(target,q,8)]','[(source,p,114),(target,q,8)]')
    path=ROOT/'validation/register_madenkoy_field_outline_v2_20261005.py'
    if path.exists():assert path.read_text(encoding='utf-8')==text
    else:path.write_text(text,encoding='utf-8')
    provenance={'parent_code_sha256':file_sha256(base),'new_code_sha256':file_sha256(path),
       'raw_field_fig3_sha256':file_sha256(p),'extracted_original_pre_aerial_panel_sha256':file_sha256(target),
       'source_red_perimeter_area_original_px':float(cv2.contourArea(contour)),
       'changes':'original pre-aerial panel without contour/DEM overlays;coarsen aerial geometrymatchingcopy to10mphysicalscale theninterpolateboth4x;source-specific scalebar normalization;nativeholdout30/median1.5/p903/separation4/sourcehull criteriaunchanged',
       'prior_failed_registration_preserved':True,'detector_or_modelscore_used':False,
       'no_finer_satellite_observation_or_professional_original_GIS_claim':True}
    (folder/'scale_matched_registration_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Fixedphysicalscale source registration v2 prepared;originals and v1failure preserved')

if __name__=='__main__':prepare()
