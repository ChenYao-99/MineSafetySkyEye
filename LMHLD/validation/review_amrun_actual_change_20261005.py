"""Original-band visibility and observed-change QA, no classifier output or safety label."""
from pathlib import Path
import argparse, json, sys
import cv2, numpy as np, rasterio
from PIL import Image, ImageDraw
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from model_registry import file_sha256
from prepare_winter_constructed_normals import alignment
DATA=ROOT/'datasets/source_backed_operation_change_20261005'
CASE=DATA/'AU_AMRUN202309';OUT=CASE/'visibility_review_v1';OUT.mkdir(exist_ok=True)

def run():
    acquisition=json.loads((CASE/'acquisition.json').read_text(encoding='utf-8'))
    rgbs=[];ndvis=[];visible=[];profiles=[]
    for period in ('pre','post'):
        row=acquisition['images'][period]
        for name,key in ((f'{period}_BGRN_10m.tif','sha256'),(f'{period}_SCL_10m.tif','scl_sha256'),(f'{period}_product.xml','product_XML_sha256')):
            assert file_sha256(CASE/name)==row[key]
        with rasterio.open(CASE/f'{period}_BGRN_10m.tif') as src:
            raw=src.read().astype(np.float32)
            refl=raw*np.asarray(src.scales,np.float32)[:,None,None]+np.asarray(src.offsets,np.float32)[:,None,None]
            valid=np.all(src.read_masks()>0,axis=0)&np.all(raw>0,axis=0)
            profiles.append((str(src.crs),tuple(src.transform),src.width,src.height))
        with rasterio.open(CASE/f'{period}_SCL_10m.tif') as src:scl=src.read(1)
        valid&=~np.isin(scl,[0,1,3,6,7,8,9,10,11]);visible.append(valid)
        rgb=np.uint8(255*np.power(np.clip(refl[[2,1,0]].transpose(1,2,0)/.45,0,1),.65));rgb[~valid]=[80,0,100]
        rgbs.append(rgb)
        den=refl[3]+refl[2];ndvi=np.divide(refl[3]-refl[2],den,out=np.zeros_like(den),where=np.abs(den)>1e-6)
        ndvis.append(ndvi)
        Image.fromarray(rgb).save(OUT/f'{period}_calibrated_original_RGB.png')
    assert profiles[0]==profiles[1],'Input grids must match'
    common=visible[0]&visible[1]
    loss=common&(ndvis[0]>.45)&(ndvis[1]<.3)&((ndvis[0]-ndvis[1])>.25)
    # QA excludes changed surfaces from ground matches; it is not ground-survey accuracy.
    exclude=~common|(np.abs(ndvis[0]-ndvis[1])>.15)
    try:pair=alignment(rgbs[0],rgbs[1],exclude)
    except ValueError as exc:pair={'raw_pair_gate_pass':False,'reason':str(exc),'raw_images_warped':False}
    h,w=common.shape;canvas=Image.new('RGB',(2*w,h+32),'white');draw=ImageDraw.Draw(canvas)
    for i,(period,rgb) in enumerate(zip(('pre','post'),rgbs)):
        canvas.paste(Image.fromarray(rgb),(i*w,32));draw.text((i*w+5,5),period+' '+acquisition['images'][period]['datetime'][:10]+' | shared reflectance display',fill='black')
    canvas.save(OUT/'original_pair_review.png')
    np.save(OUT/'vegetation_loss_candidate_NOT_CAUSE_TRUTH.npy',loss)
    result={'case_id':CASE.name,'mine_group':'Amrun','actual_dates':{p:acquisition['images'][p]['datetime'] for p in ('pre','post')},
            'input_acquisition_sha256':file_sha256(CASE/'acquisition.json'),
            'identical_native_grid':True,'raw_DN_unchanged':True,'display_stretch_shared_across_epochs':True,
            'common_visible_fraction':float(common.mean()),'pair_alignment_QA_not_ground_survey':pair,
            'SCL_excluded_classes':[0,1,3,6,7,8,9,10,11],
            'observed_vegetation_loss_candidate_pixels':int(loss.sum()),
            'observed_vegetation_loss_NOT_confirmed_operation_or_accident_pixels':True,
            'monthly_audited_total_NOT_expected_downloaded_window_area':True,
            'project_field_trial_or_expert_review':False,'classifier_predictions':0,'normal_or_failure_labels':0,
            'frozen_QA_assets':{name:file_sha256(OUT/name) for name in ('pre_calibrated_original_RGB.png','post_calibrated_original_RGB.png',
                                'original_pair_review.png','vegetation_loss_candidate_NOT_CAUSE_TRUTH.npy')}}
    p=OUT/'review.json'
    if p.exists():
        saved=json.loads(p.read_text(encoding='utf-8'))
        if saved!=result:
            corrected=dict(saved);corrected['case_id']=CASE.name
            if corrected!=result:raise ValueError('Frozen visibility review changed')
            correction={'reason':'correct full-window case ID in derived QA metadata;all inputs,arrays and metrics unchanged',
                        'old_case_id':saved['case_id'],'corrected_case_id':CASE.name,'prior_review_sha256':file_sha256(p)}
            (OUT/'review_before_case_ID_correction.json').write_bytes(p.read_bytes())
            (OUT/'case_ID_correction.json').write_text(json.dumps(correction,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('actual_dates','common_visible_fraction','pair_alignment_QA_not_ground_survey','observed_vegetation_loss_candidate_pixels')},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full-window',action='store_true');args=parser.parse_args()
    if args.full_window:
        DATA=DATA/'full_published_geometry_window_v2';CASE=DATA/'AU_AMRUN202309_FULL'
        OUT=CASE/'visibility_review_v1';OUT.mkdir(exist_ok=True)
    run()
