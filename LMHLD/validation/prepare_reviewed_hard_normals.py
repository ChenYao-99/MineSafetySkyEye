# -*- coding: utf-8 -*-
"""Versioned constructed-surface weak normals; preserve every original reference."""
from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
import rasterio
from mine_data_protocol import DATA,assert_development_cases,verify_assets
from finetune_negatives import norm_patch
from register_las_cruces_boundary import rgb

OUT=DATA/'review_v2_20261003';CASES=('Amyntaio','LasCruces','Hpakant')
OLD=DATA.parents[1]/'experiments/full_resolution_study_20261003_invalid_nan'
PRIOR=DATA.parents[1]/'experiments/full_resolution_study_safe_20261003'
# Project visual review of unannotated pre/post imagery, before any new predictions.
# These are event-background weak references, not a statement of slope safety.
REGIONS={
 'Amyntaio':[
  ('eastern_constructed_bench',[[610,160],[630,145],[655,147],[658,175],[643,196],[619,194]]),
  ('eastern_haul_road',[[610,365],[623,370],[651,349],[673,321],[687,289],[676,284],[662,318],[643,346]]),
  ('plant_access_surface',[[735,280],[770,281],[797,296],[794,313],[759,311],[735,299]]),
  ('eastern_bench_access',[[676,228],[685,233],[670,277],[650,311],[638,306],[659,271]]),
  ('northern_constructed_platform',[[472,78],[506,71],[527,90],[528,107],[504,110],[481,98]])],
 'LasCruces':[
  ('eastern_bench_road',[[286,192],[293,198],[291,230],[285,252],[275,271],[267,268],[279,246],[285,224]]),
  ('southern_bench_access',[[196,270],[218,270],[231,263],[238,269],[226,282],[207,285],[193,279]]),
  ('eastern_haul_access_apron',[[313,207],[330,216],[337,239],[328,261],[308,266],[299,253],[307,235]]),
  ('western_plant_access',[[59,151],[73,152],[79,175],[73,192],[58,188],[54,173]])],
 'Hpakant':[
  ('northeastern_haul_road',[[291,160],[303,171],[314,177],[320,188],[314,198],[307,188],[296,181],[288,170]]),
  ('eastern_constructed_access',[[349,229],[357,232],[350,252],[334,269],[322,269],[328,261],[343,247]]),
  ('southern_access_road',[[256,387],[281,381],[300,369],[307,374],[285,391],[265,398],[257,396]]),
  ('northwestern_access_surface',[[81,160],[100,153],[106,166],[99,178],[84,176]])]}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def prepare():
    assert_development_cases(list(CASES));verify_assets();OUT.mkdir(exist_ok=True);path=OUT/'review_manifest.json'
    if path.exists():
        p=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in p['frozen_outputs'].items():
            if sha(OUT/name)!=digest:raise ValueError('Frozen review-v2 data changed')
        return p
    original=json.loads((PRIOR/'protocol.json').read_text(encoding='utf-8'))
    result={'cases':{},'frozen_outputs':{},'original_assets':{},'project_detector_outputs_used_for_annotation':False,
        'independent_expert_feedback_received':False,'reference_quality':'project-reviewed intact constructed surfaces visible at both dates, weak0.25',
        'normal_is_not_geotechnical_safety':True,'outside_polygons_unknown':True,'native_size':128,
        'original_positive_labels_preserved':True,'original_benchmark_not_replaced':True,
        'source_corroboration':'independent-author optical-change support and qualitativeESA SAR; neither is original expert GIS',
        'held_mines_have_new_normal_annotations_only_for_evaluation_not_training':True}
    for eid in CASES:
        folder=DATA/f'development/{eid}';out=OUT/eid;out.mkdir(exist_ok=True)
        ref=json.loads((folder/'label_reference.json').read_text(encoding='utf-8'))
        for name,digest in ref['frozen_assets'].items():
            if sha(folder/name)!=digest:raise ValueError('Original reference asset changed')
        result['original_assets'][eid]=ref['frozen_assets']
        with rasterio.open(folder/'post_BGRN_10m.tif') as f:x=f.read().astype(np.float32);profile=f.profile.copy();valid=np.all(f.read_masks()>0,axis=0)
        with rasterio.open(folder/'partial_labels.tif') as f:old=f.read(1)
        with rasterio.open(folder/'label_confidence.tif') as f:weight=f.read(1)
        protected=np.zeros(old.shape,np.uint8)
        for name in ('published_debris_reference.tif','published_accident_reference.tif','published_source_reference.tif','published_runout_unknown.tif'):
            if (folder/name).exists():
                with rasterio.open(folder/name) as f:protected|=(f.read(1)>0).astype(np.uint8)
        if eid=='Amyntaio':
            support=OUT/'boundary_audit/amyntaio_external_change_support.tif'
            audit=json.loads((OUT/'boundary_audit/independent_author_registration.json').read_text(encoding='utf-8'))
            if not audit['registration_gate_pass']:raise ValueError('External reference not registered')
            with rasterio.open(support) as f:protected|=(f.read(1)>0).astype(np.uint8)
            result['external_support_sha256']=sha(support)
        #100m around the union of all known affected/source/runout support, not
        # just around the narrow positive core. Old reference scoring is untouched.
        prohibited=cv2.dilate(protected,np.ones((21,21),np.uint8))>0
        for period in ('pre','post'):
            scl=folder/f'{period}_SCL_10m.tif'
            if scl.exists():
                with rasterio.open(scl) as f:valid&=~np.isin(f.read(1),[0,1,3,6,8,9,10,11])
            with rasterio.open(folder/f'{period}_BGRN_10m.tif') as f:valid&=np.all(f.read_masks()>0,axis=0)
        mask=np.zeros(old.shape,bool);regions=[]
        for name,points in REGIONS[eid]:
            candidate=np.zeros(old.shape,np.uint8);cv2.fillPoly(candidate,[np.array(points,np.int32)],1)
            core=cv2.erode(candidate,np.ones((3,3),np.uint8))>0
            accepted=core&valid&~prohibited&(old!=1)
            mask|=accepted
            regions.append({'id':name,'polygon_pixel_xy':points,'requested_pixels':int(core.sum()),'retained_pixels':int(accepted.sum()),
                'excluded_affected_or_buffer':int((core&prohibited).sum()),'exclusion_policy':'all-source union100m,1pixel polygon edge, invalid/cloud/snow/shadow/water unknown',
                'reviewer_type':'project original-pair visual review; no independent expert signoff','status':'accepted_weak_reference' if accepted.sum()>=20 else 'excluded_insufficient_support'})
        if any(r['retained_pixels']<20 for r in regions):
            mask[:]=False
            for r in regions:
                if r['retained_pixels']>=20:
                    candidate=np.zeros(old.shape,np.uint8);cv2.fillPoly(candidate,[np.array(r['polygon_pixel_xy'],np.int32)],1)
                    mask|=(cv2.erode(candidate,np.ones((3,3),np.uint8))>0)&valid&~prohibited&(old!=1)
        if mask.sum()<100:raise ValueError('Too little justified constructed-surface support, leave pending')
        y=old.copy();w=weight.copy();fresh=mask&(old==255);y[fresh]=0;w[fresh]=.25
        assert np.array_equal(y[old==1],old[old==1]) and not np.any(mask&prohibited)
        for name,array,dtype,nodata in (('reviewed_normal_mask.tif',mask.astype(np.uint8),'uint8',None),('labels_v2.tif',y,'uint8',255),('confidence_v2.tif',w,'float32',None)):
            pr=profile.copy();pr.update(count=1,dtype=dtype,nodata=nodata,compress='deflate')
            with rasterio.open(out/name,'w',**pr) as f:f.write(array,1)
            result['frozen_outputs'][f'{eid}/{name}']=sha(out/name)
        hard=[]
        for b in range(64,y.shape[0]-64,8):
            for a in range(64,y.shape[1]-64,8):
                if mask[b,a] and mask[b-64:b+64,a-64:a+64].sum()>=20:hard.append({'xy':[a,b],'kind':'hard_normal'})
        if not hard:raise ValueError('No128px hard-normal centre with genuine context')
        centres=original['cases'][eid]+hard
        xx=[];yy=[];ww=[]
        for c in centres:
            a,b=c['xy'];sl=np.s_[b-64:b+64,a-64:a+64];xx.append(x[:,sl[0],sl[1]]);yy.append(y[sl]);ww.append(w[sl])
        for name,array in (('x',norm_patch(np.stack(xx))),('y',np.stack(yy)),('w',np.stack(ww))):
            p=out/(name+'.npy');np.save(p,array);result['frozen_outputs'][f'{eid}/{name}.npy']=sha(p)
        result['cases'][eid]={'regions':regions,'centres':centres,'original_centres_count':len(original['cases'][eid]),'hard_normal_centres':len(hard),
            'reviewed_normal_pixels':int(mask.sum()),'newly_supervised_pixels':int(fresh.sum()),'positive_pixels_unchanged':int((old==1).sum()),
            'weak_normal_confidence':.25,'dates':[json.loads((folder/'acquisition.json').read_text(encoding='utf-8'))['images'][p]['datetime'] for p in ('pre','post')]}
        pre,post=rgb(folder/'pre_BGRN_10m.tif'),rgb(folder/'post_BGRN_10m.tif')
        for image in (pre,post):
            for r in regions:cv2.polylines(image,[np.array(r['polygon_pixel_xy'],np.int32)],True,(40,220,240),1)
            image[mask]=(.65*image[mask]+.35*np.array([240,160,30])).astype(np.uint8)
        cv2.imencode('.png',np.hstack([pre,post]))[1].tofile(str(out/'normal_surface_review.png'))
        print(eid,'normal pixels',mask.sum(),'new',fresh.sum(),'hard centres',len(hard),flush=True)
    result['normal_annotations_are_not_new_independent_test']=True;write(path,result);return result


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');prepare()
