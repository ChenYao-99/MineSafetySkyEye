"""Repeat unchanged SAR relative gates once on a fixed larger background."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from rasterio.warp import reproject,Resampling
from PIL import Image
from model_registry import file_sha256
DATA=ROOT/'datasets/sar_development_pairs_20261005'
OUT=DATA/'LasCruces/terrain_background_v2'
OPTICAL=ROOT/'experiments/cause_terrain_round_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    previous=load(DATA/'relative_alignment_protocol.json')
    processing=load(OUT/'processing.json')
    for r in processing['rows']:assert file_sha256(r['path'])==r['sha256']
    proposal={k:v for k,v in previous.items() if k not in ['registered_at','code_sha256','cases']}
    proposal.update(registered_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),code_sha256=file_sha256(__file__),cases=['LasCruces'],
        original_gate_protocol_sha256=file_sha256(DATA/'relative_alignment_protocol.json'),
        expanded_background_processing_sha256=file_sha256(OUT/'processing.json'),
        margin_outside_original_optical_AOI_m=900,previous_insufficient_candidates=42,
        independent_new_background_coverage_not_new_mine_or_true_fieldsurvey=True,
        fixed_single_coverage_extension_no_matcher_or_threshold_search=True)
    pp=OUT/'relative_alignment_protocol.json'
    if pp.exists():assert all(load(pp).get(k)==v for k,v in proposal.items() if k!='registered_at')
    else:write(pp,proposal)
    images=[];supports=[];grids=[]
    for period in ['pre','post']:
        bands=[];valid=None
        for polarization in ['vv','vh']:
            with rasterio.open(OUT/(period+'_'+polarization+'_gamma0_30m.tif')) as src:
                power=src.read(1);present=np.isfinite(power)&(power>0);grids.append((str(src.crs),list(src.transform),src.width,src.height));profile=src.profile.copy()
            valid=present if valid is None else valid&present
            db=np.full(power.shape,-35,np.float32);db[present]=10*np.log10(power[present]);bands.append(db)
        image=np.uint8((np.clip((bands[0]+bands[1])*.5,-35,5)+35)/40*255)
        images.append(cv2.GaussianBlur(image,(3,3),0));supports.append(valid)
    assert all(g==grids[0] for g in grids)
    folder=Path(load(OPTICAL/'round_plan.json')['all_sources']['LasCruces'])
    exclusion=np.zeros(images[0].shape,np.uint8);dependencies={}
    for p in sorted(folder.rglob('published*.tif')):
        with rasterio.open(p) as src:
            a=src.read(1);a=((a>0)&(a!=255)).astype(np.uint8);mapped=np.zeros(exclusion.shape,np.uint8)
            reproject(a,mapped,src_transform=src.transform,src_crs=src.crs,dst_transform=profile['transform'],dst_crs=profile['crs'],resampling=Resampling.nearest)
        exclusion|=mapped;dependencies[str(p)]=file_sha256(p)
    core=OPTICAL/'inputs/LasCruces/old_reference.npy';positive=np.load(core)==1
    with rasterio.open(folder/'post_BGRN_10m.tif') as src:
        a=np.zeros(exclusion.shape,np.uint8);reproject(positive.astype(np.uint8),a,src_transform=src.transform,src_crs=src.crs,dst_transform=profile['transform'],dst_crs=profile['crs'],resampling=Resampling.nearest)
    exclusion|=a;dependencies[str(core)]=file_sha256(core)
    exclusion=cv2.dilate(exclusion,np.ones((11,11),np.uint8))>0
    common=supports[0]&supports[1]&~exclusion;mask=np.uint8(common)*255
    sift=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.005);ka,da=sift.detectAndCompute(images[0],mask);kb,db=sift.detectAndCompute(images[1],mask)
    pairs=[]
    if da is not None and db is not None:
        matcher=cv2.BFMatcher();reverse=matcher.knnMatch(db,da,k=2)
        backward={a.queryIdx:a.trainIdx for g in reverse if len(g)==2 for a,b in [g] if a.distance<.8*b.distance}
        forward=matcher.knnMatch(da,db,k=2)
        candidates=sorted([a for g in forward if len(g)==2 for a,b in [g] if a.distance<.8*b.distance and backward.get(a.trainIdx)==a.queryIdx],key=lambda m:m.distance)
        for match in candidates:
            a=np.asarray(ka[match.queryIdx].pt);b=np.asarray(kb[match.trainIdx].pt)
            if np.linalg.norm(a-b)>6:continue
            if any(np.linalg.norm(a-c)<2 or np.linalg.norm(b-d)<2 for c,d in pairs):continue
            pairs.append((a,b))
    row={'case':'LasCruces','unique_candidate_ground_features':len(pairs),'reference_exclusion_sources':dependencies,
        'field_independent_survey':False,'relative_pair_gate_pass':False,'common_background_cells':int(common.sum()),
        'same_original_scene_dates':True,'output_shape':list(common.shape),'matcher_and_acceptance_gates_unchanged':True}
    if len(pairs)>=60:
        a=np.asarray([p[0] for p in pairs],np.float32);b=np.asarray([p[1] for p in pairs],np.float32)
        order=np.random.default_rng(61006).permutation(len(a));held=order[:30]
        fit=np.asarray([i for i in order[30:] if np.min(np.linalg.norm(a[i]-a[held],axis=1))>=2],int)
        row.update(held_indices=held.tolist(),fit_indices=fit.tolist(),before_xy=a.tolist(),after_xy=b.tolist())
        if len(fit)>=30:
            cv2.setRNGSeed(61006);matrix,inliers=cv2.estimateAffinePartial2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=10000)
            if matrix is not None:
                raw=np.linalg.norm(a[held]-b[held],axis=1);after=np.linalg.norm(a[held]@matrix[:,:2].T+matrix[:,2]-b[held],axis=1)
                row.update(raw_held_displacement_median_px=float(np.median(raw)),raw_held_displacement_p90_px=float(np.percentile(raw,90)),
                    diagnostic_fit_held_p90_px=float(np.percentile(after,90)),diagnostic_affine_not_applied=matrix.tolist(),fit_inliers=int(inliers.sum()),
                    relative_pair_gate_pass=bool(inliers.sum()>=30 and np.median(raw)<=1 and np.percentile(raw,90)<=2 and np.percentile(after,90)<=1.5))
    show=np.stack([images[0],images[1],images[1]],2);show[exclusion]=np.uint8(.5*show[exclusion]+.5*np.array([220,0,160]))
    Image.fromarray(show).save(OUT/'relative_pair_exclusion_QA.png')
    write(OUT/'relative_alignment_audit.json',{'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'protocol_sha256':file_sha256(pp),'row':row,
       'no_absolute_currentpit_geo_or_field_or_model_acceptance_claim':True,'data_shifted_to_fit_reference':False,
       'prior_42point_failure_preserved_not_overwritten':True})
    print('LasCruces fixedlargerbackground unique',len(pairs),'relativegate',row['relative_pair_gate_pass'],flush=True)

if __name__=='__main__':run()
