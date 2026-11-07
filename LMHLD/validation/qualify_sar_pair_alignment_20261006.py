"""Independent held-feature relative SAR alignment; no fitting to accident labels."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from rasterio.warp import reproject,Resampling
from PIL import Image,ImageDraw
from model_registry import file_sha256
DATA=ROOT/'datasets/sar_development_pairs_20261005'
OPTICAL=ROOT/'experiments/cause_terrain_round_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    proposal={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'code_sha256':file_sha256(__file__),
        'cases':['Amyntaio','LasCruces','Hpakant'],'source':'realnoiseandterraincorrectedVV/VHgamma0;historicDSM30m',
        'reference':'independent heldfeature correspondences ofsameorbit radarbefore/after;not fieldsurvey or absolutecurrentpitaccuracy',
        'display_only_filter':'clipdB[-35,5];3x3blur for matching;originalrasters unchanged',
        'matcher':'mutualSIFT ratio.8;confidence not classifier;nativecandidate distance<=6px',
        'candidate_answer_regions_excluded':'all available independent publishedsource/affected/runout masks plusknownpositivecore and150m margin',
        'unique_sites_min':60,'held_sites':30,'fit_inliers_min':30,'heldfit_separation_native_px':2,
        'heldout_raw_grid_displacement_median_max_px':1,'heldout_raw_grid_p90_max_px':2,
        'heldout_after_diagnostic_affine_p90_max_px':1.5,'no_original_data_warp_or_offsetcorrection':True,
        'native_analysis_posting_m':30,'new10m_groundobservations_or_mm_deformation_claim':False,
        'radar_model_admission_requires_currentgeometry_andlayover_shadowreview':True}
    path=DATA/'relative_alignment_protocol.json'
    if path.exists():
        old=load(path);assert all(old.get(k)==v for k,v in proposal.items() if k!='registered_at')
    else:write(path,proposal)
    source_plan=load(OPTICAL/'round_plan.json')['all_sources'];rows=[]
    for case in proposal['cases']:
        folder=DATA/case/'terrain_correction_v1';images=[];supports=[];grids=[]
        for period in ['pre','post']:
            bands=[];valid=None
            for polarization in ['vv','vh']:
                with rasterio.open(folder/(period+'_'+polarization+'_gamma0_30m.tif')) as src:
                    power=src.read(1);present=np.isfinite(power)&(power>0);grids.append((str(src.crs),list(src.transform),src.width,src.height))
                    profile=src.profile.copy()
                valid=present if valid is None else valid&present
                db=np.full(power.shape,-35,np.float32);db[present]=10*np.log10(power[present]);bands.append(db)
            intensity=np.clip((bands[0]+bands[1])*.5,-35,5)
            image=np.uint8((intensity+35)/40*255);image=cv2.GaussianBlur(image,(3,3),0)
            images.append(image);supports.append(valid)
        assert all(g==grids[0] for g in grids)
        exclusion=np.zeros(images[0].shape,np.uint8);dependencies={}
        sources=list(Path(source_plan[case]).glob('published*.tif'))
        sources+=[p for p in Path(source_plan[case]).rglob('published*.tif') if p not in sources]
        for p in sources:
            with rasterio.open(p) as src:
                a=src.read(1);a=((a>0)&(a!=255)).astype(np.uint8);mapped=np.zeros(exclusion.shape,np.uint8)
                reproject(a,mapped,src_transform=src.transform,src_crs=src.crs,dst_transform=profile['transform'],dst_crs=profile['crs'],resampling=Resampling.nearest)
            exclusion|=mapped;dependencies[str(p)]=file_sha256(p)
        optical=Path(source_plan[case])/'post_BGRN_10m.tif'
        positive=np.load(OPTICAL/'inputs'/case/'old_reference.npy')==1
        with rasterio.open(optical) as src:
            a=np.zeros(exclusion.shape,np.uint8);reproject(positive.astype(np.uint8),a,src_transform=src.transform,src_crs=src.crs,
                dst_transform=profile['transform'],dst_crs=profile['crs'],resampling=Resampling.nearest)
        exclusion|=a;exclusion=cv2.dilate(exclusion,np.ones((11,11),np.uint8))>0
        common=supports[0]&supports[1]&~exclusion;mask=np.uint8(common)*255
        sift=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.005);key_a,desc_a=sift.detectAndCompute(images[0],mask);key_b,desc_b=sift.detectAndCompute(images[1],mask)
        pairs=[]
        if desc_a is not None and desc_b is not None:
            matcher=cv2.BFMatcher();reverse=matcher.knnMatch(desc_b,desc_a,k=2)
            backward={a.queryIdx:a.trainIdx for group in reverse if len(group)==2 for a,b in [group] if a.distance<.8*b.distance}
            forward=matcher.knnMatch(desc_a,desc_b,k=2)
            candidates=sorted([a for group in forward if len(group)==2 for a,b in [group] if a.distance<.8*b.distance and backward.get(a.trainIdx)==a.queryIdx],key=lambda m:m.distance)
            for match in candidates:
                a=np.asarray(key_a[match.queryIdx].pt);b=np.asarray(key_b[match.trainIdx].pt)
                if np.linalg.norm(a-b)>6:continue
                if any(np.linalg.norm(a-c)<2 or np.linalg.norm(b-d)<2 for c,d in pairs):continue
                pairs.append((a,b))
        row={'case':case,'unique_candidate_ground_features':len(pairs),'reference_exclusion_sources':dependencies,
             'field_independent_survey':False,'relative_pair_gate_pass':False}
        if len(pairs)>=60:
            a=np.asarray([p[0] for p in pairs],np.float32);b=np.asarray([p[1] for p in pairs],np.float32)
            rng=np.random.default_rng(61006);order=rng.permutation(len(a));held=order[:30]
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
        Image.fromarray(show).save(folder/'relative_pair_exclusion_QA.png')
        rows.append(row);print(case,'relativeSAR unique',len(pairs),'gate',row['relative_pair_gate_pass'],flush=True)
    write(DATA/'relative_alignment_audit.json',{'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'protocol_sha256':file_sha256(path),'rows':rows,'relative_pair_gates_all_pass':all(r['relative_pair_gate_pass'] for r in rows),
        'no_absolute_currentpit_geo_or_field_or_model_acceptance_claim':True,'data_shifted_to_fit_reference':False})

if __name__=='__main__':run()
