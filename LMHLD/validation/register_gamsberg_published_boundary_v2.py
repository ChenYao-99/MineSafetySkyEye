"""Register a published, geographically labelled 2020 accident outline using held-out ground.

This creates a new research-reference version, never survey-grade GIS or full pixel truth.
No accident detector, prediction, old presence bag, or target segmentation enters the fit.
"""
from pathlib import Path
from datetime import datetime
import json,sys
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform as coordinates
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from prepare_winter_constructed_normals import display
SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004/wang_supplement_original_media/image13.png'
CASE=ROOT/'datasets/arid_accident_operation_20261004/ZA_GAMS2020S'
OUT=CASE/'reference_registration_v2_wang2026'
# Centreline of the author's white failure outline in original866x577 FigureS13.
# Digitised solely from the original source, before looking at fitted target polygons.
POLYGON=np.array([[244,292],[253,292],[258,294],[261,298],[266,301],[270,306],
                  [273,310],[265,313],[258,314],[254,309],[252,305],[248,301],
                  [245,299],[243,295]],np.float64)
AXES={'longitude':{'source_x':[220.,454.,685.],'degrees':[18.95,18.975,19.]},
      'latitude':{'source_y':[180.,409.],'degrees':[-(29+14/60+7/3600),-(29+15/60+30/3600)]}}

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def printed_prior(points,frame):
    # Only printed graticule positions, never adjusted from target correspondences.
    lon=np.polyval(np.polyfit(AXES['longitude']['source_x'],AXES['longitude']['degrees'],1),points[:,0])
    lat=np.polyval(np.polyfit(AXES['latitude']['source_y'],AXES['latitude']['degrees'],1),points[:,1])
    x,y=coordinates('EPSG:4326',frame.crs,lon.tolist(),lat.tolist());inv=~frame.transform
    return np.array([inv*(a,b) for a,b in zip(x,y)])-.5

def run():
    OUT.mkdir(exist_ok=True)
    protocol={'created_at':datetime.now().isoformat(timespec='seconds'),'source_sha256':file_sha256(SOURCE),
      'source_url':'https://ars.els-cdn.com/content/image/1-s2.0-S1569843226001330-mmc1.docx',
      'source_figure':'Wang etal2026 supplementary FigureS13; explicit2020Gamsberg heading and S14 dated17November2020',
      'source_base_google_earth_capture_date':None,'event_date':'2020-11-17','polygon_source_xy':POLYGON.tolist(),
      'graticule':AXES,'matching_search_prior_tolerance_px':15,'fit_validation_split_seed':24004,
      'spatial_holdout_selection':'30 sites by deterministic round-robin source spatial cells, before ANY affine/RANSAC fit',
      'fit_holdout_exclusion_radius_native_px':4,'gate':{'unique_sites':60,'fit_inliers':30,'heldout_sites':30,'heldout_median_px_max':1.5,'heldout_p90_px_max':3,'heldout_within3px_fraction_min':.9},
      'detector_or_model_predictions_used':False,'source_is_not_original_survey_GIS':True,
      'all_replicated_paper_outlines_are_one_reference':True,'target_input_resampling':False,
      'no_label_promotion_until_geometric_and_optical_review':True}
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text(encoding='utf-8'))
        if any(old[k]!=v for k,v in protocol.items() if k!='created_at'):raise ValueError('Existing protocol changed; use a new version')
        if (OUT/'registration.json').exists():print('Frozen registration exists; no re-fit');return
    else:write(pp,protocol)
    source=cv2.cvtColor(cv2.imdecode(np.fromfile(SOURCE,np.uint8),1),cv2.COLOR_BGR2RGB)
    # Mine workings, accident body, measurement clusters, legend, north arrow, scale and map borders.
    # Fit/validation use only the unchanged surrounding land and roads outside this mask.
    source_ok=np.zeros(source.shape[:2],np.uint8);source_ok[45:533,45:821]=255
    source_ok[50:347,60:421]=0;source_ok[:110,:85]=0
    source_ok[370:,:180]=0;source_ok[485:,680:]=0
    source_ok[source.min(2)>230]=0
    clahe=cv2.createCLAHE(2,(8,8));sift=cv2.SIFT_create(nfeatures=12000,nOctaveLayers=5,contrastThreshold=.005,edgeThreshold=12,sigma=1.)
    sg=clahe.apply(cv2.cvtColor(source,cv2.COLOR_RGB2GRAY));sk,sd=sift.detectAndCompute(sg,source_ok)
    rows=[]
    for period in ('pre','post'):
        target,tvalid=display(CASE/f'{period}_BGRN_10m.tif')
        tmask=(tvalid.astype(np.uint8)*255);tmask[150:550,100:440]=0
        with rasterio.open(CASE/f'{period}_BGRN_10m.tif') as frame:prior=printed_prior(np.array([k.pt for k in sk]),frame)
        ground=[]
        for target_scale in (1.,1.5):
            enlarged=cv2.resize(target,None,fx=target_scale,fy=target_scale,interpolation=cv2.INTER_CUBIC) if target_scale!=1. else target
            mask=cv2.resize(tmask,(enlarged.shape[1],enlarged.shape[0]),interpolation=cv2.INTER_NEAREST)
            tk,td=sift.detectAndCompute(clahe.apply(cv2.cvtColor(enlarged,cv2.COLOR_RGB2GRAY)),mask)
            if td is None or sd is None:continue
            for pair in cv2.BFMatcher().knnMatch(sd,td,k=2):
                if len(pair)!=2:continue
                m,n=pair
                if m.distance>=.8*n.distance:continue
                a=np.array(sk[m.queryIdx].pt);b=(np.array(tk[m.trainIdx].pt)+.5)/target_scale-.5
                if np.linalg.norm(b-prior[m.queryIdx])>15:continue
                ground.append({'source_xy':a.tolist(),'target_native_xy':b.tolist(),'descriptor_ratio':float(m.distance/max(n.distance,1e-9))})
        unique=[]
        for point in sorted(ground,key=lambda p:p['descriptor_ratio']):
            if any(np.linalg.norm(np.array(point['source_xy'])-q['source_xy'])<5 or np.linalg.norm(np.array(point['target_native_xy'])-q['target_native_xy'])<2 for q in unique):continue
            unique.append(point)
        r={'period':period,'target_sha256':file_sha256(CASE/f'{period}_BGRN_10m.tif'),'unique_ground_sites':len(unique),'points':unique,'registration_gate_pass':False}
        if len(unique)>=60:
            a=np.array([p['source_xy'] for p in unique],np.float32);b=np.array([p['target_native_xy'] for p in unique],np.float32)
            # Spatial cells order the allocation, residuals have not been computed.
            cells={}
            for i,p in enumerate(a):cells.setdefault((int(p[0]//100),int(p[1]//100)),[]).append(i)
            rng=np.random.default_rng(24004);keys=sorted(cells);rng.shuffle(keys)
            for indices in cells.values():rng.shuffle(indices)
            order=[]
            while any(cells.values()):
                for key in keys:
                    if cells[key]:order.append(cells[key].pop())
            held=np.array(order[:30]);fit=np.array([i for i in order[30:] if np.min(np.linalg.norm(b[i]-b[held],axis=1))>=4],int)
            r.update(heldout_indices=held.tolist(),fit_indices=fit.tolist())
            if len(fit)>=30:
                cv2.setRNGSeed(24004);matrix,inliers=cv2.estimateAffine2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
                if matrix is not None:
                    residual=np.linalg.norm(a[held]@matrix[:,:2].T+matrix[:,2]-b[held],axis=1)
                    passed=bool(inliers.sum()>=30 and np.median(residual)<=1.5 and np.percentile(residual,90)<=3 and (residual<=3).mean()>=.9)
                    singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
                    passed=passed and bool(.3<singular.min() and singular.max()<2 and singular.max()/singular.min()<1.5)
                    r.update(source_to_native_affine=matrix.tolist(),fit_inliers=int(inliers.sum()),heldout_errors_px=residual.tolist(),
                        heldout_median_px=float(np.median(residual)),heldout_p90_px=float(np.percentile(residual,90)),heldout_within3px_fraction=float((residual<=3).mean()),
                        affine_singular_values=singular.tolist(),registration_gate_pass=passed)
                    # QA display only. Original images/old all-unknown presence targets stay intact.
                    mapped=POLYGON@matrix[:,:2].T+matrix[:,2]
                    review=target.copy();cv2.polylines(review,[np.rint(mapped).astype(np.int32)],True,(255,70,30),1)
                    cv2.imencode('.png',review)[1].tofile(str(OUT/f'{period}_published_outline_review.png'))
                    r['mapped_polygon_native_xy']=mapped.tolist()
        rows.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('points','heldout_indices','fit_indices','heldout_errors_px','mapped_polygon_native_xy','source_to_native_affine')},ensure_ascii=False),flush=True)
    write(OUT/'registration.json',{'source_sha256':file_sha256(SOURCE),'protocol_sha256':file_sha256(pp),'rows':rows,'both_original_dates_gate_pass':all(r['registration_gate_pass'] for r in rows),'new_model_predictions':0,'expert_GIS':False,'label_promoted':False})

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
