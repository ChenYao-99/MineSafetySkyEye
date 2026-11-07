"""Dense, independently validated ground registration of the external 2020 outline.

LoFTR is a frozen generic photograph matcher, not our mine accident detector.
Unknown pixels and previous weak training data are never promoted by this script.
"""
from pathlib import Path
from datetime import datetime
import json,sys,requests
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio,torch
from PIL import Image,ImageDraw
from kornia.feature import LoFTR
from model_registry import file_sha256
from register_gamsberg_published_boundary_v2 import SOURCE,CASE,POLYGON,write
from rasterio.warp import transform as coordinates
AXES={'longitude':{'source_x':[224.,454.,684.],'degrees':[18.95,18.975,19.]},
      'latitude':{'source_y':[198.,400.],'degrees':[-(29+14/60+17/3600),-(29+15/60+36/3600)]}}
def printed_prior(points,frame):
    lon=np.polyval(np.polyfit(AXES['longitude']['source_x'],AXES['longitude']['degrees'],1),points[:,0])
    lat=np.polyval(np.polyfit(AXES['latitude']['source_y'],AXES['latitude']['degrees'],1),points[:,1])
    x,y=coordinates('EPSG:4326',frame.crs,lon.tolist(),lat.tolist());inverse=~frame.transform
    return np.array([inverse*(a,b) for a,b in zip(x,y)])-.5

from prepare_winter_constructed_normals import display
OUT=CASE/'reference_registration_v5_wang2026_verified_axes'
WEIGHT=ROOT/'cache/reference_registration/loftr_outdoor.ckpt'
WEIGHT_URL='https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt'

def weights():
    WEIGHT.parent.mkdir(parents=True,exist_ok=True)
    if not WEIGHT.exists():
        partial=WEIGHT.with_suffix('.download')
        with requests.get(WEIGHT_URL,timeout=(10,30),stream=True) as r:
            r.raise_for_status();total=0
            with partial.open('wb') as f:
                for block in r.iter_content(1<<20):
                    total+=len(block)
                    if total>100*1024*1024:raise ValueError('Reference matcher exceeds100MiB budget')
                    f.write(block)
        # weights_only rejects arbitrary pickle code before the artifact can become a usable matcher.
        value=torch.load(partial,map_location='cpu',weights_only=True)
        if not isinstance(value.get('state_dict'),dict):raise ValueError('Unexpected official matcher checkpoint')
        partial.replace(WEIGHT)
    value=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    return value['state_dict']

def padded(image,mask):
    gray=cv2.createCLAHE(2,(8,8)).apply(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))
    gray[~mask]=0
    h,w=gray.shape;hh=(h+7)//8*8;ww=(w+7)//8*8
    x=np.zeros((hh,ww),np.float32);x[:h,:w]=gray/255.
    # Kornia0.8.1 resizes the mask internally: nearest interpolation needs float0/1.
    # Its attention converts the product back to bool; excluded support stays exactly0.
    m=np.zeros((hh,ww),np.float32);m[:h,:w]=mask.astype(np.float32)
    return torch.from_numpy(x)[None,None].cuda(),torch.from_numpy(m)[None].cuda()

def sheet(source,target,a,b,path):
    panel=Image.new('RGB',(900,720),'white');draw=ImageDraw.Draw(panel)
    # All30 held-out control sites appear; no residual-based selection or omissions.
    for i,(p,q) in enumerate(zip(a,b)):
        col=i%5;row=i//5;x=col*180;y=row*120
        draw.text((x+5,y+2),str(i+1),fill='black')
        for j,(im,point) in enumerate(((source,p),(target,q))):
            ix,iy=np.rint(point).astype(int);crop=Image.fromarray(im).crop((ix-16,iy-16,ix+17,iy+17)).resize((80,80),Image.Resampling.NEAREST)
            d=ImageDraw.Draw(crop);d.line((35,40,45,40),fill='red');d.line((40,35,40,45),fill='red')
            panel.paste(crop,(x+j*88+2,y+20))
    panel.save(path)

def run():
    OUT.mkdir(exist_ok=True)
    protocol={'created_at':datetime.now().isoformat(timespec='seconds'),'source_sha256':file_sha256(SOURCE),
      'source_url':'https://ars.els-cdn.com/content/image/1-s2.0-S1569843226001330-mmc1.docx',
      'source_figure':'Wang etal2026 FigureS13; original author outline, source event explicitly17November2020 inS14',
      'source_google_earth_capture_date':None,'event_date':'2020-11-17','polygon_source_xy':POLYGON.tolist(),'graticule':AXES,
      'matcher':'Kornia0.8.1 LoFTR pretrained outdoor / MegaDepth, no mining fine-tuning',
      'matcher_weight_url':WEIGHT_URL,'matcher_confidence_min':.5,'geographic_prior_tolerance_px':15,
      'correspondence_dedup_radius_source_px':5,'correspondence_dedup_radius_target_native_px':2,
      'spatial_holdout_seed':24004,'spatial_holdout_sites':30,'fit_holdout_separation_native_px':4,
      'allocation_before_any_fit':True,'gate':{'unique_sites':60,'fit_inliers':30,'heldout_median_px_max':1.5,'heldout_p90_px_max':3,'heldout_within3px_fraction_min':.9},
      'ground_only_mask':'source workings/accident/annotation excluded; target workings[y170:455,x150:450] and Southpit[y435:550,x248:338] excluded; western unchanged terrain retained',
      'images_padded_for_matcher_only_never_resized':True,'raw_or_training_images_changed':False,'mine_detector_predictions_used':False,
      'earlier_failed_v1_and_v2_preserved':True,'survey_GIS_received':False,'unknown_label_policy':'outside accepted conservative core stays255; no automatic safe background'}
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text(encoding='utf-8'))
        if any(old[k]!=v for k,v in protocol.items() if k!='created_at'):raise ValueError('Frozen protocol changed')
        if (OUT/'registration.json').exists():print('Frozen dense registration exists; no refit');return
    else:write(pp,protocol)
    torch.set_num_threads(4);torch.manual_seed(24004)
    model=LoFTR(pretrained=None).eval();model.load_state_dict(weights());model.cuda()
    source=cv2.cvtColor(cv2.imdecode(np.fromfile(SOURCE,np.uint8),1),cv2.COLOR_BGR2RGB)
    ok=np.zeros(source.shape[:2],bool);ok[45:533,45:821]=True
    ok[50:347,105:421]=False;ok[:110,:85]=False;ok[370:,:180]=False;ok[485:,680:]=False
    ok[source.min(2)>230]=False
    x,m=padded(source,ok);rows=[]
    for period in ('pre','post'):
        target,valid=display(CASE/f'{period}_BGRN_10m.tif');tmask=valid.copy();tmask[170:455,150:450]=False; tmask[435:550,248:338]=False
        y,n=padded(target,tmask)
        with torch.inference_mode():output=model({'image0':x,'image1':y,'mask0':m,'mask1':n})
        aa=output['keypoints0'].cpu().numpy();bb=output['keypoints1'].cpu().numpy();cc=output['confidence'].cpu().numpy()
        with rasterio.open(CASE/f'{period}_BGRN_10m.tif') as frame:prior=printed_prior(aa,frame)
        unique=[]
        for i in np.argsort(-cc):
            a,b=aa[i],bb[i]
            sa=np.rint(a).astype(int);tb=np.rint(b).astype(int)
            if cc[i]<.5 or not(0<=sa[0]<ok.shape[1] and 0<=sa[1]<ok.shape[0] and ok[sa[1],sa[0]]):continue
            if not(0<=tb[0]<valid.shape[1] and 0<=tb[1]<valid.shape[0] and tmask[tb[1],tb[0]]):continue
            if np.linalg.norm(b-prior[i])>15:continue
            if any(np.linalg.norm(a-q['source_xy'])<5 or np.linalg.norm(b-q['target_native_xy'])<2 for q in unique):continue
            unique.append({'source_xy':a.tolist(),'target_native_xy':b.tolist(),'matcher_confidence':float(cc[i])})
        r={'period':period,'target_sha256':file_sha256(CASE/f'{period}_BGRN_10m.tif'),'raw_matcher_pairs':len(aa),'unique_ground_sites':len(unique),'points':unique,'registration_gate_pass':False}
        if len(unique)>=60:
            a=np.array([p['source_xy'] for p in unique],np.float32);b=np.array([p['target_native_xy'] for p in unique],np.float32)
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
                    singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
                    hull=cv2.convexHull(a[fit][inliers.ravel()>0]);inside=all(cv2.pointPolygonTest(hull,tuple(p.astype(float)),False)>=0 for p in POLYGON)
                    passed=bool(inliers.sum()>=30 and np.median(residual)<=1.5 and np.percentile(residual,90)<=3 and (residual<=3).mean()>=.9 and .3<singular.min() and singular.max()<2 and singular.max()/singular.min()<1.5 and inside)
                    r.update(source_to_native_affine=matrix.tolist(),fit_inliers=int(inliers.sum()),heldout_errors_px=residual.tolist(),heldout_median_px=float(np.median(residual)),heldout_p90_px=float(np.percentile(residual,90)),heldout_within3px_fraction=float((residual<=3).mean()),polygon_within_fit_ground_convex_hull=inside,affine_singular_values=singular.tolist(),registration_gate_pass=passed)
                    mapped=POLYGON@matrix[:,:2].T+matrix[:,2];r['mapped_polygon_native_xy']=mapped.tolist()
                    review=target.copy();cv2.polylines(review,[np.rint(mapped).astype(np.int32)],True,(255,70,30),1)
                    cv2.imencode('.png',review)[1].tofile(str(OUT/f'{period}_published_outline_review.png'))
                    sheet(source,target,a[held],b[held],OUT/f'{period}_all30_holdout_sites.png')
        rows.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('points','heldout_indices','fit_indices','heldout_errors_px','mapped_polygon_native_xy','source_to_native_affine')},ensure_ascii=False),flush=True)
        del output,y,n;torch.cuda.empty_cache()
    write(OUT/'registration.json',{'created_at':datetime.now().isoformat(timespec='seconds'),'source_sha256':file_sha256(SOURCE),'protocol_sha256':file_sha256(pp),'matcher_weights_sha256':file_sha256(WEIGHT),'rows':rows,'both_original_dates_gate_pass':all(r['registration_gate_pass'] for r in rows),'new_mine_detector_predictions':0,'new_expert_GIS':False,'label_promoted':False})

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
