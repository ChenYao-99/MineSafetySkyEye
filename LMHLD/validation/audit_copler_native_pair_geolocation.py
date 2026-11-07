"""Validate original common-grid geolocation on held-out unchanged ground, never warp labels/images."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from PIL import Image,ImageDraw
from model_registry import file_sha256
from mine_data_protocol import registry,assert_mine_group_role,verify_assets
from prepare_winter_constructed_normals import display
CASE=ROOT/'datasets/mine_benchmark_20261002/reserved_candidates/TR_COP2024'
OUT=CASE/'native_pair_geolocation_v1'
REG=CASE/'reference_registration_v2_field2025_map_prior/registration.json'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 value=registry();verify_assets(value);assert_mine_group_role('TR_COP2024','reserved_final_test',value)
 if value['cases']['TR_COP2024']['model_predictions_run']!=0:raise ValueError('Detector has already observed this mine')
 OUT.mkdir(exist_ok=True);pp=OUT/'protocol.json'
 protocol={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'TR_COP2024','source_registration_sha256':file_sha256(REG),
  'source_pre_registration_must_already_pass':True,'source_post_direct_registration_failure_not_changed_or_accepted':True,
  'method':'SIFT descriptor ratio.85 + published same-grid80m searchprior; no RANSAC before holdout allocation',
  'dedup_native_pixels':3,'heldout_sites':30,'heldout_seed':27004,'source_holdout_separation_native_pixels':4,
  'known_mine_and_accident_works_excluded':'source rectangle[x330:970,y325:695] projected by acceptedsourcepre fit, plus100m dilation; common strictSCL weather exclusion',
  'gate':{'unique_sites':90,'fit_inliers':30,'native_unshifted_holdout_median_px_max':1,'native_unshifted_holdout_p90_px_max':2,'native_unshifted_holdout_within2px_min':.9},
  'image_or_label_warping':False,'ground_fit_is_not_failure_displacement':True,'no_mine_detector_predictions':True}
 if pp.exists():
  old=json.loads(pp.read_text(encoding='utf-8'))
  if any(old[k]!=v for k,v in protocol.items() if k!='created_at'):raise ValueError('Native pair protocol changed')
  if (OUT/'audit.json').exists():print('Frozen native pair audit exists; no repeat');return
 else:write(pp,protocol)
 ref=json.loads(REG.read_text(encoding='utf-8'));pre=next(r for r in ref['rows'] if r['period']=='pre')
 if not pre['registration_gate_pass']:raise ValueError('External sourcepre registration not accepted')
 matrix=np.asarray(pre['source_to_native_affine']);corner=np.array([[330,325],[970,325],[970,695],[330,695]])@matrix[:,:2].T+matrix[:,2]
 images={};common=None;metadata=[]
 for period in ('pre','post'):
  images[period],valid=display(CASE/f'{period}_BGRN_10m.tif')
  with rasterio.open(CASE/f'{period}_BGRN_10m.tif') as f:metadata.append((f.crs,f.transform,f.shape))
  with rasterio.open(CASE/f'{period}_SCL_10m.tif') as f:weather=~np.isin(f.read(1),[0,1,3,8,9,10,11])
  common=(valid & weather) if common is None else (common & valid & weather)
 if metadata[0]!=metadata[1]:raise ValueError('Original two-date grid differs')
 exclusion=np.zeros(common.shape,np.uint8);cv2.fillPoly(exclusion,[np.rint(corner).astype(np.int32)],1);exclusion=cv2.dilate(exclusion,np.ones((21,21),np.uint8))
 mask=((common & (exclusion==0)).astype(np.uint8)*255)
 sift=cv2.SIFT_create(nfeatures=12000,nOctaveLayers=5,contrastThreshold=.005,edgeThreshold=12,sigma=1.)
 clahe=cv2.createCLAHE(2,(8,8));sk,sd=sift.detectAndCompute(clahe.apply(cv2.cvtColor(images['pre'],cv2.COLOR_RGB2GRAY)),mask);tk,td=sift.detectAndCompute(clahe.apply(cv2.cvtColor(images['post'],cv2.COLOR_RGB2GRAY)),mask)
 points=[]
 if sd is not None and td is not None:
  matches=[]
  for pair in cv2.BFMatcher().knnMatch(sd,td,k=2):
   if len(pair)==2 and pair[0].distance<.85*pair[1].distance:matches.append(pair[0])
  for m in sorted(matches,key=lambda m:m.distance):
   a=np.array(sk[m.queryIdx].pt);b=np.array(tk[m.trainIdx].pt)
   if np.linalg.norm(a-b)>8:continue
   if any(np.linalg.norm(a-q['pre_native_xy'])<3 or np.linalg.norm(b-q['post_native_xy'])<3 for q in points):continue
   points.append({'pre_native_xy':a.tolist(),'post_native_xy':b.tolist(),'descriptor_distance':float(m.distance)})
 result={'created_at':datetime.now().isoformat(timespec='seconds'),'protocol_sha256':file_sha256(pp),'unique_ground_sites':len(points),'points':points,'common_grid_gate_pass':False,'images_warped':False,'mine_detector_predictions':0}
 if len(points)>=90:
  a=np.array([p['pre_native_xy'] for p in points],np.float32);b=np.array([p['post_native_xy'] for p in points],np.float32)
  cells={}
  for i,p in enumerate(a):cells.setdefault((int(p[0]//80),int(p[1]//80)),[]).append(i)
  rng=np.random.default_rng(27004);keys=sorted(cells);rng.shuffle(keys)
  for indices in cells.values():rng.shuffle(indices)
  order=[]
  while any(cells.values()):
   for key in keys:
    if cells[key]:order.append(cells[key].pop())
  held=np.array(order[:30]);fit=np.array([i for i in order[30:] if np.min(np.linalg.norm(a[i]-a[held],axis=1))>=4],int)
  residual=np.linalg.norm(a[held]-b[held],axis=1)
  cv2.setRNGSeed(27004);affine,inliers=cv2.estimateAffinePartial2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
  result.update(heldout_indices=held.tolist(),fit_indices=fit.tolist(),fit_inliers=int(inliers.sum()) if inliers is not None else 0,diagnostic_affine=affine.tolist() if affine is not None else None,
    unshifted_holdout_errors_px=residual.tolist(),unshifted_holdout_median_px=float(np.median(residual)),unshifted_holdout_p90_px=float(np.percentile(residual,90)),unshifted_holdout_within2px=float((residual<=2).mean()))
  result['common_grid_gate_pass']=bool(result['fit_inliers']>=30 and np.median(residual)<=1 and np.percentile(residual,90)<=2 and (residual<=2).mean()>=.9)
  panel=Image.new('RGB',(900,720),'white');draw=ImageDraw.Draw(panel)
  for i,idx in enumerate(held):
   x=(i%5)*180;y=(i//5)*120;draw.text((x+5,y+2),str(i+1),fill='black')
   for j,(period,p) in enumerate((('pre',a[idx]),('post',b[idx]))):
    ix,iy=np.rint(p).astype(int);tile=Image.fromarray(images[period]).crop((ix-16,iy-16,ix+17,iy+17)).resize((80,80),Image.Resampling.NEAREST)
    d=ImageDraw.Draw(tile);d.line((35,40,45,40),fill='red');d.line((40,35,40,45),fill='red');panel.paste(tile,(x+j*88+2,y+20))
  panel.save(OUT/'all30_native_pair_holdout_sites.png')
 result['frozen_dependencies']={str(CASE/f'{period}_{suffix}'):file_sha256(CASE/f'{period}_{suffix}') for period in ('pre','post') for suffix in ('BGRN_10m.tif','SCL_10m.tif','product.xml')}
 write(OUT/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('points','heldout_indices','fit_indices','unshifted_holdout_errors_px','frozen_dependencies','diagnostic_affine')},ensure_ascii=False))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
