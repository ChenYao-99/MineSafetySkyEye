"""Register professionally mapped partial flood extent; cropped edge is never a failure boundary."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio,torch
from PIL import Image,ImageDraw
from kornia.feature import LoFTR
from model_registry import file_sha256
from mine_data_protocol import assert_development_cases,verify_assets
from prepare_winter_constructed_normals import display
from register_gamsberg_published_boundary_v3 import weights,sheet
SRC=ROOT/'datasets/maxclosure_sources_20261005/jagersfontein/charter_product_28498588.img'
CASE=ROOT/'datasets/jagersfontein_development_20261005/ZA_JAGS2022';OUT=CASE/'official_reference_registration_v2_actual_viewport'
CROP=(48,48,3025,2420);SCALE=.25
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def tensor(rgb,mask):
    grey=cv2.createCLAHE(2,(8,8)).apply(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY));grey[~mask]=0
    h,w=grey.shape;hh=(h+7)//8*8;ww=(w+7)//8*8;x=np.zeros((hh,ww),np.float32);m=np.zeros((hh,ww),np.float32);x[:h,:w]=grey/255.;m[:h,:w]=mask
    return torch.from_numpy(x)[None,None].cuda(),torch.from_numpy(m)[None].cuda()
def run():
    assert_development_cases(['ZA_JAGS2022']);verify_assets();OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():print('Frozen Jagersregistration exists');return
    p={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'ZA_JAGS2022','role':'development','source_sha256':file_sha256(SRC),
      'source_authority':'SANSA professional reference map13Sept2022Pleiades;legendexplicitFloodedarea','source_product_url':'https://disasterscharter.org/cos-api/api/file/public/article-image/28498588',
      'source_capture_date':'2022-09-13','actual_original_target_date':'2022-09-16','cropped_source_frame_xyxy':CROP,'reference_scale_for_matching_only':SCALE,
      'source_red_line_rule':'R>180,G<100,B<110,R>1.8G,R>1.8B;largest connected component after3pxclose;legendoutsideviewportexcluded',
      'source_flood_interior_seed_original_xy':[1500,1700],'crop_border_closure':'topological region extraction only;NEVER claimed physical accidentedge',
      'source_viewport_edges_and_control_extrapolation_must_be_unknown':True,'positive_reference_scope':'dated professionalmapped deposit interior visible in partialviewport;not entire downstreamaffectedarea',
      'matcher':'frozenLoFTRoutdoor;genericgeometrynotmineclassifier','unique_ground_sites_min':60,'fit_inliers_min':30,'spatial_holdouts':30,'holdout_seed':55005,
      'holdout_median_px_max':1.5,'holdout_p90_px_max':3,'holdout_within3px_fraction_min':.9,'source_or_target_resized_as_training_observations':False,'model_predictions_used':False}
    write(OUT/'protocol.json',p)
    raw=np.asarray(Image.open(SRC).convert('RGB'));x0,y0,x1,y1=CROP;source=raw[y0:y1,x0:x1].copy()
    r,g,b=[source[:,:,i].astype(np.float32) for i in range(3)];red=(r>180)&(g<100)&(b<110)&(r>1.8*g)&(r>1.8*b)
    red=cv2.morphologyEx(red.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));n,ids,stats,_=cv2.connectedComponentsWithStats(red)
    if n<2:raise ValueError('Officialred boundary not detected')
    selected=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));line=ids==selected
    if stats[selected,cv2.CC_STAT_AREA]<10000:raise ValueError('No coherent mapped extent stroke')
    barrier=line.astype(np.uint8);barrier[[0,-1]]=1;barrier[:,[0,-1]]=1
    filled=barrier.copy();cv2.floodFill(filled,None,(1500-x0,1700-y0),2);body=filled==2
    fraction=float(body.mean())
    if not .03<fraction<.55:raise ValueError('Partialextent fill escaped orsource seedinvalid')
    cv2.imencode('.png',body.astype(np.uint8)*255)[1].tofile(str(OUT/'source_partial_mapped_interior.png'))
    source_ok=np.ones(source.shape[:2],bool);source_ok[cv2.dilate(body.astype(np.uint8),np.ones((81,81),np.uint8))>0]=False
    source_ok[350:1450,50:1200]=False;source_ok[:30]=False;source_ok[-30:]=False;source_ok[:,:30]=False;source_ok[:,-30:]=False
    image=cv2.resize(source,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_AREA);sm=cv2.resize(source_ok.astype(np.uint8),image.shape[1::-1],interpolation=cv2.INTER_NEAREST)>0
    target,valid=display(CASE/'post_BGRN_10m.tif')
    with rasterio.open(CASE/'post_SCL_10m.tif') as f:valid &= ~np.isin(f.read(1),[0,1,3,7,8,9,10,11])
    # SourceROI carries an independently declared site location;all classes are fromtheactual16Septproducer.
    xx,mm=tensor(image,sm);yy,nn=tensor(target,valid);torch.set_num_threads(4);torch.manual_seed(55005)
    model=LoFTR(pretrained=None).eval();model.load_state_dict(weights());model.cuda()
    with torch.inference_mode():out=model({'image0':xx,'image1':yy,'mask0':mm,'mask1':nn})
    a=(out['keypoints0'].cpu().numpy()+.5)/SCALE-.5;b=out['keypoints1'].cpu().numpy();score=out['confidence'].cpu().numpy()
    unique=[]
    for idx in np.argsort(-score):
      if score[idx]<.5:continue
      sa=np.rint(a[idx]).astype(int);tb=np.rint(b[idx]).astype(int)
      if not(0<=sa[0]<source_ok.shape[1] and 0<=sa[1]<source_ok.shape[0] and source_ok[sa[1],sa[0]]):continue
      if not(0<=tb[0]<valid.shape[1] and 0<=tb[1]<valid.shape[0] and valid[tb[1],tb[0]]):continue
      if any(np.linalg.norm(a[idx]-p['source_xy'])<12 or np.linalg.norm(b[idx]-p['target_native_xy'])<2 for p in unique):continue
      unique.append({'source_xy':a[idx].tolist(),'target_native_xy':b[idx].tolist(),'match_confidence':float(score[idx])})
    v={'created_at':datetime.now().isoformat(timespec='seconds'),'source_sha256':file_sha256(SRC),'target_sha256':file_sha256(CASE/'post_BGRN_10m.tif'),'protocol_sha256':file_sha256(OUT/'protocol.json'),
       'unique_ground_sites':len(unique),'points':unique,'source_mapped_partial_interior_fraction':fraction,'source_mapped_interior_pixels':int(body.sum()),'source_map_cropped_not_completeextent':True,'registration_gate_pass':False,'mine_detector_predictions':0,'labels_promoted':False}
    if len(unique)>=60:
      aa=np.array([p['source_xy'] for p in unique],np.float32);bb=np.array([p['target_native_xy'] for p in unique],np.float32);cells={}
      for i,p in enumerate(aa):cells.setdefault((int(p[0]//300),int(p[1]//300)),[]).append(i)
      rng=np.random.default_rng(55005);keys=sorted(cells);rng.shuffle(keys)
      for indices in cells.values():rng.shuffle(indices)
      order=[]
      while any(cells.values()):
        for key in keys:
          if cells[key]:order.append(cells[key].pop())
      held=np.array(order[:30]);fit=np.array([i for i in order[30:] if np.min(np.linalg.norm(bb[i]-bb[held],axis=1))>=4],int)
      cv2.setRNGSeed(55005);matrix,inliers=cv2.estimateAffine2D(aa[fit],bb[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
      if matrix is not None:
        error=np.linalg.norm(aa[held]@matrix[:,:2].T+matrix[:,2]-bb[held],axis=1);singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
        passed=bool(inliers.sum()>=30 and np.median(error)<=1.5 and np.percentile(error,90)<=3 and (error<=3).mean()>=.9 and .05<singular.min() and singular.max()<.5 and singular.max()/singular.min()<1.5)
        hull=cv2.convexHull(aa[fit][inliers.ravel()>0])
        v.update(sourcecrop_to_native_affine=matrix.tolist(),fit_inliers=int(inliers.sum()),heldout_indices=held.tolist(),fit_indices=fit.tolist(),heldout_errors_px=error.tolist(),
          holdout_median_px=float(np.median(error)),holdout_p90_px=float(np.percentile(error,90)),holdout_within3px_fraction=float((error<=3).mean()),affine_singular_values=singular.tolist(),source_fit_ground_convex_hull=hull.reshape(-1,2).tolist(),registration_gate_pass=passed)
        mapped=cv2.warpAffine(body.astype(np.uint8),matrix,target.shape[1::-1],flags=cv2.INTER_NEAREST)>0;review=target.copy();review[mapped]=(.6*review[mapped]+.4*np.array([255,50,10])).astype(np.uint8)
        cv2.imencode('.png',review)[1].tofile(str(OUT/'mapped_partial_extent_on_original_post_review.png'));sheet(source,target,aa[held],bb[held],OUT/'all30_source_ground_holdout_sites.png')
    write(OUT/'registration.json',v);print(json.dumps({k:x for k,x in v.items() if k not in ('points','sourcecrop_to_native_affine','source_fit_ground_convex_hull','heldout_indices','fit_indices','heldout_errors_px')},ensure_ascii=False))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
