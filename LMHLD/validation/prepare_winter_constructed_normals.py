"""Project-reviewed appearance weak normals on real winter pairs, no safety/accident assertion."""
from pathlib import Path
import json,sys
from datetime import datetime
import cv2
import numpy as np
import rasterio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DATA=ROOT/'datasets/arid_operation_pairs_20261004'
OUT=DATA/'review_v1_20261004'
REGIONS={
 'ZhundongSouth':[
  ('north_curved_bench_access',[[145,69],[162,80],[172,91],[175,125],[188,145],[210,158],[207,165],[181,151],[166,132],[165,95],[153,84],[142,77]]),
  ('southwest_constructed_platform',[[8,298],[52,294],[87,307],[83,329],[10,326]]),
  ('central_diagonal_access',[[155,292],[172,296],[240,343],[260,344],[290,373],[283,383],[251,354],[234,351],[166,307],[150,299]])],
 'ZhundongMine1':[
  ('central_constructed_access_platform',[[218,229],[241,230],[257,243],[260,260],[248,273],[226,263],[214,246]]),
  ('eastern_access_strip',[[275,276],[286,271],[301,292],[295,297],[283,283]]),
  ('southern_bench_access',[[74,344],[103,336],[134,334],[156,342],[156,350],[132,343],[103,345],[76,354]])]}


def write(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def display(path):
    with rasterio.open(path) as f:raw=f.read().transpose(1,2,0).astype(np.float32);valid=f.dataset_mask()>0
    x=raw[:,:,:3].copy()
    for i in range(3):
        lo,hi=np.percentile(x[:,:,i][valid],[2,98]);x[:,:,i]=np.clip((x[:,:,i]-lo)/max(hi-lo,1e-6),0,1)
    return (x[:,:,::-1]*255).astype(np.uint8),valid


def alignment(pre,post,exclude):
    sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.005)
    mask=(~cv2.dilate(exclude.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)).astype(np.uint8)*255
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    a,ad=sift.detectAndCompute(clahe.apply(cv2.cvtColor(pre,cv2.COLOR_RGB2GRAY)),mask)
    b,bd=sift.detectAndCompute(clahe.apply(cv2.cvtColor(post,cv2.COLOR_RGB2GRAY)),mask)
    if ad is None or bd is None:raise ValueError('No independent ground features')
    matches=sorted([m for p in cv2.BFMatcher().knnMatch(ad,bd,k=2) if len(p)==2 for m,n in [p] if m.distance<.85*n.distance],key=lambda m:m.distance)
    aa=[];bb=[]
    for m in matches:
        x,y=np.asarray(a[m.queryIdx].pt),np.asarray(b[m.trainIdx].pt)
        if np.linalg.norm(x-y)>8:continue  # Same published native grid, no response-based search.
        if any(np.linalg.norm(x-q)<2 or np.linalg.norm(y-r)<2 for q,r in zip(aa,bb)):continue
        aa.append(x);bb.append(y)
    aa=np.asarray(aa,np.float32);bb=np.asarray(bb,np.float32)
    if len(aa)<30:raise ValueError('Too few unique stable-ground matches')
    cv2.setRNGSeed(4104);matrix,inliers=cv2.estimateAffinePartial2D(aa,bb,method=cv2.RANSAC,ransacReprojThreshold=2)
    if matrix is None:raise ValueError('No stable alignment')
    keep=inliers.ravel()>0;residual=np.linalg.norm(aa[keep]-bb[keep],axis=1)
    return {'unique_matches':len(aa),'inliers':int(keep.sum()),'common_grid_median_px':float(np.median(residual)),
            'common_grid_p90_px':float(np.percentile(residual,90)),'diagnostic_affine':matrix.tolist(),
            'raw_pair_gate_pass':bool(keep.sum()>=30 and np.median(residual)<=1.5 and np.percentile(residual,90)<=2.),
            'raw_images_warped':False,'not_a_displacement_measurement':True}


def prepare():
    OUT.mkdir(exist_ok=True);manifest=OUT/'review_manifest.json'
    if manifest.exists():
        p=json.loads(manifest.read_text(encoding='utf-8'))
        for name,digest in p['frozen_assets'].items():
            if file_sha256(OUT/name)!=digest:raise ValueError('Winter reference changed')
        return p
    result={'registered_at':datetime.now().isoformat(timespec='seconds'),'status':'draft_pending_visual_check',
        'label_semantics':'visibly intact constructed appearance at both dates; confidence0.25; no geotechnical stability assertion',
        'normal_pixels_not_proof_of_no_hidden_instability':True,'outside_selected_surfaces_unknown':True,
        'model_predictions_used_for_selection':False,'independent_expert_review_received':False,
        'new_accident_positive_cases_added':0,'scope':'auxiliary development, observed Zhundong region','sites':{},'frozen_assets':{}}
    for site,regions in REGIONS.items():
        folder=DATA/site;dest=OUT/site;dest.mkdir(exist_ok=True);images=[];valids=[];quality=[]
        for period in ('pre','post'):
            image,valid=display(folder/f'{period}_BGRN_10m.tif');images.append(image);valids.append(valid)
            with rasterio.open(folder/f'{period}_SCL_10m.tif') as f:quality.append(f.read(1))
        shape=images[0].shape[:2];proposal=np.zeros(shape,np.uint8)
        for name,points in regions:cv2.fillPoly(proposal,[np.asarray(points,np.int32)],1)
        try:
            pair=alignment(images[0],images[1],proposal)
        except ValueError as exc:
            pair={'raw_pair_gate_pass':False,'reason':str(exc),'raw_images_warped':False,'not_a_displacement_measurement':True}
            write(dest/'alignment_failure.json',pair)
        visible=valids[0]&valids[1]
        for scl in quality:visible &= ~np.isin(scl,[0,1,3,8,9,10,11])
        core=cv2.erode(proposal,np.ones((3,3),np.uint8)).astype(bool)&visible
        proposed_count=int(core.sum())
        if not pair['raw_pair_gate_pass']:core[:]=False
        y=np.full(shape,255,np.uint8);y[core]=0;confidence=np.zeros(shape,np.float32);confidence[core]=.25
        with rasterio.open(folder/'post_BGRN_10m.tif') as f:profile=f.profile.copy()
        for name,array,dtype in (('labels.tif',y,'uint8'),('confidence.tif',confidence,'float32'),('normal_mask.tif',core.astype(np.uint8),'uint8')):
            with rasterio.open(dest/name,'w',**{**profile,'count':1,'dtype':dtype,'nodata':None}) as f:f.write(array,1)
        canvas=Image.new('RGB',(shape[1]*2,shape[0]+42),'white')
        for j,(period,image) in enumerate(zip(('pre','post'),images)):
            annotated=image.copy()
            for name,points in regions:cv2.polylines(annotated,[np.asarray(points,np.int32)],True,(255,50,20),1)
            overlay=annotated.astype(float);overlay[core]=.65*overlay[core]+.35*np.array([30,210,80]);overlay=np.clip(overlay,0,255).astype(np.uint8)
            canvas.paste(Image.fromarray(overlay),(j*shape[1],42));ImageDraw.Draw(canvas).text((j*shape[1]+5,5),period+' | polygons=project review, green=weak normal',fill='black')
        canvas.save(dest/'review_overlay.png')
        result['sites'][site]={'polygons':[{'id':name,'pixel_edge_polygon':points} for name,points in regions],
             'normal_pixels':int(core.sum()),'unknown_pixels':int((~core).sum()),'confidence':.25,'pair_alignment':pair,
             'proposed_pixels_before_pair_gate':proposed_count,
             'SCL_water_label_not_assumed_ground_truth_over_dark_coal':True,'SCL_weather_excluded_classes':[0,1,3,8,9,10,11],
             'annotated_original_sha256':{p:file_sha256(folder/f'{p}_BGRN_10m.tif') for p in ('pre','post')}}
        for file in (dest/'labels.tif',dest/'confidence.tif',dest/'normal_mask.tif',dest/'review_overlay.png'):
            result['frozen_assets'][file.relative_to(OUT).as_posix()]=file_sha256(file)
    write(manifest,result);return result


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');p=prepare();print(json.dumps(p['sites'],ensure_ascii=False))
