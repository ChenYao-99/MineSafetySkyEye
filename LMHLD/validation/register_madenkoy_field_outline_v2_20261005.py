"""Fixed, held-ground registration of a publicly reported engineer-surveyed outline."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,cv2,rasterio,torch
from PIL import Image,ImageDraw
from kornia.feature import LoFTR
from model_registry import file_sha256
from mine_data_protocol import verify_assets,assert_development_cases
from register_gamsberg_published_boundary_v5 import weights,WEIGHT,padded
SOURCE_ROOT=ROOT/'datasets/madenkoy_development_20261005';CASE=SOURCE_ROOT/'TR_MAD2016'
SOURCE=SOURCE_ROOT/'carla2018_original_pre_aerial_panel.png';OUT=CASE/'field_registration_v2_scale_matched'
POLYGON=np.asarray([[217.0, 141.0], [208.0, 155.0], [194.0, 212.0], [162.0, 312.0], [184.0, 379.0], [206.0, 416.0], [216.0, 467.0], [228.0, 483.0], [257.0, 493.0], [266.0, 499.0], [267.0, 519.0], [281.0, 533.0], [295.0, 532.0], [320.0, 519.0], [326.0, 519.0], [341.0, 539.0], [359.0, 538.0], [372.0, 555.0], [385.0, 555.0], [474.0, 532.0], [509.0, 518.0], [591.0, 507.0], [643.0, 510.0], [688.0, 446.0], [735.0, 392.0], [730.0, 355.0], [700.0, 325.0], [724.0, 278.0], [692.0, 272.0], [650.0, 231.0], [639.0, 214.0], [622.0, 197.0], [590.0, 192.0], [544.0, 171.0], [509.0, 159.0], [459.0, 128.0], [450.0, 132.0], [433.0, 149.0], [409.0, 164.0], [381.0, 157.0], [336.0, 139.0], [281.0, 111.0], [244.0, 118.0]],np.float64)
ROI=[140,230,320,420]
# A rough search prior from the published200m scale/north arrow and the visible
# whole-pit landmark correspondence. This prior is not the fitted boundary.
PRIOR=np.asarray([[1/14.2,0,149],[0,1/14.2,278]],np.float64)

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def display(path,scl):
    with rasterio.open(path) as src:
        a=src.read().transpose(1,2,0).astype(np.float32);valid=np.all(src.read_masks()>0,axis=0)&np.all(a>0,axis=2)
        a=a*np.asarray(src.scales,np.float32)+np.asarray(src.offsets,np.float32)
    # Class7 can supply a visually checked geometric landmark, but never a
    # positive training label; the stricter labelmask is checked separately.
    valid&=~np.isin(scl,[0,1,3,6,8,9,10,11])
    rgb=np.ascontiguousarray(np.uint8(255*np.power(np.clip(a[:,:,[2,1,0]]/.45,0,1),.65)))
    rgb[~valid]=0
    return rgb,valid

def sheet(source,target,a,b,path):
    canvas=Image.new('RGB',(900,720),'white');draw=ImageDraw.Draw(canvas)
    for i,(p,q) in enumerate(zip(a,b)):
        x=(i%5)*180;y=(i//5)*120;draw.text((x+5,y+2),str(i+1),fill='black')
        for j,(im,point,half) in enumerate([(source,p,114),(target,q,8)]):
            px,py=np.rint(point).astype(int);crop=Image.fromarray(im).crop((px-half,py-half,px+half+1,py+half+1)).resize((80,80),Image.Resampling.NEAREST)
            d=ImageDraw.Draw(crop);d.line((34,40,46,40),fill='red');d.line((40,34,40,46),fill='red');canvas.paste(crop,(x+j*88+2,y+20))
    canvas.save(path)

def run():
    assert_development_cases(['TR_MAD2016']);verify_assets();OUT.mkdir(exist_ok=True)
    protocol={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'case_id':'TR_MAD2016','role':'development',
       'source_figure':str(SOURCE),'source_sha256':file_sha256(SOURCE),'original_polygon_xy':POLYGON.tolist(),
       'source_engineer_field_survey_assertion':'Carla2018 case study states engineers defined external instability boundary through post-event field surveys;Fig2 actualboundary,not Fig11 precursorzone',
       'source_anonymized_identity':'candidate correspondence withYavuz2023 namedMadenkoy;requires realmine geometry and heldground checks;matchingdate aloneinsufficient',
       'source_aerial_exact_flight_date_unknown':True,'source_original_professional_GIS_received':False,
       'approximate_search_prior_original_to_native':PRIOR.tolist(),'target_native_ROI':ROI,
       'matcher':'generic frozenLoFTR outdoor;neither mining detector nor geotechnical labels',
       'matcher_weight_sha256':file_sha256(WEIGHT),'matcher_only_common_resolution_m':10,'matcher_only_interpolation_factor':4,'actual_source_scalebar':{'original_x_0m':783,'original_x_100m':925},'only_source_aerial_texture_coarsened_to_actualS2resolution':True,
       'matcher_resampling_not_new_native_resolution_or_training_pixels':True,'matcher_confidence_min':.5,
       'candidate_prior_tolerance_native_px':15,'source_duplicate_radius_original_px':28.4,'target_duplicate_radius_native_px':2,
       'heldout_sites':30,'heldout_seed':51006,'fit_distance_from_heldout_native_px_min':4,
       'fit_ransac_native_px':2,'unique_ground_sites_min':60,'fit_inliers_min':30,
       'heldout_median_native_px_max':1.5,'heldout_p90_native_px_max':3,'heldout_within3px_min':.9,
       'fit_hull_must_enclose_entire_original_polygon':True,'same_source_line_for_both_dates':True,
       'all_geometry_and_matches_exclude_affected_field_polygon_and_graphical_annotations':True,
       'geometry_class7_only_optional_landmarks_not_training_observations':True,
       'planned_positive_core_uncertainty_buffer_m_min':40,'raw_original_images_or_masks_modified':False,
       'accident_detector_predictions_used':0,'registration_code_sha256':file_sha256(__file__)}
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));assert all(old.get(k)==v for k,v in protocol.items() if k!='registered_at')
        if (OUT/'registration.json').exists():print('Frozen source registration reused;no refit');return
    else:write(path,protocol)
    source=np.asarray(Image.open(SOURCE).convert('RGB'));height,width=source.shape[:2]
    polygon_mask=np.zeros((height,width),np.uint8);cv2.fillPoly(polygon_mask,[np.rint(POLYGON).astype(np.int32)],1)
    source_mask=np.ones((height,width),np.uint8);source_mask[:20]=0;source_mask[-20:]=0;source_mask[:,:20]=0;source_mask[:,-20:]=0
    source_mask[cv2.dilate(polygon_mask,np.ones((93,93),np.uint8))>0]=0
    source_mask[:85,880:]=0;source_mask[510:,930:]=0;source_mask[650:,750:]=0
    r,g,b=[source[:,:,i].astype(float) for i in range(3)]
    annotation=((r>190)&(g>170)&(b<120))|((r>190)&(g<80)&(b<100))|(source.min(2)>220)
    source_mask[cv2.dilate(annotation.astype(np.uint8),np.ones((9,9),np.uint8))>0]=0
    preview=source.copy();cv2.polylines(preview,[np.rint(POLYGON).astype(np.int32)],True,(255,210,0),4);Image.fromarray(preview).save(OUT/'original_field_boundary_digitisation_review.png')
    coarse_width=int(round(width/14.2));source_scale=coarse_width*4/width;coarse=cv2.resize(source,(coarse_width,int(round(height*coarse_width/width))),interpolation=cv2.INTER_AREA);source_small=cv2.resize(coarse,(coarse_width*4,coarse.shape[0]*4),interpolation=cv2.INTER_LINEAR)
    small_mask=cv2.resize(source_mask,(source_small.shape[1],source_small.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    torch.set_num_threads(4);torch.manual_seed(51006);model=LoFTR(pretrained=None).eval();model.load_state_dict(weights());model.cuda()
    x,m=padded(source_small,small_mask);rows=[]
    for period in ['pre','post']:
        with rasterio.open(CASE/(period+'_SCL_10m.tif')) as src:scl=src.read(1)
        target,valid=display(CASE/(period+'_BGRN_10m.tif'),scl)
        approximate=POLYGON@PRIOR[:,:2].T+PRIOR[:,2]
        excluded=np.zeros(valid.shape,np.uint8);cv2.fillPoly(excluded,[np.rint(approximate).astype(np.int32)],1)
        allowed=valid&~(cv2.dilate(excluded,np.ones((11,11),np.uint8))>0)
        left,top,right,bottom=ROI;target_crop=target[top:bottom,left:right];mask_crop=allowed[top:bottom,left:right]
        target_scale=4;resized=cv2.resize(target_crop,None,fx=target_scale,fy=target_scale,interpolation=cv2.INTER_LINEAR)
        resized_mask=cv2.resize(mask_crop.astype(np.uint8),(resized.shape[1],resized.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        y,n=padded(resized,resized_mask)
        with torch.inference_mode():result=model({'image0':x,'image1':y,'mask0':m,'mask1':n})
        a=(result['keypoints0'].cpu().numpy()+.5)/source_scale-.5
        bb=(result['keypoints1'].cpu().numpy()+.5)/target_scale-.5+np.array([left,top]);confidence=result['confidence'].cpu().numpy()
        expected=a@PRIOR[:,:2].T+PRIOR[:,2];points=[]
        for i in np.argsort(-confidence):
            p,q=a[i],bb[i];sx,sy=np.rint(p).astype(int);tx,ty=np.rint(q).astype(int)
            if confidence[i]<.5 or not(0<=sx<width and 0<=sy<height and source_mask[sy,sx]):continue
            if not(0<=tx<valid.shape[1] and 0<=ty<valid.shape[0] and allowed[ty,tx]):continue
            if np.linalg.norm(q-expected[i])>15:continue
            if any(np.linalg.norm(p-v['source_xy'])<28.4 or np.linalg.norm(q-v['native_xy'])<2 for v in points):continue
            points.append({'source_xy':p.tolist(),'native_xy':q.tolist(),'confidence':float(confidence[i])})
        row={'period':period,'target_sha256':file_sha256(CASE/(period+'_BGRN_10m.tif')),'raw_matcher_pairs':len(a),
             'unique_native_ground_sites':len(points),'points':points,'registration_gate_pass':False}
        if len(points)>=60:
            aa=np.asarray([p['source_xy'] for p in points],np.float32);b=np.asarray([p['native_xy'] for p in points],np.float32)
            cells={}
            for i,p in enumerate(aa):cells.setdefault((int(p[0]//250),int(p[1]//250)),[]).append(i)
            rng=np.random.default_rng(51006);keys=sorted(cells);rng.shuffle(keys)
            for bucket in cells.values():rng.shuffle(bucket)
            order=[]
            while any(cells.values()):
                for key in keys:
                    if cells[key]:order.append(cells[key].pop())
            held=np.asarray(order[:30]);fit=np.asarray([i for i in order[30:] if np.min(np.linalg.norm(b[i]-b[held],axis=1))>=4],int)
            row.update(heldout_indices=held.tolist(),fit_indices=fit.tolist())
            if len(fit)>=30:
                cv2.setRNGSeed(51006);matrix,inliers=cv2.estimateAffine2D(aa[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
                if matrix is not None:
                    errors=np.linalg.norm(aa[held]@matrix[:,:2].T+matrix[:,2]-b[held],axis=1)
                    singular=np.linalg.svd(matrix[:,:2],compute_uv=False);hull=cv2.convexHull(aa[fit][inliers.ravel()>0])
                    inside=all(cv2.pointPolygonTest(hull,tuple(p.astype(float)),False)>=0 for p in POLYGON)
                    passed=bool(inliers.sum()>=30 and np.median(errors)<=1.5 and np.percentile(errors,90)<=3 and (errors<=3).mean()>=.9 and .05<singular.min() and singular.max()<.1 and singular.max()/singular.min()<1.5 and inside)
                    mapped=POLYGON@matrix[:,:2].T+matrix[:,2]
                    row.update(source_to_native_affine=matrix.tolist(),mapped_polygon_native_xy=mapped.tolist(),fit_inliers=int(inliers.sum()),
                        heldout_errors_native_px=errors.tolist(),heldout_median_px=float(np.median(errors)),heldout_p90_px=float(np.percentile(errors,90)),
                        heldout_within3px_fraction=float((errors<=3).mean()),affine_singular_values=singular.tolist(),field_polygon_inside_ground_fit_hull=inside,
                        registration_gate_pass=passed)
                    overlay=target.copy();cv2.polylines(overlay,[np.rint(mapped).astype(np.int32)],True,(255,80,20),1)
                    Image.fromarray(overlay).save(OUT/(period+'_registered_field_boundary_review.png'));sheet(source,target,aa[held],b[held],OUT/(period+'_all30_independent_ground_sites.png'))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['points','heldout_indices','fit_indices','heldout_errors_native_px','source_to_native_affine','mapped_polygon_native_xy']}),flush=True)
        del result,y,n;torch.cuda.empty_cache()
    write(OUT/'registration.json',{'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'protocol_sha256':file_sha256(path),
          'rows':rows,'both_dates_registration_pass':all(r['registration_gate_pass'] for r in rows),'mine_identity_and_visual_holdouts_still_require_review':True,
          'detector_predictions':0,'pixel_labels_accepted':False})

if __name__=='__main__':run()
