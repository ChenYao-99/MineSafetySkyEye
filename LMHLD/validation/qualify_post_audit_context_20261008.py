"""Input/source qualification of different mines, preserving all causal uncertainty."""
from datetime import datetime
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from PIL import Image
DATA=ROOT/'datasets/post_audit_development_20261008'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quality-retry',action='store_true');args=parser.parse_args()
    acquisition=read(DATA/'acquisition.json');plan=read(DATA/'native_pair_plan.json')
    for item in acquisition['files']:assert sha(item['path'])==item['sha256']
    groups=[]
    for site in ['Chile1','Chile2']:
        folder=DATA/'MineNetCD/MCD'/site;sizes=[]
        for name in ['im1.jpg','im2.jpg','ref.png']:
            with Image.open(folder/name) as image:
                sizes.append(image.size)
                if name=='ref.png':
                    mask=np.asarray(image);values=np.unique(mask).tolist();assert set(values)<={0,255}
                    count=int((mask==255).sum())
        assert len(set(sizes))==1
        thumb=DATA/(site+'_original_post_preview.png')
        with Image.open(folder/'im2.jpg') as image:
            image.thumbnail((900,900));image.save(thumb)
        case='AUX_ARID_'+site.upper()
        native=DATA/('native_pairs_quality_v2' if args.quality_retry and site=='Chile1' else 'native_pairs')/case
        record=read(native/'acquisition.json')
        assert set(record['images'])=={'pre','post'}
        epochs=[];images=[];valid=[];grids=[]
        for epoch in ['pre','post']:
            info=record['images'][epoch]
            for name,digest in info['assets_sha256'].items():assert sha(native/name)==digest
            with rasterio.open(native/(epoch+'_BGRN_native_10m.tif')) as source:
                assert source.count==4 and source.res==(10,10) and source.descriptions==('B02','B03','B04','B08')
                images.append(source.read().transpose(1,2,0));grids.append((source.crs,source.transform,source.shape))
            with rasterio.open(native/(epoch+'_SCL_on_10m_grid.tif')) as source:scl=source.read(1)
            observed=~np.isin(scl,[0,1,3,8,9,10,11]);observed &= np.all(images[-1]>0,axis=2);valid.append(observed)
            epochs.append({'epoch':epoch,'datetime':info['datetime'],'valid_fraction':float(observed.mean()),
                'calibration_and_XML_preserved':True,'native_multispectral_pixels_unwarped':True})
        assert grids[0]==grids[1]
        common=valid[0]&valid[1]
        def grey(image,mask):
            x=image[:,:,:3].mean(axis=2).astype(np.float32);lo,hi=np.quantile(x[mask],[.02,.98])
            return np.clip((x-lo)/max(hi-lo,1),0,1).astype(np.float32)*255
        sift=cv2.SIFT_create(nfeatures=3000)
        keys=[];desc=[]
        for image,mask in zip(images,valid):
            key,d=sift.detectAndCompute(grey(image,mask).astype(np.uint8),mask.astype(np.uint8)*255);keys.append(key);desc.append(d)
        alignment={'independent_ground_accuracy_verified':False,'images_warped':False,'gate_pass':False}
        if all(d is not None for d in desc):
            pairs=cv2.BFMatcher().knnMatch(desc[0],desc[1],k=2);matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
            if len(matches)>=24:
                a=np.float32([keys[0][m.queryIdx].pt for m in matches]);b=np.float32([keys[1][m.trainIdx].pt for m in matches])
                matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2)
                if inliers is not None:
                    keep=inliers.ravel().astype(bool);unshifted=np.linalg.norm(a-b,axis=1)[keep]
                    if unshifted.size:
                        med=float(np.median(unshifted));p90=float(np.quantile(unshifted,.9))
                        alignment.update(matches=len(matches),inliers=int(keep.sum()),unshifted_median_px=med,unshifted_p90_px=p90,
                            gate_pass=bool(keep.sum()>=24 and keep.mean()>=.5 and med<=1.5 and p90<=2))
        ready=bool(common.mean()>=.6 and alignment['gate_pass'])
        groups.append({'author_site':site,'role':'development_auxiliary_change_context_only','RGB_dimensions':list(sizes[0]),
            'author_change_pixels':count,'author_change_values':values,'native_pair_case':case,'native_pair_directory':str(native),'epochs':epochs,
            'native_common_valid_fraction':float(common.mean()),'relative_alignment':alignment,'paired_input_quality_gate_pass':ready,
            'RGB_long_period_change_mask_transferred_to_native_year_pair':False,
            'new_accident_positive_or_verified_safe_negative_pixels':0,'classifier_predictions_or_training':0,
            'mine_operator_identity':'candidate geographic association;not independently ground confirmed',
            'use_contract':'auxiliary mining-change/context only;failed quality gates exclude paired quantitative use'})
    result={'verified_at':datetime.now().astimezone().isoformat(),'source_acquisition_integrity_passed':True,
        'different_geographic_mine_groups':2,'original_author_RGB_pairs':2,'original_author_change_masks':2,
        'original_native_BGRN_pairs':2,'paired_input_quality_groups_passed':sum(row['paired_input_quality_gate_pass'] for row in groups),
        'groups':groups,'reserved_mine_used_for_followup_development':False,
        'accident_or_safe_background_truth_admitted':0,'new_classifier_predictions_or_fits':0,
        'limits':['new contexts address observed domain mismatch;no claim of improved model accuracy',
            'original change labels do not distinguish excavation,accident or safe background',
            'RGB catalogue time span differs from native pairs;labels are not transferred',
            'operator-level activity evidence does not certify the cause or safety of each changed pixel']}
    if args.quality_retry:result.update(prior_quality_failure_preserved=str(DATA/'qualification.json'),date_choice_based_on_source_quality_only=True)
    name='qualification_quality_v2.json' if args.quality_retry else 'qualification.json'
    (DATA/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('QUALIFIED source groups2;native pairs2;paired quality passed',result['paired_input_quality_groups_passed'],';accident/safety labels0;modelcalls0',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
