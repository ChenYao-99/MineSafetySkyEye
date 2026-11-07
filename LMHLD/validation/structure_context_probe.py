"""Matched native RGB structure/temporal diagnostic; all four mines are already observed."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import sys,json,time
from datetime import datetime
import numpy as np
import cv2
import rasterio
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from four_mine_calibrated_probe import CASES,REVIEW,DATA,ROOT,sha,write,metric,features,prepare as prepare_colour
from mine_data_protocol import assert_development_cases,verify_assets

COLOUR=ROOT/'experiments/four_mine_calibrated_probe_20261004'
OUT=ROOT/'experiments/structure_context_probe_20261004'
MODELS=ROOT/'models_prior/structure_context_probe_20261004'
ARMS=('RGBstructurePost','RGBstructurePair')


def structure_features(image):
    """Rotation-invariant local gradient coherence and roughness at real pixel scales."""
    x=np.asarray(image,np.float32)
    grey=x[:,:,:3].mean(2)
    valid=np.isfinite(grey)&(grey>0)
    gain=float(np.percentile(grey[valid],98)) if valid.any() else 1.0
    grey=np.clip(grey/max(gain,1e-6),0,2)
    dx=cv2.Sobel(grey,cv2.CV_32F,1,0,ksize=3)/8
    dy=cv2.Sobel(grey,cv2.CV_32F,0,1,ksize=3)/8
    lap=np.abs(cv2.Laplacian(grey,cv2.CV_32F,ksize=3))/8
    result=[]
    for size in (3,9,17,33):
        blur=lambda a:cv2.blur(a,(size,size))
        a,b,c=blur(dx*dx),blur(dx*dy),blur(dy*dy)
        energy=a+c
        coherence=np.sqrt(np.maximum((a-c)**2+4*b*b,0))/(energy+1e-8)
        mean=blur(grey);variance=np.maximum(blur(grey*grey)-mean*mean,0)
        result.extend([coherence,np.sqrt(energy)/(mean+.02),
                       np.sqrt(variance)/(mean+.02),blur(lap)/(mean+.02)])
    return np.nan_to_num(np.stack(result,2),nan=0,posinf=0,neginf=0).astype(np.float32)


def structural_features(post,pre,arm,available=None):
    if arm not in ARMS:raise ValueError('Unknown structural feature arm')
    after=structure_features(post)
    if arm=='RGBstructurePost':return np.concatenate([features(post,None,'RGBpost'),after],2)
    if pre is None:
        pre=post
        available=np.zeros(post.shape[:2],bool)
    if pre.shape!=post.shape:raise ValueError('Same native pair grid required')
    if available is None:available=(pre[:,:,:3].sum(2)>0)&(post[:,:,:3].sum(2)>0)
    available=np.asarray(available,bool)
    delta=after-structure_features(pre);delta[~available]=0
    return np.concatenate([features(post,pre,'RGBpair',pre_available=available),after,delta,np.abs(delta)],2)


def read_pair(eid):
    pair=[]
    folder=DATA/f'development/{"Xinjing" if eid=="CN_XJ2023" else eid}'
    for period in ('pre','post'):
        path=DATA/f'radiometry_audit_20261004/{"Xinjing" if eid=="CN_XJ2023" else eid}/{period}_BGRN_calibrated_tags.tif'
        with rasterio.open(path) as f:
            if f.descriptions!=('B02','B03','B04','B08'):raise ValueError('Verified band descriptions required')
            image=f.read().transpose(1,2,0).astype(np.float32)
            image=image*np.asarray(f.scales,np.float32)+np.asarray(f.offsets,np.float32)
            image[f.dataset_mask()==0]=0
            pair.append(np.maximum(image,0))
    pre,post=pair
    available=np.load(COLOUR/f'{eid}_RGBpair.npy',mmap_mode='r')[:,:,-1]>0
    return pre,post,available


def prepare():
    assert_development_cases(list(CASES));verify_assets();colour=prepare_colour()
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists():
        p=json.loads(path.read_text(encoding='utf-8'))
        if p['colour_protocol_sha']!=sha(COLOUR/'protocol.json') or p['baseline_results_sha']!=sha(COLOUR/'results.json'):
            raise ValueError('Matched control or source protocol changed')
        if p['implementation_sha']!=sha(__file__):raise ValueError('Frozen structural implementation changed')
        for name,digest in p['frozen_arrays'].items():
            if sha(OUT/name)!=digest:raise ValueError('Frozen structure feature changed')
        return p
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'role':'observed development, no new blind test',
       'cases':list(CASES),'folds':colour['folds'],'seeds':colour['seeds'],'arms':list(ARMS),
       'baseline':'reuse12 frozen RGBpost and12 RGBpair controls with identical samples/budget/classifier',
       'colour_protocol_sha':sha(COLOUR/'protocol.json'),'baseline_results_sha':sha(COLOUR/'results.json'),
       'implementation_sha':sha(__file__),'frozen_arrays':{},'primary_threshold':.5,
       'max_iter':100,'max_leaf_nodes':15,'min_samples_leaf':50,'learning_rate':.07,'l2':1.,'early_stopping':False,
       'sample_per_mine_class_cap':6000,'same_sampling_as_colour_probe':True,
       'features':'RGBpost or real RGBpair plus local3/9/17/33px gradient-tensor coherence, relative energy/CV/Laplacian; pair real differences masked by pre availability',
       'native_10m_no_spatial_resize':True,'NIR_consumed':False,'coordinates_mine_ID_prediction_or_source_answer_lines_consumed':False,
       'weak_normal_not_geotechnical_stability':True,'expert_ground_truth_available':False,'reserved_predictions':0,
       'gates':{'macro_recall_min':.7,'worst_case_recall_min':.5,'old_weak_background_FPR_max':.15,
                'reviewed_normal_activation_max':.15,'seeds_all_required':True,'award_standard':False},
       'promotion_requires_additional_source_and_system_regression':True,'production_changed':False}
    for eid in CASES:
        pre,post,available=read_pair(eid)
        for arm in ARMS:
            a=structural_features(post,pre,arm,available)
            base=np.load(COLOUR/f'{eid}_{"RGBpost" if arm=="RGBstructurePost" else "RGBpair"}.npy')
            if not np.array_equal(a[:,:,:base.shape[2]],base):raise ValueError('Existing colour representation changed')
            dest=OUT/f'{eid}_{arm}.npy';np.save(dest,a);p['frozen_arrays'][dest.name]=sha(dest)
    write(path,p);return p


def run():
    p=prepare();complete=OUT/'results.json';partial=OUT/'results_partial.json'
    if complete.exists():print('Reusing completed24 structural probes');return
    label={};confidence={};old={};normals={}
    for eid in CASES:
        folder=DATA/f'development/{"Xinjing" if eid=="CN_XJ2023" else eid}'
        source=folder/'reference_v1_20261004' if eid=='CN_XJ2023' else REVIEW/eid
        for container,name in ((label,'labels.tif' if eid=='CN_XJ2023' else 'labels_v2.tif'),
                               (confidence,'confidence.tif' if eid=='CN_XJ2023' else 'confidence_v2.tif')):
            with rasterio.open(source/name) as f:container[eid]=f.read(1)
        with rasterio.open(source/'labels.tif' if eid=='CN_XJ2023' else folder/'partial_labels.tif') as f:old[eid]=f.read(1)
        with rasterio.open(source/('normal_mask.tif' if eid=='CN_XJ2023' else 'reviewed_normal_mask.tif')) as f:normals[eid]=f.read(1)>0
    saved=json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else None
    if saved and saved['protocol_sha']!=sha(OUT/'protocol.json'):raise ValueError('Resume protocol changed')
    rows=saved['models'] if saved else [];done={r['id'] for r in rows}
    for held in CASES:
        train=[c for c in CASES if c!=held];assert_development_cases(train)
        for seed in p['seeds']:
            indices={};weights={}
            for i,eid in enumerate(train):
                rng=np.random.default_rng(seed*100+i);y=label[eid].ravel();w=confidence[eid].ravel();idx=[];sw=[]
                for klass in (0,1):
                    candidates=np.flatnonzero(y==klass);sel=rng.choice(candidates,min(len(candidates),6000),replace=False)
                    idx.extend(sel.tolist());sw.extend((w[sel]/len(sel)).tolist())
                indices[eid]=np.asarray(idx);weights[eid]=np.asarray(sw,np.float32)
            for arm in ARMS:
                key=f'{held}_{arm}_{seed}'
                if key in done:continue
                xx=[];yy=[];ww=[]
                for eid in train:
                    a=np.load(OUT/f'{eid}_{arm}.npy',mmap_mode='r')
                    xx.append(a.reshape(-1,a.shape[2])[indices[eid]])
                    yy.append(label[eid].ravel()[indices[eid]]);ww.append(weights[eid])
                x,y,w=np.concatenate(xx),np.concatenate(yy),np.concatenate(ww);w=w/w.mean()
                model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,
                         learning_rate=.07,l2_regularization=1.,early_stopping=False,random_state=seed)
                started=time.time()
                with threadpool_limits(limits=4):
                    model.fit(x,y,sample_weight=w)
                    a=np.load(OUT/f'{held}_{arm}.npy',mmap_mode='r');flat=a.reshape(-1,a.shape[2])
                    probability=np.concatenate([model.predict_proba(flat[i:i+32768])[:,1] for i in range(0,len(flat),32768)]).reshape(a.shape[:2])
                if not np.isfinite(probability).all():raise ValueError('Invalid probability')
                dest=MODELS/key;dest.mkdir(parents=True,exist_ok=True);joblib.dump(model,dest/'model.joblib',compress=3)
                np.save(OUT/f'{key}_probability.npy',probability.astype(np.float32))
                r={'id':key,'held':held,'train':train,'seed':seed,'arm':arm,'training_pixels':len(y),
                   'weights_sha':sha(dest/'model.joblib'),'protocol_sha':sha(OUT/'protocol.json'),'held_predictions_during_fit':0,
                   'old_reference':metric(probability,old[held],old[held]!=255),
                   'reviewed_normal_activation':float((probability[normals[held]]>=.5).mean()),'elapsed_s':time.time()-started,
                   'production_model':False,'real_NIR_consumed':False}
                rows.append(r);write(partial,{'protocol_sha':sha(OUT/'protocol.json'),'models':rows})
                print(f'{key}: recall={r["old_reference"]["recall"]:.3f}, oldweakBG={r["old_reference"]["background_FPR"]:.3f}, newNormal={r["reviewed_normal_activation"]:.3f}',flush=True)
    write(complete,{'protocol_sha':sha(OUT/'protocol.json'),'models':rows,'new_blind_or_field_claim':False,'production_changed':False})


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
