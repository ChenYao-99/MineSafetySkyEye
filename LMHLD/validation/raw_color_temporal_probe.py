# -*- coding: utf-8 -*-
"""Fixed-budget colour/temporal separability probe, not an automatic service model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import sys,json,hashlib,time
from datetime import datetime
import numpy as np
import cv2
import rasterio
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from mine_data_protocol import DATA,assert_development_cases,verify_assets

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/raw_color_temporal_probe_20261003'
REVIEW=DATA/'review_v2_20261003';MODELS=ROOT/'models_prior/raw_color_temporal_probe_20261003'
CASES=('Amyntaio','LasCruces','Hpakant');ARMS=('RGBpost','RGBpair','NIRpair')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def frame_features(image,actual_nir=False):
    x=np.asarray(image,np.float32)
    if x.ndim!=3 or x.shape[2]<3:raise ValueError('Visible B/G/R channels required')
    if actual_nir and x.shape[2]!=4:raise ValueError('NIR features require explicitly verified actual4 spectral bands')
    b,g,r=x[:,:,0],x[:,:,1],x[:,:,2];total=b+g+r;valid=np.isfinite(x[:,:,:3]).all(2)&(total>0)
    # A common gain preservesR:G:B; per-channel minmax is deliberately absent.
    relative=np.stack([b,g,r],2)/np.maximum(total[:,:,None],1e-6)
    brightness=total/3;scale=float(np.percentile(brightness[valid],98)) if valid.any() else 1
    grey=np.clip(brightness/max(scale,1e-6),0,2)
    f=[relative[:,:,c] for c in range(3)]+[grey,2*relative[:,:,1]-relative[:,:,0]-relative[:,:,2],relative.max(2)-relative.min(2)]
    for size in (3,9):
        f.extend([cv2.blur(relative[:,:,c],(size,size)) for c in range(3)])
        mean=cv2.blur(grey,(size,size));std=np.sqrt(np.maximum(cv2.blur(grey*grey,(size,size))-mean*mean,0));f.extend([mean,std])
    sobel=np.hypot(cv2.Sobel(grey,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(grey,cv2.CV_32F,0,1,ksize=3));f.extend([cv2.blur(sobel,(3,3)),valid.astype(np.float32)])
    if actual_nir:
        nir=x[:,:,3];ndi=(nir-r)/np.maximum(nir+r,1e-6);water=(g-nir)/np.maximum(g+nir,1e-6)
        f.extend([ndi,water,cv2.blur(ndi,(9,9)),cv2.blur(water,(9,9))])
    return np.nan_to_num(np.stack(f,2),nan=0,posinf=0,neginf=0).astype(np.float32)


def features(post,pre,arm,actual_nir=False,pre_available=None):
    if arm not in ARMS:raise ValueError('Unknown feature arm')
    if arm=='NIRpair' and not actual_nir:raise ValueError('RGB/RGBA/syntheticNIR cannot provide measured-NIR features')
    nir=arm=='NIRpair';fp=frame_features(post,nir)
    if arm=='RGBpost':return fp
    if pre is None:pre=post;pre_available=np.zeros(post.shape[:2],bool)
    if pre.shape!=post.shape:raise ValueError('Pair must have the same native image grid')
    avail=(np.asarray(pre)[:,:,:3].sum(2)>0)&(np.asarray(post)[:,:,:3].sum(2)>0) if pre_available is None else np.asarray(pre_available,bool)
    before=frame_features(pre,nir);delta=fp-before
    delta[~avail]=0
    return np.concatenate([fp,delta,np.abs(delta),avail[:,:,None].astype(np.float32)],2)


def metric(q,y,valid):
    p=q>=.5;t=y==1;v=valid
    tp,fp,fn,tn=[int(a.sum()) for a in (p&t&v,p&~t&v,~p&t&v,~p&~t&v)]
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/max(tp+fn,1),'precision':tp/max(tp+fp,1),'F1':2*tp/max(2*tp+fp+fn,1),'background_FPR':fp/max(fp+tn,1)}


def prepare():
    assert_development_cases(list(CASES));verify_assets();OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists():
        p=json.loads(path.read_text(encoding='utf-8'))
        if p['review_sha']!=sha(REVIEW/'review_manifest.json'):raise ValueError('Frozen review changed')
        for name,digest in p['frozen_arrays'].items():
            if sha(OUT/name)!=digest:raise ValueError('Probe arrays changed')
        return p
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'role':'post108-run observed-development feature separability diagnostic, not final model or new blind test',
       'arms':list(ARMS),'seeds':[42,43,44],'cases':list(CASES),'folds':[{'train':[c for c in CASES if c!=held],'held':held} for held in CASES],
       'max_iter':100,'max_leaf_nodes':15,'min_samples_leaf':50,'learning_rate':.07,'l2':1.,'early_stopping':False,
       'sample_per_mine_class_cap':6000,'sampling':'same seed per case/class across arms; no held labels in sampling, fits or checkpoint choice',
       'features':'gain-preserved RGB ratios/brightness/local3/9 texture; real pair adds differences and availability; NIR arm uses verified4 bands only',
       'no_image_coordinates_mine_ID_or_annotation_lines_as_features':True,'NIR_features_are_DN_ratios_not_field_measurements':True,
       'native_10m_not_resized':True,'review_sha':sha(REVIEW/'review_manifest.json'),'frozen_arrays':{},'raw_input_hashes':{},
       'primary_threshold':.5,'reserved_predictions':0,'production_changed':False,
       'comparison_limits':'Different classifier family/compute budget from CNN; cannot attribute difference only to normalization.',
       'legacy_source_pair':'source32 has no real pre, paired features unavailable/zero; do not call it real temporal regression'}
    for eid in CASES:
        folder=DATA/f'development/{eid}';pair=[]
        for period in ('pre','post'):
            path=folder/f'{period}_BGRN_10m.tif'
            with rasterio.open(path) as src:
                if src.descriptions!=('B02','B03','B04','B08'):raise ValueError('Verified actual spectral descriptions required')
                pair.append(src.read().transpose(1,2,0).astype(np.float32))
            p['raw_input_hashes'][str(path)]=sha(path)
        pre,post=pair
        available=(pre[:,:,:3].sum(2)>0)&(post[:,:,:3].sum(2)>0)
        for period in ('pre','post'):
            scl=folder/f'{period}_SCL_10m.tif'
            if scl.exists():
                with rasterio.open(scl) as f:available&=~np.isin(f.read(1),[0,1,8,9,10,11])
        for arm in ARMS:
            array=features(post,pre,arm,actual_nir=True,pre_available=available);dest=OUT/f'{eid}_{arm}.npy';np.save(dest,array);p['frozen_arrays'][dest.name]=sha(dest)
    write(OUT/'protocol.json',p);return p


def run():
    p=prepare();models=[];label={};confidence={};old={};normals={}
    for eid in CASES:
        folder=DATA/f'development/{eid}'
        with rasterio.open(REVIEW/eid/'labels_v2.tif') as f:label[eid]=f.read(1)
        with rasterio.open(REVIEW/eid/'confidence_v2.tif') as f:confidence[eid]=f.read(1)
        with rasterio.open(folder/'partial_labels.tif') as f:old[eid]=f.read(1)
        with rasterio.open(REVIEW/eid/'reviewed_normal_mask.tif') as f:normals[eid]=f.read(1)>0
    complete=OUT/'results.json';partial=OUT/'results_partial.json'
    if complete.exists():print('Reusing complete27 classifier probes');return
    models=json.loads(partial.read_text(encoding='utf-8'))['models'] if partial.exists() else [];done={r['id'] for r in models}
    for held in CASES:
        train=[c for c in CASES if c!=held];assert_development_cases(train)
        for seed in p['seeds']:
            indices={};weights={}
            for i,eid in enumerate(train):
                rng=np.random.default_rng(seed*100+i);y=label[eid].ravel();w=confidence[eid].ravel();idx=[];sw=[]
                for klass in (0,1):
                    available=np.flatnonzero(y==klass);sel=rng.choice(available,min(len(available),6000),replace=False)
                    idx.extend(sel.tolist());sw.extend((w[sel]/len(sel)).tolist())
                indices[eid]=np.asarray(idx);weights[eid]=np.asarray(sw,np.float32)
            for arm in ARMS:
                key=f'{held}_{arm}_{seed}'
                if key in done:continue
                xx=[];yy=[];ww=[]
                for eid in train:
                    a=np.load(OUT/f'{eid}_{arm}.npy',mmap_mode='r');xx.append(a.reshape(-1,a.shape[2])[indices[eid]]);yy.append(label[eid].ravel()[indices[eid]]);ww.append(weights[eid])
                x,y,w=np.concatenate(xx),np.concatenate(yy),np.concatenate(ww);w=w/w.mean()
                model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,learning_rate=.07,l2_regularization=1.,early_stopping=False,random_state=seed)
                started=time.time()
                with threadpool_limits(limits=4):
                    model.fit(x,y,sample_weight=w);a=np.load(OUT/f'{held}_{arm}.npy',mmap_mode='r');q=model.predict_proba(a.reshape(-1,a.shape[2]))[:,1].reshape(a.shape[:2])
                if not np.isfinite(q).all():raise ValueError('Nonfinite probe output')
                dest=MODELS/key;dest.mkdir(parents=True,exist_ok=True);joblib.dump(model,dest/'model.joblib',compress=3)
                np.save(OUT/f'{key}_probability.npy',q.astype(np.float32))
                m=metric(q,old[held],old[held]!=255)
                r={'id':key,'held':held,'train':train,'seed':seed,'arm':arm,'training_pixels':len(y),'weights_sha':sha(dest/'model.joblib'),'protocol_sha':sha(OUT/'protocol.json'),
                   'held_predictions_during_fit':0,'old_reference':m,'reviewed_normal_activation':float((q[normals[held]]>=.5).mean()),'elapsed_s':time.time()-started,
                   'real_NIR_consumed':arm=='NIRpair','actual_RGB_fourth_channel_not_used':arm!='NIRpair','production_model':False}
                models.append(r);write(partial,{'protocol_sha':sha(OUT/'protocol.json'),'models':models})
                print(f'{key}: recall={m["recall"]:.3f}, oldweakBG={m["background_FPR"]:.3f}, newNormal={r["reviewed_normal_activation"]:.3f}, F1={m["F1"]:.3f}',flush=True)
    write(complete,{'protocol_sha':sha(OUT/'protocol.json'),'models':models,'not_complete_source_domain_regression':True,'no_new_blind_or_field_claim':True})


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
