"""Whole-component screening on frozen probability maps; no production hook.

Prediction features never accept ground truth. Supervised fitting is explicitly
source-only. Whole-component removal cannot split retained prediction objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import cv2
import numpy as np

FEATURES = (
    'log_pixel_area', 'log_nominal_area_m2', 'bbox_fill', 'absolute_log_aspect',
    'four_edge_compactness', 'touches_parent_border', 'mean_margin', 'p90_margin',
    'peak_margin', 'global_mean', 'local_mean', 'mean_absolute_disagreement',
    'mean_local_minus_global', 'global_half_support', 'global_entropy', 'local_entropy',
)


@dataclass
class PredictionComponents:
    labels: np.ndarray
    features: np.ndarray
    anchored: np.ndarray
    areas: np.ndarray


@dataclass
class ObservedComponents:
    prediction: PredictionComponents
    truth: np.ndarray
    gt_components: dict
    group: int
    position: int


def source_keys(keys, held):
    if set(keys) != set(range(6)) - {held}:
        raise ValueError('Exactly five source regions required; held-out forbidden')


def extract_components(global_probability, local_probability, pixel_threshold, anchor_threshold, nominal_gsd):
    p, q = [np.asarray(v, np.float32) for v in (global_probability, local_probability)]
    if p.ndim != 2 or p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Finite aligned probability maps required')
    if p.min()<0 or p.max()>1 or q.min()<0 or q.max()>1 or not np.isfinite(nominal_gsd) or nominal_gsd<=0:
        raise ValueError('Invalid probability or nominal GSD')
    if not np.isfinite(pixel_threshold) or not np.isfinite(anchor_threshold):
        raise ValueError('Finite source thresholds required')
    mean = .5*(p+q)
    mask = np.asarray(mean, np.float64) >= pixel_threshold
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:,cv2.CC_STAT_AREA].astype(np.int64)
    dtype=np.uint16 if n<=65536 else np.uint32
    lab=lab.astype(dtype)
    if n==1:
        return PredictionComponents(lab,np.empty((0,len(FEATURES)),np.float32),np.zeros(0,bool),areas)
    flat=lab.ravel().astype(np.int64)
    def average(values):
        return np.bincount(flat,weights=np.asarray(values).ravel(),minlength=n)[1:]/areas
    anchored=np.bincount(lab[np.asarray(p,np.float64)>=anchor_threshold],minlength=n)[1:]>0
    perimeter=np.zeros(n,np.float64)
    for a,b in ((lab[:,:-1],lab[:,1:]),(lab[:-1],lab[1:])):
        different=a!=b
        perimeter+=np.bincount(a[different],minlength=n)+np.bincount(b[different],minlength=n)
    for edge in (lab[0],lab[-1],lab[:,0],lab[:,-1]):
        perimeter+=np.bincount(edge,minlength=n)
    perimeter=perimeter[1:]
    widths=stats[1:,cv2.CC_STAT_WIDTH];heights=stats[1:,cv2.CC_STAT_HEIGHT]
    left=stats[1:,cv2.CC_STAT_LEFT];top=stats[1:,cv2.CC_STAT_TOP]
    border=(left==0)|(top==0)|(left+widths==lab.shape[1])|(top+heights==lab.shape[0])
    margin=np.clip((mean-pixel_threshold)/max(1-pixel_threshold,1e-6),0,1)
    p90=[];peak=[]
    for j,(x,y,w,h) in enumerate(zip(left,top,widths,heights),1):
        values=margin[y:y+h,x:x+w][lab[y:y+h,x:x+w]==j]
        p90.append(np.quantile(values,.9));peak.append(values.max())
    def entropy(values):
        a=np.clip(values,1e-6,1-1e-6)
        return -(a*np.log(a)+(1-a)*np.log1p(-a))
    features=np.stack([
        np.log1p(areas),np.log1p(areas*nominal_gsd**2),areas/(widths*heights),
        np.abs(np.log(widths/heights)),np.minimum(4*np.pi*areas/np.maximum(perimeter**2,1),1),
        border,average(margin),p90,peak,average(p),average(q),average(np.abs(p-q)),
        average(q-p),average(p>=.5),average(entropy(p)),average(entropy(q)),
    ],axis=1).astype(np.float32)
    if not np.isfinite(features).all():raise ValueError('Nonfinite component features')
    return PredictionComponents(lab,features,anchored,areas)


def observed(prediction, truth, gt_components, group, position):
    truth=np.asarray(truth,np.uint8)
    if truth.shape!=prediction.labels.shape or not ((truth==0)|(truth==1)).all():
        raise ValueError('Native binary source/evaluation truth required')
    return ObservedComponents(prediction,truth,gt_components,int(group),int(position))


def positive_overlap(case):
    return np.bincount(case.prediction.labels[case.truth>0],minlength=len(case.prediction.areas)+1)[1:].astype(np.int64)


def fit_source_classifier(cases_by_group, held):
    source_keys(cases_by_group,held)
    xs=[];ys=[];groups=[];region_info={}
    for g,cases in sorted(cases_by_group.items()):
        if any(c.group!=g for c in cases):raise ValueError('Mislabeled source group')
        xx=[];yy=[]
        for c in cases:
            eligible=c.prediction.areas>=2
            xx.append(c.prediction.features[eligible]);yy.append((positive_overlap(c)[eligible]>0).astype(np.int64))
        x=np.concatenate(xx) if xx else np.empty((0,len(FEATURES)),np.float32)
        y=np.concatenate(yy) if yy else np.zeros(0,np.int64)
        region_info[str(g)]={'candidates':len(y),'positive':int(y.sum()),'pure_background':int((y==0).sum())}
        xs.append(x);ys.append(y);groups.append(np.full(len(y),g,np.int64))
    x=np.concatenate(xs);y=np.concatenate(ys);group=np.concatenate(groups)
    if len(x)==0 or len(np.unique(y))<2:
        return {'kind':'keep_all','features':list(FEATURES),'source_groups':sorted(cases_by_group),
                'held_out_excluded':True,'fit_regions':region_info,'reason':'insufficient_source_classes'}
    present=np.unique(group);moment_weights=np.zeros(len(y),np.float64);weights=np.zeros(len(y),np.float64)
    for g in present:
        where=group==g;moment_weights[where]=1/(len(present)*where.sum())
        classes=np.unique(y[where])
        for cls in classes:
            idx=where&(y==cls);weights[idx]=len(y)/(len(present)*len(classes)*idx.sum())
    mean=(x*moment_weights[:,None]).sum(0)
    std=np.maximum(np.sqrt(((x-mean)**2*moment_weights[:,None]).sum(0)),.001)
    z=np.clip((x-mean)/std,-8,8)
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-7)
    started=time.perf_counter();model.fit(z,y,sample_weight=weights)
    if int(model.n_iter_[0])>=1000:raise ValueError('Source logistic fit did not converge')
    return {'kind':'logistic','features':list(FEATURES),'mean':mean.tolist(),'std':std.tolist(),
            'weights':model.coef_[0].tolist(),'bias':float(model.intercept_[0]),
            'source_groups':sorted(cases_by_group),'held_out_excluded':True,'fit_regions':region_info,
            'n_iter':int(model.n_iter_[0]),'fit_seconds':time.perf_counter()-started,
            'positive_definition':'any_GT_overlap; negatives_have_zero_GT_overlap',
            'regularization_C':1.,'normalization':'source region-balanced moments; std floor.001; clip+/-8'}


def predict_scores(prediction, classifier):
    if tuple(classifier['features'])!=FEATURES:raise ValueError('Component feature schema changed')
    if classifier['kind']=='keep_all':return np.ones(len(prediction.areas),np.float64)
    if classifier['kind']!='logistic':raise ValueError('Unknown classifier')
    mean,std,weight=[np.asarray(classifier[k],np.float64) for k in ('mean','std','weights')]
    if any(v.shape!=(len(FEATURES),) for v in (mean,std,weight)) or np.any(std<=0) or not all(np.isfinite(v).all() for v in (mean,std,weight)):
        raise ValueError('Invalid fitted component parameters')
    z=np.clip((prediction.features.astype(np.float64)-mean)/std,-8,8)
    logit=np.clip(z@weight+classifier['bias'],-35,35)
    return 1/(1+np.exp(-logit))


def component_scores(prediction, classifier, arm):
    if arm=='area_guard':return prediction.features[:,1].astype(np.float64)
    if arm=='confidence_guard':return prediction.features[:,6].astype(np.float64)
    if arm in ('learned_guard','learned_no_guard'):return predict_scores(prediction,classifier)
    raise ValueError('Unknown component screening arm')


def keep_components(prediction, scores, threshold, protect_anchor=True):
    scores=np.asarray(scores,np.float64)
    if scores.shape!=prediction.areas.shape or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Invalid component scores/cutoff')
    keep=(prediction.areas<2)|(scores>=threshold)
    if protect_anchor:keep|=prediction.anchored
    return keep


def component_mask(prediction, keep):
    keep=np.asarray(keep,bool)
    if keep.shape!=prediction.areas.shape:raise ValueError('Component decision shape changed')
    return np.r_[False,keep][prediction.labels]


class CalibrationPool:
    """Exact TP/FP/coverage summaries for whole-component deletion; source only."""
    def __init__(self,cases_by_group,held):
        source_keys(cases_by_group,held)
        self.groups=sorted(cases_by_group);self.cases=[]
        comp_groups=[];tp=[];area=[];guard=[];features=[];gt_areas=[];gt_small=[];gt_groups=[]
        edge_gt=[];edge_pred=[];edge_counts=[];co=0;go=0
        self.pixels=np.zeros(6,np.int64);self.truth_pixels=np.zeros(6,np.int64)
        for g,cases in sorted(cases_by_group.items()):
            for c in cases:
                if c.group!=g:raise ValueError('Mislabeled source calibration group')
                self.cases.append(c);p=c.prediction;n=len(p.areas);gc=c.gt_components
                comp_groups.append(np.full(n,g,np.int64));tp.append(positive_overlap(c));area.append(p.areas)
                guard.append(p.anchored|(p.areas<2));features.append(p.features)
                gt_areas.append(gc['areas'][1:]);gt_small.append(gc['small'][1:]);gt_groups.append(np.full(len(gc['areas'])-1,g,np.int64))
                valid=(gc['labels']>0)&(p.labels>0)
                keys,nums=np.unique(gc['labels'][valid].astype(np.int64)*(n+1)+p.labels[valid],return_counts=True)
                edge_gt.append(keys//(n+1)-1+go);edge_pred.append(keys%(n+1)-1+co);edge_counts.append(nums)
                self.pixels[g]+=c.truth.size;self.truth_pixels[g]+=int(c.truth.sum())
                co+=n;go+=len(gc['areas'])-1
        self.comp_groups=np.concatenate(comp_groups);self.tp=np.concatenate(tp);self.area=np.concatenate(area)
        self.guard=np.concatenate(guard);self.features=np.concatenate(features)
        self.gt_areas=np.concatenate(gt_areas);self.gt_small=np.concatenate(gt_small);self.gt_groups=np.concatenate(gt_groups)
        self.edge_gt=np.concatenate(edge_gt);self.edge_pred=np.concatenate(edge_pred);self.edge_counts=np.concatenate(edge_counts)

    def summary(self,keep):
        keep=np.asarray(keep,bool)
        if keep.shape!=self.area.shape:raise ValueError('Calibration keep shape changed')
        tp=np.bincount(self.comp_groups,weights=self.tp*keep,minlength=6).astype(np.int64)
        fp=np.bincount(self.comp_groups,weights=(self.area-self.tp)*keep,minlength=6).astype(np.int64)
        false=np.bincount(self.comp_groups,weights=((self.tp==0)&(self.area>=2)&keep),minlength=6).astype(np.int64)
        hits=np.bincount(self.edge_gt,weights=self.edge_counts*keep[self.edge_pred],minlength=len(self.gt_areas))
        covered=self.gt_small&(hits>=.5*self.gt_areas)
        nn=np.bincount(self.gt_groups,weights=self.gt_small,minlength=6).astype(np.int64)
        hh=np.bincount(self.gt_groups,weights=covered,minlength=6).astype(np.int64)
        regions={}
        for g in self.groups:
            fn=int(self.truth_pixels[g]-tp[g]);tn=int(self.pixels[g]-self.truth_pixels[g]-fp[g])
            regions[str(g)]={'tp':int(tp[g]),'fp':int(fp[g]),'fn':fn,'tn':tn,'small_n':int(nn[g]),'small_cover50_count':int(hh[g]),
                'f1':float(2*tp[g]/max(2*tp[g]+fp[g]+fn,1)),
                'small_cover50':float(hh[g]/nn[g]) if nn[g] else None,
                'background_fpr':float(fp[g]/max(fp[g]+tn,1)),
                'false_components_per_mpix':float(false[g]*1e6/self.pixels[g])}
        valid=[r for r in regions.values() if r['small_cover50'] is not None]
        return {'regions':regions,'macro_f1':float(np.mean([r['f1'] for r in regions.values()])),
                'macro_cover50':float(np.mean([r['small_cover50'] for r in valid])) if valid else 0.,
                'macro_false_components_per_mpix':float(np.mean([r['false_components_per_mpix'] for r in regions.values()]))}
