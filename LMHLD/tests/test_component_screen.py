"""Component inference isolation, exact calibration summaries and safe cutoffs."""
from pathlib import Path
import inspect
import json
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
import component_filter as filt
import component_screen as exp


def scene(group=0,novel_positive=False):
    p=np.zeros((16,16),np.float32);q=p.copy();y=np.zeros((16,16),np.uint8)
    p[1:4,1:4]=.95;q[1:4,1:4]=.95;y[1:4,1:4]=1
    p[8:10,1:3]=.1;q[8:10,1:3]=.95
    p[12,12]=.1;q[12,12]=.95
    if novel_positive:y[8:10,1:3]=1
    pred=filt.extract_components(p,q,.5,.8,3.)
    return filt.observed(pred,y,exp.base.gt_components(y,3.),group,0)


def sources(novel_positive=False):
    return {g:[scene(g,novel_positive)] for g in range(6) if g!=2}


class ComponentTests(unittest.TestCase):
    def test_prediction_interfaces_accept_no_truth(self):
        for fn in (filt.extract_components,filt.predict_scores,filt.keep_components):
            names=inspect.signature(fn).parameters
            self.assertFalse(any('truth' in name or name.startswith('gt') for name in names))

    def test_eight_connectivity_and_finite_features(self):
        p=np.zeros((8,8),np.float32);p[2,2]=p[3,3]=.9
        pred=filt.extract_components(p,p,.5,.8,1.)
        self.assertEqual(len(pred.areas),1);self.assertEqual(pred.areas[0],2)
        self.assertEqual(pred.features.shape,(1,16));self.assertTrue(np.isfinite(pred.features).all())

    def test_native_threshold_uses_float64_boundary_comparison(self):
        p=np.zeros((4,4),np.float32)
        p[1,1]=np.nextafter(np.float32(.7),np.float32(0))
        pred=filt.extract_components(p,p,.7,.7,1.)
        self.assertEqual(len(pred.areas),0)

    def test_empty_full_border_constant_maps(self):
        for value in (0.,1.):
            p=np.full((8,8),value,np.float32)
            pred=filt.extract_components(p,p,.5,.8,3.)
            self.assertEqual(len(pred.areas),int(value))
            self.assertTrue(np.isfinite(pred.features).all())
            if value:self.assertEqual(pred.features[0,5],1.)

    def test_anchor_and_single_pixel_guards(self):
        c=scene();p=c.prediction;scores=np.zeros(len(p.areas))
        keep=filt.keep_components(p,scores,1.,True)
        self.assertTrue(np.all(keep[p.anchored]))
        self.assertTrue(np.all(keep[p.areas<2]))
        self.assertFalse(np.any(keep[(~p.anchored)&(p.areas>=2)]))
        unguarded=filt.keep_components(p,scores,1.,False)
        self.assertFalse(np.any(unguarded[p.anchored&(p.areas>=2)]))

    def test_whole_component_removal_never_adds_or_splits(self):
        p=scene().prediction;keep=p.anchored|(p.areas<2)
        filtered=filt.component_mask(p,keep)
        self.assertFalse(np.any(filtered&~(p.labels>0)))
        for j,on in enumerate(keep,1):self.assertTrue(np.all(filtered[p.labels==j]==on))

    def test_source_fitting_and_calibration_reject_target(self):
        data=sources();data[2]=[scene(2)]
        with self.assertRaises(ValueError):filt.fit_source_classifier(data,2)
        with self.assertRaises(ValueError):filt.CalibrationPool(data,2)

    def test_positive_labels_mean_any_ground_truth_overlap(self):
        c=scene();truth=np.zeros_like(c.truth);j=int(np.flatnonzero(c.prediction.areas==4)[0])+1
        location=np.argwhere(c.prediction.labels==j)[0];truth[tuple(location)]=1
        obs=filt.observed(c.prediction,truth,exp.base.gt_components(truth,3.),0,0)
        self.assertEqual(filt.positive_overlap(obs)[j-1],1)

    def test_source_model_roundtrip_and_no_target_normalization(self):
        model=filt.fit_source_classifier(sources(),2)
        self.assertEqual(model['source_groups'],[0,1,3,4,5]);self.assertTrue(model['held_out_excluded'])
        pred=scene().prediction
        a=filt.predict_scores(pred,model);b=filt.predict_scores(pred,json.loads(json.dumps(model)))
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        self.assertGreater(a[pred.anchored].mean(),a[(~pred.anchored)&(pred.areas>=2)].mean())

    def test_fast_counts_match_native_recalculation_for_random_keep_sets(self):
        rng=np.random.default_rng(43);data={}
        for g in range(6):
            if g==2:continue
            p=rng.random((16,16)).astype(np.float32);q=rng.random((16,16)).astype(np.float32)
            truth=np.zeros((16,16),np.uint8);truth[2:5,2:5]=1;truth[10:12,10:12]=1
            pred=filt.extract_components(p,q,.65,.8,3.)
            data[g]=[filt.observed(pred,truth,exp.base.gt_components(truth,3.),g,0)]
        pool=filt.CalibrationPool(data,2)
        for _ in range(15):
            keep=rng.random(len(pool.area))>.4
            exp.assert_fast_matches_exact(pool.summary(keep),exp.full_calibration_metrics(pool,keep))

    def test_source_calibration_removes_background_preserves_real_objects(self):
        data=sources();pool=filt.CalibrationPool(data,2)
        baseline=exp.full_calibration_metrics(pool,np.ones(len(pool.area),bool))
        scores=np.concatenate([c.prediction.features[:,6] for _,cases in sorted(data.items()) for c in cases])
        selected=exp.calibrate(pool,scores,True,baseline,exp.CAL)
        self.assertLess(selected['calibration']['macro_false_components_per_mpix'],baseline['macro_false_components_per_mpix'])
        self.assertEqual(selected['calibration']['macro_cover50'],baseline['macro_cover50'])
        self.assertEqual(selected['calibration']['macro_iou50'],baseline['macro_iou50'])

    def test_source_calibration_keeps_unsupported_true_small_object(self):
        data=sources(True);pool=filt.CalibrationPool(data,2)
        baseline=exp.full_calibration_metrics(pool,np.ones(len(pool.area),bool))
        scores=np.concatenate([c.prediction.features[:,6] for _,cases in sorted(data.items()) for c in cases])
        selected=exp.calibrate(pool,scores,True,baseline,exp.CAL)
        self.assertEqual(selected['calibration']['macro_cover50'],baseline['macro_cover50'])
        self.assertEqual(selected['calibration']['macro_iou50'],baseline['macro_iou50'])

    def test_objective_ties_choose_keep_all(self):
        data=sources(True);pool=filt.CalibrationPool(data,2)
        baseline=exp.full_calibration_metrics(pool,np.ones(len(pool.area),bool))
        scores=np.ones(len(pool.area))
        selected=exp.calibrate(pool,scores,True,baseline,exp.CAL)
        self.assertEqual(selected['cutoff'],-1.)

    def test_one_class_source_falls_back_without_fabricated_fit(self):
        data={g:[scene(g,True)] for g in range(6) if g!=2}
        model=filt.fit_source_classifier(data,2)
        self.assertEqual(model['kind'],'keep_all')

    def test_invalid_probabilities_scores_or_model_fail_closed(self):
        p=np.zeros((8,8),np.float32);p[0,0]=np.nan
        with self.assertRaises(ValueError):filt.extract_components(p,p,.5,.8,1.)
        c=scene().prediction
        with self.assertRaises(ValueError):filt.keep_components(c,np.full(len(c.areas),np.nan),.5)
        model={'kind':'logistic','features':list(filt.FEATURES),'mean':[0.]*16,'std':[0.]*16,'weights':[0.]*16,'bias':0.}
        with self.assertRaises(ValueError):filt.predict_scores(c,model)


if __name__=='__main__':unittest.main()
