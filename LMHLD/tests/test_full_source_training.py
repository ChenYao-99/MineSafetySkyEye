"""Full-pool isolation, compact preprocessing and calibration invariants."""
from pathlib import Path
import copy
import hashlib
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import full_source_training as exp


def row(group, index, digest, label='label', legacy=False):
    return {'group':group,'index':index,'sha256':digest,'label_sha256':label,'legacy_selected':legacy}


class FullSourceTests(unittest.TestCase):
    def test_compact_parent_is_bit_identical_to_old_global_view(self):
        rng=np.random.default_rng(87)
        for side in (32,128,224):
            image=rng.normal(size=(4,side,side)).astype(np.float64)
            image[1]=9.0
            label=(rng.random((1,side,side))<.13).astype(np.float32)
            x,y=exp.compact_whole_view(image,label)
            oldx,oldy=exp.base.make_views(image,label)
            np.testing.assert_array_equal(x[0],oldx[0])
            np.testing.assert_array_equal(y[0],oldy[0])
            self.assertEqual(x.shape,(1,4,64,64));self.assertEqual(y.shape,(1,1,64,64))
            self.assertEqual(x.dtype,np.float16)

    def test_compact_downsampling_retains_fractional_labels(self):
        image=np.zeros((4,224,224),np.float32);label=np.zeros((1,224,224),np.float32)
        label[:,100:103,100:103]=1
        _,y=exp.compact_whole_view(image,label)
        self.assertTrue(np.any((y>0)&(y<1)))

    def test_compact_rejects_nonfinite_and_non_native_layout(self):
        image=np.zeros((4,32,32),np.float32);image[0,0,0]=np.nan
        with self.assertRaises(ValueError):exp.compact_whole_view(image,np.zeros((32,32)))
        with self.assertRaises(ValueError):exp.compact_whole_view(np.zeros((3,32,32)),np.zeros((32,32)))

    def test_broader_validation_hash_audit_excludes_training_overlap(self):
        entries=[row(0,1,'val_overlap',legacy=True),row(0,2,'clean')]
        kept,removed=exp.quarantine_train(entries,{'val_overlap'})
        self.assertEqual([r['index'] for r in kept],[2])
        self.assertEqual(removed[0]['reason'],'matches_official_val')

    def test_cross_region_and_conflicting_labels_quarantined(self):
        entries=[row(0,1,'cross'),row(2,2,'cross'),row(1,3,'conflict','a'),row(1,4,'conflict','b')]
        kept,removed=exp.quarantine_train(entries,set())
        self.assertFalse(kept)
        self.assertEqual({r['reason'] for r in removed},{'cross_region_duplicate','conflicting_labels'})

    def test_duplicate_canonical_choice_preserves_legacy_parent(self):
        entries=[row(0,1,'same'),row(0,8,'same',legacy=True),row(0,2,'other',legacy=True),row(0,9,'new')]
        kept,removed=exp.quarantine_train(entries,set())
        self.assertEqual([r['index'] for r in kept],[2,8,9])
        self.assertEqual(removed[0]['index'],1)

    def test_source_backbone_choice_ignores_target_and_prefers_old_on_tie(self):
        full={'source_macro_f1':.7,'target_f1':1.}
        old={'source_macro_f1':.8,'target_f1':0.}
        self.assertEqual(exp.source_backbone_choice(full,old),'prior512')
        old['source_macro_f1']=.7
        self.assertEqual(exp.source_backbone_choice(full,old),'prior512')
        full['source_macro_f1']=.71
        self.assertEqual(exp.source_backbone_choice(full,old),'full')

    def test_identical_pool_rng_replays_prior4800_schedule(self):
        groups=np.repeat(np.arange(6),512);held=2;seed=42
        legacy=np.random.default_rng(seed+held*100);full=np.random.default_rng(seed+held*100)
        trace=hashlib.sha256()
        for _ in range(4800):
            ids,k,flip=exp.legacy_augmentation(groups,held,legacy)
            new_ids,new_k,new_flip=exp.legacy_augmentation(groups,held,full)
            np.testing.assert_array_equal(ids,new_ids)
            self.assertEqual((k,flip),(new_k,new_flip))
            trace.update(np.asarray([*ids,k,int(flip)],np.int64).tobytes())
        old=exp.base.read(exp.PRIOR/'fold_2_Hokkaido_Japan_128/seed42/result.json')
        self.assertEqual(trace.hexdigest(),old['training']['sampling_sha256'])

    def test_source_histogram_threshold_matches_original_exact_grid(self):
        rng=np.random.default_rng(5);held=3;probs={};data={};histograms={}
        for g in range(6):
            if g==held:continue
            p=rng.uniform(size=(16,16)).astype(np.float32)
            p.flat[:4]=np.array([0.,np.nextafter(np.float32(.7),np.float32(0)),.7,1.],np.float32)
            y=np.zeros((16,16),np.uint8);y[5:8,5:8]=1
            probs[g]=[p];data[g]={'labels':[y]}
            histograms[g]=exp.source_histogram([p],[y])
        for budget in exp.BUDGETS:
            a=exp.threshold_from_histograms(histograms,held,budget)
            b=exp.base.source_threshold(probs,data,[g for g in range(6) if g!=held],held,budget)
            self.assertEqual(a,b)

    def test_histograms_refuse_target_invalid_probabilities_and_fractional_counts(self):
        h=np.zeros(1001,np.int64);h[0]=10
        with self.assertRaises(ValueError):exp.threshold_from_histograms({g:h for g in range(6)},2,.02)
        with self.assertRaises(ValueError):exp.source_histogram([np.array([[np.nan]])],[np.array([[0]])])
        with self.assertRaises(ValueError):exp.source_histogram([np.array([[1.1]])],[np.array([[0]])])
        with self.assertRaises(ValueError):exp.threshold_from_histograms({g:h.astype(float) for g in range(6) if g!=2},2,.02)

    def test_stricter_source_budget_cannot_lower_threshold(self):
        h=np.ones(1001,np.int64);hist={g:h for g in range(6) if g!=2}
        thresholds=[exp.threshold_from_histograms(hist,2,b)[0] for b in exp.BUDGETS]
        self.assertEqual(thresholds,sorted(thresholds))

    def test_both_rng_streams_resume_exactly(self):
        full_groups=np.repeat(np.arange(6),[513,600,520,700,640,800]);legacy_groups=np.repeat(np.arange(6),512)
        a=np.random.default_rng(242);b=np.random.default_rng(242)
        for _ in range(27):
            exp.legacy_augmentation(full_groups,2,a);exp.legacy_augmentation(legacy_groups,2,b)
        cp=(copy.deepcopy(a.bit_generator.state),copy.deepcopy(b.bit_generator.state))
        expected=(exp.legacy_augmentation(full_groups,2,a),exp.legacy_augmentation(legacy_groups,2,b))
        c=np.random.default_rng();d=np.random.default_rng();c.bit_generator.state=cp[0];d.bit_generator.state=cp[1]
        actual=(exp.legacy_augmentation(full_groups,2,c),exp.legacy_augmentation(legacy_groups,2,d))
        for left,right in zip(expected,actual):
            np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:],right[1:])


if __name__=='__main__':unittest.main()
