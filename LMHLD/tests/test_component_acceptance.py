"""Disjoint sample admission and fixed acceptance guardrails."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import component_screen_acceptance as exp


def fixture():
    return {'regions':[{'splits':{'val':{'indices':[0],'raw_parent_sha256':[f'seen{g}']}},
                        'all_official_val_image_sha256':[f'seen{g}',f'new{g}a',f'new{g}b']} for g in range(6)],
            'kept_train':[{'sha256':'train'}]}


class AcceptanceTests(unittest.TestCase):
    def test_additional_selection_never_reuses_selected_parents(self):
        m=fixture();rows,_=exp.select_additional(m,[2,4,3])
        self.assertEqual(len(rows),6);self.assertTrue(all(r['index']>0 for r in rows))
        self.assertEqual({r['group'] for r in rows},{2,3,4})

    def test_hash_overlap_and_cross_region_duplicates_are_excluded(self):
        m=fixture();m['regions'][2]['all_official_val_image_sha256'][1]='seen0'
        m['regions'][3]['all_official_val_image_sha256'][1]='train'
        m['regions'][4]['all_official_val_image_sha256'][1]='cross'
        m['regions'][2]['all_official_val_image_sha256'][2]='cross'
        rows,excluded=exp.select_additional(m,[2,4,3])
        self.assertFalse({r['image_sha256'] for r in rows}&{'seen0','train','cross'})
        self.assertEqual(len(excluded),4)

    def test_same_region_duplicates_keep_first_index_without_labels(self):
        m=fixture();m['regions'][2]['all_official_val_image_sha256'][2]='new2a'
        rows,excluded=exp.select_additional(m,[2])
        self.assertEqual([r['index'] for r in rows],[1]);self.assertEqual(excluded[0]['index'],2)

    def test_fixed_guardrails_do_not_accept_large_recall_loss(self):
        limits={'min_false_component_reduction_each_seed':.2,'max_cover_loss_each_seed':.01,
            'max_region_cover_loss':.03,'max_iou_loss_each_seed':.01,'max_f1_loss_each_seed':.005,
            'max_false_component_ratio_vs_tta_each_seed':1.1}
        extra={'min_cover_gain_vs_tta_each_seed':.02,'min_iou_gain_vs_tta_each_seed':.02}
        raw={'f1':.65,'small_cover50':.55,'small_iou50':.32,'false_components_per_mpix':46.}
        screened={'f1':.649,'small_cover50':.549,'small_iou50':.319,'false_components_per_mpix':35.}
        tta={'f1':.65,'small_cover50':.48,'small_iou50':.26,'false_components_per_mpix':34.}
        means={'area_guard':screened,'multiscale_raw':raw,'tta5':tta}
        _,checks=exp.assess(means,.002,limits,extra);self.assertTrue(all(checks.values()))
        bad=copy.deepcopy(means);bad['area_guard']['small_cover50']=.50
        _,checks=exp.assess(bad,.05,limits,extra)
        self.assertFalse(checks['coverage_preserved']);self.assertFalse(checks['region_coverage_preserved'])


if __name__=='__main__':unittest.main()
