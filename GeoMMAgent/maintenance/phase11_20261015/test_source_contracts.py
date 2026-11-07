"""Scientific admissibility checks against retained original sources."""
import unittest
import sqlite3
from screen_noto import DATA, HERE, read
import rasterio
import numpy as np


class SourceContracts(unittest.TestCase):
    def test_original_inventory_components_are_not_independent_mines(self):
        with sqlite3.connect((DATA/'sources/20240101noto_landslidepolygon.gpkg').as_uri()+'?mode=ro',uri=True) as con:
            count = con.execute('select count(*) from landslidepolygon').fetchone()[0]
            codes = {r[0] for r in con.execute('select distinct code from landslidepolygon')}
        self.assertEqual(count,68844)
        self.assertTrue(codes.issubset(set(range(1,11))))
        self.assertFalse(read(HERE/'source_grade_contract.json')['components_count_as_independent_mines'])

    def test_april_actual_capture_interval_not_layer_prefix(self):
        plan = read(DATA/'imagery_overviews/NIIGYO_R2_APR_CONTEXT/qualification.json')['plan']
        self.assertTrue(plan['layer_id'].startswith('20240102'))
        self.assertEqual(plan['capture_date_interval'],['2024-04-05','2024-04-26'])
        self.assertIn('not January truth',read(HERE/'source_grade_contract.json')['april_role'])

    def test_recent_pre_native_resolution_and_clear_scene(self):
        report = read(DATA/'qualification/recent_pre_satellite_result.json')
        self.assertEqual(report['scene_id'],'S2A_53SPB_20231206_0_L2A')
        for band,metres,dtype in [('red',10,'uint16'),('green',10,'uint16'),('blue',10,'uint16'),('scl',20,'uint8')]:
            with rasterio.open(DATA/'recent_pre_satellite'/(band+'_provider_native.tif')) as src:
                self.assertEqual(src.res,(metres,metres))
                self.assertEqual(src.dtypes[0],dtype)
                if band=='scl':
                    self.assertEqual(set(np.unique(src.read(1))),{4,5,6})

    def test_no_promotion_of_unmatched_bench(self):
        rows = read(DATA/'qualification/niigyo_evidence_freeze.json')['source_object_support']
        bench = next(r for r in rows if r['source_feature_index']==204)
        self.assertEqual(bench['overlapping_dem_fids'],[])
        self.assertFalse(bench['training_label_admitted'])
        self.assertFalse(any(r['large_source_priority'] for r in rows))

    def test_no_false_safe_negative_or_blind_claim(self):
        contract = read(HERE/'source_grade_contract.json')
        self.assertFalse(contract['eligible_as_new_blind_acceptance_set'])
        self.assertFalse(contract['historical_pretraining_exposure_audit_complete'])
        self.assertTrue(contract['negative_pixel_scope'].startswith('none;'))
        result = read(DATA/'qualification/niigyo_evidence_freeze.json')
        self.assertEqual(result['new_verified_negative_pixels'],0)
        self.assertEqual(result['new_admitted_supervised_positive_pixels'],0)

    def test_same_event_role_and_no_training_authorization(self):
        contract = read(HERE/'source_grade_contract.json')
        self.assertTrue(contract['same_site_same_event_across_dates'])
        self.assertFalse(contract['training_allowed'])
        self.assertFalse(contract['model_predictions_allowed'])
        self.assertEqual(contract['current_role'],'qualification_only')

    def test_alpha_is_not_cloud_snow_quality(self):
        contract = read(HERE/'source_grade_contract.json')
        self.assertIn('alpha validity is not cloud/snow quality',contract['cloud_and_snow_policy'])
        source = read(DATA/'imagery_overviews/NIIGYO_R2_JAN11/qualification.json')
        self.assertEqual(source['valid_source_pixel_fraction'],1)
        self.assertFalse(source['training_allowed'])


if __name__=='__main__':
    unittest.main(verbosity=2)
