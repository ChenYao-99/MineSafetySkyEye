"""Source dates, native pixels and scope separation for new actual-work development data."""
from pathlib import Path
import json, sys, unittest
import numpy as np, rasterio
from rasterio.windows import Window,from_bounds
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import registry,reserved_case_ids
DATA=ROOT/'datasets/source_backed_operation_change_20261005'
FULL=DATA/'full_published_geometry_window_v2'
EVIDENCE=ROOT/'datasets/actual_operation_evidence_20261005/Amrun'

def load(p):return json.loads(p.read_text(encoding='utf-8'))

class SourceBackedOperationTests(unittest.TestCase):
    def test_actual_work_evidence_matches_month_without_becoming_pixel_safety_truth(self):
        plan=load(FULL/'site_plan.json')
        self.assertIn('119.57',(EVIDENCE/'independent_audit_2025_page_060.txt').read_text(encoding='utf-8'))
        self.assertIn('30/9/23',(EVIDENCE/'independent_audit_2025_page_060.txt').read_text(encoding='utf-8'))
        for source in plan['evidence']:self.assertEqual(file_sha256(source['file']),source['sha256'])
        case=FULL/'AU_AMRUN202309_FULL';acq=load(case/'acquisition.json')
        self.assertLess(acq['images']['pre']['datetime'][:10],'2023-09-01')
        self.assertGreater(acq['images']['post']['datetime'][:10],'2023-09-30')
        self.assertEqual(acq['annotation_count'],0);self.assertEqual(acq['model_predictions'],0)
        self.assertIn('entire Amrun',plan['source_period']['area_scope'])
        review=load(case/'visibility_review_v1/review.json')
        self.assertEqual(review['case_id'],case.name);self.assertEqual(review['normal_or_failure_labels'],0)
        self.assertTrue(review['observed_vegetation_loss_NOT_confirmed_operation_or_accident_pixels'])
        self.assertFalse(review['project_field_trial_or_expert_review'])

    def test_extended_window_preserves_exact_original_crop_DN_and_native_calibration(self):
        for period in ('pre','post'):
            small=DATA/'AU_AMRUN202309';large=FULL/'AU_AMRUN202309_FULL'
            for directory in (small,large):
                row=load(directory/'acquisition.json')['images'][period]
                for name,key in ((period+'_BGRN_10m.tif','sha256'),(period+'_SCL_10m.tif','scl_sha256'),(period+'_product.xml','product_XML_sha256')):
                    self.assertEqual(file_sha256(directory/name),row[key])
            for suffix in ('BGRN_10m.tif','SCL_10m.tif'):
                with rasterio.open(small/(period+'_'+suffix)) as a,rasterio.open(large/(period+'_'+suffix)) as b:
                    self.assertEqual(a.crs,b.crs);self.assertEqual(a.res,(10,10));self.assertEqual(a.res,b.res)
                    self.assertEqual(a.scales,b.scales);self.assertEqual(a.offsets,b.offsets)
                    window=from_bounds(*a.bounds,transform=b.transform)
                    values=(window.col_off,window.row_off,window.width,window.height)
                    self.assertTrue(all(abs(x-round(x))<1e-6 for x in values))
                    exact=Window(*(int(round(x)) for x in values))
                    np.testing.assert_array_equal(a.read(),b.read(window=exact))

    def test_development_identity_and_all_four_reserves_stay_separate(self):
        value=registry();plan=load(FULL/'site_plan.json');site=plan['sites']['AU_AMRUN202309_FULL']
        self.assertEqual(site['mine_group'],'Amrun');self.assertTrue(site['not_a_new_blind_mine'])
        self.assertNotIn('Morenci',plan['training_sites'])
        for case_id in reserved_case_ids(value):
            case=value['cases'][case_id]
            self.assertNotEqual(site['mine_group'],case.get('mine_group') or case.get('mine_id') or case_id)
            self.assertEqual(case['model_predictions_run'],0)
        self.assertFalse((ROOT/'datasets/mine_benchmark_20261002/final_evaluation_once.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
