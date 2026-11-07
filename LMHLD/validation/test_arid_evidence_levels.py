import json,sys,unittest
from pathlib import Path
import numpy as np,rasterio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import registry,RESERVED,DATA as MAIN
DATA=ROOT/'datasets/arid_accident_operation_20261004'


class AridEvidenceContracts(unittest.TestCase):
    def test_dated_reference_is_original_article_capture_not_jpeg_edit_timestamp(self):
        p=json.loads((DATA/'source_review/dated_satellite_reference.json').read_text(encoding='utf-8'))
        self.assertEqual(p['capture_date_verified_from_archived_article'],'2020-12-03')
        r=json.loads((DATA/'source_review/dated_archive_revision.json').read_text(encoding='utf-8'))
        self.assertTrue(r['date_verified']);self.assertEqual(r['archive_timestamp'],'20201218083125')
        for row in p['sources'].values():self.assertEqual(file_sha256(row['path']),row['sha256'])
    def test_same_mine_dates_not_independent_cases_and_uncertain_old_outline_not_fresh_truth(self):
        cases=registry()['cases']
        a,b=cases['ZA_GAMS2020S'],cases['ZA_GAMS2023W']
        self.assertEqual(a['mine_id'],b['mine_id']);self.assertEqual(a['role'],b['role']);self.assertEqual(a['role'],'development')
        p=json.loads((DATA/'ZA_GAMS2023W/presence_reference_v1/presence_reference.json').read_text(encoding='utf-8'))
        self.assertTrue(p['2022_black_polygon_not_2023_truth']);self.assertFalse(p['weak_presence_training_eligible'])
    def test_failed_source_registration_never_becomes_pixel_truth(self):
        reg=json.loads((DATA/'ZA_GAMS2020S/reference_registration_v1/registration.json').read_text(encoding='utf-8'))
        self.assertFalse(reg['registration_gate_pass'])
        for case in ('ZA_GAMS2020S','ZA_GAMS2023W'):
            p=json.loads((DATA/case/'presence_reference_v1/presence_reference.json').read_text(encoding='utf-8'))
            self.assertFalse(p['pixel_segmentation_training_or_IoU_scoring_eligible']);self.assertFalse(p['source_GIS_transferred'])
            with rasterio.open(DATA/case/'presence_reference_v1/pixel_labels_unknown.tif') as f:self.assertTrue(np.all(f.read(1)==255))
    def test_actual_raw_images_quality_and_coefficients_frozen(self):
        for case in ('ZA_GAMS2020S','ZA_GAMS2023W','FaxeDeposition2021','UK_NEWPORT2025'):
            p=json.loads((DATA/case/'acquisition.json').read_text(encoding='utf-8'))
            for period,row in p['images'].items():
                for name,digest in row['assets_sha256'].items():self.assertEqual(file_sha256(DATA/case/name),digest)
                with rasterio.open(DATA/case/f'{period}_BGRN_10m.tif') as f:
                    self.assertEqual(f.descriptions,('B02','B03','B04','B08'));self.assertEqual(f.res,(10,10))
                    np.testing.assert_allclose(f.scales,row['calibration']['scales']);np.testing.assert_allclose(f.offsets,row['calibration']['reflectance_offsets'])
    def test_mixed_inputs_keep_presence_unknown_and_operation_weak(self):
        p=json.loads((DATA/'mixed_supervision_inputs_v1/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(p['positive_pixel_truth_count'],0);self.assertEqual(p['presence_confidence'],.25)
        for name,digest in p['arrays'].items():self.assertEqual(file_sha256(DATA/'mixed_supervision_inputs_v1'/name),digest)
        self.assertTrue(np.all(np.load(DATA/'mixed_supervision_inputs_v1/presence_pixel_targets_unknown.npy')==255))
        y=np.load(DATA/'mixed_supervision_inputs_v1/operation_labels.npy');w=np.load(DATA/'mixed_supervision_inputs_v1/operation_confidence.npy')
        self.assertEqual(set(np.unique(y)),{0,255});self.assertTrue(np.all(w[y==255]==0));self.assertTrue(np.all(w[y==0]==.25))
        self.assertIn('FaxeDeposition2021',p['not_used_cases']);self.assertIn('UK_NEWPORT2023',p['not_used_cases'])
    def test_reserves_and_default_never_consumed_by_collection(self):
        for case in RESERVED:self.assertEqual(registry()['cases'][case]['model_predictions_run'],0)
        self.assertFalse((MAIN/'final_evaluation_once.json').exists())
        self.assertEqual(file_sha256(ROOT/'models_prior/real_ft_T1b_ours_seed42/best_model.pth'),'0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef')


if __name__=='__main__':unittest.main(verbosity=2)
