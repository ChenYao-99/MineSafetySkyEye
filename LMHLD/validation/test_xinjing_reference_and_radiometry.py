# -*- coding: utf-8 -*-
import unittest,json
from pathlib import Path
import numpy as np
import cv2
import rasterio
from audit_development_radiometry import parse_xml
from mine_data_protocol import DATA,registry,verify_assets


class CalibrationReferenceTests(unittest.TestCase):
    def test_actual_XML_coefficients_and_derived_DN_preservation(self):
        audit=json.loads((DATA/'radiometry_audit_20261004/audit.json').read_text(encoding='utf-8'))
        for eid,folder in (('Amyntaio','Amyntaio'),('LasCruces','LasCruces'),('Hpakant','Hpakant'),('CN_XJ2023','Xinjing')):
            for period in ('pre','post'):
                r=audit['cases'][eid][period];self.assertEqual(r['DN_offsets'],[-1000.]*4 if eid=='CN_XJ2023' else [0.]*4)
                self.assertEqual(r['BOA_quantification'],10000)
                with rasterio.open(DATA/f'development/{folder}/{period}_BGRN_10m.tif') as f:old=f.read();geometry=f.transform
                with rasterio.open(r['derivative']) as f:
                    np.testing.assert_array_equal(old,f.read());self.assertEqual(geometry,f.transform)
                    np.testing.assert_allclose(f.scales,r['scales']);np.testing.assert_allclose(f.offsets,r['reflectance_offsets'])
        self.assertTrue(verify_assets())

    def test_modern_XML_without_offsets_is_not_treated_as_zero(self):
        bad=b'<root><PROCESSING_BASELINE>04.00</PROCESSING_BASELINE><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE></root>'
        with self.assertRaisesRegex(ValueError,'provide all selected BOA offsets'):parse_xml(bad)

    def test_partial_duplicate_or_nonfinite_offsets_cannot_create_calibration(self):
        base='<root><PROCESSING_BASELINE>04.00</PROCESSING_BASELINE><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>{}</root>'
        rows=''.join(f'<BOA_ADD_OFFSET band_id="{i}">-1000</BOA_ADD_OFFSET>' for i in (1,2,3,7))
        with self.assertRaises(ValueError):parse_xml(base.format(rows.replace('band_id="7"','band_id="9"')).encode())
        with self.assertRaises(ValueError):parse_xml(base.format(rows+'<BOA_ADD_OFFSET band_id="1">-1000</BOA_ADD_OFFSET>').encode())
        with self.assertRaises(ValueError):parse_xml(base.format(rows.replace('-1000','nan',1)).encode())

    def test_historical_event_reference_has_unknowns_and_no_safety_claim(self):
        folder=DATA/'development/Xinjing/reference_v1_20261004';r=json.loads((folder/'label_reference.json').read_text(encoding='utf-8'))
        self.assertTrue(r['not_new_blind_test']);self.assertFalse(r['author_original_expert_GIS_received']);self.assertFalse(r['independent_project_expert_review'])
        with rasterio.open(folder/'labels.tif') as f:y=f.read(1)
        with rasterio.open(folder/'confidence.tif') as f:w=f.read(1)
        with rasterio.open(folder/'published_footprint.tif') as f:body=f.read(1)>0
        with rasterio.open(folder/'normal_mask.tif') as f:normal=f.read(1)>0
        self.assertEqual(int((y==1).sum()),1188);self.assertEqual(int(normal.sum()),1313);self.assertFalse(np.any((y==1)&~body))
        self.assertFalse(np.any(normal&(cv2.dilate(body.astype(np.uint8),np.ones((21,21),np.uint8))>0)))
        self.assertFalse(np.any(w[y==255]!=0));self.assertTrue(r['pair_alignment']['raw_pair_gate_pass'])
        self.assertEqual(registry()['cases']['CN_XJ2023']['role'],'development')
        self.assertFalse((DATA/'final_evaluation_once.json').exists())

    def test_SWIR_nearest_resampling_is_not_new10m_sensor_detail(self):
        q=json.loads((DATA/'radiometry_audit_20261004/Xinjing/winter_observation_audit.json').read_text(encoding='utf-8'))
        for r in q.values():
            self.assertEqual(r['native_SWIR_resolution_m'],20);self.assertEqual(r['analysis_grid_m'],10);self.assertTrue(r['native_detail_not_increased'])
            self.assertFalse(r['model_predictions_used']);self.assertTrue(r['no_old_labels_changed'])


if __name__=='__main__':unittest.main(verbosity=2)
