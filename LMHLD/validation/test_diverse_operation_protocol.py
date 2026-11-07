import json,sys,unittest
from pathlib import Path
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import registry,RESERVED,DATA as MAIN
DATA=ROOT/'datasets/diverse_operation_pairs_20261004';REVIEW=DATA/'review_v1_20261004';INPUTS=DATA/'inputs_v1_20261004'


class DiverseOperationContracts(unittest.TestCase):
    def test_mine_roles_and_source_split_preserved(self):
        p=json.loads((DATA/'site_plan.json').read_text(encoding='utf-8'))
        self.assertEqual(p['training_sites'],['Amrun','Carajas','Faxe']);self.assertEqual(p['normal_holdout_sites'],['Morenci'])
        self.assertEqual(p['sites']['Morenci']['source_split'],'test')
        self.assertFalse(set(p['training_sites'])&set(p['normal_holdout_sites']))
        for site in p['sites'].values():
            b=site['bbox_wgs84']
            for limit in p['reserved_envelopes_excluded'].values():self.assertFalse(b[0]<=limit[2] and b[2]>=limit[0] and b[1]<=limit[3] and b[3]>=limit[1])
    def test_frozen_original_products_and_actual_calibration(self):
        for site in ('Morenci','Amrun','Carajas','Faxe'):
            p=json.loads((DATA/site/'acquisition.json').read_text(encoding='utf-8'))
            for period,row in p['images'].items():
                for name,key in ((period+'_BGRN_10m.tif','sha256'),(period+'_SCL_10m.tif','scl_sha256'),(period+'_product.xml','product_XML_sha256')):
                    self.assertEqual(file_sha256(DATA/site/name),row[key])
                with rasterio.open(DATA/site/(period+'_BGRN_10m.tif')) as f:
                    self.assertEqual(f.descriptions,('B02','B03','B04','B08'));self.assertEqual(f.res,(10,10))
                    np.testing.assert_allclose(f.scales,row['calibration']['scales']);np.testing.assert_allclose(f.offsets,row['calibration']['reflectance_offsets'])
    def test_weak_normals_only_and_uncertainty_not_training_background(self):
        p=json.loads((REVIEW/'review_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(p['status'],'accepted_project_weak_appearance_reference');self.assertFalse(p['model_predictions_used_for_selection'])
        self.assertEqual(p['expert_feedback_received'],0);self.assertFalse(p['geotechnical_stability_verified'])
        for name,digest in p['frozen_assets'].items():self.assertEqual(file_sha256(REVIEW/name),digest)
        for site,entry in p['sites'].items():
            self.assertTrue(entry['pair_alignment']['raw_pair_gate_pass'])
            with rasterio.open(REVIEW/site/'labels.tif') as f:y=f.read(1)
            with rasterio.open(REVIEW/site/'confidence.tif') as f:w=f.read(1)
            self.assertEqual(set(np.unique(y)),{0,255});self.assertTrue(np.all(w[y==255]==0));self.assertTrue(np.all(w[y==0]==.25))
            for period in ('pre','post'):
                with rasterio.open(DATA/site/(period+'_SCL_10m.tif')) as f:scl=f.read(1)
                self.assertFalse(np.isin(scl[y==0],[0,1,3,6,8,9,10,11]).any())
    def test_no_predictions_during_input_preparation_and_no_reserved_exposure(self):
        p=json.loads((INPUTS/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(p['model_predictions_during_preparation'],0)
        for name,digest in p['arrays'].items():self.assertEqual(file_sha256(INPUTS/name),digest)
        self.assertEqual(p['holdout_sites'],['Morenci'])
        for case in RESERVED:self.assertEqual(registry()['cases'][case]['model_predictions_run'],0)
        self.assertFalse((MAIN/'final_evaluation_once.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
