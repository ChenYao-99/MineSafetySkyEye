import json,sys,unittest
from pathlib import Path
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DATA=ROOT/'datasets/mine_benchmark_20261002'
OUT=DATA/'development/Brumadinho/official_boundary_review_v1_20261004'


class OfficialBoundaryProvenance(unittest.TestCase):
    def test_actual_government_vector_originals_and_derivatives_are_frozen(self):
        audit=json.loads((OUT/'boundary_audit.json').read_text(encoding='utf-8'))
        for name,digest in audit['original_assets'].items():self.assertEqual(file_sha256(DATA/name),digest)
        for name,digest in audit['derived_asset_sha256'].items():self.assertEqual(file_sha256(OUT/name),digest)
        source=ROOT/'datasets/operational_boundary_sources_20261004/brumadinho/official_impact_manifest.json'
        self.assertEqual(file_sha256(source),audit['source_manifest_sha256'])
        manifest=json.loads(source.read_text(encoding='utf-8'))
        for row in manifest['sources']:self.assertEqual(file_sha256(row['path']),row['sha256'])
    def test_auxiliary_boundary_is_not_automatically_truth_or_normal_background(self):
        audit=json.loads((OUT/'boundary_audit.json').read_text(encoding='utf-8'))
        self.assertFalse(audit['accepted_for_training_or_scoring']);self.assertEqual(audit['model_inference_run'],0)
        self.assertTrue(audit['comparison_is_not_model_accuracy']);self.assertTrue(audit['exact_original_Pleiades_acquisition_date_not_given'])
        self.assertEqual(audit['original_official_crs'],'EPSG:4674')
        with rasterio.open(OUT/'government_impact_outline_on_original_grid.tif') as f:
            y=f.read(1);self.assertEqual(str(f.crs),'EPSG:32723');self.assertEqual(f.res,(10,10))
            self.assertEqual(f.tags()['labels_accepted'],'false');self.assertEqual(set(np.unique(y)),{0,1})


if __name__=='__main__':unittest.main(verbosity=2)
