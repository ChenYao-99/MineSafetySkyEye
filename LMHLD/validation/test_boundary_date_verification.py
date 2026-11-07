import json,sys,unittest
from pathlib import Path
import rasterio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DATA=ROOT/'datasets/mine_benchmark_20261002'
SOURCE=ROOT/'datasets/operational_boundary_sources_20261004/brumadinho'
OUT=DATA/'development/Brumadinho/official_boundary_review_v1_20261004'


class BoundaryDateContracts(unittest.TestCase):
    def test_publication_dates_and_different_provider_dates_do_not_become_capture(self):
        p=json.loads((OUT/'boundary_date_verification.json').read_text(encoding='utf-8'))
        self.assertFalse(p['SEMAD_boundary_image_capture_date_verified']);self.assertFalse(p['automatic_training_or_scoring_acceptance'])
        self.assertTrue(p['IBGE_date_cannot_be_assigned_to_SEMAD_vector']);self.assertFalse(p['IBGE_actual_dated_boundary_obtained'])
        self.assertEqual(p['IBGE_named_boundary_image_dates']['post_GeoEye1'],'2019-01-29')
        dates={r['value'] for r in p['official_SEMAD_metadata_dates']};self.assertIn('2020-10-26',dates)
        self.assertEqual(file_sha256(SOURCE/'official_impact_metadata.xml'),p['official_SEMAD_Pleiades_metadata_sha256'])
    def test_actual_same_day_sensor_quality_is_frozen_without_changing_originals(self):
        folder=DATA/'development/Brumadinho/exact_date_quality_20261004'
        p=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        raw=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/original_scenes/BR0001/BR0001__None__None__MASK.tif'
        self.assertEqual(file_sha256(raw),p['mask_sha']);self.assertEqual(p['new_model_predictions'],0)
        with rasterio.open(raw) as f:shape=f.shape;grid=f.transform;crs=f.crs
        for period,date in (('pre','2019-01-07'),('post','2019-02-01')):
            row=p['images'][period];self.assertEqual(row['datetime'][:10],date)
            for name,digest in row['assets'].items():self.assertEqual(file_sha256(folder/name),digest)
            with rasterio.open(folder/(period+'_SCL_original_grid.tif')) as f:
                self.assertEqual(f.shape,shape);self.assertEqual(f.transform,grid);self.assertEqual(f.crs,crs)


if __name__=='__main__':unittest.main(verbosity=2)
