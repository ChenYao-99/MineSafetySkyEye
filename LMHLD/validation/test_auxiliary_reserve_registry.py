"""Protect new auxiliary holdouts without decoding or predicting their pixels."""
from pathlib import Path
import tempfile,json,sys,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import protected_inputs as guard


class AuxiliaryReserveTests(unittest.TestCase):
    def test_known_hash_or_declaration_cannot_be_renamed_into_development(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache/runtime/tmp') as folder:
            base=Path(folder);(base/'benchmark').mkdir()
            case={'role':'reserved_final_test','scope':'auxiliary_change_not_accident_truth',
                  'input_sha256_values':['a'*64],'protected_geographic_extent_wgs84':[10,20,11,21]}
            (base/'input_reserve_registry.json').write_text(json.dumps({'schema_version':1,'cases':{'aux':case}}),encoding='utf-8')
            with patch.object(guard,'DATA',base/'benchmark'):
                raw=guard.input_split({'input_sha256':'a'*64,'declared_data_role':'development'})
                self.assertTrue(raw['reserved']);self.assertTrue(raw['identity_verified'])
                self.assertTrue(guard.input_split({'declared_case_id':'aux','declared_data_role':'development'})['reserved'])
                spatial=guard.input_split({'georeference':{'width':10,'height':10,'crs':'EPSG:4326','transform':[.01,0,10.5,0,-.01,20.8]}})
                self.assertTrue(spatial['reserved']);self.assertIn('auxiliary',spatial['scope'])
                self.assertFalse(guard.input_split({'input_sha256':'b'*64})['reserved'])


if __name__=='__main__':unittest.main()
