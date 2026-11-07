import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
BASE=ROOT/'experiments/rgb_context_residual_pilot_20261004'
QUALITY=ROOT/'experiments/quality_brightness_inputs_20261004'
OUT=ROOT/'experiments/diverse_material_validation_20261004'


class ContrastProtocolContracts(unittest.TestCase):
    def test_brightness_contrast_preserves_ratios_availability_and_label_inputs(self):
        p=json.loads((QUALITY/'protocol.json').read_text(encoding='utf-8'))
        for case in p['cases']:
            old=np.load(BASE/f'{case}_features.npy',mmap_mode='r');new=np.load(QUALITY/f'{case}_features.npy',mmap_mode='r')
            np.testing.assert_array_equal(old[:3],new[:3]);np.testing.assert_array_equal(old[12],new[12])
            self.assertTrue(np.isfinite(new).all())
        for name,digest in p['arrays'].items():self.assertEqual(file_sha256(QUALITY/name),digest)
        for name,digest in p['original_dependencies'].items():self.assertEqual(file_sha256(name),digest)
    def test_three_contrasts_have_separate_factors_and_no_holdout_sampler(self):
        variants={'diverse_material_data_20261004':(False,True),'quality_brightness_only_20261004':(True,False),'quality_diverse_material_20261004':(True,True)}
        for family,(quality,aux) in variants.items():
            folder=ROOT/'experiments'/family;p=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(p['uses_masked_brightness_statistics'],quality);self.assertEqual(p['uses_diverse_auxiliary_data'],aux)
            self.assertEqual(set(p['aux_sites']),{'Amrun','Carajas','Faxe'});self.assertNotIn('Morenci',p['aux_sites'])
            self.assertTrue(p['original_sampler_RNG_unchanged']);self.assertEqual(p['threshold'],.5);self.assertEqual(p['epochs'],20)
            rows=json.loads((folder/'results.json').read_text(encoding='utf-8'))['models'];self.assertEqual(len(rows),4)
            for row in rows:self.assertEqual(file_sha256(row['weights']),row['weights_sha'])
    def test_normal_holdout_only_after_all_fits_and_unknowns_not_true_negatives(self):
        p=json.loads((OUT/'normal_holdout_protocol.json').read_text(encoding='utf-8'))
        for family,digest in p['source_result_sha256'].items():self.assertEqual(file_sha256(ROOT/f'experiments/{family}/results.json'),digest)
        rows=json.loads((OUT/'normal_holdout_results.json').read_text(encoding='utf-8'))['models'];self.assertEqual(len(rows),16)
        for row in rows:
            self.assertEqual(row['normal_holdout_site'],'Morenci');self.assertEqual(row['known_constructed_pixels'],1940)
            self.assertTrue(row['zero_pixels_outside_reference_not_counted_as_true_negatives'])
            self.assertFalse(row['geotechnical_no_accident_ground_truth']);self.assertEqual(row['protocol_sha256'],file_sha256(OUT/'normal_holdout_protocol.json'))


if __name__=='__main__':unittest.main(verbosity=2)
