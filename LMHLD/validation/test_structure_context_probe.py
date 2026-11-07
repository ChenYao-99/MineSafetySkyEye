import unittest
import numpy as np
from structure_context_probe import structure_features,structural_features,CASES
from mine_data_protocol import RESERVED


class StructuralInformation(unittest.TestCase):
    def setUp(self):self.image=np.random.default_rng(92).uniform(2,2000,(45,65,4)).astype(np.float32)
    def test_common_gain_and_unused_fourth_channel(self):
        expected=structure_features(self.image)
        np.testing.assert_allclose(structure_features(self.image*.0001),expected,rtol=5e-4,atol=5e-5)
        modified=self.image.copy();modified[:,:,3]*=100
        np.testing.assert_array_equal(structure_features(modified),expected)
    def test_rotation_invariant_structure_and_constant_is_not_informative(self):
        expected=structure_features(self.image)
        rotated=structure_features(np.rot90(self.image))
        np.testing.assert_allclose(rotated,np.rot90(expected),rtol=1e-3,atol=5e-5)
        self.assertLess(float(structure_features(np.ones((45,65,4),np.float32)).max()),1e-3)
    def test_unavailable_pre_has_no_fabricated_change(self):
        a=structural_features(self.image,None,'RGBstructurePair')
        self.assertEqual(a[:,:,-32:].sum(),0)
        before=self.image.copy();before[:,:,:3]*=2
        b=structural_features(self.image,before,'RGBstructurePair',np.zeros((45,65),bool))
        np.testing.assert_array_equal(a,b)
        self.assertFalse(RESERVED.intersection(CASES))


if __name__=='__main__':unittest.main(verbosity=2)
