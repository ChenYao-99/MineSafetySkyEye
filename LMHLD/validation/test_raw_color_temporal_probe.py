# -*- coding: utf-8 -*-
import unittest
import numpy as np
from raw_color_temporal_probe import frame_features,features


class RawFeatureTests(unittest.TestCase):
    def test_RGB_ratios_survive_common_gain_and_ignore_fourth_channel(self):
        rng=np.random.default_rng(11);x=rng.uniform(30,500,(32,32,4)).astype(np.float32)
        a=frame_features(x);b=frame_features(x*3);np.testing.assert_allclose(a,b,atol=1e-5)
        np.testing.assert_allclose(a,frame_features(x*.0001),atol=1e-5)
        changed=x.copy();changed[:,:,3]*=10;np.testing.assert_array_equal(a,frame_features(changed))

    def test_unavailable_or_identity_pre_has_zero_change_and_explicit_status(self):
        x=np.ones((32,32,4),np.float32)*200;f=frame_features(x);n=f.shape[2]
        a=features(x,x,'RGBpair');np.testing.assert_array_equal(a[:,:,n:3*n],0);self.assertTrue(np.all(a[:,:,-1]==1))
        b=features(x,None,'RGBpair');np.testing.assert_array_equal(b[:,:,n:3*n],0);self.assertTrue(np.all(b[:,:,-1]==0))

    def test_NIR_arm_rejects_unverified_or_three_band_input(self):
        x=np.ones((32,32,4),np.float32)
        with self.assertRaises(ValueError):features(x,x,'NIRpair',actual_nir=False)
        with self.assertRaises(ValueError):features(x[:,:,:3],x[:,:,:3],'NIRpair',actual_nir=True)


if __name__=='__main__':unittest.main(verbosity=2)
