"""Observed support and cropped fragments must not invent changes or outside risk."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detector_service as service
from detector import LandslideDetector


class Stub(LandslideDetector):
    def __init__(self, maps): self.maps=iter(maps);self.threshold=.5;self.device='cpu';self.patch_size=32
    def detect_image(self, image, **kw): return next(self.maps).astype(np.float32)


class ScopeStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.client=service.app.test_client()
        self.image=np.random.default_rng(12).uniform(10,200,(128,128,4)).astype(np.float32)
    def upload(self, maps, valid, roi=None):
        meta={'nir_source':'red_proxy','format':'PNG','note':'test','_valid_mask':valid}
        with patch.object(service,'_prepare_upload',return_value=(self.image,meta)), patch.object(service,'input_split',return_value={'reserved':False}), patch.object(service,'_load_upload_detector',return_value=(Stub(maps),'minmax',1)):
            return self.client.post('/api/detect_upload',json={'image':'fixture','roi':roi})
    def pair(self, maps, valid):
        meta={'nir_source':'red_proxy','format':'PNG','note':'test','_valid_mask':valid}
        with patch.object(service,'_prepare_upload',return_value=(self.image,meta)), patch.object(service,'input_split',return_value={'reserved':False}), patch.object(service,'_load_upload_detector',return_value=(Stub(maps),'minmax',1)):
            return self.client.post('/api/compare',json={'image_t1':'fixture','image_t2':'fixture','registration':'prealigned'})
    def test_added_and_total_ratios_use_common_observed_denominator(self):
        valid=np.ones((128,128),bool);valid[:,:32]=False
        first=np.zeros(valid.shape,np.uint8);second=first.copy();second[10:30,50:70]=1
        response=self.pair([first,second],valid);self.assertEqual(response.status_code,200,response.get_json())
        stats=response.get_json()['stats'];self.assertEqual(stats['added_ratio'],round(400/int(valid.sum()),4));self.assertEqual(stats['added_ratio'],stats['t2_ratio'])
    def test_net_ratio_uses_counts_before_rounding(self):
        valid=np.ones((128,128),bool);a=np.zeros(valid.shape,np.uint8);b=a.copy();a.flat[0]=1;b.flat[:2]=1
        response=self.pair([a,b],valid);self.assertEqual(response.status_code,200,response.get_json())
        self.assertEqual(response.get_json()['stats']['net_ratio'],round(1/a.size,4))
    def test_roi_fragment_does_not_inflate_outside_count(self):
        valid=np.ones((128,128),bool);mask=np.zeros(valid.shape,np.uint8);mask[20,55:65]=1
        response=self.upload([mask],valid,[0,0,64,128]);self.assertEqual(response.status_code,200,response.get_json())
        r=response.get_json()['roi'];self.assertEqual(r['outside_reported_pixels'],1);self.assertEqual(r['risk_outside'],round(1/(128*64),4))
    def test_roi_with_no_observed_outside_has_no_outside_level(self):
        valid=np.ones((128,128),bool);mask=np.zeros(valid.shape,np.uint8)
        for roi in ([0,0,128,128],[0,0,64,128]):
            v=valid.copy()
            if roi[2]==64:v[:,64:]=False
            response=self.upload([mask],v,roi);self.assertEqual(response.status_code,200,response.get_json())
            r=response.get_json()['roi'];self.assertFalse(r['outside_available']);self.assertIsNone(r['risk_outside']);self.assertIsNone(r['level_outside'])
    def test_vectorized_classes_retain_existing_component_semantics(self):
        import cv2
        rng=np.random.default_rng(4)
        for _ in range(8):
            a=(rng.random((40,41))>.6).astype(np.uint8);b=(rng.random(a.shape)>.6).astype(np.uint8)
            actual=service._classify_changes(a,b);expected={k:np.zeros_like(a) for k in ('added','expanded','stable','removed')}
            n,lab,st,_=cv2.connectedComponentsWithStats(b,8)
            for i in range(1,n):
                comp=lab==i;overlap=int(a[comp].sum());kind='added' if overlap==0 else 'expanded' if st[i,cv2.CC_STAT_AREA]>overlap*service.EXPAND_RATIO else 'stable';expected[kind][comp]=1
            n,lab,st,_=cv2.connectedComponentsWithStats(((a>0)&(b==0)).astype(np.uint8),8)
            for i in range(1,n):
                if st[i,cv2.CC_STAT_AREA]>=10:expected['removed'][lab==i]=1
            for k in expected:np.testing.assert_array_equal(actual[k],expected[k])


if __name__=='__main__':unittest.main()
