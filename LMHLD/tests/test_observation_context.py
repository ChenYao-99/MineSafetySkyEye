"""Missing observations cannot supply different context to a paired comparison."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detector_service as service
from detector import LandslideDetector
from input_quality import prepare_inference_context


class ContextSensitiveDetector(LandslideDetector):
    def __init__(self):
        self.threshold=.5;self.patch_size=32;self.device='cpu';self.inputs=[]
    def predict_batch(self,patches):return np.asarray(patches[:,0]/255,np.float32)
    def detect_image(self,image,**kwargs):
        self.inputs.append(np.asarray(image).copy())
        return super().detect_image(image,**kwargs)


class ObservationContextTests(unittest.TestCase):
    def setUp(self):
        self.image=np.random.default_rng(8).uniform(100,200,(128,128,4)).astype(np.float32)
        self.det=ContextSensitiveDetector();self.client=service.app.test_client()
    def pair(self,a,b,valid1,valid2):
        def meta(valid):return {'format':'PNG','nir_source':'red_proxy','note':'test','_valid_mask':valid,'nodata_pixels':int((~valid).sum())}
        with patch.object(service,'_prepare_upload',side_effect=[(a,meta(valid1)),(b,meta(valid2))]),patch.object(service,'input_split',return_value={'reserved':False}),patch.object(service,'_load_upload_detector',return_value=(self.det,'minmax',1)):
            return self.client.post('/api/compare',json={'image_t1':'fixture','image_t2':'fixture','registration':'prealigned'})
    def test_identical_common_observations_do_not_produce_missing_edge_changes(self):
        v1=np.ones((128,128),bool);v2=v1.copy();v1[:,:16]=False;v2[:,112:]=False
        a=self.image.copy();b=a.copy();a[~v1]=0;b[~v2]=0
        response=self.pair(a,b,v1,v2);self.assertEqual(response.status_code,200,response.get_json().get('error'))
        np.testing.assert_array_equal(self.det.inputs[0],self.det.inputs[1])
        stats=response.get_json()['stats'];self.assertEqual(stats['n_added_regions'],0);self.assertEqual(stats['n_expanded_regions'],0);self.assertEqual(stats['n_removed_regions'],0);self.assertEqual(stats['net_ratio'],0)
        self.assertEqual(stats['valid_pixels'],128*96)
    def test_noncommon_values_do_not_change_inputs_or_quantitative_facts(self):
        v1=np.ones((128,128),bool);v2=v1.copy();v1[:,:16]=False;v2[:,112:]=False
        common=v1&v2;a=self.image.copy();b=a.copy();a[:,112:]=1.e6;b[:,:16]=0
        response=self.pair(a,b,v1,v2);self.assertEqual(response.status_code,200,response.get_json().get('error'))
        np.testing.assert_array_equal(self.det.inputs[0][common],self.image[common]);np.testing.assert_array_equal(self.det.inputs[0],self.det.inputs[1])
        self.assertEqual(response.get_json()['band_meta']['t1']['nodata_pixels'],128*16)

    def test_context_fill_preserves_observations_and_all_channel_relationships(self):
        valid=np.ones((128,128),bool);valid[30:35,30:35]=False
        image=self.image.copy();image[~valid]=np.nan;before=image.copy()
        filled,record=prepare_inference_context(image,valid)
        np.testing.assert_array_equal(filled[valid],image[valid]);self.assertTrue(np.isfinite(filled).all())
        np.testing.assert_array_equal(image,before);self.assertEqual(record['context_only_pixels'],25);self.assertFalse(record['filled_pixels_are_observations'])
        original_tuples={tuple(x) for x in image[valid]}
        self.assertTrue(all(tuple(x) in original_tuples for x in filled[~valid]))

    def test_fully_observed_route_is_numerically_unchanged(self):
        valid=np.ones((128,128),bool);filled,record=prepare_inference_context(self.image,valid)
        self.assertIs(filled,self.image);self.assertEqual(record['context_only_pixels'],0)
        with self.assertRaises(ValueError):prepare_inference_context(self.image,np.zeros_like(valid))
        with self.assertRaises(ValueError):prepare_inference_context(self.image,np.ones((2,2),bool))

    def test_single_roi_keeps_missing_pixels_excluded_from_statistics(self):
        valid=np.ones((128,128),bool);valid[:,:16]=False
        image=self.image.copy();image[~valid]=0
        meta={'format':'PNG','nir_source':'red_proxy','note':'test','_valid_mask':valid}
        with patch.object(service,'_prepare_upload',return_value=(image,meta)),patch.object(service,'input_split',return_value={'reserved':False}),patch.object(service,'_load_upload_detector',return_value=(self.det,'minmax',1)):
            response=self.client.post('/api/detect_upload',json={'image':'fixture','roi':[0,0,64,128]})
        self.assertEqual(response.status_code,200,response.get_json().get('error'));r=response.get_json()
        self.assertEqual(r['stats']['valid_pixels'],128*48);self.assertEqual(r['roi']['valid_area_px'],128*48)
        self.assertEqual(r['input_quality']['inference_context']['context_only_pixels'],128*16)
        np.testing.assert_array_equal(self.det.inputs[0][valid],self.image[valid])

    def test_true_change_inside_common_observations_is_retained(self):
        v1=np.ones((128,128),bool);v2=v1.copy();v1[:,:16]=False;v2[:,112:]=False
        a=self.image.copy();b=a.copy();b[48:80,48:80,0]+=500
        response=self.pair(a,b,v1,v2);self.assertEqual(response.status_code,200,response.get_json().get('error'))
        np.testing.assert_array_equal(self.det.inputs[0][48:80,48:80],a[48:80,48:80])
        np.testing.assert_array_equal(self.det.inputs[1][48:80,48:80],b[48:80,48:80])
        stats=response.get_json()['stats'];self.assertNotEqual(stats['t1_ratio'],stats['t2_ratio'])


if __name__=='__main__':unittest.main()
