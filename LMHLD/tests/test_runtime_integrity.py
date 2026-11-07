"""Concurrent request isolation, rejected artifacts, input defects and bounded streaming."""
import base64
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import detector_service as service
from detector import LandslideDetector, ModelComponentError, ModelNumericalError
from model_registry import ModelRegistry, ModelArtifactError, file_sha256
from upload_utils import decode_upload


def encoded(image):
    ok,data=cv2.imencode('.png',image)
    assert ok
    return base64.b64encode(data).decode()


class Stub(LandslideDetector):
    def __init__(self):
        self.threshold,self.device,self.patch_size=.4,'cpu',32
        self.batches=[]
    def predict_batch(self,x):
        self.batches.append(len(x))
        return np.clip(x[:,0]/255,0,1).astype(np.float32)


class RuntimeIntegrity(unittest.TestCase):
    def setUp(self):
        self.image=np.random.default_rng(9).integers(1,255,(65,97,3),dtype=np.uint8)
        self.stub=Stub()
        self.loader=patch.object(service,'_load_upload_detector',return_value=(self.stub,'minmax',1.0))
        self.load_mock=self.loader.start();self.addCleanup(self.loader.stop)
        self.client=service.app.test_client()

    def test_overlapping_requests_reject_second_and_keep_first_model(self):
        entered,release=threading.Event(),threading.Event()
        result=[]
        def predict(image,**kwargs):
            entered.set()
            if not release.wait(8):raise RuntimeError('test did not release inference')
            return np.zeros(image.shape[:2],np.float32)
        def worker():
            with service.app.test_client() as client:
                result.append(client.post('/api/detect_upload',json={'image':encoded(self.image)}))
        with patch.object(self.stub,'detect_image',side_effect=predict):
            thread=threading.Thread(target=worker);thread.start()
            try:
                self.assertTrue(entered.wait(5))
                busy=self.client.post('/api/detect_upload',json={'image':encoded(self.image),'model_tag':'realfit'})
                self.assertEqual(busy.status_code,503)
                self.assertEqual(busy.get_json()['inference_status'],'busy')
                self.assertEqual(busy.headers['Retry-After'],'2')
                self.assertEqual(self.client.get('/api/health').status_code,200)
                self.assertEqual(self.load_mock.call_count,1)
            finally:
                release.set();thread.join(8)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0].status_code,200)
        self.assertEqual(result[0].get_json()['model_tag'],'mine')
        self.assertFalse(service._inference_lock.locked())

    def test_black_uniform_and_transparent_images_never_produce_low_risk(self):
        for value in (0,200):
            response=self.client.post('/api/detect_upload',json={'image':encoded(np.full((32,32,3),value,np.uint8))})
            self.assertEqual(response.status_code,422)
            self.assertNotIn('risk_level',response.get_json())
        rgba=np.dstack([self.image,np.zeros(self.image.shape[:2],np.uint8)])
        response=self.client.post('/api/detect_upload',json={'image':encoded(rgba)})
        self.assertEqual(response.status_code,400)
        self.assertNotIn('stats',response.get_json())

    def test_alpha_mask_excludes_hidden_prediction_and_returns_trace(self):
        alpha=np.full(self.image.shape[:2],255,np.uint8);alpha[:,50:]=0
        image,meta=decode_upload(encoded(np.dstack([self.image,alpha])))
        self.assertEqual(meta['observed_pixels'],65*50)
        self.assertEqual(image[:,50:].sum(),0)
        response=self.client.post('/api/detect_upload',json={'image':encoded(np.dstack([self.image,alpha]))})
        body=response.get_json()
        self.assertEqual(body['stats']['valid_pixels'],65*50)
        self.assertEqual(body['request_id'],response.headers['X-KAT-Request-ID'])
        self.assertEqual(body['input_quality']['weather_assessment'],'not_verified')

    def test_model_failure_releases_lock_without_conclusion(self):
        for error in (ModelComponentError('failed'),torch.cuda.OutOfMemoryError('failed')):
            with patch.object(self.stub,'detect_image',side_effect=error):
                response=self.client.post('/api/detect_upload',json={'image':encoded(self.image)})
            self.assertEqual(response.status_code,503)
            self.assertNotIn('stats',response.get_json())
            self.assertFalse(service._inference_lock.locked())

    def test_capacity_precedes_JSON_allocation_and_invalid_body_releases_slot(self):
        service._inference_lock.acquire()
        try:
            with patch('flask.wrappers.Request.get_json',side_effect=AssertionError('must reject before decoding a body')):
                response=self.client.post('/api/detect_upload',data='not json',content_type='application/json')
            self.assertEqual(response.status_code,503)
        finally:service._inference_lock.release()
        response=self.client.post('/api/detect_upload',data='not json',content_type='application/json')
        self.assertEqual(response.status_code,400)
        self.assertFalse(service._inference_lock.locked())

    def test_oom_cleanup_occurs_while_capacity_is_still_owned(self):
        ownership=[]
        with patch.object(self.stub,'detect_image',side_effect=torch.cuda.OutOfMemoryError('test')), \
             patch.object(service,'_release_models',side_effect=lambda:ownership.append(service._inference_lock.locked())):
            response=self.client.post('/api/detect_upload',json={'image':encoded(self.image)})
        self.assertEqual(response.status_code,503)
        self.assertEqual(ownership,[True])
        self.assertFalse(service._inference_lock.locked())

    def test_missing_requested_region_never_runs_previous_region(self):
        for num,start in ((0,0),(65,0),(True,0),(1,-1),('2',0)):
            response=self.client.post('/api/detect',json={'num':num,'start':start})
            self.assertEqual(response.status_code,400)
        with patch.object(service,'_load_detector_region',return_value=None):
            response=self.client.post('/api/detect',json={'region':service.DEFAULT_REGION})
        self.assertEqual(response.status_code,503)
        self.assertNotIn('samples',response.get_json())

    def test_conflicting_evidence_is_unknown_and_not_low_risk(self):
        current=Stub();current.region=service.FUSION_REGION
        images=np.full((1,4,32,32),.99,np.float32)
        labels=np.zeros((1,1,32,32),np.uint8)
        with patch.object(service,'_load_detector_region',return_value=current), \
             patch.object(service,'detector',current), \
             patch.object(service,'_get_region_npy',return_value=(images,labels)), \
             patch.object(current,'predict_batch',return_value=np.full((1,32,32),.99,np.float32)), \
             patch.object(service,'_load_detector32',return_value=current), \
             patch.object(current,'detect_image',return_value=np.full((32,32),.01,np.float32)):
            response=self.client.post('/api/detect',json={'region':service.FUSION_REGION,'num':1,'fuse':True,'ds_reject':True})
        self.assertEqual(response.status_code,200)
        sample=response.get_json()['samples'][0]
        self.assertEqual(sample['risk_level'],'unknown')
        self.assertEqual(sample['stats']['unverified_pixels'],1024)
        self.assertEqual(sample['stats']['valid_pixels'],0)
        self.assertFalse(sample['stats']['area_ratio_available'])
        self.assertIsNone(sample['stats']['iou'])
        self.assertIn('待复核',sample['risk_name'])

    def test_negative_result_says_not_detected_instead_of_safe(self):
        with patch.object(self.stub,'detect_image',return_value=np.zeros(self.image.shape[:2],np.float32)):
            body=self.client.post('/api/detect_upload',json={'image':encoded(self.image)}).get_json()
        self.assertEqual(body['risk_level'],'blue')
        self.assertIn('未检出',body['risk_name'])
        self.assertNotIn('低风险',body['risk_name'])

    def test_streaming_identical_normalization_with_bounded_window_allocation(self):
        image=np.dstack([self.image,self.image[:,:,2]]).astype(np.float32)
        for norm in (None,'minmax','zscore'):
            reference=self.stub.detect_image(image,patch_norm=norm,max_batch=512)
            self.stub.batches=[]
            with patch('detector.np.stack',wraps=np.stack) as allocation:
                actual=self.stub.detect_image(image,patch_norm=norm,max_batch=2)
                self.assertTrue(all(len(call.args[0])<=2 for call in allocation.call_args_list))
            np.testing.assert_array_equal(actual,reference)
            self.assertLessEqual(max(self.stub.batches),2)

    def test_overlapping_nondivisible_stride_covers_bottom_and_right(self):
        image=np.ones((65,97,4),np.float32)*255
        actual=self.stub.detect_image(image,stride=21,max_batch=2)
        np.testing.assert_array_equal(actual,np.ones((65,97),np.float32))
        with self.assertRaises(ValueError):self.stub.detect_image(image,stride=33)

    def test_synth_is_bounded_and_failure_cannot_fall_back(self):
        image=np.dstack([self.image,self.image[:,:,2]]).astype(np.float32)
        calls=[]
        def good(x):
            calls.append(len(x));return x[:,:1]
        self.stub.detect_image(image,patch_norm='minmax',nir_synth_model=good,max_batch=2)
        self.assertLessEqual(max(calls),2)
        def bad(x):raise RuntimeError('broken component')
        with self.assertRaises(ModelComponentError):self.stub.detect_image(image,nir_synth_model=bad,max_batch=2)
        def invalid(x):return torch.full((len(x),1,32,32),float('nan'))
        with self.assertRaises(ModelNumericalError):self.stub.detect_image(image,nir_synth_model=invalid,max_batch=2)


class RegistryIntegrity(unittest.TestCase):
    def test_content_and_protocol_changes_reject_previously_cached_artifact(self):
        folder=ROOT/'cache/tests';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            root=Path(temp);weights=root/'weights.pth';weights.write_bytes(b'original')
            protocol={'threshold':.4,'patch_size':32}
            config={'schema_version':1,'artifacts':{'upload:mine':{'path':'weights.pth','sha256':file_sha256(weights),
                    'protocol':protocol,'validation_status':'development_recall_insufficient'}}}
            manifest=root/'registry.json';manifest.write_text(json.dumps(config),encoding='utf-8')
            registry=ModelRegistry(manifest,root)
            self.assertEqual(registry.verify('upload:mine',weights,protocol)['weights_sha256'],file_sha256(weights))
            with self.assertRaises(ModelArtifactError):registry.verify('upload:mine',weights,{'threshold':.5,'patch_size':32})
            weights.write_bytes(b'modified')
            with self.assertRaises(ModelArtifactError):registry.verify('upload:mine',weights,protocol)
            config['artifacts']['upload:mine']['path']='../outside.pth'
            manifest.write_text(json.dumps(config),encoding='utf-8')
            with self.assertRaises(ModelArtifactError):registry.verify('upload:mine',weights,protocol)


if __name__=='__main__':unittest.main(verbosity=2)
