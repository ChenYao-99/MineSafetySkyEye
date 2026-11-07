"""Source version checks include helper/catalog edits and exclude credentials/data."""
from pathlib import Path
import importlib.util,sys,tempfile,unittest
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
from runtime_identity import runtime_fingerprint


class RuntimeIdentity(unittest.TestCase):
    def test_helper_edit_changes_identity_while_credentials_do_not(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
            root=Path(directory);(root/'backend.py').write_text('initial',encoding='utf-8')
            before=runtime_fingerprint(root,'backend')
            (root/'.env').write_text('PRIVATE_SENTINEL',encoding='utf-8')
            self.assertEqual(before,runtime_fingerprint(root,'backend'))
            (root/'case_archive.py').write_text('changed helper',encoding='utf-8')
            self.assertNotEqual(before,runtime_fingerprint(root,'backend'))

    def test_backend_expected_detector_matches_detector_computation(self):
        path=Path('D:/LMHLD/runtime_identity.py')
        spec=importlib.util.spec_from_file_location('detector_runtime_identity',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(runtime_fingerprint(path.parent,'detector'),module.runtime_fingerprint(path.parent))

    def test_stale_detector_is_rejected_before_payload_is_sent(self):
        import backend as service
        from unittest.mock import patch
        from io import BytesIO
        import json
        class Response(BytesIO):
            status=200
        with patch.object(service.urllib.request,'urlopen',return_value=Response(json.dumps({'status':'ok','runtime_sha256':'old'}).encode())) as opened:
            result=service.detector_request('/api/detect_upload',{'image':'must-not-be-sent'})
        self.assertEqual(result['status_code'],503)
        self.assertEqual(result['inference_status'],'runtime_version_mismatch')
        self.assertEqual(opened.call_count,1)
        self.assertTrue(str(opened.call_args.args[0]).endswith('/api/health'))

    def test_same_code_from_another_directory_is_not_a_compatible_detector(self):
        import backend as service
        self.assertFalse(service.detector_identity_matches({'service':'kat-detector',
            'runtime_sha256':service.EXPECTED_DETECTOR_RUNTIME_SHA256,'runtime_root':'D:/another-deployment'}))
        self.assertTrue(service.detector_identity_matches({'service':'kat-detector',
            'runtime_sha256':service.EXPECTED_DETECTOR_RUNTIME_SHA256,'runtime_root':service.LMHLD_DIR}))

    def test_health_probe_does_not_inherit_long_analysis_timeout(self):
        import backend as service
        from unittest.mock import patch
        with patch.object(service.urllib.request,'urlopen',side_effect=TimeoutError) as opened:
            self.assertIsNone(service.detector_request('/api/health',method='GET'))
        self.assertEqual(opened.call_args.kwargs['timeout'],3)


if __name__=='__main__':unittest.main(verbosity=2)
