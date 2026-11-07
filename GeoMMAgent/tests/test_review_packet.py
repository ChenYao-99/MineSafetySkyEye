"""Offline handoff preserves manual locations, scoped facts, escaping and no reruns."""
from pathlib import Path
from unittest.mock import patch
import base64,io,json,sys,unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import backend as service
from archive_test_support import isolate_archive
from test_case_archive import fixture
from review_packet import review_html


def png():
    buffer=io.BytesIO();Image.new('RGB',(30,20),'#8c9f85').save(buffer,format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


class ReviewPacket(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service);self.client=service.app.test_client()
        value=fixture();value['detection'].update(size=[20,30],rgb_b64=png(),overlay_b64=png())
        value['final_answer']='<script>secret()</script>'
        self.saved=service.cache_report(value);self.rid=self.saved['report_id']
        self.updated=service.case_archive.submit_manual_annotation(self.rid,{'expected_revision':0,'box_px':[2,3,12,15],
            'reviewer':'内部功能验收','evidence':'<img src=x onerror=alert(1)>','note':'<script>alert(1)</script> 此位置待核查'})

    def test_offline_html_reads_current_revision_without_model_or_cloud_and_preserves_sources(self):
        with patch.object(service,'detector_request',side_effect=AssertionError('no inference')),patch.object(service,'analyze_with_agents',side_effect=AssertionError('no cloud')):
            response=self.client.post('/api/review_record',json={'report_id':self.rid,'format':'html'})
        self.assertEqual(response.status_code,200);text=base64.b64decode(response.get_json()['content_b64']).decode()
        self.assertIn('viewBox="0 0 30 20"',text);self.assertIn('<rect x="2.0" y="3.0" width="10.0" height="12.0"',text)
        self.assertIn(self.updated['archive']['snapshot_sha256'],text);self.assertIn('b'*64,text)
        self.assertIn('橙色框是人工核查位置，不是事故边界',text)
        self.assertNotIn('<script>alert(1)</script>',text);self.assertNotIn('<img src=x onerror=alert(1)>',text)
        self.assertIn('&lt;script&gt;',text);self.assertIn("connect-src 'none'",text)
        self.assertEqual(self.client.post('/api/review_record',json={'report_id':self.rid,'format':'unsupported'}).status_code,400)

    def test_missing_raw_image_is_explicit_and_external_image_urls_are_not_embedded(self):
        value=fixture(True);value['detection'].update(size=[20,30],t2_rgb_b64='https://untrusted.invalid/a.svg')
        saved=service.cache_report(value);text=review_html(saved)
        self.assertIn('本记录未保存原图显示',text);self.assertNotIn('https://untrusted.invalid',text)
        self.assertIn('净变化',text);self.assertIn('个百分点',text)

    def test_independent_failure_metrics_are_bound_to_weight_and_not_this_case_accuracy(self):
        evidence={'weights_sha256':'a'*64,'metrics':{'recall':.1044845,'precision':.0085879,'IoU':.00799915}}
        text=review_html(self.updated,evidence)
        self.assertIn('另一个矿区的结果，不是本页案例的精度',text);self.assertIn('10.45%',text)
        evidence['weights_sha256']='z'*64
        self.assertNotIn('CL0002公开清单',review_html(self.updated,evidence))


if __name__=='__main__':unittest.main()
