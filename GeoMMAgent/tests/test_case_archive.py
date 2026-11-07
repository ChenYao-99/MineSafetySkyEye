"""Persistence, provenance, concurrent review conflicts and real archive routes."""
from pathlib import Path
import base64,copy,hashlib,io,json,os,sys,sqlite3,unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['YOLO_AUTOINSTALL']='false'
import backend as service
from case_archive import CaseArchive,ArchiveError
from archive_test_support import isolate_archive


def fixture(paired=False):
    region={'id':1,'area_px':20,'bbox':[0,0,5,4]}
    stats={'area_ratio':.1,'num_regions':1,'num_regions_total':1,'max_region_area':20,'regions':[region]}
    metadata={'input_sha256':'b'*64,'acquisition_datetime':'2020-01-01T10:00:00Z'}
    detection={'request_id':'request-1','model_info':{'weights_sha256':'a'*64},'model_tag':'mine','threshold':.4,
               'stats':stats,'band_meta':metadata,'overlay_b64':'SAVED_OVERLAY','input_quality':{'status':'usable_observation'}}
    if paired:
        detection['stats']={'t1_ratio':.1,'t2_ratio':.2,'net_ratio':.1,'valid_pixels':200,
                            'n_added_regions':1,'n_expanded_regions':0,'n_removed_regions':1}
        detection['band_meta']={'t1':metadata,'t2':{**metadata,'acquisition_datetime':'2020-02-01T10:00:00Z'}}
        detection['change_instances']={'instances':[{**region,'change':'added'}],
                                      'removed_instances':[{**region,'change':'removed'}],
                                      'summary':{'n_added':1,'n_removed':1}}
    return {'region':'compare' if paired else 'upload','target_index':None,'detection':detection,'agents':[],
            'knowledge':[],'final_answer':'原始自动事实汇总','generation':{'status':'fallback','source':'detection_summary'},
            'quality':{'degraded':True}}


class CaseArchiveTests(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service);service._report_cache.clear();self.client=service.app.test_client()
        self.case=service.case_archive.create_case('测试矿区','工程验收案例','内部测试，不是现场试用')
        self.saved=service.cache_report(fixture(),self.case['case_id'])

    def decision(self,revision=0,target='report',status='needs_followup'):
        return {'expected_revision':revision,'target_key':target,'status':status,'reviewer':'测试复核员',
                'evidence':'对照本次原图和公开资料；身份未作独立核验','note':'需要继续核查变化原因'}

    def test_reopen_after_cache_eviction_and_time_expiry_never_reruns(self):
        for _ in range(9):service.cache_report(fixture(),self.case['case_id'])
        self.assertNotIn(self.saved['report_id'],service._report_cache)
        service._report_cache.clear()
        reopened=CaseArchive(service.case_archive.path)
        with patch.object(service,'case_archive',reopened),patch.object(service,'detector_request',side_effect=AssertionError('no inference')), \
             patch.object(service,'analyze_with_agents',side_effect=AssertionError('no cloud')):
            loaded=service.report_for_export({'report_id':self.saved['report_id']},lambda:None)
        self.assertEqual(loaded['detection'],self.saved['detection'])
        self.assertEqual(loaded['final_answer'],self.saved['final_answer'])
        self.assertEqual(loaded['review_handoff']['input_evidence']['image']['input_sha256'],'b'*64)
        self.assertEqual(service.case_archive.list_reports(self.case['case_id'])['total'],10)

    def test_review_changes_only_human_metadata_preserves_machine_facts_and_hash(self):
        result=service.case_archive.submit_review(self.saved['report_id'],self.decision(target='image:1'))
        self.assertEqual(result['detection'],self.saved['detection'])
        self.assertEqual(result['objective_facts'],self.saved['objective_facts'])
        self.assertEqual(result['archive']['snapshot_sha256'],self.saved['archive']['snapshot_sha256'])
        record=result['review_handoff'];self.assertEqual(record['review_items'][0]['review_status'],'human_note_recorded')
        self.assertFalse(record['project_field_trial_or_expert_endorsement'])
        self.assertFalse(record['objective_facts']['site_stability_confirmed'])
        self.assertFalse(record['accepted_human_review_decisions'][0]['reviewer_identity_verified'])
        self.assertFalse(record['accepted_human_review_decisions'][0]['interpretation_is_verified_ground_truth'])

    def test_correction_and_withdrawal_keep_all_previous_versions(self):
        store=service.case_archive;rid=self.saved['report_id']
        first=store.submit_review(rid,self.decision())
        second=store.submit_review(rid,self.decision(1,status='likely_work_change'))
        withdrawn=store.submit_review(rid,self.decision(2,status='withdrawn'))
        history=withdrawn['review_handoff']['review_history']
        self.assertEqual(len(history),3);self.assertEqual(history[1]['supersedes_event_id'],history[0]['event_id'])
        self.assertEqual(withdrawn['review_handoff']['accepted_human_review_decisions'],[])
        self.assertEqual(withdrawn['review_handoff']['review_state'],'not_reviewed')

    def test_two_concurrent_writers_cannot_silently_overwrite(self):
        def submit():
            try:return service.case_archive.submit_review(self.saved['report_id'],self.decision())['archive']['review_revision']
            except ArchiveError as error:return error.status
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:submit(),range(2)))
        self.assertEqual(sorted(results),[1,409])
        self.assertEqual(len(service.case_archive.get_report(self.saved['report_id'])['review_handoff']['review_history']),1)

    def test_paired_same_numeric_ids_remain_distinct_review_targets(self):
        pair=service.cache_report(fixture(True),self.case['case_id']);store=service.case_archive
        store.submit_review(pair['report_id'],self.decision(target='t1:1'))
        result=store.submit_review(pair['report_id'],self.decision(1,target='t2:1'))
        self.assertEqual({row['target_key'] for row in result['review_handoff']['accepted_human_review_decisions']},{'t1:1','t2:1'})

    def test_invalid_targets_and_attempted_fact_changes_are_rejected(self):
        for change in ({'target_key':'image:999'},{'expected_revision':True},{'status':'confirmed_safe'},
                       {'objective_facts':{'site_stability_confirmed':True}},{'reviewer':''},{'evidence':''}):
            with self.subTest(change=change):
                with self.assertRaises(ArchiveError):service.case_archive.submit_review(self.saved['report_id'],{**self.decision(),**change})
        self.assertEqual(service.case_archive.get_report(self.saved['report_id'])['archive']['review_revision'],0)

    def test_api_submission_download_and_conflict_share_latest_record(self):
        url='/api/archive/reports/'+self.saved['report_id']+'/reviews'
        response=self.client.post(url,json=self.decision());self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.post(url,json=self.decision()).status_code,409)
        exported=self.client.post('/api/review_record',json={'report_id':self.saved['report_id']}).get_json()
        record=json.loads(base64.b64decode(exported['content_b64']))
        self.assertEqual(record['review_revision'],1);self.assertEqual(len(record['review_history']),1)
        self.assertEqual(self.client.get('/api/archive/reports/'+self.saved['report_id']).get_json()['review_handoff'],record)

    def test_backup_is_reopenable_with_history_and_model_identifiers(self):
        service.case_archive.submit_review(self.saved['report_id'],self.decision())
        backup=service.case_archive.backup();copied=CaseArchive(backup).get_report(self.saved['report_id'])
        self.assertEqual(copied['review_handoff']['review_revision'],1)
        self.assertEqual(copied['review_handoff']['trace']['weights_sha256'],['a'*64])

    def test_corrupt_snapshot_is_not_returned_even_if_memory_cache_exists(self):
        with service.case_archive.connection() as db:db.execute('UPDATE reports SET payload_sha256=? WHERE report_id=?',('0'*64,self.saved['report_id']))
        response=self.client.get('/api/archive/reports/'+self.saved['report_id'])
        self.assertEqual(response.status_code,503)

    def test_case_edit_requires_current_revision_and_search_is_literal(self):
        current=service.case_archive.get_case(self.case['case_id'])
        body={'expected_revision':current['revision'],'mine_name':'石灰岩矿区','title':'边坡复核','notes':'资料待核查'}
        response=self.client.patch('/api/cases/'+self.case['case_id'],json=body);self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.patch('/api/cases/'+self.case['case_id'],json=body).status_code,409)
        self.assertEqual(self.client.get('/api/cases?q=石灰岩').get_json()['total'],1)
        self.assertEqual(self.client.get('/api/cases?q=%25').get_json()['total'],0)
        history=service.case_archive.get_case(self.case['case_id'])['metadata_history']
        self.assertEqual(len(history),2);self.assertEqual(history[0]['metadata']['mine_name'],'测试矿区')
        self.assertEqual(history[1]['metadata']['mine_name'],'石灰岩矿区')

    def test_unknown_schema_and_pagination_do_not_silently_rewrite_archive(self):
        self.assertEqual(self.client.get('/api/cases?limit=0').status_code,400)
        self.assertEqual(self.client.get('/api/cases?offset=-1').status_code,400)
        with service.case_archive.connection() as db:db.execute('PRAGMA user_version=999')
        self.assertEqual(self.client.get('/api/cases').status_code,503)
        db=sqlite3.connect(str(service.case_archive.path))
        try:self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],999)
        finally:db.close()

    def test_unknown_case_is_rejected_before_pipeline_execution(self):
        with patch.object(service,'detector_request',side_effect=AssertionError('must not infer')):
            response=self.client.post('/api/pipeline_upload',json={'image':'input','case_id':'unknown'})
        self.assertEqual(response.status_code,404)

    def test_private_fields_are_not_persisted_and_reason_for_missing_index_is_kept(self):
        result=fixture();result.update(api_key='SECRET',image='RAW_INPUT',image_path='private path')
        result['detection']['quant']={'ndvi':{'available':False,'reason':'缺少真实近红外'}}
        result['generation']['reason']='PROVIDER_DIAGNOSTIC';saved=service.cache_report(result)
        encoded=json.dumps(saved,ensure_ascii=False)
        self.assertNotIn('SECRET',encoded);self.assertNotIn('RAW_INPUT',encoded);self.assertNotIn('PROVIDER_DIAGNOSTIC',encoded)
        self.assertEqual(saved['detection']['quant']['ndvi']['reason'],'缺少真实近红外')


if __name__=='__main__':unittest.main(verbosity=2)
