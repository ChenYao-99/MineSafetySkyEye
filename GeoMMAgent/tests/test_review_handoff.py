"""Actual cached report handoff, including failed-analysis and paired scope behavior."""
import base64,copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
import backend as service
from archive_test_support import isolate_archive
from review_handoff import build_review_record

class ReviewHandoffTests(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service)
        service._report_cache.clear();self.client=service.app.test_client()
        self.fixture=json.loads((ROOT/'cache/competition_readiness_20261005/real_detection.json').read_text(encoding='utf-8'))
        self.result={'region':'upload','detection':copy.deepcopy(self.fixture),'knowledge':[],
                     'final_answer':'检测事实汇总','generation':{'status':'fallback','source':'detection_summary'},
                     'quality':{'degraded':True},'agents':[]}
    def test_actual_cached_handoff_does_not_repeat_detection_or_claim_field_review(self):
        cached=service.cache_report(self.result)
        with patch.object(service,'detector_request',side_effect=AssertionError('export must not infer')),patch.object(service,'analyze_with_agents',side_effect=AssertionError('export must not generate')):
            response=self.client.post('/api/review_record',json={'report_id':cached['report_id'],'meters_per_pixel':1,'roi':[0,0,1,1]})
        self.assertEqual(response.status_code,200);body=response.get_json();r=body['record']
        self.assertEqual(r['trace']['request_id'],self.fixture['request_id'])
        self.assertEqual(r['trace']['weights_sha256'],[self.fixture['model_info']['artifact']['weights_sha256']])
        self.assertEqual(r['selected_roi'],self.fixture['roi']);self.assertEqual(r['measurement_scale']['meters_per_pixel'],10)
        self.assertFalse(r['project_field_trial_or_expert_endorsement']);self.assertEqual(r['accepted_human_review_decisions'],[])
        self.assertEqual(json.loads(base64.b64decode(body['content_b64'])),r)
        self.assertNotIn('overlay_b64',json.dumps(r));self.assertTrue(r['trace']['trace_complete'])
    def test_weather_and_ground_location_unknown_not_hidden_by_readable_pixels(self):
        r=service.cache_report(self.result)['review_handoff'];codes={x['code'] for x in r['pending_checks']}
        self.assertIn('WEATHER_UNVERIFIED',codes);self.assertIn('GROUND_LOCATION_UNVERIFIED',codes)
        self.assertIn('AUTOMATIC_ANALYSIS_INCOMPLETE',codes)
        self.assertFalse(r['summary']['area_ratio_is_failure_probability'])
    def test_pair_record_preserves_two_dates_and_percentage_points(self):
        a=copy.deepcopy(self.fixture['band_meta']);b=copy.deepcopy(a);b['acquisition_datetime']='2019-02-01T11:13:29Z'
        self.result['region']='compare';d=self.result['detection'];d['band_meta']={'t1':a,'t2':b}
        d['stats']={'t1_ratio':.1,'t2_ratio':.2,'net_ratio':.1,'n_added_regions':2,'n_expanded_regions':3,'n_removed_regions':1}
        r=service.cache_report(self.result)['review_handoff']
        self.assertEqual(r['summary']['net_change_percentage_points'],10)
        self.assertEqual(r['input_evidence']['t1']['acquisition_datetime'],a['acquisition_datetime'])
        self.assertEqual(r['input_evidence']['t2']['acquisition_datetime'],b['acquisition_datetime'])
        self.assertIn('CHANGE_CAUSE_UNVERIFIED',{x['code'] for x in r['pending_checks']})
        self.assertIsNone(r['summary']['reported_regions'])
    def test_untraceable_expired_or_wrong_request_rejected_without_generation(self):
        self.assertEqual(self.client.post('/api/review_record',json={}).status_code,400)
        self.assertEqual(self.client.post('/api/review_record',json=[]).status_code,400)
        self.assertEqual(self.client.post('/api/review_record',json={'report_id':'missing'}).status_code,410)
        self.result['detection'].pop('model_info');cached=service.cache_report(self.result)
        self.assertEqual(self.client.post('/api/review_record',json={'report_id':cached['report_id']}).status_code,422)
    def test_conflict_and_truncated_regions_remain_pending(self):
        d=self.result['detection'];d['stats']['unverified_pixels']=999;d['stats']['area_ratio_available']=False
        d['stats']['num_regions_total']=1000;cached=service.cache_report(self.result);r=cached['review_handoff']
        self.assertIsNone(r['summary']['area_ratio']);self.assertEqual(r['summary']['unknown_conflict_pixels'],999)
        self.assertTrue(r['review_items_truncated']);self.assertIn('CONFLICT_SUPPORT_UNKNOWN',{x['code'] for x in r['pending_checks']})
        self.assertTrue(all(x['review_status']=='not_reviewed' for x in r['review_items']))

    def test_real_pair_geometry_preserves_epoch_ids_and_both_truncation_paths(self):
        bank=json.loads(Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005/detector_bank.json').read_text(encoding='utf-8'))
        d=copy.deepcopy(next(t['detection'] for t in bank['tasks'] if t['id']=='LasCruces_pair'))
        self.result['region']='compare';self.result['detection']=d
        record=service.cache_report(self.result)['review_handoff']
        self.assertTrue(record['review_items']);self.assertTrue(record['review_items_truncated'])
        self.assertTrue(any(r['source_epoch']=='t1' for r in record['review_items']))
        self.assertTrue(any(r['source_epoch']=='t2' for r in record['review_items']))
        keys=[r['review_item_key'] for r in record['review_items']];self.assertEqual(len(keys),len(set(keys)))
        lookup={(epoch,str(r['id'])):r for epoch,rows in (('t1',d['change_instances']['removed_instances']),('t2',d['change_instances']['instances'])) for r in rows}
        for row in record['review_items']:
            source=lookup[row['source_epoch'],str(row['id'])]
            self.assertEqual(row['bbox'],source['bbox']);self.assertEqual(row['area_px'],source['area_px'])
            self.assertEqual(row['coordinate_frame'],'t2_comparison_grid')
        self.assertEqual(record['summary']['n_added_regions'],d['stats']['n_added_regions'])
        self.assertEqual(record['summary']['matched_instance_summary'],d['change_instances']['summary'])

    def test_machine_facts_ignore_supplied_generated_values_and_remain_cached(self):
        self.result['objective_facts']={'largest_area_m2':1,'site_stability_confirmed':True}
        cached=service.cache_report(self.result)
        self.assertEqual(cached['objective_facts']['largest_area_m2'],148100)
        self.assertFalse(cached['objective_facts']['site_stability_confirmed'])
        self.result['detection']['stats']['regions'][0]['area_m2']=1
        body=self.client.post('/api/review_record',json={'report_id':cached['report_id']}).get_json()
        self.assertEqual(body['record']['objective_facts']['largest_area_m2'],148100)
        self.assertEqual(body['record']['objective_facts_source'],'same_detector_result')

    def test_quality_and_work_evidence_do_not_turn_into_confirmed_safety(self):
        self.result['detection']['band_meta']['observation_quality']={'source':'provided_Sentinel2_SCL','quality_labels_independently_verified':False}
        self.result['detection']['work_context']={'status':'matched_public_expected_work_reference',
            'predicted_pixels_overlapping_work_reference':10,'geotechnical_safety_or_accident_absence_verified':False}
        record=service.cache_report(self.result)['review_handoff'];codes={x['code'] for x in record['pending_checks']}
        self.assertIn('WEATHER_LABEL_REVIEW_REQUIRED',codes);self.assertIn('WORK_CONTEXT_CAUSE_UNVERIFIED',codes)
        self.assertFalse(record['objective_facts']['site_stability_confirmed'])
        self.assertEqual(record['work_context']['predicted_pixels_overlapping_work_reference'],10)

    def test_work_reference_is_supplied_to_report_reasoning_without_live_model_call(self):
        d=copy.deepcopy(self.fixture);d['overlay_b64']='cached_image'
        d['work_context']={'status':'matched_public_expected_work_reference','predicted_pixels_overlapping_work_reference':10,
            'source_title':'public annual mine clearing','source_period':'2023-2024','source_url':'https://example.org/public-source'}
        quality={'generation':{'status':'fallback','source':'detection_summary'}}
        with patch.object(service,'detector_request',return_value=d),patch.object(service,'analyze_with_agents',return_value=([], '本地事实汇总',None,quality)) as call:
            result=service.run_pipeline_upload('fixture')
        self.assertIn('[WORKCTX-001]',call.call_args.args[0])
        self.assertIn('不确认具体日级作业时间',call.call_args.args[0])
        self.assertEqual(result['knowledge'][-1]['id'],'WORKCTX-001')
        self.assertFalse(result['objective_facts']['site_stability_confirmed'])

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');unittest.main(verbosity=2)
