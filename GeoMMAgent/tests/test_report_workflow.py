"""Evidence roles, dependency checks and truthful workflow status."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidence_integrity import evidence_context
from report_workflow import ReportCoordinator,choose_report_route,workflow_status


class ReportWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.facts={'stats':{'scope':'whole_image','area_ratio':.1,'num_regions':2},
                    'band_meta':{'acquisition_datetime':'2021-02-18T10:30:00Z','spectral_indices_available':False}}
        self.context=evidence_context(case_id='test_case',request_id='test_request',
            records=[{'evidence_id':'TEST-001','scope':'global_reference'}])
        self.audit={'case_id':'test_case','request_id':'test_request','decision':'needs_review',
                    'checks':['保留本次量测'],'unknowns':['现场状态未确认'],'issues':[],'source_ids':['TEST-001']}

    def make_coord(self,audit=None,review='FINAL_REVIEW',truncated=False):
        captured=[]
        value=copy.deepcopy(self.audit if audit is None else audit)
        class Agent:
            def __init__(self,**kw):self.system=kw.get('system_message','')
            def step(self,msg):
                name='EvidenceAuditAgent' if '你是EvidenceAuditAgent' in self.system else 'ReportReviewAgent' if '你是ReportReviewAgent' in self.system else 'ReasoningAgent'
                captured.append((name,msg.content))
                content=json.dumps(value,ensure_ascii=False) if name=='EvidenceAuditAgent' else review if name=='ReportReviewAgent' else 'WRITER_DRAFT'
                return SimpleNamespace(msgs=[SimpleNamespace(content=content)],info={'termination_reasons':['length' if truncated and name=='EvidenceAuditAgent' else 'stop']})
        with patch('report_workflow.create_model_from_config',return_value=object()), \
             patch('report_workflow.ChatAgent',Agent),patch('coordinator.coordinator.ChatAgent'):
            coord=ReportCoordinator({},self.facts)
        coord.set_evidence_context(self.context)
        return coord,captured

    def test_complete_single_epoch_uses_two_distinct_roles(self):
        self.assertEqual(choose_report_route(self.facts)['roles'],['EvidenceAuditAgent','ReasoningAgent'])
        coord,captured=self.make_coord()
        trace=coord.run_with_trace('write report')
        self.assertEqual([name for name,_ in captured],['EvidenceAuditAgent','ReasoningAgent'])
        self.assertIn('现场状态未确认',captured[1][1])
        self.assertIn('同次工具字段',captured[1][1])
        self.assertEqual(trace.final_answer,'WRITER_DRAFT')
        self.assertFalse(coord.status(trace)['interrupted'])

    def test_pair_has_review_and_receives_the_actual_writer_draft(self):
        self.facts.update(kind='compare',band_meta={'t1':self.facts['band_meta'],'t2':self.facts['band_meta']})
        coord,captured=self.make_coord();trace=coord.run_with_trace('paired report')
        self.assertEqual([name for name,_ in captured],['EvidenceAuditAgent','ReasoningAgent','ReportReviewAgent'])
        self.assertIn('WRITER_DRAFT',captured[2][1])
        self.assertEqual(trace.final_answer,'FINAL_REVIEW')

    def test_unknown_time_and_conflicting_support_add_review(self):
        self.facts['band_meta']['acquisition_datetime']=None
        self.assertIn('ReportReviewAgent',choose_report_route(self.facts)['roles'])
        self.facts['band_meta']['acquisition_datetime']='2021-02-18'
        self.facts['stats']['unverified_pixels']=350
        self.assertIn('ReportReviewAgent',choose_report_route(self.facts)['roles'])

    def test_wrong_case_audit_stops_before_writer_and_marks_planned_steps(self):
        self.audit['case_id']='foreign_case'
        coord,captured=self.make_coord();trace=coord.run_with_trace('report')
        self.assertEqual([name for name,_ in captured],['EvidenceAuditAgent'])
        self.assertEqual(trace.final_answer,'')
        status=coord.status(trace)
        self.assertTrue(status['interrupted'])
        self.assertEqual(status['roles'][1]['name'],'ReasoningAgent')
        self.assertEqual(status['roles'][1]['status'],'not_executed')

    def test_unprovided_source_and_blocked_decision_do_not_flow_downstream(self):
        for field,value in [('source_ids',['FAKE-999']),('decision','block')]:
            audit=copy.deepcopy(self.audit);audit[field]=value
            coord,captured=self.make_coord(audit);trace=coord.run_with_trace('report')
            self.assertEqual(len(captured),1)
            self.assertEqual(trace.agent_steps[0].status,'invalid')
            self.assertTrue(trace.agent_steps[0].validation_errors)

    def test_truncated_audit_is_not_consumed(self):
        coord,captured=self.make_coord(truncated=True);trace=coord.run_with_trace('report')
        self.assertEqual(len(captured),1)
        self.assertEqual(trace.agent_steps[0].status,'incomplete')

    def test_reviewer_blocks_irreconcilable_case(self):
        self.facts['band_meta']['acquisition_datetime']=None
        coord,captured=self.make_coord(review='[REVIEW_BLOCKED]关键证据冲突')
        trace=coord.run_with_trace('report')
        self.assertEqual(len(captured),3)
        self.assertEqual(trace.final_answer,'')
        self.assertTrue(coord.status(trace)['interrupted'])

    def test_one_planned_QA_step_is_not_an_interrupted_multi_role_chain(self):
        trace=SimpleNamespace(dispatch_plan=[{'agent':0,'subtask':'describe'}],
            agent_steps=[SimpleNamespace(step=1,agent_name='ReasoningAgent',status='complete')])
        status=workflow_status(trace)
        self.assertTrue(status['single_worker_planned'])
        self.assertFalse(status['interrupted'])
        self.assertFalse(status['multiple_workers_executed'])


if __name__=='__main__':unittest.main(verbosity=2)
