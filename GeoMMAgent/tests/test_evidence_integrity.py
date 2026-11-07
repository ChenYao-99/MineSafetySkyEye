"""Regression of actual numeric/provenance failures and response completeness."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_integrity import (bind_structured_report, completion_status, evidence_context,
                                resolve_scoped_evidence)
from report_grounding import check_detection_numbers


class EvidenceContracts(unittest.TestCase):
    def test_actual_old_numeric_prose_is_withheld_after_fields_are_corrected(self):
        study = json.loads((ROOT/'maintenance/phase2_20261013/results.json').read_text(encoding='utf-8'))
        protocol = json.loads((ROOT/'maintenance/phase2_20261013/protocol.json').read_text(encoding='utf-8'))
        targets = {t['id']:t['target'] for t in protocol['tasks']}
        for task, wrong_text in [('P1-02','8679'),('P1-06','48.67')]:
            raw = next(r['report'] for r in study['rows'] if r['task_id']==task and r['arm']=='B_single_checked')
            fixed = bind_structured_report(raw, targets[task])
            self.assertNotIn(wrong_text, fixed['conclusion'])
            self.assertEqual(fixed['facts'], targets[task])
            self.assertIn(wrong_text, fixed['raw_generated_report']['conclusion'])
            self.assertTrue(fixed['review_required'])

    def test_case_and_event_replacement_is_explicit_and_preserves_history(self):
        old = dict(evidence_id='old',scope='case',case_id='GAMS',event_id='2020',content_sha256='abc')
        new = dict(evidence_id='new',scope='case',case_id='GAMS',event_id='2020',supersedes=['old'],accepts_sha256='abc')
        result = resolve_scoped_evidence([old,new])
        self.assertEqual([x['evidence_id'] for x in result['active']],['new'])
        self.assertEqual([x['evidence_id'] for x in result['history']],['old'])
        self.assertNotIn('supersedes',old)
        wrong = copy.deepcopy(new); wrong['case_id']='AND'
        with self.assertRaisesRegex(ValueError,'Cross-case'):
            resolve_scoped_evidence([old,wrong])
        wrong=copy.deepcopy(new);wrong['event_id']='2023'
        with self.assertRaisesRegex(ValueError,'Cross-case'):
            resolve_scoped_evidence([old,wrong])

    def test_hash_mismatch_and_cycles_do_not_upgrade_references(self):
        a=dict(evidence_id='a',scope='case',case_id='one',event_id=None,content_sha256='x')
        b=dict(evidence_id='b',scope='case',case_id='one',event_id=None,supersedes=['a'],accepts_sha256='wrong')
        with self.assertRaisesRegex(ValueError,'hash'):
            resolve_scoped_evidence([a,b])
        b.pop('accepts_sha256');a['supersedes']=['b']
        with self.assertRaisesRegex(ValueError,'cycle'):
            resolve_scoped_evidence([a,b])

    def test_current_case_does_not_take_foreign_observations(self):
        record=dict(evidence_id='other',scope='case',case_id='AND')
        with self.assertRaisesRegex(ValueError,'another case'):
            evidence_context(case_id='GAMS',records=[record])
        record=dict(evidence_id='standard',scope='global_reference')
        self.assertEqual(len(evidence_context(case_id='GAMS',records=[record])['evidence']['active']),1)

    def test_flat_report_cannot_merge_two_case_acceptance_records(self):
        records=[dict(evidence_id='G',scope='case',case_id='GAMS'),dict(evidence_id='A',scope='case',case_id='AND')]
        raw={'facts':{'known':True},'conclusion':'AND替代GAMS。','evidence_ids':['G','A']}
        fixed=bind_structured_report(raw,{'known':True},evidence_records=records)
        self.assertTrue(fixed['review_required'])
        self.assertNotIn('替代',fixed['conclusion'])
        self.assertIn('mixed_case_citations_require_separate_reports',fixed['source_scope_errors'])

    def test_superseded_record_is_history_not_an_active_citation(self):
        records=[dict(evidence_id='old',scope='case',case_id='one'),dict(evidence_id='new',scope='case',case_id='one',supersedes=['old'])]
        fixed=bind_structured_report({'facts':{'known':True},'conclusion':'old text','evidence_ids':['old']},
                                      {'known':True},evidence_records=records,case_id='one')
        self.assertEqual(fixed['source_scope_errors'],['unresolved_or_superseded_citation'])

    def test_camel_and_provider_truncation_are_both_detected(self):
        self.assertFalse(completion_status(SimpleNamespace(info={'termination_reasons':['length']}))['complete'])
        self.assertFalse(completion_status(SimpleNamespace(choices=[SimpleNamespace(finish_reason='length')]))['complete'])
        self.assertFalse(completion_status(SimpleNamespace(info={'termination_reasons':['max_num_tokens_exceeded']}))['complete'])
        self.assertTrue(completion_status(SimpleNamespace(info={'termination_reasons':['stop']}))['complete'])

    def test_paired_values_counts_and_percentage_points_are_checked(self):
        facts={'kind':'compare','stats':{'t1_ratio':.0884,'t2_ratio':.0889,'net_ratio':.0005,
                'n_added_regions':2,'n_expanded_regions':3,'n_removed_regions':1}}
        self.assertEqual(check_detection_numbers('T1占比8.84%，T2占比8.89%，净变化+0.05个百分点，新增2处，扩大3处，消退1处。',facts),[])
        self.assertTrue(check_detection_numbers('T1占比18.84%，净变化0.05%，新增9处。',facts))

    def test_rounding_is_allowed_but_nearby_wrong_coverage_is_not(self):
        facts={'stats':{'area_ratio':.0871,'num_regions':1}}
        self.assertEqual(check_detection_numbers('面积占比8.7%，共1处。',facts),[])
        self.assertTrue(check_detection_numbers('面积占比8.9%，共1处。',facts))


class CoordinatorCompleteness(unittest.TestCase):
    def coordinator(self, workers):
        from coordinator import AgentCoordinator
        with patch('coordinator.coordinator.ChatAgent'):
            coord=AgentCoordinator(model=object())
        coord.register_workers(workers)
        return coord

    def test_truncated_intermediate_is_retained_but_not_consumed(self):
        called=[]
        class Agent:
            def step(self, msg):
                called.append(msg)
                return SimpleNamespace(msgs=[SimpleNamespace(content='partial evidence')],info={'termination_reasons':['length']})
        first=Agent(); second=SimpleNamespace(step=lambda msg: self.fail('downstream must not run'))
        coord=self.coordinator([dict(name='EvidenceAgent',description='evidence',agent=first),dict(name='ReasoningAgent',description='reason',agent=second)])
        final,steps,_=coord._execute_traced([dict(agent=0,subtask='first'),dict(agent=1,subtask='second')],'task',None)
        self.assertEqual(final,'')
        self.assertEqual(len(steps),1)
        self.assertEqual(steps[0].status,'incomplete')
        self.assertEqual(steps[0].output,'partial evidence')

    def test_open_reasoning_receives_prior_outputs_and_scope(self):
        class Agent:
            def __init__(self, answer): self.answer=answer
            def step(self,msg): return SimpleNamespace(msgs=[SimpleNamespace(content=self.answer)],info={'termination_reasons':['stop']})
        coord=self.coordinator([dict(name='ClsAgent',description='scene',agent=Agent('VERIFIED_PRIOR')),dict(name='ReasoningAgent',description='reason',agent=Agent('final report'))])
        coord.set_evidence_context(evidence_context(case_id='mine_one',request_id='request_one'))
        final,steps,_=coord._execute_traced([dict(agent=0,subtask='scene'),dict(agent=1,subtask='reason')],'OPEN_TASK',None)
        self.assertEqual(final,'final report')
        self.assertIn('VERIFIED_PRIOR',steps[1].input_text)
        self.assertIn('mine_one',steps[1].input_text)
        self.assertIn('reason',steps[1].input_text)
        self.assertIsNone(steps[1].trace_only_context)

    def test_mcq_reference_path_remains_explicit(self):
        agent=SimpleNamespace(step=lambda msg: SimpleNamespace(msgs=[SimpleNamespace(content='A')],info={'termination_reasons':['stop']}))
        coord=self.coordinator([dict(name='ReasoningAgent',description='reason',agent=agent)])
        final,steps,_=coord._execute_traced([dict(agent=0,subtask='reason')],'MCQ_TASK',None,answer_text='A')
        self.assertEqual(final,'A')
        self.assertIsNotNone(steps[0].trace_only_context)


if __name__=='__main__':
    unittest.main(verbosity=2)
