import unittest
from evidence_experiment import parse_answer, score, checked_report, run


class ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.task=dict(target={'pixels':38,'full_boundary':False},documents=[dict(id='SOURCE',content={})])
        self.report=dict(facts={'pixels':38,'full_boundary':False},conclusion='仅为部分核心。',evidence_ids=['SOURCE'],uncertainties=['外围未知'])

    def test_substantive_error_and_missing_field_are_distinct(self):
        self.report['facts']={'pixels':380}
        s=score(self.report,self.task)
        self.assertEqual(len(s['wrong_fields']),1)
        self.assertEqual(s['missing_fields'],['full_boundary'])
        self.assertFalse(s['objective_contract_passed'])

    def test_type_coercion_does_not_hide_format_errors(self):
        self.report['facts']={'pixels':'38','full_boundary':'false'}
        s=score(self.report,self.task)
        self.assertEqual(len(s['type_or_format_fields']),2)
        self.assertEqual(s['wrong_fields'],[])

    def test_invalid_or_missing_citations_fail_contract(self):
        self.report['evidence_ids']=['INVENTED']
        self.assertFalse(score(self.report,self.task)['objective_contract_passed'])
        self.report['evidence_ids']=[]
        self.assertFalse(score(self.report,self.task)['objective_contract_passed'])

    def test_guard_corrects_facts_but_does_not_certify_generated_prose(self):
        self.report['facts']['pixels']=380
        self.report['conclusion']='已确认完整边界。'
        checked=checked_report(self.report,self.task)
        self.assertEqual(checked['facts']['pixels'],38)
        self.assertEqual(checked['generated_facts_before_check']['pixels'],380)
        self.assertEqual(checked['conclusion'],'已确认完整边界。')
        self.assertFalse(checked['generated_conclusion_semantically_verified'])
        self.assertTrue(checked['manual_review_required'])

    def test_truncated_or_non_object_answer_is_not_a_report(self):
        self.assertIsNone(parse_answer('{"facts":'))
        self.assertIsNone(parse_answer('[]'))
        self.assertFalse(score(None,self.task)['objective_contract_passed'])

    def test_external_call_is_blocked_before_any_client_without_explicit_approval(self):
        with self.assertRaisesRegex(RuntimeError,'Explicit user approval'):
            run(None)


if __name__=='__main__':
    unittest.main(verbosity=2)
