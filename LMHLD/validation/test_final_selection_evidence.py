"""Test admission evidence without creating the real one-shot test lock."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from validation.mine_data_protocol import validate_final_selection,sha,DATA

class SelectionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.real_lock=DATA/'final_evaluation_once.json'
        self.real_lock_before=self.real_lock.read_bytes() if self.real_lock.exists() else None
        base=ROOT/'experiments/final_gate_unit_fixtures';base.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name);self.protocol=self.folder/'input.json';self.protocol.write_text('{}',encoding='utf-8')
        self.selection=self.folder/'selection.json'
        self.record={'status':'accepted_for_final_evaluation','development_gates_passed':True,'source_regression_passed':True,
                     'reserved_predictions_used_for_selection':0,'threshold':.5,'selected_weights_sha256':['a'*64],
                     'input_protocol':{'path':str(self.protocol),'sha256':sha(self.protocol)}}
        self.plan={'threshold':.5,'metrics':['IoU_on_scored_published_reference'],
                   'weights':[{'sha256':'a'*64}], 'development_selection_record':{}}
        self.freeze()
    def freeze(self):
        self.selection.write_text(json.dumps(self.record),encoding='utf-8')
        self.plan['development_selection_record']={'path':str(self.selection),'sha256':sha(self.selection)}
    def test_valid_evidence_validation_does_not_open_test(self):
        self.assertEqual(validate_final_selection(self.plan),self.record)
        self.assertEqual(self.real_lock.read_bytes() if self.real_lock.exists() else None,self.real_lock_before)
    def test_empty_metrics_and_boolean_threshold_rejected(self):
        self.plan['metrics']=[]
        with self.assertRaisesRegex(ValueError,'nonempty'):validate_final_selection(self.plan)
        self.plan['metrics']=['x'];self.plan['threshold']=True
        with self.assertRaisesRegex(ValueError,'finite'):validate_final_selection(self.plan)
    def test_unverified_statement_or_modified_evidence_rejected(self):
        self.plan['development_selection_record']='we tested it'
        with self.assertRaisesRegex(ValueError,'SHA256'):validate_final_selection(self.plan)
        self.freeze();self.selection.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'changed'):validate_final_selection(self.plan)
    def test_failed_selection_reserved_use_or_different_weights_rejected(self):
        for key,bad in [('development_gates_passed',False),('source_regression_passed',False),
                        ('reserved_predictions_used_for_selection',1),('selected_weights_sha256',['b'*64]),('threshold',.4)]:
            old=self.record[key];self.record[key]=bad;self.freeze()
            with self.assertRaises(ValueError):validate_final_selection(self.plan)
            self.record[key]=old
    def test_preprocessing_tamper_rejected(self):
        self.protocol.write_text('{"different":true}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'input protocol changed'):validate_final_selection(self.plan)

    def test_explicit_frozen_audit_preserves_failed_development_without_promoting_model(self):
        self.plan['evaluation_purpose']='frozen_version_independent_audit'
        self.record.update(status='accepted_for_frozen_version_audit',evaluation_purpose=self.plan['evaluation_purpose'],
            development_gates_passed=False,reserved_results_may_select_or_tune_model=False,
            user_authorization={'source':'user_message','instruction':'Freeze current version and conduct one independent acceptance.'})
        self.freeze()
        self.assertFalse(validate_final_selection(self.plan)['development_gates_passed'])
        for key,bad in [('source_regression_passed',False),('reserved_results_may_select_or_tune_model',True),('user_authorization',{})]:
            old=self.record[key];self.record[key]=bad;self.freeze()
            with self.assertRaises(ValueError):validate_final_selection(self.plan)
            self.record[key]=old

if __name__=='__main__':unittest.main(verbosity=2)
