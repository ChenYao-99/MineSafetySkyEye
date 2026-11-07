import json,sys,unittest
from pathlib import Path
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
from published_inventory_contract import load_inventory_reference,reference_metrics
from validation.mine_data_protocol import registry,DATA,register_final_run
class PublishedInventoryContractTests(unittest.TestCase):
    def test_real_accepted_independent_sources_and_scopes(self):
        for case,scope,mode,count in [('CL0002','post_observed_reference','post_only',2163),('CL0002','paired_observed_reference','paired_observed_only',521),('KG0001','paired_observed_reference','paired_observed_only',1751)]:
            d=load_inventory_reference(case,scope,mode);self.assertEqual(int((d['labels']==1).sum()),count)
            self.assertFalse(d['contract']['source_original_field_survey']);self.assertFalse(d['contract']['outside_mapped_event_is_geotechnically_safe'])
    def test_paired_only_cannot_claim_postonly_label_support(self):
        with self.assertRaises(ValueError):load_inventory_reference('CL0002','post_observed_reference','paired_observed_only')
    def test_training_and_unopened_independent_evaluation_rejected(self):
        for use in ('development_training','evaluation'):
            with self.assertRaises(ValueError):load_inventory_reference('CL0002','post_observed_reference','post_only',use)
        self.assertFalse((DATA/'final_evaluation_once.json').exists())
    def test_unknown_output_cannot_change_reference_metrics(self):
        labels=np.array([[1,0,255],[1,0,255]],np.uint8);p=np.array([[.9,.2,0],[.8,.7,0]],np.float32);q=p.copy();q[:,2]=1
        self.assertEqual(reference_metrics(p,labels,.5),reference_metrics(q,labels,.5))
        self.assertAlmostEqual(reference_metrics(p,labels,.5)['IoU_on_scored_published_reference'],2/3)
    def test_positive_core_only_cannot_be_full_IoU(self):
        with self.assertRaises(ValueError):reference_metrics(np.ones((2,2)),np.array([[1,255],[1,255]]),.5)
    def test_final_gate_requires_frozen_input_and_mask_scope(self):
        path=DATA/'quality_audit/test_rejected_scope_plan.json';path.write_text(json.dumps({'case_ids':['CL0002']}),encoding='utf-8')
        self.addCleanup(lambda:path.unlink(missing_ok=True))
        with self.assertRaisesRegex(ValueError,'input mode'):register_final_run(path,ROOT/'experiments/test_not_started')
        self.assertFalse((DATA/'final_evaluation_once.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
