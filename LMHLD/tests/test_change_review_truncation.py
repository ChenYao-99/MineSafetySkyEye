"""Removed components need a truncation flag even when the current mask is empty."""
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detector_service import _change_instances

class ChangeReviewTruncationTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((64,64),np.uint8)
        for x in (5,25,45):self.mask[10:20,x:x+10]=1
        self.empty=np.zeros_like(self.mask)

    def test_removed_only_records_are_explicitly_truncated(self):
        result=_change_instances(self.mask,self.empty,max_regions=1)
        self.assertEqual(result['instances'],[]);self.assertEqual(len(result['removed_instances']),1)
        self.assertEqual(result['summary']['n_removed'],3)
        self.assertTrue(result['summary']['truncated']);self.assertTrue(result['summary']['truncated_removed'])
        self.assertFalse(result['summary']['truncated_t2'])

    def test_current_only_records_keep_separate_epoch_counts(self):
        result=_change_instances(self.empty,self.mask,max_regions=1)
        self.assertEqual(result['removed_instances'],[]);self.assertEqual(len(result['instances']),1)
        self.assertEqual(result['summary']['n_t1_instances'],0);self.assertEqual(result['summary']['n_t2_instances'],3)
        self.assertTrue(result['summary']['truncated_t2']);self.assertFalse(result['summary']['truncated_removed'])

if __name__=='__main__':unittest.main(verbosity=2)
