from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import brumadinho_auxiliary_reference as a


class AuxiliaryReferenceTests(unittest.TestCase):
    def test_auxiliary_data_never_opens_for_primary_or_negative_tasks(self):
        for task in ('pitwall_binary_training', 'normal_work_background', 'full_incident_iou', 'independent_test'):
            with self.assertRaises(ValueError): a.load_reference(Path('nonexistent'), task)


if __name__ == '__main__': unittest.main()
