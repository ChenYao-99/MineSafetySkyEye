"""Later reserves must participate in protection audits and remain invisible to development."""
import copy, unittest
from mine_data_protocol import RESERVED,registry,reserved_case_ids,assert_development_cases

class ReservedInventoryTests(unittest.TestCase):
    def test_four_current_reserves_are_included_and_unobserved(self):
        value=registry()
        self.assertEqual(reserved_case_ids(value),{'CL0002','KG0001','US_BC2021','TR_COP2024'})
        self.assertEqual(RESERVED,reserved_case_ids(value))
        for case_id in reserved_case_ids(value):
            self.assertEqual(value['cases'][case_id]['model_predictions_run'],0)
            with self.assertRaises(ValueError):assert_development_cases([case_id],value)

    def test_newly_registered_reserve_is_discovered_without_reimport(self):
        value=copy.deepcopy(registry())
        value['cases']['FUTURE_RESERVE_TEST_ONLY']={'role':'reserved_final_test','model_predictions_run':0,'mine_group':'TEST_ONLY'}
        self.assertIn('FUTURE_RESERVE_TEST_ONLY',reserved_case_ids(value))
        with self.assertRaises(ValueError):assert_development_cases(['FUTURE_RESERVE_TEST_ONLY'],value)

if __name__=='__main__':unittest.main(verbosity=2)
