import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mine_data_protocol import registry,assert_mine_group_role,assert_development_cases


class ExplicitMineGroups(unittest.TestCase):
    def test_gamsberg_pits_and_dates_remain_one_development_group(self):
        v=registry()
        self.assertEqual(assert_mine_group_role('ZA_GAMS2020S','development',v),'ZA_GAMSBERG')
        self.assertEqual(assert_mine_group_role('ZA_GAMS2023W','development',v),'ZA_GAMSBERG')
    def test_future_same_mine_event_cannot_be_declared_unseen_final(self):
        v=copy.deepcopy(registry());v['cases']['FAKE_FUTURE_EVENT']={'mine_id':'ZA_GAMSBERG','role':'reserved_final_test','quality_review_status':'accepted'}
        with self.assertRaisesRegex(ValueError,'mine group conflict'):assert_mine_group_role('FAKE_FUTURE_EVENT','reserved_final_test',v)
        with self.assertRaisesRegex(ValueError,'mine group conflict'):assert_development_cases(['ZA_GAMS2020S'],v)
    def test_duplicate_reserved_group_cannot_be_imported_as_auxiliary(self):
        v=copy.deepcopy(registry());source=v['cases']['CL0002'];group=source.get('mine_group') or source.get('mine_id') or 'CL0002'
        v['cases']['FAKE_AUX_IMPORT']={'mine_id':group,'role':'auxiliary_development'}
        with self.assertRaisesRegex(ValueError,'mine group conflict'):assert_development_cases(['FAKE_AUX_IMPORT'],v)


if __name__=='__main__':unittest.main(verbosity=2)
