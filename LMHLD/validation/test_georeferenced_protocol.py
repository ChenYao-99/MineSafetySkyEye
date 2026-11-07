# -*- coding: utf-8 -*-
"""Mine split and dataset-integrity checks; never run detector inference."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"validation"))
import mine_data_protocol as protocol


class GeoreferencedProtocol(unittest.TestCase):
    def test_new_mines_cannot_be_training_or_threshold_cases(self):
        for eid in protocol.RESERVED:
            with self.assertRaisesRegex(ValueError,"Reserved"):
                protocol.assert_development_cases([eid])

    def test_tailings_auxiliary_is_separate_from_pit_wall_test(self):
        cases=protocol.assert_development_cases(["BR0001"])
        self.assertEqual(cases[0]["scope"],"tailings_dam_mudflow_auxiliary")
        self.assertEqual(cases[0]["role"],"auxiliary_development")

    def test_directory_glob_cannot_replace_explicit_case_ids(self):
        with self.assertRaises(ValueError):
            protocol.assert_development_cases([])
        with self.assertRaises(ValueError):
            protocol.assert_development_cases(["*"])

    def test_original_assets_are_frozen_on_D(self):
        value=protocol.registry()
        self.assertEqual(len(value["frozen_assets"]),38)
        self.assertTrue(protocol.verify_assets(value))
        self.assertEqual(value["new_test_model_predictions_run"],0)
        for name in value["frozen_assets"]:
            self.assertTrue((protocol.DATA/name).resolve().is_relative_to(protocol.DATA.resolve()))

    def test_published_mask_and_actual_CRS_are_audited(self):
        quality=json.loads((protocol.DATA/"quality_audit/quality_audit.json").read_text(encoding="utf-8"))
        for case in quality["cases"]:
            self.assertEqual(case["published_mask_vs_vector_IoU"],1.)
            self.assertTrue(case["geometric_gate_pass"])
            self.assertEqual(case["raster_metadata"]["mask"]["resolution"],[10.,10.])
            self.assertNotEqual(case["raster_metadata"]["mask"]["crs"],"EPSG:32633")
            self.assertEqual(case["model_predictions_run"],0)

    def test_final_gate_rejects_pending_quality_before_lock(self):
        plan=protocol.DATA/"quality_audit/test_rejected_final_plan.json"
        plan.write_text(json.dumps({"case_ids":["US_BC2021"]}),encoding="utf-8")
        self.addCleanup(lambda:plan.unlink(missing_ok=True))
        with self.assertRaisesRegex(ValueError,"not accepted"):
            protocol.register_final_run(plan,ROOT/"experiments/test_not_started")
        self.assertFalse((protocol.DATA/"final_evaluation_once.json").exists())

    def test_old_xinjing_event_cannot_be_relabelled_blind_by_new_files(self):
        value=protocol.registry();case=value['cases']['CN_XJ2023']
        self.assertEqual(case['scope'],'open_pit_wall_failure')
        self.assertEqual(case['quality_review_status'],'accepted_for_development_reference_only')
        self.assertEqual(case['role'],'development')
        self.assertGreaterEqual(case['model_predictions_run'],8)
        self.assertTrue(case['historical_event_model_exposure'])
        self.assertTrue(case['no_new_blind_test_claim'])
        self.assertEqual(case['new_original_pair_model_predictions_run'],case['auxiliary_feature_classifiers_scored']+case['native_CNN_checkpoints_scored'])
        self.assertGreaterEqual(case['new_original_pair_model_predictions_run'],9)
        self.assertEqual(protocol.assert_development_cases(['CN_XJ2023'])[0]['role'],'development')
        xinjing_assets={k:v for k,v in value['additional_frozen_assets'].items() if k.startswith('development/Xinjing/')}
        self.assertEqual(len(xinjing_assets),4)
        acquisition=json.loads((protocol.DATA/case['acquisition']).read_text(encoding='utf-8'))
        self.assertEqual(acquisition['images']['pre']['datetime'][:10],'2023-02-15')
        self.assertEqual(acquisition['images']['post']['datetime'][:10],'2023-02-23')
        self.assertFalse(acquisition['images']['post']['reference_calendar_day_match'])
        self.assertFalse(case['independent_expert_or_field_audit'])
        key=next(iter(value['additional_frozen_assets']))
        value['additional_frozen_assets'][key]='0'*64
        with self.assertRaisesRegex(ValueError,'Frozen mine asset changed'):
            protocol.verify_assets(value)

    def test_bingham_candidate_is_not_yet_accepted_or_predicted(self):
        value=protocol.registry();case=value['cases']['US_BC2021']
        self.assertEqual(case['role'],'reserved_final_test')
        self.assertEqual(case['quality_review_status'],'pending')
        self.assertEqual(case['model_predictions_run'],0)
        self.assertFalse(case['new_unseen_main_event_accepted'])
        self.assertFalse(case['boundary_available'])
        self.assertFalse(case['new_unseen_mine_eligible'])
        self.assertTrue(case['same_mine_historical_exposure_verified'])
        acquisition=json.loads((protocol.DATA/case['acquisition']).read_text(encoding='utf-8'))
        self.assertLess(acquisition['images']['pre']['datetime'][:10],case['event_date'])
        self.assertGreater(acquisition['images']['post']['datetime'][:10],case['event_date'])
        self.assertEqual(acquisition['images']['pre']['shape'],acquisition['images']['post']['shape'])
        assets={k:v for k,v in value['additional_frozen_assets'].items() if k.startswith('reserved_candidates/US_BC2021/')}
        self.assertEqual(len(assets),4)


if __name__=="__main__":
    unittest.main(verbosity=2)
