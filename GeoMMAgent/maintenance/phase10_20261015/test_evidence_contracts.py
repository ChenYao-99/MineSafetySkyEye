"""Reject concrete provenance mistakes exposed by this source batch."""
import unittest
from pathlib import Path
from qualify_mine_context import DATA, read, sha


class EvidenceContracts(unittest.TestCase):
    def test_ground_inspections_do_not_create_satellite_negatives(self):
        pack = read(DATA / "qualification/normal_field_reference_pack.json")
        self.assertEqual(pack["dated_reports"], 2)
        self.assertEqual(pack["independent_mine_groups"], 1)
        self.assertEqual(pack["physical_inspected_face_types"], 2)
        self.assertEqual(pack["verified_satellite_negative_pixels"], 0)
        self.assertEqual(len({row["image_sha256"] for row in pack["observations"]}), 4)
        for row in pack["observations"]:
            self.assertFalse(row["georeferenced"])
            self.assertFalse(row["pixel_labels_admitted"])
            self.assertFalse(row["training_allowed"])
            self.assertFalse(row["blind_test_allowed"])
            self.assertEqual(sha(row["image"]), row["image_sha256"])

    def test_future_plan_and_small_internal_residual_do_not_certify_safety(self):
        plan = read(DATA / "qualification/xincheng_plan_native_vectors.json")
        self.assertEqual(plan["plan_year"], 2025)
        self.assertFalse(plan["may_be_applied_as_2024_working_face_truth"])
        self.assertFalse(plan["safe_negative_label"])
        self.assertFalse(plan["absolute_survey_accuracy_certified"])
        self.assertLess(max(row["residual_m"] for row in plan["independent_control_symbol_checks"]), 2)
        geo = read(DATA / "qualification/xincheng_2025_planned_work_context.geojson")
        for row in geo["features"]:
            self.assertFalse(row["properties"]["training_allowed"])
            self.assertFalse(row["properties"]["use_as_2024_working_face_truth_allowed"])

    def test_crs_interpretation_does_not_certify_pixel_registration(self):
        row = read(DATA / "qualification/guohua2022_nominal_crs_clarification.json")
        self.assertTrue(row["nominal_definition_clarified"])
        self.assertFalse(row["survey_epoch_and_pixel_registration_independently_certified"])
        self.assertFalse(row["source_coordinates_modified"])
        self.assertFalse(row["training_allowed"])
        self.assertLess(row["difference_from_previous_coarse_transform_m_for_selected_object"], .01)

    def test_global_footprint_coverage_is_not_all_real_mines(self):
        row = read(DATA / "qualification/global_mine_discovery.json")
        self.assertEqual(row["discovery_region_mine_polygons"], 1)
        self.assertEqual(row["mine_accident_labels_admitted"], 0)
        self.assertEqual(row["new_predictions"], 0)
        self.assertEqual(sum(source["fresh_large_priority_objects"] for source in row["sources"]), 0)

    def test_official_area_agreement_is_not_an_original_expert_polygon(self):
        row = read(DATA / "qualification/xinjing_official_extent_and_chronology.json")
        self.assertEqual(row["official_approximate_affected_area_m2"], 230000)
        self.assertFalse(row["original_expert_GIS_received"])
        self.assertFalse(row["area_agreement_certifies_boundary"])
        self.assertFalse(row["pre_event_image_implies_safe_background"])
        self.assertFalse(row["new_blind_mine"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
