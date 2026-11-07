"""Scientific provenance checks; no network, detector or held-out predictions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import expand_event_references as expand


class SourceContracts(unittest.TestCase):
    def test_dates_preserve_missing_and_bad_values(self):
        for value in [None, float("nan"), float("inf"), "20220230", "20230714.1", ""]:
            self.assertIsNone(expand.date_value(value))
        self.assertEqual(str(expand.date_value(20230714.0)), "2023-07-14")
        self.assertEqual(str(expand.date_value("2023-07-14")), "2023-07-14")

    def test_cached_name_cannot_silently_change_remote_source(self):
        with tempfile.TemporaryDirectory(dir=expand.DATA, prefix="contract_check_") as folder:
            root = Path(folder)
            path = root / "sources/sample.json"
            expand.write(path, {"original": True})
            receipt = {"ok": True, "path": str(path), "sha256": expand.sha(path),
                       "requested_url": "https://example.org/original"}
            expand.write(root / "receipts/sample.json.json", receipt)
            with patch.object(expand, "DATA", root), patch.object(expand.ACQ, "acquire") as network:
                self.assertEqual(expand.acquire("sample.json", receipt["requested_url"]), receipt)
                with self.assertRaises(ValueError):
                    expand.acquire("sample.json", "https://example.org/another-date")
                network.assert_not_called()

    def test_all_qualified_windows_still_block_training_and_blind_scores(self):
        for case in ["TW_DAYU2023_SOURCE_CANDIDATE", "TW_GUOHUA2022_CLEAR_SOURCE_CANDIDATE"]:
            report = expand.read(expand.DATA / "imagery" / case / "qualification.json")
            self.assertFalse(report["training_allowed"])
            self.assertFalse(report["predictions_allowed"])
            self.assertFalse(report["final_blind_test_allowed"])
            self.assertFalse(report["full_accident_extent_confirmed"])
            for obj in report["objects"]:
                self.assertFalse(obj["mask_is_truth"])

    def test_fallback_only_changed_source_quality_not_model_selection(self):
        old = expand.read(expand.DATA / "qualification/guohua2022_rejected_pair.json")
        new = expand.read(expand.DATA / "qualification/guohua2022_clear_source_selection.json")
        search = expand.read(expand.DATA / "qualification/guohua2022_search_plan.json")
        self.assertEqual(old["qualification"]["joint_clear_fraction"], 0)
        self.assertEqual(new["bbox_wgs84"], search["bbox_wgs84"])
        self.assertTrue(search["windows"]["post"][0] <= new["source_dates"][1] <= search["windows"]["post"][1])
        self.assertEqual(new["original_reference_sensors"], ["SPOT-6/7", "Pleiades"])
        self.assertTrue(new["source_imagery_is_substitute_not_original_high_resolution"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
