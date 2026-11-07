"""Data/metric checks that require no model inference."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "experiments" / "core_ablation_20261001"
STUDY = ROOT / "experiments" / "mining_validation_20261001"


class StudyProtocol(unittest.TestCase):
    def test_source_and_internal_validation_do_not_share_identifiers(self):
        protocol = json.loads((TRAIN / "protocol.json").read_text(encoding="utf-8"))
        def key(row):
            if row.get("kind") == "baige_weak_positive":
                return ("baige", tuple(row["box"]))
            return (row.get("kind", "lmhld"), row.get("region", row.get("array", "")), row.get("split", ""), row["index"])
        a = {key(row) for row in protocol["common_train_provenance"]}
        b = {key(row) for row in protocol["internal_validation_provenance"]}
        self.assertFalse(a & b)
        self.assertTrue(any(row.get("kind") == "baige_weak_positive" for row in protocol["common_train_provenance"]))
        self.assertFalse(any(row.get("kind") == "baige_weak_positive" for row in protocol["internal_validation_provenance"]))

    def test_pair_pools_have_the_same_size_and_same_arid_half(self):
        a, b = np.load(TRAIN / "ours_extra.npy"), np.load(TRAIN / "naive_extra.npy")
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a[:600], b[:600])
        self.assertFalse(np.array_equal(a[600:], b[600:]))

    def test_case_mask_and_excluded_windows_do_not_leak_the_published_line(self):
        case = json.loads((STUDY / "validation_cases.json").read_text(encoding="utf-8"))["cases"][0]
        def image(key):
            return cv2.imdecode(np.fromfile(STUDY / case[key], np.uint8), cv2.IMREAD_UNCHANGED)
        raw, valid, target = image("image"), image("valid") > 0, image("reference") > 0
        b, g, r = [ch.astype(np.int16) for ch in cv2.split(raw)]
        red = (r > 100) & (r-g > 25) & (r-b > 20)
        self.assertFalse((red & valid).any())
        for y in range(0, len(valid), 32):
            for x in range(0, valid.shape[1], 32):
                part = valid[y:y+32, x:x+32]
                self.assertTrue(part.all() or not part.any())
        self.assertEqual(int((target & valid).sum()), case["scored_positive_pixels"])

    def test_known_event_without_pixel_truth_is_not_a_pixel_benchmark(self):
        cases = json.loads((STUDY / "validation_cases.json").read_text(encoding="utf-8"))["cases"]
        nasa = next(row for row in cases if row["id"].startswith("bingham"))
        self.assertNotIn("reference", nasa)
        self.assertEqual(nasa["evaluation_mode"], "confirmed_event_without_pixel_truth")

    def test_confusion_metrics_respect_ignore_pixels(self):
        code = ast.parse((ROOT / "validation/evaluate_mine_study.py").read_text(encoding="utf-8"))
        node = next(n for n in code.body if isinstance(n, ast.FunctionDef) and n.name == "metrics")
        ns = {"np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "metrics", "exec"), ns)
        result = ns["metrics"](np.array([1, 1, 0, 1]), np.array([1, 0, 1, 0]), np.array([1, 1, 1, 0]))
        self.assertEqual([result[k] for k in ("TP", "FP", "FN", "TN")], [1, 1, 1, 0])
        self.assertAlmostEqual(result["F1"], .5)

    def test_background_holdout_is_invariant_to_affine_brightness(self):
        ns = {"np": np}
        for file, name in ((ROOT / "finetune_negatives.py", "norm_patch"),
                           (ROOT / "validation/evaluate_mine_study.py", "prepare_background_holdout")):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), ns)
        images = np.random.default_rng(0).uniform(40, 140, (2, 4, 32, 32)).astype(np.float32)
        original = ns["prepare_background_holdout"](images, [1])
        shifted = ns["prepare_background_holdout"](images * 2 + 10, [1])
        np.testing.assert_allclose(original, shifted, atol=.001)
        self.assertAlmostEqual(float(original.min()), 0, places=3)
        self.assertAlmostEqual(float(original.max()), 255, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
