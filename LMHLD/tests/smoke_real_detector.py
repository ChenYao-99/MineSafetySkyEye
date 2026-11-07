"""One real-weight GPU/CPU smoke check, no external model API calls."""
import base64
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import detector_service as service


class RealDetectorSmoke(unittest.TestCase):
    def test_real_weights_large_image_and_roi(self):
        client = service.app.test_client()
        path = Path(r"D:\竞赛材料\真实场景验证\影像\alxa_xinjing.png")
        if not path.is_file():
            self.skipTest("public mine test image is unavailable")
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image)
        ok, raw = cv2.imencode(".png", image)
        self.assertTrue(ok)
        roi = {"unit": "fraction", "x0": .165, "y0": .13, "x1": .937, "y1": .937}
        result = client.post("/api/detect_upload", json={"image": base64.b64encode(raw).decode(),
            "model_tag": "mine", "roi": roi, "meters_per_pixel": 2})
        self.assertEqual(result.status_code, 200, result.get_json().get("error"))
        body = result.get_json()
        self.assertEqual(body["model_tag"], "mine")
        self.assertEqual(body["stats"]["scope"], "roi")
        self.assertEqual(body["stats"]["area_ratio"], body["roi"]["risk_inside"])
        self.assertFalse(body["quant"]["ndvi"]["available"])
        out = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else Path("experiments/smoke")
        out.mkdir(parents=True, exist_ok=True)
        (out / "real_detection.json").write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"image_shape": list(image.shape), "model": body["model_info"],
                          "scope": body["stats"]["scope"], "area_ratio": body["stats"]["area_ratio"],
                          "regions": body["stats"]["num_regions"],
                          "band_meta": body["band_meta"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # Optional output directory is deliberately not a unittest test-name argument.
    unittest.main(argv=[sys.argv[0]], verbosity=2)
