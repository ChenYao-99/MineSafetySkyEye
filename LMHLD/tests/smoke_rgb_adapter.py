"""Real GPU endpoint check: model blend, batch bounds, ROI and band provenance."""
import base64
import json
import sys
import time
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
import detector_service as service


class RGBAdapterSmoke(unittest.TestCase):
    def test_real_adapter_on_large_RGB_upload_and_paired_periods(self):
        path=Path("D:/竞赛材料/真实场景验证/影像/alxa_xinjing.png")
        if not path.is_file():
            self.skipTest("Mine image unavailable")
        raw=cv2.imdecode(np.fromfile(path,np.uint8),1)
        client=service.app.test_client()
        payload={"image":base64.b64encode(cv2.imencode(".png",raw)[1]).decode(),
                 "model_tag":"rgb_guarded","roi":[.165,.13,.937,.937]}
        started=time.perf_counter()
        response=client.post("/api/detect_upload",json=payload)
        self.assertEqual(response.status_code,200,response.get_json().get("error"))
        body=response.get_json()
        self.assertEqual(body["model_tag"],"rgb_guarded")
        self.assertAlmostEqual(body["model_info"]["adapter_weight"],.15)
        self.assertEqual(body["model_info"]["validation_status"],"public_reference_regression_only")
        self.assertEqual(body["threshold"],.5)
        self.assertFalse(body["quant"]["ndvi"]["available"])
        self.assertEqual(body["stats"]["area_ratio"],body["roi"]["risk_inside"])
        self.assertEqual(body["stats"]["scope"],"roi")
        elapsed=time.perf_counter()-started
        detector=service._upload_detectors["rgb_guarded"][0]
        # Actual model output must match the val-calibrated probability mixture.
        patch=raw[:32,:32].astype(np.float32)
        patch=np.concatenate([patch,patch[:,:,2:3]],2).transpose(2,0,1)[None]
        mn,mx=patch.min((2,3),keepdims=True),patch.max((2,3),keepdims=True)
        patch=(patch-mn)/(mx-mn+1e-6)*255
        combined=detector.predict_batch(patch)
        with torch.no_grad():
            current=torch.sigmoid(detector.model(torch.as_tensor(np.ascontiguousarray(patch),device=detector.device)))[:,0].cpu().numpy()
        expected=.85*current+.15*detector.adapter.predict_batch(patch)
        np.testing.assert_allclose(combined,expected,rtol=1e-6,atol=1e-6)
        small=base64.b64encode(cv2.imencode(".png",raw[:96,:96])[1]).decode()
        pair=client.post("/api/compare",json={"image_t1":small,"image_t2":small,
             "registration":"prealigned","model_tag":"rgb_guarded"})
        self.assertEqual(pair.status_code,200,pair.get_json().get("error"))
        self.assertAlmostEqual(pair.get_json()["stats"]["net_ratio"],0)
        self.assertEqual(pair.get_json()["model_info"]["tag"],"rgb_guarded")
        out=ROOT/"experiments/rgb_guarded_20261001/smoke"
        out.mkdir(exist_ok=True)
        summary={"image_shape":list(raw.shape),"elapsed_seconds":elapsed,"device":detector.device,
                 "area_ratio":body["stats"]["area_ratio"],"model_info":body["model_info"],
                 "physical_NDVI_available":body["quant"]["ndvi"]["available"],"max_batch":256,
                 "paired_identical_image_net_ratio":pair.get_json()["stats"]["net_ratio"]}
        (out/"real_adapter_smoke.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
