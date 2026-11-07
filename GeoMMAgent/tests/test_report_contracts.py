"""Report contract integration tests: actual Flask routes, isolated model/network calls."""
import base64
import argparse
import io
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend as service
from archive_test_support import isolate_archive
from report_grounding import (build_grounding, check_structured_geometry, fallback_report,
                              check_detection_numbers, check_report_citations, check_unsupported_safety_claims)

SMOKE_DIR = None


class ReportContracts(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service)
        self.client = service.app.test_client()
        service._report_cache.clear()
        service._llm_check_cache.update(t=0, ok=None)
        self.check = patch.object(service, "llm_fast_check", return_value=False)
        self.check.start()
        self.addCleanup(self.check.stop)
        self.coord = patch.object(service, "coordinator", object())
        self.coord.start()
        self.addCleanup(self.coord.stop)
        self.region = {"id": 1, "area_px": 937, "perimeter_px": 179.3,
                       "major_axis_px": 54.94, "minor_axis_px": 12.0}
        self.det = {"overlay_b64": "", "risk_level": "orange", "risk_name": "橙色-高风险",
                    "stats": {"area_ratio": .125, "num_regions": 1, "num_regions_total": 1,
                              "max_region_area": 937, "scope": "roi", "regions": [self.region]},
                    "quant": {"ndvi": {"available": False}, "region": None},
                    "roi": {"box_px": [0, 0, 32, 32]}, "model_tag": "mine", "threshold": .4,
                    "band_meta": {"nir_source": "red_proxy", "note": "缺少真实NIR"}}

    def pipeline(self):
        with patch.object(service, "detector_request", return_value=self.det) as detector:
            response = self.client.post("/api/pipeline_upload", json={"image": "dummy",
                "roi": {"x0": .1, "y0": .1, "x1": .9, "y1": .9}, "model_tag": "mine",
                "channel_order": "auto", "instance_tau": .6, "nir_synth": False})
        return response, detector

    def test_upload_options_are_forwarded_to_the_detector(self):
        response, detector = self.pipeline()
        self.assertEqual(response.status_code, 200)
        payload = detector.call_args.args[1]
        self.assertIn("roi", payload)
        self.assertEqual(payload["instance_tau"], .6)
        self.assertEqual(payload["channel_order"], "auto")
        body = response.get_json()
        self.assertEqual(body["detection"]["roi"], self.det["roi"])
        self.assertIn("监测范围内", body["question"])
        self.assertNotIn("NDVI 指数:", body["question"])

    def test_reference_grid_is_preserved_through_pipeline_and_cached_export(self):
        self.det['band_meta']['spatial_input_contract']={'mode':'reference_10m','reference_meters_per_pixel':10,'accuracy_on_real_high_resolution_sensors_verified':False}
        with patch.object(service,'detector_request',return_value=self.det) as call:
            response=self.client.post('/api/pipeline_upload',json={'image':'dummy','analysis_grid':'reference_10m'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(call.call_args.args[1]['analysis_grid'],'reference_10m')
        result=response.get_json()
        archived=service.case_archive.get_report(result['report_id'])
        self.assertEqual(archived['detection']['band_meta']['spatial_input_contract']['mode'],'reference_10m')
        self.assertFalse(archived['detection']['band_meta']['spatial_input_contract']['accuracy_on_real_high_resolution_sensors_verified'])

    def test_unavailable_api_gives_measured_summary_not_simulated_risk(self):
        body = self.pipeline()[0].get_json()
        self.assertEqual(body["generation"]["status"], "fallback")
        self.assertEqual(body["generation"]["source"], "detection_summary")
        self.assertTrue(body["quality"]["degraded"])
        self.assertFalse(body["is_mock"])
        self.assertEqual(body["agents"], [])
        self.assertIn("12.5%", body["final_answer"])
        self.assertIn("橙色", body["final_answer"])
        self.assertNotIn("Overall risk: LOW", body["final_answer"])

    def test_export_uses_the_same_cached_result(self):
        body = self.pipeline()[0].get_json()
        with patch.object(service, "run_pipeline_upload", side_effect=AssertionError("must not rerun")), \
             patch.object(service, "render_report_png", return_value="png") as renderer:
            response = self.client.post("/api/report_upload", json={"report_id": body["report_id"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(renderer.call_args.args[0]["final_answer"], body["final_answer"])
        self.assertEqual(renderer.call_args.args[0]["detection"]["roi"], body["detection"]["roi"])

    def test_expired_or_unknown_export_returns_410(self):
        response = self.client.post("/api/report_upload", json={"report_id": "missing"})
        self.assertEqual(response.status_code, 410)

    def test_detector_validation_status_is_preserved(self):
        with patch.object(service, "detector_request", return_value={"error": "roi invalid", "status_code": 400}):
            response = self.client.post("/api/pipeline_upload", json={"image": "dummy"})
            self.assertEqual(response.status_code, 400)
            response = self.client.post("/api/detect_upload", json={"image": "dummy"})
            self.assertEqual(response.status_code, 400)

    def test_quality_rejection_and_model_trace_are_preserved(self):
        for status,kind in ((422,'unusable_observation'),(503,'busy'),(503,'failed_component')):
            error={'error':'no conclusion','status_code':status,'inference_status':kind,'request_id':'rejected-request'}
            with patch.object(service,'detector_request',return_value=error):
                response=self.client.post('/api/pipeline_upload',json={'image':'dummy'})
            self.assertEqual(response.status_code,status)
            self.assertNotIn('detection',response.get_json())
            self.assertNotIn('final_answer',response.get_json())
        self.det.update(request_id='verified-request',input_quality={'status':'usable_observation'},
                        model_info={'artifact':{'weights_sha256':'frozen-sha'}})
        body=self.pipeline()[0].get_json()
        self.assertEqual(body['detection']['request_id'],'verified-request')
        self.assertEqual(service.report_facts(body['detection'])['model_info']['artifact']['weights_sha256'],'frozen-sha')
        with patch.object(service,'run_pipeline_upload',side_effect=AssertionError('must not rerun')):
            cached=service.report_for_export({'report_id':body['report_id']},lambda: None)
        self.assertEqual(cached['detection']['input_quality'],self.det['input_quality'])

    def test_typed_guard_checks_measurement_name_and_unit(self):
        grounding = build_grounding([self.region], 1, 2)
        cited, bad = check_structured_geometry("周长179.3像素，长轴54.94像素。", grounding)
        self.assertTrue(cited)
        self.assertEqual(bad, [])
        self.assertTrue(check_structured_geometry("长轴179.3像素。", grounding)[1])
        self.assertTrue(check_structured_geometry("面积937米。", grounding)[1])
        self.assertTrue(check_structured_geometry("长轴54.94米。", grounding)[1])
        self.assertEqual(check_structured_geometry("长轴109.88米。", grounding)[1], [])

    def test_wrong_geometry_fails_the_production_guard(self):
        grounding = build_grounding([self.region], 1)
        answer = "橙色风险，面积937像素，长轴179.3像素。建议监测。" * 12
        self.assertIn("几何数值一致", service.validate_report(answer, self.det["risk_name"], grounding))

    def test_mask_cannot_certify_safety_or_resolve_conflicting_pixels(self):
        facts=service.report_facts(self.det)
        for answer in ('矿区安全，已排除失稳风险。','边坡处于稳定状态，不会发生滑坡。'):
            self.assertTrue(check_unsupported_safety_claims(answer,facts))
            self.assertIn('筛查结论不越界',service.validate_report(answer*20,self.det['risk_name'],facts=facts))
        self.assertFalse(check_unsupported_safety_claims('当前结果不能证明边坡安全。建议评估边坡稳定状态。',facts))
        facts['stats']={**facts['stats'],'unverified_pixels':1024,'area_ratio_available':False}
        facts['risk_name']='待复核-证据冲突'
        report=fallback_report(facts,'test')
        self.assertIn('1024',report)
        self.assertIn('待复核',report)
        self.assertIn('面积占比不可计算',report)
        self.assertTrue(check_detection_numbers('异常面积占比为0%',facts))
        self.assertNotIn('风险等级一致',service.validate_report(report,facts['risk_name'],facts=facts))

    def test_wrong_coverage_count_and_citation_are_rejected(self):
        self.assertTrue(check_detection_numbers("面积占比为90%，共5处异常。", service.report_facts(self.det)))
        self.assertEqual(check_detection_numbers("面积占比为12.5%，共1处异常。", service.report_facts(self.det)), [])
        self.assertTrue(check_report_citations("依据[REG-002]", "资料[REG-002]"))
        self.assertFalse(check_report_citations("依据[REG-999]", "资料[REG-002]"))
        self.assertFalse(check_report_citations("没有来源编号", "资料[REG-002]"))

    def test_repair_has_original_facts_and_expected_risk(self):
        prompts = []
        class Agent:
            def __init__(self, **kwargs):
                pass
            def step(self, prompt):
                prompts.append(prompt)
                return SimpleNamespace(msgs=[SimpleNamespace(content="repaired")])
        with patch.object(service, "llm_fast_check", return_value=True), \
             patch.object(service, "create_model", return_value=object()), \
             patch("camel.agents.ChatAgent", Agent):
            result = service.repair_report("wrong draft", ["风险等级一致"], "真实等级橙色，面积937像素")
        self.assertEqual(result, "repaired")
        self.assertIn("真实等级橙色，面积937像素", prompts[0])

    def test_coordinator_instances_are_request_owned(self):
        import configs
        import coordinator
        class Coordinator:
            def __init__(self, **kwargs):
                self.workers = []
            def register_workers(self, workers):
                self.workers = workers
        config = {"agents": {"ReasoningAgent": {}, "MatchingAgent": {}, "SuperResolutionAgent": {}}}
        with patch.object(service, "_cfg", config), \
             patch.object(configs, "build_agents_from_config", side_effect=lambda cfg, **kwargs: [
                 {"name": name, "agent": object()} for name in cfg["agents"]
             ]) as build_workers, \
             patch.object(coordinator, "AgentCoordinator", Coordinator), \
             patch.object(service, "create_model", return_value=object()):
            first, second = service.get_open_coordinator(), service.get_open_coordinator()
        self.assertIsNot(first, second)
        self.assertEqual(list(build_workers.call_args.args[0]["agents"]), ["ReasoningAgent"])
        self.assertEqual(list(config["agents"]), ["ReasoningAgent", "MatchingAgent", "SuperResolutionAgent"])
        self.assertEqual([worker["name"] for worker in first.workers], ["ReasoningAgent"])
        self.assertIsNot(first.workers[0]["agent"], second.workers[0]["agent"])

    def test_outstanding_timeouts_cannot_spawn_unlimited_tasks(self):
        gate = threading.Event()
        started = []
        slots = threading.BoundedSemaphore(2)
        def slow():
            started.append(1)
            gate.wait(2)
        with patch.object(service, "_llm_slots", slots):
            try:
                service.run_with_timeout(slow, timeout=.02)
                service.run_with_timeout(slow, timeout=.02)
                service.run_with_timeout(slow, timeout=.02)
                self.assertEqual(len(started), 2)
            finally:
                gate.set()

    def test_knowledge_endpoint_exposes_real_sources(self):
        response = self.client.post("/api/knowledge", json={"query": "GB/T 37573是什么标准"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["docs"][0]["id"], "REG-002")
        self.assertTrue(body["docs"][0]["source_url"])
        self.assertEqual(body["answer"], "")

    def test_invalid_json_object_is_a_client_error(self):
        self.assertEqual(self.client.post("/api/pipeline_upload", json=[1]).status_code, 400)

    def test_compare_fallback_does_not_call_added_count_total_instances(self):
        answer = fallback_report({"kind": "compare", "risk_name": "黄色-中风险", "stats": {
            "t1_ratio": .1, "t2_ratio": .2, "net_ratio": .1,
            "n_added_regions": 2, "n_expanded_regions": 3, "n_removed_regions": 1}}, "服务不可用")
        self.assertIn("新增2处", answer)
        self.assertIn("扩大3处", answer)
        self.assertIn("10.00个百分点", answer)

    def test_render_real_detection(self):
        if not SMOKE_DIR:
            self.skipTest("optional real-detector fixture not requested")
        import json
        from pathlib import Path
        from PIL import Image
        root = Path(SMOKE_DIR)
        actual = json.loads((root / "real_detection.json").read_text(encoding="utf-8"))
        with patch.object(service, "detector_request", return_value=actual):
            result = service.run_pipeline_upload("dummy", model_tag="mine", roi={"unit": "fraction"})
        png = base64.b64decode(service.render_report_png(result))
        image = Image.open(io.BytesIO(png))
        self.assertGreater(image.width, 850)
        self.assertLess(image.width, 1100)
        self.assertGreater(image.height, image.width)
        self.assertGreater(len(png), 50000)
        (root / "verified_report.png").write_bytes(png)
        (root / "verified_report.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-dir")
    args, remaining = parser.parse_known_args()
    SMOKE_DIR = args.smoke_dir
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
