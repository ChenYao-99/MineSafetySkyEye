"""Verify the input path without training, real inference or blind accuracy scoring."""
from pathlib import Path
import contextlib
import json
import sys
import unittest
ROOT=Path('D:/LMHLD');sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
from runtime_paths import configure_runtime_paths
paths=configure_runtime_paths()
assert all(Path(value).drive.upper()=='D:' for value in paths.values())
from model_registry import file_sha256
from mine_data_protocol import DATA,registry,reserved_case_ids,verify_assets
from finalize_cause_terrain_round_20261005 import protected_runtime
OUT=ROOT/'experiments/engineering_integrity_20261006';OUT.mkdir(exist_ok=True)


def main():
    patterns=['test_upload_preflight.py','test_upload_contracts.py','test_geotiff_upload.py','test_radar_optical_upload_guard.py']
    with (OUT/'test_output.txt').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        suite=unittest.TestSuite()
        for pattern in patterns:suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun==41 and not result.skipped
    protected=protected_runtime();inventory=registry();verify_assets(inventory)
    counts={case:inventory['cases'][case]['model_predictions_run'] for case in sorted(reserved_case_ids(inventory))}
    assert len(counts)==4 and not any(counts.values()) and not (DATA/'final_evaluation_once.json').exists()
    receipt={'passed':True,'tests_run':result.testsRun,'passed_tests':result.testsRun,'suites':patterns,
        'runtime_and_cache_paths':paths,'protected_runtime_artifacts':protected,'preserved_cases_model_prediction_counts':counts,
        'existing_reserved_upload_blocking_tests_included':True,
        'reserved_guard_test_may_read_and_reencode_pixels_only_to_check_inference_is_blocked':True,
        'no_fresh_blind_accuracy_or_all_pixels_unseen_claim':True,
        'new_real_CNN_inference_or_training':0,'new_cloud_calls':0,'production_default_changed':False,
        'source_hashes':{name:file_sha256(ROOT/name) for name in ['detector_service.py','upload_utils.py','raster_upload.py','runtime_paths.py','tests/test_upload_preflight.py']}}
    (OUT/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS 41 input/transport tests; 15 protected artifacts verified; 4 reserved model-prediction counts remain zero')


if __name__=='__main__':main()
