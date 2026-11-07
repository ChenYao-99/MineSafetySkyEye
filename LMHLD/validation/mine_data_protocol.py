# -*- coding: utf-8 -*-
"""Mine-level split policy and a one-shot final-test gate for the new data."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets/mine_benchmark_20261002"
REGISTRY=DATA/"mine_case_registry.json"
ORIGINAL_RESERVED_SOURCE_CASES=frozenset({"CL0002","KG0001","US_BC2021"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry():
    value=json.loads(REGISTRY.read_text(encoding="utf-8"))
    if value["schema_version"]!=1:
        raise ValueError("Unsupported mine registry")
    return value


def reserved_case_ids(value=None):
    """Include later registered reserves, rather than only the initial source cases."""
    value=registry() if value is None else value
    return frozenset(case_id for case_id,case in value['cases'].items()
                     if case.get('role')=='reserved_final_test')


# Compatibility for existing diagnostics. New callers can supply a current
# registry to reserved_case_ids() when cases are registered during a process.
RESERVED=reserved_case_ids() if REGISTRY.exists() else ORIGINAL_RESERVED_SOURCE_CASES


def assert_mine_group_role(case_id,expected_role,value=None):
    """An explicit same-mine group cannot be split across development and final roles."""
    value=registry() if value is None else value
    case=value['cases'].get(case_id)
    if case is None:raise ValueError(f'Unregistered event: {case_id}')
    group=case.get('mine_group') or case.get('mine_id') or case_id
    final=expected_role=='reserved_final_test'
    for other_id,other in value['cases'].items():
        other_group=other.get('mine_group') or other.get('mine_id') or other_id
        if other_group==group and (other.get('role')=='reserved_final_test')!=final:
            raise ValueError(f'Reserved/development mine group conflict: {group}; {case_id}/{other_id}')
    return group


def assert_development_cases(case_ids,value=None):
    value=registry() if value is None else value
    if not case_ids:
        raise ValueError("Development cases must be explicit, not a directory glob")
    for eid in case_ids:
        if eid not in value["cases"]:
            raise ValueError(f"Unregistered event: {eid}")
        case=value["cases"][eid]
        if case["role"]!="auxiliary_development" and case["role"]!="development":
            raise ValueError(f"Reserved final-test event cannot enter development: {eid}")
        assert_mine_group_role(eid,case['role'],value)
    return [value["cases"][eid] for eid in case_ids]


def verify_assets(value=None):
    value=registry() if value is None else value
    assets={**value["frozen_assets"],**value.get("additional_frozen_assets",{})}
    for name,digest in assets.items():
        path=(DATA/name).resolve()
        if not path.is_relative_to(DATA.resolve()) or sha(path)!=digest:
            raise ValueError(f"Frozen mine asset changed or escaped dataset root: {name}")
    return True


def validate_final_selection(plan):
    """A boolean alone must not open an untouched test: freeze reviewable evidence."""
    metrics=plan.get('metrics')
    if not isinstance(metrics,list) or not metrics or any(not isinstance(m,str) or not m for m in metrics) or len(set(metrics))!=len(metrics):
        raise ValueError('Explicit nonempty unique final metrics are required')
    threshold=plan.get('threshold')
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<threshold<1:
        raise ValueError('Final threshold must be a fixed finite number in (0,1)')
    evidence=plan.get('development_selection_record')
    if not isinstance(evidence,dict) or not {'path','sha256'}.issubset(evidence):
        raise ValueError('Freeze a development selection record path and SHA256, not an unverified statement')
    path=Path(evidence['path']).resolve()
    if not path.is_relative_to((ROOT/'experiments').resolve()) or not path.is_file() or sha(path)!=evidence['sha256']:
        raise ValueError('Frozen development selection record changed or escaped D: experiments')
    record=json.loads(path.read_text(encoding='utf-8'))
    audit=plan.get('evaluation_purpose')=='frozen_version_independent_audit'
    if audit:
        if record.get('status')!='accepted_for_frozen_version_audit' or record.get('evaluation_purpose')!=plan['evaluation_purpose']:
            raise ValueError('Frozen version audit requires its explicit evidence record')
        authorization=record.get('user_authorization',{})
        if not isinstance(authorization,dict) or not authorization.get('instruction') or authorization.get('source')!='user_message':
            raise ValueError('Frozen version audit requires recorded user authorization')
        if not isinstance(record.get('development_gates_passed'),bool):
            raise ValueError('Audit must preserve the actual development gate outcome')
        if record.get('reserved_results_may_select_or_tune_model') is not False:
            raise ValueError('Audit results cannot select or tune a model')
    elif record.get('status')!='accepted_for_final_evaluation' or record.get('development_gates_passed') is not True:
        raise ValueError('Development selection requires completed passing model evidence')
    if record.get('source_regression_passed') is not True:
        raise ValueError('Completed passing source regression evidence is required')
    if record.get('reserved_predictions_used_for_selection')!=0 or record.get('threshold')!=threshold:
        raise ValueError('Final selection used reserved predictions or changed its frozen threshold')
    expected=sorted(item['sha256'] for item in plan.get('weights',[]))
    if not expected or record.get('selected_weights_sha256')!=expected:
        raise ValueError('Final weights differ from development selection')
    protocol=record.get('input_protocol',{})
    if not {'path','sha256'}.issubset(protocol):
        raise ValueError('Freeze the selected model input/preprocessing protocol')
    pp=Path(protocol['path']).resolve()
    if not pp.is_relative_to((ROOT/'experiments').resolve()) or not pp.is_file() or sha(pp)!=protocol['sha256']:
        raise ValueError('Frozen selected input protocol changed or escaped D: experiments')
    return record


def register_final_run(plan_path,output_dir):
    value=registry()
    verify_assets(value)
    plan=json.loads(Path(plan_path).read_text(encoding="utf-8"))
    ids=plan.get("case_ids",[])
    if not ids or len(set(ids))!=len(ids):
        raise ValueError("Explicit, unique final case IDs are required")
    for eid in ids:
        case=value["cases"].get(eid,{})
        if case.get("role")!="reserved_final_test" or case.get("quality_review_status")!="accepted":
            raise ValueError(f"Final case is not accepted/frozen: {eid}")
        if case.get('new_unseen_mine_eligible') is False:
            raise ValueError(f'Previously observed mine cannot enter new-mine final evaluation: {eid}')
        assert_mine_group_role(eid,'reserved_final_test',value)
        if case.get('benchmark_usage_contract'):
            contract_path=Path(case['benchmark_usage_contract'])
            if sha(contract_path)!=case.get('benchmark_usage_contract_sha256'):
                raise ValueError('Frozen independent benchmark contract changed')
            contract=json.loads(contract_path.read_text(encoding='utf-8'))
            scope=plan.get('reference_scopes',{}).get(eid)
            input_mode=plan.get('input_modes',{}).get(eid)
            if scope not in contract['scopes'] or input_mode not in contract['scopes'][scope]['input_modes']:
                raise ValueError('Freeze benchmark input mode and matching scored-reference scope before final evaluation')
            if not set(plan.get('metrics',[])).issubset(set(contract['allowed_metrics'])):
                raise ValueError('Final metrics exceed accepted independent reference scope')
    required=("weights","threshold","metrics","source_regression_passed","development_selection_record")
    if any(key not in plan for key in required) or plan["source_regression_passed"] is not True:
        raise ValueError("Freeze weights, threshold, metrics and passing development regression before final test")
    selection=validate_final_selection(plan)
    for item in plan["weights"]:
        if sha(item["path"])!=item["sha256"]:
            raise ValueError("Final weights differ from frozen hashes")
    if not plan["weights"]:
        raise ValueError("Final model weights are required")
    output=Path(output_dir).resolve()
    if not output.is_relative_to((ROOT/"experiments").resolve()):
        raise ValueError("Final output must stay in D:/LMHLD/experiments")
    output.mkdir(parents=True,exist_ok=True)
    lock=DATA/"final_evaluation_once.json"
    record={"registered_at":datetime.now().isoformat(timespec="seconds"),"plan_sha256":sha(plan_path),
            "registry_sha256":sha(REGISTRY),"case_ids":ids,"output":str(output),"plan":plan,
            "gate_implementation_sha256":sha(__file__),"frozen_development_selection":selection}
    # Never silently rerun or select another winner on an already opened test set.
    with lock.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
    return record


def prepare():
    if REGISTRY.exists():
        verify_assets()
        print("Reusing reserved mine registry")
        return registry()
    quality=json.loads((DATA/"quality_audit/quality_audit.json").read_text(encoding="utf-8"))
    ledger=json.loads((DATA/"selected_archive_members.json").read_text(encoding="utf-8"))
    assets={str(Path(row["path"]).relative_to(DATA)).replace("\\","/"):row["sha256"] for row in ledger.values()}
    cases={}
    for row in quality["cases"]:
        eid=row["EID"]
        cases[eid]={"mine_id":eid,"event_date":row["event_date"],"scope":row["scope"],
            "role":"reserved_final_test" if eid in ORIGINAL_RESERVED_SOURCE_CASES else "auxiliary_development",
            "quality_review_status":"pending","geometric_gate_pass":row["geometric_gate_pass"],
            "raw_source":"https://zenodo.org/records/17007637","license":"CC BY 4.0",
            "audit":f"quality_audit/{eid}/audit.json","use_all_times_with_same_mine_role":True,
            "model_predictions_run":0,"independent_expert_or_field_audit":False,
            "unresolved_quality":["cloud/observation coverage review"]+(["same-day acquisition/event chronology review"] if row["same_day_post"] else [])}
    value={"schema_version":1,"registered_at":datetime.now().isoformat(timespec="seconds"),
        "split_unit":"mine + event group; all dates and overlapping patches stay together",
        "cases":cases,"frozen_assets":assets,"new_test_model_predictions_run":0,
        "original_cases_status":"Las Cruces/Bingham/Hpakant/Collolar and all previously probed mine mosaics are development/regression, never new blind tests",
        "quality_review_scope":"file/geometry/date/visibility checks are allowed before any model predictions; do not choose methods from final-test outcomes",
        "test_scope_reporting":"pit-wall, mine-adjacent and tailings-dam failures must be reported separately"}
    REGISTRY.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    verify_assets(value)
    print(f"Reserved {len(reserved_case_ids(value))} mine-related final candidates; {len(assets)} frozen original assets; no model predictions")
    return value


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        prepare()
    if args.verify:
        verify_assets(); print("Frozen assets verified")
