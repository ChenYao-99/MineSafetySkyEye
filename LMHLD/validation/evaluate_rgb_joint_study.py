"""One-shot external evaluation AFTER completed RGB development; no test selection."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME",str(ROOT/"cache/torch"))
os.environ["SMP_DISABLE_VERSION_CHECK"]="1"

import argparse
import hashlib
import json
import sys
from datetime import datetime
import cv2
import numpy as np
import torch

sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from finetune_negatives import norm_patch,DATA_ROOT,NEG_NPY
from evaluate_mine_study import metrics,predict_patches,prepare_background_holdout

OLD=ROOT/"experiments/mining_validation_20261001"
POS=ROOT/"experiments/positive_expansion_20261001"
STUDY=ROOT/"experiments/rgb_balanced_20261001"
OUT=STUDY/"evaluation"


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def image(path):
    result=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED)
    if result is None:
        raise ValueError(f"Image decoding failed: {path}")
    return result


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"evaluation_plan.json"
    expected={"training_protocol_sha256":sha(STUDY/"protocol.json"),
              "positive_manifest_sha256":sha(POS/"positive_cases.json"),
              "old_cases_sha256":sha(OLD/"validation_cases.json")}
    if path.exists():
        plan=json.loads(path.read_text(encoding="utf-8"))
        if any(plan[k]!=v for k,v in expected.items()):
            raise ValueError("Evaluation inputs differ from registration")
        return plan
    plan={"registered_at":datetime.now().isoformat(timespec="seconds"),**expected,
          "primary_threshold":.5,"secondary_threshold":.4,
          "primary_pair":"rgb_joint_minepos - rgb_joint, paired seeds 42/43/44, balanced recipe",
          "target_limit":"Hpakant collapsed-source interior and outside-runout background only; other runout is unknown",
          "old_test_status":"Las Cruces already observed before RGB study; regression only, not a fresh blind test",
          "new_test_status":"Hpakant reference frozen before first prediction and excluded from all development",
          "uniform_pilot":"retained as internal development failure, not eligible for promotion",
          "choose_candidate":"higher 3-seed mean internal selection score among arms with all internal gates; deployment seed fixed 42",
          "promotion_regression_gates":{"source_NIR_base_retention":.85,"source_RGB_current_retention":1.0,
                                        "generic_background_block_max":.10},
          "no_test_winner_or_threshold_selection":True,"no_metric_for_Bingham":True}
    write(path,plan)
    return plan


def completed():
    protocol=json.loads((STUDY/"protocol.json").read_text(encoding="utf-8"))
    rows=[]
    for arm in protocol["arms"]:
        for seed in protocol["seeds"]:
            path=STUDY/f"train_{arm}_seed{seed}.json"
            if not path.exists():
                raise ValueError(f"Training incomplete: {arm}/{seed}")
            row=json.loads(path.read_text(encoding="utf-8"))
            if row["completed_epochs"]!=protocol["epochs"] or row["protocol_sha256"]!=sha(STUDY/"protocol.json"):
                raise ValueError("All preregistered runs must finish before first external prediction")
            if sha(Path(row["weights"]))!=row["weights_sha256"]:
                raise ValueError("Checkpoint changed")
            rows.append(row)
    candidates=[]
    for arm in protocol["arms"]:
        group=[r for r in rows if r["arm"]==arm]
        if all(r["best_eligible"] for r in group):
            candidates.append((float(np.mean([r["best_internal_score"] for r in group])),arm))
    candidate=max(candidates)[1] if candidates else None
    write(OUT/"candidate_before_external_test.json",{"candidate_arm":candidate,"seed":42,
          "criterion":"internal validation only; committed before external predictions",
          "scores":{a:float(np.mean([r["best_internal_score"] for r in rows if r["arm"]==a])) for a in protocol["arms"]}})
    return rows,candidate


def evaluate(plan):
    final=OUT/"results.json"
    if final.exists():
        old=json.loads(final.read_text(encoding="utf-8"))
        if old["plan"]!=plan:
            raise ValueError("Existing evaluation belongs to another plan")
        print("Reusing complete external evaluation",flush=True)
        return
    rows,candidate=completed()
    pos=json.loads((POS/"positive_cases.json").read_text(encoding="utf-8"))
    for name,digest in pos["frozen_files"].items():
        if sha(POS/name)!=digest:
            raise ValueError("Frozen positive reference changed")
    cases=json.loads((OLD/"validation_cases.json").read_text(encoding="utf-8"))["cases"]
    work=[{"arm":"source_base","seed":None,"weights":str(ROOT/"models_prior/Emilia-Romagna_Italy_32_upload_seed42/best_model.pth")},
          {"arm":"current_mine","seed":42,"weights":str(ROOT/"models_prior/real_ft_T1b_ours_seed42/best_model.pth")},*rows]
    region=Path(DATA_ROOT)/"Emilia-Romagna_Italy_32"
    sx=norm_patch(np.load(region/"test_images.npy").astype(np.float32))
    sy=np.load(region/"test_labels.npy")[:,0]>0
    split=json.loads((ROOT/"experiments/nir_train/neg_split_seed42.json").read_text(encoding="utf-8"))
    neg=prepare_background_holdout(np.load(NEG_NPY),split["holdout_idx"])
    targets=[("hpakant_20200702",POS,pos["hpakant"]),
             (cases[0]["id"],OLD,cases[0]),(cases[1]["id"],OLD,cases[1])]
    partial=OUT/"results_partial.json"
    prior=json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {"plan":plan,"models":[]}
    if prior["plan"]!=plan:
        raise ValueError("Partial evaluation protocol mismatch")
    all_results=prior["models"]
    for record in work:
        if any(r["arm"]==record["arm"] and r["seed"]==record["seed"] for r in all_results):
            continue
        arm,seed=record["arm"],record["seed"]
        detector=LandslideDetector(record["weights"],threshold=.5,patch_size=32)
        source=metrics(predict_patches(detector,sx)>=.5,sy)
        rgb=sx.copy(); rgb[:,3]=rgb[:,2]
        rgb_source=metrics(predict_patches(detector,rgb)>=.5,sy)
        npred=predict_patches(detector,neg)>=.5
        result={"arm":arm,"seed":seed,"weights":record["weights"],"weights_sha256":sha(Path(record["weights"])),
                "source_test":source,"source_RGB_proxy_test":rgb_source,
                "generic_background_holdout":{"n":len(neg),"block_activation":float((npred.sum((1,2))>=10).mean()),
                    "pixel_activation":float(npred.mean()),"limit":"same-pool pseudo-background, not expert audited mine FPR"},
                "best_internal_eligible":record.get("best_eligible"),"cases":{}}
        folder=OUT/f"{arm}_seed{seed}"
        folder.mkdir(exist_ok=True)
        for key,root,case in targets:
            raw=image(root/case["image"])
            four=np.concatenate([raw,raw[:,:,2:3]],2).astype(np.float32)
            prob=detector.detect_image(four,patch_norm="minmax",max_batch=128)
            np.save(folder/f"{key}_probability.npy",prob)
            if "reference" in case:
                truth,valid=image(root/case["reference"])>0,image(root/case["valid"])>0
                m={"primary@0.5":metrics(prob>=.5,truth,valid),"secondary@0.4":metrics(prob>=.4,truth,valid),
                   "label_quality":case["label_quality"],"scored_pixels":int(valid.sum())}
                for threshold in (.5,.4):
                    p=prob>=threshold
                    review=raw.copy()
                    for mask,color in ((p&truth&valid,(40,190,40)),(p&~truth&valid,(0,190,240)),(~p&truth&valid,(40,40,230))):
                        review[mask]=(.45*review[mask]+.55*np.array(color)).astype(np.uint8)
                    review[~valid]=(.3*review[~valid]).astype(np.uint8)
                    cv2.imencode(".png",review)[1].tofile(str(folder/f"{key}_review_t{threshold}.png"))
            else:
                m={"qualitative_only":True,"whole_image_activation@0.5":float((prob>=.5).mean()),
                   "note":"event-confirmed image without a mask; activation is not accuracy"}
            result["cases"][key]=m
        all_results.append(result)
        write(partial,{"plan":plan,"models":all_results})
        print(f"{arm}/{seed}: source NIR={source['F1']:.4f}, RGB={rgb_source['F1']:.4f}, Hpakant F1={result['cases']['hpakant_20200702']['primary@0.5']['F1']:.4f}, Las F1={result['cases'][cases[0]['id']]['primary@0.5']['F1']:.4f}",flush=True)
        del detector
        torch.cuda.empty_cache()
    base,current=[next(r for r in all_results if r["arm"]==name) for name in ("source_base","current_mine")]
    chosen=next((r for r in all_results if r["arm"]==candidate and r["seed"]==42),None)
    gates={}
    if chosen:
        gates={"source_NIR_retention":chosen["source_test"]["F1"]>=.85*base["source_test"]["F1"],
               "source_RGB_retention":chosen["source_RGB_proxy_test"]["F1"]>=current["source_RGB_proxy_test"]["F1"],
               "generic_background":chosen["generic_background_holdout"]["block_activation"]<=.10}
    outcome={"candidate_arm":candidate,"seed":42,"regression_gates":gates,
             "eligible_for_optional_experimental_service":bool(gates) and all(gates.values()),
             "default_mine_replacement":False,
             "reason":"small figure-derived test references are insufficient to validate broad operational warning"}
    write(final,{"completed_at":datetime.now().isoformat(timespec="seconds"),"plan":plan,"models":all_results,"selection":outcome})


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--register-only",action="store_true")
    args=parser.parse_args()
    plan=register()
    if not args.register_only:
        evaluate(plan)
