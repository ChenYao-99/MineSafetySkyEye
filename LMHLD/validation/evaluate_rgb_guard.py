"""Check a val-selected blend on reused tests, explicitly as regression checks."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME",str(ROOT/"cache/torch"))
os.environ["SMP_DISABLE_VERSION_CHECK"]="1"
import json
import sys
from datetime import datetime
import cv2
import numpy as np
import torch

sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from finetune_negatives import DATA_ROOT,NEG_NPY,norm_patch
from evaluate_mine_study import metrics,predict_patches,prepare_background_holdout
from run_rgb_joint_study import sha,write
from calibrate_rgb_guard import prepare,STUDY,MODELS,PREVIOUS


def read(path):
    return cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED)


def evaluate():
    protocol=prepare()
    out=STUDY/"evaluation"
    out.mkdir(exist_ok=True)
    final=out/"results.json"
    if final.exists():
        existing=json.loads(final.read_text(encoding="utf-8"))
        if existing["protocol_sha256"]!=sha(STUDY/"protocol.json"):
            raise ValueError("Existing regression belongs to another calibration")
        print("Reusing completed blend regression",flush=True)
        return
    prev=json.loads((PREVIOUS/"evaluation/results.json").read_text(encoding="utf-8"))
    if sha(PREVIOUS/"evaluation/results.json")!=protocol["standalone_result_sha256"]:
        raise ValueError("Standalone regression changed after calibration")
    base,current=[next(r for r in prev["models"] if r["arm"]==tag) for tag in ("source_base","current_mine")]
    source=Path(DATA_ROOT)/"Emilia-Romagna_Italy_32"
    sx=norm_patch(np.load(source/"test_images.npy").astype(np.float32))
    sy=np.load(source/"test_labels.npy")[:,0]>0
    rgb=sx.copy(); rgb[:,3]=rgb[:,2]
    split=json.loads((ROOT/"experiments/nir_train/neg_split_seed42.json").read_text(encoding="utf-8"))
    neg=prepare_background_holdout(np.load(NEG_NPY),split["holdout_idx"])
    old_detector=LandslideDetector(MODELS/protocol["components"]["current"]["path"])
    old_probs=[predict_patches(old_detector,view) for view in (sx,rgb,neg)]
    del old_detector
    torch.cuda.empty_cache()
    pos_root=ROOT/"experiments/positive_expansion_20261001"
    pos=json.loads((pos_root/"positive_cases.json").read_text(encoding="utf-8"))
    old_root=ROOT/"experiments/mining_validation_20261001"
    old=json.loads((old_root/"validation_cases.json").read_text(encoding="utf-8"))
    for name,digest in pos["frozen_files"].items():
        if sha(pos_root/name)!=digest:
            raise ValueError("Positive reference changed")
    cases=[("hpakant_20200702",pos_root,pos["hpakant"]),
           (old["cases"][0]["id"],old_root,old["cases"][0]),
           (old["cases"][1]["id"],old_root,old["cases"][1])]
    alpha=protocol["adapter_weight"]
    rows=[]
    for seed in protocol["seeds"]:
        joint=next(r for r in prev["models"] if r["arm"]=="rgb_joint" and r["seed"]==seed)
        if sha(Path(joint["weights"]))!=joint["weights_sha256"]:
            raise ValueError("Adapter weights changed")
        detector=LandslideDetector(joint["weights"])
        adapter_probs=[predict_patches(detector,view) for view in (sx,rgb,neg)]
        probabilities=[(1-alpha)*a+alpha*b for a,b in zip(old_probs,adapter_probs)]
        negative=probabilities[2]>=.5
        row={"arm":"rgb_guarded","seed":seed,"adapter_weight":alpha,
            "source_test":metrics(probabilities[0]>=.5,sy),"source_RGB_proxy_test":metrics(probabilities[1]>=.5,sy),
            "generic_background_holdout":{"n":len(neg),"block_activation":float((negative.sum((1,2))>=10).mean()),
                "pixel_activation":float(negative.mean()),"limit":"pseudo-background from same original scene pool"},"cases":{}}
        folder=out/f"rgb_guarded_seed{seed}"
        folder.mkdir(exist_ok=True)
        for key,root,case in cases:
            # Blend existing, frozen component predictions; targets never affect alpha.
            oldp=np.load(PREVIOUS/f"evaluation/current_mine_seed42/{key}_probability.npy")
            newp=np.load(PREVIOUS/f"evaluation/rgb_joint_seed{seed}/{key}_probability.npy")
            prob=(1-alpha)*oldp+alpha*newp
            np.save(folder/f"{key}_probability.npy",prob)
            if "reference" in case:
                raw,truth,valid=read(root/case["image"]),read(root/case["reference"])>0,read(root/case["valid"])>0
                item={"primary@0.5":metrics(prob>=.5,truth,valid),"secondary@0.4":metrics(prob>=.4,truth,valid),
                      "label_quality":case["label_quality"],"evaluation_role":"reused figure-reference regression"}
                for threshold in (.5,.4):
                    p=prob>=threshold
                    review=raw.copy()
                    for mask,color in ((p&truth&valid,(40,190,40)),(p&~truth&valid,(0,190,240)),(~p&truth&valid,(40,40,230))):
                        review[mask]=(.45*review[mask]+.55*np.array(color)).astype(np.uint8)
                    review[~valid]=(.3*review[~valid]).astype(np.uint8)
                    cv2.imencode(".png",review)[1].tofile(str(folder/f"{key}_review_t{threshold}.png"))
            else:
                item={"qualitative_only":True,"whole_image_activation@0.5":float((prob>=.5).mean()),"note":"activation is not accuracy"}
            row["cases"][key]=item
        row["regression_gates"]={"source_NIR_retention":row["source_test"]["F1"]>=.85*base["source_test"]["F1"],
            "source_RGB_retention":row["source_RGB_proxy_test"]["F1"]>=current["source_RGB_proxy_test"]["F1"],
            "generic_background":row["generic_background_holdout"]["block_activation"]<=.10}
        rows.append(row)
        print(json.dumps({"seed":seed,"RGB":row["source_RGB_proxy_test"]["F1"],"background":row["generic_background_holdout"]["block_activation"],"gates":row["regression_gates"]}),flush=True)
        del detector
        torch.cuda.empty_cache()
    passes=all(rows[0]["regression_gates"].values())
    result={"completed_at":datetime.now().isoformat(timespec="seconds"),"protocol_sha256":sha(STUDY/"protocol.json"),
            "baselines":[base,current],"models":rows,"service_candidate_seed":42,"service_candidate_passed":passes,
            "test_status":protocol["test_status"],"default_mine_replacement":False}
    write(final,result)
    if passes:
        folder=MODELS/"rgb_guarded_20261001"
        folder.mkdir(exist_ok=True)
        config={"schema_version":1,"adapter_weight":alpha,"components":protocol["components"],
                "regression_passed":True,"threshold":.5,"calibration_protocol_sha256":sha(STUDY/"protocol.json"),
                "validation_note":"已做源域/背景与公开矿山图示参考回归；图示参考未独立专家审核，尚无现场试用验证"}
        write(folder/"adapter.json",config)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate()
