"""Choose a blend ONLY on internal val; previously tested mine scenes are regression."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME",str(ROOT/"cache/torch"))
os.environ["SMP_DISABLE_VERSION_CHECK"]="1"
import json
import sys
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from run_rgb_joint_study import validation,confusion,sha,write
from evaluate_mine_study import predict_patches

STUDY=ROOT/"experiments/rgb_guarded_20261001"
MODELS=ROOT/"models_prior"
PREVIOUS=ROOT/"experiments/rgb_balanced_20261001"


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    path=STUDY/"protocol.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # These constraints are committed before any ensemble test predictions.
    recipe={"registered_at":datetime.now().isoformat(timespec="seconds"),
        "reason":"standalone paired RGB model failed the preregistered generic-background regression gate",
        "candidate_source":"paired study's internal-validation winner rgb_joint, seed 42 fixed",
        "adapter_weight_grid":[round(i/20,2) for i in range(21)],"threshold":.5,
        "grid_refinement":"coarse internal grid had no eligible mixture; refined to .05 steps without changing constraints or consulting external mixture results",
        "internal_background_cap":"current internal block activation + .01",
        "RGB_gain_min":.02,"source_NIR_base_retention":.85,
        "selection":"highest internal NIR/RGB harmonic F1 satisfying background cap and RGB gain; no external metric access",
        "regression_gates":{"source_NIR_base_retention":.85,"source_RGB_current_retention":1.,"generic_background_block_max":.10},
        "test_status":"all mine references already observed in the standalone study; regression, not a new blind claim",
        "no_target_threshold_selection":True,"seeds":[42,43,44],
        "training_changes":False,"adapter_weight_fixed_for_all_seeds":True}
    prior_failure=STUDY/"internal_calibration_failure.json"
    archive=STUDY/"internal_calibration_coarse_failure.json"
    if prior_failure.exists() and not archive.exists():
        archive.write_bytes(prior_failure.read_bytes())
    write(STUDY/"calibration_plan.json",recipe)
    vx=np.load(ROOT/"experiments/core_ablation_20261001/val_x.npy")
    vy=np.load(ROOT/"experiments/core_ablation_20261001/val_y.npy")
    rgb=vx.copy(); rgb[:,3]=rgb[:,2]
    current_path=MODELS/"real_ft_T1b_ours_seed42/best_model.pth"
    adapter_path=MODELS/"rgb_balanced_20261001/rgb_joint_seed42/best_model.pth"
    outputs=[]
    for weights in (current_path,adapter_path):
        detector=LandslideDetector(weights)
        outputs.append([predict_patches(detector,view) for view in (vx,rgb)])
        del detector
        torch.cuda.empty_cache()
    old=json.loads((PREVIOUS/"protocol.json").read_text(encoding="utf-8"))["initial_validation"]
    source_base=old["source_base"]
    current_rgb=confusion(outputs[0][1][:400],vy[:400])["F1"]
    current_bg=float(((outputs[0][1][400:]>=.5).sum((1,2))>=10).mean())
    rows=[]
    for alpha in recipe["adapter_weight_grid"]:
        nir,rgb_prob=[(1-alpha)*outputs[0][view]+alpha*outputs[1][view] for view in (0,1)]
        mn,mr=confusion(nir[:400],vy[:400]),confusion(rgb_prob[:400],vy[:400])
        bg=float(((rgb_prob[400:]>=.5).sum((1,2))>=10).mean())
        passes=(mn["F1"]>=.85*source_base["source_NIR"]["F1"] and mr["F1"]>=current_rgb+.02 and bg<=current_bg+.01+1e-12)
        score=2*mn["F1"]*mr["F1"]/max(mn["F1"]+mr["F1"],1e-9)
        rows.append({"adapter_weight":alpha,"NIR":mn,"RGB":mr,"background_block_activation":bg,"eligible":passes,"score":score})
    eligible_rows=[r for r in rows if r["eligible"]]
    if not eligible_rows:
        write(STUDY/"internal_calibration_failure.json",{"plan":recipe,"grid":rows})
        raise ValueError("No blend improves RGB while respecting current background cap")
    chosen=max(eligible_rows,key=lambda r:r["score"])
    protocol={**recipe,"adapter_weight":chosen["adapter_weight"],"internal_calibration":rows,
        "initial_source_base_validation":source_base,"current_validation_RGB_F1":current_rgb,
        "components":{"current":{"path":"real_ft_T1b_ours_seed42/best_model.pth","sha256":sha(current_path)},
                      "adapter":{"path":"rgb_balanced_20261001/rgb_joint_seed42/best_model.pth","sha256":sha(adapter_path)}},
        "frozen_validation":{"x":sha(ROOT/"experiments/core_ablation_20261001/val_x.npy"),"y":sha(ROOT/"experiments/core_ablation_20261001/val_y.npy")},
        "standalone_result_sha256":sha(PREVIOUS/"evaluation/results.json")}
    write(path,protocol)
    print(json.dumps({"selected_alpha":chosen["adapter_weight"],"validation_RGB":chosen["RGB"]["F1"],"current_RGB":current_rgb,"background":chosen["background_block_activation"],"grid":[[r["adapter_weight"],r["RGB"]["F1"],r["background_block_activation"],r["eligible"]] for r in rows]},ensure_ascii=False),flush=True)
    return protocol


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    prepare()
