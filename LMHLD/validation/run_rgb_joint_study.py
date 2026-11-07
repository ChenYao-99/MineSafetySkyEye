"""Paired RGB/full-band supervision, with a mine-positive replay ablation.

All weights/selection use training + internal validation only. Mine-event test
images are never read here. Outputs stay under D:/LMHLD.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key, folder in (("TORCH_HOME","torch"),("HF_HOME","huggingface")):
    os.environ.setdefault(key,str(ROOT/"cache"/folder))
os.environ["SMP_DISABLE_VERSION_CHECK"] = "1"

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime

sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import segmentation_models_pytorch as smp
from finetune_negatives import DS, norm_patch
from prepare_positive_expansion import prepare as prepare_positives

OLD = ROOT/"experiments/core_ablation_20261001"
POS = ROOT/"experiments/positive_expansion_20261001"
OUT = ROOT/"experiments/rgb_joint_20261001"
WEIGHTS = ROOT/"models_prior/rgb_joint_20261001"
BASE = ROOT/"models_prior/Emilia-Romagna_Italy_32_upload_seed42/best_model.pth"
CURRENT = ROOT/"models_prior/real_ft_T1b_ours_seed42/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = ("rgb_joint","rgb_joint_minepos")


def sha(path):
    h = hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def model_at(path):
    model = smp.UnetPlusPlus(encoder_name="resnet34",encoder_weights=None,in_channels=4,classes=1).to(DEVICE)
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
    return model


def confusion(prob,y):
    p,t = prob>=.5,y>0
    tp,fp,fn,tn = [int(a.sum()) for a in (p&t,p&~t,~p&t,~p&~t)]
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,
            "F1":2*tp/max(2*tp+fp+fn,1),"recall":tp/max(tp+fn,1),
            "background_FPR":fp/max(fp+tn,1)}


def validation(model,vx,vy):
    model.eval()
    outputs = []
    with torch.no_grad():
        for rgb in (False,True):
            chunks = []
            for start in range(0,len(vx),64):
                batch = np.array(vx[start:start+64],copy=True)
                if rgb:
                    batch[:,3] = batch[:,2]
                logits = model(torch.as_tensor(batch,device=DEVICE))
                chunks.append(torch.sigmoid(logits)[:,0].cpu().numpy())
            outputs.append(np.concatenate(chunks))
    # First 400 are genuine source-val masks; last 200 are pseudo-background.
    nir,rgb = [confusion(p[:400],vy[:400]) for p in outputs]
    negative = outputs[1][400:] >= .5
    background = {"block_activation":float((negative.sum((1,2))>=10).mean()),
                  "pixel_activation":float(negative.mean()),"n":len(negative)}
    harmonic = 2*nir["F1"]*rgb["F1"]/max(nir["F1"]+rgb["F1"],1e-9)
    return {"source_NIR":nir,"source_RGB_proxy":rgb,"generic_background":background,
            "selection_score":harmonic-.1*background["block_activation"]}


def eligible(readout,base):
    return (readout["source_NIR"]["F1"] >= .85*base["source_NIR"]["F1"]
            and readout["source_RGB_proxy"]["F1"] >= .95*base["source_RGB_proxy"]["F1"]
            and readout["generic_background"]["block_activation"] <= .10)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    pos = prepare_positives()
    old = json.loads((OLD/"protocol.json").read_text(encoding="utf-8"))
    frozen = {str(OLD/name):digest for name,digest in old["frozen_files"].items()}
    frozen.update({str(POS/name):digest for name,digest in pos["frozen_files"].items()})
    for name,digest in frozen.items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Frozen data changed: {name}")
    path = OUT/"protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if protocol["frozen_files"] != frozen or protocol["base_sha256"] != sha(Path(protocol["initial_weights"])):
            raise ValueError("Study inputs changed")
        return protocol
    val_origins = old["internal_validation_provenance"]
    if len(val_origins)!=600 or any(p.get("kind")=="generic_negative" for p in val_origins[:400]):
        raise ValueError("Unexpected validation layout")
    vx,vy = np.load(OLD/"val_x.npy"),np.load(OLD/"val_y.npy")
    initial = {}
    for name,weights in (("source_base",BASE),("current_mine",CURRENT)):
        model = model_at(weights)
        initial[name] = validation(model,vx,vy)
        del model
        torch.cuda.empty_cache()
    balanced = OUT.name.startswith("rgb_balanced")
    initial_weights = CURRENT if balanced else BASE
    protocol = {"registered_at":datetime.now().isoformat(timespec="seconds"),
        "recipe":"balanced_bce_dice" if balanced else "uniform_bce",
        "arms":list(ARMS),"seeds":[42,43,44],"epochs":15,"updates_per_epoch":120,
        "batch":32,"training_budget":"paired equal updates; smaller pilot than prior 30-epoch study",
        "initial_weights":str(initial_weights),"base_sha256":sha(initial_weights),
        "source_base_sha256":sha(BASE),
        "optimizer":"SGD lr=.0002 momentum=.9 weight_decay=.0001, ExponentialLR gamma=.96",
        "common_batch":28,"replay_batch":4,"replay_weight":.25,
        "common_data":"prior frozen common (incl. Baige weak positives) + same 2000 stratified negatives",
        "paired_difference":"4 weak Collolar positives vs 4 source positive replays per batch",
        "supervision":"mean of BCE on actual 4-channel input and paired NIR:=R view, same mask; no inferred NIR measurement",
        "augmentation":"paired horizontal/vertical flips and quarter rotations",
        "positive_balance":"14 source-positive + 14 background per common batch" if balanced else "uniform common sampling",
        "loss":"0.5 BCE + 0.5 smoothed per-image soft Dice" if balanced else "BCE",
        "development_reason":"v1 internal RGB validation collapse; no external mine predictions used" if balanced else "first paired RGB pilot",
        "selection":"internal source-val NIR/RGB harmonic F1 minus .1 * generic background block activation, threshold .5",
        "eligibility":{"source_NIR_retention":.85,"source_RGB_base_retention":.95,"generic_background_block_max":.10},
        "eligible_checkpoint_priority":True,"if_none_eligible":"best internal score saved but prohibited from automatic promotion",
        "thresholds":{"primary":.5,"secondary":.4},"initial_validation":initial,
        "external_tests":["new Hpakant collapsed-source figure reference","old Las Cruces interior figure reference","NASA Bingham qualitative"],
        "no_external_selection":True,"frozen_files":frozen,
        "positive_manifest_sha256":sha(POS/"positive_cases.json")}
    write(path,protocol)
    print("Registered paired RGB study; all mine tests remain excluded from selection.",flush=True)
    print(json.dumps(initial,ensure_ascii=False),flush=True)
    return protocol


def train(arm,seed,protocol):
    path = OUT/f"train_{arm}_seed{seed}.json"
    folder = WEIGHTS/f"{arm}_seed{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    best_path,resume_path = folder/"best_model.pth",folder/"resume.pth"
    digest = sha(OUT/"protocol.json")
    if path.exists():
        old_record = json.loads(path.read_text(encoding="utf-8"))
        if old_record["protocol_sha256"] != digest:
            raise ValueError("Existing run belongs to another protocol")
        if old_record["completed_epochs"]==protocol["epochs"] and best_path.exists():
            print(f"Reusing completed {arm}/{seed}",flush=True)
            return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    images = np.concatenate([np.load(OLD/"common_x.npy"),np.load(OLD/"ours_extra.npy")])
    targets = np.concatenate([np.load(OLD/"common_y.npy"),np.zeros((2000,32,32),np.float32)])
    replay_indices = np.flatnonzero(targets.mean((1,2))>=.1)
    background_indices = np.flatnonzero(targets.mean((1,2))==0)
    if arm=="rgb_joint_minepos":
        rx,ry = norm_patch(np.load(POS/"collolar_positive_x.npy")),np.load(POS/"collolar_positive_y.npy")
    else:
        rx,ry = images[replay_indices],targets[replay_indices]
    main,replay = DS(images,targets,train=True,seed=seed),DS(rx,ry,train=True,seed=seed+100)
    vx,vy = np.load(OLD/"val_x.npy"),np.load(OLD/"val_y.npy")
    model = model_at(Path(protocol["initial_weights"]))
    optimizer = torch.optim.SGD(model.parameters(),lr=.0002,momentum=.9,weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.96)
    history,best,best_epoch,best_eligible,start_epoch = [],-float("inf"),0,False,1
    if resume_path.exists():
        state = torch.load(resume_path,map_location=DEVICE,weights_only=False)
        if state["protocol_sha256"]!=digest:
            raise ValueError("Resume protocol mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        history,best,best_epoch,best_eligible = state["history"],state["best"],state["best_epoch"],state["best_eligible"]
        main.rng.bit_generator.state = state["main_rng"]
        replay.rng.bit_generator.state = state["replay_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        start_epoch = state["epoch"]+1
        print(f"Resuming {arm}/{seed} from epoch {start_epoch}",flush=True)
    started = time.time()
    for epoch in range(start_epoch,protocol["epochs"]+1):
        # Common sampling is identical between arms. Replay uses separate RNG so
        # changing its pool size cannot change common batches or augmentations.
        rng = np.random.default_rng(seed*10000+epoch)
        rrng = np.random.default_rng(seed*10000+epoch+1000000)
        model.train(); losses=[]
        for step in range(protocol["updates_per_epoch"]):
            if protocol.get("recipe")=="balanced_bce_dice":
                positive_ids = rng.choice(replay_indices,14,replace=True)
                background_ids = rng.choice(background_indices,14,replace=True)
                common_ids = np.concatenate([positive_ids,background_ids])
            else:
                common_ids = rng.integers(0,len(main),28)
            pairs = [main[int(i)] for i in common_ids]
            pairs += [replay[int(i)] for i in rrng.integers(0,len(replay),4)]
            x = torch.stack([p[0] for p in pairs]).to(DEVICE)
            y = torch.stack([p[1] for p in pairs]).to(DEVICE).unsqueeze(1)
            rgb = torch.cat([x[:,:3],x[:,2:3]],dim=1)
            logits = model(torch.cat([x,rgb]))
            per_image = torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.cat([y,y]),reduction="none").mean((1,2,3))
            if protocol.get("recipe")=="balanced_bce_dice":
                truth,prob = torch.cat([y,y]),torch.sigmoid(logits)
                dice = 1-(2*(truth*prob).sum((1,2,3))+1)/(truth.sum((1,2,3))+prob.sum((1,2,3))+1)
                per_image = .5*per_image+.5*dice
            weights = torch.ones(32,device=DEVICE); weights[28:]=.25
            weights = torch.cat([weights,weights])
            loss = (per_image*weights).sum()/weights.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss))
        schedule.step()
        readout = validation(model,vx,vy)
        passes = eligible(readout,protocol["initial_validation"]["source_base"])
        score = readout["selection_score"]
        if (passes and not best_eligible) or (passes==best_eligible and score>best):
            best,best_epoch,best_eligible = score,epoch,passes
            torch.save(model.state_dict(),best_path)
        row = {"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":readout,"eligible":passes,
               "elapsed_seconds_this_invocation":round(time.time()-started,2)}
        history.append(row)
        state = {"protocol_sha256":digest,"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                 "schedule":schedule.state_dict(),"history":history,"best":best,"best_epoch":best_epoch,"best_eligible":best_eligible,
                 "main_rng":main.rng.bit_generator.state,"replay_rng":replay.rng.bit_generator.state,
                 "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
        torch.save(state,resume_path)
        record = {"arm":arm,"seed":seed,"protocol_sha256":digest,"completed_epochs":epoch,
                  "best_epoch":best_epoch,"best_internal_score":best,"best_eligible":best_eligible,
                  "best_validation":history[best_epoch-1]["validation"],"weights":str(best_path),"history":history}
        write(path,record)
        print(f"{arm}/{seed} epoch {epoch}/15: NIR={readout['source_NIR']['F1']:.4f}, RGB={readout['source_RGB_proxy']['F1']:.4f}, neg={readout['generic_background']['block_activation']:.3f}, eligible={passes}, elapsed={time.time()-started:.0f}s",flush=True)
    record["weights_sha256"] = sha(best_path)
    write(path,record)
    del model,state,optimizer
    torch.cuda.empty_cache()


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant",choices=["uniform","balanced"],default="uniform")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    args=parser.parse_args()
    if args.variant=="balanced":
        OUT = ROOT/"experiments/rgb_balanced_20261001"
        WEIGHTS = ROOT/"models_prior/rgb_balanced_20261001"
    protocol=prepare()
    if not args.prepare_only:
        for seed in args.seeds:
            for arm in args.arms:
                train(arm,seed,protocol)
