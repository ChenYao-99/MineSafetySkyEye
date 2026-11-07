# -*- coding: utf-8 -*-
"""Matched-budget native-mine label-policy study; reserved final mines are never read."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME",str(ROOT/"cache/torch"))
os.environ.setdefault("HF_HOME",str(ROOT/"cache/huggingface"))
os.environ["SMP_DISABLE_VERSION_CHECK"]="1"

import argparse
import json
import random
import sys
import time
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0,str(ROOT))
from finetune_negatives import DS
from run_rgb_joint_study import model_at,validation,sha,write,DEVICE,CURRENT
from evaluate_mine_study import metrics
from mine_data_protocol import assert_development_cases,verify_assets,REGISTRY

OUT=ROOT/"experiments/native_boundary_20261002"
DATA_OUT=OUT
OLD=ROOT/"experiments/core_ablation_20261001"
MODEL_OUT=ROOT/"models_prior/native_boundary_20261002"
ARMS=("source_replay","weak_patch","partial_boundary")
DISTILLATION_WEIGHT=0.


def partial_loss(logits,labels,confidence):
    """Unknown pixels contribute neither BCE nor Dice gradient; confidence is retained."""
    target=(labels==1).to(logits.dtype)
    weight=confidence.to(logits.dtype)*(labels!=255).to(logits.dtype)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction="none")
    bce=(bce*weight).mean((1,2,3))
    prob=torch.sigmoid(logits)
    intersection=(prob*target*weight).sum((1,2,3))
    denominator=(prob*weight).sum((1,2,3))+(target*weight).sum((1,2,3))
    dice=(1-(2*intersection+1)/(denominator+1))*weight.mean((1,2,3))
    return (.5*bce+.5*dice).mean()


class PartialDataset:
    def __init__(self,x,y,w,seed):
        self.x,self.y,self.w=x,y,w
        self.rng=np.random.default_rng(seed)

    def get(self,index,weak=False,kind=None):
        x,y,w=self.x[index].copy(),self.y[index].copy(),self.w[index].copy()
        if weak:
            value=1 if kind=="positive" else 0
            y=np.full_like(y,value); w=np.full_like(w,.6 if value else .25)
        # Always consume the same augmentation draws for weak/partial arms.
        if self.rng.random()<.5:
            x,y,w=x[:,:,::-1],y[:,::-1],w[:,::-1]
        if self.rng.random()<.5:
            x,y,w=x[:,::-1,:],y[::-1,:],w[::-1,:]
        k=int(self.rng.integers(0,4))
        x,y,w=np.rot90(x,k,(1,2)),np.rot90(y,k),np.rot90(w,k)
        return tuple(torch.as_tensor(np.ascontiguousarray(a)) for a in (x,y,w))


def native_validation(model,x,y,indices):
    model.eval(); results={}
    valid=y[indices]!=255
    truth=y[indices]==1
    for tag,copy_red in (("measured_NIR",False),("RGB_proxy",True)):
        probabilities=[]
        with torch.no_grad():
            for start in range(0,len(indices),64):
                rows=indices[start:start+64]
                batch=np.array(x[rows],copy=True)
                if copy_red:
                    batch[:,3]=batch[:,2]
                p=torch.sigmoid(model(torch.as_tensor(batch,device=DEVICE)))[:,0].cpu().numpy()
                probabilities.append(p)
        results[tag]=metrics(np.concatenate(probabilities)>=.5,truth,valid)
    return results


def readout(model,vx,vy,mx,my,indices):
    source=validation(model,vx,vy)
    mine=native_validation(model,mx,my,indices)
    harmonic=2*source["source_NIR"]["F1"]*source["source_RGB_proxy"]["F1"]/max(source["source_NIR"]["F1"]+source["source_RGB_proxy"]["F1"],1e-9)
    return {"source":source,"native_spatial_validation":mine,
            "selection_score":.4*harmonic+.6*mine["RGB_proxy"]["F1"]-.1*source["generic_background"]["block_activation"]}


def eligible(value,initial):
    s=value["source"]; b=initial["source"]
    return (s["source_NIR"]["F1"]>=.85*b["source_NIR"]["F1"] and
            s["source_RGB_proxy"]["F1"]>=.95*b["source_RGB_proxy"]["F1"] and
            s["generic_background"]["block_activation"]<=.10 and
            value["native_spatial_validation"]["RGB_proxy"]["recall"]>=.50)


def prepare():
    assert_development_cases(["Amyntaio"]); verify_assets()
    data=json.loads((DATA_OUT/"development_data.json").read_text(encoding="utf-8"))
    for name,digest in data["frozen_arrays"].items():
        if sha(DATA_OUT/name)!=digest:
            raise ValueError("Native development data changed")
    old=json.loads((OLD/"protocol.json").read_text(encoding="utf-8"))
    for name,digest in old["frozen_files"].items():
        if sha(OLD/name)!=digest:
            raise ValueError("Frozen source data changed")
    path=OUT/"training_protocol.json"
    if path.exists():
        plan=json.loads(path.read_text(encoding="utf-8"))
        if plan["development_data_sha256"]!=sha(DATA_OUT/"development_data.json") or plan["initial_weights_sha256"]!=sha(CURRENT):
            raise ValueError("Training protocol inputs changed")
        return plan
    indices=[i for i,w in enumerate(data["val_windows"]) if w["box"][0]%32==0 and w["box"][1]%32==0]
    if len(indices)<8:
        raise ValueError("Too few non-overlapping native validation windows")
    plan={"registered_at":datetime.now().isoformat(timespec="seconds"),"arms":list(ARMS),"seeds":[42,43,44],
        "epochs":15,"updates_per_epoch":120,"batch":32,"common_batch":24,"replay_batch":8,
        "common_sampling":"12 source positive + 12 background, identical RNG between arms",
        "replay_sampling":"4 positive + 4 negative; weak/partial arms use identical actual native mine image windows",
        "source_replay":"source positive/negative patches instead of mine; reference arm, not a label-only single-factor comparison",
        "primary_pair":"partial_boundary minus weak_patch, same images, seeds, update budget and geometric augmentations",
        "weak_labels":"all selected positive patch pixels assigned 1/.6 and selected negative patches 0/.25",
        "partial_labels":"known pixels only; positive confidence .6, weak negative .25, unknown255 zero gradient",
        "coverage_limit":"label coverage and effective weighted supervision differ by policy; report rather than claiming equal labeled pixels",
        "initial_weights":str(CURRENT),"initial_weights_sha256":sha(CURRENT),
        "optimizer":"SGD lr=.0002 momentum=.9 weight_decay=.0001; ExponentialLR .96",
        "supervision":"paired actual four-band and NIR:=R views, .5 BCE+.5 soft Dice",
        "native_validation_indices":indices,"native_validation_overlap":"32px grid, no duplicated validation image pixels",
        "thresholds":{"primary":.5,"secondary":.4},
        "selection":".4 source NIR/RGB harmonic F1 + .6 native spatial-val RGB F1 - .1 generic background block activation",
        "eligible_gate":{"source_NIR_current_retention":.85,"source_RGB_current_retention":.95,"generic_background_block_max":.10,"native_spatial_RGB_recall_min":.50},
        "if_no_eligible":"save best diagnostic score, no promotion",
        "distillation_weight":DISTILLATION_WEIGHT,
        "source_guard":"fixed current teacher, Bernoulli KL only on common source/generic replay views; never impose teacher negatives on actual mine positives",
        "development_reason":"first native study internal generic-background gate failed; no new reserved prediction used" if DISTILLATION_WEIGHT else "first native label-policy study",
        "development_data_sha256":sha(DATA_OUT/"development_data.json"),"source_protocol_sha256":sha(OLD/"protocol.json"),
        "mine_registry_sha256":sha(REGISTRY),"training_mines":["Amyntaio"],
        "cross_domain_regression_only":["old Las Cruces figure","old Hpakant figure"],
        "new_reserved_mines_read_for_prediction":False,"single_native_development_mine_limit":True}
    write(path,plan)
    model=model_at(CURRENT)
    value=readout(model,np.load(OLD/"val_x.npy"),np.load(OLD/"val_y.npy"),np.load(DATA_OUT/"mine_val_x.npy"),np.load(DATA_OUT/"mine_val_y.npy"),indices)
    write(OUT/"initial_validation.json",value)
    del model; torch.cuda.empty_cache()
    print(f"Preregistered native boundary study, {len(indices)} non-overlapping val windows; reserved mines excluded.",flush=True)
    print(json.dumps({"initial_native":value["native_spatial_validation"]}),flush=True)
    return plan


def train(arm,seed,plan):
    result_path=OUT/f"train_{arm}_seed{seed}.json"
    folder=MODEL_OUT/f"{arm}_seed{seed}"; folder.mkdir(parents=True,exist_ok=True)
    digest=sha(OUT/"training_protocol.json")
    best_path=folder/"best_model.pth"
    resume_path=folder/"resume.pth"
    if result_path.exists():
        previous=json.loads(result_path.read_text(encoding="utf-8"))
        if previous["protocol_sha256"]!=digest:
            raise ValueError("Run protocol mismatch")
        if previous["completed_epochs"]==plan["epochs"]:
            print(f"Reusing {arm}/{seed}",flush=True); return
        if not resume_path.exists():
            raise ValueError("Incomplete run has no recoverable checkpoint")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    images=np.concatenate([np.load(OLD/"common_x.npy"),np.load(OLD/"ours_extra.npy")])
    labels=np.concatenate([np.load(OLD/"common_y.npy"),np.zeros((2000,32,32),np.float32)])
    positive=np.flatnonzero(labels.mean((1,2))>=.1); negative=np.flatnonzero(labels.mean((1,2))==0)
    common=DS(images,labels,train=True,seed=seed)
    mx,my,mw=[np.load(DATA_OUT/f"mine_train_{name}.npy") for name in ("x","y","weight")]
    metadata=json.loads((DATA_OUT/"development_data.json").read_text(encoding="utf-8"))
    mp=np.array([i for i,w in enumerate(metadata["train_windows"]) if w["kind"]=="positive"])
    mn=np.array([i for i,w in enumerate(metadata["train_windows"]) if w["kind"]=="negative"])
    replay=PartialDataset(mx,my,mw,seed+200)
    source_replay=DS(images,labels,train=True,seed=seed+200)
    vx,vy=np.load(OLD/"val_x.npy"),np.load(OLD/"val_y.npy")
    vmx,vmy=np.load(DATA_OUT/"mine_val_x.npy"),np.load(DATA_OUT/"mine_val_y.npy")
    initial=json.loads((OUT/"initial_validation.json").read_text(encoding="utf-8"))
    model=model_at(CURRENT)
    teacher=model_at(CURRENT).eval() if plan.get("distillation_weight",0) else None
    if teacher is not None:
        teacher.requires_grad_(False)
    optimizer=torch.optim.SGD(model.parameters(),lr=.0002,momentum=.9,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.96)
    history=[]; best=-float("inf"); best_epoch=0; best_eligible=False; started=time.time()
    start_epoch=1
    if result_path.exists():
        state=torch.load(resume_path,map_location=DEVICE,weights_only=False)
        if state["protocol_sha256"]!=digest:
            raise ValueError("Resume checkpoint protocol mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history,best,best_epoch,best_eligible=state["history"],state["best"],state["best_epoch"],state["best_eligible"]
        common.rng.bit_generator.state=state["common_rng"]
        replay.rng.bit_generator.state=state["replay_rng"]
        source_replay.rng.bit_generator.state=state["source_replay_rng"]
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([a.cpu() for a in state["cuda_rng"]])
        start_epoch=state["epoch"]+1
        print(f"Resuming {arm}/{seed} at epoch {start_epoch}",flush=True)
    for epoch in range(start_epoch,plan["epochs"]+1):
        rng=np.random.default_rng(seed*10000+epoch)
        rrng=np.random.default_rng(seed*10000+epoch+1000000)
        model.train(); losses=[]; supervision=[]; distillation_losses=[]
        for _ in range(plan["updates_per_epoch"]):
            ids=np.concatenate([rng.choice(positive,12,replace=True),rng.choice(negative,12,replace=True)])
            pairs=[(*common[int(i)],torch.ones((32,32))) for i in ids]
            if arm=="source_replay":
                for group,kind in ((positive,"positive"),(negative,"negative")):
                    for i in rrng.choice(group,4,replace=True):
                        xx,yy=source_replay[int(i)]
                        pairs.append((xx,yy,torch.full((32,32),.6 if kind=="positive" else .25)))
            else:
                for group,kind in ((mp,"positive"),(mn,"negative")):
                    for i in rrng.choice(group,4,replace=True):
                        pairs.append(replay.get(int(i),weak=arm=="weak_patch",kind=kind))
            x=torch.stack([p[0] for p in pairs]).to(DEVICE)
            y=torch.stack([p[1] for p in pairs]).to(DEVICE).unsqueeze(1)
            w=torch.stack([p[2] for p in pairs]).to(DEVICE).unsqueeze(1)
            rgb=torch.cat([x[:,:3],x[:,2:3]],1)
            logits=model(torch.cat([x,rgb]))
            loss=partial_loss(logits,torch.cat([y,y]),torch.cat([w,w]))
            if teacher is not None:
                source_logits=torch.cat([logits[:24],logits[32:56]])
                with torch.no_grad():
                    teacher_logits=teacher(torch.cat([x[:24],rgb[:24]]))
                    teacher_prob=torch.sigmoid(teacher_logits)
                    entropy=torch.nn.functional.binary_cross_entropy_with_logits(teacher_logits,teacher_prob,reduction="none")
                kl=(torch.nn.functional.binary_cross_entropy_with_logits(source_logits,teacher_prob,reduction="none")-entropy).mean()
                loss=loss+plan["distillation_weight"]*kl
                distillation_losses.append(float(kl))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss)); supervision.append(float(w.mean()))
        scheduler.step()
        value=readout(model,vx,vy,vmx,vmy,plan["native_validation_indices"])
        passes=eligible(value,initial); score=value["selection_score"]
        if (passes and not best_eligible) or (passes==best_eligible and score>best):
            best,best_epoch,best_eligible=score,epoch,passes
            torch.save(model.state_dict(),best_path)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"mean_pixel_supervision_weight":float(np.mean(supervision)),
                        "source_guard_KL":float(np.mean(distillation_losses)) if distillation_losses else 0.,
                        "validation":value,"eligible":passes,"elapsed_seconds":round(time.time()-started,2)})
        record={"arm":arm,"seed":seed,"completed_epochs":epoch,"protocol_sha256":digest,"weights":str(best_path),
            "best_epoch":best_epoch,"best_eligible":best_eligible,"best_score":best,
            "best_validation":history[best_epoch-1]["validation"],"history":history}
        write(result_path,record)
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                    "epoch":epoch,"protocol_sha256":digest,"history":history,"best":best,"best_epoch":best_epoch,
                    "best_eligible":best_eligible,"common_rng":common.rng.bit_generator.state,
                    "replay_rng":replay.rng.bit_generator.state,"source_replay_rng":source_replay.rng.bit_generator.state,
                    "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},resume_path)
        if epoch==1 or epoch%5==0:
            m=value["native_spatial_validation"]["RGB_proxy"]
            print(f"{arm}/{seed} {epoch}/15: nativeRGB F1={m['F1']:.4f}, recall={m['recall']:.3f}, FPR={m['background_FPR']:.3f}, eligible={passes}",flush=True)
    record["weights_sha256"]=sha(best_path); write(result_path,record)
    del model,optimizer,teacher; torch.cuda.empty_cache()


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--distillation-weight",type=float,choices=[0.,.5,1.],default=0.)
    args=parser.parse_args()
    if args.distillation_weight:
        DISTILLATION_WEIGHT=args.distillation_weight
        OUT=ROOT/f"experiments/native_source_guard_20261002/lambda{DISTILLATION_WEIGHT}"
        MODEL_OUT=ROOT/f"models_prior/native_source_guard_20261002/lambda{DISTILLATION_WEIGHT}"
        ARMS=("partial_boundary",)
        args.arms=["partial_boundary"]
        OUT.mkdir(parents=True,exist_ok=True)
    if DEVICE!="cuda" and not args.prepare_only:
        raise ValueError("CUDA required for this preregistered GPU budget")
    plan=prepare()
    if not args.prepare_only:
        for seed in args.seeds:
            for arm in args.arms:
                train(arm,seed,plan)
