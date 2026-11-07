# -*- coding: utf-8 -*-
"""True prior-image information vs identical-context control, same 8-channel model."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME",str(ROOT/"cache/torch"))
os.environ["SMP_DISABLE_VERSION_CHECK"]="1"
import argparse
import json
import random
import sys
import time
from datetime import datetime
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

sys.path.insert(0,str(ROOT))
from finetune_negatives import DS,norm_patch
from run_rgb_joint_study import CURRENT,DEVICE,model_at,validation,sha,write
from run_native_boundary_study import partial_loss,PartialDataset,eligible
from evaluate_mine_study import metrics
from mine_data_protocol import assert_development_cases,verify_assets

DATA=ROOT/"experiments/native_boundary_20261002"
OLD=ROOT/"experiments/core_ablation_20261001"
CASE=ROOT/"datasets/mine_benchmark_20261002/development/Amyntaio"
OUT=ROOT/"experiments/native_period_20261002"
MODEL_OUT=ROOT/"models_prior/native_period_20261002"
ARMS=("identity_context","real_pair")
FUSION="early_pair"


def period_model():
    model=smp.UnetPlusPlus(encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEVICE)
    state=torch.load(CURRENT,map_location=DEVICE,weights_only=True)
    key="encoder.conv1.weight"
    if state[key].shape[1]!=4:
        raise ValueError("Unexpected initial input stem")
    state[key]=(torch.cat([state[key],torch.zeros_like(state[key])],dim=1) if FUSION=="post_change"
                else torch.cat([state[key]*.5,state[key]*.5],dim=1))
    model.load_state_dict(state)
    return model


class SingleViewContext(torch.nn.Module):
    """Explicit identity-context fallback for source single-date inputs."""
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,x):
        return self.model(torch.cat([x,torch.zeros_like(x) if FUSION=="post_change" else x],1))


def observed_pair(split):
    metadata=json.loads((DATA/"development_data.json").read_text(encoding="utf-8"))
    with rasterio.open(CASE/"pre_BGRN_10m.tif") as src:
        before=src.read().astype(np.float32)
    with rasterio.open(CASE/"post_BGRN_10m.tif") as src:
        after=src.read().astype(np.float32)
    windows=metadata[split+"_windows"]
    pre=np.stack([before[:,w["box"][1]:w["box"][3],w["box"][0]:w["box"][2]] for w in windows])
    post=np.stack([after[:,w["box"][1]:w["box"][3],w["box"][0]:w["box"][2]] for w in windows])
    if FUSION=="post_change":
        # Keep exactly the existing post-only normalization. Additional changes
        # have a shared pre/post range, not independent delta minmax amplification.
        low=np.minimum(pre.min((2,3),keepdims=True),post.min((2,3),keepdims=True))
        high=np.maximum(pre.max((2,3),keepdims=True),post.max((2,3),keepdims=True))
        change=np.clip(np.abs(post-pre)/(high-low+1e-6)*255,0,255)
        return np.concatenate([norm_patch(post),change],1).astype(np.float32)
    return np.concatenate([norm_patch(pre),norm_patch(post)],1)


def native_validation(model,pair,y,indices,arm):
    results={}; model.eval()
    x=np.array(pair[indices],copy=True)
    if arm=="identity_context":
        if FUSION=="post_change":
            x[:,4:]=0
        else:
            x[:,:4]=x[:,4:]
    for name,rgb in (("measured_NIR",False),("RGB_proxy",True)):
        view=x.copy()
        if rgb:
            view[:,3]=view[:,2]; view[:,7]=view[:,6]
        with torch.no_grad():
            prob=torch.sigmoid(model(torch.as_tensor(view,device=DEVICE)))[:,0].cpu().numpy()
        results[name]=metrics(prob>=.5,y[indices]==1,y[indices]!=255)
    return results


def readout(model,pair,my,vx,vy,indices,arm):
    source=validation(SingleViewContext(model),vx,vy)
    mine=native_validation(model,pair,my,indices,arm)
    h=2*source["source_NIR"]["F1"]*source["source_RGB_proxy"]["F1"]/max(source["source_NIR"]["F1"]+source["source_RGB_proxy"]["F1"],1e-9)
    return {"source":source,"native_spatial_validation":mine,
        "selection_score":.4*h+.6*mine["RGB_proxy"]["F1"]-.1*source["generic_background"]["block_activation"]}


def prepare():
    assert_development_cases(["Amyntaio"]); verify_assets()
    OUT.mkdir(parents=True,exist_ok=True)
    alignment=json.loads((CASE/"period_alignment.json").read_text(encoding="utf-8"))
    if not alignment["raw_pair_gate_pass"]:
        raise ValueError("Real period imagery not aligned sufficiently")
    path=OUT/"training_protocol.json"
    assets={"pre":sha(CASE/"pre_BGRN_10m.tif"),"post":sha(CASE/"post_BGRN_10m.tif"),
            "data":sha(DATA/"development_data.json"),"alignment":sha(CASE/"period_alignment.json"),"initial":sha(CURRENT)}
    if path.exists():
        plan=json.loads(path.read_text(encoding="utf-8"))
        if plan["assets"]!=assets:
            raise ValueError("Period study inputs changed")
        return plan
    prior=json.loads((DATA/"training_protocol.json").read_text(encoding="utf-8"))
    plan={"registered_at":datetime.now().isoformat(timespec="seconds"),"arms":list(ARMS),"seeds":[42,43,44],
        "epochs":15,"updates_per_epoch":120,"batch":32,"distillation_weight":.5,
        "fusion":FUSION,
        "architecture":"8-channel UNet++; post-preserving initial weights + zero change branch" if FUSION=="post_change" else "same 8-channel UNet++ ResNet34, first stem initialized half/half from existing 4-channel model",
        "only_paired_information_difference":"real prior-date Amyntaio image vs duplicated post image as prior context",
        "dates":{"pre":"2017-06-01","post":"2017-06-28","event":"2017-06-10"},
        "source_replay_context":"post duplicated, explicit identity control for single-date source data; not additional observed dates",
        "target_semantics":"post-event debris reference segmentation assisted by prior context, not pure new-change labels or pre-event warning",
        "no_pre_negative_label_assumption":True,"optimizer":"same SGD .0002/.9/.0001, exponential .96",
        "change_semantics":"normalized absolute radiometric difference features; not physical displacement or spectral-index measurements",
        "RGB_policy":"both post NIR and change NIR replaced by their red counterparts, no genuine-NIR leak in RGB proxy",
        "initial_function":"zero change stem preserves current post-only model for any observed pair" if FUSION=="post_change" else "identity contexts preserve current single-date function at initialization",
        "sampling":"same 12source positive/12background +4mine positive/4weak negative; geometric augmentation matched",
        "loss":"confidence-weighted partial BCE/Dice + .5 fixed-teacher Bernoulli KL on common source only",
        "native_validation_indices":prior["native_validation_indices"],"selection":prior["selection"],
        "eligible_gate":prior["eligible_gate"],"thresholds":prior["thresholds"],
        "assets":assets,"new_reserved_mines_read_for_prediction":False,
        "limit":"one native development mine and publication-derived reference, not independent mine testing"}
    write(path,plan)
    return plan


def train(arm,seed,plan):
    result=OUT/f"train_{arm}_seed{seed}.json"
    folder=MODEL_OUT/f"{arm}_seed{seed}"; folder.mkdir(parents=True,exist_ok=True)
    digest=sha(OUT/"training_protocol.json")
    checkpoint=folder/"best_model.pth"; resume=folder/"resume.pth"
    if result.exists():
        previous=json.loads(result.read_text(encoding="utf-8"))
        if previous["protocol_sha256"]!=digest:
            raise ValueError("Existing period run protocol differs")
        if previous["completed_epochs"]==plan["epochs"]:
            print(f"Reusing {arm}/{seed}",flush=True); return
        if not resume.exists():
            raise ValueError("Partial run cannot resume")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    sx=np.concatenate([np.load(OLD/"common_x.npy"),np.load(OLD/"ours_extra.npy")])
    sy=np.concatenate([np.load(OLD/"common_y.npy"),np.zeros((2000,32,32),np.float32)])
    sp,sn=np.flatnonzero(sy.mean((1,2))>=.1),np.flatnonzero(sy.mean((1,2))==0)
    source=DS(sx,sy,train=True,seed=seed)
    mx=observed_pair("train"); vxmine=observed_pair("val")
    if arm=="identity_context":
        if FUSION=="post_change":
            mx[:,4:]=0
        else:
            mx[:,:4]=mx[:,4:]
    my,mw=np.load(DATA/"mine_train_y.npy"),np.load(DATA/"mine_train_weight.npy")
    replay=PartialDataset(mx,my,mw,seed+200)
    data=json.loads((DATA/"development_data.json").read_text(encoding="utf-8"))
    mp=np.array([i for i,w in enumerate(data["train_windows"]) if w["kind"]=="positive"])
    mn=np.array([i for i,w in enumerate(data["train_windows"]) if w["kind"]=="negative"])
    vsx,vsy=np.load(OLD/"val_x.npy"),np.load(OLD/"val_y.npy")
    vmy=np.load(DATA/"mine_val_y.npy")
    model=period_model(); teacher=model_at(CURRENT).eval().requires_grad_(False)
    initial=json.loads((DATA/"initial_validation.json").read_text(encoding="utf-8"))
    optimizer=torch.optim.SGD(model.parameters(),lr=.0002,momentum=.9,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.96)
    history=[]; best=-float("inf"); best_epoch=0; best_eligible=False; begin_epoch=1
    if result.exists():
        state=torch.load(resume,map_location=DEVICE,weights_only=False)
        if state["protocol_sha256"]!=digest:
            raise ValueError("Resume protocol changed")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        history,best,best_epoch,best_eligible=state["history"],state["best"],state["best_epoch"],state["best_eligible"]
        source.rng.bit_generator.state=state["source_rng"]; replay.rng.bit_generator.state=state["replay_rng"]
        begin_epoch=state["epoch"]+1
    start=time.time()
    for epoch in range(begin_epoch,plan["epochs"]+1):
        rng=np.random.default_rng(seed*10000+epoch); rrng=np.random.default_rng(seed*10000+epoch+1000000)
        model.train(); losses=[]
        for _ in range(120):
            ids=np.concatenate([rng.choice(sp,12,True),rng.choice(sn,12,True)])
            common=[source[int(i)] for i in ids]
            x4=torch.stack([p[0] for p in common]).to(DEVICE)
            pairs=[(torch.cat([p[0],torch.zeros_like(p[0]) if FUSION=="post_change" else p[0]]),p[1],torch.ones((32,32))) for p in common]
            for pool in (mp,mn):
                pairs.extend(replay.get(int(i)) for i in rrng.choice(pool,4,True))
            x=torch.stack([p[0] for p in pairs]).to(DEVICE)
            y=torch.stack([p[1] for p in pairs]).to(DEVICE).unsqueeze(1)
            w=torch.stack([p[2] for p in pairs]).to(DEVICE).unsqueeze(1)
            rgb=x.clone(); rgb[:,3]=rgb[:,2]; rgb[:,7]=rgb[:,6]
            logits=model(torch.cat([x,rgb]))
            loss=partial_loss(logits,torch.cat([y,y]),torch.cat([w,w]))
            x4rgb=torch.cat([x4[:,:3],x4[:,2:3]],1)
            with torch.no_grad():
                tl=teacher(torch.cat([x4,x4rgb])); tp=torch.sigmoid(tl)
                entropy=torch.nn.functional.binary_cross_entropy_with_logits(tl,tp,reduction="none")
            sl=torch.cat([logits[:24],logits[32:56]])
            kl=(torch.nn.functional.binary_cross_entropy_with_logits(sl,tp,reduction="none")-entropy).mean()
            loss=loss+.5*kl
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss))
        scheduler.step()
        value=readout(model,vxmine,vmy,vsx,vsy,plan["native_validation_indices"],arm)
        passes=eligible(value,initial); score=value["selection_score"]
        if (passes and not best_eligible) or (passes==best_eligible and score>best):
            best,best_epoch,best_eligible=score,epoch,passes; torch.save(model.state_dict(),checkpoint)
        history.append({"epoch":epoch,"loss":float(np.mean(losses)),"validation":value,"eligible":passes,"elapsed_seconds":round(time.time()-start,2)})
        record={"arm":arm,"seed":seed,"completed_epochs":epoch,"protocol_sha256":digest,"weights":str(checkpoint),
                "best_epoch":best_epoch,"best_eligible":best_eligible,"best_score":best,"best_validation":history[best_epoch-1]["validation"],"history":history}
        write(result,record)
        torch.save({"epoch":epoch,"protocol_sha256":digest,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
            "history":history,"best":best,"best_epoch":best_epoch,"best_eligible":best_eligible,
            "source_rng":source.rng.bit_generator.state,"replay_rng":replay.rng.bit_generator.state},resume)
        if epoch==1 or epoch%5==0:
            n=value["native_spatial_validation"]["RGB_proxy"]; bg=value["source"]["generic_background"]["block_activation"]
            print(f"{arm}/{seed} {epoch}/15: RGB F1={n['F1']:.4f}, recall={n['recall']:.3f}, generic={bg:.3f}, eligible={passes}",flush=True)
    record["weights_sha256"]=sha(checkpoint); write(result,record)
    del model,teacher,optimizer; torch.cuda.empty_cache()


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--fusion",choices=["early_pair","post_change"],default="early_pair")
    args=parser.parse_args()
    if args.fusion=="post_change":
        FUSION=args.fusion
        OUT=ROOT/"experiments/native_change_residual_20261002"
        MODEL_OUT=ROOT/"models_prior/native_change_residual_20261002"
    plan=prepare()
    if not args.prepare_only:
        for seed in args.seeds:
            for arm in args.arms:
                train(arm,seed,plan)
