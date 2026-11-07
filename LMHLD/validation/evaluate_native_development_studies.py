# -*- coding: utf-8 -*-
"""Development regressions after frozen native training; reserved mines never predicted."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME",str(ROOT/"cache/torch"))
os.environ["SMP_DISABLE_VERSION_CHECK"]="1"
import hashlib
import json
import sys
from datetime import datetime
import cv2
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from finetune_negatives import DATA_ROOT,NEG_NPY,norm_patch
from evaluate_mine_study import metrics,predict_patches,prepare_background_holdout
from run_rgb_joint_study import sha,write,CURRENT
from mine_data_protocol import verify_assets

NATIVE=ROOT/"experiments/native_boundary_20261002"
GUARD=ROOT/"experiments/native_source_guard_20261002/lambda0.5"
PERIOD=ROOT/"experiments/native_period_20261002"
CHANGE=ROOT/"experiments/native_change_residual_20261002"
OUT=ROOT/"experiments/native_development_20261002/evaluation"
AMY=ROOT/"datasets/mine_benchmark_20261002/development/Amyntaio"


class PairDetector(LandslideDetector):
    def __init__(self,weights,fusion="early_pair"):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.patch_size,self.threshold=32,.5
        self.model=smp.UnetPlusPlus(encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(self.device)
        self.model.load_state_dict(torch.load(weights,map_location=self.device,weights_only=True))
        self.model.eval()
        self.fusion=fusion

    def detect_image(self,image_hwc,**kwargs):
        if self.fusion!="post_change":
            return super().detect_image(image_hwc,**kwargs)
        if image_hwc.shape[2]!=8 or kwargs.get("stride") not in (None,32):
            raise ValueError("Post/change diagnostic expects raw observed pre/post and fixed32 stride")
        h,w=image_hwc.shape[:2]; ps=32
        padded=np.pad(image_hwc,((0,(-h)%ps),(0,(-w)%ps),(0,0)),mode="reflect")
        coords=[(y,x) for y in range(0,padded.shape[0],ps) for x in range(0,padded.shape[1],ps)]
        pre=np.stack([padded[y:y+ps,x:x+ps,:4].transpose(2,0,1) for y,x in coords])
        post=np.stack([padded[y:y+ps,x:x+ps,4:].transpose(2,0,1) for y,x in coords])
        low=np.minimum(pre.min((2,3),keepdims=True),post.min((2,3),keepdims=True))
        high=np.maximum(pre.max((2,3),keepdims=True),post.max((2,3),keepdims=True))
        difference=np.clip(np.abs(post-pre)/(high-low+1e-6)*255,0,255)
        features=np.concatenate([norm_patch(post),difference],1).astype(np.float32)
        prob=predict_patches(self,features,batch=kwargs.get("max_batch",128))
        result=np.zeros(padded.shape[:2],np.float32)
        for (y,x),patch in zip(coords,prob):
            result[y:y+ps,x:x+ps]=patch
        return result[:h,:w]


def records():
    groups=[(NATIVE,"native",3),(GUARD,"source_guard",1),(PERIOD,"period",2),(CHANGE,"post_change",2)]
    rows=[]
    for folder,study,n_arms in groups:
        plan=json.loads((folder/"training_protocol.json").read_text(encoding="utf-8"))
        paths=[folder/f"train_{arm}_seed{seed}.json" for arm in plan["arms"] for seed in plan["seeds"]]
        if len(paths)!=n_arms*3:
            raise ValueError("Unexpected registered experiment family")
        for path in paths:
            if not path.exists():
                raise ValueError("All declared development runs must finish before regression")
            row=json.loads(path.read_text(encoding="utf-8"))
            if row["completed_epochs"]!=plan["epochs"] or row["protocol_sha256"]!=sha(folder/"training_protocol.json"):
                raise ValueError("Incomplete/mismatched development run")
            if sha(Path(row["weights"]))!=row["weights_sha256"]:
                raise ValueError("Development weight changed")
            row={**row,"family":study}
            rows.append(row)
    return rows


def read(path):
    image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image decoding failed")
    return image


def evaluate():
    verify_assets()
    rows=records()
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[NATIVE/"training_protocol.json",GUARD/"training_protocol.json",PERIOD/"training_protocol.json",CHANGE/"training_protocol.json"]
    plan={"registered_at":datetime.now().isoformat(timespec="seconds"),"primary_threshold":.5,"secondary_threshold":.4,
        "protocols":{str(p):sha(p) for p in paths},"regression_role":"old source test, old Las/Hpak figure references, same-mine native annotated subset",
        "no_new_reserved_mine_prediction":True,"no_claim_of_final_blind_evaluation":True,
        "native_whole_labeled_limit":"includes training locations; diagnostic only, not independent accuracy",
        "period_legacy_source_policy":"single-date old/source inputs duplicated to identity context, not true temporal validation"}
    final=OUT/"results.json"; partial=OUT/"results_partial.json"
    if final.exists():
        print("Reusing complete development regressions"); return
    plan_path=OUT/"evaluation_plan.json"
    if plan_path.exists():
        prior=json.loads(plan_path.read_text(encoding="utf-8"))
        if prior["protocols"]!=plan["protocols"]:
            raise ValueError("Regression protocols changed")
        plan=prior
    else:
        write(plan_path,plan)
    # Commit candidate choice using internal validation BEFORE regressions.
    candidates=[r for r in rows if r["best_eligible"]]
    selection={"candidate":None,"criterion":"best 3-seed mean internal score among families/arms with all 3 eligible; seed fixed42"}
    groups={}
    for row in rows:
        groups.setdefault((row["family"],row["arm"]),[]).append(row)
    qualified=[(float(np.mean([r["best_score"] for r in group])),key) for key,group in groups.items() if len(group)==3 and all(r["best_eligible"] for r in group)]
    if qualified:
        score,key=max(qualified)
        selection["candidate"]={"family":key[0],"arm":key[1],"seed":42,"score":score}
    write(OUT/"candidate_before_regression.json",selection)
    region=Path(DATA_ROOT)/"Emilia-Romagna_Italy_32"
    sx=norm_patch(np.load(region/"test_images.npy").astype(np.float32))
    sy=np.load(region/"test_labels.npy")[:,0]>0
    split=json.loads((ROOT/"experiments/nir_train/neg_split_seed42.json").read_text(encoding="utf-8"))
    negatives=prepare_background_holdout(np.load(NEG_NPY),split["holdout_idx"])
    with rasterio.open(AMY/"post_BGRN_10m.tif") as src:
        post=src.read().transpose(1,2,0).astype(np.float32)
    with rasterio.open(AMY/"pre_BGRN_10m.tif") as src:
        before=src.read().transpose(1,2,0).astype(np.float32)
    with rasterio.open(AMY/"partial_labels.tif") as src:
        y=src.read(1)
    valid,target=y!=255,y==1
    las_root=ROOT/"experiments/mining_validation_20261001"
    las=json.loads((las_root/"validation_cases.json").read_text(encoding="utf-8"))["cases"][0]
    hp_root=ROOT/"experiments/positive_expansion_20261001"
    hp=json.loads((hp_root/"positive_cases.json").read_text(encoding="utf-8"))["hpakant"]
    cases=[("las_cruces_201901",las_root,las),("hpakant_20200702",hp_root,hp)]
    work=[{"family":"baseline","arm":"current_mine","seed":42,"weights":str(CURRENT)},
          {"family":"baseline","arm":"source_base","seed":42,"weights":str(ROOT/"models_prior/Emilia-Romagna_Italy_32_upload_seed42/best_model.pth")},*rows]
    results=json.loads(partial.read_text(encoding="utf-8"))["models"] if partial.exists() else []
    for row in work:
        key=f"{row['family']}_{row['arm']}_seed{row['seed']}"
        if any(r["id"]==key for r in results):
            continue
        paired=row["family"] in ("period","post_change")
        change=row["family"]=="post_change"
        detector=PairDetector(row["weights"],fusion="post_change" if change else "early_pair") if paired else LandslideDetector(row["weights"],threshold=.5,patch_size=32)
        def predict_source(images):
            return predict_patches(detector,np.concatenate([images,np.zeros_like(images) if change else images],axis=1) if paired else images)
        source=metrics(predict_source(sx)>=.5,sy)
        rgb=sx.copy(); rgb[:,3]=rgb[:,2]
        rgb_source=metrics(predict_source(rgb)>=.5,sy)
        npred=predict_source(negatives)>=.5
        item={"id":key,"family":row["family"],"arm":row["arm"],"seed":row["seed"],"weights":row["weights"],
            "weights_sha256":sha(Path(row["weights"])),"best_internal_eligible":row.get("best_eligible"),
            "source_test":source,"source_RGB_proxy_test":rgb_source,
            "generic_background_holdout":{"n":len(negatives),"block_activation":float((npred.sum((1,2))>=10).mean()),
                                          "pixel_activation":float(npred.mean()),"limit":"same-pool pseudo-background"},"cases":{}}
        folder=OUT/key; folder.mkdir(exist_ok=True)
        native={}
        for mode in ("measured_NIR","RGB_proxy"):
            p=post.copy(); b=before.copy()
            if mode=="RGB_proxy":
                p[:,:,3]=p[:,:,2]; b[:,:,3]=b[:,:,2]
            if paired:
                first=p if row["arm"]=="identity_context" else b
                image=np.concatenate([first,p],-1)
            else:
                image=p
            probability=detector.detect_image(image,patch_norm="minmax",max_batch=128)
            native[mode]={"primary@0.5":metrics(probability>=.5,target,valid),"secondary@0.4":metrics(probability>=.4,target,valid),
                          "role":"whole annotated subset includes training: diagnostic"}
            np.save(folder/f"amyntaio_{mode}_probability.npy",probability)
            if mode=="RGB_proxy":
                display=post[:,:,:3].copy()
                for c in range(3):
                    lo,hi=np.percentile(display[:,:,c],[2,98]); display[:,:,c]=np.clip((display[:,:,c]-lo)/max(hi-lo,1),0,1)*255
                review=display.astype(np.uint8)
                pred=probability>=.5
                for mask,color in ((pred&target&valid,(40,190,40)),(pred&~target&valid,(0,190,240)),(~pred&target&valid,(40,40,230))):
                    review[mask]=(.45*review[mask]+.55*np.array(color)).astype(np.uint8)
                review[~valid]=(.3*review[~valid]).astype(np.uint8)
                cv2.imencode(".png",review)[1].tofile(str(folder/"amyntaio_RGB_diagnostic.png"))
        item["native_annotated_diagnostic"]=native
        for name,root,case in cases:
            raw=read(root/case["image"])
            four=np.concatenate([raw,raw[:,:,2:3]],-1).astype(np.float32)
            image=np.concatenate([four,four],-1) if paired else four
            prob=detector.detect_image(image,patch_norm="minmax",max_batch=128)
            truth,v=read(root/case["reference"])>0,read(root/case["valid"])>0
            item["cases"][name]={"primary@0.5":metrics(prob>=.5,truth,v),"secondary@0.4":metrics(prob>=.4,truth,v),
                                 "role":"previously observed figure-reference development regression"}
            np.save(folder/f"{name}_probability.npy",prob)
        results.append(item)
        write(partial,{"plan":plan,"models":results})
        print(f"{key}: sourceRGB={rgb_source['F1']:.4f}, generic={item['generic_background_holdout']['block_activation']:.3f}, nativeRGB={native['RGB_proxy']['primary@0.5']['F1']:.4f}, Las={item['cases']['las_cruces_201901']['primary@0.5']['F1']:.4f}, Hpak={item['cases']['hpakant_20200702']['primary@0.5']['F1']:.4f}",flush=True)
        del detector; torch.cuda.empty_cache()
    write(final,{"completed_at":datetime.now().isoformat(timespec="seconds"),"plan":plan,"models":results,"internal_candidate":selection,
                 "production_weights_changed":False,"reserved_final_predictions":0})


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate()
