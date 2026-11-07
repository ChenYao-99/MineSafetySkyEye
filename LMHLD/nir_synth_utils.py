# -*- coding: utf-8 -*-
"""
nir_synth_utils.py — RGB→NIR 合成器的载入与推理（无 Flask 依赖）
=================================================================
从 eval_nir_synth.py 抽出，供 finetune_negatives.py 的**训练期注入**复用。

为什么单独成模块：eval_nir_synth.py 第 37 行有
`from detector_service import UPLOAD_MODELS`，会把 Flask/服务端依赖链拉进来。
训练脚本不应依赖服务端 —— 故把这两个纯函数抽到此处，两边共用一份实现，
避免出现第三份复制（仓内已有三份内联 F1 的教训）。

协议（不可动摇，见第十五组报告）：合成器在**逐图块 min-max → [0,1]** 域训练，
故调用方传入的 patches 必须已是**逐图块 min-max → [0,255]** 域。
"""
import os
import sys

import numpy as np
import torch
import segmentation_models_pytorch as smp

sys.stdout.reconfigure(encoding="utf-8")

MODELS = r"D:\LMHLD\models_prior"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_synth(tag="nir_synth", seed=42, dev=None):
    """载入合成器权重并置 eval 模式。tag/seed 决定目录 {tag}_seed{seed}。"""
    dev = dev or DEV
    m = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                 in_channels=3, classes=1).to(dev)
    p = os.path.join(MODELS, f"{tag}_seed{seed}", "best_model.pth")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    m.load_state_dict(torch.load(p, weights_only=True, map_location=dev))
    m.eval()
    return m


def synth_nir(model, patches_255, dev=None):
    """patches_255: (N,4,P,P) 已逐图块 min-max 到 [0,255]。
    返回合成 NIR (N,P,P) 同域 [0,255]，可直接替换第 4 通道。

    只用前 3 通道（RGB）作输入，第 4 通道被丢弃 —— 故传入真 NIR 或 NIR:=R
    都不影响结果，可在推理链路中无条件调用。
    """
    dev = dev or DEV
    with torch.no_grad():
        rgb01 = torch.as_tensor(np.ascontiguousarray(patches_255[:, :3] / 255.0),
                                dtype=torch.float32, device=dev)
        out = []
        for i in range(0, len(rgb01), 256):
            out.append(model(rgb01[i:i + 256]).cpu().numpy())
    return np.clip(np.concatenate(out), 0, 1)[:, 0] * 255.0
