# -*- coding: utf-8 -*-
"""
detector.py — 矿安天眼 · LMHLD 滑坡检测推理模块
=================================================
模型: UnetPlusPlus (encoder=resnet34, 无预训练), 4 通道 [B,G,R,NIR], 二分类
与 train.py 完全一致的模型结构, 加载已训练权重进行推理。

功能:
  1. predict_batch()  — 对 (N,4,P,P) 数据块批量推理 (数据集 npy 场景)
  2. detect_image()   — 对任意大影像 (H,W,4) 滑窗推理, 拼接整幅掩膜
  3. analyze_mask()   — 连通域统计 + 四级风险分级 (蓝/黄/橙/红)
  4. overlay()        — 掩膜叠加到 RGB 影像生成标注图
  5. to_rgb()         — LMHLD 通道序 [B,G,R,NIR] → 可视化 RGB

使用 (在 D:\\LMHLD 目录, lmhld conda 环境):
    python -c "from detector import LandslideDetector; d=LandslideDetector('best_model_Emilia-Romagna_Italy_32.pth')"
"""
import os
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
os.environ["SM_FRAMEWORK"] = "torch"
os.environ["SMP_DISABLE_VERSION_CHECK"] = "1"

import numpy as np
import torch
import segmentation_models_pytorch as smp
import cv2

# 多粒度几何引擎 (纯 numpy+cv2, 不引入新模型权重)
from geo_geometry import (mask_to_regions, MAX_REGIONS_LITE,
                          LITE_MAX_VERTICES, DEFAULT_SIMPLIFY_EPS)

NUM_CHANNELS = 4    # B,G,R,NIR (LMHLD 官方通道顺序)
PATCH_SIZE = 32     # 训练 patch 尺寸
NUM_CLASSES = 1     # 二分类

# 四级预警: 风险区像素占比阈值 (MVP 规则, 后续可替换为多智能体评估)
RISK_RULES = [
    (0.15, "red"),    # ≥15% 红色-极高风险
    (0.05, "orange"), # ≥5%  橙色-高风险
    (0.01, "yellow"), # ≥1%  黄色-中风险
]
RISK_NAMES = {
    "blue":   ("蓝色-未检出明显异常", "#1E6FD9"),
    "yellow": ("黄色-少量疑似异常", "#E8B830"),
    "orange": ("橙色-较多疑似异常", "#E06A1E"),
    "red":    ("红色-大量疑似异常", "#D92B2B"),
    "unknown": ("待复核-证据冲突", "#737B87"),
}


class ModelNumericalError(ValueError):
    """Inference cannot produce a valid scientific conclusion from invalid logits."""


class ModelComponentError(RuntimeError):
    """A requested inference component failed; substitution is not authorized."""


class LandslideDetector:
    """LMHLD 滑坡检测推理器"""

    def __init__(self, weights_path, device=None, threshold=0.5, patch_size=PATCH_SIZE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=NUM_CHANNELS,
            classes=NUM_CLASSES,
        )
        state = torch.load(weights_path, weights_only=True, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.threshold = threshold
        self.patch_size = patch_size
        print(f"[detector] 权重加载成功: {os.path.basename(weights_path)} | device={self.device}")

    # ── 核心推理 ──────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(self, patches):
        """
        批量推理 32×32 数据块
        patches: (N,4,P,P) float32 (原始 DN 值, 与训练数据分布一致)
        返回: (N,P,P) float32 滑坡概率 [0,1]
        """
        if not isinstance(patches, torch.Tensor):
            patches = torch.as_tensor(np.ascontiguousarray(patches), dtype=torch.float32, device=self.device)
        else:
            patches = patches.to(self.device).float()
        logits = self.model(patches)
        if not torch.isfinite(logits).all():
            raise ModelNumericalError('检测模型输出非有限数，已停止推理，不能生成筛查等级')
        probabilities=torch.sigmoid(logits).squeeze(1)
        if not torch.isfinite(probabilities).all():
            raise ModelNumericalError('检测模型输出非有限概率，已停止推理，不能生成筛查等级')
        return probabilities.cpu().numpy()

    @torch.no_grad()
    def detect_image(self, image_hwc, stride=None, patch_norm=None,
                     nir_synth_model=None, max_batch=None):
        """
        对整幅影像流式滑窗推理，逐批归一化、合成、预测与回填。

        max_batch: 每批窗口数上限，None=256；上限512，同时限制CPU和GPU窗口内存。
                   2304² 图按 32px 滑窗有 5184 个窗口, 一次性推理的显存峰值在
                   8GB 卡上会 OOM(实测 torch.cat 单次要 972MB)。
                   模型在 eval() 下 BN 用 running stats → **逐样本独立**, 故分块与
                   整批**结果逐位相同**, 只是削掉峰值。见 tests 的等价性验证。
        image_hwc: (H,W,4) float32 [B,G,R,NIR]
        patch_norm: None=不归一化(原始DN域模型);
                    'minmax'=逐图块 min-max→[0,255] (与上传模型训练协议一致);
                    'zscore'=逐图块标准化(均值127.5/标准差70, 实测选择性更佳)
        nir_synth_model: 可选的 RGB→NIR 合成器(第十五组)。给定时, 在**逐图块归一化之后**
                    用合成的 NIR 替换第 4 通道 —— 合成器正是在归一化域训练的, 二者协议一致。
                    默认 None = 沿用 NIR:=R(复制 R), 即现有部署行为不变。
                    实测: 真实滑坡场景分离度 +0.06(base)/+0.02(realfit);
                          未见矿山场景 base 误报下降、realfit 略升(见第十五组报告)。
        返回: (H,W) float32 滑坡概率图 (边缘做了 padding, 与原图同尺寸)
        """
        image_hwc = np.asarray(image_hwc, dtype=np.float32)
        if image_hwc.ndim != 3 or image_hwc.shape[2] != NUM_CHANNELS or min(image_hwc.shape[:2]) < 2:
            raise ValueError('检测输入须为至少2×2的真实BGRN或显式RGB代理数组')
        if not np.isfinite(image_hwc).all():
            raise ModelNumericalError('检测输入含非有限数')
        H, W, C = image_hwc.shape
        ps = self.patch_size
        stride = ps if stride is None else stride
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or not 1 <= stride <= ps:
            raise ValueError('滑窗步长须为1到patch_size之间的整数，不能产生未覆盖像素')
        batch_size = 256 if max_batch is None else max_batch
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or not 1 <= batch_size <= 512:
            raise ValueError('max_batch须为1到512之间的整数')
        if patch_norm not in (None, 'global', 'minmax', 'zscore'):
            raise ValueError('未知影像归一化协议')

        # 边缘 padding 到 patch 整数倍 (reflect 减少边界效应)
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        img = np.pad(image_hwc, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        # 整图全局归一化 (场景适配模型 'baige' 使用的协议)
        if patch_norm == "global":
            img = img.copy()
            for c in range(img.shape[2]):
                ch = img[..., c]
                lo, hi = ch.min(), ch.max()
                img[..., c] = (ch - lo) / (hi - lo + 1e-6) * 255.0
            patch_norm = None   # 已归一化, 后续不再处理

        Hp, Wp = img.shape[:2]
        ys = list(range(0, Hp - ps + 1, stride))
        xs = list(range(0, Wp - ps + 1, stride))
        if ys[-1] != Hp - ps:
            ys.append(Hp - ps)
        if xs[-1] != Wp - ps:
            xs.append(Wp - ps)
        # Only this batch's windows/probabilities exist at once, including NIR synthesis.
        prob_map = np.zeros((Hp, Wp), dtype=np.float32)
        cnt_map = np.zeros((Hp, Wp), dtype=np.float32)
        from itertools import islice, product
        positions = iter(product(ys, xs))
        while True:
            batch_positions = list(islice(positions, batch_size))
            if not batch_positions:
                break
            patches = np.stack([img[y:y+ps, x:x+ps].transpose(2, 0, 1)
                                for y, x in batch_positions])
            if patch_norm == 'minmax':
                mn = patches.min(axis=(2, 3), keepdims=True)
                mx = patches.max(axis=(2, 3), keepdims=True)
                patches = (patches-mn)/(mx-mn+1e-6)*255.0
            elif patch_norm == 'zscore':
                mu = patches.mean(axis=(2, 3), keepdims=True)
                sd = patches.std(axis=(2, 3), keepdims=True)
                patches = (patches-mu)/(sd+1e-6)*70.0+127.5
            if nir_synth_model is not None:
                try:
                    with torch.no_grad():
                        rgb01 = torch.as_tensor(np.ascontiguousarray(patches[:, :3]/255.0),
                                                dtype=torch.float32, device=self.device)
                        synthetic = nir_synth_model(rgb01)
                        if synthetic.shape != (len(patches), 1, ps, ps) or not torch.isfinite(synthetic).all():
                            raise ModelNumericalError('NIR合成器输出无效，未执行替代推理')
                        patches[:, 3] = synthetic.clamp(0, 1).cpu().numpy()[:, 0]*255.0
                        del rgb01, synthetic
                except (ModelNumericalError, torch.cuda.OutOfMemoryError):
                    raise
                except Exception as exc:
                    raise ModelComponentError('所选NIR合成器执行失败，未改用红波段') from exc
            probs = self.predict_batch(patches)
            if probs.shape != (len(patches), ps, ps) or not np.isfinite(probs).all() or np.any((probs < 0)|(probs > 1)):
                raise ModelNumericalError('检测模型概率形状或数值无效')
            for (y, x), probability in zip(batch_positions, probs):
                prob_map[y:y+ps, x:x+ps] += probability
                cnt_map[y:y+ps, x:x+ps] += 1.0
        if np.any(cnt_map == 0):
            raise ModelNumericalError('滑窗覆盖不完整，未生成检测结论')
        prob_map /= cnt_map
        return prob_map[:H, :W]

    # ── 统计分析 ──────────────────────────────────────────
    def analyze_mask(self, mask, min_region_area=10, with_regions=True,
                     max_regions=None, simplify_eps=None):
        """
        连通域统计 + 四级风险分级 + 多粒度实例几何 (对单个 2D 掩膜)
        mask: (H,W) 0/1
        min_region_area: 小于该面积的连通域视为噪点丢弃 (默认10px)
        with_regions: 是否计算实例几何 (关掉可省 ~ms 级开销, 供应急回退)
        返回原有 6 键 (landslide_pixels/area_ratio/num_regions/max_region_area/
        region_areas/risk_level) + 新增几何键:
          · regions               : 实例几何列表 (区域级bbox+对象级多边形+形态量)
          · num_regions_total     : 实例总数 (口径显式, 不受 region_areas 前10截断影响)
          · region_areas_truncated: region_areas 是否被截断
          · geometry_meta         : 顶点上限/简化容差等口径信息

        注: num_regions 由 len(region_areas) 改为真实计数 —— 原实现因
        region_areas 只保留前 10 而使 num_regions 被静默截断在 10
        (实测影响面 1.0% 的图块, 最大 17 处); 改正后与 geo_tools.compute_region_stats
        口径一致。region_areas 本身仍保持前 10, 故老消费者字节输出不变。
        """
        mask = (mask > 0).astype(np.uint8)
        n_regions_total = 0
        geom = None

        if mask.sum() > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            # 最小区域过滤: 仅保留 >= min_region_area 的连通域
            keep = np.where(areas >= min_region_area)[0] + 1
            if len(keep) == 0:
                landslide_px = 0
                region_areas = []
                max_region = 0
            else:
                keep_labels = set(keep.tolist())
                mask_clean = np.isin(labels, list(keep_labels))
                landslide_px = int(mask_clean.sum())
                areas = areas[keep - 1]
                # 真实实例数 (未截断) —— 排序后同时供 region_areas 与几何使用
                areas_sorted = sorted([int(a) for a in areas], reverse=True)
                n_regions_total = len(areas_sorted)
                region_areas = areas_sorted[:10]
                max_region = int(areas.max())
                if with_regions:
                    # 多粒度几何: 掩膜 -> 区域级bbox / 对象级多边形+形态量
                    # (在已过滤的 mask_clean 上做, min_area=1 避免二次过滤逻辑漂移)
                    geom = mask_to_regions(
                        mask_clean, min_area=1,
                        max_regions=(max_regions if max_regions is not None
                                     else MAX_REGIONS_LITE),
                        simplify_eps=(simplify_eps if simplify_eps is not None
                                      else DEFAULT_SIMPLIFY_EPS),
                        max_vertices=LITE_MAX_VERTICES)
        else:
            landslide_px = 0
            region_areas = []
            max_region = 0

        ratio = landslide_px / mask.size
        out = {
            "landslide_pixels": landslide_px,
            "area_ratio": round(float(ratio), 4),
            "num_regions": n_regions_total,
            "max_region_area": max_region,
            "region_areas": region_areas,
            "risk_level": self._risk_level(ratio),
            "num_regions_total": n_regions_total,
            "region_areas_truncated": bool(n_regions_total > 10),
            "regions": (geom["regions"] if geom else []),
            "geometry_meta": {
                "enabled": bool(with_regions and geom is not None),
                "min_region_area": int(min_region_area),
                "simplify_eps": float(simplify_eps if simplify_eps is not None
                                      else DEFAULT_SIMPLIFY_EPS),
                "num_regions_returned": (geom["num_regions_returned"] if geom else 0),
                "truncated": (geom["truncated"] if geom else False),
            },
        }
        return out

    def _risk_level(self, ratio):
        for thr, level in RISK_RULES:
            if ratio >= thr:
                return level
        return "blue"

    # ── 可视化 ──────────────────────────────────────────
    def overlay(self, rgb, mask, color=(255, 0, 0), alpha=0.55,
                outline=False, outline_color=(255, 255, 255), thickness=1):
        """红色半透明标注叠加到 RGB 影像

        outline=False (默认) 时输出与历史版本逐字节一致 —— 已有基线/前端不受影响;
        置 True 时在填充之上再描实例外轮廓, 便于分辨相邻滑坡体。
        """
        overlay = rgb.astype(np.float32).copy()
        m = mask > 0
        overlay[m] = overlay[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
        out = np.clip(overlay, 0, 255).astype(np.uint8)
        if outline and m.any():
            cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2 用 BGR 序, 故按 RGB 反序传入
            cv2.drawContours(out, cnts, -1,
                             tuple(int(c) for c in outline_color[::-1]),
                             thickness, cv2.LINE_8)
        return out

    @staticmethod
    def to_rgb(bgrnir):
        """(...,4)[B,G,R,NIR] → (...,3) RGB, 逐通道 min-max 归一化 (可视化用)"""
        rgb = bgrnir[..., [2, 1, 0]].astype(np.float32).copy()
        for i in range(3):
            c = rgb[..., i]
            lo, hi = float(c.min()), float(c.max())
            rgb[..., i] = (c - lo) / (hi - lo + 1e-6)
        return (rgb * 255).astype(np.uint8)


# ── 便捷入口: 数据集 npy 批量检测 ──────────────────────
def demo_on_npy(region_dir, weights_path, split="test", max_samples=100,
                save_fig=None, show_stats=True):
    """
    对数据集 npy 批量检测并出效果图 (演示/视频素材用)
    region_dir: 区域目录 (含 *_images.npy / *_labels.npy)
    返回: (detector, results列表, 图片grid)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体 (Windows)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    det = LandslideDetector(weights_path)
    images = np.load(os.path.join(region_dir, f"{split}_images.npy"))
    labels = np.load(os.path.join(region_dir, f"{split}_labels.npy"))
    if max_samples:
        images, labels = images[:max_samples], labels[:max_samples]

    N = images.shape[0]
    probs = det.predict_batch(images)                       # (N,32,32)
    masks = (probs >= det.threshold).astype(np.uint8)
    gt = (labels[:, 0] > 0).astype(np.uint8)                # (N,32,32)

    # 逐样本统计
    results = []
    for i in range(N):
        r = det.analyze_mask(masks[i])
        # 与真实标签对比的 IoU / F1
        inter = np.logical_and(masks[i], gt[i]).sum()
        union = np.logical_or(masks[i], gt[i]).sum()
        r["iou"] = round(float(inter) / (union + 1e-6), 4)
        r["label_ratio"] = round(float(gt[i].sum() / gt[i].size), 4)
        results.append(r)

    # 展示统计 (逐样本判断是否含滑坡)
    if show_stats:
        pos = gt.sum(axis=(1, 2)) > 0
        print(f"[demo] {os.path.basename(region_dir)} {split}集 前{max_samples}样本: "
              f"含滑坡 {int(pos.sum())}/{N}")
        risks = {}
        for r in results:
            risks[r["risk_level"]] = risks.get(r["risk_level"], 0) + 1
        print(f"[demo] 预警分级分布: {risks}")

    # 出图: 每行 4 个样本, 列=RGB/GT/预测掩膜/红色标注
    n = min(N, 8)
    if save_fig:
        fig, axes = plt.subplots(n // 4 + (1 if n % 4 else 0), 4,
                                 figsize=(16, 4 * (n // 4 + (1 if n % 4 else 0))))
        axes = np.atleast_2d(axes)
        for i in range(n):
            ax = axes[i // 4, i % 4]
            rgb = det.to_rgb(images[i].transpose(1, 2, 0))
            overlay = det.overlay(rgb, masks[i])
            ax.imshow(overlay)
            level = RISK_NAMES[results[i]["risk_level"]][0]
            ax.set_title(f"#{i} {level} IoU={results[i]['iou']:.2f}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"[demo] 效果图已保存: {save_fig}")
        plt.close(fig)

    return det, results


if __name__ == "__main__":
    import sys
    # 用法: python detector.py <weights.pth> <region_dir> [保存图片路径]
    weights = sys.argv[1] if len(sys.argv) > 1 else "best_model_Emilia-Romagna_Italy_32.pth"
    region = sys.argv[2] if len(sys.argv) > 2 else r"F:\LMHLD\LMHLD_dataset_different_patch_sizes\Emilia-Romagna_Italy_32"
    out_png = sys.argv[3] if len(sys.argv) > 3 else "demo_detection.png"
    demo_on_npy(region, weights, save_fig=out_png)
