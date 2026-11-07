# 七来源原生RGB工程候选：已完成

2026-10-20完成两个种子、每个4800步的三通道训练，来源阈值及8个矿区结果重算通过，两个真实GPU分批推理重放通过。真实模型CPU烟雾测试也已通过。这是全来源工程候选，尚未登记为默认模型。

## 设置与数据角色

- Unet++/ResNet34，真正3通道BGR，随机初始化，无预训练权重、NIR或合成NIR。根据此前开发对照选择普通父图训练＋五次TTA；不自动采用未过门槛的尺度扰动。
- 七来源训练可用池17753个父图，来源验证2210个父图；对5072张原官方val图像做BGR精确查重，无需要剔除的重复。batch21且每来源3个父图，每模型100800次父图采样，比此前batch20实验多5%，不是纯数据量因果对照。
- Rio de Janeiro的478张train在本阶段纳入来源；其136张val在本阶段用于来源选择/校准。上一阶段的新地区验证已经冻结，新阶段不再声称Rio独立或跨地区测试。Rio使用RapidEye标准前三BGR映射，NPY本身缺乏语义通道标签，映射假设仍需保留。
- 来源选择半集按宏平均native TTA5 F1@.5选检查点；另一半校准统一像素阈值，使七来源平均背景FPR≤2%。来源均值约束不保证新矿区误报上限。
- 每模型共4800步，检查点1200/2400/3600/4800。该预算与当前曲线不构成已证明训练饱和或性能极限的结论。实际被梯度访问的独立样本比例见training.json，并非可用池每张都保证访问。

## 入选候选

| 种子 | 来源选中步数 | 来源选择F1@.5 | 来源校准阈值 | 来源平均背景FPR | 实际独立训练父图覆盖 |
|---|---:|---:|---:|---:|---:|
|42|4800|69.32%|0.709|1.99%|97.26%|
|43|4800|69.07%|0.692|2.00%|97.12%|

以上是来源域验证，不是新的跨地区分数。三地区配对研究的原生RGB TTA5宏平均F1 60.32%及相关对照见../rgb_native_20261020/README.md。

## 四个历史开发矿区

只评分部分事故正例核心、部分背景及独立复核正常区；未知255不当负例，NoData评分排除，禁止正例/正常掩膜冲突。两个候选使用各自来源阈值，生产对照为原默认的固定.4。新模型用native32父图、0..1、64输入和TTA5，旧默认用patch32和0..255，因此属于完整管线回归，不能归因为单个算法因素。

| 矿区 | 核心覆盖：默认 → 候选42 / 43 | 正常激活：默认 → 候选42 / 43 |
|---|---:|---:|
|Amyntaio|15.78% → 27.92% / 25.41%|6.08% → 13.39% / 14.79%|
|LasCruces|5.13% → 13.71% / 3.40%|18.95% → 19.92% / 16.95%|
|Hpakant|20.66% → 24.18% / 22.42%|1.74% → 0.87% / 0.94%|
|CN_XJ2023|0.00% → 23.06% / 0.76%|0.46% → 5.03% / 5.03%|

| 种子 | 平均核心覆盖变化/百分点 | 平均正常激活变化/百分点 | 开发门槛 |
|---|---:|---:|---|
|42|+11.82|+3.00|未通过|
|43|+2.60|+2.62|未通过|

两个种子均未通过正常作业控制门槛，默认模型、服务和前端保持不变；也未根据矿区分数挑选某个种子作生产权重。平均提升不代表逐矿稳定、完整事故边界、真实用户收益或现场预警能力。

## 独立推理接口与必要文件

- D:/LMHLD/rgb_native_detector.py：独立三通道CPU/GPU候选推理，不修改原服务选择器。输入必须为有限BGR HWC3；实际NoData需调用方先显式处理。RGBA或测量NIR四通道输入会拒绝。native32父图归一化0..1、缩放64、TTA5，流式拼接保持边缘覆盖。
- seed42与seed43目录各保留一份source_selected.pth、candidate.json、source_curve.json、training.json及必要校准/开发预测缓存；不保留其他epoch权重或已完成优化器。约0.54GiB训练缓存已自动清理。
- protocol.json/data_manifest.json保存设置、来源角色和精确样本；exposure_ledger.json另记开发矿区曝光；results.json/verification.json/state.json保存最终状态，inference_smoke.json只证明CPU接口运行，不代表准确率。

运行示例（已有D:/Miniconda3/envs/lmhld环境，BGR图像）：

```python
import sys
sys.path.insert(0, r"D:\LMHLD")
import torch, cv2
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
from rgb_native_detector import load_candidate
detector = load_candidate(r"D:\LMHLD\experimentsgb_final_all_regions_20261020\seed42\candidate.json")
image_bgr = cv2.imread(r"D:\your_image.png", cv2.IMREAD_COLOR)
probability = detector.detect_image(image_bgr, max_batch=64)
mask = probability.astype("float64") >= detector.threshold
```

示例种子仅展示接口，不表示推荐该种子上线。原始训练/校准入口为D:/LMHLD/validation/rgb_final_training.py；已完成状态应先查看现有verification.json和冻结缓存，重新训练应使用新的实验目录。

官方test与矿区保留集未预测，无外部模型调用。两个新模型和全部临时缓存均在D盘，原始数据只读。
