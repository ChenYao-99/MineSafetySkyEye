# 第三方来源与致谢 / Third-Party Attribution

> 本文件用于标注本项目所引用的**他人技术、算法、数据**。
> 竞赛规则明确要求："引用他人技术、算法或数据，需在参赛材料中明确标注"，
> 且"版权不清晰"属取消资格情形。**提交材料前请核对本清单是否完整。**

---

## 一、数据集与底图来源

### 1.1 LMHLD 滑坡数据集（本项目检测模型的训练与测试数据）

| | |
|---|---|
| **全称** | LMHLD: A Large-Scale Multisource High-Resolution Landslide Dataset for Landslide Detection Based on Deep Learning |
| **作者** | Guanting Liu, Yi Wang, Xi Chen, Baoyu Du, Penglei Li, Yuan Wu, Zhice Fang, Peifeng Ma |
| **发表** | IEEE Transactions on Geoscience and Remote Sensing (TGRS), Vol. 63, 2025 |
| **DOI** | 10.1109/TGRS.2025.3619062 |
| **数据集获取** | 开放获取：https://doi.org/10.5281/zenodo.11424987 |
| **规模** | 25 365 个图像图块、32 296 个滑坡实例；5 种卫星传感器（含 UAV），覆盖 7 个研究区 |
| **许可证** | 论文未明确写明，**以 Zenodo 页面标注为准**（引用前请到该页面确认） |

**本项目对 LMHLD 的使用**

- 早期四波段 B/G/R/NIR 实验使用 Emilia-Romagna、Gorkha、Hokkaido、Jiuzhaigou、Wenchuan 和 Taiwan 六个地区。Rio de Janeiro 的原始数据是五通道，未进入这套四波段协议。
- 后续[七来源原生 RGB 实验](experiments/rgb_final_all_regions_20261020/README.md)将 Rio de Janeiro 的 RGB 输入纳入训练与来源选择。该实验的模型属于研究候选，现役模型见 [model_registry.json](model_registry.json)。
- 早期实验沿用原数据的区域图块尺寸和 BLDS 指标，并在部分训练中使用自行切分的 `LMHLD_tiled`。各组具体的数据划分、波段和指标以对应实验协议为准。

数据从原发布页获取，使用和引用时核对对应版本的许可。

### 1.2 Sentinel-2 L2A（多光谱数据源，第十四组接入）

| | |
|---|---|
| **数据来源** | Sentinel-2 L2A（ESA Copernicus），经 **AWS Earth Search**（Element 84 镜像）获取 |
| **获取方式** | STAC 检索 + COG 分块读取（`fetch_sentinel2.py`），无需注册 |
| **波段** | B02/B03/B04/**B08(NIR)**，10 m 分辨率 |
| **许可证** | Copernicus Sentinel Data License —— **需署名 ESA/Copernicus** |

第十四组验证了多光谱数据接入，并发现该案例在 10 m 影像上的目标分离度较弱。后续版本支持合规的光学四波段 GeoTIFF 输入，具体检测效果随场景、空间分辨率和模型变化。

### 1.3 Esri World Imagery（真实场景验证底图）

真实场景验证所用的卫星影像由 `fetch_mine_imagery.py` 从 Esri World Imagery 服务拉取
（白格滑坡、抚顺西露天矿）。使用时应遵守 Esri 的服务条款并标注来源。
相关元数据（含分辨率与经纬度）保存在 `D:\竞赛材料\真实场景验证\影像\*_meta.json`。

---

## 二、本团队实现与实验

| 内容 | 位置 |
|---|---|
| 滑坡检测推理模块与多尺度评估 | `detector.py`、`detector_service.py` |
| 多粒度几何引擎（掩膜→区域/对象/像素三粒度） | `geo_geometry.py` |
| 定量分析工具集（NDVI/区域统计/趋势变点/两期对比） | `geo_tools.py` |
| 安全知识检索与来源登记 | `knowledge_base.py` |
| 训练脚本与实验（先验加权损失、小样本适配、统一模型） | `train_prior.py`、`train_unified.py`、`train_upload_adapt.py` |
| 评估脚本与实验记录 | `eval_*.py`、`experiments/` |

---

## 三、使用的开源软件与在线服务

| 组件 | 用途 | 许可/说明 |
|---|---|---|
| PyTorch（检测侧 2.5.1） | 深度学习框架 | BSD-3-Clause |
| segmentation-models-pytorch | UNet++ 模型 | MIT |
| albumentations | 数据增强 | MIT |
| OpenCV | 图像处理 | Apache 2.0 |
| NumPy / SciPy | 数值计算 | BSD |
| Flask / flask-cors | 服务接口 | BSD / MIT |
| CAMEL (camel-ai) | 多智能体框架 | Apache 2.0 |
| **GeoMMAgent / GeoMMBench** | 多智能体框架与基准（上游项目） | **Apache 2.0**，详见 [GeoMMAgent NOTICE](../GeoMMAgent/NOTICE) |
| Qwen API（阿里云百炼） | 大模型推理服务 | 商业服务，遵守其服务条款 |
| Esri World Imagery | 真实场景验证影像 | 见第一节 |

---

## 四、学术参考工作

本项目的评估协议与方法论参考了以下论文（**已在对应实验报告中标注**）：

| 论文 | 本项目中的应用 | 标注位置 |
|---|---|---|
| Earth-Agent (ICLR 2026) | 多智能体推理轨迹双层评估协议 | 《多智能体推理轨迹双层评估报告》 |
| ThinkGeo (2026) | 工具智能体失败模式分类（指导稳定性加固） | 同上（附录） |
| RemoteReasoner (AAAI 2026) | 决策级任务变换与三粒度评估协议 | 《第十组》《第十一组》实验报告 |

> 以上均为**方法参考**，非代码/数据复制；各实验报告已在正文中注明来源与口径差异。

---

*来源说明复核日期为 2026-11-07。下列增补保留当时的数据范围。*


## 2026-10-03原始矿区与复核材料增补

- 三座开发矿原始两期Sentinel-2 L2A、SCL及新井煤矿历史开发资料升级，经Microsoft Planetary Computer读取；保留Copernicus/ESA来源、原始日期、波段、坐标和哈希。
- DLR/GFZ Landslide Reference Data DOI10.5281/zenodo.17007637，保存的发布元数据注明CC BY4.0；只取得38项所需资产，事故类型分开。未使用其矿区训练权重。
- Amyntaio：FIG/JISDM2019 Vassilakis等研究Figure1；LasCruces：López-Vinielles等论文DOI10.5281/zenodo.4704955；Hpakant：Lin等DOI10.1016/j.isprsjprs.2021.05.015。参考为作者图示的项目地理化派生，不是原始专家GIS或新专家认证；图示及底层影像权利分别保留。
- 新井定位/事故参考：Chang等Remote Sensing2025,17,19，DOI10.3390/rs17010019，作者GF-2参考2023-02-24与项目Sentinel-2 post02-23不同日。新原始影像对尚未预测、边界未接纳，但9/30第18组已在同一事件做检测；因此只作开发回归，不增加新盲测数。
- Unet++/ResNet、配准、BCE/Dice、已知支持归一化、源域教师、模态丢弃、概率融合、RAG/多智能体均为已有组件；团队贡献限定于场景需求、输入/标签纪律、匹配研究和事实量测/报告流程。
- 复核工具不展示预测或作者答案；自动浏览器测试不属于专家反馈。所有新数据/权重/缓存保存D盘。源码审阅包不含论文原文、IEEE机构水印、凭据、运行库或浏览器缓存；不是已发布提交包。
