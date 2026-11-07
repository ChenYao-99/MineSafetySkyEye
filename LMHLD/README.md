# 矿安天眼 · 滑坡分割与遥感量测

这个目录是矿安天眼的影像检测模块。它读取遥感影像，在像素上标出可能需要复核的滑坡区域，再把相邻的候选像素合成对象，生成叠加图、边界、面积等结果。网页和桌面端通过 [GeoMMAgent 工作台](../GeoMMAgent/README.md) 调用这里的服务。

`LMHLD` 也是训练与评测所用的公开滑坡数据集名称。此目录包含团队基于该数据集和矿区影像开展的分割训练、场景适配、推理和评测代码。可用能力取决于输入条件和对应的模型。疑似区用于影像筛查与人工复核，跨矿识别效果仍需独立验证。

## 影像是怎样处理的

1. **检查输入。** 读取 RGB 或配置明确的光学四波段影像，检查监测范围、无效像元、波段、空间尺度和影像是否适合比较。
2. **生成候选。** 将大幅影像分块送入分割模型，拼回概率结果，再按模型配置生成候选掩膜。
3. **输出可核查的量测。** 将掩膜转成候选对象，有可靠空间尺度时计算米制面积等指标，并提供可导出的几何结果。
4. **比较两期影像。** 核查两期的空间匹配关系，在共同有效观测范围内统计候选区域变化，供工作台生成复核报告。

现役服务有不同的影像和模型配置。训练中使用的真实近红外波段、RGB 输入及实验性合成代理分别记录。光谱指数需符合真实波段和辐射条件。

## 目录怎么读

| 入口 | 负责的工作 |
| --- | --- |
| [detector_service.py](detector_service.py) | 提供检测、影像检查、两期对比和矢量导出 API，负责调度推理与量测 |
| [detector.py](detector.py) | 加载模型，执行滑窗推理与概率拼接 |
| [raster_upload.py](raster_upload.py) 与 [input_quality.py](input_quality.py) | 读取上传影像，处理波段、有效像元和观测质量 |
| [geo_geometry.py](geo_geometry.py) 与 [geo_tools.py](geo_tools.py) | 计算候选对象几何、空间量测和适用的定量指标 |
| [model_registry.json](model_registry.json) | 记录模型路径、文件校验值和推理参数 |
| [knowledge_base.py](knowledge_base.py) 与 [knowledge_sources.json](knowledge_sources.json) | 为工作台提供知识检索和来源信息 |
| [train.py](train.py)、[train_prior.py](train_prior.py) 与 [train_unified.py](train_unified.py) | 基线训练、先验加权训练和跨地区统一训练 |
| [experiments](experiments/)、[validation](validation/) 与 [tests](tests/) | 实验记录、验证工具和回归检查 |

## 本地使用

只想浏览代码时，可以从 [detector_service.py](detector_service.py) 的上传检测和两期对比接口入手。实际运行请按[团队统一安装](../GeoMMAgent/deployment/DEPLOYMENT.md#团队统一安装)建立检测环境。部署验收采用 Python 3.10.20，安装脚本按 [GPU 部署锁定清单](../GeoMMAgent/deployment/requirements.lmhld.cuda124.lock.txt)固定 `torch==2.5.1+cu124` 和其他依赖版本。[requirements.txt](requirements.txt) 是检测模块的依赖参考。

| 资源或设置 | 如何准备 |
| --- | --- |
| 模型权重 | 默认在本目录的 `models_prior/` 查找。也可以用 `KAT_MODELS_ROOT` 指向外部目录。运行所需模型以登记文件为准，矿区上传模型的登记键为 `upload:mine` |
| 保留案例索引 | 按部署说明配置矿区参考和保留集隔离索引，使系统可以识别应隔离的评估输入 |
| 区域评测和训练数据 | 取得所需数据后，按 [data_paths.py](data_paths.py) 配置 `KAT_DATA_ROOT` 或 `KAT_TILED_ROOT`。历史训练脚本也可能有各自的命令参数或本机默认路径 |
| 数据、底图与引用 | 从相应来源获取。出处和使用范围见 [ATTRIBUTION.md](ATTRIBUTION.md) |

激活检测侧 Python 环境后，在本目录运行下面的命令可以单独启动检测服务。正常使用整套系统时，由工作台启动器负责启动或复用它。

```powershell
python detector_service.py 5101
```

| 接口 | 用途 |
| --- | --- |
| `GET /api/health` | 查看服务状态 |
| `POST /api/inspect_upload` | 核查影像输入及可用信息 |
| `POST /api/detect_upload` | 单期影像检测 |
| `POST /api/compare` | 两期候选变化对比 |
| `POST /api/geometry_export` | 导出量测和矢量结果 |

服务启动后可访问 [本地健康接口](http://127.0.0.1:5101/api/health)。模型按任务加载，完成一次实际检测还需要与输入匹配的权重和测试影像。

## 实验从哪里看

| 想了解的问题 | 对应记录 |
| --- | --- |
| 多尺度融合怎样影响像素精度和小区域召回 | [三尺度融合实验](experiments/fusion/三尺度融合实验报告.md) |
| 怎样同时检查对象误报与漏检 | [实例级误差归因](experiments/instance_diag/第十七组实验报告_实例级误差归因与概率过滤.md) |
| RGB 与近红外输入变化怎样影响真实影像 | [真实影像域差异](experiments/domain_gap/第十三组实验报告_真实影像泛化的成因与改进.md)与[评估协议校正](experiments/nir_train/第十六组实验报告_合成NIR训练期注入与评估协议校正.md) |
| 多尺度提高小对象检出后，怎样减少背景候选 | [冻结面积筛选验收](experiments/component_acceptance_20261020/README.md) |
| 原生三通道模型的开发进展 | [七来源 RGB 候选](experiments/rgb_final_all_regions_20261020/README.md)，当前属于研究候选 |
| 两期影像可见范围不同，怎样避免边缘假变化 | [2026-11-06 共同观测范围验证](../GeoMMAgent/maintenance/observation_context_20261106/README.md) |

例如，冻结面积筛选在同地区、同事件的 682 张新增图块上减少了约 21.8% 的纯背景候选，保持了该批样本的小对象检出计数，像素 F1 略降。这个实验用于衡量候选筛选的取舍，结果适用范围见记录中的样本划分和固定阈值。

复现实验时，应使用对应的代码版本、数据划分和检查点。运行配置、研究候选和历史比较结果分别以 [模型登记文件](model_registry.json) 与各实验协议为准。
