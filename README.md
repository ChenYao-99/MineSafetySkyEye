# 矿安天眼 · MineSafetySkyEye

主项目在 `GeoMMAgent/`，负责工作台、任务调度和复核报告。`LMHLD/` 提供它调用的影像检测服务和实验代码。

矿安天眼是一个面向露天矿遥感影像的筛查与复核原型。上传一张影像，可以在原图上查看模型标出的疑似区域和量测结果。上传同一矿区的两期影像，可以在共同有效观测范围内检查候选变化。工作台把结果、来源和复核意见整理成案例记录。

| 想了解什么 | 从哪里开始 |
| --- | --- |
| 页面、报告、知识检索、归档和复核 | [GeoMMAgent 工作台与报告服务](GeoMMAgent/README.md) |
| 滑坡分割、影像质量、候选量测与两期对比 | [LMHLD 滑坡分割与遥感量测](LMHLD/README.md) |
| 模型训练、对照实验与研究候选 | [检测侧实验索引](LMHLD/README.md#实验从哪里看) |
| 环境、资产路径与部署检查 | [部署说明](GeoMMAgent/deployment/DEPLOYMENT.md) |

系统的用途是历史案例复盘、影像疑似区筛查和人工复核。在未见矿区的稳定识别仍待验证。

## 本地查看和运行

```powershell
git clone https://github.com/ChenYao-99/MineSafetySkyEye.git
cd MineSafetySkyEye
```

只看代码或分析实验时，先读上表对应的 README 和实验记录。当前的完整运行方式面向 Windows，默认以 D 盘作为项目与运行缓存位置。工作台服务使用端口 5100，检测服务使用端口 5101，两端各有自己的 Python 环境。

1. 按[部署说明](GeoMMAgent/deployment/DEPLOYMENT.md)准备依赖。检测环境的现有部署清单使用 PyTorch 的 CUDA 12.4 构建。
2. 按[检测模块说明](LMHLD/README.md#本地使用)配置权重和输入索引。权重通过团队共享渠道或相应来源获取，使用 [model_registry.json](LMHLD/model_registry.json) 中的路径和校验值核对。保留案例索引和评估资产按团队的隔离规则配置。
3. 从 `GeoMMAgent` 目录启动工作台，使用[启动示例](GeoMMAgent/README.md#在自己的电脑上运行)指定旁边的 `LMHLD` 目录和本机解释器。先使用 `-CheckOnly` 检查路径，再使用 `-LocalFactsOnly` 验证本地检测与量测流程。
4. 从 [本地工作台](http://127.0.0.1:5100/) 打开页面，选择有权限使用的测试影像。在线报告模式的模型配置和本地凭据模板见 [GeoMMAgent README](GeoMMAgent/README.md)。

公开源码与运行资产分别管理。训练数据、原始影像、权重、案例库、评估保留集、账户凭据和大型实验中间产物通过各自的授权渠道配置。部分历史脚本依赖当时的本机路径和冻结资产，复现时以对应实验的协议为准。源目录中的小型实验结果用于阅读和对照。

## 修改与验证

- 页面与复核流程从 `GeoMMAgent` 的前端、后端和测试入手。
- 模型、输入协议、地理量测和评测从 `LMHLD` 对应模块入手。
- 提交改动时说明修改场景、验证范围和所需资源。可以 Fork 后用 Pull Request 提交改进。

## 来源与许可

智能体模块基于 GeoMMAgent / GeoMMBench，保留上游的 [LICENSE](GeoMMAgent/LICENSE) 和 [NOTICE](GeoMMAgent/NOTICE)。检测侧的数据、工具和引用出处见 [ATTRIBUTION.md](LMHLD/ATTRIBUTION.md)。数据、影像、模型权重和第三方组件分别遵循各自来源的使用条件。
