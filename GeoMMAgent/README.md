# 矿安天眼 · 工作台与报告服务

矿安天眼是一个面向露天矿遥感影像的筛查与复核原型。用户可以上传一张影像，圈定需要检查的范围，查看模型标出的疑似区域。上传同一矿区的两期影像时，工作台还可以对比共同可见范围内的候选变化。

这个目录负责界面、任务调度、报告、案例归档和人工复核。实际分割和几何量测由 [LMHLD 检测模块](../LMHLD/README.md) 完成。

## 从一张影像到一份复核记录

1. 在网页或桌面工作台选择影像和监测范围。
2. 工作台调用检测模块，取回候选区域、面积、变化等结果，并在原图上叠加显示。
3. 报告模块将可用观测范围、模型结果和来源依据组织成复核材料。可选的在线模型负责撰写解释，关键数值由检测与量测结果提供。
4. 用户可以对照原图补充复核意见，并将结果、来源和修改记录保存在同一案例中。

两期结果只在经过空间核查的共同有效范围内比较。工作台提供的是影像疑似区和复核依据，适合历史案例复盘与辅助筛查。真实矿区之间的识别稳定性仍需独立验证。

## 目录怎么读

| 入口 | 负责的工作 |
| --- | --- |
| [前端页面](<矿安天眼_前端演示(后端版).html>) | 上传影像、选择区域、展示结果和操作案例 |
| [backend.py](backend.py) | 提供网页与 API，启动或连接检测服务，组织分析和报告 |
| [coordinator](coordinator/) 与 [configs](configs/) | 配置多智能体角色和调度方式 |
| [report_workflow.py](report_workflow.py) 与 [report_grounding.py](report_grounding.py) | 整理报告流程，检查生成内容与事实字段的对应关系 |
| [case_archive.py](case_archive.py) 与 [review_handoff.py](review_handoff.py) | 保存案例、复核意见及交接材料 |
| [desktop](desktop/) | Windows 桌面窗口与工作台布局，复用网页服务 |
| [deployment](deployment/) 与 [tests](tests/) | 部署清单、诊断和回归检查 |

## 在自己的电脑上运行

新成员按[团队统一安装](deployment/DEPLOYMENT.md#团队统一安装)操作。当前启动脚本面向 Windows，要求两套源码位于 D 盘。工作台默认使用 `127.0.0.1:5100`，检测服务默认使用 `127.0.0.1:5101`，两端使用各自的 Python 环境。检测侧的团队安装基准是 PyTorch 2.5.1 的 CUDA 12.4 构建。

完成[权重、索引和启动配置](deployment/DEPLOYMENT.md#配置运行资产并启动)后，从 [本地工作台](http://127.0.0.1:5100/) 打开影像检测页。首次运行可选 `-LocalFactsOnly`，用本地检测与量测结果生成事实汇总。需要在线报告时，按 [.env_template](.env_template) 与 [模型配置](configs/GeoMMBench.yaml) 配置本地 `.env`。

已有原项目默认目录和解释器的电脑可以双击 [启动矿安天眼演示.bat](启动矿安天眼演示.bat)。合仓后的克隆目录和自定义环境应按统一安装页明确指定路径。需要桌面窗口时，参见[桌面工作台说明](desktop/README.md#在团队安装环境下启动)。

模型、影像和案例库按[检测模块说明](../LMHLD/README.md#本地使用)在本地配置。日志、上传影像和 SQLite 案例库保存在各自电脑上。

## 验证记录

- [部署与实际任务验收](deployment/DEPLOYMENT.md)记录同一台 Windows 机器上重建两套依赖环境后的影像任务、归档和恢复检查。换到其他电脑时，仍需验证依赖、驱动和模型资产。
- [2026-11-04 工程复核](maintenance/review_cleanup_20261104/README.md)记录共同观测范围内的统计修正和变化后处理验证。
- [2026-11-06 观测范围修订](maintenance/observation_context_20261106/README.md)记录缺失观测对单期与两期结果的影响，以及固定条件下的复验。
- [跨矿算法记录](maintenance/phase12_20261015/README.md)介绍已验证的 RGB 适配结果和实际矿区误报、漏检的局限。

## 上游与团队工作

智能体侧以 [GeoMMAgent / GeoMMBench](https://github.com/Shihao-Cheng/GeoMMAgent) 为基础。上游提供面向遥感与地学任务的多智能体框架。本团队在其上实现矿区工作台、检测服务对接、证据与量测注入、报告检查、案例归档及复核流程。代码来源、修改声明和上游 Apache 2.0 许可证见 [NOTICE](NOTICE) 和 [LICENSE](LICENSE)。上游框架的基准、模型和安装方法请查阅其项目主页。
