# 当前工程精简与验收记录

## 2026-10-16 再次精简

按本次明确授权，永久删除1,484个冗余文件（主清单1,478个、收尾一次性脚本6个），逻辑体积约3.521 GiB；执行时D盘可用空间增加约3.524 GiB。
删除范围包括40个独立材料/汇总入口、制作build目录中的排版脚本及依赖、旧方案草稿和渲染预览、
淘汰实验的派生数组/预测缓存、未启用的单事件微调候选。完整最终实验结论、来源和协议仍保留。

4个无用目录联接仅以非递归方式移除，真实LMHLD数据/模型以及共享PPT依赖目标均完整。
此前把联接目标计成副本的盘点已纠正；没有将其目标大小计入本次释放量。

全部14份运行权重及额外CLI权重、原始数据、保护测试登记、生产案例库、当前前端与运行报告模块、
最新可编辑Word/PPT、视频脚本、矢量图和必要实际检测图片保留。生成材料的脚本已删，后续材料可直接编辑成品。

清理后：后端127项中126通过/1可选跳过；检测侧86项通过；前端32运行、13异步、6栅格及复核交接检查通过。
真实GPU验证单期、双期、RGB适配、可选NIR合成、源域评测与北海道多尺度融合，保留与演示一致的量测结果。
3,109个保护文件核验通过，其中1,231项哈希与1,878项原始数据元信息核对；预期文档链接调整另行记录。
本轮没有训练、下载、外部生成调用或新增保留矿区预测。

当前清单为 `delete_manifest_20261016.json.gz`，执行与保护验真为 `cleanup_execution_20261016.json`、
`preservation_verification_20261016.json`，测试日志为 `backend_tests_20261016.log`、`detector_tests_20261016.log`。
这些是清理/可用性记录，不是识别性能或现场验收证据。以下保留上一次已完成清理的历史说明。

已按确认清单删除117116个文件，释放约25.4 GiB。清理范围为淘汰模型和训练检查点、旧部署与SAR环境、重复验收副本、无用中间数组，以及155个未采用方案的代码和相关测试文件。

## 当前入口

正常使用仍从 `D:\GeoMMAgent\启动矿安天眼演示.bat` 启动，打开 `http://127.0.0.1:5100/`。检测侧为 `D:\LMHLD`，两套当前Conda环境保持完整。5198真实任务评估入口及其任务库也保留。

## 保留依据

默认模型、所有界面选项需要的权重、RGB联合适配依赖、NIR可选组件、六区评测和北海道融合权重均保留。models_prior中保留14份权重及适配配置，根目录另保留独立detector.py CLI使用的意大利基座权重。默认模型、阈值与输入协议没有调整。

原始数据、保留集索引、知识库46片段与15来源、当前案例库、关键实验原始结果、最新PPT和视频脚本、Las Cruces原始输入与离线回看资料保持完整。旧实验的结果和来源记录保留，已淘汰权重不再支持直接重放。

## 清理后验收

- 后端83项回归中82项通过、1项可选真实检测fixture未启用；真实GPU业务链另行验证。
- 检测侧86项回归通过，前端运行、异步、栅格和JSON与HTML交接检查通过。
- 真实GPU单期、双期、RGB适配、NIR可选组件、源域评测与多尺度融合可运行。
- 单期8.71%，双期8.84%与8.89%、净变化0.05个百分点，与录制素材保持一致。
- 单期与双期报告、定位补记、原始事实保持、JSON与HTML导出、知识检索和归档完整性通过。
- 核心运行源码、全部14份运行权重及额外CLI权重的指纹与清理前相同。

额外修正为前端镜像同步，以及更新过期的前端导出测试断言并补核对归档HTML导出。规范前端和业务算法没有更换。参赛PPT在维护期间有保存版本变化，文件不在删除清单中，保留其当前版本；Word临时锁文件不作为源文档校验。

## 保留权重

|用途|保留文件|
|---|---|
|upload:base|models_prior/Emilia-Romagna_Italy_32_upload_seed42/best_model.pth|
|upload:baige|models_prior/baige_adapted_ps32/best_model.pth|
|upload:unified|models_prior/unified_pre_seed42/best_model.pth|
|upload:realfit|models_prior/real_ft_v1_seed42/best_model.pth|
|rgb-component:current|models_prior/real_ft_T1b_ours_seed42/best_model.pth|
|rgb-component:adapter|models_prior/rgb_balanced_20261001/rgb_joint_seed42/best_model.pth|
|region:Emilia-Romagna_Italy_32|models_prior/Emilia-Romagna_Italy_32_prior1.0_seed42_pre_dice/best_model.pth|
|region:Gorkha_Nepal_32|models_prior/Gorkha_Nepal_32_prior1.0_seed42/best_model.pth|
|region:Hokkaido_Japan_128|models_prior/Hokkaido_Japan_128_prior1.0_seed42/best_model.pth|
|region:Jiuzhaigou_China_128|models_prior/Jiuzhaigou_China_128_prior0.5_seed42_pre_dice/best_model.pth|
|region:Wenchuan_China_224|models_prior/Wenchuan_China_224_prior1.0_seed42_pre_dice/best_model.pth|
|region:Taiwan_China_224|models_prior/Taiwan_China_224_prior1.0_seed42/best_model.pth|
|component:fusion32|models_prior/Hokkaido_Japan_32_prior1.0_seed42/best_model.pth|
|component:nir_synth|models_prior/nir_synth_seed42/best_model.pth|
|standalone-detector-CLI|best_model_Emilia-Romagna_Italy_32.pth|

清理清单与原盘点压缩为JSON GZip，旧模型的少量训练元数据保存在retired_model_metadata.json。运行原始回执和清理后功能验收结果位于同目录。此验收说明工程可用性，不构成新的矿区识别准确率或独立事故验证。
