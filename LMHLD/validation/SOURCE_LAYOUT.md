# 当前代码入口与历史归档

2026-10-16：额外清理已完成的报告/材料生成脚本、旧阶段汇总器和淘汰实验缓存。
运行报告模块、数据保护合同、关键训练/推理/评估实现、被其他实现引用的共享核验函数保留。
当前状态以 `D:/GeoMMAgent/maintenance/phase12_20261015/README.md` 与
`D:/GeoMMAgent/maintenance/slim_project/README.md` 为准；下文旧优先级与归档属于历史记录。

2026-10-05按用户要求清理已完成的一次性生成器、固定案例访问/图面诊断、空样板及根目录调试输出。正式运行、训练评估和数据来源校验实现保留原路径，避免破坏导入、冻结哈希和复现协议。

| 用途 | 入口 |
|---|---|
| 检测服务、原始栅格与几何输出 | `D:/LMHLD/detector_service.py`、`detector.py`、`raster_upload.py`、`geo_geometry.py` |
| 报告服务、前端与事实约束 | `D:/GeoMMAgent/backend.py`、`report_grounding.py`、`矿安天眼_前端演示(后端版).html` |
| 启动与检查 | `D:/GeoMMAgent/启动矿安天眼演示.bat`、`start_backend_5100.bat`、`check_services.py` |
| 权重及输入协议 | `D:/LMHLD/model_registry.json`、`model_registry.py`、`models_prior/` |
| 来源、数据角色和保留测试保护 | `D:/LMHLD/validation/mine_data_protocol.py`、`pixel_reference_contract.py`、`published_inventory_contract.py`、`protected_inputs.py` |
| 后续优先级 | `D:/LMHLD/validation/competition_readiness_assessment_20261005.json`、`validation/STATE.md` |
| 数据及复现实验 | `D:/LMHLD/datasets/`、`experiments/`与保留的`run_*`、`fetch_*`、`evaluate_*`脚本 |

历史脚本的产物已存在，清理的是生成器和一次性诊断入口；保留的最终实现可直接运行。原始数据、标签、协议、实验结果、当前权重、凭据和第三方署名未删除。科学实验脚本中的失败版本若仍被导入或哈希约束，继续保留。

归档与逐文件删除清单位于`D:/LMHLD/cache/source_cleanup_20261005/`。归档文件`retired_sources_and_debug_outputs.zip`保存36个源脚本或调试输出的原始字节，压缩后已校验CRC与SHA256。Python编译缓存可从保留源码重新生成，不列入归档。

查看或恢复历史文件：

```powershell
& 'D:\Miniconda3\envs\lmhld\python.exe' 'D:\LMHLD\validation\restore_retired_source.py' --list
& 'D:\Miniconda3\envs\lmhld\python.exe' 'D:\LMHLD\validation\restore_retired_source.py' --file 'LMHLD/validation/make_normal_focal_runner.py'
```

恢复工具只接受清单中的文件，校验归档与内容哈希，且拒绝覆盖已经修改的现用文件。恢复历史脚本不会自动执行训练、下载或改变数据角色。
