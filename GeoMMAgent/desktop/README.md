# 矿安天眼桌面工作台

Windows本机原型，采用pywebview 6.2.1和系统已有WebView2。独立窗口打开原有本地页面，
再加载桌面专用的`workbench.css`和`workbench.js`，重排同一组控件。
原网页源文件、后端、旧启动器及算法实现保持原样；浏览器不会加载桌面布局。

## 工作台布局

- 左侧按任务导航：影像检测、变化分析、智能体问答、案例归档、安全知识库、模型评测。
- 影像检测以大幅视图为中心，可切换原图/候选叠加，直接拖动圈定监测范围。
- 右侧按“分析设置 / 检测结果 / 复核报告”分组。高级尺度与输入条件折叠保留，
  检测后突出“生成复核报告”，重新检测作为次要操作。
- 变化分析分为两期影像、变化结果与报告面板；案例库常驻，记录/报告/人工复核以页签切换。
- 顶部操作指引说明完整流程；文件、尺度、服务状态与当前归档案例均有独立位置。
- 字体、间距、按钮和空态统一，1000×720与1440×900工作区均完成布局检查。

## 在团队安装环境下启动

先完成[团队统一安装](../deployment/DEPLOYMENT.md#团队统一安装)和[运行资产配置](../deployment/DEPLOYMENT.md#配置运行资产并启动)。桌面窗口使用 WebView2，并在 `GeoMMAgent/cache/desktop_runtime` 下安装独立的窗口依赖。以下命令从 `GeoMMAgent` 目录执行：

```powershell
$runtimeRoot = 'D:\MineSafetySkyEye-runtime'
$agentPython = Join-Path $runtimeRoot 'GeoMMA\Scripts\python.exe'
$detectorPython = Join-Path $runtimeRoot 'lmhld\Scripts\python.exe'
$detectorDirectory = (Resolve-Path -LiteralPath '..\LMHLD').Path
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\desktop\install.ps1 -BasePython $agentPython
$desktopPython = '.\cache\desktop_runtime\Scripts\pythonw.exe'
& $desktopPython -B .\desktop\app.py --agent-python $agentPython `
    --detector-python $detectorPython --detector-dir $detectorDirectory --local-facts-only
```

已有原项目默认目录和解释器的电脑可双击 `启动矿安天眼桌面版.bat`。克隆后的并列目录使用上面的显式路径参数。桌面端复用同一套 5100 和 5101 服务，导出到当前工作台目录下的 `data/desktop_exports`，同名文件自动编号。

关闭桌面窗口后，网页版仍可在共享服务运行时使用。停止整套服务时，使用 `.\stop_demo.ps1 -DetectorDirectory $detectorDirectory`。启动日志位于 `cache/desktop/` 和 `cache/launcher/`。在线报告与网页使用同一套模型配置。

## 验收入口

- `tests/test_desktop_host.py`：服务来源校验、启动不等待后台管道EOF、导出不覆盖、参数契约。
- `tests/browser_desktop_smoke.mjs`：真实WebView2文件导入、GPU检测、双期、本地报告及导出字节核对。
- `tests/desktop_native.ps1`：限定本项目窗口的截图或正常关闭，不操作其他应用。
- `cache/desktop_acceptance_20261018`：本机验收数据与精简回执；使用独立案例库和已见开发影像，
  不计作专业用户试用、模型精度或新矿区盲测，验收不进行云端生成调用。

2026-10-18初版记录：桌面真实GeoTIFF/GPU单期、双期、本地报告及3项JSON/HTML导出通过；
关闭桌面后，独立Edge浏览器再次完成真实影像导入和GPU检测。正式入口冷启动通过，
默认不开放调试端口；重复打开保持同一桌面进程。后端134项中133通过、1项可选用例跳过，
另有32项前端运行、13项异步、6项栅格及复核交接检查通过。
精简验收回执见 `maintenance/desktop_prototype_20261018/acceptance.json`。

WebView2下载适配使用固定版本pywebview的原生控件接口，升级该依赖前应重跑导出验收。

2026-10-19工作台验收另见`maintenance/desktop_workbench_20261019/acceptance.json`。
覆盖真实单期/双期GPU分析、缩放画布的鼠标ROI、切换范围后的旧结果失效、导出、
隔离库复核意见保存、知识检索、问答图片输入、六项导航与操作指引。原生窗口可见性
另行核对；自动化内部复核不作为外部用户、矿方或专家验证。
本轮后端135项中134通过、1项可选用例跳过；原网页32项运行、13项异步、6项栅格与交接检查通过。
