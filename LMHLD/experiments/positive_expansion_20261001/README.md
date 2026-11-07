# 公开矿山正样本补充

本目录全部位于D盘。数据清单及冻结SHA-256见`positive_cases.json`，下载记录见`sources/acquisition_manifest.json`。

| 资产 | 来源 | 使用与限制 |
|---|---|---|
| Çöllolar 200×4×32×32弱正图块 | UNOSAT官方报告v2，WorldView-2公开图示 | 一个矿区的保守失稳内部；不是200次独立事件；仅训练，权重0.25；不把其余像素赋背景 |
| Hpakant 2020-08-03 RGB与参考 | Lin等2021论文Figure 7D | 只数字化作者蓝色虚线坍塌源区；其他运动范围保留未知；剔除标注与边界不确定性窗口 |

Çöllolar使用PDF独立图像对象，文字标注未进入输入；32px图块不跨条带接缝。提取图示没有原始地理坐标和原始产品分辨率，不按0.5m原产品计量。

Hpakant主评估包含23个前景、238个背景完整窗口；不评估参考边界精度。首次用于`rgb_balanced_20261001`完成后的外部评估；随后`rgb_guarded_20261001`只作已见测试回归，不能再称首次盲测。原Las Cruces与NASA Bingham从未加入训练；白格仍为早期已见弱正训练场景。

Hpakant背景仅表示该事件图示范围外的空间参考，未全面审核矿区的其他历史失稳；该参考FPR不能解释为矿区所有滑坡的现场误报率。

标签为公开图示或保守弱标注，未获独立专家审查。UNOSAT明确报告未现场独立核验；DigitalGlobe/Planet底层影像版权保留，未主张原始图示可按开放数据自由再分发。

来源：[UNOSAT事故页](https://unosat.org/products/2117)、[原报告PDF](https://unosat.org/static/unosat_filesystem/2117/UNOSAT_TUK_LS2011-Report_v2_HR.pdf)、[Hpakant论文](https://doi.org/10.1016/j.isprsjprs.2021.05.015)、[作者Figure 7](https://ars.els-cdn.com/content/image/1-s2.0-S0924271621001489-gr7_lrg.jpg)。

本轮结果：[完整报告](../rgb_guarded_20261001/正样本扩充与RGB适配结果.md)。
