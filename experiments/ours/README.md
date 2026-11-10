# Ours / PAC

以下四个代码目录及其 runs/ 中的结果均已实际移动到本目录。

| 目录 | 内容 |
|---|---|
| [yolov5-advpad-20260908](yolov5-advpad-20260908/) | 本次指定：YOLOv5s AdvPatch 30.40% 及对应 PAC-P/R/T 三种子实验 |
| [yolov5-master](yolov5-master/) | 最初 PAC 代码与原始实验 |
| [yolov5-fixed-20260907](yolov5-fixed-20260907/) | PAC 修正版及 squash / letterbox 比较 |
| [yolov5-advreal-recolor-20260910](yolov5-advreal-recolor-20260910/) | AdvReal 换色扩展与完整评估矩阵 |

## YOLOv5s PAC-P / R / T

代码、补丁和原始评估位于上表第一个实际目录；s000 的 AdvPatch 对应当前统一评估 mAP50=30.40%。
同次实验保留 s000 / s001 / s002 三个种子，下表链接展示 s000，其他种子在同级目录。

| 系列 | 版本 1 | 版本 2 |
|---|---|---|
| P 系列 | [P1](yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_p1/) | [P2](yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_p2/) |
| R 系列 | [R1](yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_r1/) | [R2](yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_r2/) |
| T 系列 | [T1](yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_t1/) | [T2](yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_t2/) |

[全部补丁](yolov5-advpad-20260908/runs/official_pad3_20260908/patches/) · [原始评估](yolov5-advpad-20260908/runs/official_pad3_20260908/eval/) · [训练与实验记录](yolov5-advpad-20260908/runs/official_pad3_20260908/)

原始 eval/ 保留当时 val.py 的结果；30.40% / 38.13% / 33.92% 是后续统一评估的 AdvPatch / P1 / P2 指标，
[统一评估结果](yolov5-advreal-recolor-20260910/runs/advreal_recolor_20260910/eval/yolov5s/) 保持在对应评估实验中。

## 其他 PAC 配对结果

- [AdvPatch / CAP 与 PAC](../baselines/advpatch_cap_completion_20260911/report.md)
- [T-SEA / NAP 与 PAC](../baselines/tsea_nap_20260912/report.md)
- [PNAP 与 PAC 最终比较](../baselines/pnap_yolo_20261102/final_comparison_20261108/)
- [补丁图与画廊](../analysis/pac_patch_visualizations_20260910/)

配对结果在所属实验目录中保存一份，通过文档链接访问。
