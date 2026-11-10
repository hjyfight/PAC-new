# PNAP 与论文 PAC 方法的同源比较

比较范围固定为上一份报告中完成500轮并完成评估的YOLOv2、YOLOv5s、Faster R-CNN。各方法均取相同3个白盒单元和15个黑盒单元；这是共同子集比较，不是完整6WB/30BB论文主表。

主比较使用训练损失选择的best。PNAP-final另列供观察，不据测试分数切换主结果。PAC-P1/P2来自AdvPatch基底，PAC(T-SEA, Bc1/Bc2)来自T-SEA基底，二者不要混用。

| 方法 | 白盒mAP50↓ | 黑盒mAP50↓ | 白盒ASR↑ | 黑盒ASR↑ |
|---|---:|---:|---:|---:|
| PNAP-best | 29.04 | 49.52 | 70.66 | 38.70 |
| PNAP-final | 27.36 | 49.30 | 74.19 | 39.97 |
| AdvPatch | 20.25 | 51.42 | 80.78 | 40.53 |
| PAC-P1 | 32.06 | 52.90 | 66.92 | 37.10 |
| PAC-P2 | 26.10 | 52.37 | 74.86 | 38.52 |
| T-SEA | 5.85 | 22.17 | 88.19 | 64.44 |
| PAC(T-SEA, Bc1) | 10.65 | 23.54 | 81.40 | 61.05 |
| PAC(T-SEA, Bc2) | 11.91 | 23.40 | 81.06 | 61.83 |
| NAP | 30.38 | 41.75 | 61.06 | 40.47 |

所有数值单位为%；均为单元等权平均，ASR不是把各检测器GT分母合并计算。

## 当前结论

- PNAP-best相对PAC-P1：白盒mAP50降低3.02个百分点，黑盒降低3.39个百分点。
- 相对PAC-P2：PNAP-best白盒mAP50高2.93个百分点（较弱），黑盒低2.85个百分点（较强）；白盒ASR也低于PAC-P2。
- PAC(T-SEA, Bc1/Bc2)的白盒、黑盒mAP50及ASR均优于PNAP-best；在这3个源模型各自的白盒mAP50与五目标黑盒平均mAP50上也全部优于PNAP。
- PNAP在不同训练源上的表现不一致：其YOLOv5s源的黑盒平均mAP50为56.25，低于PAC-P1的79.53；Faster R-CNN源则为37.29，高于PAC-P1的20.31。
- 这些是当前单次训练、共同子集上的数值比较，不构成统计显著性结论；剩余源完成前不宣称完整六源排名。

## 与原论文数值的关系

原稿article/cas-sc-sample.tex的Table 1中，PAC-P1在YOLOv2、YOLOv5s、Faster R-CNN上的mAP50分别为14.37、32.70、9.80。但原稿clean基准不同：例如YOLOv2为76.71，当前为95.03；Faster R-CNN为81.39，当前为97.32。因此不能直接把新PNAP行并入原表并称为同条件对比。上表采用项目内统一协议重新评估的PAC数据。

这里仅比较数字攻击指标；尚无PNAP与PAC的同协议隐蔽性、人评或物理攻击结果，不能由mAP/ASR推出这些维度上的优劣。

## 数据及核验

本次逐项读取162份metrics.json，确认数据集规模、预处理、补丁尺寸/面积、置信度/NMS/ASR阈值、batch、检测器权重哈希和clean基准一致。

- [对比汇总CSV](paper_matched_summary.csv)
- [逐源结果CSV](paper_matched_sources.csv)
- [逐单元结果及原始路径](paper_matched_cells.csv)
- [参数及文件哈希核验](paper_matched_audit.json)
- [原PNAP进度报告](report.md)
