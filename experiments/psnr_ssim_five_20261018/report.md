# 随机五图 PSNR/SSIM 试测

随机种子 20261018；INRIA Test 288 张中无放回抽取 5 张；来源 YOLOv5s。
共 3 donor × 3 版本；以同一干净图为参考，沿用固定色板 Bc1/Bc2。没有真实环境取色或攻击评测。

| 图片 | 人体数 | 补丁支持像素/640² |
|---|---:|---:|
| person_and_bike_095.png | 1 | 2.90% |
| person_250.png | 3 | 6.06% |
| person_073.png | 1 | 1.68% |
| person_and_bike_086.png | 1 | 3.63% |
| person_and_bike_029.png | 2 | 2.29% |

人体 ROI 先在每张图内对所有有效人体取平均，再对五图等权平均。SSIM 窗口为 11×11 高斯窗口，σ=1.5，RGB 三通道平均，data_range=1。
数值来自精确 float32 贴图张量（转 float64 计算），PNG 仅用于展示；整图列包含人工灰色 padding。去 padding 结果另存于 summary.csv 的 content_rect。

| Donor | 版本 | 整图 PSNR↑ (dB) | 整图 SSIM↑ | 人体 ROI PSNR↑ (dB) | 人体 ROI SSIM↑ |
|---|---|---:|---:|---:|---:|
| AdvPatch | Original | 23.7338 | 0.965468 | 14.4348 | 0.719663 |
| AdvPatch | Bc1 | 24.0797 | 0.966245 | 14.7731 | 0.725633 |
| AdvPatch | Bc2 | 24.2534 | 0.965675 | 14.9418 | 0.721858 |
| T-SEA | Original | 27.0178 | 0.966549 | 17.7249 | 0.725185 |
| T-SEA | Bc1 | 28.0360 | 0.967214 | 18.6954 | 0.729482 |
| T-SEA | Bc2 | 27.5507 | 0.966537 | 18.2120 | 0.724786 |
| AdvReal-2D | Original | 26.5320 | 0.966129 | 17.2404 | 0.722936 |
| AdvReal-2D | Bc1 | 27.4705 | 0.966776 | 18.1376 | 0.727622 |
| AdvReal-2D | Bc2 | 27.0537 | 0.966150 | 17.7228 | 0.723014 |

这些是五图试测，不是全测试集结论；高分表示相对干净参考图的改变更小，不能直接证明与环境更协调或攻击更强。

文件：selection.json（抽样）、manifest.json（协议/审计）、per_image.csv（逐图）、per_person.csv（逐人体）、summary.csv（均值/图间SD）、tensors/*.npz（精确输入）、index.html（图片对照）。

复现命令（脚本保护已有完整结果，重跑需复制到独立 experiments 子目录并创建/复用依赖环境）：
`experiments/psnr_ssim_five_20261018/.venv/bin/python -B experiments/psnr_ssim_five_20261018/run_five.py`


## LPIPS

使用官方 LPIPS AlexNet（`lpips==0.1.4`, version `0.1`），输入由 `[0,1]` 映射到 `[-1,1]`。LPIPS 越低表示与干净参考图的深层特征越相似。整图直接使用 640×640 张量；人体 ROI 统一双三次缩放到 224×224 后计算，先在每张图内平均，再对五张图等权平均。

| Donor | 版本 | 整图 LPIPS↓ | 人体 ROI LPIPS↓ |
|---|---|---:|---:|
| AdvPatch | Original | 0.039189 | 0.301435 |
| AdvPatch | Bc1 | 0.036328 | 0.286294 |
| AdvPatch | Bc2 | 0.034651 | 0.275172 |
| T-SEA | Original | 0.038412 | 0.304293 |
| T-SEA | Bc1 | 0.035856 | 0.285865 |
| T-SEA | Bc2 | 0.034910 | 0.280665 |
| AdvReal-2D | Original | 0.040524 | 0.317048 |
| AdvReal-2D | Bc1 | 0.037469 | 0.295694 |
| AdvReal-2D | Bc2 | 0.037123 | 0.292202 |

LPIPS 只描述相对干净参考图的深层特征距离，不能单独代表真实环境中的隐蔽性；完整协议和逐图/逐人体结果见 `lpips_manifest.json`、`lpips_per_image.csv` 和 `lpips_per_person.csv`。
