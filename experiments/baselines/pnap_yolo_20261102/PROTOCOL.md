# PNAP-YOLO（2025）：统一 PAC 协议适配

论文：Jun Li, Chenwu Shan, Liyan Shen, Yawei Ren, Jiajie Zhang，
[PNAP-YOLO: An Improved Prompts-Based Naturalistic Adversarial Patch Model for Object Detectors](https://doi.org/10.1007/s40745-025-00604-0)，Annals of Data Science，2025-05-02 在线。
源码：[Kwizii/PNAP](https://github.com/Kwizii/PNAP)，固定提交 `dc1b1f025f3492de61e6dd1890e11d116f9fc649`，Apache-2.0。
`author/` 保持下载原样；文件 SHA256 和论文 Crossref 元数据保存在 `evidence/`。这是公开实现的 PAC 协议适配，不声称复刻原论文数值。

PNAP 与已有 ICCV 2021 NAP 是两个方法。PNAP 使用扩散模型；本次按用户明确点名纳入，不沿用旧候选清单的“排除扩散”筛选结果。

## 保留机制

- 冻结 Stable Diffusion 1.5 的 UNet、VAE、文本编码器，只更新 `(1,4,64,64)` 潜变量；不训练直接 RGB 像素。
- 权重使用 `runwayml/stable-diffusion-v1-5` 重定向后的官方公共仓库 `stable-diffusion-v1-5/stable-diffusion-v1-5`，固定版本 `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`。保留 float32 完整权重。
- 原默认 Pomeranian 文本、negative prompt、guidance=7；先做50步初始去噪，再加噪至601，训练时按 `[601,401,201,1]` 四步 DDIM 生成。
- 原 `patch_embed.py` 的 `prompt_embedding()`、`gen_patch()` 直接调用，保留其 scheduler 行为，包括 `num_inference_steps=4`。不自行重新解释时间步。
- 损失：检测损失 + 0.01×NPS + max(2.5×TV, 0.1)。NPS 使用原30色打印色表；NPS/TV 在原生512×512纹理上计算，直接加载作者定义。
- Adam AMSGrad，LR=0.001，默认 betas；ReduceLROnPlateau，patience=50，监控每轮平均总训练损失。500轮，seed=1176426343；614图、batch=8，每轮77次更新，共38500次更新/源。**500轮来自作者源码，用户于2026-11-02明确同意采用；尚未独立核实付费论文正文中的轮数，不将源码默认值称为已核实的论文设置。**

## 检测损失及公开代码的差异

公开代码中各模型并不共用一个统一的最大 person 分数；不能把 AdvPatch loss 直接换名为 PNAP。

| 源模型 | 当前实现 |
|---|---|
| YOLOv2 | 保留原筛选：argmax 类别为 person，obj×class>0.4，无 NMS；对剩余框的 objectness 求均值，再平均非空图。使用等价向量化，避免作者逐框 CPU 循环。 |
| YOLOv3 | 原 `MaxProbExtractor_yolov5`，objectness 再做 sigmoid 后逐图取最大值；保留已发布代码的二次 sigmoid 行为。 |
| YOLOv5s | 原 `MeanProbExtractor_yolov5` 及 `non_max_suppression_gradable`，conf=0.4、IoU=0.45、max_det=300。 |
| YOLOv5m | 显式扩展：同 YOLOv5s 损失，使用共同协议的 v5m 权重。 |
| YOLOv4 | 显式扩展：作者注册表有 `yolov4` 名称但缺少对应类定义。重建共同后端的原生 obj/class，采用作者 v5 mean extractor。 |
| Faster R-CNN | 显式扩展：使用作者已定义的 `MeanProbExtractor_frcnn`；作者没有可直接运行的该模型训练配置。 |

作者可微 NMS 的当前代码实际以 `obj*class^2` 筛选，并以该连续分数而非类别 ID 构造 NMS 坐标偏移，最终返回 objectness 供 mean loss 使用。本适配保留该行为，没有静默修正成标准 NMS。这是公开实现的局限，报告时应披露；标准 PAC 评测仍使用原共同评测代码。空检测批次的 CPU 零改为设备一致、可反传的零，数值不变。

## 共同约束与适配变化

- 使用已冻结的六个 COCO 检测器及 INRIA Train614/Test288（589个GT）。图像/标签与检测器权重逐文件校验 SHA256，训练/测试图像无交集。
- 图像在原始分辨率居中填充 RGB127 至正方形，再 PIL 双线性缩放到640；使用 GT 人框中心，方形面积为人框25%。采用现有 `utils.patch.apply_patch` 的取整、覆盖和边界处理。
- 原生512纹理双线性缩放到共同300×300纹理；原作者416输入、batch16、矩形/对角线尺度贴图改为上述共同约束。
- 训练增强保留范围：中值滤波7、对比度0.8–1.2、亮度−0.2–0.2、噪声−0.1–0.1、旋转±15°；在共同300纹理上逐贴图应用，无随机位置。使用共同旋转覆盖掩码，避免作者重复乘插值掩码的边缘行为。随机采样使用当前工程 RNG，非逐位重放原作者增强。
- 将硬编码双 GPU 改为单 GPU。UNet/VAE使用激活重计算以减少显存，冻结参数但保留输入梯度；文本编码后移到 CPU。没有降低扩散步数或替换生成器。
- 2026-11-02共享GPU短程验证发现显存不足后，YOLOv2/3/4/5m使用独立的 `run_offloaded.py` 将检测器保存的反传张量暂存CPU固定页内存，按需传回原GPU；计算仍为float32、batch8。两张真实训练图的原入口/新入口损失完全相同，梯度相对L2误差≤6.27e-8，记录见 `evidence/detector_offload_equivalence.json`；不声称长程逐位重现。各源执行配置另存 `execution_profile.json`，原训练代码、权重及 `protocol.json` 不变。新任务使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少显存碎片。
- 每轮结束导出 best/final；best 按该轮平均训练总损失选择，重新生成轮末潜变量对应补丁。保存 optimizer/scheduler/RNG，训练可从轮末恢复，不使用测试集挑选补丁、提示或超参数。

## 评测与完成标准

正式评测使用原 `pac` Python 环境调用现有 `eval_transfer.py`，避免 PNAP 的依赖版本改变历史评测。conf=0.001、NMS IoU=0.6、max_det=300、batch8、无测试EOT。ASR 是 clean 检出的GT在贴图后漏检比例，conf/IoU均0.5。

每源 best/final 在六模型评测：12个攻击单元+6个clean重放。六源合计72个攻击单元；缺失值留空，不伪造均值。clean 的 mAP50、ASR 分子分母须与已审计基线精确一致。

`tables/comparison_best_with_pnap.csv` 从既有统一协议主表引入 AdvPatch、PAC、CAP、AdvReal、T-SEA、NAP，并追加 PNAP-YOLO。只有6个白盒/30个黑盒单元齐全才报告对应平均值；单源结果可直接在 `tables/comparison.csv` 查看。

`smoke/` 仅用于短程训练/评测验证；不进入 `tables/comparison.csv`。正式500轮结果在 `sources/`，完成标记分别为 `train_complete.json` 和 `evaluation_complete.json`。

## 操作

在 PAC-yolo 根目录运行；首次已创建 `.venv`，所有缓存位于项目内。

```bash
experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/download_weights.py
experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/run_experiments.py preflight
env CUDA_VISIBLE_DEVICES=0 experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/validate_integration.py
env CUDA_VISIBLE_DEVICES=0 experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/run_experiments.py train --model yolov5s --smoke-steps 2
env CUDA_VISIBLE_DEVICES=0 experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/run_experiments.py evaluate --model yolov5s --smoke
env CUDA_VISIBLE_DEVICES=0 experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/run_experiments.py train --model yolov5s
env CUDA_VISIBLE_DEVICES=0 experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/run_experiments.py evaluate --model yolov5s
experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/run_experiments.py summarize
```

其他源模型用相应 `--model`。已有 `protocol.json` 时不要重复 prepare；已有 smoke 产物时不要将其作为正式训练断点。
