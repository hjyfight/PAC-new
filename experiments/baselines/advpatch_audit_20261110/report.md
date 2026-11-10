# 当前 AdvPatch 训练与作者实现核对

审查日期：2026-11-10。范围：当前统一比较所用六个 AdvPatch best 补丁、实际训练配置/日志、作者公开代码和 Thys 等人的原论文，以及 PAC 稿件。没有改动训练实现、实验结果或论文，也没有重训。

## 结论

当前实现是有原论文依据的 **AdvPatch OBJ-CLS 变体在 PAC 统一协议下的适配**，并非作者 README 推荐的 `paper_obj` 配置的严格复现。核心优化公式没有发现无效反传、NPS/TV 算错或初始化为自然图片的问题；但损失版本、几何变换、贴图框与预算确实存在差别，应当披露，不能笼统称为“完全与官方一致”。

## 作者来源

本次联网下载的作者 `README.md`、`patch_config.py`、`train_patch.py`、`load_data.py` 与项目原有 `experiments/baselines/adversarial-yolo-master/` 对应文件逐字节相同。下载地址和 SHA256 见 [source_manifest.json](source_manifest.json)。

- [作者仓库与 README](https://gitlab.com/EAVISE/adversarial-yolo)
- [作者配置](https://gitlab.com/EAVISE/adversarial-yolo/-/blob/master/patch_config.py)
- [作者训练脚本](https://gitlab.com/EAVISE/adversarial-yolo/-/blob/master/train_patch.py)
- [作者变换与损失实现](https://gitlab.com/EAVISE/adversarial-yolo/-/blob/master/load_data.py)
- [原论文：Fooling automated surveillance cameras: adversarial patches to attack person detection](https://arxiv.org/abs/1904.08653)，重点为 §3.1、§3.2、§4.1 和 Figure 4–6。

## 一、最重要的区别：OBJ-CLS 与 OBJ

当前 YOLO 分支：

`mean_over_images(max_over_candidates(objectness * person_probability))`

对应作者 `BaseConfig.loss_target = lambda obj, cls: obj * cls`，以及原论文的 OBJ-CLS。作者 README 推荐执行 `python train_patch.py paper_obj`，该配置采用 `lambda obj, cls: obj`，只优化 objectness；原论文实验中 OBJ 的 AP 最低，并将其用于最佳补丁示例。

因此，当前版本不是没有论文出处，而是没有选用作者推荐复现的损失版本。作者在 Figure 4 的讨论中指出，OBJ-CLS 可能生成类似 teddy bear 的图案，让另一类别压过 person。当前补丁呈现类似自然物体的结构，不能据此外观判定训练错误；但也不能据此外观确认某一具体物体类别，尚未做补丁自身的类别检测分析。

YOLOv5s 实际调用位于 `experiments/ours/yolov5-advpad-20260908/train_patch.py`，`utils/patch.py::attack_loss`。其余五源使用 `experiments/baselines/yolov5-advpatch-cap-20260911/train_patch.py` 与 `detector_backends.py::advpatch_loss`。Faster R-CNN 优化逐图最大 pre-NMS ROI softmax person 概率，是项目的检测器适配，不是原论文发布的 FRCNN 训练实现。

## 二、与作者代码一致的部分

- 直接优化 RGB 像素，初始化为 0.5 灰色；当前配置 `source: null`，没有以自然图像或其它补丁初始化。
- 300×300 补丁参数，Adam AMSGrad，初始学习率 0.03。
- `ReduceLROnPlateau(mode=min, patience=50)`。
- `loss = det + 0.01*NPS + max(2.5*TV, 0.1)`。
- 使用作者的 30 色 NPS 调色表与归一化方式；TV 为作者代码的相邻像素差绝对值和，按像素通道元素数归一化。
- 更新 RGB 后裁剪至 [0,1]。
- 7×7 median，contrast [0.8,1.2]，加性 brightness [-0.1,0.1]，噪声 [-0.1,0.1]，角度 [-20°,20°]。

注意：作者配置中虽然有 `max_tv=0.165` 字段，公开训练脚本实际执行的是硬编码的 `max(tv_loss,0.1)`；本审查比较实际执行语句。

## 三、不能忽略的协议与实现差异

| 项目 | 作者实现 / 原论文 | 当前比较 |
|---|---|---|
| 推荐检测损失 | README `paper_obj`：OBJ；论文也研究 OBJ-CLS | YOLO：OBJ-CLS |
| 训练贴图框 | 由干净检测器预测得到的框 | INRIA 人工标注的人体框 |
| 补丁边长 | `0.2*sqrt(w²+h²)` | `round(sqrt(0.25*w*h))` |
| 位置 | 框中心，并在图像归一化 y 上减 0.05；训练 `rand_loc=False` | 框中心，无上述上移 |
| 旋转实现 | 先将补丁填充到图像画布，再仿射旋转缩放 | 在补丁自身 300×300 画布旋转，再缩放贴入 |
| 训练轮数 | 公开脚本 `n_epochs=10000`；这不是对论文实际收敛轮数的断言 | 固定 200 轮 |
| 检测器与输入 | 公开实现是 YOLOv2，本地作者 cfg 为 416×416 | 六个检测器，统一 640×640 |
| 评估 | 原论文以干净检测结果作参考框，并包含随机变换 | 人工 GT AP；评估不启用 EOT |

尺寸公式也不是同义改写：对于 100×200 的框，原版边长约 44.7，面积约为框面积的 10%；当前边长约 70.7（取整前），面积为 25%。作者的 0.2 是对角线比例，不是面积比例。

**旋转还有一个具体实现偏差。** 当前在补丁自身画布上旋转，转出画布的角会被裁掉。CPU 几何检查中，300×300 补丁转 20° 后，保留覆盖面积约为原面积的 87.65%；填充到更大画布再旋转约为 100%。这不是原版完整旋转的等价实现。若目标是忠实复现作者 EOT，这一处需要修正；若保留当前共享变换，则需披露。该检查只证明几何差异，没有测量它对 AP 的具体影响。

## 四、实际训练记录

| 源模型 | 完成轮数 | 最低训练总损失所在轮 |
|---|---:|---:|
| YOLOv2 | 200 | 162 |
| YOLOv3 | 200 | 183 |
| YOLOv4 | 200 | 200 |
| YOLOv5s | 200 | 199 |
| YOLOv5m | 200 | 200 |
| Faster R-CNN | 200 | 200 |

六份 `results.csv` 均连续记录 1–200 轮，所有数值有限；没有发现提前中断。五个新增源的完成摘要分别记录 15,400 次更新。YOLOv5s 复用 s000，其完整 200 轮日志和 best/last 均存在。所有用于本次画廊的 AdvPatch checkpoint SHA256 与对应当前评估记录一致。

best 的选取依据是 epoch 平均训练总损失，并非测试 AP。完成训练和损失下降不等于充分收敛；多个源的最优训练损失出现在最后一轮，现有记录不能证明延长训练没有收益。也不能把官方脚本的 10,000 轮直接当作必须执行的统一训练预算。

## 五、与 PAC 稿件的一致性

`article/cas-sc-sample.tex:574` 中的 640×640、batch 8、Adam lr 0.03、200 轮与当前 AdvPatch 配置一致；25% 在当前代码中明确解释为人体框面积比例。

但正文没有说明 AdvPatch 选用 OBJ-CLS，没有详细披露从原作者协议改动的贴图规则。附录 `appendix:eot` 描述的是 PAC 的颜色概率矩阵 m，包含均值平滑、brightness [0.9,1.1]、noise [0,0.1]、独立缩放 [0.9,1.1]；不能将这一组描述直接当作当前 AdvPatch RGB 分支的设置。建议把两类 EOT 分开说明，而不是断言二者已经逐项一致。

稿件表格还包含旧结果，例如白盒 AdvPatch YOLOv5s 为 31.60，当前所展示 best 的 AP50 为 30.4028。这份稿件尚不能视为当前统一实验结果的完整记录。

## 六、建议

1. 当前已有结果可保留，方法名称明确写为 **AdvPatch (OBJ-CLS, shared protocol)**，并列明协议调整。
2. 如果论文要声称“按作者推荐配置复现 AdvPatch”，应补做 OBJ-only 版本，并解决旋转裁剪差异。应分别设置对照，避免一次同时改变损失、尺寸和轮数而无法解释差异来源。
3. OBJ 在原论文中更强，不意味着在本项目所有源模型和设置下必然更强；须经相同协议的实验判断。不能根据外观或当前训练曲线提前断言。

## 验证与产物

- [audit_checks.py](audit_checks.py)：只读 CPU 检查脚本。
- [audit_checks.json](audit_checks.json)：NPS/TV/OBJ-CLS 数值及梯度、灰色初始化、旧/新 AdvPatch 合成输入输出及梯度、旋转覆盖面积、训练日志与 checkpoint 哈希核对结果。
- NPS/TV 梯度与作者公式一致；OBJ-CLS 输出及梯度与作者 YOLOv2 提取器一致，输出误差为浮点求和量级。
- 合成输入中，复用 s000 与新增源代码的 AdvPatch 贴图输出、梯度完全一致。
- 检查未重新运行六个检测器的完整训练或评估，不能替代收敛与攻击强度对照实验。
- [AdvPatch 六源高清图](../../analysis/pac_patch_visualizations_20260910/current_20261110/advpatch_all_sources_best.png)。
- [YOLOv5s 主要方法高清图](../../analysis/pac_patch_visualizations_20260910/current_20261110/yolov5s_main_methods_best.png)。
- 画廊包含 84 张攻击补丁和 6 张 Gray，按训练损失选择 best，原 PNG 没有调色或增强；Node VM 模拟 DOM 下已检查六源切换、放大/收起和状态恢复。
