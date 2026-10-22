**PAC 新增开源对比方法核查（2026-09-11）**

按用户最新要求：用于最终实验的攻击目标是 person 漏检，贴图位置使用现有 PAC 协议。保留“2025–2026 正式发表优先、中科院一区/二区或 CCF A/B、非扩散、公开攻击训练代码、二维补丁”的筛选条件；不按攻击强弱挑选。CAP、AdvReal 已有实验，RFA-Tex 按用户要求暂缓。本次只查资料与源码，没有启动训练。

**建议：优先考察 DynamicPAE；CDUPatch 作为需核查核心机制的 RGB 适配候选；TAOB、DSAP 列为扩展候选。不能把四个项目都写成当前协议下已能直接运行的原版基线。**

公开训练源码已核实，但未在本轮安装依赖或复现训练。前四项基于 2026-09-09 固定版本的实际源码核验，本轮结合“中心贴图、person 漏检”的要求重新评估适配性。此前证据见 [源码核查记录](/datacc05/huotianlin/code-hu/PAC/third_party/verification/open_attack_baselines_20260909.md)。

| 方法 | 论文 / 正式发表 | 等级 | 公开代码 | 当前协议下的适配判断 |
|---|---|---|---|---|
| DynamicPAE | [DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time](https://doi.org/10.1109/TPAMI.2025.3626068)，TPAMI，2025 在线 / 2026 卷期 | 中科院大类一区 | [hujinCN/DynamicPAE](https://github.com/hujinCN/DynamicPAE) | 与当前任务最接近：行人检测、INRIA/COCO、T-SEA detlib。位置和面积可统一，但要保留场景条件生成，明确每张图生成的补丁不同。 |
| CDUPatch | [CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors](https://doi.org/10.1145/3746027.3755188)，ACM MM 2025 | CCF A | [Jiahuan-Long/CDUpatch](https://github.com/Jiahuan-Long/CDUpatch) | 通用二维 RGB 补丁、颜色相关方法。原版联合可见光/红外，当前 RGB person 实验须标注适配；应先判断保留哪些核心约束，不能删掉跨模态设计后仍声称完整复现原法。 |
| TAOB | [Targeted attack via adversarial patch outside bounding box](https://doi.org/10.1016/j.patcog.2025.112244)，Pattern Recognition，2025 在线 / 2026 卷期 | 中科院大类一区 | [mayfly227/TAOB](https://github.com/mayfly227/TAOB) | 原版是框外定向攻击。可以研究中心贴图、person 目标的适配，但框外掩码和定向目标重构涉及核心设计；仅改坐标、删损失并不足以构成忠实复现。 |
| DSAP | [A Unified Framework for Generating Diverse and Stealthy Adversarial Patches Against Aerial Object Detection](https://doi.org/10.1109/TGRS.2025.3594008)，TGRS 2025 | 中科院地球科学大类一区；遥感小类二区 | [Yongyx99/DSpatch](https://github.com/Yongyx99/DSpatch) | GAN 生成背景相似的二维补丁，原目标是航拍虚假目标。改为 person 漏检需改变检测损失并接入当前贴图器，保留生成器/判别器及隐蔽性机制，作为任务适配扩展实验报告。 |

期刊分区统一采用**中科院 2025 年 3 月升级版大类分区**，不是 JCR Q1/Q2，也不是 LetPub 的“2026 新锐分区”。来源：[TPAMI](https://www.letpub.com.cn/index.php?page=journalapp&view=detail&journalid=3411)、[Pattern Recognition](https://www.letpub.com.cn/index.php?page=journalapp&view=detail&journalid=6515)、[TGRS](https://www.letpub.com.cn/index.php?page=journalapp&view=detail&journalid=3389)、[CCF 图形学与多媒体目录](https://www.ccf.org.cn/Academic_Evaluation/CGAndMT/)。对应页面快照与分区核验保存在前述 2026-09-09 记录中。

DynamicPAE 的 [arXiv 首版](https://arxiv.org/abs/2412.08053)发表于 **2024 年 12 月**。这里按正式期刊发表年份纳入；若要求算法首次公开也必须在 2025 年之后，它不满足该更严格条件。该方法使用生成器，不是扩散补丁；README 中出现 diffusers 或可选 AutoencoderTiny，不能据此把核心算法归为扩散攻击。

**真正的训练入口与固定版本**

- DynamicPAE：`sources_root/dynamic_example/experiments/run_pretrain.py` → `trainer/detection_atk_pretraining.py`；固定提交 `2481550ea43c2bed2b3f3e0bdf04577828434905`。已核对生成器、攻击/重建损失和反向传播。
- CDUPatch：`train_patch.py`、`test_patch.py`、`vis2infra/`；固定提交 `0418d7d967ba0d1b73203d3882aa725a2debb187`。当前原仓库配置包含微调 YOLOv5x、五类头及可见光/红外流程，不能直接把其检测器权重作为 PAC 的共同 COCO 权重。
- TAOB：`TAOB.py`、`utils/loss.py`、`utils/patchapply.py`；固定提交 `6208542979bdcc0b6f93df6c5ebb3e494bae8553`。已有像素梯度更新，以及检测抑制、上下文、定向目标和 TV 损失。
- DSAP：`adversarial_patch_uav_rod/train_DSAP_uavrod.py`，另有 `train_DSAP_uavrod_phys.py`；固定提交 `fce529074064f077bcdae5a40df201516fedc0bf`。**不要误用 `train_advpatch_uavrod.py`**：它是直接像素 AdvPatch 路径，不能代表 DSAP 的 GAN 方法。

**统一协议与方法身份**

实验端沿用已核查的 PAC 协议：GT 人框中心、方形补丁面积为人框面积的 25%、300×300 纹理；图像原分辨率中心填充 RGB127 至正方形，再 PIL 双线性缩放到 640；batch size 8。现有数据是 INRIA Train 614 张 / Test 288 张，`images/val` 实际对应 Test，测试人框 589 个。AP50 评估 conf=0.001、NMS IoU=0.6、max_det=300、测试 EOT 关闭；ASR 另用 conf=0.5 和匹配 IoU=0.5。YOLOv2/v3/v4/v5s/v5m/Faster R-CNN 使用相同已冻结评测权重。

- 原论文贴图位置或类别不同，不自动否决；需要检查能否在统一约束下保留核心优化方法。改动应记录为 `method (RGB/person adaptation)`，并报告与原法的差异。
- 通用补丁与场景条件补丁应明确标记。DynamicPAE 若每图生成新补丁，不能描述成全测试集共用一张纹理；简单固定某张生成结果也不能代表原动态方法。
- 白盒实验训练与评估使用同一规定模型；迁移黑盒实验只在源模型训练，不能通过目标模型梯度或目标测试表现选择补丁。动态方法也必须明确测试时能读取什么输入。
- 对方法自有的损失、正则、优化器与学习率保留原始设计；位置、面积、数据划分和评测管线统一。若适配删除了核心贡献，应换候选，而不是仅保留论文名字。

**本轮新核实但没有纳入主推荐的项目**

1. **Higher-Order Adversarial Patches for Real-Time Object Detectors**：确有 [训练代码](https://github.com/JensBayer/HigherOrder)，使用 INRIA/person evasion；论文 [arXiv:2601.04991](https://arxiv.org/abs/2601.04991)。正式出版已核实为 [ICPR 2026 论文集章节](https://doi.org/10.1007/978-3-032-31583-0_41)，2026-08-04 在线，论文集版权年 2027。[Springer 论文集页面](https://link.springer.com/book/10.1007/978-3-032-31583-0)明确写明 ICPR 2026，**不是 Elsevier 的 Pattern Recognition 期刊**；本轮没有将其列为已核实的 CCF A/B。固定提交 `efd0778134f3ce54b90cc7ccf29ddf12e56abb1a`，真实补丁训练入口 `scripts/train/trainYolov10Patch.py`。正文所谓 higher-order 指交替训练补丁与加固检测器，不是高阶梯度优化。它需要额外检测器训练；只在原冻结检测器上跑一次其补丁脚本，不是完整高阶流程。
2. [SmuDPatch](https://github.com/visheshrajput2408/SmuDPatch)：仓库声明 ECCV 2026，有 `deformable_patch_working_catmull.py`。主机制是 Catmull–Rom 轮廓及进化形状搜索，原目标为 ImageNet 分类；COCO 部分是补丁检测/OOD 测试，不等于 person 漏检训练。固定方形约束会删除形状搜索核心。本轮未独立核验其会议录用记录，未把仓库自报年份当作已核实正式出版。
3. [Fourier-shape attack](https://github.com/Yongyx99/Fourier-shape-attack)：仓库声明 ICML 2026，确有 `attack_fouriershape_yolov3.py`、`base.py`。攻击对象是红外行人检测，LLVIP、逐实例傅里叶轮廓优化。不能不加说明地当成通用 RGB 方形补丁；本轮未独立核验其会议录用记录。
4. [Tang 航拍补丁仓库](https://github.com/tang-agui/adversarial_patch_attacks_against_optical_object_detectors)：2025 年上传，但对应 [Neurocomputing 2023 论文](https://doi.org/10.1016/j.neucom.2023.03.050)。其 false-positive 续作对应 [2024 论文](https://doi.org/10.1016/j.neucom.2024.127431)。仓库创建时间不代表论文年份，因此均不作为新增 2025–2026 方法。
5. [SRAP](https://github.com/HerTenSu/TIFS26_SRAP) 是 TIFS 2026 的分类/隐私保护补丁示例；PhysPatch 是多模态大模型自动驾驶攻击，UPA-RFAS 是 VLA 机器人攻击。不能仅因期刊会议等级合适就当作已验证的 person 检测基线。
6. [Carson1391/yodo](https://github.com/Carson1391/yodo) 有 YOLOv3 person 攻击代码，但 README 定位为研究作品集，未核实到符合要求的正式论文；[Human-imperceptible-adversarial-patch](https://github.com/addiajsjjs/Human-imperceptible-adversarial-patch) 是 NIR/VIS 人脸识别，并非行人漏检。

**论文很相关，但未核实到公开训练实现**

- [Fooling human detectors via robust and visually natural adversarial patches](https://doi.org/10.1016/j.neucom.2024.128915)，Neurocomputing 2025，中科院二区。
- [Transferable and Robust Dynamic Adversarial Attack Against Object Detection Models](https://doi.org/10.1109/JIOT.2025.3531852)，IEEE Internet of Things Journal 2025。
- [Focus Matters: Advancing Adversarial Patch Transferability through Contextual Perception and Gradient-Prior Masking](https://doi.org/10.1109/TCSVT.2026.3727098)，TCSVT 2026。`cvsp-lab/FocusMatters` 实际是另一篇 VLM 幻觉工作，不能作为这篇论文的开源链接。
- [Robust Adversarial Patch for Object Detection Using Self-Similarity for Multiscale Attacks](https://doi.org/10.1109/TDSC.2026.3681111)，TDSC 2026。

“未核实到”不是证明作者没有代码。这几项目前不计入满足用户开源条件的名单，也没有根据题名虚构实现或简称。

出版信息来自 Crossref、OpenAlex、论文正文和出版商页面；源码证据来自作者仓库。当前目录的 `metadata/` 保存元数据，`sources/` 保存可核查正文/代码片段。Google/Bing 未产生有效结果的搜索缓存不用于支持推荐。静态读取训练入口不等于完整环境及实验已跑通。
