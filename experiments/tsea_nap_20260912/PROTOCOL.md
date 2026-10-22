# T-SEA、NAP 与 PAC 配对实验协议

本轮不改动论文正文，也不改动历史 Table 1/2 数字。新增实验和导出表全部放在本目录；PSNR/SSIM 按此前要求继续暂停。

## 训练与评测条件

- INRIA：Train 614 张，官方测试部分 288 张、589 个 person GT。使用既有划分；逐文件复核图像和标签 SHA256，训练/测试图像哈希无交集。测试集不用于选 epoch、色板、潜变量类别或学习率。
- 六个模型：YOLOv2、YOLOv3、YOLOv4、YOLOv5s、YOLOv5m、Faster R-CNN。权重及来源见 manifest.json。
- 输入：PIL RGB → 原尺寸居中填充 RGB(127,127,127) 正方形 → PIL 双线性缩放至 640×640。不是直接拉伸，也不是先缩放再填充的 letterbox。
- 2D 通用补丁，输出纹理 300×300；每个 person GT 框中心放置，方形边长为 round(sqrt(0.25×框宽×框高))。各方法共用同一整数取整、边界裁剪、重叠覆盖实现。测试关闭所有 EOT。
- batch size 8，workers 4，单卡独立优化；动态排队使用空闲 GPU，不进行多卡联合训练。
- AP：confidence=0.001，NMS IoU=0.6，max_det=300；指标为 person AP50（单类时记作 mAP50）。另保存条件消失 ASR：以 clean 在 confidence≥0.5、IoU≥0.5 时一对一匹配到的 GT 为分母。
- WB：六个源模型对应的六个对角单元；BB：三十个非对角单元。单源 BB 平均为另外五个目标。缺失数据留空，不用零代替，也不计算不完整平均值。
- 同预算灰色遮挡控制在六个检测器分别评估，位于 controls/gray/；它没有训练源，不伪造六源迁移矩阵，不计入288个新增攻击单元。该控制加入新增白盒表，原稿 Gray 行保持不动。
- 本轮两种新方法均按作者默认训练 1000 轮，即每源 77,000 次更新；不早停。seed=1788272031，沿用选定的 PAC 协议。best 取 epoch 平均训练总损失最小的状态，最后一轮独立评估，禁止用测试集挑选 best。

## 保留的基线机制与适配

| 项目 | T-SEA | NAP |
|---|---|---|
| 官方来源 | https://github.com/VDIGPKU/T-SEA | https://github.com/aiiu-lab/Naturalistic-Adversarial-Patch |
| 参数化 | 灰色初始、直接 RGB，更新后截断到 [0,1] | 冻结作者 BigGAN G_ema，仅优化 120 维潜变量 |
| 初始 LR | 0.03 | YOLO 0.005，Faster R-CNN 0.02 |
| 优化器 | Adam AMSGrad，默认 betas | Adam AMSGrad，betas=(0.5,0.999) |
| 调度器 | 原作者 ALRS，patience=10，decay=0.97 | detection loss 上 ReduceLROnPlateau，patience=50 |
| 检测目标 | 原作者 detlib 暴露的逐图最大置信度 | YOLO：逐图最大 person objectness×class；FRCNN：最终 person 得分≥0.5 的最大值，无匹配时零 |
| 正则 | max(2.5×TV,0.1)，无 NPS | 原生 128×128 生成纹理的 0.1×TV，无 NPS、判别器或重叠损失 |
| 核心机制 | 图像增强、残差 ShakeDrop、补丁 Cutout 三者保留 | 潜变量每次更新前舍入至四位小数，限制 [-10,10]；固定生成类别 259 |

T-SEA 的图像增强在填充/缩放前进行：0.5 概率启用，内部 0.5 概率水平翻转、0.1 色彩抖动、±5° 旋转；GT 框同步变换。补丁执行中值滤波7、光度抖动、截断、Cutout、共享几何变换。Cutout 的批级开关概率0.9、边长比例0.4、填充值0.5、实例中心范围[-0.05,1.05]，保留作者边缘插值行为。

YOLOv3/v4 逐块验证原配置与作者 ShakeDrop 配置，仅改变指定 shortcut；YOLOv5 在第4/6/8阶段的残差支路施加作者函数；Faster R-CNN 仅在 backbone.layer3 的第1–5号块施加。YOLOv2 原作者配置就没有残差扰动，不人为添加；YOLOv5m 沿用 YOLOv5s 的相同阶段对应，这是架构扩展。

T-SEA 原配置使用模型预测框、416 输入以及不同位置/面积；这里明确改为共享 GT 中心、面积与640输入。其 YOLOv2/3/5 损失使用 raw objectness，YOLOv4 使用 detlib 暴露的最大联合置信度，FRCNN 使用最终框得分；不会为取得更好结果擅自换目标。

NAP 使用作者精确 EMA 权重，SHA256：c9706dfbaad7687617921f45d6d1a95d2e8b6aeeb7b3af8f89cb87063b87d741。生成器原生128×128输出双线性变为共同的300×300纹理，不训练生成器，不把自由像素优化冒充 NAP。作者默认矩形/位置/输入/batch 改为共享条件；保留无旋转、无随机位置、无模糊的默认光度增强。作者未提供 YOLOv5，两个 v5 版本采用相同 person 目标和0.005学习率的显式扩展。作者副本仅删除 BigGAN.py 中未使用的 ipdb 调试导入，记录在 author_changes.json。

因此结果应称为“统一 PAC 协议下的基线适配”，不能声称完全复刻各基线原论文实验设置。不同方法的训练轮次不要求全部相同；共同攻击约束与评测条件保持一致，并公开预算。

## PAC 换色

T-SEA 的 best/final 各生成 Bc1、Bc2 两个版本，不再训练。既有 AdvPatch、CAP、AdvReal 的配对结果从已审计记录读取，不重写原产物。

```
P_new[h,w,c] = sum_k P_raw[k,h,w] * Bc[k,c] / 255
Bc1 = [[119,49,72],[2,204,1],[134,2,182]]
Bc2 = [[199,21,131],[40,165,4],[16,69,120]]
```

P_raw 已在[0,1]，不重复除以255，不用 softmax、不按通道和归一化、不匹配亮度、不对补丁做 K-means。两套色板各列 RGB 和均为255，当前输入范围不会产生越界裁剪。每个产物用独立 float64 公式核验。CPU 换色计时预热10次、重复100次，排除磁盘I/O和模型加载；额外训练轮次为0。

## 论文表格安排

1. `tables/tsea_paired_best.tex`：建议的 T-SEA 新表。每个源模型三行：T-SEA、T-SEA+PAC(Bc1)、T-SEA+PAC(Bc2)。列为源模型、版本、WB mAP50、另外五个模型的 BB 均值、相对原补丁的 ΔWB、ΔBB、换色毫秒。正 Δ 表示攻击损失。正文可用此表，完整迁移矩阵放补充材料。
2. `tables/paired_best.tex`：AdvPatch、CAP、AdvReal、T-SEA 各自 Original/+PAC(Bc1)/+PAC(Bc2)，报告6模型WB均值、30单元BB均值及代价。这检验已测试基础攻击的可复用性，不等同于证明任意攻击都兼容。
3. `tables/whitebox_shared_best.tex` 与 `transfer_shared_best_*.tex`：统一协议下单独导出的攻击对照，包含已有方法和新增 T-SEA/NAP。原论文 Table 1/2 不自动覆盖。历史 clean 与当前 clean（尤其 YOLOv2、FRCNN）不同，不能把新旧数字拼接成一次受控排名。
4. `tables/best_final_sensitivity.csv`：新方法 best 与 final 独立披露。CAP 既有 best/final 在 FRCNN 上差别大，需同时查看上一轮完整报告，不能只用较弱的一组概括其能力。

新训练结束前，T-SEA/NAP 表格对应格为 --；已完成的 AdvPatch/CAP/AdvReal 数字来自实际结果文件。单种子结果不使用“显著优于”等统计结论。

## 隐蔽性对照

当前先生成同场景材料，暂不计算或改写 PSNR/SSIM。材料目录为 `appearance_comparison/`。

- 同图同位置同面积，测试 EOT 关闭；既展示完整图，也展示相同人体 ROI，并附原补丁纹理。整图边框/字体不会混入待评分图片。
- 主要对照：AdvPatch、PAC-P1/P2、NAP；加上 Gray/clean 控制，以及 CAP、AdvReal、T-SEA 与可用的 PAC 版本。NAP 为自然外观目标的基线；T-SEA 为迁移基线，不能把两者角色混为一谈。
- 展示样本按原始 GT 面积的四分位分层、固定 seed 抽取，不根据攻击结果或好看的样例挑选。量化若恢复，建议覆盖全部288张测试图，而不是只测展示样本。
- 如恢复 PSNR/SSIM：reference 必须是同一张干净图。原始整图结果与人体 ROI/补丁区域结果分开报告；先锁定颜色空间、data_range、SSIM窗口和聚合规则。大面积未变背景与正方形填充会抬高整图分数，因此整图相似度不能单独充当隐蔽性证据。
- 匿名主观评价沿用论文的7级尺度，分别记录环境一致性（高分好）、显眼程度（低分好）和自然程度（高分好）；环境一致性须使用保留背景的整图。不要将三个概念合为一个分数。数字图片的新评分与原有实物实验独立报告，不合并参与者或均值。脚本提供匿名图片、方法对照密钥及空白评分 CSV；尚无参与者数据。评分页面关闭或刷新前须导出 CSV，页面不自动持久保存。
- 所有外观结果需与同一补丁的 WB/BB 攻击效果联合展示，给出攻击—外观折衷。Gray 本身也可能明显降低检测性能，应使用同预算灰色控制区分遮挡与优化图案的作用。不能单凭高 PSNR/SSIM 宣称方法更好。

当前能检验的论点是：PAC 能在所测试的基础攻击上复用空间权重，使用零训练换色，并量化攻击变化；更强基础攻击是否保留优势以新矩阵为准。隐蔽性改善仍需要匹配对照和实际测量，不以适配成功代替隐蔽性证据。

## 新增复核记录

Faster R-CNN 源 T-SEA 在 YOLOv5s 上出现较强迁移。使用上一轮工作树的检测器后端独立复测，clean=95.702364、T-SEA=1.390074，与正式结果数值完全一致；对应同预算 Gray=42.780098。复测记录位于 audits/tsea_frcnn_previous_backend/validation.json。六检测器的 Gray 控制均已通过 clean 重放、图像与 GT 数量、权重和评测参数核验。原稿 YOLOv5s 的 Gray=85.30，因此共同协议的新结果须单独报告，不能直接与历史表组成受控排名。
