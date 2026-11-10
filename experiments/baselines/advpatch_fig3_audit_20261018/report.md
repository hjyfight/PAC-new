# Fig. 3 与当前 AdvPatch 的来源核查

核查日期：2026-10-18。用户确认只有论文截图，没有 Fig. 3 所用的原始补丁 PNG、张量或训练目录。

## 已确认的事实

1. 当前稿 `article/cas-sc-sample.tex:461` 引用 `article/images/main/patches_01.png`，其内容与用户提供的 Fig. 3 一致。
2. 五图 PSNR/SSIM 试测使用独立重训产物：`experiments/ours/yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_advpatch/best.pt`，SHA256 为 `bb27f8379b9b31f5bf7e02de674bdb0cf9a71b28deb795870ff73f42bcf385c3`。
3. 该重训运行的 seed 为 1788272031；训练 200 轮，batch 8，Adam AMSGrad，初始 LR 0.03，300×300 补丁，640 输入，人框中心贴图、面积为人框面积的 25%。best 按 epoch 平均训练总损失选择；日志最低点为第 199 轮。
4. 已独立核验 `experiments/analysis/pac_patch_visualizations_20260910/patches/yolov5s_advpatch.png` 与该 checkpoint 的 `round(RGB*255)` 完全一致，且五图试测 manifest 引用同一 checkpoint 哈希。因此当前展示确实对应实际测量补丁。
5. Fig. 3 与当前补丁的空间图案明显不同。已有另外两个重训种子 1788272032/1788272033 的图案也不同，最低训练总损失均出现在第 200 轮。
6. 当前稿给出的 200 轮、batch 8、LR 0.03、640 输入、25% 人框面积等主要条件与重训配置一致；Fig. 3 的原始随机种子、确切检测器权重、训练损失选项、EOT/预处理细节及 checkpoint 选择尚不能追溯。不能把未知条件写成已确认不一致。

展示对照见 `fig3_vs_retrained_seeds.png`。第一列只是从论文拼图裁出的显示区域，已经缩放，仅供目视比较；不作为原始实验纹理，也没有用于 PSNR/SSIM 或检测性能评价。来源坐标和哈希见 `provenance.json`，渲染验证见 `render_check.json`。

## 攻击目标存在需要核对的选项

当前 YOLOv5s 重训路径在 `experiments/ours/yolov5-advpad-20260908/utils/patch.py:314` 使用：

```python
(pred[..., 4] * pred[..., 5 + cls]).amax(dim=1).mean()
```

即逐图最大 `objectness × person probability` 后取 batch 平均。

仓库中 Thys 原作者代码 `experiments/baselines/adversarial-yolo-master/patch_config.py` 同时含两种选项：

- `BaseConfig`：`obj * cls`。
- `ReproducePaperObj`（README 推荐的 `paper_obj`）：仅 `obj`。

原作者这一仓库针对 YOLOv2，不等同于 PAC 论文 Fig. 3 的 YOLOv5s 实验。上述对照说明“AdvPatch”名称下有不同优化目标，值得核查；**它不能证明 Fig. 3 使用了 objectness-only，也不能证明目前实现一定错误**。当前实现还包含 NPS、TV 正则与 EOT；这些细节均会影响最终图案。

## 对旧 PSNR/SSIM 表的影响

五图试测的结论应限定为：“当前重训 donor 在五张固定随机测试图上的配对外观结果”。它不是对 Fig. 3 原补丁的重放，也不是旧表中 `14.82/0.43` 和 `24.16/0.78` 的同条件复算。

目前也没有建立 Fig. 3 显示纹理与旧 PSNR/SSIM 表原始输入之间的直接关联。图案不同、评价区域不同、补丁/色板不同与样本不同均可能改变指标；不能把差异只归因于其中一项，更不能凭五图否定旧表。

## 当前应采用的做法

- 将当前实验明确标为统一协议下的 AdvPatch 重训/适配结果；按实际 checkpoint 生成对应的 PAC-P1/P2，联合评价攻击与外观。
- 复现方法应核对检测目标、检测器权重、正则项、EOT、预处理及面积定义；不以“看起来像截图”或“数值接近旧表”选择训练设置或随机种子。
- 若继续使用 Fig. 3，只能保留其原有示意/历史实验身份，不将当前新指标附到它上面。修订稿若要让图像、数值和代码形成完整对应，应使用新实验的实际补丁与相应数据。
- 原补丁与原记录不可获得时，旧表仍处于不可核验状态；正式修订可以用完整、可追溯的新评价替换。五图试测本身不足以替代全测试集表。
- 不从论文截图裁切、重采样后充当原始补丁去声称复现原指标；这样会引入未知插值、量化和图像排版处理。

另：Fig. 3 的 (b)、(c) 都标为 PAC-P1；正文对两套色板的命名是 PAC-P1/P2，应在最终确认图像对应关系后修正 (c) 标签。本轮没有修改论文图或正文，也没有启动重训。
