# 实验目录索引

所有分类下的实验与代码均为实际目录。根目录代码已于 2026-11-10 迁入对应分类；旧目录入口软链接已移除。

| 分类 | 内容 |
|---|---|
| [ours](ours/README.md) | YOLOv5s PAC-P/R/T、PAC 原始与修正版、AdvReal 换色代码及结果 |
| [baselines](baselines/README.md) | 对比方法代码、实验、作者参考实现和文献资料 |
| [NCQ](NCQ/README.md) | NCQ、随机色板与结构置换实验 |
| [analysis](analysis/README.md) | 外观分析、PSNR / SSIM / LPIPS 与补丁图 |

同一实验的原方法与 PAC 配对结果仍共同保存，通过各分类 README 中的普通链接访问。
代码目录中的 runs/、权重和训练产物随目录一起移动；压缩包位于项目根目录 archives/。

迁移清单与核验结果见 [layout.json](layout.json)。路径字段和受影响的文件校验值已更新；
数值结果及模型文件保持不变。原始脚本、配置与记录备份在 .tmp_root_organization_20261110/before/。

项目原有的数据集、外部权重链接仍存在，运行环境也依赖本机安装；复制到另一台机器仍需准备这些依赖。
本次新增的目录兼容软链接已全部移除。
