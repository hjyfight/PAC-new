# PNAP-YOLO（2025）

已接入原版扩散潜变量优化，复用 PAC 的六模型检测器、300×300纹理、人框中心25%面积和完整评测流程。训练采用用户确认的作者源码500轮。论文、固定源码、算法细节及适配差异见 [PROTOCOL.md](PROTOCOL.md)。

## 最新核查：2026-11-08 14:19（北京时间）

六个源模型均已完成500轮训练（每源38500步），best/final六目标评估共72/72项全部完成，当前没有PNAP训练或评估进程。本次核验了训练记录、权重来源、评估协议、36次clean重放，并重算72项攻击结果的逐图ASR，全部通过。

PNAP-best的6WB/30BB平均mAP50分别为23.80%/41.37%，ASR为74.43%/43.76%；final分别为22.75%/40.81%和76.71%/44.67%。主比较继续使用训练损失选择的best。

- [完整六源结果与同协议方法比较](reports/20261108_141900/report.md)
- [逐源汇总CSV](reports/20261108_141900/source_summary.csv) / [72项结果CSV](reports/20261108_141900/completed_metrics.csv)
- [完整核验记录](reports/20261108_141900/audit.json)

## 历史核查：2026-11-03 19:03（北京时间）

YOLOv2、YOLOv5s、Faster R-CNN 均已完成500轮及best/final六目标评估，共36/72项攻击结果。已有评估的权重、参数、clean重放和逐图ASR计数全部核验通过，本次复用有效结果。按用户要求，仅汇总完成500轮的源，不评估在训中间权重。

该时刻YOLOv3完成438/500轮、YOLOv4完成309/500轮、YOLOv5m完成496/500轮，三个训练进程均存活；控制器将在500轮结束后自动评估并更新正式结果表。六源未齐，暂不计算整体6WB/30BB平均值。

- [完整进度及best/final结果报告](reports/20261103_190334/report.md)
- [逐单元结果CSV](reports/20261103_190334/completed_metrics.csv) / [源模型汇总CSV](reports/20261103_190334/source_summary.csv)
- [核验记录与权重来源](reports/20261103_190334/audit.json)

可从项目根目录再次运行 `experiments/baselines/pnap_yolo_20261102/.venv/bin/python -B experiments/baselines/pnap_yolo_20261102/summarize_completed.py`，生成新的时间戳报告；`reports/latest.json` 指向最新报告。此命令只核验和汇总已有完整评估，不启动训练或推理。

## 2026-11-02启动记录（历史）

2026-11-02：环境和完整权重校验通过；六模型检测损失梯度检查通过；YOLOv2向量化损失与作者原函数值/梯度误差均为0。YOLOv5s已通过2步真实扩散训练和288张测试图评测，clean重放精确一致。上述短程结果不用于正式比较。

**YOLOv5s源、GPU0、500轮** 正式训练持续运行。其余五源使用 **tmux 会话 `pnap_20261102`**，每源独立窗口，先做2步真实扩散训练验证，通过后自动进入500轮正式训练。每源训练结束后自动评测六个目标检测器的best/final并更新结果表。

**2026-11-02 20:39（北京时间）启动核查通过：六源均在正式训练，五个新增源的tmux窗口全部存活，GPU映射正确，每源只有一个正式训练进程。** 新增四个YOLO源已完成至少30次正式更新，Faster R-CNN进入第5轮，原YOLOv5s进入第24轮。此为启动状态，不是500轮完成结果。完整快照见 [evidence/startup_status.json](evidence/startup_status.json)。

**六源训练及评测的 `batch_size` 均为8，按用户论文要求保持一致。** CPU暂存没有拆小训练批次，也没有使用梯度累积。614张训练图每轮为76个完整8图批次加1个6图尾批，保留全部样本，共77次更新；288张评测图为36个完整8图批次。

| 源模型 | GPU | tmux窗口 | 训练入口 |
|---|---:|---|---|
| YOLOv5s | 0 | 原后台进程 | 原入口，继续运行 |
| YOLOv2 | 3 | `yolov2` | 检测器反传张量暂存CPU |
| YOLOv3 | 4 | `yolov3` | 检测器反传张量暂存CPU |
| YOLOv5m | 5 | `yolov5m` | 检测器反传张量暂存CPU |
| Faster R-CNN | 6 | `fasterrcnn` | 原入口 |
| YOLOv4 | 7 | `yolov4` | 检测器反传张量暂存CPU |

首次短程验证中 Faster R-CNN 通过，其余四源因共享GPU的可用显存不足而OOM。新增独立入口 `run_offloaded.py`，仅把检测器为反向传播保存的张量暂存到CPU固定页内存，保持batch=8、float32、损失、扩散步骤及500轮设置。四模型两张真实训练图的对照检查中损失完全一致，梯度最大绝对误差≤1.87e-9、相对L2误差≤6.27e-8；详见 `evidence/detector_offload_equivalence.json`。该检查是数值等价验证，不声称长程训练逐位一致。原训练文件和协议未改写；各源 `execution_profile.json` 记录新入口校验值。首次失败日志保留，新尝试追加到相同日志。

重新进行的真实扩散短程训练五源全部通过，检测损失到扩散潜变量的梯度非零且有限。PyTorch峰值分配显存：YOLOv2 9.828 GiB、YOLOv3 9.869 GiB、YOLOv5m 9.717 GiB、Faster R-CNN 12.638 GiB、YOLOv4 9.878 GiB；这些是各自进程的短程峰值，不含其他任务及CUDA上下文/缓存。

- 启动信息：原YOLOv5s为 [launch.json](launch.json)，新增源为 `launch_<源>_gpu<编号>.json`，记录PID、tmux窗格和显存配置；实时状态：`jobs/<源>.json`、`sources/<源>/status.json`。
- 正式日志：`logs/<源>_train.log`；短程日志：`logs/<源>_smoke.log`；新增控制器日志：`logs/controller_<源>_gpu<编号>.log`；后续评测日志位于 `sources/<源>/logs/`。
- 主对比表：[comparison_best_with_pnap.csv](tables/comparison_best_with_pnap.csv)。引入既有共同协议方法，追加 PNAP；缺失值留空，完整6源结果前不计算6WB/30BB平均值。
- PNAP逐单元结果：[comparison.csv](tables/comparison.csv)；机器状态：[results_status.json](results_status.json)。
- 验证记录：`preflight.json`、`evidence/detector_validation_all.json`、`evidence/yolov2_author_equivalence.json`、`smoke/yolov5s/smoke_complete.json`、`smoke/yolov5s/evaluation_complete.json`。

## 两个最高优先级实验

详见 [实验核查报告](existing_experiments_report.md)。原始日志与294份指标已复核；NCQ/PAC对照和十组图案×配色控制均有完成结果，但结论有明确适用范围。

## 恢复或扩展运行

在 PAC-yolo 根目录运行。当前任务仍在运行时不要重复启动。训练从每轮保存的状态恢复，代码或协议变更后拒绝混用旧断点。命名控制器允许不同源并行，并拒绝重复源；CPU暂存入口还检查独立执行配置。短程结果与正式训练分别保存，正式训练从原种子重新开始。

```bash
# 查看已有会话；Ctrl-b d 脱离，不会停止训练。
tmux attach -t pnap_20261102
tmux list-windows -t pnap_20261102
# 直接跟踪单源训练。
tail -f experiments/baselines/pnap_yolo_20261102/logs/yolov2_train.log
# 初次启动五源用此脚本；已有会话时默认拒绝重复启动。
bash experiments/baselines/pnap_yolo_20261102/start_remaining_tmux.sh
```

若正式训练中断，先确认对应控制器和训练进程均已退出，再在该tmux窗口运行原 `launch_<源>_gpu<编号>.json` 中的命令，并将 `--smoke-steps 2` 改为 `--smoke-steps 0`，使用已通过的短程验证和轮末断点。CPU暂存源须保留 `--offload-detector`，启动环境保留 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。不要删除短程验证或正式断点来强行重跑。

所有依赖、缓存和任务文件均在本项目目录内。独立环境在 `.venv/`，原 `pac` 环境保留用于正式评测。
