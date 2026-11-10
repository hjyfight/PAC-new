# 用户工作目录约定

- 用户明确要求：不要在 home 目录（包括 `/home/huotianlin`）创建或写入任务文件。
- 项目代码、脚本、下载文件、图表、报告、日志及临时任务产物，均放在对应项目目录下。
- PAC-yolo 任务的产物根目录为 `/datacc05/huotianlin/code-hu/PAC-yolo`，按需在其下建立独立任务目录。
- 工具若默认把任务输出或缓存写到 home，应将相应输出或缓存目录改到项目下；不要改写 HOME 或 CODEX_HOME。
- 用户已要求并完成将此次补丁可视化迁移至 `experiments/analysis/pac_patch_visualizations_20260910/`。
