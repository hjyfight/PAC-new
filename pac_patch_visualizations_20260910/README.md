# 已评估补丁可视化（best）

- `gallery.html`：可离线打开，按训练来源筛选，点击放大并下载原始 PNG。
- `yolov5s_all_methods.png`：共同 YOLOv5s 来源的全部七种方法。第一行是 AdvPatch、PAC-P1、PAC-P2、CAP；第二行是 AdvReal、PAC-R1 (AdvReal)、PAC-R2 (AdvReal) 及色板。
- `all_sources_cap_advreal_recolors.png`：六个来源，列依次为 CAP、AdvReal、PAC-R1 (AdvReal)、PAC-R2 (AdvReal)。
- `patches/`：27 张单独的 300×300 RGB PNG。
- `manifest.json`：原始 checkpoint 路径与哈希、白盒结果、颜色范围及数值核验。

原始 PAC-P1/P2 来自 AdvPatch；PAC-R1/R2 (AdvReal) 来自 AdvReal，不是论文中的随机 PAC-R 控制组。
CAP 和 AdvReal 使用既有训练损失选定的 best。CAP / Faster R-CNN 的 best 为第 3 轮，接近纯色；此图保留该实际评估版本。
AdvPatch、AdvReal、CAP 的源训练预算分别为 200、800、1000 轮。换色不重新训练，展示未应用 EOT 或 ISP。
图片来自评估使用的精确 RGB 张量。所有单图按 round(RGB×255) 导出，未归一化、调亮或调对比度。
总览 PNG 中补丁按原生 300×300 粘贴；预览 JPEG 仅供屏幕浏览，论文和后续分析请使用 PNG。
