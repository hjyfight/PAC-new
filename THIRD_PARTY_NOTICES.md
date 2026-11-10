# Third-party sources

This repository archives research code and experiment records; it is not an official release of the reference methods.

- PNAP-YOLO (Li et al., Annals of Data Science 2025): https://github.com/Kwizii/PNAP — Apache-2.0 author snapshot in `experiments/baselines/pnap_yolo_20261102/author/`, commit `dc1b1f025f3492de61e6dd1890e11d116f9fc649`. Its bundled YOLO-derived files retain their own upstream notices. Local adapters are documented separately in that experiment's protocol.

- YOLOv5: https://github.com/ultralytics/yolov5 — derived code retains its AGPL-3.0 license and copyright notices.
- AdvPatch (Thys et al., CVPRW 2019): https://gitlab.com/EAVISE/adversarial-yolo — reference snapshot in `experiments/baselines/adversarial-yolo-master/`.
- T-SEA (CVPR 2023): https://github.com/VDIGPKU/T-SEA — detector source subset in `external/T-SEA-pac/`; upstream commit and local changes are recorded in the snapshot manifest.
- CAP: https://github.com/weihui1308/CAP — source subset in `external/CAP_neurips2024/`, including the ISP network used by the experiments.
- AdvReal: local author source was maintained separately in `/datacc05/huotianlin/code-hu/PAC/baseline/AdvReal-main`; the 2D adaptation used here is preserved in the dated experiment directories and their manifests.

Original notices and available license files are retained. An absent upstream license is not replaced by a new license here. Detector weights and datasets remain separate downloads. Manuscript and figure rights remain with their respective authors.
