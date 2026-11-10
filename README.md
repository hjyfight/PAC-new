# PAC — physical adversarial patch experiments (YOLOv5)

Code and experiment protocols for reproducing and extending PAC-style physical
adversarial patch attacks against YOLO-family detectors. This repository is a
curated source snapshot: datasets, detector weights, training outputs and the
manuscript are intentionally **not** included.

## Layout

| Path | Contents |
|---|---|
| `experiments/ours/yolov5-master/` | Early YOLOv5 working tree used for the first PAC runs (large run outputs not archived) |
| `experiments/ours/yolov5-fixed-20260907/` | EOT, rotation-mask and linear recolor fixes; squash vs. letterbox comparison |
| `experiments/ours/yolov5-advpad-20260908/` | YOLOv5s AdvPatch (30.40% mAP50) and PAC-P/R/T code, patches and evaluations |
| `experiments/baselines/yolov5-advreal2d-20260908/` | AdvReal 2D adaptation, 800-epoch runs, white/black-box evaluation on six detectors |
| `experiments/baselines/yolov5-cap-rfatex-20260909/` | CAP source-specific 1000-epoch runs; RFA-Tex baseline |
| `experiments/ours/yolov5-advreal-recolor-20260910/` | AdvReal recolor variant and the complete evaluation matrix |
| `experiments/baselines/yolov5-advpatch-cap-20260911/` | AdvPatch + CAP completion experiments |
| `experiments/baselines/yolov5-tsea-nap-20260912/` | T-SEA / NAP baseline comparison |
| `experiments/` | [Experiment index](experiments/README.md): baselines, ours, NCQ and analysis; code, results and figures organized into real directories |
| `experiments/baselines/pnap_yolo_20261102/` | PNAP-YOLO (2025) author source, PAC protocol adapter, training/evaluation entry points, and audit of NCQ / pattern-color experiments |
| `experiments/analysis/pac_patch_visualizations_20260910/` | Patch PNGs, comparison figures and a static HTML gallery |
| `experiments/baselines/adversarial-yolo-master/` | Reference snapshot of the official AdvPatch / YOLOv2 code (third-party) |
| `experiments/baselines/literature_baselines_20260911/` | Survey notes and baseline metadata collected for the paper |

The local experiment code and outputs live under `experiments/`. Git worktree
metadata is kept inside this project; `.git-worktrees/` stores the imported
source repository metadata. Dataset/weight dependencies and Python environments
still need to be prepared when moving to another machine.

## Not included

The following are excluded by `.gitignore` and must be obtained separately:

- datasets (`INRIAPerson/`, `**/datasets/`, detector hub caches);
- detector and generator weights (`*.pt`, `*.pth`, `*.onnx`, `**/weights/`);
- training/evaluation outputs (`**/runs/`, `**/logs*/`, `**/tensors/`, W&B);
- the manuscript and paper figures (`article/`);
- local caches, virtual environments and archives.

YOLOv5 weights can be downloaded from the upstream release page; INRIAPerson is
available from its original distribution. Experiment configs and YAMLs that
reference those paths are kept in-tree.

## Reproducing

Each experiment directory contains its own driver scripts and protocol notes
(`PROTOCOL.md`, `report.md`, `manifest.json`). Typical entry points are
`train_patch.py`, `detect.py` and `val.py` inside an `experiments/ours/yolov5-*` or `experiments/baselines/yolov5-*` tree, and
`run_*.py` / `run_*.sh` under `experiments/`. Check the protocol file for the
exact seed, preprocessing and evaluation settings of each run.

## Third-party code

See `THIRD_PARTY_NOTICES.md`. Derived YOLOv5 code retains its AGPL-3.0 license
(see `LICENSE`). Other baseline sources remain under their original licenses.
