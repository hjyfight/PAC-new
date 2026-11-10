# Experiment protocol

- New AdvPatch sources: YOLOv2, YOLOv3, YOLOv4, YOLOv5m, Faster R-CNN. Existing YOLOv5s s000 is unchanged.
- AdvPatch: 200 epochs, seed 1788272031, batch 8, Adam AMSGrad LR 0.03, ReduceLROnPlateau patience 50, NPS 0.01, max(2.5*TV,0.1). No early stopping.
- EOT and RGB parameterization exactly reuse the selected s000 path; median7, additive brightness/noise, rotations, matched coverage.
- YOLO objective: mean per-image peak objectness*person probability. Faster R-CNN adaptation: mean per-image peak pre-NMS ROI softmax person probability. This detector-family extension is explicit, not claimed as an original Thys FRCNN implementation.
- Dataset: fixed INRIA train614/test288; test has589 person GT. Content hashes show no train/test image overlap.
- All images: PIL RGB, original-resolution centered RGB127 square padding, then PIL bilinear640.
- Patch: 300x300 texture, square centered on each GT person, side round(sqrt(0.25*bbox_width*bbox_height)). Shared clipping and overlap rules.
- Evaluation: no EOT, conf0.001, NMS IoU0.6, max_det300, batch8. Conditional disappearance ASR uses clean detected GT at conf0.5/IoU0.5.
- Best selection: minimum epoch-mean total training loss for AdvPatch; existing minimum CAP patch-phase training loss for CAP. Test metrics never select checkpoints.
- Recolor formula: out[h,w,c]=sum_k raw[k,h,w]*base_colors_255[k,c]/255. No channel normalization, softmax, clustering, brightness matching, or retraining.
- Names: AdvPatch recolors PAC-P1/P2; CAP recolors PAC(CAP,Bc1/Bc2), separate from original random PAC-R1/R2 and existing AdvReal recolors.
- New evaluations: AdvPatch90 attack cells + CAP recolor72 attack cells +36 clean replays. Reused references: selected YOLOv5s18 cells and CAP raw36 cells. Complete output: six6x6 attack matrices,216 cells.
- White-box averages require all six diagonal cells; black-box averages require all30 off-diagonal cells. Missing values are not zero-filled.
- One process per assigned GPU, no distributed/multi-GPU training. Existing compute jobs are excluded from scheduling.
- PSNR/SSIM and the manuscript are unchanged.
- Test status:10 available unit tests passed; one external-reference AdvReal TV test was skipped because its author loss.py is absent. The shared TV/EOT/placement/evaluator files are unchanged from the existing CAP worktree.
- All task outputs and caches are under PAC-yolo. TMPDIR is the short PAC-yolo/.tmp_ac11 path to satisfy Unix socket path limits.
