# TriPatch
This repository contains the source code for our paper "Transferable physical-world adversarial patches against pedestrian detection models".

## Abstract
Physical adversarial patch attacks pose serious risks to pedestrian detection in surveillance and autonomous driving. Existing methods face two main limitations: they lack systematic disruption of the multi-stage decision pipeline, allowing residual modules to offset perturbations, and they fail to model complex physical variations, resulting in poor robustness. To address these issues, we propose TriPatch, a pedestrian adversarial patch generation method that combines multi-stage collaborative attacks with robustness enhancements. TriPatch adopts a triplet loss consisting of detection confidence loss, bounding box loss and non-maximum suppression (NMS) loss, together with an appearance regularization and data augmentation to improve realism and physical robustness. Experiments demonstrate that TriPatch achieves higher attack success rates compared with existing approaches.

