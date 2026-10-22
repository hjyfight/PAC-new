# UPA-RFAS

Official code repository for our CVPR 2026 accepted paper
["When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models"](https://arxiv.org/abs/2511.21192).

![Main Figure](figs/mainfig.png)

---

UPA-RFAS is a research code release for patch-based adversarial attacks against vision-language-action policies, with an OpenVLA / OpenVLA-OFT evaluation pipeline on LIBERO.

This repository contains:

- adversarial patch optimization code
- OpenVLA-OFT evaluation scripts for LIBERO
- released adversarial patches
- demo videos for simulated and physical settings


## Demo Videos

GitHub renders GIF previews directly in the repository homepage. Click any preview below to open the corresponding MP4 file.

### Benign

<table>
  <tr>
    <td align="center"><a href="videos/benign_1.mp4"><img src="videos/benign_1.gif" alt="benign_1" width="240"></a></td>
    <td align="center"><a href="videos/benign_2.mp4"><img src="videos/benign_2.gif" alt="benign_2" width="240"></a></td>
    <td align="center"><a href="videos/benign_3.mp4"><img src="videos/benign_3.gif" alt="benign_3" width="240"></a></td>
  </tr>
</table>

### Real-world Attacks

<table>
  <tr>
    <td align="center"><a href="videos/our_patch_1.mp4"><img src="videos/our_patch_1.gif" alt="our_patch_1" width="240"></a></td>
    <td align="center"><a href="videos/our_patch_2.mp4"><img src="videos/our_patch_2.gif" alt="our_patch_2" width="240"></a></td>
    <td align="center"><a href="videos/our_patch_3.mp4"><img src="videos/our_patch_3.gif" alt="our_patch_3" width="240"></a></td>
  </tr>
</table>

### Baseline

<table>
  <tr>
    <td align="center"><a href="videos/baseline--openvla_oft--task=LIBERO-long.mp4"><img src="videos/baseline--openvla_oft--task=LIBERO-long.gif" alt="baseline--openvla_oft--task=LIBERO-long" width="320"></a></td>
  </tr>
</table>

### Attacks in Simulated setting

<table>
  <tr>
    <td align="center"><a href="videos/our_simulated--openvla_oft--task=LIBERO-long.mp4"><img src="videos/our_simulated--openvla_oft--task=LIBERO-long.gif" alt="our_simulated--openvla_oft--task=LIBERO-long" width="320"></a></td>
    <td align="center"><a href="videos/our_simulated--openvla_oft--task=LIBERO-spatial.mp4"><img src="videos/our_simulated--openvla_oft--task=LIBERO-spatial.gif" alt="our_simulated--openvla_oft--task=LIBERO-spatial" width="320"></a></td>
  </tr>
</table>

### Attacks in physical setting

<table>
  <tr>
    <td align="center"><a href="videos/our_physical--openvla_oft--task=LIBERO-spatial.mp4"><img src="videos/our_physical--openvla_oft--task=LIBERO-spatial.gif" alt="our_physical--openvla_oft--task=LIBERO-spatial" width="320"></a></td>
    <td align="center"><a href="videos/our_physical_openvla_oft--task=LIBERO-long.mp4"><img src="videos/our_physical_openvla_oft--task=LIBERO-long.gif" alt="our_physical_openvla_oft--task=LIBERO-long" width="320"></a></td>
  </tr>
</table>


## Project Structure

```text
.
├── adversarial_patches/      # released patch checkpoints used for evaluation
├── figs/                     # figures for the project page / README
├── openvla-oft/              # OpenVLA-OFT-based evaluation code
├── videos/                   # demo videos shown in the README
├── white_patch/              # attack and patch optimization modules
├── trans.py                  # training entry for adversarial patch optimization
└── sbatch_train.sh           # example training script
```

## Requirements

This codebase is built on top of [roboticAttack](https://github.com/William-wAng618/roboticAttack) and [openvla-oft](https://github.com/moojink/openvla-oft).

Please prepare the environment by following those upstream repositories first, especially:

- Python / PyTorch / CUDA setup from `openvla-oft`
- LIBERO installation
- OpenVLA or OpenVLA-OFT checkpoint access
- optional `dlimp` dependency if your local setup requires it


## Released Patches

The following patch checkpoints are included in this repository:

- `adversarial_patches/patch_sim_v1.pt`
- `adversarial_patches/patch_sim_v2.pt`
- `adversarial_patches/patch_phy.pt`

They can be passed directly to `--patch_path` during evaluation.

## Training

`trans.py` is the main entry for adversarial patch optimization.

Example:

```bash
bash sbatch_train.sh
```

The provided script is a cleaned slurm-based public template. Adjust the dataset path, environment name, and cluster options to match your machine.

Training data is resolved from either:

- `--server /path/to/project_root` and then `${server}/openvla-main/dataset`
- or the environment variable `OPENVLA_DATA_ROOT`

## Evaluation on LIBERO

We provide a concise OpenVLA-OFT evaluation script in `openvla-oft/sbatch_test.sh`.

Example:

```bash
cd openvla-oft
bash sbatch_test.sh
```

The script evaluates released adversarial patches on multiple LIBERO task suites with OpenVLA-OFT checkpoints from Hugging Face, and the default `patch_path` now points to files under `adversarial_patches/`.

For a single run, you can also call the evaluator directly:

```bash
cd openvla-oft
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --patch_path ../adversarial_patches/patch_sim_v1.pt
```

## Acknowledgment

This repository builds upon:

- [roboticAttack](https://github.com/William-wAng618/roboticAttack)
- [openvla-oft](https://github.com/moojink/openvla-oft)
- [OpenVLA](https://github.com/openvla/openvla)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

We thank the original authors for open-sourcing their code and models.

## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{lu2025when,
  title={When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models},
  author={Lu, Hui and Yu, Yi and Yang, Yiming and Yi, Chenyu and Zhang, Qixin and Shen, Bingquan and Kot, Alex C. and Jiang, Xudong},
  journal={arXiv preprint arXiv:2511.21192},
  year={2025}
}
```
