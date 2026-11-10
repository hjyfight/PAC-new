# Breaking Rigidity in Adversarial Patch Attacks (ECCV2026)

### SmuDPatch — Smooth Deformable Universal Adversarial Patches 

> Official code release for the paper **"Breaking Rigidity in Adversarial Patch Attacks"** (ECCV 2026).


[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/Breaking_Rigidity_in_Adversarial_Patch_Attacks.pdf?raw=1)

**➡️ [Download the ECCV 2026 Paper](paper/Breaking_Rigidity_in_Adversarial_Patch_Attacks.pdf?raw=1)**


Most existing adversarial patch attacks are restricted to **fixed shapes** (squares or circles). This rigidity is something current patch detectors — including Vision-Language Models (VLMs) — implicitly exploit, giving a *false sense of security*. We propose a **universal optimization framework** that generates **smooth, deformable** adversarial patches whose shape is searched with a **centripetal Catmull–Rom spline** and an **evolutionary shape optimizer**, then textured to attack an ensemble of models. The resulting **SmuDPatch** patches strongly fool state-of-the-art image classifiers and slip past recent patch detectors far more often than fixed-shape patches.

---

## 🖼️ Method Overview

<p align="center">
  <img src="../../../../../../../../assets/figures/main_fig.png" alt="SmuDPatch framework: shape search via centripetal Catmull-Rom spline + evolutionary optimization, followed by patch texture optimization over a model ensemble with affine augmentation." width="85%">
</p>


## Comparison Between Fixed and SmuDPatch Shape Patches on VLM
<p align="center">
  <img src="../../../../../../../../assets/figures/ff1.png" alt="Fixed-shape vs. deformable patches and how VLMs / patch detectors respond." width="85%">
</p>


---

## ✨ Highlights

- **Universal deformable patches.** A single optimized patch per target class is applied across many images — not optimized per-image.
- **Smooth shapes.** Anchor points are interpolated with a **centripetal Catmull–Rom spline**, avoiding the jagged, self-intersecting boundaries of prior deformable methods.
- **Area-matched fairness.** Every shape is normalized to a fixed target area (default **2500 px ≈ 5%** of a 224×224 image; **4000/5000 px** variants ≈ 8% / 10% also provided), so *shape*, not area, drives the attack.
- **Evolutionary shape search.** A non-differentiable pipeline (spline sampling, rasterization, thresholding) is optimized with an evolutionary algorithm (population 30, 20 generations).
- **Robust to affine transforms.** Random rotation / translation / scale are applied during texture optimization for real-world stability.
- **Two large-scale benchmarks.** ImageNet-based **SmuDPatch** dataset and a COCO-based **out-of-distribution** detection dataset — 140,000 adversarial patch images in total.

---

## 📂 Repository Layout

```
SmuDPatch/
├── README.md
├── LICENSE                              # CC BY 4.0
├── requirements.txt
│
├── deformable_patch_working_catmull.py  # ① generate the deformable patches (shape search + texture opt.)
├── gz_create.py                         # ② pack patches/masks into a .gz archive  (PLACEHOLDER — author to add)
├── apply_patches.py                     # ③ build the SmuDPatch-ImageNet dataset (deformable)
├── apply_circular_patch.py              #    build the circular-patch variant of the dataset
├── evaluate.py                          # ④ evaluate classifiers on the generated dataset
├── SmuDPatch_eval.ipynb                 # ⑤ one-shot: apply patches + evaluate, no dataset saved to disk
│
├── transforms/                          # patch-application math (DO NOT EDIT — reproduces paper numbers)
│   ├── apply_patch_deformable.py        #    ApplyPatch for deformable, mask-based patches
│   ├── apply_patch.py                   #    ApplyPatch for square / circular patches
│   └── my_random_affine.py              #    affine applied jointly to patch + mask
│
├── utils/
│   ├── utils.py                         # set_all_seed, target_transforms, normalization constants
│   └── visualization.py                 # preview helpers
│
├── patches/                             # pre-generated patches + their binary masks (10 classes each)
│   ├── deformable_patch_2500/           #   the 10 SmuDPatch patches  (≈5% area)   0.png … 9.png
│   ├── deformable_mask_2500/            #   matching binary masks                  0.png … 9.png
│   ├── circular_patch_2500/             #   circular baseline patches
│   ├── deformable_patch_4000/  + _mask  #   ≈8%  area variant
│   └── deformable_patch_5000/  + _mask  #   ≈10% area variant
│
├── OOD_Detection/                       # ⑥ out-of-distribution patch-detection benchmark (real vs patch)
│   ├── train_detector.py                #    fine-tune a binary detector per patch
│   ├── evaluate_detector.py             #    cross-patch (OOD) evaluation -> accuracy tables
│   ├── checkpoints/                     #    downloaded / fine-tuned detector .pth files
│   └── README.md                        #    detailed OOD-benchmark instructions
│
├── assets/
│   ├── gz/                              # packed patch tensors used by the scripts/notebook
│   │   ├── SmuDPatch_2500.gz            #   deformable patches (2500 px)
│   │   ├── Circular_patch_2500.gz       #   circular patches
│   │   └── Square_imagenet_patch.gz     #   square baseline (ImageNet-Patch style)
│   ├── data/                            # tiny sample ImageFolder for a quick smoke test
│   ├── figures/                         # << put main_fig.png / ff1.png here >>
│   └── imagenet1000_clsidx_to_labels.txt
│
└── results/
    ├── results_smudpatch.json           # reference ResNet-50 numbers (deformable)
    └── results_circular.json            # reference ResNet-50 numbers (circular)
```

> **Note on imports:** the runnable scripts and the notebook import the local `transforms/` and `utils/` packages, so **run everything from the repository root**.

---

## 📥 Datasets & Downloads

The full datasets (optimization images, the 5,000-image ImageNet test subset, the COCO out-of-distribution split, and the large generated patch datasets) are hosted on Google Drive:

**➡️ [Download datasets](https://drive.google.com/drive/u/0/folders/1D3GCvk8UKm1YkS4MNGffl-dPqSpchf_y)**

A small sample ImageFolder is bundled under `assets/data/` so you can run the pipeline end-to-end immediately without any download.

### Dataset summary (from the paper)

| Dataset | Source | Composition | Size |
|---|---|---|---|
| **SmuDPatch-ImageNet** | ImageNet val | 5,000 test images × 10 target classes × 2 shapes (deformable + circular) | **100,000** patched images |
| **Out-of-Distribution (COCO)** | COCO val | 2,000 clean images × 20 patches + 2,000 clean; 3:2 train/test split | **40,000** patched images |

**10 target classes (patch-0 … patch-9):** soap dispenser (804), cornet (513), plate (923), banana (954), cup (968), typewriter keyboard (878), electric guitar (546), hair spray (585), sock (806), cellular phone (487).

---

## ⚙️ Installation

```bash
git clone <your-repo-url> SmuDPatch
cd SmuDPatch

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Developed and tested with PyTorch on an **NVIDIA A6000 (48 GB)**.

---

## 🚀 Workflow

The attack pipeline has five stages. Stages ③–⑤ work out of the box on the bundled `assets/data/` sample. Stage ⑥ is the detection benchmark.

### ① Generate the deformable patches

Runs the shape search (Catmull–Rom spline + evolutionary optimization) and then optimizes the patch texture against the model ensemble (ViT-B/16 + ResNet-50 + MobileNetV2).

```bash
python3 deformable_patch_working_catmull.py
```

> Paths to the ImageNet validation images and the synset-mapping file are set near the top of the script — update them to your local ImageNet location before running. The generated patches and masks are what you find pre-computed under `patches/`.

### ② Pack patches into a `.gz` archive

Use `gz_create.py` to bundle the per-class patch images (and masks, for the deformable variant) into the `.gz` format consumed by the apply/eval scripts.

```bash
python3 gz_create.py
```

### ③ Build the SmuDPatch-ImageNet dataset

Applies the 10 deformable patches to the source images and writes **one ready-to-evaluate copy of the dataset per patch**.

```bash
python3 apply_patches.py \
    --patch-file  assets/gz/SmuDPatch_2500.gz \
    --mask-dir    patches/deformable_mask_2500 \
    --data-dir    assets/data \
    --output-root generated_dataset \
    --format      pt
```

**Output**

```
generated_dataset/
├── patch_00/data/<class>/<image>.pt   # adversarial image, lossless float [0,1], (C,H,W)
├── patch_01/ ...
├── ...
├── patch_09/...
└── manifest.json                      # params, targets, per-image labels, per-patch checksums
```

Circular variant: use `apply_circular_patch.py` with `--patch-file assets/gz/Circular_patch_2500.gz` (no mask dir needed).

**Reproducibility** below.

### ④ Evaluate classifiers

Loads the per-patch folders and scores any torchvision model, reporting **clean accuracy**, **per-patch robust accuracy**, **per-patch attack-success rate (ASR)**, and pooled aggregates.

```bash
python3 evaluate.py \
    --output-root generated_dataset \
    --models      resnet50 \
    --results-json results/my_run.json
```


**Metric definitions** (unchanged from the paper):
`robust_accuracy = 100 · correct_adv / N`, `success_rate = 100 · (pred == patch_target) / N`, aggregates pooled over all patches.

### ⑤ One-shot apply + evaluate (no dataset saved)

`SmuDPatch_eval.ipynb` applies the patches and scores the models **in memory**, without writing the patched dataset to disk — handy for quick experiments.

```bash
jupyter notebook SmuDPatch_eval.ipynb
```

### ⑥ Out-of-Distribution patch detection (binary classification benchmark)

The `OOD_Detection/` folder holds the **real vs patch** detection benchmark: a
detector is fine-tuned on **one** patch type and evaluated on **all 10**,
measuring generalization to **unseen / out-of-distribution** patch shapes. You
can either **download released detector checkpoints** or **fine-tune them
yourself**, then run the cross-patch evaluation.

**Expected dataset layout** (one subfolder per patch, each split into
`train/`/`test/` with two classes — *real* vs *patch*):

```
<DATA_DIR>/
├── 0/{train,test}/{real,patch}/...
├── 1/{train,test}/{real,patch}/...
├── ...
└── 9/{train,test}/{real,patch}/...
```

Download the OOD detection datasets from the [Drive link](https://drive.google.com/drive/u/0/folders/1D3GCvk8UKm1YkS4MNGffl-dPqSpchf_y) above (or build them following the paper).

**Before running**, open both scripts and set the variables at the top — keep
`DATA_DIR`, `FINETUNED_DIR`, and the active `model_modification_funcs` dict
**consistent across the two files** (see `OOD_Detection/README.md` for the full
note; the shipped defaults intentionally differ and must be aligned).

**Option A — download checkpoints**

Drop released detector checkpoints into `OOD_Detection/checkpoints/` preserving
the `<ModelName>/<patch>/model.pth` structure, set `FINETUNED_DIR="checkpoints"`
in both scripts, then go straight to evaluation.

**Option B — fine-tune the detectors**

```bash
cd OOD_Detection
python3 train_detector.py        # one binary detector per patch (0..9) per model
```

**Evaluate (cross-patch / OOD generalization)**

```bash
python3 evaluate_detector.py     # train-patch x test-patch accuracy matrix
```

Outputs an `.xlsx` (one sheet per model) and a `.csv`. Off-diagonal entries
(train patch ≠ test patch) are the out-of-distribution scores. Full details in
[`OOD_Detection/README.md`](OOD_Detection/README.md).

---

## 🔁 Reproducibility

The patched images are produced by a **random affine** and the source loader uses `shuffle=True`, all drawn from the **global PyTorch RNG**. The final numbers therefore depend on the exact order in which random numbers are consumed.

- `apply_patches.py` defaults to `--replay-notebook-rng` (toggle off with `--no-replay-notebook-rng`) to replay the original notebook's RNG sequence in order.
- `--replay-model resnet50` (default) rebuilds the model so the RNG lines up bit-for-bit with the paper's ResNet-50 run. `weights=None` is used purely to advance the RNG; no download is required.
- `--replay-model none` produces a **clean, model-independent** dataset — identical for every model you later evaluate. **Recommended for the public multi-model benchmark**; numbers land within a tiny epsilon of the paper.

**Pretrained weights matter.** The original code used `models.resnet50(pretrained=True)` → the legacy **`IMAGENET1K_V1`** weights (top-1 ≈ 76.1%). Modern torchvision `weights="DEFAULT"` resolves to **`IMAGENET1K_V2`** (top-1 ≈ 80.9%), a *different* model. `evaluate.py` defaults to `--weights IMAGENET1K_V1` to reproduce the paper's clean accuracy (≈ 76.7%).

`manifest.json` records a `norm_adv_sha256` per patch — matching checksums across two runs mean identical images were produced.

> Hyperparameters in the code may differ slightly from the paper in places, due to different evaluation settings. **Feel free to set the hyperparameters reported in the paper** for exact-match runs.

---

## 📊 Reference Results (ResNet-50)

ResNet-50, `IMAGENET1K_V1` weights, clean accuracy **76.74%** (full per-patch numbers in `results/`):

| Patch set                       | Overall Robust Acc. ↓ | Overall ASR ↑ |
|---------------------------------|:---------------------:|:-------------:|
| **SmuDPatch** (deformable)      | **28.39%**            | **62.11%**    |
| Circular baseline               | 49.81%                | 27.74%        |

Lower robust accuracy and higher ASR ⇒ a stronger attack. Across the paper's 15-model benchmark, the deformable patches consistently outperform fixed-shape patches on standard, adversarially-robust, augmentation, and more-data model groups, and transfer to unseen architectures.

---

## 📜 License

Released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license — see [`LICENSE`](LICENSE). Dataset images derive from ImageNet and COCO and are subject to their respective terms.

---

## 📝 Citation

If you use this code, the patches, or the datasets, please cite:

```bibtex
@inproceedings{smudpatch_eccv2026,
  title     = {Breaking Rigidity in Adversarial Patch Attacks},
  author    = {Vishesh Kumar, Guha Balakrishnan, Akshay Agarwal},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
