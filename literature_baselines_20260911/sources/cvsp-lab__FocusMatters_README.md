# Focus Matters: Phas-Aware Suppression for Hallucination in Vision-Language Models

[🌐 Project Page](https://paper.pnu-cvsp.com/FocusMatters/)

> This code reproduces the results reported in Table 1 of the main paper for LLaVA-1.5-7B, covering both the Origin baseline and our proposed DPP masking method. It includes evaluation scripts for CHAIR (CHAIR_I / CHAIR_S) and POPE (Random, Popular, and Adversarial settings) on the COCO val2014 dataset.

## Environment Setup

```bash
# Create conda environment from the provided file
conda env create -f environment.yml
conda activate focusmatters
```

**Requirements:** Python 3.10, PyTorch 2.1.0, CUDA-capable GPU

## Data Preparation

### MSCOCO 2014 Validation Set
- Download the **2014 Val images** from the [MSCOCO website](https://cocodataset.org/#download)
- Extract the downloaded archive and place the images into `dataset/val2014/`


```
dataset/
└── val2014/       # COCO 2014 validation images
```

## Model Weights

LLaVA-1.5 weights are automatically downloaded from HuggingFace at first run.

So, no manual download is required.

## Running Evaluations

### Masking modes

- `origin` — No masking (baseline)
- `ours` — DPP masking (proposed method)


### CHAIR
- From the `focusmatters/` :
```bash
# For Origin
bash evaluation/run_chair.sh --masking origin

# For Ours (DPP masking)
bash evaluation/run_chair.sh --masking ours
```

### POPE
- From the `focusmatters/` :
```bash
# For Origin
bash evaluation/run_pope.sh --masking origin

# For Ours (DPP masking)
bash evaluation/run_pope.sh --masking ours
```

## Output Structure

Results are saved to `results/{chair,pope}/<date>/<experiment_name>/`:


### CHAIR output
```
results/chair/<date>/<exp_name>/
├── config.json                # Experiment configuration
├── captions.jsonl             # Generated captions (one per line)
├── chair_metric_results.json  # CHAIR-i, CHAIR-s metrics
└── captions_eval_results.json # Per-caption evaluation details
```

### POPE output
```
results/pope/<date>/<exp_name>/
├── config.json
├── captions_pope_random.jsonl      # Generated answers (random split)
├── captions_pope_popular.jsonl     # Generated answers (popular split)
├── captions_pope_adversarial.jsonl # Generated answers (adversarial split)
├── output_random.json              # Post-processed answers
├── output_random_label.json        # Ground truth labels
├── output_popular.json
├── output_popular_label.json
├── output_adversarial.json
├── output_adversarial_label.json
└── pope_metric_results.json        # Unified metrics for all 3 splits
```

## Project Structure

```
focusmatters/
├── evaluation/              # Evaluation scripts (CHAIR, POPE)
│   ├── run_chair.sh         # CHAIR evaluation runner
│   ├── run_pope.sh          # POPE evaluation runner
│   ├── eval_chair.py        # CHAIR caption generation
│   └── eval_pope.py         # POPE answer generation
├── eval_utils/              # Metric computation (CHAIR, POPE)
├── model_utils/             # Model framework (config, registry, processors)
│   ├── models/
│   │   ├── llava.py         # LLaVA model definition
│   │   └── dpp_utils.py     # DPP masking implementation
│   ├── configs/             # Model default configurations
│   ├── common/              # Config system, registry
│   └── processors/          # Image/text preprocessors
├── configs/                 # Evaluation YAML configurations
├── decoder_zoo/             # Alternative decoding strategies (VCD, PAI, Devils)
├── transformers-4.37.2/     # Custom transformers (required for OPERA/Devils)
├── dataset/                 # Datasets (user-provided)
└── environment.yml          # Conda environment specification
```
