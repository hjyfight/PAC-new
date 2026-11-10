"""Prepare T-SEA/NAP runs and PAC recolors; adapted from the existing PAC experiment owner."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
PROJECT = ROOT / "experiments/baselines/yolov5-tsea-nap-20260912"
PYTHON = "/datacc05/huotianlin/miniconda3/envs/pac/bin/python"
MODELS = ("yolov2", "yolov3", "yolov4", "yolov5s", "yolov5m", "fasterrcnn")
METHODS = ("tsea", "nap")
RUNTIME = json.loads((RUN / "runtime_environment.json").read_text())
os.environ.update(RUNTIME)
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from train_patch import save_patch
from utils.patch import BASE_COLORS, load_patch, recolor

PROTOCOL_KEYS = ("seed", "data_yaml", "train_images", "test_images", "test_targets", "batch_size", "patch_size",
                 "patch_area_fraction", "placement", "imgsz", "preprocess", "pixel_pipeline", "conf_thres",
                 "nms_iou", "max_det", "asr_conf", "asr_iou", "asr_definition", "detectors")
METRIC_KEYS = ("batch_size", "preprocess", "imgsz", "patch_size", "patch_area_fraction", "conf_thres",
               "nms_iou", "max_det", "asr_conf", "asr_iou")
CODE_FILES = ("train_patch.py", "detector_backends.py", "eval_transfer.py", "utils/patch.py", "utils/augmentations.py",
              "utils/dataloaders.py", "utils/general.py", "utils/metrics.py", "models/experimental.py", "models/common.py", "models/yolo.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_code(config):
    assert {name: sha(PROJECT / name) for name in config["source_code_sha256"]} == config["source_code_sha256"]
    for path, expected in config.get("author_assets_sha256", {}).items():
        assert sha(path) == expected, path


def audit_metric(path, config, source, target, patch_hash):
    record = read(path)
    assert record["optimization_source"] == source and record["detector"] == target
    assert record["images"] == config["test_images"] and record["targets"] == config["test_targets"]
    assert record["patch_sha256"] == patch_hash
    assert record["weights_sha256"] == config["detectors"][target]["sha256"]
    assert record["eot"] is False and record["rect"] is False
    assert record["relation"] == ("control" if source is None else "whitebox" if source == target else "blackbox_transfer")
    for key in METRIC_KEYS:
        assert record[key] == config[key], (path, key)
    assert np.isfinite(record["map50_pct"]) and 0 <= record["map50_pct"] <= 100
    return record


def generate(method, source):
    directory = RUN / method / source
    config = read(directory / "manifest.json")
    complete = read(directory / "train_complete.json")
    assert complete["passed"]
    items = [{"id": "clean", "label": "Clean", "path": None, "optimization_source": None}]
    audits = []
    for selection in ("best", "last"):
        checkpoint = complete["checkpoints"][selection]
        path = Path(checkpoint["path"])
        raw = load_patch(path)
        assert raw.shape == (3, 300, 300) and raw.isfinite().all() and 0 <= raw.min() <= raw.max() <= 1
        assert sha(path) == checkpoint["sha256"]
        epoch = complete["best_training_loss_epoch"] if selection == "best" else config["epochs"]
        item = {"id": selection + "_raw", "label": f"{method.upper()} {source} ({selection})",
                "variant": "raw", "selection": selection, "path": str(path), "sha256": sha(path),
                "optimization_source": source, "source_epoch": epoch}
        items.append(item)
        if method != "tsea":
            continue
        for index, color_name in enumerate(("bc1", "bc2"), 1):
            colors = BASE_COLORS[color_name]
            expected = np.einsum("khw,kc->chw", raw.numpy().astype(np.float64), colors.astype(np.float64) / 255)
            result = recolor(raw, colors).contiguous()
            error = float(np.max(np.abs(result.numpy().astype(np.float64) - expected)))
            assert error < 2e-7 and expected.min() >= 0 and expected.max() <= 1
            # Isolate the deterministic CPU operation, excluding disk I/O and model loading.
            for _ in range(10):
                recolor(raw, colors)
            samples = []
            for _ in range(100):
                started = time.perf_counter()
                recolor(raw, colors)
                samples.append((time.perf_counter() - started) * 1000)
            target = directory / "patches" / f"{selection}_{color_name}.pt"
            save_patch({"patch": result, "patch_size": 300, "method": "fixed-rgb-linear",
                        "source_path": str(path), "source_sha256": sha(path), "source_epoch": epoch,
                        "base_colors_255": colors.tolist(), "recolor_training_epochs": 0,
                        "formula": "out[h,w,c] = sum_k raw[k,h,w] * base_colors_255[k,c] / 255"}, result, target)
            assert torch.equal(load_patch(target), result)
            items.append({"id": selection + "_" + color_name, "label": f"T-SEA+PAC(Bc{index}) {source} ({selection})",
                          "variant": color_name, "selection": selection, "optimization_source": source,
                          "path": str(target), "sha256": sha(target), "source_path": str(path),
                          "source_sha256": sha(path), "source_epoch": epoch, "recolor_ms_median": float(np.median(samples))})
            audits.append({"selection": selection, "palette": color_name, "max_absolute_formula_error": error,
                           "roundtrip_max_error": 0, "softmax": False, "weight_normalization": False,
                           "clipped_values": 0, "unclipped_min": float(expected.min()), "unclipped_max": float(expected.max()),
                           "extra_training_epochs": 0, "timing_device": "CPU", "timing_threads": torch.get_num_threads(),
                           "timing_warmups": 10, "timing_repeats": 100, "median_ms": float(np.median(samples))})
    config["patches"] = items
    write(directory / "manifest.json", config)
    write(directory / "generation_audit.json", {"passed": True, "method": method, "source": source,
          "attack_artifacts": len(items) - 1, "recolor_audits": audits, "created_utc": now()})


def main():
    torch.set_num_threads(2)
    if (RUN / "manifest.json").exists():
        raise RuntimeError("Preparation already exists; do not overwrite the experiment manifest")
    previous = ROOT / "experiments/baselines/advpatch_cap_completion_20260911"
    base = read(previous / "manifest.json")
    common = {key: base[key] for key in PROTOCOL_KEYS}
    assert common["batch_size"] == 8 and common["preprocess"] == "advpatch"
    assert read(RUN / "smoke/unit_tests.json")["passed"]
    assert read(RUN / "smoke/integration_audit.json")["passed"]
    for method in METHODS:
        for source in MODELS:
            assert read(RUN / "smoke" / f"{method}_{source}.json")["passed"]
    for record in read(RUN / "smoke/end_to_end_runs.json"):
        assert record["returncode"] == 0
    for model in common["detectors"].values():
        assert sha(model["weights"]) == model["sha256"]
    dataset = read(previous / "dataset_audit.json")
    for rows in dataset["splits"].values():
        for row in rows:
            assert sha(row["image"]) == row["image_sha256"]
            assert sha(row["label"]) == row["label_sha256"]
    assert dataset["passed"] and dataset["image_hash_overlap"] == 0
    shutil.copyfile(previous / "dataset_audit.json", RUN / "dataset_audit.json")
    files = {name: sha(PROJECT / name) for name in CODE_FILES}
    for name in CODE_FILES:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    author = RUN / "author/Naturalistic-Adversarial-Patch"
    author_files = {str(path.relative_to(author)): sha(path) for path in
                    (author / "GANLatentDiscovery/models/BigGAN").rglob("*.py")}
    tsea = Path("/datacc05/huotianlin/code-hu/PAC/third_party/T-SEA-pac")
    asset_names = (
        "detlib/HHDet/yolov2/yolov2/darknet.py",
        "detlib/HHDet/yolov3/PyTorch_YOLOv3/pytorchyolo/models.py",
        "detlib/HHDet/yolov3/PyTorch_YOLOv3/pytorchyolo/utils/parse_config.py",
        "detlib/HHDet/yolov3/PyTorch_YOLOv3/pytorchyolo/shakedrop/shake_drop.py",
        "detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3-shakedrop.cfg",
        "detlib/HHDet/yolov4/Pytorch_YOLOv4/tool/darknet2pytorch.py",
        "detlib/HHDet/yolov4/Pytorch_YOLOv4/tool/yolo_layer.py",
        "detlib/HHDet/yolov4/Pytorch_YOLOv4/tool/shakedrop/shake_drop.py",
        "detlib/HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4-shakedrop.cfg",
        "detlib/HHDet/yolov5/yolov5/models/shakedrop/shake_drop.py",
        "detlib/HHDet/yolov5/yolov5/models/yolov5s-shakedrop.yaml",
        "detlib/torchDet/faster_rcnn/shakedrop/shake_drop.py",
        "detlib/torchDet/faster_rcnn/shakedrop/resnet.py",
        "utils/solver/lr_scheduler.py", "utils/solver/loss.py", "attack/uap/transformer.py", "utils/loader.py",
    )
    author_assets = {str(tsea / name): sha(tsea / name) for name in asset_names}
    author_assets.update({str(author / name): value for name, value in author_files.items()})
    generator_config = author / "GANLatentDiscovery/models/BigGAN/generator_config.json"
    author_assets[str(generator_config)] = sha(generator_config)
    for method in METHODS:
        for source in MODELS:
            directory = RUN / method / source
            (directory / "logs").mkdir(parents=True, exist_ok=True)
            (directory / "patches").mkdir(exist_ok=True)
            config = {**common, "method": method, "optimization_source": source,
                      "epochs": 1000, "expected_optimizer_steps": 77000, "early_stopping": False,
                      "initial_lr": 0.03 if method == "tsea" else 0.02 if source == "fasterrcnn" else 0.005,
                      "tv_weight": 2.5 if method == "tsea" else 0.1, "nps_weight": 0,
                      "checkpoint_selection": "minimum epoch-mean training total loss; final reported separately",
                      "source_code_sha256": files, "nap_author_code_sha256": author_files,
                      "author_assets_sha256": author_assets,
                      "nap_class": 259, "nap_weights": str(RUN / "weights/NAP_BigGAN_G_ema.pth"),
                      "nap_weights_sha256": read(RUN / "weights/nap_weights.json")["sha256"],
                      "patches": [], "created_utc": now()}
            write(directory / "manifest.json", config)
    manifest = {**common, "experiment": "T-SEA + PAC paired adaptation and NAP appearance comparison",
                "project": str(PROJECT), "sources": list(MODELS), "methods": list(METHODS),
                "epochs": 1000, "allowed_gpus": list(range(8)), "parallelism": "independent single-GPU processes, no DDP",
                "checkpoint_selection": "minimum training total loss; never test-selected; final sensitivity exported separately",
                "source_code_sha256": files, "nap_author_code_sha256": author_files,
                      "author_assets_sha256": author_assets,
                "previous_protocol_manifest": str(previous / "manifest.json"),
                "previous_protocol_manifest_sha256": sha(previous / "manifest.json"),
                "previous_results": str(previous / "matrix_results.json"),
                "clean_reference_paths": {target: str(previous / "cap_recolor/eval" / target / "clean/metrics.json") for target in MODELS},
                "expected_training_runs": 12, "expected_attack_cells_best": 144, "expected_attack_cells_final": 144,
                "psnr_ssim": "Paused per prior user instruction; not calculated or modified",
                "manuscript": "Existing Table 1/2 and numerical entries are not modified",
                "created_utc": now()}
    write(RUN / "manifest.json", manifest)
    (RUN / "logs").mkdir(exist_ok=True)
    print("Prepared 12 source runs, 1000 epochs each, best/final evaluation on six fixed target models.", flush=True)


if __name__ == "__main__":
    main()
