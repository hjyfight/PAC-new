"""Prepare the existing CAP best recolors and the five missing AdvPatch source runs.

Adapted from the existing AdvReal prepare_recolor.py. Recoloring and patch saving
reuse the shared implementation; all numerical audits use independent float64 math.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PROJECT = ROOT / "yolov5-advpatch-cap-20260911"
PYTHON = "/datacc05/huotianlin/miniconda3/envs/pac/bin/python"
CAP_RUN = ROOT / "yolov5-cap-rfatex-20260909/runs/cap_rfatex_whitebox_20260909"
REFERENCE_RUN = ROOT / "yolov5-advreal-recolor-20260910/runs/advreal_recolor_20260910"
MODELS = ("yolov2", "yolov3", "yolov4", "yolov5s", "yolov5m", "fasterrcnn")
NEW_SOURCES = tuple(x for x in MODELS if x != "yolov5s")
RUNTIME = {
    "XDG_CACHE_HOME": str(RUN / "cache"), "XDG_CONFIG_HOME": str(RUN / "config"),
    "MPLCONFIGDIR": str(RUN / "mplconfig"), "TORCH_HOME": str(RUN / "torch"),
    "TMPDIR": str(ROOT / ".tmp_ac11"), "YOLOV5_CONFIG_DIR": str(RUN / "yolo_config"),
    "YOLO_CONFIG_DIR": str(RUN / "yolo_config"), "NUMBA_CACHE_DIR": str(RUN / "cache/numba"),
    "YOLOv5_DATASETS_DIR": str(ROOT / "datasets"),
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
    "MPLBACKEND": "Agg", "YOLOv5_AUTOINSTALL": "false", "GIT_OPTIONAL_LOCKS": "0",
}
os.environ.update(RUNTIME)
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
import yaml
from train_patch import save_patch
from utils.patch import BASE_COLORS, load_patch, recolor

PROTOCOL_KEYS = (
    "seed", "data_yaml", "train_images", "test_images", "test_targets",
    "batch_size", "patch_size", "patch_area_fraction", "placement", "imgsz",
    "preprocess", "pixel_pipeline", "conf_thres", "nms_iou", "max_det",
    "asr_conf", "asr_iou", "asr_definition", "detectors",
)
METRIC_KEYS = (
    "batch_size", "preprocess", "imgsz", "patch_size", "patch_area_fraction",
    "conf_thres", "nms_iou", "max_det", "asr_conf", "asr_iou",
)
CODE_FILES = (
    "train_patch.py", "detector_backends.py", "eval_transfer.py", "utils/patch.py",
    "utils/dataloaders.py", "utils/general.py", "utils/metrics.py", "models/experimental.py",
)


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
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_metric(path, config, source, target, patch_hash):
    record = read(path)
    assert record["optimization_source"] == source and record["detector"] == target
    assert record["images"] == config["test_images"] and record["targets"] == config["test_targets"]
    assert record["patch_sha256"] == patch_hash
    assert record["weights_sha256"] == config["detectors"][target]["sha256"]
    assert record["eot"] is False and record["rect"] is False
    assert record["relation"] == ("control" if source is None else ("whitebox" if source == target else "blackbox_transfer"))
    for key in METRIC_KEYS:
        assert record[key] == config[key], (path, key)
    assert np.isfinite(record["map50_pct"]) and 0 <= record["map50_pct"] <= 100
    return record


def make_colors(raw_item, output_dir, prefix, family, best_epoch):
    raw = load_patch(raw_item["path"])
    assert sha(raw_item["path"]) == raw_item["sha256"]
    assert torch.equal(raw, torch.load(raw_item["path"], map_location="cpu", weights_only=True)["patch"])
    assert raw.shape == (3, 300, 300) and raw.isfinite().all() and 0 <= raw.min() <= raw.max() <= 1
    items, audits = [], []
    for index, color_name in enumerate(("bc1", "bc2"), 1):
        colors = BASE_COLORS[color_name]
        assert np.array_equal(colors.sum(axis=0), [255, 255, 255])
        result = recolor(raw, colors).contiguous()
        expected = np.einsum("khw,kc->chw", raw.numpy().astype(np.float64), colors.astype(np.float64) / 255)
        error = float(np.max(np.abs(result.numpy().astype(np.float64) - expected)))
        assert error < 2e-7 and expected.min() >= 0 and expected.max() <= 1
        path = Path(output_dir) / f"{prefix}{index}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_patch({
            "patch": result, "patch_size": 300, "method": "fixed-rgb-linear",
            "source_detector": raw_item["optimization_source"], "source_path": raw_item["path"],
            "source_sha256": raw_item["sha256"], "source_best_epoch": best_epoch,
            "base_colors_255": colors.tolist(), "recolor_training_epochs": 0,
            "formula": "out[h,w,c] = sum_k raw[k,h,w] * base_colors_255[k,c] / 255",
        }, result, path)
        assert torch.equal(load_patch(path), result)
        source = raw_item["optimization_source"]
        items.append({
            "id": f"p{index}" if family == "AdvPatch" else f"cap_bc{index}_{source}",
            "label": f"PAC-P{index} ({source})" if family == "AdvPatch" else f"PAC(CAP, Bc{index}) ({source})",
            "variant": f"p{index}" if family == "AdvPatch" else f"cap_bc{index}",
            "optimization_source": source, "path": str(path), "sha256": sha(path),
            "source_path": raw_item["path"], "source_sha256": raw_item["sha256"],
            "source_best_epoch": best_epoch, "base_colors": color_name,
        })
        audits.append({
            "source": source, "family": family, "base_colors": color_name, "path": str(path),
            "source_sha256": raw_item["sha256"], "sha256": sha(path), "shape": list(result.shape),
            "max_absolute_formula_error": error, "roundtrip_max_error": 0.0,
            "unclipped_min": float(expected.min()), "unclipped_max": float(expected.max()),
            "clipped_values": 0, "softmax": False, "weight_normalization": False,
        })
    return items, audits


def generate_advpatch(source):
    directory = RUN / "advpatch" / source
    config = read(directory / "manifest.json")
    complete = read(directory / "train_complete.json")
    assert complete["passed"] and complete["epochs"] == config["epochs"]
    raw = dict(next(x for x in config["patches"] if x["id"] == "advpatch"))
    raw.update(complete["checkpoints"]["best"])
    raw["source_best_epoch"] = complete["best_training_loss_epoch"]
    items, audits = make_colors(raw, directory / "patches", "p", "AdvPatch", raw["source_best_epoch"])
    config["patches"] = [config["patches"][0], raw, *items]
    write(directory / "generation_audit.json", {"passed": True, "count": 2, "patches": audits})
    write(directory / "manifest.json", config)


def main():
    torch.set_num_threads(2)
    if (RUN / "manifest.json").exists():
        raise RuntimeError("Experiment is already prepared; do not overwrite its provenance.")
    base = read(CAP_RUN / "manifest.json")
    common = {key: base[key] for key in PROTOCOL_KEYS}
    assert common["preprocess"] == "advpatch" and common["batch_size"] == 8 and common["seed"] == 1788272031
    for info in common["detectors"].values():
        assert sha(info["weights"]) == info["sha256"]
    files = {name: sha(PROJECT / name) for name in CODE_FILES}
    for name in CODE_FILES:
        if name not in ("train_patch.py", "detector_backends.py"):
            assert sha(PROJECT / name) == sha(CAP_RUN.parents[1] / name), name
        path = RUN / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PROJECT / name).read_bytes())

    data = yaml.safe_load(Path(base["data_yaml"]).read_text())
    dataset = Path(data["path"])
    assert dataset.resolve().is_relative_to(ROOT)
    split_records = {}
    for split, expected in (("train", 614), ("val", 288)):
        image_dir = dataset / data[split]
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        assert len(images) == expected, (split, len(images))
        rows = []
        for path in images:
            label = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
            values = np.loadtxt(label, ndmin=2)
            assert values.shape[1] == 5 and (values[:, 0] == 0).all()
            rows.append({"image": str(path), "image_sha256": sha(path), "label": str(label),
                         "label_sha256": sha(label), "targets": len(values)})
        split_records[split] = rows
    assert not ({x["image_sha256"] for x in split_records["train"]} &
                {x["image_sha256"] for x in split_records["val"]})
    assert sum(x["targets"] for x in split_records["val"]) == 589
    write(RUN / "dataset_audit.json", {"passed": True, "image_hash_overlap": 0,
          "data_yaml": base["data_yaml"], "data_yaml_sha256": sha(base["data_yaml"]), "splits": split_records})

    cap = {**common, "experiment": "CAP training-selected best recolored with fixed Bc1/Bc2",
           "sources": list(MODELS), "targets": list(MODELS), "evaluation_batch_size": 8,
           "expected_evaluations": 78, "new_attack_evaluations": 72,
           "source_code_sha256": files, "source_manifest": str(CAP_RUN / "manifest.json"),
           "source_manifest_sha256": sha(CAP_RUN / "manifest.json"),
           "recolor_training_epochs": 0, "softmax": False, "weight_normalization": False,
           "naming": "PAC(CAP, Bc1/Bc2), distinct from existing random PAC-R and AdvReal recolors",
           "checkpoint_selection": "Existing minimum epoch-mean CAP patch-phase training loss; no test selection",
           "patches": [{"id": "clean", "label": "Clean", "path": None, "optimization_source": None}],
           "raw_reference_metrics": {}}
    audits = []
    for source in MODELS:
        previous = Path(base["source_runs"][source])
        old_config, complete = read(previous / "manifest.json"), read(previous / "train_complete.json")
        assert complete["passed"] and complete["epochs"] == 1000 and complete["early_stopping"] is False
        for key in PROTOCOL_KEYS:
            assert old_config[key] == common[key], (source, key)
        raw = dict(next(x for x in old_config["patches"] if x["id"] == "cap"))
        assert raw["sha256"] == complete["checkpoints"]["best"]["sha256"]
        items, checks = make_colors(raw, RUN / "cap_recolor/patches" / source, "bc", "CAP",
                                    complete["best_training_loss_epoch"])
        cap["patches"].extend(items)
        audits.extend(checks)
        cap["raw_reference_metrics"][source] = {}
        for target in MODELS:
            path = previous / "eval" / target / "cap/metrics.json"
            audit_metric(path, common, source, target, raw["sha256"])
            cap["raw_reference_metrics"][source][target] = {"path": str(path), "sha256": sha(path)}
    write(RUN / "cap_recolor/manifest.json", cap)
    write(RUN / "cap_recolor/generation_audit.json",
          {"passed": True, "count": len(audits), "max_absolute_formula_error":
           max(x["max_absolute_formula_error"] for x in audits), "patches": audits})

    reference_config = read(REFERENCE_RUN / "manifest.json")
    reference = {}
    for variant in ("advpatch", "p1", "p2"):
        item = next(x for x in reference_config["patches"] if x["id"] == variant)
        assert sha(item["path"]) == item["sha256"] and item["optimization_source"] == "yolov5s"
        reference[variant] = {}
        for target in MODELS:
            path = REFERENCE_RUN / "eval" / target / variant / "metrics.json"
            audit_metric(path, common, "yolov5s", target, item["sha256"])
            reference[variant][target] = {"path": str(path), "sha256": sha(path)}
    sources = {}
    for source in NEW_SOURCES:
        directory = RUN / "advpatch" / source
        (directory / "logs").mkdir(parents=True, exist_ok=True)
        conf = {**common, "method": "AdvPatch", "optimization_source": source,
                "epochs": 200, "initial_lr": 0.03, "early_stopping": False,
                "tv_weight": 2.5, "tv_floor": 0.1, "nps_weight": 0.01,
                "optimizer": "Adam AMSGrad", "lr_schedule": "ReduceLROnPlateau(mode=min, patience=50)",
                "batches_per_epoch": 77, "expected_optimizer_steps": 15400, "workers": 4,
                "checkpoint_selection": "minimum epoch-mean total training loss",
                "test_used_during_training": False, "eval_eot": False,
                "train_eot": {"median_kernel": 7, "contrast": [0.8, 1.2],
                              "brightness": [-0.1, 0.1], "noise": [-0.1, 0.1],
                              "rotation_degrees": [-20, 20], "scale": [1, 1],
                              "rgb_clamp": [1e-6, 0.99999]},
                "detector_confidence": ("mean per-image peak pre-NMS ROI softmax person probability"
                                       if source == "fasterrcnn" else
                                       "mean per-image peak objectness * person probability"),
                "adaptation": "Shared PAC budget/placement; FRCNN person-confidence extension is explicitly documented.",
                "source_code_sha256": files, "targets": list(MODELS),
                "patches": [{"id": "clean", "label": "Clean", "path": None, "optimization_source": None},
                            {"id": "advpatch", "label": f"AdvPatch ({source})",
                             "path": str(directory / "patches/advpatch/best.pt"),
                             "optimization_source": source, "variant": "advpatch"}]}
        write(directory / "manifest.json", conf)
        sources[source] = str(directory)
    master = {**common, "experiment": "AdvPatch missing five sources and fixed CAP recolors",
              "project": str(PROJECT), "sources": list(MODELS), "new_advpatch_sources": sources,
              "advpatch_reference_yolov5s": reference, "advpatch_reference_run": str(REFERENCE_RUN),
              "cap_manifest": str(RUN / "cap_recolor/manifest.json"),
              "source_code_sha256": files, "allowed_gpus": list(range(8)),
              "new_advpatch_training_epochs": 200, "new_advpatch_attack_evaluations": 90,
              "new_cap_recolor_attack_evaluations": 72, "new_clean_evaluations": 36,
              "parallelism": "Independent single-GPU jobs; no DDP",
              "checkpoint_selection": "Training loss only; unchanged selected s000 YOLOv5s reference",
              "psnr_ssim": "Unchanged; not calculated in this experiment",
              "created_utc": datetime.now(timezone.utc).isoformat()}
    write(RUN / "runtime_environment.json", RUNTIME)
    write(RUN / "manifest.json", master)
    print(json.dumps({"prepared": True, "new_sources": list(NEW_SOURCES), "epochs": 200,
                      "cap_recolors": len(audits), "max_formula_error": max(x["max_absolute_formula_error"] for x in audits),
                      "run": str(RUN)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
