"""Matched NCQ and random-three-color experiments using the existing PAC evaluator.

Artifacts belong to this directory; the detector/evaluation implementation is
imported without editing the existing experiment or detector checkouts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
PROJECT = ROOT / "experiments/baselines/yolov5-advpatch-cap-20260911"
PREVIOUS = ROOT / "experiments/baselines/advpatch_cap_completion_20260911"
REFERENCE = ROOT / "experiments/ours/yolov5-advreal-recolor-20260910/runs/advreal_recolor_20260910"
MODELS = ("yolov2", "yolov3", "yolov4", "yolov5s", "yolov5m", "fasterrcnn")
SEEDS = tuple(range(5))
CODE_FILES = (
    "eval_transfer.py", "detector_backends.py", "utils/patch.py",
    "utils/dataloaders.py", "utils/general.py", "utils/metrics.py",
    "models/experimental.py",
)

# Set these before importing plotting, detector, or torch libraries.
for key, value in {
    "XDG_CACHE_HOME": "runtime/cache", "XDG_CONFIG_HOME": "runtime/config",
    "MPLCONFIGDIR": "runtime/mplconfig", "TORCH_HOME": "runtime/torch",
    "YOLO_CONFIG_DIR": "runtime/yolo_config",
    "YOLOV5_CONFIG_DIR": "runtime/yolo_config", "NUMBA_CACHE_DIR": "runtime/numba",
}.items():
    path = RUN / value
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
temporary_directory = ROOT / ".tmp_ncq25"
temporary_directory.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(temporary_directory)
os.environ.update({
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
    "MPLBACKEND": "Agg", "YOLOv5_AUTOINSTALL": "false", "GIT_OPTIONAL_LOCKS": "0",
    "YOLOv5_DATASETS_DIR": str(ROOT / "datasets"),
})
sys.path.insert(0, str(PROJECT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_code(config):
    assert {name: sha(PROJECT / name) for name in CODE_FILES} == config["source_code_sha256"]


def prepare():
    import numpy as np
    from PIL import Image
    import torch
    import yaml
    from utils.patch import BASE_COLORS, load_patch, recolor

    if (RUN / "protocol.json").exists():
        raise RuntimeError("Protocol already exists; use run/report to resume without regenerating patches.")
    torch.set_num_threads(2)
    master = read(PREVIOUS / "manifest.json")
    common = {key: master[key] for key in (
        "seed", "data_yaml", "train_images", "test_images", "test_targets", "batch_size",
        "patch_size", "patch_area_fraction", "placement", "imgsz", "preprocess",
        "pixel_pipeline", "conf_thres", "nms_iou", "max_det", "asr_conf", "asr_iou",
        "asr_definition", "detectors",
    )}
    common["source_code_sha256"] = {name: sha(PROJECT / name) for name in CODE_FILES}
    assert common["source_code_sha256"] == {name: master["source_code_sha256"][name] for name in CODE_FILES}
    for item in common["detectors"].values():
        assert sha(item["weights"]) == item["sha256"]

    dataset = yaml.safe_load(Path(common["data_yaml"]).read_text())
    records = []
    for path in sorted((Path(dataset["path"]) / dataset["val"]).glob("*")):
        if path.suffix.lower() not in (".jpg", ".png", ".jpeg", ".bmp"):
            continue
        label = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
        labels = np.loadtxt(label, ndmin=2)
        assert labels.shape[1] == 5 and (labels[:, 0] == 0).all()
        records.append({"image": str(path), "image_sha256": sha(path), "label": str(label),
                        "label_sha256": sha(label), "targets": len(labels)})
    assert len(records) == common["test_images"] == 288
    assert sum(row["targets"] for row in records) == common["test_targets"] == 589
    write(RUN / "dataset_audit.json", records)

    palettes = {name: value.tolist() for name, value in BASE_COLORS.items()}
    for seed in SEEDS:
        palettes[f"random_s{seed}"] = np.random.default_rng(seed).integers(0, 256, (3, 3)).tolist()
    write(RUN / "palettes.json", palettes)
    audit, source_items = [], {}
    for model in MODELS:
        old = read(REFERENCE / "manifest.json") if model == "yolov5s" else read(
            PREVIOUS / "advpatch" / model / "manifest.json")
        source = next(item for item in old["patches"] if item["id"] == "advpatch")
        assert sha(source["path"]) == source["sha256"]
        raw = load_patch(source["path"])
        assert raw.shape == (3, 300, 300) and torch.isfinite(raw).all()
        source_items[model] = source
        directory = RUN / "sources" / model
        items = [
            {"id": "clean", "label": "Clean", "path": None, "optimization_source": None},
            {"id": "gray", "label": "Gray", "path": "gray", "optimization_source": None},
            {**source, "label": "AdvPatch", "optimization_source": model},
        ]

        def save(item_id, patch, colors, **metadata):
            path = directory / "patches" / f"{item_id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            patch = patch.contiguous()
            torch.save({"patch": patch, "patch_size": 300, "source_path": source["path"],
                        "source_sha256": source["sha256"], "base_colors_255": colors,
                        "extra_optimization_steps": 0, **metadata}, path)
            assert torch.equal(load_patch(path), patch)
            Image.fromarray((patch.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
                path.with_suffix(".png"))
            items.append({"id": item_id, "label": item_id, "path": str(path),
                          "sha256": sha(path), "optimization_source": model, **metadata})

        def pac(item_id, colors, patch=raw, **metadata):
            result = recolor(patch, colors)
            unbounded = np.einsum("khw,kc->chw", patch.numpy().astype(np.float64),
                                   np.asarray(colors, dtype=np.float64) / 255)
            error = float(np.abs(result.numpy() - unbounded.clip(0, 1)).max())
            assert error < 3e-7
            audit.append({"model": model, "id": item_id, "formula_max_error": error,
                          "clipped_fraction": float(((unbounded < 0) | (unbounded > 1)).mean()),
                          "unclipped_min": float(unbounded.min()), "unclipped_max": float(unbounded.max())})
            save(item_id, result, colors, operation="PAC linear RGB weights, then clamp [0,1]", **metadata)

        for index in (1, 2):
            colors = palettes[f"bc{index}"]
            pac(f"p{index}", colors)
            old_patch = next(item for item in old["patches"] if item["id"] == f"p{index}")
            assert torch.equal(load_patch(old_patch["path"]), recolor(raw, colors))
            centers = torch.tensor(colors, dtype=torch.float32) / 255
            pixels = raw.permute(1, 2, 0)
            nearest = ((pixels[:, :, None] - centers)**2).sum(-1).argmin(-1)
            independent = ((pixels.numpy().astype(np.float64)[:, :, None] -
                            np.asarray(colors, dtype=np.float64) / 255)**2).sum(-1).argmin(-1)
            assert np.array_equal(nearest.numpy(), independent)
            quantized = centers[nearest].permute(2, 0, 1)
            save(f"ncq{index}", quantized, colors, operation="nearest RGB squared Euclidean distance",
                 palette_counts=torch.bincount(nearest.flatten(), minlength=3).tolist())

        for seed in SEEDS:
            colors = palettes[f"random_s{seed}"]
            pac(f"random_s{seed}", colors, palette_seed=seed, structure="retained")
            permutation = np.random.default_rng(10000 + seed).permutation(300 * 300)
            shuffled = raw.reshape(3, -1)[:, permutation].reshape_as(raw)
            assert torch.equal(shuffled.reshape(3, -1).sort(1).values, raw.reshape(3, -1).sort(1).values)
            for palette_name, color_set in (("bc1", palettes["bc1"]), ("random", colors)):
                item_id = f"shuffle_{palette_name}_s{seed}"
                pac(item_id, color_set, patch=shuffled, structure="pixel-vector permutation",
                    permutation_seed=10000 + seed, palette_seed=seed if palette_name == "random" else None)
                original = recolor(raw, color_set).reshape(3, -1).sort(1).values
                rendered = recolor(shuffled, color_set).reshape(3, -1).sort(1).values
                assert torch.equal(original, rendered)
        config = {**common, "optimization_source": model, "patches": items,
                  "evaluation_batch_size": 8, "palette_seeds": list(SEEDS)}
        write(directory / "manifest.json", config)
    write(RUN / "generation_audit.json", {"passed": True, "patches": audit})
    protocol = {
        **common, "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS), "palette_seeds": list(SEEDS), "source_items": source_items,
        "random_palette_sampling": "NumPy default_rng(seed).integers(0,256,(3,3)); ordered RGB triples; no selection",
        "structure_control": "Full RGB pixel vectors permuted together, same permutation for Bc1 and random palette",
        "ncq": "Each original donor pixel mapped to nearest of the same three base colors in RGB; no optimization",
        "pac": "Unnormalized original RGB channels used as weights; same existing recolor() including clamp",
        "variance": "Sample SD across five palette/permutation draws, NOT five independent training runs",
        "scope": "Six white-box detectors, existing single donor training seed; Bc1/Bc2 are fixed synthetic palettes",
        "expected_evaluations": 6 * 22, "script_sha256_at_prepare": sha(__file__),
        "dataset_audit_sha256": sha(RUN / "dataset_audit.json"),
    }
    write(RUN / "protocol.json", protocol)
    print(json.dumps({"prepared": True, "evaluations": 132, "palettes": palettes}, ensure_ascii=False), flush=True)


def worker(model_name):
    import numpy as np
    import torch
    from detector_backends import DetectorBackend
    from eval_transfer import evaluate
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset

    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    directory = RUN / "sources" / model_name
    config = read(directory / "manifest.json")
    check_code(config)
    model = DetectorBackend(model_name, device=device, image_size=config["imgsz"], fused=True,
                            conf_threshold=config["conf_thres"], nms_iou=config["nms_iou"],
                            max_det=config["max_det"], cache_dir=RUN / "runtime/torchhub")
    weight_hash = sha(model.weight_path)
    assert weight_hash == config["detectors"][model_name]["sha256"]
    data = check_dataset(config["data_yaml"])
    loader, _ = create_dataloader(data["val"], config["imgsz"], 8, 32, workers=2,
                                 rect=False, pad=0, shuffle=False, preprocess=config["preprocess"])
    with torch.inference_mode():
        model.predict(torch.zeros(1, 3, config["imgsz"], config["imgsz"], device=device))
    for item in config["patches"]:
        output = directory / "eval" / model_name / item["id"]
        if not (output / "metrics.json").exists():
            evaluate(model, loader, item, config, output, weight_hash)
    check_code(config)
    write(directory / "complete.json", {"passed": True, "evaluations": len(config["patches"])})


def run(gpus):
    check_code(read(RUN / "protocol.json"))

    def lane(index, gpu):
        for model in MODELS[index::len(gpus)]:
            log_path = RUN / "logs" / f"{model}.log"
            log_path.parent.mkdir(exist_ok=True)
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--model", model]
            print(f"START GPU {gpu}: {model}", flush=True)
            with log_path.open("a") as log:
                result = subprocess.run(command, cwd=PROJECT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{model} exited {result.returncode}: {log_path}")
            print(f"DONE GPU {gpu}: {model}", flush=True)

    write(RUN / "launch.json", {"gpus": gpus, "pid": os.getpid(), "command": sys.argv,
                                "script_sha256": sha(__file__), "temporary_directory": str(temporary_directory),
                                "started_utc": datetime.now(timezone.utc).isoformat()})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs = [pool.submit(lane, i, gpu) for i, gpu in enumerate(gpus)]
        for job in jobs:
            job.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "worker"))
    parser.add_argument("--gpus", default="1,6,7")
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "worker":
        worker(args.model)
    else:
        run([int(value) for value in args.gpus.split(",")])


if __name__ == "__main__":
    main()
