"""Run the authorized YOLOv5s OBJ-only comparison on physical GPU 7.

Uses existing train_patch.py, recolor(), and the unchanged shared evaluator.
The isolated experiment keeps both best and final checkpoints and their recolors.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
PROJECT = ROOT / "experiments/baselines/yolov5-advpatch-objonly-20261110"
PYTHON = "/datacc05/huotianlin/miniconda3/envs/pac/bin/python"
REFERENCE = ROOT / "experiments/ours/yolov5-advpad-20260908/runs/official_pad3_20260908/patches/ap_s000_advpatch"
GPU = 7
ENV = {
    "XDG_CACHE_HOME": str(RUN / "cache"), "XDG_CONFIG_HOME": str(RUN / "config"),
    "MPLCONFIGDIR": str(RUN / "mplconfig"), "TORCH_HOME": str(RUN / "torch"),
    "TMPDIR": str(ROOT / ".tmp_obj11"), "YOLOV5_CONFIG_DIR": str(RUN / "yolo_config"),
    "YOLO_CONFIG_DIR": str(RUN / "yolo_config"), "NUMBA_CACHE_DIR": str(RUN / "cache/numba"),
    "YOLOv5_DATASETS_DIR": str(ROOT / "datasets"), "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg", "YOLOv5_AUTOINSTALL": "false",
    "GIT_OPTIONAL_LOCKS": "0",
}
os.environ.update(ENV)
sys.path.insert(0, str(PROJECT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(directory):
    path = directory / "results.csv"
    return list(csv.DictReader(path.open())) if path.exists() else []


def status(state, **values):
    write(RUN / "status.json", {"state": state, "physical_gpu": GPU, "controller_pid": os.getpid(),
                               "updated_utc": datetime.now(timezone.utc).isoformat(), **values})


def environment(evaluation=False):
    env = {**os.environ, **ENV}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if evaluation:
        env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    return env


def training_command(epochs, project, name):
    config = read(RUN / "protocol.json")
    return [PYTHON, "-B", "train_patch.py", "--method", "advpatch", "--advpatch-objective", "obj",
            "--data", config["data_yaml"], "--weights", config["detectors"]["yolov5s"]["weights"],
            "--preprocess", "advpatch", "--epochs", str(epochs), "--batch-size", "8",
            "--patch-size", "300", "--imgsz", "640", "--frac", "0.25", "--lr", "0.03",
            "--tv-weight", "2.5", "--nps-weight", "0.01", "--seed", "1788272031",
            "--workers", "4", "--device", str(GPU), "--project", str(project), "--name", name]


def train(epochs, parent, name, phase):
    destination = parent / name
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing training {destination}")
    command = training_command(epochs, parent, name)
    with (RUN / "logs" / f"{phase}.log").open("w") as log:
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=PROJECT, env=environment(), stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        write(RUN / f"{phase}_launch.json", {"command": command, "pid": process.pid,
                                            "physical_gpu": GPU, "started_utc": datetime.now(timezone.utc).isoformat()})
        while process.poll() is None:
            history = rows(destination)
            status(phase, training_pid=process.pid, completed_epochs=len(history), expected_epochs=epochs,
                   elapsed_seconds=time.monotonic() - started, last_epoch=history[-1] if history else None)
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f"{phase} exited {process.returncode}; see logs/{phase}.log")
    summary = audit_training(destination, epochs)
    write(RUN / f"{phase}_complete.json", summary)
    return summary


def audit_training(directory, expected):
    import torch
    history = rows(directory)
    assert [int(row["epoch"]) for row in history] == list(range(1, expected + 1))
    for row in history:
        assert all(math.isfinite(float(v)) for v in row.values())
        assert math.isclose(float(row["loss"]), sum(float(row[k]) for k in ("det", "nps", "tv")), rel_tol=2e-6)
        assert float(row["tv"]) >= 0.1 - 1e-7 and float(row["nps"]) >= 0
    summary = read(directory / "training_summary.json")
    assert summary["advpatch_objective"] == "obj"
    assert summary["detector_confidence"] == "mean per-image peak objectness"
    assert summary["completed_epochs"] == expected and summary["optimizer_steps"] == 77 * expected
    assert summary["test_used_during_training"] is False and summary["early_stopping"] is False
    best = min(history, key=lambda r: float(r["loss"]))
    assert summary["best_training_loss_epoch"] == int(best["epoch"])
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    assert state["completed_epoch"] == expected and len(state["optimizer"]["state"]) == 1
    assert all(int(v["step"]) == expected * 77 and "max_exp_avg_sq" in v
               for v in state["optimizer"]["state"].values())
    checkpoints = {}
    for selection in ("best", "last"):
        path = directory / f"{selection}.pt"
        patch = torch.load(path, map_location="cpu", weights_only=True)["patch"]
        assert patch.shape == (3, 300, 300) and patch.isfinite().all()
        assert 0 <= patch.min() < patch.max() <= 1
        checkpoints[selection] = {"path": str(path), "sha256": sha(path)}
    return {"passed": True, "summary": summary, "checkpoints": checkpoints}


def prepare_patches():
    import numpy as np
    import torch
    from train_patch import save_patch
    from utils.patch import BASE_COLORS, load_patch, recolor
    config = read(RUN / "protocol.json")
    items = [{"id": "clean", "label": "Clean", "path": None, "optimization_source": None},
             {"id": "gray", "label": "Gray", "path": "gray", "optimization_source": None}]
    audits = []
    for objective, folder in (("obj-cls", REFERENCE), ("obj", RUN / "patches/advpatch")):
        history = rows(folder)
        best_epoch = int(min(history, key=lambda r: float(r["loss"]))["epoch"])
        for selection in ("best", "last"):
            source = folder / f"{selection}.pt"
            raw = load_patch(source)
            source_hash = sha(source)
            prefix = objective.replace("-", "_") + "_" + selection
            epoch = best_epoch if selection == "best" else 200
            common = {"objective": objective, "selection": selection, "epoch": epoch,
                      "optimization_source": "yolov5s"}
            items.append({**common, "id": prefix + "_raw", "label": f"{objective} {selection} raw",
                          "variant": "raw", "path": str(source), "sha256": source_hash})
            for index, color_name in enumerate(("bc1", "bc2"), 1):
                colors = BASE_COLORS[color_name]
                transformed = recolor(raw, colors).contiguous()
                expected = np.einsum("khw,kc->chw", raw.numpy().astype(np.float64), colors.astype(np.float64) / 255)
                error = float(np.max(np.abs(transformed.numpy() - expected)))
                assert error < 2e-7 and expected.min() >= 0 and expected.max() <= 1
                path = RUN / "patches/recolors" / f"{prefix}_p{index}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_patch({"patch": transformed, "patch_size": 300, "base_colors_255": colors.tolist(),
                            "source_path": str(source), "source_sha256": source_hash,
                            "recolor_training_epochs": 0}, transformed, path)
                assert torch.equal(load_patch(path), transformed)
                items.append({**common, "id": prefix + f"_p{index}", "label": f"{objective} {selection} PAC-P{index}",
                              "variant": f"p{index}", "path": str(path), "sha256": sha(path),
                              "source_sha256": source_hash, "base_colors": color_name})
                audits.append({"id": items[-1]["id"], "maximum_float64_formula_error": error})
    config["patches"] = items
    write(RUN / "manifest.json", config)
    write(RUN / "generation_audit.json", {"passed": True, "recolors": audits, "count": len(audits)})
    return config


def main():
    for key, value in ENV.items():
        if key.endswith("_HOME") or key.endswith("_DIR") or key in ("TMPDIR", "MPLCONFIGDIR"):
            Path(value).mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    write(RUN / "runtime_environment.json", ENV)
    with (RUN / "logs/unit_tests.log").open("w") as log:
        result = subprocess.run([PYTHON, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_patch.py", "-v"],
                                cwd=PROJECT, env=environment(evaluation=True), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Existing patch tests failed; see logs/unit_tests.log")
    train(1, RUN / "smoke", "obj", "smoke")
    train(200, RUN / "patches", "advpatch", "training")
    config = prepare_patches()
    status("evaluating", expected_evaluations=len(config["patches"]))
    with (RUN / "logs/evaluation.log").open("w") as log:
        command = [PYTHON, "-B", "eval_transfer.py", "--manifest", str(RUN / "manifest.json"),
                   "--detector", "yolov5s", "--device", "cuda:0", "--batch-size", "8", "--workers", "4",
                   "--patches", ",".join(item["id"] for item in config["patches"])]
        result = subprocess.run(command, cwd=PROJECT, env=environment(evaluation=True),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError("Evaluation failed; see logs/evaluation.log")
    result = subprocess.run([PYTHON, "-B", str(RUN / "finish_report.py")], cwd=PROJECT, env=environment())
    if result.returncode:
        raise RuntimeError("Reporting failed")
    status("complete", completed_epochs=200, completed_evaluations=len(config["patches"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=str(error))
        traceback.print_exc()
        raise
