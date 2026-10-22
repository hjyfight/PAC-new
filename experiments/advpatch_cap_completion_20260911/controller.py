"""Run independent single-GPU source pipelines and CAP recolor evaluations.

Adapted from the existing CAP matrix/train controllers. Optimization remains in
train_patch.py; all evaluation uses the unchanged eval_transfer.py.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from prepare_recolor import (
    MODELS, PROJECT, PYTHON, RUN, RUNTIME, audit_metric, generate_advpatch, read, sha, write,
)
import torch


def now():
    return datetime.now(timezone.utc).isoformat()


def read_epochs(directory):
    path = directory / "patches/advpatch/results.csv"
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def check_code(config):
    assert {name: sha(PROJECT / name) for name in config["source_code_sha256"]} == config["source_code_sha256"]


def gpu_environment(gpu, training=False):
    env = {**os.environ, **RUNTIME}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if not training:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def evaluate_target(directory, target, gpu):
    config = read(directory / "manifest.json")
    check_code(config)
    log_path = directory / "logs" / f"eval_{target}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    expected = [directory / "eval" / target / item["id"] / "metrics.json" for item in config["patches"]]
    command = [PYTHON, "-B", "eval_transfer.py", "--manifest", str(directory / "manifest.json"),
               "--detector", target, "--device", "cuda:0", "--batch-size", "8", "--workers", "4",
               "--patches", ",".join(item["id"] for item in config["patches"])]
    if not all(path.exists() for path in expected):
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=PROJECT, env=gpu_environment(gpu),
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(f"{target} evaluation exited {result.returncode}; {log_path}")
    records = []
    for item, path in zip(config["patches"], expected, strict=True):
        records.append(audit_metric(path, config, item["optimization_source"], target, item.get("sha256")))
    # Reused raw CAP baselines must share the same clean result under this replay.
    master = read(RUN / "manifest.json")
    cap = read(master["cap_manifest"])
    reference_path = Path(cap["raw_reference_metrics"]["yolov5s"][target]["path"])
    previous_clean = read(reference_path.parents[1] / "clean/metrics.json")
    current_clean = next(x for x in records if x["patch_id"] == "clean")
    difference = abs(previous_clean["map50_pct"] - current_clean["map50_pct"])
    assert difference <= 1e-3, (target, "clean replay changed", difference)
    assert current_clean["conditional_asr_denominator"] == previous_clean["conditional_asr_denominator"]
    result = {"passed": True, "target": target, "evaluations": len(records),
              "clean_map50_pct": current_clean["map50_pct"], "clean_reference_max_difference": difference,
              "command": command, "physical_gpu": gpu, "completed_utc": now()}
    write(directory / "eval" / target / "complete.json", result)
    return result


def audit_training(directory, config, gpu):
    rows = read_epochs(directory)
    assert len(rows) == config["epochs"]
    assert [int(row["epoch"]) for row in rows] == list(range(1, config["epochs"] + 1))
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    for row in rows:
        assert float(row["tv"]) >= 0.1 - 1e-7 and float(row["nps"]) >= 0
        assert math.isclose(float(row["loss"]), sum(float(row[x]) for x in ("det", "tv", "nps")),
                            rel_tol=2e-6, abs_tol=2e-7)
    assert float(rows[0]["lr"]) == config["initial_lr"]
    assert all(float(b["lr"]) <= float(a["lr"]) for a, b in zip(rows, rows[1:]))
    output = directory / "patches/advpatch"
    summary = read(output / "training_summary.json")
    assert summary["completed_epochs"] == summary["requested_epochs"] == config["epochs"]
    assert summary["early_stopping"] is False and summary["stop_reason"] == "epoch_limit"
    assert summary["optimizer_steps"] == config["expected_optimizer_steps"]
    assert summary["test_used_during_training"] is False
    assert summary["detector_confidence"] == config["detector_confidence"]
    state = torch.load(output / "training_state.pt", map_location="cpu", weights_only=True)
    assert state["completed_epoch"] == config["epochs"]
    assert len(state["optimizer"]["state"]) == 1
    assert all(int(item["step"]) == config["expected_optimizer_steps"] and "max_exp_avg_sq" in item
               for item in state["optimizer"]["state"].values())
    assert all(group["amsgrad"] for group in state["optimizer"]["param_groups"])
    assert state["scheduler"]["mode"] == "min" and state["scheduler"]["patience"] == 50
    best = min(rows, key=lambda row: float(row["loss"]))
    assert int(best["epoch"]) == summary["best_training_loss_epoch"]
    checkpoints = {}
    for name in ("best", "last"):
        path = output / f"{name}.pt"
        patch = torch.load(path, map_location="cpu", weights_only=True)["patch"]
        assert patch.shape == (3, 300, 300) and patch.isfinite().all()
        assert 0 <= patch.min() <= patch.max() <= 1
        checkpoints[name] = {"path": str(path), "sha256": sha(path),
                             "min": float(patch.min()), "max": float(patch.max()), "std": float(patch.std())}
    check_code(config)
    report = {"passed": True, "method": "AdvPatch", "source": config["optimization_source"],
              "epochs": len(rows), "optimizer_steps": config["expected_optimizer_steps"],
              "best_training_loss_epoch": int(best["epoch"]), "best_training_loss": float(best["loss"]),
              "checkpoint_selection": config["checkpoint_selection"], "early_stopping": False,
              "test_used_during_training": False, "checkpoints": checkpoints,
              "training_summary": summary, "physical_gpu": gpu, "completed_utc": now()}
    write(directory / "train_complete.json", report)
    return report


def source_pipeline(source, gpu):
    directory = RUN / "advpatch" / source
    config = read(directory / "manifest.json")
    check_code(config)
    assert config["epochs"] == 200 and config["batch_size"] == 8 and config["early_stopping"] is False
    assert read(RUN / "smoke/unit_tests.json")["passed"]
    assert read(RUN / "smoke/yolov5s.json")["passed"]
    started = time.monotonic()
    if not (directory / "train_complete.json").exists():
        if (directory / "train_launch.json").exists() or read_epochs(directory):
            raise RuntimeError(f"Refusing to overwrite incomplete training: {directory}")
        write(directory / "train_status.json", {"state": "preflight", "physical_gpu": gpu, "updated_utc": now()})
        with (directory / "logs/preflight.log").open("w") as log:
            result = subprocess.run([PYTHON, "-B", str(RUN / "validate_training.py"), "--detector", source],
                                    cwd=PROJECT, env=gpu_environment(gpu), stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(f"{source} batch8 preflight failed; logs/preflight.log")
        profile = read(RUN / "smoke" / f"{source}.json")
        assert profile["passed"] and profile["batch_size"] == 8
        command = [PYTHON, "-B", "train_patch.py", "--data", config["data_yaml"],
                   "--weights", config["detectors"][source]["weights"], "--detector", source,
                   "--method", "advpatch", "--preprocess", "advpatch", "--epochs", "200",
                   "--batch-size", "8", "--patch-size", "300", "--imgsz", "640", "--frac", "0.25",
                   "--lr", "0.03", "--tv-weight", "2.5", "--nps-weight", "0.01",
                   "--seed", str(config["seed"]), "--workers", "4", "--device", str(gpu),
                   "--project", str(directory / "patches"), "--name", "advpatch"]
        with (directory / "logs/train.log").open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=gpu_environment(gpu, training=True),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(directory / "train_launch.json", {"command": command, "training_pid": process.pid,
                  "controller_pid": os.getpid(), "physical_gpu": gpu, "preflight": profile,
                  "started_utc": now(), "code_sha256": config["source_code_sha256"]})
            while process.poll() is None:
                rows = read_epochs(directory)
                write(directory / "train_status.json", {"state": "running", "epochs": len(rows),
                      "expected_epochs": 200, "last_epoch": rows[-1] if rows else None,
                      "training_pid": process.pid, "physical_gpu": gpu,
                      "elapsed_seconds": time.monotonic() - started, "updated_utc": now()})
                time.sleep(10)
        if process.returncode:
            raise RuntimeError(f"{source} training exited {process.returncode}; logs/train.log")
        audit_training(directory, config, gpu)
    else:
        audit_training(directory, config, gpu)
    write(directory / "train_status.json", {"state": "complete", "epochs": 200, "physical_gpu": gpu,
          "elapsed_seconds": time.monotonic() - started, "updated_utc": now()})
    if not (directory / "generation_audit.json").exists():
        generate_advpatch(source)
    jobs = {}
    for target in MODELS:
        write(directory / "evaluation_status.json", {"state": "running", "active_target": target,
              "jobs": jobs, "updated_utc": now()})
        jobs[target] = evaluate_target(directory, target, gpu)
        print(f"AdvPatch {source} -> {target}: raw/P1/P2 complete", flush=True)
    write(directory / "evaluation_status.json", {"state": "complete", "jobs": jobs, "updated_utc": now()})
    write(directory / "source_complete.json", {"passed": True, "source": source, "epochs": 200,
          "attack_evaluations": 18, "whitebox_evaluations": 3, "blackbox_evaluations": 15,
          "clean_evaluations": 6, "completed_utc": now()})


VARIANTS = {
    "advpatch": "AdvPatch", "p1": "PAC-P1", "p2": "PAC-P2",
    "cap": "CAP", "cap_bc1": "PAC(CAP, Bc1)", "cap_bc2": "PAC(CAP, Bc2)",
}


def report():
    master = read(RUN / "manifest.json")
    cap = read(master["cap_manifest"])
    cap_items = {p["id"]: p for p in cap["patches"]}
    matrices, rows = {}, []
    for variant, label in VARIANTS.items():
        cells = {}
        for source in MODELS:
            cells[source] = {}
            for target in MODELS:
                pointer = None
                if variant == "cap":
                    pointer = cap["raw_reference_metrics"][source][target]
                    path = Path(pointer["path"])
                elif variant.startswith("cap_"):
                    path = RUN / "cap_recolor/eval" / target / f"{variant}_{source}" / "metrics.json"
                elif source == "yolov5s":
                    pointer = master["advpatch_reference_yolov5s"][variant][target]
                    path = Path(pointer["path"])
                else:
                    path = RUN / "advpatch" / source / "eval" / target / variant / "metrics.json"
                if not path.exists():
                    cells[source][target] = None
                    continue
                if pointer:
                    assert sha(path) == pointer["sha256"], path
                    record = read(path)
                elif variant.startswith("cap_"):
                    item = cap_items[f"{variant}_{source}"]
                    record = audit_metric(path, cap, source, target, item["sha256"])
                else:
                    config = read(RUN / "advpatch" / source / "manifest.json")
                    item = next(p for p in config["patches"] if p["id"] == variant)
                    record = audit_metric(path, config, source, target, item["sha256"])
                cells[source][target] = {"map50_pct": record["map50_pct"], "conditional_asr": record["conditional_asr"],
                                         "relation": record["relation"], "metrics_path": str(path)}
                rows.append({"method": label, "variant": variant, "source": source, "target": target,
                             "relation": record["relation"], "map50_pct": record["map50_pct"],
                             "conditional_asr": record["conditional_asr"], "metrics_path": str(path)})
        white = [cells[s][s]["map50_pct"] for s in MODELS if cells[s][s] is not None]
        black = [cells[s][t]["map50_pct"] for s in MODELS for t in MODELS if s != t and cells[s][t] is not None]
        matrices[variant] = {"label": label, "cells": cells, "whitebox_cells": len(white),
                             "blackbox_cells": len(black), "complete": len(white) == 6 and len(black) == 30,
                             "whitebox_mean": sum(white) / 6 if len(white) == 6 else None,
                             "blackbox_mean": sum(black) / 30 if len(black) == 30 else None}
    pairs = {}
    for raw, colors in (("advpatch", ("p1", "p2")), ("cap", ("cap_bc1", "cap_bc2"))):
        for color in colors:
            pairs[color] = {key + "_penalty": matrices[color][key] - matrices[raw][key]
                            if matrices[color][key] is not None and matrices[raw][key] is not None else None
                            for key in ("whitebox_mean", "blackbox_mean")}
    result = {"updated_utc": now(), "complete": all(x["complete"] for x in matrices.values()),
              "attack_cells_present": len(rows), "expected_attack_cells": 216,
              "model_order": list(MODELS), "matrices": matrices, "recolor_penalties": pairs,
              "average_rule": "WB: six diagonal cells; BB: 30 off-diagonal cells. No incomplete averages or zero imputation.",
              "psnr_ssim": "Not calculated or changed"}
    write(RUN / "matrix_results.json", result)
    tables = RUN / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / "all_best_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ["method", "variant", "source", "target", "relation", "map50_pct",
                                 "conditional_asr", "metrics_path"])
        writer.writeheader()
        writer.writerows(rows)
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"
    text = [
        "# AdvPatch 五个缺失源模型及 CAP 换色实验",
        "", f"更新：{now()}；攻击结果 {len(rows)}/216。mAP50 越低越好。",
        "", "best 均按训练损失选择；保留 s000 的 YOLOv5s 参考结果。Faster R-CNN 的 AdvPatch 使用逐图最高的 pre-NMS ROI 人类别概率。",
        "黑盒均值只计 30 个非对角单元；缺失结果留空，不计算不完整均值。PSNR/SSIM 未修改或计算。",
        "", "| 方法 | " + " | ".join(MODELS) + " | 白盒均值 | 黑盒均值 |",
        "|---|" + "---:|" * 8,
    ]
    for variant, data in matrices.items():
        values = [data["cells"][source][source]["map50_pct"] if data["cells"][source][source] else None for source in MODELS]
        text.append("| " + data["label"] + " | " + " | ".join(map(fmt, values + [data["whitebox_mean"], data["blackbox_mean"]])) + " |")
    for variant, data in matrices.items():
        text += ["", "## " + data["label"], "", "| 源模型 | " + " | ".join(MODELS) + " | 黑盒均值（5个目标） |",
                 "|---|" + "---:|" * 7]
        for source in MODELS:
            values = [data["cells"][source][target]["map50_pct"] if data["cells"][source][target] else None for target in MODELS]
            offdiag = [data["cells"][source][target]["map50_pct"] for target in MODELS
                       if target != source and data["cells"][source][target] is not None]
            mean = sum(offdiag)/5 if len(offdiag) == 5 else None
            text.append("| " + source + " | " + " | ".join(map(fmt, values + [mean])) + " |")
    (RUN / "report.md").write_text("\n".join(text) + "\n")
    return result


def idle_gpus():
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in apps.splitlines() if line.startswith("GPU-")}
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    available = {}
    for line in output.splitlines():
        index, uuid, free, used = [value.strip() for value in line.split(",")]
        if uuid not in occupied and int(used) < 500:
            available[int(index)] = int(free)
    return available


def controller():
    lock = (RUN / ".controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(RUN / "manifest.json")
    check_code(config)
    assert read(RUN / "smoke/unit_tests.json")["passed"]
    assert read(RUN / "smoke/yolov5s.json")["passed"]
    if (RUN / "matrix_launch.json").exists():
        raise RuntimeError("Controller already launched; inspect existing jobs before recovery.")
    pending = [("source", name) for name in ("yolov4", "yolov3", "fasterrcnn", "yolov5m", "yolov2")]
    pending += [("cap", name) for name in ("fasterrcnn", "yolov4", "yolov2", "yolov5s", "yolov3", "yolov5m")]
    active, jobs = {}, {}
    started = time.monotonic()
    last_report = 0
    write(RUN / "matrix_launch.json", {"controller_pid": os.getpid(), "started_utc": now(),
          "command": [PYTHON, "-B", str(Path(__file__))], "allowed_gpus": config["allowed_gpus"],
          "controller_sha256": sha(__file__), "prepare_sha256": sha(RUN / "prepare_recolor.py"),
          "validate_sha256": sha(RUN / "validate_training.py"), "runtime_environment": RUNTIME})
    while pending or active:
        for key, (process, log, gpu, kind, name) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            complete = (RUN / "advpatch" / name / "source_complete.json" if kind == "source" else
                        RUN / "cap_recolor/eval" / name / "complete.json")
            good = code == 0 and complete.exists() and read(complete)["passed"]
            jobs[key].update(state="complete" if good else "failed", exit_code=code, completed_utc=now())
            print(f"{key}: {jobs[key]['state']} (exit {code})", flush=True)
            del active[key]
        available = {gpu: free for gpu, free in idle_gpus().items()
                     if gpu in config["allowed_gpus"] and gpu not in {job[2] for job in active.values()}}
        for kind, name in list(pending):
            candidates = [gpu for gpu, free in available.items() if free >= (23000 if kind == "source" else 10240)]
            if not candidates:
                continue
            gpu = min(candidates)
            key = f"{kind}_{name}"
            command = [PYTHON, "-B", str(Path(__file__)), "--kind", kind, "--model", name, "--gpu", str(gpu)]
            log_path = RUN / "logs" / f"{key}.log"
            log = log_path.open("w")
            env = {**os.environ, **RUNTIME}
            env.pop("CUDA_VISIBLE_DEVICES", None)
            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            jobs[key] = {"state": "running", "kind": kind, "model": name, "physical_gpu": gpu,
                         "pipeline_pid": process.pid, "log": str(log_path), "command": command,
                         "free_mib_at_launch": available[gpu], "started_utc": now()}
            active[key] = process, log, gpu, kind, name
            pending.remove((kind, name))
            del available[gpu]
            print(f"Started {key} on physical GPU {gpu}, PID {process.pid}", flush=True)
        sources = {}
        for source, value in config["new_advpatch_sources"].items():
            directory = Path(value)
            progress = read(directory / "train_status.json") if (directory / "train_status.json").exists() else {"state": "queued", "epochs": 0}
            sources[source] = {**progress, "completed_eval_targets":
                               sum((directory / "eval" / target / "complete.json").exists() for target in MODELS)}
        write(RUN / "matrix_status.json", {"state": "running", "jobs": jobs,
              "queued": [{"kind": kind, "model": name} for kind, name in pending],
              "sources": sources, "controller_pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
              "updated_utc": now()})
        if time.monotonic() - last_report >= 60 or not pending and not active:
            report()
            last_report = time.monotonic()
        if pending or active:
            time.sleep(10)
    results = report()
    passed = all(job["state"] == "complete" for job in jobs.values()) and results["complete"]
    status = read(RUN / "matrix_status.json")
    status.update(state="complete" if passed else "failed", jobs=jobs, updated_utc=now())
    write(RUN / "matrix_status.json", status)
    write(RUN / "experiment_complete.json", {"passed": passed, "jobs": len(jobs),
          "attack_cells": results["attack_cells_present"], "new_attack_evaluations": 162,
          "new_clean_evaluations": 36, "completed_utc": now()})
    if not passed:
        raise RuntimeError("Some jobs failed; inspect matrix_status.json and task logs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("source", "cap"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        result = report()
        print({"attack_cells": result["attack_cells_present"], "complete": result["complete"]}, flush=True)
    elif args.kind == "source":
        try:
            source_pipeline(args.model, args.gpu)
        except Exception as error:
            directory = RUN / "advpatch" / args.model
            write(directory / "train_error.json", {"error": repr(error), "utc": now()})
            raise
    elif args.kind == "cap":
        evaluate_target(RUN / "cap_recolor", args.model, args.gpu)
    else:
        controller()


if __name__ == "__main__":
    main()
