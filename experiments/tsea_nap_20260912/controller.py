"""Queue independent single-GPU T-SEA/NAP pipelines using the established PAC evaluator."""
import argparse
import csv
import fcntl
import math
import os
from pathlib import Path
import subprocess
import time

from prepare_recolor import MODELS, METHODS, PROJECT, PYTHON, RUN, RUNTIME, audit_metric, check_code, generate, now, read, sha, write
import torch


def read_epochs(directory, method):
    path = directory / "patches" / method / "results.csv"
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def gpu_environment(gpu, training=False):
    env = {**os.environ, **RUNTIME}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if not training:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def audit_training(directory, config, gpu):
    method = config["method"]
    rows = read_epochs(directory, method)
    assert len(rows) == config["epochs"]
    assert [int(row["epoch"]) for row in rows] == list(range(1, config["epochs"] + 1))
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    for row in rows:
        assert float(row["nps"]) == 0 and float(row["tv"]) >= 0
        if method == "tsea":
            assert float(row["tv"]) >= .1 - 1e-7
        assert math.isclose(float(row["loss"]), sum(float(row[key]) for key in ("det", "tv", "nps")), rel_tol=2e-6, abs_tol=2e-7)
    assert float(rows[0]["lr"]) == config["initial_lr"]
    assert all(float(b["lr"]) <= float(a["lr"]) for a, b in zip(rows, rows[1:]))
    output = directory / "patches" / method
    summary = read(output / "training_summary.json")
    assert summary["completed_epochs"] == summary["requested_epochs"] == config["epochs"]
    assert summary["optimizer_steps"] == config["expected_optimizer_steps"]
    assert summary["early_stopping"] is False and summary["stop_reason"] == "epoch_limit"
    assert summary["test_used_during_training"] is False
    state = torch.load(output / "training_state.pt", map_location="cpu", weights_only=True)
    assert state["completed_epoch"] == config["epochs"]
    assert len(state["optimizer"]["state"]) == 1
    assert all(int(item["step"]) == config["expected_optimizer_steps"] and "max_exp_avg_sq" in item for item in state["optimizer"]["state"].values())
    assert all(group["amsgrad"] for group in state["optimizer"]["param_groups"])
    assert state["scheduler"]["patience"] == (10 if method == "tsea" else 50)
    if method == "nap":
        assert state["module"]["latent"].shape == (120,) and summary["generator_frozen"]
        assert sha(config["nap_weights"]) == config["nap_weights_sha256"]
        assert state["optimizer"]["param_groups"][0]["betas"] == (.5, .999)
    best = min(rows, key=lambda row: float(row["loss"]))
    assert int(best["epoch"]) == summary["best_training_loss_epoch"]
    checkpoints = {}
    for selection in ("best", "last"):
        path = output / (selection + ".pt")
        patch = torch.load(path, map_location="cpu", weights_only=True)["patch"]
        assert patch.shape == (3, 300, 300) and patch.isfinite().all() and 0 <= patch.min() <= patch.max() <= 1
        checkpoints[selection] = {"path": str(path), "sha256": sha(path), "min": float(patch.min()),
                                  "max": float(patch.max()), "std": float(patch.std())}
    check_code(config)
    result = {"passed": True, "method": method, "source": config["optimization_source"], "epochs": len(rows),
              "optimizer_steps": config["expected_optimizer_steps"], "best_training_loss_epoch": int(best["epoch"]),
              "best_training_loss": float(best["loss"]), "checkpoints": checkpoints, "training_summary": summary,
              "physical_gpu": gpu, "early_stopping": False, "test_used_during_training": False, "completed_utc": now()}
    write(directory / "train_complete.json", result)
    return result


def evaluate_target(directory, target, gpu):
    config = read(directory / "manifest.json")
    check_code(config)
    expected = [directory / "eval" / target / item["id"] / "metrics.json" for item in config["patches"]]
    command = [PYTHON, "-B", "eval_transfer.py", "--manifest", str(directory / "manifest.json"),
               "--detector", target, "--device", "cuda:0", "--batch-size", "8", "--workers", "4",
               "--patches", ",".join(item["id"] for item in config["patches"])]
    log_path = directory / "logs" / f"eval_{target}.log"
    if not all(path.exists() for path in expected):
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=PROJECT, env=gpu_environment(gpu),
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(f"{target} evaluation exited {result.returncode}; {log_path}")
    records = [audit_metric(path, config, item["optimization_source"], target, item.get("sha256"))
               for item, path in zip(config["patches"], expected, strict=True)]
    previous = read(read(RUN / "manifest.json")["clean_reference_paths"][target])
    clean = next(record for record in records if record["patch_id"] == "clean")
    difference = abs(previous["map50_pct"] - clean["map50_pct"])
    assert difference <= 1e-3, (target, "clean evaluation changed", difference)
    assert clean["conditional_asr_denominator"] == previous["conditional_asr_denominator"]
    result = {"passed": True, "target": target, "evaluations": len(records), "physical_gpu": gpu,
              "clean_map50_pct": clean["map50_pct"], "clean_reference_max_difference": difference,
              "command": command, "completed_utc": now()}
    write(directory / "eval" / target / "complete.json", result)
    return result


def source_pipeline(method, source, gpu):
    directory = RUN / method / source
    config = read(directory / "manifest.json")
    check_code(config)
    assert read(RUN / "smoke/unit_tests.json")["passed"]
    profile = read(RUN / "smoke" / f"{method}_{source}.json")
    assert profile["passed"] and profile["batch_size"] == 8 and config["epochs"] == 1000
    started = time.monotonic()
    if not (directory / "train_complete.json").exists():
        if (directory / "train_launch.json").exists() or read_epochs(directory, method):
            raise RuntimeError(f"Refusing to overwrite incomplete training: {directory}")
        command = [PYTHON, "-B", "train_patch.py", "--data", config["data_yaml"],
                   "--weights", config["detectors"][source]["weights"], "--detector", source,
                   "--method", method, "--preprocess", "advpatch", "--epochs", str(config["epochs"]),
                   "--batch-size", "8", "--patch-size", "300", "--imgsz", "640", "--frac", "0.25",
                   "--lr", str(config["initial_lr"]), "--tv-weight", str(config["tv_weight"]), "--nps-weight", "0",
                   "--seed", str(config["seed"]), "--workers", "4", "--device", str(gpu),
                   "--project", str(directory / "patches"), "--name", method]
        with (directory / "logs/train.log").open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=gpu_environment(gpu, training=True),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(directory / "train_launch.json", {"command": command, "training_pid": process.pid,
                  "pipeline_pid": os.getpid(), "physical_gpu": gpu, "preflight": profile,
                  "started_utc": now(), "code_sha256": config["source_code_sha256"]})
            while process.poll() is None:
                rows = read_epochs(directory, method)
                write(directory / "train_status.json", {"state": "running", "epochs": len(rows),
                      "expected_epochs": config["epochs"], "last_epoch": rows[-1] if rows else None,
                      "training_pid": process.pid, "physical_gpu": gpu,
                      "elapsed_seconds": time.monotonic() - started, "updated_utc": now()})
                time.sleep(10)
        if process.returncode:
            raise RuntimeError(f"{method}/{source} training exited {process.returncode}; logs/train.log")
    audit_training(directory, config, gpu)
    write(directory / "train_status.json", {"state": "complete", "epochs": config["epochs"],
          "physical_gpu": gpu, "elapsed_seconds": time.monotonic() - started, "updated_utc": now()})
    if not (directory / "generation_audit.json").exists():
        generate(method, source)
    jobs = {}
    for target in MODELS:
        write(directory / "evaluation_status.json", {"state": "running", "active_target": target, "jobs": jobs, "updated_utc": now()})
        jobs[target] = evaluate_target(directory, target, gpu)
        print(f"{method} {source} -> {target}: evaluated best and final", flush=True)
    write(directory / "evaluation_status.json", {"state": "complete", "jobs": jobs, "updated_utc": now()})
    write(directory / "source_complete.json", {"passed": True, "method": method, "source": source,
          "epochs": config["epochs"], "attack_evaluations": 36 if method == "tsea" else 12, "completed_utc": now()})


def idle_gpus():
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in apps.splitlines() if line.startswith("GPU-")}
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used", "--format=csv,noheader,nounits"], text=True)
    available = {}
    for line in rows.splitlines():
        index, uuid, free, used = [value.strip() for value in line.split(",")]
        if uuid not in occupied and int(used) < 500:
            available[int(index)] = int(free)
    return available


def controller():
    lock = (RUN / ".controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(RUN / "manifest.json")
    check_code(config)
    if (RUN / "matrix_launch.json").exists():
        raise RuntimeError("This controller has already launched; inspect it before any recovery")
    pending = [(method, source) for source in ("yolov4", "yolov3", "fasterrcnn", "yolov5m", "yolov2", "yolov5s") for method in METHODS]
    active, jobs = {}, {}
    started, last_report = time.monotonic(), 0.0
    write(RUN / "matrix_launch.json", {"controller_pid": os.getpid(), "started_utc": now(),
          "command": [PYTHON, "-B", str(Path(__file__))], "allowed_gpus": config["allowed_gpus"],
          "controller_sha256": sha(__file__), "runtime_environment": RUNTIME})
    while pending or active:
        for key, (process, log, gpu, method, source) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            complete = RUN / method / source / "source_complete.json"
            good = code == 0 and complete.exists() and read(complete)["passed"]
            jobs[key].update(state="complete" if good else "failed", exit_code=code, completed_utc=now())
            print(f"{key}: {jobs[key]['state']} (exit {code})", flush=True)
            del active[key]
        available = {gpu: free for gpu, free in idle_gpus().items()
                     if gpu in config["allowed_gpus"] and gpu not in {job[2] for job in active.values()}}
        for method, source in list(pending):
            profile = read(RUN / "smoke" / f"{method}_{source}.json")
            required = profile["reserved_mib"] + 1536
            candidates = [gpu for gpu, free in available.items() if free >= required]
            if not candidates:
                continue
            gpu = min(candidates)
            key = method + "_" + source
            command = [PYTHON, "-B", str(Path(__file__)), "--method", method, "--source", source, "--gpu", str(gpu)]
            log_path = RUN / "logs" / (key + ".log")
            log = log_path.open("w")
            env = {**os.environ, **RUNTIME}
            env.pop("CUDA_VISIBLE_DEVICES", None)
            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            jobs[key] = {"state": "running", "method": method, "source": source, "physical_gpu": gpu,
                         "pipeline_pid": process.pid, "log": str(log_path), "command": command,
                         "free_mib_at_launch": available[gpu], "started_utc": now()}
            active[key] = process, log, gpu, method, source
            pending.remove((method, source))
            del available[gpu]
            print(f"Started {key} on physical GPU {gpu}, PID {process.pid}", flush=True)
        sources = {}
        for method in METHODS:
            for source in MODELS:
                directory = RUN / method / source
                path = directory / "train_status.json"
                progress = read(path) if path.exists() else {"state": "queued", "epochs": 0}
                key = method + "_" + source
                if jobs.get(key, {}).get("state") == "failed":
                    progress["state"] = "failed"
                sources[key] = {**progress, "completed_eval_targets": sum((directory / "eval" / target / "complete.json").exists() for target in MODELS)}
        write(RUN / "matrix_status.json", {"state": "running", "jobs": jobs,
              "queued": [{"method": method, "source": source} for method, source in pending], "sources": sources,
              "controller_pid": os.getpid(), "elapsed_seconds": time.monotonic() - started, "updated_utc": now()})
        if time.monotonic() - last_report >= 60 or not pending and not active:
            with (RUN / "logs/report.log").open("a") as log:
                result = subprocess.run([PYTHON, "-B", str(RUN / "finalize_results.py")], cwd=PROJECT,
                                        env={**os.environ, **RUNTIME}, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f"Report export failed: see {RUN / 'logs/report.log'}", flush=True)
            last_report = time.monotonic()
        if pending or active:
            time.sleep(10)
    passed = all(job["state"] == "complete" for job in jobs.values()) and read(RUN / "matrix_results.json")["complete"]
    status = read(RUN / "matrix_status.json")
    status.update(state="complete" if passed else "failed", jobs=jobs, updated_utc=now())
    write(RUN / "matrix_status.json", status)
    write(RUN / "experiment_complete.json", {"passed": passed, "jobs": len(jobs), "attack_cells": 288 if passed else read(RUN / "matrix_results.json")["attack_cells_present"], "completed_utc": now()})
    if not passed:
        raise RuntimeError("Some pipelines failed; inspect matrix_status.json and per-source logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--source", choices=MODELS)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.method:
        try:
            source_pipeline(args.method, args.source, args.gpu)
        except Exception as error:
            write(RUN / args.method / args.source / "train_error.json", {"error": repr(error), "utc": now()})
            raise
    else:
        controller()


if __name__ == "__main__":
    main()
