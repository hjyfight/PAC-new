"""Run selected PNAP sources, then automatically evaluate six targets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from common import MODELS, ROOT, RUN, read, runtime, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="yolov5s")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--run-id", default="", help="Separate launch/log records for disjoint source groups")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Require a fresh short training run before formal training")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--record-launch", action="store_true", help="Record a foreground controller, e.g. inside tmux")
    parser.add_argument("--offload-detector", action="store_true", help="Store detector tensors needed for backward in CPU memory")
    args = parser.parse_args()
    sources, gpus = args.sources.split(","), args.gpus.split(",")
    if not sources or any(s not in MODELS for s in sources) or len(set(sources)) != len(sources):
        parser.error("Invalid/duplicate sources")
    if any(not g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error("Invalid/duplicate GPU indices")
    if args.run_id and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        parser.error("Invalid run ID")
    if not 0 <= args.smoke_steps <= 77:
        parser.error("--smoke-steps must be between 0 and 77")
    if args.detach and args.record_launch:
        parser.error("--detach and --record-launch are mutually exclusive")
    runtime()
    if not read(RUN / "preflight.json")["passed"]:
        raise RuntimeError("Preflight has not passed")
    if not read(RUN / "smoke/yolov5s/evaluation_complete.json")["passed"]:
        raise RuntimeError("YOLOv5s end-to-end smoke evaluation has not passed")
    logs = RUN / "logs"; logs.mkdir(exist_ok=True)
    suffix = "_" + args.run_id if args.run_id else ""
    if args.detach or args.record_launch:
        launch = RUN / f"launch{suffix}.json"
        with (RUN / ".controller_launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for recorded in [RUN / "launch.json", *RUN.glob("launch_*.json")]:
                if not recorded.exists():
                    continue
                previous = read(recorded)
                try:
                    os.kill(previous["pid"], 0)
                except ProcessLookupError:
                    continue
                if recorded == launch or set(sources) & set(previous["sources"]):
                    raise RuntimeError(f"A recorded controller already owns this run/source: {recorded}")
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--sources", args.sources, "--gpus", args.gpus,
                       "--smoke-steps", str(args.smoke_steps)]
            if args.run_id:
                command += ["--run-id", args.run_id]
            if args.offload_detector:
                command += ["--offload-detector"]
            if args.detach:
                with (logs / f"controller{suffix}.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=os.environ, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                pid = child.pid
            else:
                pid = os.getpid()
                command = [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
            write(launch, dict(pid=pid, command=command, sources=sources, gpus=gpus,
                               smoke_steps=args.smoke_steps,
                               tmux_pane=os.environ.get("TMUX_PANE"),
                               cuda_allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                               offload_detector=args.offload_detector,
                               started_utc=datetime.now(timezone.utc).isoformat()))
        print("Launched controller", pid, flush=True)
        if args.detach:
            return
    def lane(index, gpu):
        for source in sources[index::len(gpus)]:
            try:
                phases = ("smoke", "train", "evaluate") if args.smoke_steps else ("train", "evaluate")
                for phase in phases:
                    write(RUN / "jobs" / (source + ".json"), dict(state="running", phase=phase, gpu=gpu))
                    runner = "run_offloaded.py" if args.offload_detector and phase != "evaluate" else "run_experiments.py"
                    command = [sys.executable, "-B", str(RUN / runner),
                               "train" if phase == "smoke" else phase, "--model", source]
                    if phase == "smoke":
                        command += ["--smoke-steps", str(args.smoke_steps)]
                    print("START", source, phase, "GPU", gpu, flush=True)
                    with (logs / f"{source}_{phase}.log").open("a") as log:
                        subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if phase == "smoke":
                        result = read(RUN / "smoke" / source / "smoke_complete.json")
                        if not (result["passed"] and result["smoke"] and result["steps"] == args.smoke_steps
                                and result["latent_grad_norm"] > 0
                                and result["detection_latent_grad_norm"] > 0):
                            raise RuntimeError(f"Smoke validation failed: {source}")
                        print("SMOKE PASSED", source, result, flush=True)
                write(RUN / "jobs" / (source + ".json"), dict(state="complete", gpu=gpu))
            except Exception as exc:
                write(RUN / "jobs" / (source + ".json"), dict(state="failed", error=str(exc), gpu=gpu))
                raise
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs = [pool.submit(lane, index, gpu) for index, gpu in enumerate(gpus)]
        for job in jobs: job.result()
    with (RUN / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([sys.executable, "-B", str(RUN / "run_experiments.py"), "summarize"], cwd=ROOT, check=True)
    write(RUN / f"selected_sources_complete{suffix}.json", dict(passed=True, sources=sources))


if __name__ == "__main__":
    main()
