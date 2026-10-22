"""Read-only compact status for the AdvPatch/CAP completion experiment."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

RUN = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    master = read(RUN / "manifest.json")
    status = read(RUN / "matrix_status.json")
    result = read(RUN / "matrix_results.json")
    sources = {}
    current = datetime.now(timezone.utc)
    for source, folder in master["new_advpatch_sources"].items():
        directory = Path(folder)
        progress = read(directory / "train_status.json")
        launch = read(directory / "train_launch.json")
        path = directory / "patches/advpatch/results.csv"
        with path.open() if path.exists() else open(os.devnull) as stream:
            rows = list(csv.DictReader(stream))
        completed = len(rows)
        elapsed = ((current - datetime.fromisoformat(launch["started_utc"])).total_seconds()
                   if launch.get("started_utc") else None)
        remaining = max(0, 200 - completed) * elapsed / completed / 60 if completed and elapsed is not None else None
        best = min(rows, key=lambda r: float(r["loss"])) if rows else None
        complete = read(directory / "source_complete.json")
        error = read(directory / "train_error.json")
        state = ("failed" if error else "complete" if complete.get("passed")
                 else "evaluating" if (directory / "train_complete.json").exists()
                 else progress.get("state", "queued"))
        sources[source] = {
            "state": state, "epochs": completed, "expected_epochs": 200,
            "gpu": launch.get("physical_gpu", progress.get("physical_gpu")),
            "training_pid": launch.get("training_pid"),
            "estimated_training_minutes_left": round(remaining, 1) if remaining is not None else None,
            "best_training_loss_epoch_so_far": int(best["epoch"]) if best else None,
            "latest_loss": float(rows[-1]["loss"]) if rows else None,
            "completed_targets": sum((directory / "eval" / target / "complete.json").exists()
                                     for target in master["sources"]),
            "error": error or None,
        }
    cap = {}
    for key in ("cap", "cap_bc1", "cap_bc2"):
        matrix = result.get("matrices", {}).get(key, {})
        cap[key] = {k: matrix.get(k) for k in ("complete", "whitebox_mean", "blackbox_mean")}
    pid = status.get("controller_pid")
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            pass
    print(json.dumps({
        "utc": current.isoformat(), "state": status.get("state"), "controller_pid": pid,
        "controller_alive": alive, "advpatch_sources": sources, "cap_results": cap,
        "attack_cells_present": result.get("attack_cells_present"),
        "expected_attack_cells": result.get("expected_attack_cells"),
        "report": str(RUN / "report.md"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
