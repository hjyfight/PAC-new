"""Independently audit completed best matrices and export CSV, LaTeX and patch panels."""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

from prepare_recolor import MODELS, PROJECT, RUN, read, sha, write, BASE_COLORS, load_patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = {
    "advpatch": "AdvPatch", "p1": "PAC-P1", "p2": "PAC-P2",
    "cap": "CAP", "cap_bc1": "PAC(CAP, Bc1)", "cap_bc2": "PAC(CAP, Bc2)",
}
MODEL_LABELS = {"yolov2": "YOLOv2", "yolov3": "YOLOv3", "yolov4": "YOLOv4",
                "yolov5s": "YOLOv5s", "yolov5m": "YOLOv5m", "fasterrcnn": "Faster R-CNN"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    last_update = 0
    completion = RUN / "experiment_complete.json"
    while not (completion.exists() and read(completion).get("passed")):
        state = read(RUN / "matrix_status.json")
        if state.get("state") == "failed":
            raise RuntimeError("Experiment failed; no complete table exported.")
        if not args.wait:
            raise RuntimeError("Some experiments are still running.")
        if time.monotonic() - last_update >= 60:
            print({name: f"{value.get('epochs', 0)}/200, {value.get('completed_eval_targets', 0)}/6 targets"
                   for name, value in state.get("sources", {}).items()}, flush=True)
            last_update = time.monotonic()
        time.sleep(20)
    master = read(RUN / "manifest.json")
    result = read(RUN / "matrix_results.json")
    assert result["complete"] and result["attack_cells_present"] == result["expected_attack_cells"] == 216
    assert {name: sha(PROJECT / name) for name in master["source_code_sha256"]} == master["source_code_sha256"]
    for source in master["new_advpatch_sources"]:
        complete = read(RUN / "advpatch" / source / "train_complete.json")
        assert complete["passed"] and complete["epochs"] == 200 and complete["optimizer_steps"] == 15400
        assert complete["early_stopping"] is False and complete["test_used_during_training"] is False
        profile = read(RUN / "smoke" / f"{source}.json")
        assert profile["passed"] and profile["batch_size"] == 8 and profile["detector_frozen"]
        for name, item in complete["checkpoints"].items():
            assert sha(item["path"]) == item["sha256"]
    dataset = read(RUN / "dataset_audit.json")
    for records in dataset["splits"].values():
        for item in records:
            assert sha(item["image"]) == item["image_sha256"]
            assert sha(item["label"]) == item["label_sha256"]
    assert len(dataset["splits"]["train"]) == 614 and len(dataset["splits"]["val"]) == 288
    clean = {target: read(RUN / "cap_recolor/eval" / target / "clean/metrics.json") for target in MODELS}
    new_metrics = list((RUN / "advpatch").glob("*/eval/*/*/metrics.json"))
    new_metrics += list((RUN / "cap_recolor/eval").glob("*/*/metrics.json"))
    assert len(new_metrics) == 198, len(new_metrics)
    evaluated, summary, records_by_key = set(), [], {}
    for variant, label in LABELS.items():
        matrix = result["matrices"][variant]
        white, black = [], []
        for source in MODELS:
            for target in MODELS:
                cell = matrix["cells"][source][target]
                assert cell is not None
                path = Path(cell["metrics_path"])
                assert str(path) not in evaluated
                evaluated.add(str(path))
                record = read(path)
                assert record["detector"] == target and record["optimization_source"] == source
                assert record["images"] == 288 and record["targets"] == 589
                assert record["weights_sha256"] == master["detectors"][target]["sha256"]
                assert record["patch_sha256"] == sha(record["patch"])
                assert record["relation"] == ("whitebox" if source == target else "blackbox_transfer")
                for key in ("batch_size", "preprocess", "imgsz", "patch_size", "patch_area_fraction",
                            "conf_thres", "nms_iou", "max_det", "asr_conf", "asr_iou"):
                    assert record[key] == master[key]
                assert record["eot"] is False and record["rect"] is False
                value = record["map50_pct"]
                assert math.isfinite(value) and 0 <= value <= 100
                assert math.isclose(value, record["map50"] * 100, abs_tol=1e-10)
                assert value == cell["map50_pct"]
                numerator, denominator = record["conditional_asr_numerator"], record["conditional_asr_denominator"]
                assert 0 <= numerator <= denominator <= 589
                assert denominator == clean[target]["conditional_asr_denominator"]
                assert math.isclose(record["conditional_asr"], numerator / denominator, abs_tol=1e-12)
                (white if source == target else black).append(value)
                records_by_key[variant, source, target] = record
        assert len(white) == 6 and len(black) == 30
        wb, bb = math.fsum(white) / 6, math.fsum(black) / 30
        assert math.isclose(wb, matrix["whitebox_mean"], abs_tol=1e-10)
        assert math.isclose(bb, matrix["blackbox_mean"], abs_tol=1e-10)
        summary.append({"method": label, "whitebox_mean": wb, "blackbox_mean": bb,
                        "whitebox_cells": 6, "blackbox_cells": 30})
    assert len(evaluated) == 216
    tables = RUN / "tables"
    with (tables / "summary_best.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (tables / "whitebox_best.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", *MODELS, "Avg_WB_6"])
        writer.writerow(["Clean", *(clean[t]["map50_pct"] for t in MODELS),
                         math.fsum(clean[t]["map50_pct"] for t in MODELS) / 6])
        for variant, label in LABELS.items():
            values = [records_by_key[variant, t, t]["map50_pct"] for t in MODELS]
            writer.writerow([label, *values, math.fsum(values) / 6])
    with (tables / "blackbox_best.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Source", "Method", *MODELS, "Avg_all_6", "Avg_BB_5"])
        for source in MODELS:
            for variant, label in LABELS.items():
                values = [records_by_key[variant, source, t]["map50_pct"] for t in MODELS]
                transfer = [records_by_key[variant, source, t]["map50_pct"] for t in MODELS if t != source]
                writer.writerow([source, label, *values, math.fsum(values) / 6, math.fsum(transfer) / 5])

    white_tex = [
        r"% Standalone export; does not modify the manuscript. Requires booktabs and graphicx.",
        r"\begin{table*}[t]", r"\centering", r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{White-box attack performance ($mAP_{50}$, lower is better) under the shared INRIA evaluation protocol. All attack checkpoints are selected by training loss.}",
        r"\label{tab:advpatch_cap_completion_whitebox}", r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        "Method & " + " & ".join(MODEL_LABELS[t] for t in MODELS) + r" & Avg. \\",
        r"\midrule",
    ]
    clean_values = [clean[t]["map50_pct"] for t in MODELS]
    white_tex.append("Clean & " + " & ".join(f"{v:.2f}" for v in [*clean_values, math.fsum(clean_values) / 6]) + r" \\")
    white_tex.append(r"\midrule")
    for variant, label in LABELS.items():
        values = [records_by_key[variant, t, t]["map50_pct"] for t in MODELS]
        white_tex.append(label + " & " + " & ".join(f"{v:.2f}" for v in [*values, math.fsum(values) / 6]) + r" \\")
    white_tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (tables / "whitebox_best.tex").write_text("\n".join(white_tex) + "\n")
    black_tex = [
        r"% Requires booktabs, graphicx and multirow. Diagonal entries are white-box controls.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Transfer attack performance ($mAP_{50}$) under the shared INRIA protocol. All Avg.\ includes the six targets; BB Avg.\ excludes the source-model diagonal and averages five targets.}",
        r"\label{tab:advpatch_cap_completion_transfer}", r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccccccc}", r"\toprule",
        "Source & Method & " + " & ".join(MODEL_LABELS[t] for t in MODELS) + r" & All Avg. & BB Avg. \\",
        r"\midrule",
    ]
    for source_index, source in enumerate(MODELS):
        if source_index:
            black_tex.append(r"\midrule")
        for index, (variant, label) in enumerate(LABELS.items()):
            source_label = r"\multirow{6}{*}{" + MODEL_LABELS[source] + "}" if index == 0 else ""
            values = [records_by_key[variant, source, t]["map50_pct"] for t in MODELS]
            transfer = [records_by_key[variant, source, t]["map50_pct"] for t in MODELS if t != source]
            black_tex.append(source_label + " & " + label + " & " +
                             " & ".join(f"{v:.2f}" for v in [*values, math.fsum(values) / 6, math.fsum(transfer) / 5]) + r" \\")
    black_tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (tables / "blackbox_best.tex").write_text("\n".join(black_tex) + "\n")

    # Visualize the exact evaluated tensors. This is not a PSNR/SSIM computation.
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    panel = Image.new("RGB", (2060, 2000), "white")
    draw = ImageDraw.Draw(panel)
    for column, label in enumerate(LABELS.values()):
        draw.text((190 + column * 310, 15), label, fill="black", font=font)
    formulas = []
    for row, source in enumerate(MODELS):
        patches = {}
        y = 65 + row * 320
        draw.text((10, y + 140), MODEL_LABELS[source], fill="black", font=font)
        for column, variant in enumerate(LABELS):
            record = records_by_key[variant, source, source]
            patch = load_patch(record["patch"])
            patches[variant] = patch.numpy().astype(np.float64)
            pixels = (patch.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            panel.paste(Image.fromarray(pixels), (185 + column * 310, y))
        for raw, colors in (("advpatch", ("p1", "p2")), ("cap", ("cap_bc1", "cap_bc2"))):
            for index, color in enumerate(colors, 1):
                expected = np.einsum("khw,kc->chw", patches[raw], BASE_COLORS[f"bc{index}"].astype(np.float64) / 255)
                error = float(np.abs(expected - patches[color]).max())
                assert error < 2e-7
                formulas.append({"source": source, "raw": raw, "recolor": color, "max_absolute_error": error})
    panel.save(RUN / "patches_best_all_sources.png")
    audit = {"passed": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
             "attack_cells": 216, "new_evaluation_records": 198, "new_attack_records": 162,
             "reused_attack_records": 54, "clean_replays": 36, "source_trainings_completed": 5,
             "epochs_per_new_source": 200, "optimizer_steps_per_new_source": 15400,
             "source_and_target_weights_verified": True, "dataset_hashes_unchanged": True,
             "independently_recomputed_means": summary, "recolor_checks": formulas,
             "max_formula_error": max(item["max_absolute_error"] for item in formulas),
             "test_status": read(RUN / "smoke/unit_tests.json"), "psnr_ssim_changed": False,
             "export_script_sha256": sha(__file__)}
    write(RUN / "evaluation_audit.json", audit)
    write(RUN / "final_summary.json", {"summary": summary, "audit_passed": True,
          "report": str(RUN / "report.md"), "tables": str(tables),
          "patch_panel": str(RUN / "patches_best_all_sources.png")})
    print({"audit_passed": True, "summary": summary, "tables": str(tables)}, flush=True)


if __name__ == "__main__":
    main()
