"""Export audited paired tables without modifying the manuscript or imputing missing cells."""
import csv
import math
import os
import subprocess
from pathlib import Path

from prepare_recolor import MODELS, METHODS, PROJECT, PYTHON, ROOT, RUN, audit_metric, check_code, now, read, sha, write

LABELS = {"advpatch": "AdvPatch", "p1": "PAC-P1", "p2": "PAC-P2",
          "cap": "CAP", "cap_bc1": "PAC(CAP, Bc1)", "cap_bc2": "PAC(CAP, Bc2)",
          "advreal": "AdvReal", "advreal_bc1": "PAC-R1* (AdvReal)", "advreal_bc2": "PAC-R2* (AdvReal)",
          "tsea": "T-SEA", "tsea_bc1": "PAC(T-SEA, Bc1)", "tsea_bc2": "PAC(T-SEA, Bc2)", "nap": "NAP"}
MODEL_LABELS = {"yolov2": "YOLOv2", "yolov3": "YOLOv3", "yolov4": "YOLOv4", "yolov5s": "YOLOv5s",
                "yolov5m": "YOLOv5m", "fasterrcnn": "Faster R-CNN"}
FAMILIES = {"AdvPatch": ("advpatch", "p1", "p2"), "CAP": ("cap", "cap_bc1", "cap_bc2"),
            "AdvReal": ("advreal", "advreal_bc1", "advreal_bc2"), "T-SEA": ("tsea", "tsea_bc1", "tsea_bc2")}


def empty_cells():
    return {source: {target: None for target in MODELS} for source in MODELS}


def summarize(cells, label):
    white = [cells[s][s]["map50_pct"] for s in MODELS if cells[s][s] is not None]
    black = [cells[s][t]["map50_pct"] for s in MODELS for t in MODELS if s != t and cells[s][t] is not None]
    return {"label": label, "cells": cells, "whitebox_cells": len(white), "blackbox_cells": len(black),
            "whitebox_mean": math.fsum(white) / 6 if len(white) == 6 else None,
            "blackbox_mean": math.fsum(black) / 30 if len(black) == 30 else None,
            "complete": len(white) == 6 and len(black) == 30}


def cell(path, config, source, target, patch_hash):
    record = audit_metric(path, config, source, target, patch_hash)
    return {"map50_pct": record["map50_pct"], "conditional_asr": record["conditional_asr"],
            "relation": record["relation"], "metrics_path": str(path), "patch_path": record["patch"],
            "patch_sha256": record["patch_sha256"]}


def fmt(value):
    return "--" if value is None else f"{value:.2f}"


def delta(a, b):
    return None if a is None or b is None else a - b


def csv_file(path, rows, columns):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def table(path, caption, label, columns, rows):
    lines = [r"% Standalone export; requires booktabs and graphicx. Does not edit the manuscript.",
             r"\begin{table*}[t]", r"\centering", r"\renewcommand{\arraystretch}{1.12}",
             r"\caption{" + caption + "}", r"\label{" + label + "}", r"\resizebox{\textwidth}{!}{",
             r"\begin{tabular}{" + "l" * min(2, len(columns)) + "c" * (len(columns) - min(2, len(columns))) + "}",
             r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    master = read(RUN / "manifest.json")
    check_code(master)
    matrices, rows, comparison = {}, [], {}
    for selection in ("best", "last"):
        for method in METHODS:
            for variant in (("raw", "bc1", "bc2") if method == "tsea" else ("raw",)):
                key = method if variant == "raw" else method + "_" + variant
                cells = empty_cells()
                for source in MODELS:
                    directory = RUN / method / source
                    config = read(directory / "manifest.json")
                    item = next((p for p in config["patches"] if p["id"] == selection + "_" + variant), None)
                    if item is None:
                        continue
                    for target in MODELS:
                        path = directory / "eval" / target / item["id"] / "metrics.json"
                        if not (path.parent.parent / "complete.json").exists():
                            continue
                        cells[source][target] = cell(path, config, source, target, item["sha256"])
                        rows.append({"method": method, "variant": variant, "selection": selection, "source": source,
                                     "target": target, **cells[source][target]})
                result = summarize(cells, LABELS[key])
                matrices[key + "_" + selection] = result
                if selection == "best":
                    comparison[key] = result
    previous = read(master["previous_results"])
    assert previous["complete"]
    for key in ("advpatch", "p1", "p2", "cap", "cap_bc1", "cap_bc2"):
        cells = empty_cells()
        for source in MODELS:
            for target in MODELS:
                path = previous["matrices"][key]["cells"][source][target]["metrics_path"]
                record = read(path)
                cells[source][target] = cell(path, master, source, target, record["patch_sha256"])
        comparison[key] = summarize(cells, LABELS[key])
    advreal = read(ROOT / "yolov5-advreal-recolor-20260910/runs/advreal_recolor_20260910/matrix_results.json")
    assert advreal["complete"]
    for old, key in (("advreal2d", "advreal"), ("advreal_r1", "advreal_bc1"), ("advreal_r2", "advreal_bc2")):
        cells = empty_cells()
        for row in advreal["rows"]:
            if row["variant"] == old:
                source, target = row["source"], row["target"]
                cells[source][target] = cell(row["metrics_path"], master, source, target, row["patch_sha256"])
        comparison[key] = summarize(cells, LABELS[key])
        assert comparison[key]["complete"]
    comparison = {key: comparison[key] for key in LABELS}
    gray_controls = {}
    for target in MODELS:
        directory = RUN / "controls/gray/eval" / target
        if (directory / "complete.json").exists():
            record = audit_metric(directory / "gray/metrics.json", master, None, target, None)
            assert record["patch"] == "gray" and read(directory / "complete.json")["passed"]
            gray_controls[target] = {"map50_pct": record["map50_pct"], "conditional_asr": record["conditional_asr"],
                                     "metrics_path": str(directory / "gray/metrics.json")}
    gray_values = [gray_controls[target]["map50_pct"] if target in gray_controls else None for target in MODELS]
    gray_mean = math.fsum(gray_values) / 6 if len(gray_controls) == 6 else None
    complete = all(result["complete"] for result in matrices.values())
    result = {"complete": complete, "updated_utc": now(), "attack_cells_present": len(rows), "expected_attack_cells": 288,
              "best_cells_present": sum(row["selection"] == "best" for row in rows), "expected_best_cells": 144,
              "matrices": matrices, "comparison_best": comparison, "model_order": list(MODELS),
              "gray_controls": gray_controls, "gray_mean": gray_mean,
              "average_rule": "WB: six diagonals; BB: 30 off-diagonals; no incomplete means or zero imputation",
              "psnr_ssim": "Paused; not calculated or modified", "single_seed": master["seed"]}
    write(RUN / "matrix_results.json", result)
    tables = RUN / "tables"
    tables.mkdir(exist_ok=True)
    csv_file(tables / "gray_controls.csv", [{"target": target, **gray_controls.get(target, {})} for target in MODELS],
             ["target", "map50_pct", "conditional_asr", "metrics_path"])
    csv_file(tables / "all_new_cells.csv", rows, ["method", "variant", "selection", "source", "target", "map50_pct",
                                                   "conditional_asr", "relation", "metrics_path", "patch_path", "patch_sha256"])
    paired = []
    for family, keys in FAMILIES.items():
        raw = comparison[keys[0]]
        for index, key in enumerate(keys):
            data = comparison[key]
            timing = []
            if family == "T-SEA" and index:
                for source in MODELS:
                    cfg = read(RUN / "tsea" / source / "manifest.json")
                    item = next((x for x in cfg["patches"] if x["id"] == f"best_bc{index}"), None)
                    if item:
                        timing.append(item["recolor_ms_median"])
            paired.append({"base": family, "variant": "Original" if index == 0 else f"+PAC(Bc{index})", "method": LABELS[key],
                           "wb_map50": data["whitebox_mean"], "bb_map50": data["blackbox_mean"],
                           "delta_wb_pp": delta(data["whitebox_mean"], raw["whitebox_mean"]) if index else None,
                           "delta_bb_pp": delta(data["blackbox_mean"], raw["blackbox_mean"]) if index else None,
                           "extra_training_epochs": 0 if index else None,
                           "recolor_cpu_ms": math.fsum(timing) / 6 if len(timing) == 6 else None})
    csv_file(tables / "paired_best.csv", paired, list(paired[0]))
    table(tables / "paired_best.tex",
          r"Paired PAC adaptation under the shared INRIA protocol. WB averages six source-model diagonals; BB averages 30 off-diagonal transfers. Lower $mAP_{50}$ is better; positive $\Delta$ indicates an attack penalty. Checkpoints use training loss only. Missing runs are --; conversion time excludes I/O (CPU, two threads).",
          "tab:pac_paired_shared", ["Base attack", "Variant", "WB", "BB", r"$\Delta$WB", r"$\Delta$BB", "Extra epochs", "Recolor (ms)"],
          [[x["base"], x["variant"], fmt(x["wb_map50"]), fmt(x["bb_map50"]), fmt(x["delta_wb_pp"]), fmt(x["delta_bb_pp"]),
            "--" if x["extra_training_epochs"] is None else x["extra_training_epochs"], fmt(x["recolor_cpu_ms"])] for x in paired])
    tsea_rows = []
    for source in MODELS:
        baseline = comparison["tsea"]["cells"][source]
        raw_wb = baseline[source]["map50_pct"] if baseline[source] else None
        raw_bb = [baseline[t]["map50_pct"] for t in MODELS if t != source and baseline[t]]
        raw_bb = math.fsum(raw_bb)/5 if len(raw_bb) == 5 else None
        cfg = read(RUN / "tsea" / source / "manifest.json")
        for index, key in enumerate(("tsea", "tsea_bc1", "tsea_bc2")):
            current = comparison[key]["cells"][source]
            wb = current[source]["map50_pct"] if current[source] else None
            bb = [current[t]["map50_pct"] for t in MODELS if t != source and current[t]]
            bb = math.fsum(bb)/5 if len(bb) == 5 else None
            item = next((x for x in cfg["patches"] if x["id"] == f"best_bc{index}"), None) if index else None
            tsea_rows.append({"source": source, "variant": "T-SEA" if index == 0 else f"T-SEA+PAC(Bc{index})",
                              "wb_map50": wb, "bb_map50_5": bb, "delta_wb_pp": delta(wb,raw_wb) if index else None,
                              "delta_bb_pp": delta(bb,raw_bb) if index else None,
                              "recolor_ms": item["recolor_ms_median"] if item else None})
    csv_file(tables / "tsea_paired_best.csv", tsea_rows, list(tsea_rows[0]))
    table(tables / "tsea_paired_best.tex",
          r"T-SEA with and without PAC recoloring under identical patch constraints. WB is evaluated on the source model; BB averages the other five detectors. Positive $\Delta$ denotes an increase in $mAP_{50}$ after recoloring. Both palettes are fixed in advance; no recoloring retraining is performed.",
          "tab:tsea_pac_shared", ["Source", "Variant", "WB", "BB Avg.", r"$\Delta$WB", r"$\Delta$BB", "Recolor (ms)"],
          [[MODEL_LABELS[x["source"]], x["variant"], *(fmt(x[c]) for c in ("wb_map50", "bb_map50_5", "delta_wb_pp", "delta_bb_pp", "recolor_ms"))] for x in tsea_rows])
    clean = [read(master["clean_reference_paths"][target])["map50_pct"] for target in MODELS]
    white = [["Clean", *(fmt(v) for v in clean), fmt(math.fsum(clean)/6)],
             ["Gray", *(fmt(v) for v in gray_values), fmt(gray_mean)]]
    for key, data in comparison.items():
        values = [data["cells"][source][source]["map50_pct"] if data["cells"][source][source] else None for source in MODELS]
        white.append([LABELS[key], *(fmt(value) for value in values), fmt(data["whitebox_mean"])])
    table(tables / "whitebox_shared_best.tex", r"White-box $mAP_{50}$ under one shared INRIA protocol (lower is better). Gray is an unoptimized occlusion control with the same patch budget and placement. This separate export does not replace the manuscript's historical table. NAP optimizes a frozen BigGAN latent; asterisked PAC-R variants use AdvReal and differ from the original random PAC-R controls.",
          "tab:shared_whitebox", ["Method", *(MODEL_LABELS[t] for t in MODELS), "WB Avg."], white)
    for source in MODELS:
        black = []
        for key, data in comparison.items():
            values = [data["cells"][source][t]["map50_pct"] if data["cells"][source][t] else None for t in MODELS]
            offdiag = [value for t, value in zip(MODELS,values) if t != source and value is not None]
            bb = math.fsum(offdiag)/5 if len(offdiag) == 5 else None
            black.append([LABELS[key], *(fmt(value) for value in values), fmt(bb)])
        table(tables / f"transfer_shared_best_{source}.tex", r"Transfer $mAP_{50}$ from " + MODEL_LABELS[source] + r" under the shared protocol. The source-model column is a white-box control; BB Avg. excludes that column.",
              "tab:shared_transfer_" + source, ["Method", *(MODEL_LABELS[t] for t in MODELS), "BB Avg."], black)
    summary = []
    for key,data in comparison.items():
        summary.append({"method": LABELS[key], "wb_map50": data["whitebox_mean"], "bb_map50": data["blackbox_mean"],
                        "wb_cells": data["whitebox_cells"], "bb_cells": data["blackbox_cells"]})
    csv_file(tables / "comparison_best.csv", summary, list(summary[0]))
    final_rows = []
    for method in ("tsea", "tsea_bc1", "tsea_bc2", "nap"):
        for selection in ("best", "last"):
            data = matrices[method + "_" + selection]
            final_rows.append({"method": LABELS[method], "checkpoint": selection, "wb_map50": data["whitebox_mean"], "bb_map50": data["blackbox_mean"]})
    csv_file(tables / "best_final_sensitivity.csv", final_rows, list(final_rows[0]))
    text = ["# T-SEA + PAC 与 NAP 统一协议实验", "", f"更新时间：{now()}；新增攻击单元 {len(rows)}/288（best/final 各 144）。",
            "", "mAP50 越低越好；白盒均值含 6 个对角单元，黑盒均值含 30 个非对角单元。缺失项留空，不计算不完整均值。",
            "所有 best 按训练损失选择；最后一轮单列。原论文 Table 1/2 未修改。PSNR/SSIM 保持暂停。", "",
            "| 方法 | 白盒均值 | 黑盒均值 | 已完成 WB/BB |", "|---|---:|---:|---:|"]
    text += [f"| {x['method']} | {fmt(x['wb_map50'])} | {fmt(x['bb_map50'])} | {x['wb_cells']}/6，{x['bb_cells']}/30 |" for x in summary]
    text += ["", f"同预算 Gray 已完成 {len(gray_controls)}/6 个检测器，mAP50 均值 {fmt(gray_mean)}；详见 tables/gray_controls.csv 和新增白盒表。Gray 属于未优化的遮挡控制，不计作六源迁移攻击矩阵。",
             f"当前协议下 Gray 在 YOLOv5s 上为 {fmt(gray_controls.get('yolov5s', {}).get('map50_pct'))}，原稿为 85.30；结合 clean 差异，新旧实验不能直接拼接为同条件排名。当前 Gray 已按共同面积、中心贴图和全部288张测试图重新评估。",
             "T-SEA 表格：tables/tsea_paired_best.tex；多基线配对表：tables/paired_best.tex。",
             "两种色板均预先固定，不能根据测试结果逐模型挑选。单种子结果不支持显著性或对任意方法的普遍兼容性结论。",
             "NAP 是自然外观目标的对照；T-SEA 是迁移攻击对照。图片相似度或单张示例均不能单独证明隐蔽性。",
             "CAP 的训练最优与最后一轮结果差异较大，需同时阅读前一实验的 best/final 分析，不能只凭 best 排名概括其强弱。"]
    (RUN / "report.md").write_text("\n".join(text)+"\n")
    if complete:
        for method in METHODS:
            for source in MODELS:
                cfg = read(RUN / method / source / "manifest.json")
                assert read(RUN / method / source / "train_complete.json")["passed"]
                for item in cfg["patches"]:
                    if item["path"]:
                        assert sha(item["path"]) == item["sha256"]
        write(RUN / "evaluation_audit.json", {"passed": True, "attack_cells": len(rows), "new_best_cells": 144,
              "new_final_cells": 144, "wb_cells_per_method": 6, "bb_cells_per_method": 30, "completed_utc": now()})
    # Refresh the matched gallery when a source pipeline finishes. Incomplete
    # training checkpoints never enter the appearance comparison.
    pending_views = []
    for source in MODELS:
        finished = [method for method in METHODS if (RUN / method / source / "source_complete.json").exists()]
        path = RUN / "appearance_comparison/sources" / source / "manifest.json"
        rendered = read(path)["new_methods_complete"] if path.exists() else None
        if rendered != finished:
            pending_views.append(source)
    if pending_views:
        command = [PYTHON, "-B", str(RUN / "prepare_appearance.py"), "--sources", ",".join(pending_views)]
        subprocess.run(command, cwd=PROJECT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
    print({"new_attack_cells": len(rows), "complete": complete}, flush=True)


if __name__ == "__main__":
    main()
