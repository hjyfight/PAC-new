"""Compare final PNAP with PAC-P1/P2 derived from epoch-200 AdvPatch last.pt."""
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
from zoneinfo import ZoneInfo

from common import MODELS, PROJECT, ROOT, RUN, read, runtime, sha, write

runtime()
import numpy as np
import torch
from utils.patch import BASE_COLORS, load_patch, recolor

OUTPUT = RUN / "final_comparison_20261108"
PAC = ROOT / "experiments/baselines/advpatch_cap_completion_20260911"
REFERENCE = ROOT / "experiments/ours/yolov5-advreal-recolor-20260910/runs/advreal_recolor_20260910"
OLD = ROOT / "experiments/ours/yolov5-advpad-20260908/runs/official_pad3_20260908"
PYTHON = "/datacc05/huotianlin/miniconda3/envs/pac/bin/python"
FIELDS = ("batch_size", "imgsz", "preprocess", "patch_size", "patch_area_fraction",
          "asr_conf", "asr_iou", "conf_thres", "nms_iou", "max_det")


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def prepare(config):
    from run_experiments import verify
    verify(config, weights=False)
    OUTPUT.mkdir(exist_ok=True)
    master = read(PAC / "manifest.json")
    reference = read(REFERENCE / "manifest.json")
    entries, pending = [], []
    for source in MODELS:
        if source == "yolov5s":
            folder = OLD / "patches/ap_s000_advpatch"
            checked = read(OLD / "training_verification.json")
            assert checked["passed"] and checked["trainings"]["ap_s000_advpatch"]["epochs"] == 200
            with (folder / "results.csv").open() as stream:
                history = list(csv.DictReader(stream))
            assert len(history) == 200 and int(history[-1]["epoch"]) == 200
            source_config = reference
        else:
            folder = PAC / "advpatch" / source / "patches/advpatch"
            source_config = read(PAC / "advpatch" / source / "manifest.json")
            complete = read(PAC / "advpatch" / source / "train_complete.json")
            assert complete["passed"] and complete["epochs"] == 200
            assert sha(folder / "last.pt") == complete["checkpoints"]["last"]["sha256"]
        raw_last, raw_best = load_patch(folder / "last.pt"), load_patch(folder / "best.pt")
        assert raw_last.shape == (3, 300, 300) and torch.isfinite(raw_last).all()
        for variant in ("p1", "p2"):
            old_item = next(p for p in source_config["patches"] if p["id"] == variant)
            assert sha(old_item["path"]) == old_item["sha256"]
            previous_patch = load_patch(old_item["path"])
            colors = BASE_COLORS["bc" + variant[-1]]
            best_error = float((recolor(raw_best, colors) - previous_patch).abs().max())
            assert best_error < 2e-7, (source, variant, "best recolor mismatch", best_error)
            patch = recolor(raw_last, colors).contiguous()
            expected = np.einsum("khw,kc->chw", raw_last.numpy().astype(np.float64),
                                 colors.astype(np.float64) / 255)
            formula_error = float(np.max(np.abs(patch.numpy() - expected)))
            assert formula_error < 2e-7 and torch.isfinite(patch).all()
            identical = torch.equal(patch, previous_patch)
            identity = f"{source}_{variant}_final"
            destination = OUTPUT / "patches" / source / (variant + "_final.pt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            metadata = dict(patch=patch, patch_size=300, source_path=str(folder / "last.pt"),
                            source_sha256=sha(folder / "last.pt"), source_epoch=200,
                            optimization_source=source, base_colors_255=colors.tolist(),
                            recolor_training_epochs=0, checkpoint_selection="final")
            if destination.exists():
                existing = torch.load(destination, map_location="cpu", weights_only=True)
                assert existing["source_sha256"] == metadata["source_sha256"]
                assert torch.equal(existing["patch"], patch)
            else:
                torch.save(metadata, destination)
            item = dict(id=identity, label=f"PAC-P{variant[-1]} final ({source})",
                        path=str(destination), sha256=sha(destination), optimization_source=source)
            old_paths = {target: (master["advpatch_reference_yolov5s"][variant][target]["path"]
                         if source == "yolov5s" else str(PAC / "advpatch" / source / "eval" /
                                                        target / variant / "metrics.json"))
                         for target in MODELS}
            entries.append(dict(source=source, variant=variant, item=item, source_epoch=200,
                                source_last=str(folder / "last.pt"), source_last_sha256=metadata["source_sha256"],
                                bitwise_equal_to_best=identical, best_reconstruction_max_error=best_error,
                                final_formula_max_error=formula_error, best_patch_sha256=old_item["sha256"],
                                best_metrics=old_paths))
            if not identical:
                pending.append(item)
    manifest = {**config, "method": "PAC-P1/P2 final comparison", "epochs": 200,
                "checkpoint_selection": "last epoch, no selection by test scores",
                "patches": [dict(id="clean", label="Clean", path=None, optimization_source=None), *pending]}
    write(OUTPUT / "manifest.json", manifest)
    write(OUTPUT / "patch_provenance.json", entries)
    print(json.dumps(dict(stage="prepared", final_patches=len(entries),
                          patches_requiring_inference=len(pending),
                          patches_reusing_bitwise_identical_best=len(entries)-len(pending))), flush=True)
    return entries, manifest


def evaluate_target(target, gpu, manifest):
    directory = OUTPUT / "eval" / target
    missing = [p["id"] for p in manifest["patches"]
               if not (directory / p["id"] / "metrics.json").exists()]
    if missing:
        if "clean" not in missing:
            missing.insert(0, "clean")
        command = [PYTHON, "-B", "eval_transfer.py", "--manifest", str(OUTPUT / "manifest.json"),
                   "--detector", target, "--device", "cuda:0", "--batch-size", "8",
                   "--workers", "4", "--patches", ",".join(missing)]
        log_path = OUTPUT / "logs" / (target + ".log")
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("a") as log:
            result = subprocess.run(command, cwd=PROJECT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(f"{target}: evaluator exit {result.returncode}; {log_path}")
    print(json.dumps(dict(stage="target_complete", target=target, gpu=gpu)), flush=True)


def audit_metric(path, config, source, target, expected_hash):
    item = read(path)
    assert item["optimization_source"] == source and item["detector"] == target
    assert item["patch_sha256"] == expected_hash
    assert item["weights_sha256"] == config["detectors"][target]["sha256"]
    assert item["images"] == 288 and item["targets"] == 589 and not item["eot"] and not item["rect"]
    assert item["relation"] == ("whitebox" if source == target else "blackbox_transfer")
    for key in FIELDS:
        assert item[key] == config[key], (str(path), key)
    clean_dir = Path(path).parents[1] / "clean"
    clean = read(clean_dir / "detected_gt.json")
    baseline = RUN / "sources/yolov2/eval" / target / "clean"
    assert clean == read(baseline / "detected_gt.json")
    clean_metrics = read(clean_dir / "metrics.json")
    reference = read(baseline / "metrics.json")
    for key in ("map50", "map50_95", "precision", "recall", "conditional_asr_denominator"):
        assert clean_metrics[key] == reference[key], (str(path), "clean", key)
    attacked = read(Path(path).parent / "detected_gt.json")
    assert len(clean) == len(attacked) == 288
    denominator = numerator = targets = 0
    for before, after in zip(clean, attacked, strict=True):
        assert before["image"] == after["image"]
        a, b = before["detected_gt"], after["detected_gt"]
        assert len(a) == len(b)
        denominator += sum(a)
        targets += len(a)
        numerator += sum(x and not y for x, y in zip(a, b))
    assert targets == 589 and denominator == item["conditional_asr_denominator"]
    assert numerator == item["conditional_asr_numerator"] and item["conditional_asr"] == numerator / denominator
    return item


def summarize(config, entries):
    rows, checks = [], []
    for entry in entries:
        source, variant = entry["source"], entry["variant"]
        for target in MODELS:
            for selection in ("best", "final"):
                reused = selection == "final" and entry["bitwise_equal_to_best"]
                use_best = selection == "best" or reused
                path = (Path(entry["best_metrics"][target]) if use_best else
                        OUTPUT / "eval" / target / entry["item"]["id"] / "metrics.json")
                expected_hash = entry["best_patch_sha256"] if use_best else entry["item"]["sha256"]
                record = audit_metric(path, config, source, target, expected_hash)
                rows.append(dict(method="PAC-P" + variant[-1], selection=selection, source=source,
                                 target=target, relation="whitebox" if source == target else "blackbox",
                                 map50_pct=record["map50_pct"], asr_pct=100 * record["conditional_asr"],
                                 reused_identical_best=reused, metrics_file=str(path)))
                checks.append(dict(path=str(path), sha256=sha(path)))
    for source in MODELS:
        for target in MODELS:
            for selection in ("best", "final"):
                path = RUN / "sources" / source / "eval" / target / ("pnap_" + selection) / "metrics.json"
                expected_hash = sha(RUN / "sources" / source / "patches" / (selection + ".pt"))
                record = audit_metric(path, config, source, target, expected_hash)
                rows.append(dict(method="PNAP", selection=selection, source=source, target=target,
                                 relation="whitebox" if source == target else "blackbox",
                                 map50_pct=record["map50_pct"], asr_pct=100 * record["conditional_asr"],
                                 reused_identical_best=False, metrics_file=str(path)))
                checks.append(dict(path=str(path), sha256=sha(path)))
    summaries, by_source = [], []
    for method in ("PNAP", "PAC-P1", "PAC-P2"):
        for selection in ("best", "final"):
            for source in (*MODELS, "ALL"):
                cells = [r for r in rows if r["method"] == method and r["selection"] == selection
                         and (source == "ALL" or r["source"] == source)]
                wb = [r for r in cells if r["relation"] == "whitebox"]
                bb = [r for r in cells if r["relation"] == "blackbox"]
                assert len(wb) == (6 if source == "ALL" else 1)
                assert len(bb) == (30 if source == "ALL" else 5)
                result = dict(method=method, selection=selection, source=source,
                              wb_map50_pct=mean(r["map50_pct"] for r in wb),
                              wb_asr_pct=mean(r["asr_pct"] for r in wb),
                              bb_map50_pct=mean(r["map50_pct"] for r in bb),
                              bb_asr_pct=mean(r["asr_pct"] for r in bb))
                (summaries if source == "ALL" else by_source).append(result)
    save_csv(OUTPUT / "comparison_summary.csv", summaries)
    save_csv(OUTPUT / "source_summary.csv", by_source)
    save_csv(OUTPUT / "cells.csv", rows)
    write(OUTPUT / "audit.json", dict(passed=True, checked_metric_references=len(checks),
          unique_metric_files=len({c["path"] for c in checks}), final_attack_cells=108,
          pac_final_epoch=200, pnap_final_epoch=500, checked=checks,
          checks=["patch and detector hashes", "final epoch provenance", "recolor reconstruction",
                  "identical tensor proof before reuse", "evaluation protocol", "clean replay", "per-image ASR recount"]))
    lines = ["# PNAP 与 PAC-P1/P2：最后一轮对比", "",
             f"核查时间：{datetime.now(ZoneInfo('Asia/Shanghai')):%Y-%m-%d %H:%M:%S}（北京时间）。", "",
             "PNAP-final取第500轮；PAC-P1/P2-final由各源AdvPatch第200轮last.pt分别经Bc1/Bc2固定配色生成。两者评估协议相同，训练轮数不相同。YOLOv5s沿用原主表s000种子。", "",
             "INRIA Test288，589个人体GT；batch=8，输入640，补丁300×300，人框中心25%面积，无测试EOT。单位均为%；mAP50越低、ASR越高表示攻击越强。白盒6单元、黑盒30单元分别等权平均。", "",
             table(["方法", "补丁", "白盒mAP50↓", "白盒ASR↑", "黑盒mAP50↓", "黑盒ASR↑"],
                   [[r["method"], r["selection"], *[f'{r[k]:.2f}' for k in
                     ("wb_map50_pct", "wb_asr_pct", "bb_map50_pct", "bb_asr_pct")]] for r in summaries]), "",
             "P系列final与已有best补丁逐元素完全一致时复用原评估，其他补丁重新评估；复用证明及补丁来源见patch_provenance.json。没有按测试分数选择补丁。", ""]
    for field, label in (("map50_pct", "mAP50↓"), ("asr_pct", "ASR↑")):
        for method in ("PNAP", "PAC-P1", "PAC-P2"):
            lines += [f"## {method} final：{label}", "",
                      table(["源→目标", *MODELS], [[source, *[f'{next(r[field] for r in rows if r["method"] == method and r["selection"] == "final" and r["source"] == source and r["target"] == target):.2f}' for target in MODELS]] for source in MODELS]), ""]
    lines += ["逐源汇总：[source_summary.csv](source_summary.csv)；逐项数据：[cells.csv](cells.csv)；核验：[audit.json](audit.json)。", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines))
    print(json.dumps(dict(stage="complete", report=str(OUTPUT / "report.md"), summaries=summaries), ensure_ascii=False), flush=True)


def main():
    torch.set_num_threads(2)
    config = read(RUN / "protocol.json")
    entries, manifest = prepare(config)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(evaluate_target, target, gpu, manifest) for gpu, target in enumerate(MODELS)]
        for future in as_completed(futures):
            future.result()
    summarize(config, entries)


if __name__ == "__main__":
    main()
