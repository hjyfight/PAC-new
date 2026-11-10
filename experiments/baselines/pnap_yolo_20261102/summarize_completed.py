"""Audit completed 500-epoch PNAP runs and write a dated progress report."""
import csv
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
from statistics import mean, median
import subprocess
from zoneinfo import ZoneInfo

from common import MODELS, ROOT, RUN, read, runtime, sha, write


NAMES = dict(yolov2="YOLOv2", yolov3="YOLOv3", yolov4="YOLOv4",
             yolov5s="YOLOv5s", yolov5m="YOLOv5m", fasterrcnn="Faster R-CNN")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def save_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    runtime()
    import torch
    from run_experiments import summarize, verify

    config = read(RUN / "protocol.json")
    verify(config, weights=False)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    output = RUN / "reports" / now.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    protocol_hash = sha(RUN / "protocol.json")
    adapter_hashes = {name: sha(RUN / name) for name in
                      ("pnap.py", "run_experiments.py", "common.py")}
    processes = subprocess.check_output(
        ["ps", "-u", "huotianlin", "-o", "pid=,args="], text=True)
    progress, audited, results, summaries = [], {}, [], []
    for source in MODELS:
        folder = RUN / "sources" / source
        status = read(folder / "status.json")
        job = read(RUN / "jobs" / (source + ".json"))
        trained = (folder / "train_complete.json").exists()
        evaluated = (folder / "evaluation_complete.json").exists()
        history = []
        for line in (folder / "history.jsonl").read_text().splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                # A running trainer may currently be appending the last line.
                require(not trained, f"Incomplete history in completed run: {source}")
        recent = history[-11:]
        durations = [b["seconds"] - a["seconds"] for a, b in zip(recent, recent[1:])]
        seconds_per_epoch = median(durations)
        eta = datetime.fromtimestamp((folder / "status.json").stat().st_mtime,
                                    ZoneInfo("Asia/Shanghai")) + timedelta(
            seconds=(config["epochs"] - status["epoch"]) * seconds_per_epoch)
        matching = [line.strip() for line in processes.splitlines()
                    if f"train --model {source}" in line and "pnap_yolo_20261102" in line]
        progress.append(dict(source=source, epochs=status["epoch"], total_epochs=config["epochs"],
                             steps=status["steps"], gpu=job["gpu"], job=job,
                             train_complete=trained, evaluation_complete=evaluated,
                             training_processes=matching, median_epoch_seconds=seconds_per_epoch,
                             estimated_training_finish=eta.isoformat() if not trained else None))
        if not (trained and evaluated):
            continue
        train_done = read(folder / "train_complete.json")
        eval_done = read(folder / "evaluation_complete.json")
        require(train_done["passed"] and not train_done["smoke"] and
                train_done["epochs"] == 500 and train_done["steps"] == 38500,
                f"Invalid training completion: {source}")
        require(eval_done["passed"] and not eval_done["smoke"] and
                eval_done["evaluations"] == 18, f"Invalid evaluation completion: {source}")
        require([item["epoch"] for item in history] == list(range(1, 501)),
                f"Incomplete/duplicate training history: {source}")
        require(all(item["steps"] == item["epoch"] * 77 for item in history),
                f"Unexpected training step count: {source}")
        expected_best = min(history, key=lambda item: item["loss"][0])["epoch"]
        manifest = read(folder / "manifest.json")
        require(all(manifest[key] == value for key, value in config.items()),
                f"Manifest protocol mismatch: {source}")
        patches, patch_records = {}, {}
        for version in ("best", "final"):
            path = folder / "patches" / (version + ".pt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            epoch = expected_best if version == "best" else 500
            require(checkpoint["epoch"] == epoch and checkpoint["steps"] == epoch * 77,
                    f"Wrong checkpoint selection: {source}/{version}")
            require(checkpoint["optimization_source"] == source and not checkpoint["smoke"] and
                    checkpoint["protocol_sha256"] == protocol_hash and
                    checkpoint["adapter_sha256"] == adapter_hashes,
                    f"Checkpoint provenance mismatch: {source}/{version}")
            patch = checkpoint["patch"]
            require(tuple(patch.shape) == (3, 300, 300) and torch.isfinite(patch).all().item()
                    and patch.min().item() >= 0 and patch.max().item() <= 1,
                    f"Invalid patch tensor: {source}/{version}")
            record = dict(path=str(path), sha256=sha(path), epoch=epoch, steps=epoch * 77)
            require(next(item for item in manifest["patches"] if item["id"] == "pnap_" + version)
                    ["sha256"] == record["sha256"], f"Manifest patch mismatch: {source}/{version}")
            patches[version] = record
        for target in MODELS:
            evaluation = folder / "eval" / target
            clean = read(evaluation / "clean/metrics.json")
            clean_gt = read(evaluation / "clean/detected_gt.json")
            reference_dir = ROOT / "experiments/NCQ/ncq_random3_20261025/sources" / target / "eval" / target / "clean"
            reference = read(reference_dir / "metrics.json")
            require(clean_gt == read(reference_dir / "detected_gt.json"),
                    f"Clean per-image replay mismatch: {source}/{target}")
            for key in ("map50", "map50_95", "precision", "recall", "conditional_asr_numerator",
                        "conditional_asr_denominator", "images", "targets"):
                require(clean[key] == reference[key], f"Clean replay mismatch: {source}/{target}/{key}")
            for version in ("clean", "best", "final"):
                patch_id = "clean" if version == "clean" else "pnap_" + version
                path = evaluation / patch_id / "metrics.json"
                item = read(path)
                for key in ("images", "targets", "batch_size", "imgsz", "preprocess", "patch_size",
                            "patch_area_fraction", "asr_conf", "asr_iou", "conf_thres", "nms_iou", "max_det"):
                    expected = config[{"images": "test_images", "targets": "test_targets"}.get(key, key)]
                    require(item[key] == expected, f"Protocol mismatch: {source}/{target}/{patch_id}/{key}")
                require(item["detector"] == target and item["patch_id"] == patch_id and
                        item["weights_sha256"] == config["detectors"][target]["sha256"] and
                        not item["eot"] and not item["rect"], f"Metrics identity mismatch: {path}")
                detections = read(evaluation / patch_id / "detected_gt.json")
                require(len(detections) == len(clean_gt) == 288, f"Wrong image count: {path}")
                denominator, numerator, targets = 0, 0, 0
                for before, after in zip(clean_gt, detections, strict=True):
                    a, b = before["detected_gt"], after["detected_gt"]
                    require(before["image"] == after["image"] and len(a) == len(b),
                            f"Image/GT alignment mismatch: {path}")
                    targets += len(a)
                    denominator += sum(a)
                    numerator += sum(x and not y for x, y in zip(a, b))
                require(targets == 589 and denominator == item["conditional_asr_denominator"] and
                        numerator == item["conditional_asr_numerator"] and
                        item["conditional_asr"] == numerator / denominator,
                        f"ASR recount mismatch: {path}")
                require(abs(item["map50_pct"] - item["map50"] * 100) < 1e-10,
                        f"mAP unit mismatch: {path}")
                if version == "clean":
                    require(item["patch_sha256"] is None, f"Unexpected clean patch: {path}")
                    continue
                require(item["patch_sha256"] == patches[version]["sha256"] and
                        item["optimization_source"] == source and
                        item["relation"] == ("whitebox" if source == target else "blackbox_transfer"),
                        f"Patch/relation mismatch: {path}")
                results.append(dict(source=source, target=target, checkpoint=version,
                                    checkpoint_epoch=patches[version]["epoch"],
                                    relation="whitebox" if source == target else "blackbox",
                                    clean_map50_pct=clean["map50_pct"], map50_pct=item["map50_pct"],
                                    conditional_asr_pct=item["conditional_asr"] * 100,
                                    asr_numerator=numerator, asr_denominator=denominator,
                                    patch_sha256=item["patch_sha256"], metrics_file=str(path)))
                patch_records[str(path)] = sha(path)
        audited[source] = dict(passed=True, training_epochs=500, patches=patches,
                               checked_metric_files=18, attack_evaluations=12,
                               metrics_sha256=patch_records)
        for version in ("best", "final"):
            cells = [r for r in results if r["source"] == source and r["checkpoint"] == version]
            wb = next(r for r in cells if r["relation"] == "whitebox")
            bb = [r for r in cells if r["relation"] == "blackbox"]
            summaries.append(dict(source=source, checkpoint=version, checkpoint_epoch=patches[version]["epoch"],
                                  clean_wb_map50_pct=wb["clean_map50_pct"], wb_map50_pct=wb["map50_pct"],
                                  wb_asr_pct=wb["conditional_asr_pct"],
                                  bb_mean_map50_pct=mean(r["map50_pct"] for r in bb),
                                  bb_mean_asr_pct=mean(r["conditional_asr_pct"] for r in bb), bb_targets=len(bb)))
    require(audited, "No fully completed, audited 500-epoch sources available")
    with (RUN / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summarize()
    audit = dict(timestamp=now.isoformat(), passed=True, protocol_sha256=protocol_hash,
                 completed_sources=list(audited), attack_evaluations=len(results),
                 expected_attack_evaluations=72, formal_results_complete=len(audited) == 6,
                 checks=["dataset, detector and frozen source hashes", "500 epochs / 38500 steps",
                         "best selected by minimum training loss", "checkpoint identity and finite patch tensor",
                         "all evaluation protocol fields", "clean baseline metrics and per-image detections",
                         "ASR recounted from per-image GT detections"],
                 inference_rerun=False, progress=progress, sources=audited)
    write(output / "audit.json", audit)
    save_csv(output / "completed_metrics.csv", results)
    save_csv(output / "source_summary.csv", summaries)
    lines = ["# PNAP 训练进度及已完成权重评估汇总", "",
             f"核查时间：{now:%Y-%m-%d %H:%M:%S}（北京时间）。仅纳入已完成500轮且评估齐全的源模型。", "",
             f"**已完成并核验：{len(audited)}/6个源模型，{len(results)}/72项攻击评估。** 每源含best/final各6个目标检测器，以及6次clean重放。", "",
             "已有自动评估通过完整性和来源校验，本次复用其结果并重算逐图ASR计数，未重复运行GPU推理。", "",
             "## 训练进度", "",
             table(["源模型", "已完成轮数", "GPU", "状态", "近10轮中位耗时", "预计训练完成（北京时间）"],
                   [[NAMES[p["source"]], f'{p["epochs"]}/500', p["gpu"],
                     "训练及评估完成" if p["source"] in audited else
                     ("训练完成，评估中/待评估" if p["train_complete"] else
                      ("训练中" if p["training_processes"] else "未发现训练进程，需检查")),
                     f'{p["median_epoch_seconds"]:.1f}秒/轮',
                     datetime.fromisoformat(p["estimated_training_finish"]).strftime("%m-%d %H:%M")
                     if p["estimated_training_finish"] else "已完成"] for p in progress]), "",
             "预计时间按近期训练速度外推，仅供参考，不含训练结束后的六目标评估时间；共享GPU负载变化会影响估计。", "",
             "## 评估口径", "",
             "INRIA Test288，共589个人体GT；batch=8，输入640，补丁300×300，位于人框中心、面积占人框25%；测试时无EOT。", "",
             "mAP50与ASR均以百分数表示。攻击效果越强，mAP50越低、ASR越高。ASR分母为clean下被正确检出的GT，置信度与IoU阈值均为0.5。", "",
             "best在完整500轮训练后按最低轮均训练总损失选出，因此best所属轮数可能早于500；final固定来自第500轮，二者均来自完成的正式训练。", "",
             "## 已完成源模型汇总", "",
             table(["源模型", "权重", "权重轮数", "Clean白盒mAP50", "白盒mAP50↓", "白盒ASR↑", "5目标黑盒平均mAP50↓", "5目标黑盒平均ASR↑"],
                   [[NAMES[r["source"]], r["checkpoint"], r["checkpoint_epoch"],
                     *[f'{r[k]:.2f}' for k in ("clean_wb_map50_pct", "wb_map50_pct", "wb_asr_pct",
                                              "bb_mean_map50_pct", "bb_mean_asr_pct")]] for r in summaries]), "",
             "黑盒平均是每个已完成源在另外5个检测器上的等权平均；不代表全部6源的30个黑盒单元。六源未齐，不计算总6WB/30BB均值，也不与完整六源方法均值直接排名。"]
    for version in ("best", "final"):
        for field, label in (("map50_pct", "mAP50（%）↓"), ("conditional_asr_pct", "ASR（%）↑")):
            lines += ["", f"## {version}：{label}", "",
                      "行是训练源，列是目标检测器；加粗为白盒。", "",
                      table(["源→目标", *[NAMES[m] for m in MODELS]],
                            [[NAMES[source], *[("**" if source == target else "") +
                              f'{next(r[field] for r in results if r["source"] == source and r["target"] == target and r["checkpoint"] == version):.2f}' +
                              ("**" if source == target else "") for target in MODELS]] for source in audited])]
    lines += ["", "## 核验与文件", "",
              "数据、检测器、冻结源码、补丁SHA256、checkpoint来源、best选择轮数、评估参数、每项288图/589GT均通过；所有clean重放与已有基线一致，逐图重算的ASR分子/分母与指标文件一致。", "",
              "- [逐单元完整精度CSV](completed_metrics.csv)",
              "- [源模型best/final汇总CSV](source_summary.csv)",
              "- [进度、权重及核验记录](audit.json)",
              "- [持续更新的正式结果表](../../tables/comparison.csv)",
              "- [统一协议与适配说明](../../PROTOCOL.md)", "",
              "本报告为核查时快照；正在运行的控制器会在各源达到500轮后自动评估best/final并更新原正式结果表。"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write(RUN / "reports/latest.json", dict(report=str(output / "report.md"), audit=str(output / "audit.json"),
                                            timestamp=now.isoformat(), completed_sources=list(audited)))
    print(json.dumps(dict(report=str(output / "report.md"), summaries=summaries, progress=progress),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
