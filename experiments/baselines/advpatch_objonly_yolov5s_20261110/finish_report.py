"""Audit the shared evaluation, summarize AP/ASR, and render actual checkpoints."""

import csv
import json
import math
from pathlib import Path

from run_experiment import ROOT, RUN, PROJECT, REFERENCE, read, write, sha

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from utils.patch import load_patch

VIS = ROOT / "experiments/analysis/pac_patch_visualizations_20260910/objonly_yolov5s_20261110"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def render_grid(items, path, header, column_count=3):
    tile, gap, margin, top, foot = 320, 28, 30, 115, 85
    row_count = math.ceil(len(items) / column_count)
    width = 2 * margin + column_count * tile + (column_count - 1) * gap
    height = top + row_count * (tile + foot) + (row_count - 1) * 18 + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), header, fill="black", font=font(25))
    draw.text((margin, 61), "Native colors; no brightness or contrast adjustment", fill="#555555", font=font(18))
    for i, item in enumerate(items):
        col, row = i % column_count, i // column_count
        x, y = margin + col * (tile + gap), top + row * (tile + foot + 18)
        im = Image.open(item["png"]).convert("RGB")
        canvas.paste(im.resize((tile, tile), Image.Resampling.NEAREST), (x, y))
        draw.text((x, y + tile + 8), item["title"], fill="black", font=font(20))
        draw.text((x, y + tile + 37), item["subtitle"], fill="#444444", font=font(17))
    canvas.save(path)


def main():
    torch.set_num_threads(2)
    config = read(RUN / "manifest.json")
    assert read(RUN / "training_complete.json")["passed"]
    for name, digest in config["source_code_sha256"].items():
        assert sha(PROJECT / name) == digest, name
    previous_data = read(config["source_dataset_audit"])
    for split in previous_data["splits"].values():
        for item in split:
            assert sha(item["image"]) == item["image_sha256"]
            assert sha(item["label"]) == item["label_sha256"]

    records = {}
    for item in config["patches"]:
        metric = read(RUN / "eval/yolov5s" / item["id"] / "metrics.json")
        assert metric["images"] == 288 and metric["targets"] == 589
        assert metric["weights_sha256"] == config["detectors"]["yolov5s"]["sha256"]
        assert metric["eot"] is False and metric["rect"] is False
        for field in ("batch_size", "preprocess", "imgsz", "patch_size", "patch_area_fraction",
                      "conf_thres", "nms_iou", "max_det", "asr_conf", "asr_iou"):
            assert metric[field] == config[field], (item["id"], field)
        if item["id"] not in ("clean", "gray"):
            assert sha(item["path"]) == item["sha256"] == metric["patch_sha256"]
        assert all(math.isfinite(metric[k]) for k in ("map50", "map50_95", "conditional_asr", "precision", "recall"))
        records[item["id"]] = metric

    clean = records["clean"]
    previous_clean = read(config["reference_clean_metrics"])
    assert abs(clean["map50_pct"] - previous_clean["map50_pct"]) < 1e-3
    assert clean["conditional_asr_denominator"] == previous_clean["conditional_asr_denominator"] == 578
    replay = {}
    for variant, old_method in (("raw", "advpatch"), ("p1", "p1"), ("p2", "p2")):
        previous = read(config["reference_patches"][old_method]["metrics_path"])
        current = records["obj_cls_best_" + variant]
        delta = abs(previous["map50_pct"] - current["map50_pct"])
        assert delta < 1e-3 and current["conditional_asr_numerator"] == previous["conditional_asr_numerator"]
        replay[variant] = {"map50_absolute_difference_pp": delta, "asr_count_matches": True}

    VIS.mkdir(parents=True, exist_ok=True)
    (VIS / "patches").mkdir(exist_ok=True)
    table, visual = [], {}
    for item in config["patches"]:
        record = records[item["id"]]
        assert record["conditional_asr_denominator"] == 578
        row = {"id": item["id"], "objective": item.get("objective", "control"),
               "selection": item.get("selection", "control"), "variant": item.get("variant", item["id"]),
               "epoch": item.get("epoch"), "map50_pct": record["map50_pct"],
               "map50_drop_pp": clean["map50_pct"] - record["map50_pct"],
               "map50_95_pct": 100 * record["map50_95"],
               "asr_pct": 100 * record["conditional_asr"],
               "asr_numerator": record["conditional_asr_numerator"], "asr_denominator": 578,
               "precision_pct": 100 * record["precision"], "recall_pct": 100 * record["recall"],
               "checkpoint": item["path"], "checkpoint_sha256": record["patch_sha256"]}
        table.append(row)
        if item["id"] == "clean":
            continue
        patch = load_patch(item["path"]).permute(1, 2, 0).numpy().astype(np.float64)
        pixels = np.rint(patch * 255).clip(0, 255).astype(np.uint8)
        png = VIS / "patches" / (item["id"] + ".png")
        Image.fromarray(pixels).save(png)
        assert np.array_equal(np.asarray(Image.open(png)), pixels)
        visual[item["id"]] = {"png": str(png), "checkpoint_sha256": record["patch_sha256"],
                              "png_sha256": sha(png), "title": item["label"].replace("obj-cls", "OBJ-CLS").replace("obj ", "OBJ "),
                              "subtitle": f"AP50 {row['map50_pct']:.2f}% | ASR {row['asr_pct']:.2f}%"}

    for name, selected in (("comparison_all.csv", table),
                           ("comparison_best.csv", [r for r in table if r["selection"] in ("control", "best")])):
        with (RUN / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(selected)

    for selection in ("best", "last"):
        items = [visual[f"{objective}_{selection}_{variant}"] for objective in ("obj_cls", "obj")
                 for variant in ("raw", "p1", "p2")]
        render_grid(items, VIS / f"{selection}_comparison.png", f"YOLOv5s | OBJ-CLS vs OBJ-only | {selection}")

    official_dir = ROOT / "experiments/baselines/advpatch_audit_20261110"
    official = [
        {"png": str(official_dir / "official_class_transfer.png"), "title": "Official Fig. 4(b) | YOLOv2", "subtitle": "OBJ-CLS | published patch"},
        {"png": str(official_dir / "official_obj_patch.png"), "title": "Official Fig. 4(c) | YOLOv2", "subtitle": "OBJ-only | published patch"},
        {**visual["obj_cls_best_raw"], "title": "Current YOLOv5s | OBJ-CLS"},
        {**visual["obj_best_raw"], "title": "New YOLOv5s | OBJ-only"},
    ]
    render_grid(official, VIS / "official_vs_current.png", "Official examples and current trained patches", column_count=2)
    write(VIS / "manifest.json", {"selection": "training-loss-selected best and final", "patches": visual,
                                 "official_source": "https://arxiv.org/abs/1904.08653",
                                 "note": "Official examples use YOLOv2; current patches use YOLOv5s."})

    indexed = {r["id"]: r for r in table}
    retention = []
    for objective in ("obj_cls", "obj"):
        for selection in ("best", "last"):
            raw = indexed[f"{objective}_{selection}_raw"]
            for variant in ("p1", "p2"):
                recolored = indexed[f"{objective}_{selection}_{variant}"]
                retention.append({"objective": objective, "selection": selection, "variant": variant,
                                  "map50_change_vs_raw_pp": recolored["map50_pct"] - raw["map50_pct"],
                                  "ap_drop_retention_pct": 100 * recolored["map50_drop_pp"] / raw["map50_drop_pp"]})
    write(RUN / "recolor_retention.json", retention)
    audit = {"passed": True, "evaluations": len(records), "images": 288, "targets": 589,
             "asr_denominator": 578, "existing_obj_cls_replay": replay,
             "dataset_hashes_match_previous_audit": True, "evaluator_and_training_code_hashes_match_manifest": True}
    write(RUN / "evaluation_audit.json", audit)

    complete = read(RUN / "training_complete.json")
    lines = ["# YOLOv5s AdvPatch OBJ-only 与 OBJ-CLS 对照", "",
             "物理 GPU 7。OBJ-only 从灰色补丁训练 200 轮，共 15,400 次更新；没有测试集选模。",
             f"OBJ-only best 为第 {complete['summary']['best_training_loss_epoch']} 轮；OBJ-CLS 复用 s000，第 199 轮 best，并重新评估。",
             "仅检测项从 max(obj×person probability) 改为 max(obj)。NPS 0.01、max(2.5×TV,0.1)、EOT、图像预处理与贴图规则保持一致。",
             "训练：614 图，seed 1788272031，batch 8，Adam AMSGrad lr 0.03，ReduceLROnPlateau patience 50，640 输入，300×300 补丁，人体框面积 25%。",
             "评估：人工 GT，288 图/589 人体；conf 0.001、NMS IoU 0.6、max_det 300、batch 8、无 EOT。ASR 为干净可检出的 578 个 GT 中补丁导致漏检的比例，匹配阈值 conf/IoU 均为 0.5。", "",
             "## 按训练损失选择 best", "",
             "| 方法 | mAP50 (%) ↓ | 相对 Clean 下降 (百分点) ↑ | mAP50–95 (%) ↓ | ASR (%) ↑ |",
             "|---|---:|---:|---:|---:|"]
    for row in table:
        if row["selection"] in ("control", "best"):
            lines.append(f"| {row['id']} | {row['map50_pct']:.2f} | {row['map50_drop_pp']:.2f} | {row['map50_95_pct']:.2f} | {row['asr_pct']:.2f} |")
    lines += ["", "## 最后第 200 轮", "", "| 方法 | mAP50 (%) ↓ | 下降 (百分点) ↑ | mAP50–95 (%) ↓ | ASR (%) ↑ |", "|---|---:|---:|---:|---:|"]
    for row in table:
        if row["selection"] == "last":
            lines.append(f"| {row['id']} | {row['map50_pct']:.2f} | {row['map50_drop_pp']:.2f} | {row['map50_95_pct']:.2f} | {row['asr_pct']:.2f} |")
    lines += ["", "## 换色保留性", "", "保留率 = 换色后的 AP50 下降 / 原补丁 AP50 下降 × 100%。两套固定色板 Bc1/Bc2 均使用原有 RGB 线性换色公式，不归一化通道、不训练新补丁。", "",
              "| 目标 | 选择 | 换色 | 相对原补丁 AP50 变化 (百分点) | AP50 下降保留率 (%) |", "|---|---|---|---:|---:|"]
    for row in retention:
        lines.append(f"| {row['objective']} | {row['selection']} | {row['variant']} | {row['map50_change_vs_raw_pp']:+.2f} | {row['ap_drop_retention_pct']:.2f} |")
    delta = indexed['obj_best_raw']['map50_pct'] - indexed['obj_cls_best_raw']['map50_pct']
    gray_delta = indexed['obj_best_raw']['map50_pct'] - indexed['gray']['map50_pct']
    gray_asr_delta = indexed['obj_best_raw']['asr_pct'] - indexed['gray']['asr_pct']
    lines += ["", f"OBJ-only best 的 AP50 相对 OBJ-CLS best 变化为 {delta:+.2f} 个百分点；负值表示本次攻击更强。此结论限定为当前 YOLOv5s、单个固定种子的白盒对照。",
              f"OBJ-only best 的 AP50 相对 Gray 为 {gray_delta:+.2f} 个百分点，ASR 相对 Gray 为 {gray_asr_delta:+.2f} 个百分点。AP 衡量整条精确率-召回率曲线，ASR 使用固定置信度/IoU 阈值，因此两者可能给出不同排序。",
              "上述换色保留率以 Clean 为基线，包含补丁遮挡带来的性能下降；高保留率本身不能证明强于灰色遮挡。应结合 Gray 和原补丁的绝对 AP/ASR 一起解释。各下降值使用未舍入数值计算，显示值相加可能有 0.01 的舍入误差。",
              "", "## 补丁图", "", "![best 对照](../../analysis/pac_patch_visualizations_20260910/objonly_yolov5s_20261110/best_comparison.png)",
              "", "![官方图与当前补丁](../../analysis/pac_patch_visualizations_20260910/objonly_yolov5s_20261110/official_vs_current.png)",
              "", "官方上排为原论文 Figure 4(b)/(c)，使用 YOLOv2；本地下排使用 YOLOv5s。图片只用于外观比较，不能将跨模型和跨协议的差异单独归因于损失函数。",
              "本次保留现有 PAC 的 GT 贴图、面积和旋转实现，是检测损失的受控对照，不是作者 YOLOv2 全协议的逐项重放。",
              "", "## 文件与验证", "", "- [全部指标 CSV](comparison_all.csv)；[best 指标 CSV](comparison_best.csv)。",
              "- [实验协议](protocol.json)、[训练完成记录](training_complete.json)、[评估审查](evaluation_audit.json)、[换色数值审查](generation_audit.json)。",
              "- [新旧训练配置核对](matched_protocol_audit.json)。6 项现有 unittest 通过，1 轮完整 GPU 试跑通过，再执行 200 轮正式训练。",
              "- [代码修改](code_changes.diff)、[代码版本](source_provenance.json)。现有仓库 remote 为本地文件路径，没有可提交 PR 的托管远端；修改保存在独立工作树和提交中。",
              "- 指标全部在相同评估器下重新计算，旧 OBJ-CLS 的原补丁/P1/P2 复测 AP50 与现有记录差值均小于 0.001 个百分点，ASR 计数完全一致。",
              "- CSV 的 precision/recall 为评估器选择的 F1 工作点输出；ASR 使用固定阈值，二者分开解释。",
              "- 图片从真实 checkpoint 渲染，RGB 原值仅四舍五入到 PNG uint8；没有调色、对比度增强或以论文截图替代训练结果。"]
    (RUN / "report.md").write_text("\n".join(lines) + "\n")
    write(RUN / "final_summary.json", {"passed": True, "training": complete["summary"], "evaluation": audit,
                                       "metrics": table, "retention": retention, "visualization_dir": str(VIS)})
    print(json.dumps({"passed": True, "report": str(RUN / "report.md"), "metrics": table}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
