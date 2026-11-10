"""Audit and combine sum255 palette seeds 0..9 without altering prior runs."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
FIRST = ROOT / "experiments/NCQ/ncq_random3_sum255_20261025"
ORIGINAL = ROOT / "experiments/NCQ/ncq_random3_20261025"
MODELS = ("yolov2", "yolov3", "yolov4", "yolov5s", "yolov5m", "fasterrcnn")
SEEDS = tuple(range(10))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(name, rows):
    with (RUN / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    first, new, original = [read(p / "summary.json") for p in (FIRST, RUN, ORIGINAL)]
    protocol = read(RUN / "protocol.json")
    assert first["passed"] and first["evaluations"] == 66
    assert new["passed"] and new["evaluations"] == 96
    assert sha(FIRST / "summary.json") == protocol["first_batch_summary_sha256"]
    assert sha(FIRST / "protocol.json") == protocol["first_batch_protocol_sha256"]
    assert sha(FIRST / "palettes.json") == protocol["first_batch_palettes_sha256"]
    assert new["parent_summary_sha256"] == sha(ORIGINAL / "summary.json")
    assert sha(ORIGINAL / "protocol.json") == protocol["parent_protocol_sha256"]
    first_protocol = read(FIRST / "protocol.json")
    for key in ("source_code_sha256", "detectors", "source_items", "data_yaml",
                "patch_size", "patch_area_fraction", "imgsz", "preprocess", "batch_size",
                "conf_thres", "nms_iou", "max_det", "asr_conf", "asr_iou"):
        assert protocol[key] == first_protocol[key], key
    palettes = {**first["palettes"], **new["palettes"]}
    assert len(palettes) == 10 and set(palettes) == {f"sum255_s{s}" for s in SEEDS}
    assert len({tuple(v for color in p for v in color) for p in palettes.values()}) == 10
    for palette in palettes.values():
        assert [sum(color[c] for color in palette) for c in range(3)] == [255, 255, 255]
    for directory, count in ((FIRST, 60), (RUN, 90)):
        audit = read(directory / "generation_audit.json")
        assert audit["passed"] and len(audit["patches"]) == count
        assert all(r["clamp_changed_components"] == 0 and r["clipped_fraction_float64"] == 0
                   and r["clipped_fraction_float32"] == 0 for r in audit["patches"])
        assert all(r["paired_permutation_exact"] for r in audit["patches"] if r["id"].startswith("shuffle"))
    values, asrs = {}, {}
    checked = 0
    for model in MODELS:
        values[model], asrs[model] = {}, {}
        for directory, summary, expected in ((FIRST, first, 11), (RUN, new, 16)):
            manifest = read(directory / "sources" / model / "manifest.json")
            assert read(directory / "sources" / model / "complete.json") == {"passed": True, "evaluations": expected}
            for item in manifest["patches"]:
                path = directory / "sources" / model / "eval" / model / item["id"] / "metrics.json"
                record = read(path)
                assert record["images"] == 288 and record["targets"] == 589
                assert record["detector"] == model and record["patch_id"] == item["id"]
                assert record["map50_pct"] == summary["values"][model][item["id"]]
                assert record["weights_sha256"] == protocol["detectors"][model]["sha256"]
                assert record["eot"] is False and record["rect"] is False
                for key in ("patch_size", "patch_area_fraction", "imgsz", "preprocess", "batch_size",
                            "conf_thres", "nms_iou", "max_det", "asr_conf", "asr_iou"):
                    assert record[key] == protocol[key], (model, key)
                asr = 100 * record["conditional_asr_numerator"] / record["conditional_asr_denominator"]
                assert math.isclose(asr, summary["asr_pct"][model][item["id"]], abs_tol=1e-12)
                if item["path"]:
                    assert sha(item["path"]) == item["sha256"] == record["patch_sha256"]
                values[model][item["id"]] = record["map50_pct"]
                asrs[model][item["id"]] = asr
                checked += 1
        values[model]["p1"] = original["values"][model]["p1"]
        asrs[model]["p1"] = original["asr_pct"][model]["p1"]
        for seed in range(5):
            key = f"shuffle_bc1_s{seed}"
            values[model][key] = original["values"][model][key]
            asrs[model][key] = original["asr_pct"][model][key]
    for data in (values, asrs):
        data["mean6"] = {key: statistics.mean(data[m][key] for m in MODELS) for key in data[MODELS[0]]}

    seed_rows, factorial, batch_rows = [], [], []
    for model in (*MODELS, "mean6"):
        for seed in SEEDS:
            v, a = values[model], asrs[model]
            fixed = v["p1"]
            retained = v[f"retained_sum255_s{seed}"]
            shuffled = v[f"shuffled_sum255_s{seed}"]
            shuffled_fixed = v[f"shuffle_bc1_s{seed}"]
            seed_rows.append({
                "model": model, "palette_seed": seed, "permutation_seed": 10000 + seed,
                "fixed_bc1_map50": fixed, "retained_sum255_map50": retained,
                "shuffled_bc1_map50": shuffled_fixed, "shuffled_sum255_map50": shuffled,
                "color_effect_retained": retained - fixed, "structure_effect_bc1": shuffled_fixed - fixed,
                "structure_effect_random": shuffled - retained, "color_effect_shuffled": shuffled - shuffled_fixed,
                "retained_sum255_asr_pct": a[f"retained_sum255_s{seed}"],
                "shuffled_sum255_asr_pct": a[f"shuffled_sum255_s{seed}"],
            })
        for label, seeds in (("first5", range(5)), ("new5", range(5, 10)), ("combined10", range(10))):
            selected = [r for r in seed_rows if r["model"] == model and r["palette_seed"] in seeds]
            row = {"model": model, "batch": label, "draws": len(selected),
                   "fixed_bc1_map50": values[model]["p1"], "fixed_bc1_asr_pct": asrs[model]["p1"]}
            for key in ("retained_sum255_map50", "shuffled_bc1_map50", "shuffled_sum255_map50",
                        "color_effect_retained", "structure_effect_bc1", "structure_effect_random",
                        "color_effect_shuffled", "retained_sum255_asr_pct", "shuffled_sum255_asr_pct"):
                draws = [r[key] for r in selected]
                row[key + "_mean"] = statistics.mean(draws)
                row[key + "_sd"] = statistics.stdev(draws)
            batch_rows.append(row)
            if label == "combined10":
                factorial.append(row)
    export("combined_per_seed.csv", seed_rows)
    export("combined_factorial.csv", factorial)
    export("batch_comparison.csv", batch_rows)
    summary = {"passed": True, "palette_seeds": list(SEEDS), "palette_draws": 10,
               "raw_metrics_checked": checked, "new_evaluations": 96, "new_attack_evaluations": 90,
               "new_random_attack_evaluations": 60, "new_bc1_shuffle_evaluations": 30,
               "new_clean_replays": 6, "clipping_fraction": 0.0, "palettes": palettes,
               "factorial": factorial, "batches": batch_rows, "values": values, "asr_pct": asrs,
               "source_summaries_sha256": {str(p / "summary.json"): sha(p / "summary.json") for p in (FIRST, RUN, ORIGINAL)},
               "summary_script_sha256": sha(__file__),
               "variance": "Sample SD over palette/permutation draws, not independent training seeds; mean6 first averages models within each seed."}
    write("combined_summary.json", summary)
    lines = ["# 新增五组随机色板及十组合并结果", "",
             "新增色板 seed 5–9，列和均为 [255,255,255]；采用与首批相同的均匀整数分拆采样，未按测试结果筛选。",
             "新增 60 项随机色板攻击评估、30 项配对 Bc1 置换对照、6 项 clean 重放，共 96 项；全部完成。",
             "复用相同六个 AdvPatch donor 与 INRIA 288 图/589 框白盒协议，不新增训练；新 90 张补丁的 clamp 均不改变任何分量。",
             "完整 RGB 像素向量置换 seed 10005–10009；对应固定 Bc1 与随机色板使用同一置换。",
             "合并前核验两批 162 份原始指标及 150 张补丁哈希；原结果保持原路径。", "",
             "## 五组新色板", "", "| seed | 基色 1 | 基色 2 | 基色 3 |", "|---|---|---|---|"]
    for seed in range(5, 10):
        p = palettes[f"sum255_s{seed}"]
        lines.append(f"| {seed} | {p[0]} | {p[1]} | {p[2]} |")
    lines += ["", "## 六模型平均：批次比较", "",
              "mAP50 为百分数，越低表示攻击越强；差值为百分点。", "",
              "| 批次 | 保留/随机 | 打乱/随机 | 相对 Bc1 换色代价 | 配对结构代价 |",
              "|---|---:|---:|---:|---:|"]
    for r in batch_rows:
        if r["model"] == "mean6":
            lines.append(f"| {r['batch']} | {r['retained_sum255_map50_mean']:.2f} ± {r['retained_sum255_map50_sd']:.2f} | {r['shuffled_sum255_map50_mean']:.2f} ± {r['shuffled_sum255_map50_sd']:.2f} | {r['color_effect_retained_mean']:+.2f} | {r['structure_effect_random_mean']:+.2f} ± {r['structure_effect_random_sd']:.2f} |")
    lines += ["", "## 十组完整 2×2 结果", "",
              "| 模型 | 保留/Bc1 | 保留/随机 | 打乱/Bc1 | 打乱/随机 |",
              "|---|---:|---:|---:|---:|"]
    for r in factorial:
        lines.append(f"| {r['model']} | {r['fixed_bc1_map50']:.2f} | {r['retained_sum255_map50_mean']:.2f} ± {r['retained_sum255_map50_sd']:.2f} | {r['shuffled_bc1_map50_mean']:.2f} ± {r['shuffled_bc1_map50_sd']:.2f} | {r['shuffled_sum255_map50_mean']:.2f} ± {r['shuffled_sum255_map50_sd']:.2f} |")
    m = factorial[-1]
    lines += ["", "## 解释与统计口径", "",
              f"十组均值的保留结构换色代价为 {m['color_effect_retained_mean']:+.2f} 点；固定 Bc1 的结构打乱代价为 {m['structure_effect_bc1_mean']:+.2f} 点；随机色板下的配对结构打乱代价为 {m['structure_effect_random_mean']:+.2f} 点。",
              f"ASR 六模型均值：Bc1 {m['fixed_bc1_asr_pct']:.2f}%，保留/随机 {m['retained_sum255_asr_pct_mean']:.2f} ± {m['retained_sum255_asr_pct_sd']:.2f}%，打乱/随机 {m['shuffled_sum255_asr_pct_mean']:.2f} ± {m['shuffled_sum255_asr_pct_sd']:.2f}%。",
              "± 为对应 5 或 10 次色板/置换干预的样本标准差。mean6 先按 seed 对六模型等权平均，再跨 seed 计算标准差；ASR 也是各模型条件 ASR 的等权平均。",
              "新增五组不是独立训练种子；不能据此报告训练方差、统计显著性、黑盒或物理结果。",
              "列和约束避免换色截断，但换色仍可改变颜色、亮度和对比度；完整像素置换是强结构干预。",
              "仅能比较所测试色板与完整像素置换的影响，不能推广为颜色无关或任意换色都保持攻击性能。", "",
              "文件：combined_summary.json、combined_factorial.csv、combined_per_seed.csv、batch_comparison.csv；新增原始指标见 sources/*/eval/。", ""]
    (RUN / "combined_report.md").write_text("\n".join(lines))
    os.environ["MPLCONFIGDIR"] = str(RUN / "runtime/mplconfig")
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    labels = ["Retained / Bc1", "Retained / random", "Shuffled / Bc1", "Shuffled / random"]
    means = [m["fixed_bc1_map50"], m["retained_sum255_map50_mean"],
             m["shuffled_bc1_map50_mean"], m["shuffled_sum255_map50_mean"]]
    errors = [0, m["retained_sum255_map50_sd"], m["shuffled_bc1_map50_sd"], m["shuffled_sum255_map50_sd"]]
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.bar(labels, means, yerr=errors, capsize=5, color=["#54a24b", "#4c78a8", "#f2cf5b", "#e45756"])
    ax.set(ylabel="mAP50 (%) - lower means stronger attack", ylim=(0, 105),
           title="Six-model mean: ten zero-clipping palette/permutation draws")
    fig.savefig(RUN / "combined_factorial.png", dpi=180)
    fig.savefig(RUN / "combined_factorial.pdf")
    plt.close(fig)
    print(json.dumps({"passed": True, "raw_metrics_checked": checked,
                      "new5_mean6": next(r for r in batch_rows if r["model"] == "mean6" and r["batch"] == "new5"),
                      "combined10_mean6": m}, ensure_ascii=False))


if __name__ == "__main__":
    main()
