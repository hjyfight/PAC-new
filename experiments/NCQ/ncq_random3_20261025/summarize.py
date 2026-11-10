"""Audit the completed evaluations and export tables, figures, and a report."""

import csv
import json

from run_experiments import CODE_FILES, MODELS, PREVIOUS, PROJECT, REFERENCE, RUN, SEEDS, check_code, read, sha, write

import numpy as np


def export_csv(name, rows):
    with (RUN / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = read(RUN / "protocol.json")
    check_code(protocol)
    generation = read(RUN / "generation_audit.json")
    assert generation["passed"]
    rows, replays, values = [], [], {}
    asr = {}
    for model in MODELS:
        directory = RUN / "sources" / model
        assert read(directory / "complete.json")["passed"]
        config = read(directory / "manifest.json")
        values[model], asr[model] = {}, {}
        for item in config["patches"]:
            path = directory / "eval" / model / item["id"] / "metrics.json"
            record = read(path)
            assert record["images"] == 288 and record["targets"] == 589
            assert record["detector"] == model
            assert record["weights_sha256"] == config["detectors"][model]["sha256"]
            assert record["eot"] is False and record["rect"] is False
            for key in ("patch_size", "patch_area_fraction", "imgsz", "preprocess", "batch_size",
                        "conf_thres", "nms_iou", "max_det", "asr_conf", "asr_iou"):
                assert record[key] == config[key], (model, key)
            if item["path"] not in (None, "gray"):
                assert record["patch_sha256"] == sha(item["path"]) == item["sha256"]
            assert np.isfinite(record["map50_pct"]) and 0 <= record["map50_pct"] <= 100
            values[model][item["id"]] = record["map50_pct"]
            asr[model][item["id"]] = 100 * record["conditional_asr"]
            rows.append({"model": model, "variant": item["id"], "map50_pct": record["map50_pct"],
                         "conditional_asr_pct": asr[model][item["id"]],
                         "asr_numerator": record["conditional_asr_numerator"],
                         "asr_denominator": record["conditional_asr_denominator"], "metrics_path": str(path)})
            if item["id"] in ("clean", "advpatch", "p1", "p2"):
                previous_root = REFERENCE if model == "yolov5s" else PREVIOUS / "advpatch" / model
                previous = previous_root / "eval" / model / item["id"] / "metrics.json"
                previous_record = read(previous)
                difference = record["map50_pct"] - previous_record["map50_pct"]
                assert abs(difference) <= 1e-3, (model, item["id"], difference)
                assert record["conditional_asr_numerator"] == previous_record["conditional_asr_numerator"]
                assert record["conditional_asr_denominator"] == previous_record["conditional_asr_denominator"]
                replays.append({"model": model, "variant": item["id"], "difference": difference,
                                "previous_metrics": str(previous), "previous_sha256": sha(previous)})
    assert len(rows) == protocol["expected_evaluations"] == 132
    for row in read(RUN / "dataset_audit.json"):
        assert sha(row["image"]) == row["image_sha256"]
        assert sha(row["label"]) == row["label_sha256"]
    export_csv("all_metrics.csv", rows)

    # Across-model means are descriptive averages. Across-seed SD is calculated
    # after averaging all six models for each draw, not from 30 pooled cells.
    values["mean6"] = {key: float(np.mean([values[model][key] for model in MODELS]))
                       for key in values[MODELS[0]]}
    asr["mean6"] = {key: float(np.mean([asr[model][key] for model in MODELS]))
                    for key in asr[MODELS[0]]}
    ncq_rows, factorial_rows, seed_rows = [], [], []
    for model in (*MODELS, "mean6"):
        val = values[model]
        ncq_rows.append({"model": model, **{key: val[key] for key in
                        ("clean", "gray", "advpatch", "p1", "ncq1", "p2", "ncq2")},
                        "ncq1_minus_p1": val["ncq1"] - val["p1"],
                        "ncq2_minus_p2": val["ncq2"] - val["p2"]})
        random = np.array([val[f"random_s{seed}"] for seed in SEEDS])
        shuffled_fixed = np.array([val[f"shuffle_bc1_s{seed}"] for seed in SEEDS])
        shuffled_random = np.array([val[f"shuffle_random_s{seed}"] for seed in SEEDS])
        structure_fixed = shuffled_fixed - val["p1"]
        structure_random = shuffled_random - random
        factorial_rows.append({
            "model": model, "retained_bc1": val["p1"],
            "retained_random_mean": float(random.mean()), "retained_random_std": float(random.std(ddof=1)),
            "retained_random_min": float(random.min()), "retained_random_max": float(random.max()),
            "shuffled_bc1_mean": float(shuffled_fixed.mean()), "shuffled_bc1_std": float(shuffled_fixed.std(ddof=1)),
            "shuffled_random_mean": float(shuffled_random.mean()), "shuffled_random_std": float(shuffled_random.std(ddof=1)),
            "color_effect_retained": float(random.mean() - val["p1"]),
            "structure_effect_bc1": float(structure_fixed.mean()),
            "structure_effect_random": float(structure_random.mean()),
            "interaction": float((structure_random - structure_fixed).mean()),
        })
        for seed in SEEDS:
            seed_rows.append({"model": model, "palette_seed": seed,
                              "retained_bc1": val["p1"], "retained_random": val[f"random_s{seed}"],
                              "shuffled_bc1": val[f"shuffle_bc1_s{seed}"],
                              "shuffled_random": val[f"shuffle_random_s{seed}"]})
    export_csv("ncq_comparison.csv", ncq_rows)
    export_csv("f1_factorial.csv", factorial_rows)
    export_csv("f1_per_seed.csv", seed_rows)
    export_csv("asr_comparison.csv", [{"model": model, **data} for model, data in asr.items()])
    clipping = []
    for seed in SEEDS:
        selected = [row["clipped_fraction"] for row in generation["patches"] if row["id"] == f"random_s{seed}"]
        clipping.append({"seed": seed, "clipped_pct_min": 100 * min(selected),
                         "clipped_pct_max": 100 * max(selected)})
    summary = {"passed": True, "evaluations": len(rows), "original_baseline_replays": replays,
               "max_replay_difference": max(abs(row["difference"]) for row in replays),
               "ncq": ncq_rows, "factorial": factorial_rows, "values": values, "asr_pct": asr,
               "random_palette_clipping": clipping,
               "runtime_notes": "Initial AF_UNIX path failure was resolved by shortening TMPDIR; successful workers exited 0. NFS temporary-file cleanup warnings occurred after all metrics were saved.",
               "summary_script_sha256": sha(__file__), "execution_script_sha256": sha(RUN / "run_experiments.py"),
               "source_code_sha256": {name: sha(PROJECT / name) for name in CODE_FILES}}
    write(RUN / "summary.json", summary)
    render_figures(values, ncq_rows, factorial_rows)
    render_report(summary)
    print(json.dumps({"passed": True, "evaluations": len(rows), "ncq_mean": ncq_rows[-1],
                      "f1_mean": factorial_rows[-1], "report": str(RUN / "report.md")}, ensure_ascii=False))


def render_figures(values, ncq_rows, factorial_rows):
    import matplotlib.pyplot as plt
    from utils.patch import load_patch

    names = (*MODELS, "Mean (6)")
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(7)
    for index, (key, label, color) in enumerate((
        ("advpatch", "AdvPatch", "#4c78a8"), ("p1", "PAC Bc1", "#54a24b"),
        ("ncq1", "NCQ Bc1", "#e45756"), ("p2", "PAC Bc2", "#72b7b2"),
        ("ncq2", "NCQ Bc2", "#f2cf5b"),
    )):
        ax.bar(x + (index - 2) * 0.15, [row[key] for row in ncq_rows], 0.15, label=label, color=color)
    ax.set(xticks=x, xticklabels=names, ylabel="mAP50 (%) - lower means stronger attack", ylim=(0, 102),
           title="NCQ vs PAC: matched donor, palette, placement and full INRIA test set")
    ax.legend(ncol=5, fontsize=9, loc="upper center")
    fig.savefig(RUN / "ncq_comparison.png", dpi=180)
    fig.savefig(RUN / "ncq_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for index, (key, err, label, color) in enumerate((
        ("retained_bc1", None, "Retained / Bc1", "#54a24b"),
        ("retained_random_mean", "retained_random_std", "Retained / Random", "#4c78a8"),
        ("shuffled_bc1_mean", "shuffled_bc1_std", "Shuffled / Bc1", "#f2cf5b"),
        ("shuffled_random_mean", "shuffled_random_std", "Shuffled / Random", "#e45756"),
    )):
        ax.bar(x + (index - 1.5) * 0.19, [row[key] for row in factorial_rows], 0.19,
               yerr=[row[err] for row in factorial_rows] if err else None, capsize=2, label=label, color=color)
    ax.set(xticks=x, xticklabels=names, ylabel="mAP50 (%) - lower means stronger attack", ylim=(0, 108),
           title="F1: retained vs shuffled spatial weights, fixed vs five random palettes")
    ax.legend(ncol=4, fontsize=9, loc="upper center")
    fig.savefig(RUN / "f1_factorial.png", dpi=180)
    fig.savefig(RUN / "f1_factorial.pdf")
    plt.close(fig)

    columns = ("advpatch", "p1", "ncq1", "p2", "ncq2", *[f"random_s{s}" for s in SEEDS])
    fig, axes = plt.subplots(6, len(columns), figsize=(18, 11), constrained_layout=True)
    for row, model in enumerate(MODELS):
        items = {item["id"]: item for item in read(RUN / "sources" / model / "manifest.json")["patches"]}
        for col, key in enumerate(columns):
            patch = load_patch(items[key]["path"]).permute(1, 2, 0).numpy()
            axes[row, col].imshow(patch)
            axes[row, col].set(xticks=[], yticks=[])
            axes[row, col].set_xlabel(f'{values[model][key]:.2f}', fontsize=9)
            if row == 0:
                axes[row, col].set_title(key, fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(model, fontsize=10)
    fig.suptitle("Actual patch tensors; numbers are white-box mAP50 (%)", fontsize=14)
    fig.savefig(RUN / "patch_gallery.png", dpi=140)
    plt.close(fig)


def render_report(summary):
    n, f = summary["ncq"][-1], summary["factorial"][-1]
    text = [
        "# NCQ 与 F1 随机三色实验结果", "", "日期：2026-10-25。132/132 项评估完成，以下 mAP50 均为百分数，越低表示攻击越强。", "",
        "## 实验协议", "",
        "- INRIA 完整测试集：288 张图、589 个人体框；输入 640×640；补丁张量 300×300；中心覆盖面积为人体框的 25%。",
        "- 六模型白盒：每个模型使用自己已有的 AdvPatch donor，最低训练损失选择 checkpoint；无新增攻击训练，无测试集挑色/挑补丁。",
        "- 原始 RGB 通道作为未归一化空间权重，经现有 recolor() 做线性换色并截断到 [0,1]；没有逆映射优化或额外 softmax。",
        "- NCQ：对原始 donor 每像素计算到相同三个基色的 RGB 欧氏距离，取最近基色。Bc1/Bc2 的顺序和值与现有 PAC-P1/P2 相同。",
        "- 五个随机色板：NumPy default_rng 的 seed=0…4，RGB 各通道从 0…255 独立等概率采样，三色有序，不按攻击结果筛选。完整 RGB 见 palettes.json。",
        "- 结构控制：seed=10000…10004 的空间置换，将同一像素的三个 RGB 权重整体打乱；同一置换用于 Bc1 和对应随机色板。已核验贴图前颜色直方图完全一致。",
        "- AP 使用现有 eval_transfer.py；置信度 0.001、NMS IoU 0.6、max_det=300，测试无 EOT，batch=8。ASR 为 clean 可检出的 GT 在贴图后漏检的比例（conf/IoU 均为 0.5）。",
        f'- 重放 clean/AdvPatch/PAC-P1/PAC-P2 共 24 项，最大 mAP50 差为 {summary["max_replay_difference"]:.8f}，ASR 分子分母完全一致。',
        "", "## NCQ 对比", "",
        "| 模型 | AdvPatch | PAC-P1 | NCQ-Bc1 | PAC-P2 | NCQ-Bc2 | NCQ1−PAC1 | NCQ2−PAC2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["ncq"]:
        text.append("| " + row["model"] + " | " + " | ".join(f'{row[k]:.2f}' for k in
                    ("advpatch", "p1", "ncq1", "p2", "ncq2", "ncq1_minus_p1", "ncq2_minus_p2")) + " |")
    text += ["", f'六模型均值：PAC-P1={n["p1"]:.2f}，NCQ-Bc1={n["ncq1"]:.2f}，差 {n["ncq1_minus_p1"]:.2f} 点；'
             f'PAC-P2={n["p2"]:.2f}，NCQ-Bc2={n["ncq2"]:.2f}，差 {n["ncq2_minus_p2"]:.2f} 点。', "",
             "解释：此实验直接检验“现有 PAC 换色能否被最近基色硬量化替代”。PAC 使用连续混色，输出不限制为恰好三种 RGB；NCQ 只输出三种颜色，因此也改变了亮度、局部对比度及空间细节。结果不能证明某个学习到的 m 或额外优化优于 NCQ，因为当前 PAC-P 实现没有该学习步骤。", "",
             "## F1：固定结构、随机三色及配对结构控制", "",
             "| 模型 | 保留结构 / Bc1 | 保留结构 / 随机三色 | 打乱结构 / Bc1 | 打乱结构 / 随机三色 |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["factorial"]:
        text.append(f'| {row["model"]} | {row["retained_bc1"]:.2f} | '
                    f'{row["retained_random_mean"]:.2f} ± {row["retained_random_std"]:.2f} | '
                    f'{row["shuffled_bc1_mean"]:.2f} ± {row["shuffled_bc1_std"]:.2f} | '
                    f'{row["shuffled_random_mean"]:.2f} ± {row["shuffled_random_std"]:.2f} |')
    text += ["", "± 为五次色板/置换抽样的样本标准差，不是五个训练种子的标准差。六模型均值行先对每个 seed 的六个模型求均值，再对五个均值计算标准差；不能把六个模型当独立重复。", "",
             f'六模型均值中，随机三色相对 Bc1 的变化为 {f["color_effect_retained"]:+.2f} 点；'
             f'保持 Bc1 而打乱结构的变化为 {f["structure_effect_bc1"]:+.2f} 点；'
             f'随机三色下打乱结构的配对变化为 {f["structure_effect_random"]:+.2f} 点。', "",
             "这组数据支持：在本协议中，空间打乱造成的平均退化大于换成这五组随机色板；但随机换色也有明显影响，不能声称颜色无关或任意换色均保持原攻击性能。", "",
             "任意 RGB 换色同时影响颜色、亮度、对比度与截断；像素置换是强烈的空间/频率破坏，因此不把两种干预的差值推广为普遍的“pattern 比 color 重要”定律。", "",
             "### 随机色板的截断审计", "",
             "Bc1/Bc2 的每个输出通道基色之和均为 255，换色不产生越界；任意随机三色没有这一限制。下表记录固定结构随机换色的 RGB 分量截断比例范围（六模型），用于披露与固定色板的处理差异。", "",
             "| 色板 seed | 六模型平均 mAP50 | 最小截断比例 | 最大截断比例 |", "|---|---:|---:|---:|"]
    for row in summary["random_palette_clipping"]:
        mean_map = summary["values"]["mean6"]["random_s" + str(row["seed"])]
        text.append(f'| {row["seed"]} | {mean_map:.2f} | '
                    f'{row["clipped_pct_min"]:.2f}% | {row["clipped_pct_max"]:.2f}% |')
    text += ["",
             "## 范围与复现", "",
             "- Bc1/Bc2 是固定合成色板，不是从真实环境提取的颜色；本次评价攻击能力，未评价环境隐蔽性。",
             "- 此次使用可追溯的共同协议重训 donor，六模型原始均值为 19.33；不是对旧稿 19.58/22.92/82.38 的复现，不能混用数字。",
             "- 每模型一个已有训练 seed；五个随机色板用于测量色板敏感性，未补训练方差、黑盒或物理实验。",
             "- generation_audit.json 记录每个 PAC 版本超出 [0,1] 的截断比例；随机色板保留现有截断规则，不重新归一化权重或色板。",
             "- 首次启动的 AF_UNIX 路径错误已通过缩短临时目录修复；成功运行的进程退出码为 0。部分进程保存全部指标后，NFS 临时文件清理报资源忙；不影响 132 份完整结果及 24 份基准精确重放。",
             "- 配置及出处见 protocol.json、sources/*/manifest.json、dataset_audit.json、generation_audit.json；原始指标和逐 GT 检出状态见 sources/*/eval/。",
             "", "```bash", "cd /datacc05/huotianlin/code-hu/PAC-yolo",
             "/datacc05/huotianlin/miniconda3/envs/pac/bin/python -B experiments/NCQ/ncq_random3_20261025/run_experiments.py run --gpus 1,6,7",
             "/datacc05/huotianlin/miniconda3/envs/pac/bin/python -B experiments/NCQ/ncq_random3_20261025/summarize.py", "```", "",
             "CSV：ncq_comparison.csv、f1_factorial.csv、f1_per_seed.csv、asr_comparison.csv、all_metrics.csv。图：ncq_comparison.png/pdf、f1_factorial.png/pdf、patch_gallery.png。", ""]
    (RUN / "report.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
