"""Five-image PSNR/SSIM pilot using frozen PAC donor artifacts and renderer."""
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / "experiments/baselines/tsea_nap_20260912"
PROJECT = ROOT / "experiments/baselines/yolov5-tsea-nap-20260912"
SEED = 20261018
SOURCE = "yolov5s"

# Redirect caches/configuration before importing image and dataset libraries.
runtime = json.loads((OLD / "runtime_environment.json").read_text())
for key, value in runtime.items():
    if isinstance(value, str):
        runtime[key] = value.replace(str(OLD), str(OUT / "runtime"))
runtime["TMPDIR"] = str(OUT / "runtime/tmp")
os.environ.update(runtime)
for key in ("TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR", "YOLOV5_CONFIG_DIR"):
    Path(runtime[key]).mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))

import numpy as np
import PIL
from PIL import Image
import scipy
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.patch import apply_patch, load_patch, recolor, BASE_COLORS

METHODS = {
    "advpatch": ("AdvPatch", "Original"),
    "p1": ("AdvPatch", "Bc1"),
    "p2": ("AdvPatch", "Bc2"),
    "tsea": ("T-SEA", "Original"),
    "tsea_bc1": ("T-SEA", "Bc1"),
    "tsea_bc2": ("T-SEA", "Bc2"),
    "advreal": ("AdvReal-2D", "Original"),
    "advreal_bc1": ("AdvReal-2D", "Bc1"),
    "advreal_bc2": ("AdvReal-2D", "Bc2"),
}
SSIM_ARGS = dict(data_range=1.0, channel_axis=-1, win_size=11,
                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                 K1=0.01, K2=0.03)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(reference, candidate):
    reference, candidate = reference.astype(np.float64), candidate.astype(np.float64)
    assert reference.shape == candidate.shape and min(reference.shape[:2]) >= 11
    assert np.isfinite(reference).all() and np.isfinite(candidate).all()
    mse = float(np.mean((reference - candidate) ** 2))
    assert mse > 0, "Identical images must be handled as a separate infinite-PSNR control"
    psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    ssim = float(structural_similarity(reference, candidate, **SSIM_ARGS))
    assert np.isfinite(psnr) and np.isfinite(ssim)
    assert abs(psnr - (-10 * math.log10(mse))) < 1e-10
    return {"psnr_db": psnr, "ssim": ssim, "mse": mse}


def crop(array, box):
    x0, y0, x1, y1 = box
    return array[y0:y1, x0:x1]


def save_preview(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array.clip(0, 1) * 255).astype(np.uint8)).save(path)


def tensor_hwc(tensor):
    return tensor[0].cpu().permute(1, 2, 0).contiguous().numpy()


@torch.inference_mode()
def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit("Completed output already exists; preserve it and use a separate run directory.")
    torch.set_num_threads(2)
    config = read(OLD / "manifest.json")
    for name in ("utils/dataloaders.py", "utils/patch.py"):
        assert sha(PROJECT / name) == config["source_code_sha256"][name], name
    audit = read(OLD / "dataset_audit.json")
    population = audit["splits"]["val"]
    assert len(population) == 288 and audit["passed"]
    indices = np.random.default_rng(SEED).choice(len(population), size=5, replace=False)
    samples = [{"id": f"s{i+1:02d}", "audit_index": int(index), **population[index]}
               for i, index in enumerate(indices)]
    # Freeze the draw before loading patches or computing any appearance score.
    write_json(OUT / "selection.json", {
        "seed": SEED, "generator": "numpy.random.default_rng (PCG64)",
        "rule": "5 without replacement from the frozen audit val list (INRIA Test), no selection by results",
        "population": 288, "audit_sha256": sha(OLD / "dataset_audit.json"), "samples": samples,
    })
    print("Selected:", [Path(s["image"]).name for s in samples], flush=True)

    comparison = read(OLD / "matrix_results.json")["comparison_best"]
    patches, artifacts = {}, {}
    for key, (donor, variant) in METHODS.items():
        item = comparison[key]["cells"][SOURCE][SOURCE]
        assert sha(item["patch_path"]) == item["patch_sha256"]
        patch = load_patch(item["patch_path"])
        assert patch.shape == (3, 300, 300) and patch.isfinite().all()
        assert 0 <= patch.min() <= patch.max() <= 1
        patches[key] = patch
        artifacts[key] = {"donor": donor, "variant": variant,
                          "path": item["patch_path"], "sha256": item["patch_sha256"]}
    recolor_checks = {}
    for raw, versions in [("advpatch", ["p1", "p2"]), ("tsea", ["tsea_bc1", "tsea_bc2"]),
                          ("advreal", ["advreal_bc1", "advreal_bc2"])]:
        for palette, key in zip(("bc1", "bc2"), versions):
            error = float((recolor(patches[raw], BASE_COLORS[palette]) - patches[key]).abs().max())
            assert error < 2e-7, (key, error)
            recolor_checks[key] = error

    # Validate metric scale and identity handling without using study images.
    zeros = np.zeros((32, 32, 3), dtype=np.float64)
    assert abs(metrics(zeros, zeros + 0.1)["psnr_db"] - 20.0) < 1e-10
    assert structural_similarity(zeros, zeros, **SSIM_ARGS) == 1.0
    _, dataset = create_dataloader(check_dataset(config["data_yaml"])["val"],
                                   640, 1, 32, workers=0, shuffle=False, preprocess="advpatch")
    lookup = {path: i for i, path in enumerate(dataset.im_files)}
    assert len(lookup) == len(population) and set(lookup) == {s["image"] for s in population}
    rows, roi_rows, sample_records = [], [], []
    for sample in samples:
        assert sha(sample["image"]) == sample["image_sha256"]
        assert sha(sample["label"]) == sample["label_sha256"]
        image, targets, path, shapes = dataset[lookup[sample["image"]]]
        assert path == sample["image"] and (targets[:, 1] == 0).all()
        assert len(targets) == sample["targets"]
        clean = image.float().div(255).unsqueeze(0)
        reference = tensor_hwc(clean)
        coverage = apply_patch(torch.zeros_like(clean), targets, torch.ones(3, 300, 300), 0.25).bool()
        mask = coverage[0, 0].numpy()
        boxes = []
        for row in targets:
            xc, yc, width, height = (row[2:6] * 640).tolist()
            boxes.append([max(0, math.floor(xc-width/2)), max(0, math.floor(yc-height/2)),
                          min(640, math.ceil(xc+width/2)), min(640, math.ceil(yc+height/2))])
        valid_roi_ids = [i for i, (x0, y0, x1, y1) in enumerate(boxes) if min(x1-x0, y1-y0) >= 11]
        assert valid_roi_ids, "No eligible ROI under fixed 11x11 SSIM protocol"
        with Image.open(path) as original:
            width, height = original.size
        side = max(width, height)
        left, top = (side-width)//2, (side-height)//2
        # Conservative integer crop of source content; boundary filter mixing may remain.
        content_box = [math.ceil(left*640/side), math.ceil(top*640/side),
                       math.floor((left+width)*640/side), math.floor((top+height)*640/side)]
        sample_records.append({**sample, "dataset_index": lookup[path], "person_rois": boxes,
                               "valid_roi_ids": valid_roi_ids, "content_xyxy": content_box,
                               "patch_pixels": int(mask.sum()), "coverage_fraction": float(mask.mean()),
                               "padding_excluded_fraction": 1-(content_box[2]-content_box[0])*(content_box[3]-content_box[1])/640**2})
        arrays = {"clean": reference, "patch_mask": mask, "targets": targets.numpy()}
        save_preview(OUT / "images" / sample["id"] / "clean.png", reference)
        for j, box in enumerate(boxes):
            save_preview(OUT / "roi" / sample["id"] / f"p{j:02d}_clean.png", crop(reference, box))
        for key, patch in patches.items():
            attacked = apply_patch(clean, targets, patch, 0.25)
            assert torch.equal(attacked[~coverage], clean[~coverage])
            candidate = tensor_hwc(attacked)
            arrays[key] = candidate
            full = metrics(reference, candidate)
            masked_mse = float(np.mean((reference.astype(np.float64)[mask]-candidate.astype(np.float64)[mask])**2))
            assert abs(full["mse"] - masked_mse * mask.mean()) < 1e-12
            common = {"sample_id": sample["id"], "image": Path(path).name, "method": key,
                      "donor": METHODS[key][0], "variant": METHODS[key][1], "source": SOURCE}
            rows.append({**common, "region": "full_canvas_640", "roi_count": 0, **full})
            rows.append({**common, "region": "content_rect", "roi_count": 0,
                         **metrics(crop(reference, content_box), crop(candidate, content_box))})
            local = []
            for j in valid_roi_ids:
                values = metrics(crop(reference, boxes[j]), crop(candidate, boxes[j]))
                local.append(values)
                roi_rows.append({**common, "person_index": j, "roi_xyxy": json.dumps(boxes[j]), **values})
            rows.append({**common, "region": "person_roi_per_image", "roi_count": len(local),
                         **{metric: float(np.mean([v[metric] for v in local])) for metric in ("psnr_db", "ssim", "mse")}})
            save_preview(OUT / "images" / sample["id"] / f"{key}.png", candidate)
            for j, box in enumerate(boxes):
                save_preview(OUT / "roi" / sample["id"] / f"p{j:02d}_{key}.png", crop(candidate, box))
        (OUT / "tensors").mkdir(exist_ok=True)
        np.savez_compressed(OUT / "tensors" / f"{sample['id']}.npz", **arrays)
        print("Measured:", sample["id"], Path(path).name, "persons=", len(boxes), flush=True)

    summary = []
    for key, (donor, variant) in METHODS.items():
        for region in ("full_canvas_640", "content_rect", "person_roi_per_image"):
            selected = [r for r in rows if r["method"] == key and r["region"] == region]
            assert len(selected) == 5
            summary.append({"method": key, "donor": donor, "variant": variant, "region": region, "images": 5,
                            "psnr_mean_db": float(np.mean([r["psnr_db"] for r in selected])),
                            "psnr_sd_db": float(np.std([r["psnr_db"] for r in selected], ddof=1)),
                            "ssim_mean": float(np.mean([r["ssim"] for r in selected])),
                            "ssim_sd": float(np.std([r["ssim"] for r in selected], ddof=1))})
    write_csv(OUT / "per_image.csv", rows)
    write_csv(OUT / "per_person.csv", roi_rows)
    write_csv(OUT / "summary.csv", summary)
    manifest = {"status": "complete_five_image_pilot", "seed": SEED, "source": SOURCE,
                "test_population": 288, "selected_images": 5, "methods": artifacts, "samples": sample_records,
                "ssim_parameters": SSIM_ARGS, "data_range": 1.0, "color_space": "RGB, no gamma linearization",
                "pixel_pipeline": config["pixel_pipeline"], "patch_area_fraction": 0.25, "test_eot": False,
                "metric_input": "exact float32 rendered tensors promoted to float64; PNGs are quantized previews only",
                "aggregation": "arithmetic mean of 5 per-image scores; person ROIs averaged within each image first",
                "full_canvas": "640x640 including gray padding",
                "content_rect": "integer crop excluding artificial padding, with possible resize-filter mixing at border",
                "identical_control": {"psnr_db": "inf", "ssim": 1.0, "included_in_mean": False},
                "validations": {"patch_hashes": True, "selected_image_label_hashes": True,
                                "renderer_dataloader_hashes": True, "outside_support_unchanged": True,
                                "mse_support_identity": True, "metric_identity_and_psnr_scale": True,
                                "recolor_max_abs_error": recolor_checks},
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__,
                             "pillow": PIL.__version__, "torch": torch.__version__},
                "code_sha256": sha(__file__), "lpips_computed": False, "attack_evaluation_performed": False}
    write_json(OUT / "manifest.json", manifest)
    lines = ["# 随机五图 PSNR/SSIM 试测", "", f"随机种子 {SEED}；INRIA Test 288 张中无放回抽取 5 张；来源 YOLOv5s。",
             "共 3 donor × 3 版本；以同一干净图为参考，沿用固定色板 Bc1/Bc2。没有真实环境取色、攻击评测或 LPIPS 计算。", "",
             "| 图片 | 人体数 | 补丁支持像素/640² |", "|---|---:|---:|"]
    lines += [f"| {Path(s['image']).name} | {len(s['person_rois'])} | {100*s['coverage_fraction']:.2f}% |" for s in sample_records]
    lines += ["", "人体 ROI 先在每张图内对所有有效人体取平均，再对五图等权平均。SSIM 窗口为 11×11 高斯窗口，σ=1.5，RGB 三通道平均，data_range=1。",
              "数值来自精确 float32 贴图张量（转 float64 计算），PNG 仅用于展示；整图列包含人工灰色 padding。去 padding 结果另存于 summary.csv 的 content_rect。", "",
              "| Donor | 版本 | 整图 PSNR↑ (dB) | 整图 SSIM↑ | 人体 ROI PSNR↑ (dB) | 人体 ROI SSIM↑ |",
              "|---|---|---:|---:|---:|---:|"]
    for key, (donor, variant) in METHODS.items():
        full = next(r for r in summary if r["method"] == key and r["region"] == "full_canvas_640")
        local = next(r for r in summary if r["method"] == key and r["region"] == "person_roi_per_image")
        lines.append(f"| {donor} | {variant} | {full['psnr_mean_db']:.4f} | {full['ssim_mean']:.6f} | {local['psnr_mean_db']:.4f} | {local['ssim_mean']:.6f} |")
    lines += ["", "这些是五图试测，不是全测试集结论；高分表示相对干净参考图的改变更小，不能直接证明与环境更协调或攻击更强。",
              "", "文件：selection.json（抽样）、manifest.json（协议/审计）、per_image.csv（逐图）、per_person.csv（逐人体）、summary.csv（均值/图间SD）、tensors/*.npz（精确输入）、index.html（图片对照）。",
              "", "复现命令（脚本保护已有完整结果，重跑需复制到独立 experiments 子目录并创建/复用依赖环境）：",
              "`experiments/analysis/psnr_ssim_five_20261018/.venv/bin/python -B experiments/analysis/psnr_ssim_five_20261018/run_five.py`"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    cards = []
    for sample in samples:
        cards.append(f"<h2>{sample['id']}: {html.escape(Path(sample['image']).name)}</h2><div class='grid'>")
        for key in ("clean", *METHODS):
            label = "Clean" if key == "clean" else " / ".join(METHODS[key])
            image_path = f"images/{sample['id']}/{key}.png"
            cards.append(f"<article><p>{label}</p><a href='{image_path}'><img src='{image_path}'></a></article>")
        cards.append("</div>")
    (OUT / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>五图试测对照</title>"
        "<style>body{font:16px system-ui;margin:24px}.grid{display:grid;grid-template-columns:repeat(5,minmax(180px,1fr));gap:12px}"
        "img{width:100%}article{border:1px solid #ddd;padding:8px}</style><h1>随机五图 PSNR/SSIM 试测</h1>"
        "<p>固定色板；点击查看原尺寸。PNG 是展示图，精确指标输入保存在 tensors/。"
        "<a href='report.md'>报告</a> · <a href='summary.csv'>指标表</a></p>" + "".join(cards))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
