#!/usr/bin/env python3
"""Compute LPIPS on the frozen five-image PSNR/SSIM pilot tensors."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


OUT = Path(__file__).resolve().parent
DEPS = OUT / ".deps"
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))
import lpips  # noqa: E402


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
ROI_SIZE = 224


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_nchw(array: np.ndarray) -> torch.Tensor:
    """Convert RGB float images in [0, 1] from HWC to NCHW."""
    if array.ndim == 3:
        array = array[None]
    return torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float()


def resize_roi(array: np.ndarray) -> torch.Tensor:
    """Resize an HWC RGB ROI to the fixed AlexNet evaluation size."""
    tensor = to_nchw(array)
    return F.interpolate(tensor, size=(ROI_SIZE, ROI_SIZE), mode="bicubic", align_corners=False)


def lpips_batch(model, reference: torch.Tensor, candidate: torch.Tensor) -> np.ndarray:
    """Evaluate a batch of [0, 1] RGB tensors with official LPIPS scaling."""
    reference = reference.mul(2).sub(1).clamp(-1, 1)
    candidate = candidate.mul(2).sub(1).clamp(-1, 1)
    values = model(reference, candidate)
    return values.reshape(-1).detach().cpu().numpy().astype(np.float64)


@torch.inference_mode()
def main() -> None:
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete_five_image_pilot":
        raise RuntimeError("Unexpected source manifest status")
    if (OUT / "lpips_summary.csv").exists():
        raise SystemExit("LPIPS output already exists; preserve it and use a separate run directory.")

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", version="0.1", spatial=False).to(device)
    model.eval()

    full_rows: list[dict] = []
    person_rows: list[dict] = []
    sample_records: list[dict] = []
    for sample in manifest["samples"]:
        sample_id = sample["id"]
        tensor_path = OUT / "tensors" / f"{sample_id}.npz"
        if not tensor_path.exists():
            raise FileNotFoundError(tensor_path)
        arrays = np.load(tensor_path)
        reference = arrays["clean"].astype(np.float32, copy=False)
        boxes = sample["person_rois"]
        valid_roi_ids = sample["valid_roi_ids"]
        if not valid_roi_ids:
            raise RuntimeError(f"No valid ROI for {sample_id}")
        sample_records.append({
            "sample_id": sample_id,
            "image": sample["image"],
            "tensor_path": str(tensor_path),
            "tensor_sha256": sha256(tensor_path),
            "roi_size": [ROI_SIZE, ROI_SIZE],
            "roi_count": len(valid_roi_ids),
        })

        reference_full = to_nchw(reference).to(device)
        reference_rois = torch.cat(
            [resize_roi(reference[boxes[j][1]:boxes[j][3], boxes[j][0]:boxes[j][2]])
             for j in valid_roi_ids], dim=0
        ).to(device)
        for key, (donor, variant) in METHODS.items():
            candidate = arrays[key].astype(np.float32, copy=False)
            full_score = float(lpips_batch(model, reference_full, to_nchw(candidate).to(device))[0])
            common = {
                "sample_id": sample_id,
                "image": Path(sample["image"]).name,
                "method": key,
                "donor": donor,
                "variant": variant,
            }
            full_rows.append({**common, "region": "full_canvas_640", "roi_count": 0,
                              "lpips": full_score})

            candidate_rois = torch.cat(
                [resize_roi(candidate[boxes[j][1]:boxes[j][3], boxes[j][0]:boxes[j][2]])
                 for j in valid_roi_ids], dim=0
            ).to(device)
            roi_scores = lpips_batch(model, reference_rois, candidate_rois)
            for j, score in zip(valid_roi_ids, roi_scores):
                person_rows.append({**common, "person_index": j,
                                    "roi_xyxy": json.dumps(boxes[j]), "roi_size": ROI_SIZE,
                                    "lpips": float(score)})
            full_rows.append({**common, "region": "person_roi_per_image",
                              "roi_count": len(roi_scores), "lpips": float(np.mean(roi_scores))})
        print(f"Measured {sample_id} {Path(sample['image']).name} persons={len(valid_roi_ids)}", flush=True)

    summary: list[dict] = []
    for key, (donor, variant) in METHODS.items():
        for region in ("full_canvas_640", "person_roi_per_image"):
            selected = [r for r in full_rows if r["method"] == key and r["region"] == region]
            values = np.asarray([r["lpips"] for r in selected], dtype=np.float64)
            summary.append({
                "method": key,
                "donor": donor,
                "variant": variant,
                "region": region,
                "images": len(values),
                "lpips_mean": float(values.mean()),
                "lpips_sd": float(values.std(ddof=1)),
            })

    write_csv(OUT / "lpips_per_image.csv", full_rows)
    write_csv(OUT / "lpips_per_person.csv", person_rows)
    write_csv(OUT / "lpips_summary.csv", summary)
    lpips_manifest = {
        "status": "complete_five_image_pilot",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "selected_images": len(manifest["samples"]),
        "methods": manifest["methods"],
        "samples": sample_records,
        "metric": "LPIPS",
        "package": "lpips==0.1.4",
        "model": "AlexNet",
        "version": "0.1",
        "device": str(device),
        "input_color": "RGB",
        "input_range": "[-1, 1] after mapping from [0, 1]",
        "full_canvas": "640x640, including gray padding",
        "roi_protocol": f"Each GT person ROI is bicubic-resized to {ROI_SIZE}x{ROI_SIZE}; scores averaged within image, then across five images",
        "direction": "lower is more perceptually similar to the clean reference",
        "tensor_input": "exact float32 rendered tensors from tensors/*.npz; PNG previews are not used",
        "aggregation": "arithmetic mean of five per-image scores; person ROIs averaged within each image first",
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
        "files": {
            "per_image": "lpips_per_image.csv",
            "per_person": "lpips_per_person.csv",
            "summary": "lpips_summary.csv",
        },
    }
    (OUT / "lpips_manifest.json").write_text(json.dumps(lpips_manifest, ensure_ascii=False, indent=2) + "\n")

    report_path = OUT / "report.md"
    report = report_path.read_text()
    report = report.replace(
        "共 3 donor × 3 版本；以同一干净图为参考，沿用固定色板 Bc1/Bc2。没有真实环境取色、攻击评测或 LPIPS 计算。",
        "共 3 donor × 3 版本；以同一干净图为参考，沿用固定色板 Bc1/Bc2。没有真实环境取色或攻击评测。",
    )
    marker = "\n## LPIPS\n"
    if marker not in report:
        lines = [marker.rstrip(), "", "使用官方 LPIPS AlexNet（`lpips==0.1.4`, version `0.1`），输入由 `[0,1]` 映射到 `[-1,1]`。LPIPS 越低表示与干净参考图的深层特征越相似。整图直接使用 640×640 张量；人体 ROI 统一双三次缩放到 224×224 后计算，先在每张图内平均，再对五张图等权平均。", "", "| Donor | 版本 | 整图 LPIPS↓ | 人体 ROI LPIPS↓ |", "|---|---|---:|---:|"]
        for key, (donor, variant) in METHODS.items():
            full = next(r for r in summary if r["method"] == key and r["region"] == "full_canvas_640")
            local = next(r for r in summary if r["method"] == key and r["region"] == "person_roi_per_image")
            lines.append(f"| {donor} | {variant} | {full['lpips_mean']:.6f} | {local['lpips_mean']:.6f} |")
        lines += ["", "LPIPS 只描述相对干净参考图的深层特征距离，不能单独代表真实环境中的隐蔽性；完整协议和逐图/逐人体结果见 `lpips_manifest.json`、`lpips_per_image.csv` 和 `lpips_per_person.csv`。", ""]
        report += "\n" + "\n".join(lines)
    report_path.write_text(report)

    manifest["lpips_computed"] = True
    manifest["lpips_results"] = {
        "protocol": "official LPIPS AlexNet v0.1; full canvas direct; person ROI bicubic-resized to 224x224",
        "summary": "lpips_summary.csv",
        "manifest": "lpips_manifest.json",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote LPIPS outputs to {OUT}")


if __name__ == "__main__":
    main()
