"""Read-only CPU audit of the existing AdvPatch artifacts and author formulas."""

import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)


def read_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


current = read_module(ROOT / "experiments/baselines/yolov5-advpatch-cap-20260911/utils/patch.py", "audit_current_patch")
old_scope = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math}
old_tree = ast.parse((ROOT / "experiments/ours/yolov5-advpad-20260908/utils/patch.py").read_text())
old_selected = [n for n in old_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in {"AdvPatch", "EOT", "median_pool", "apply_patch"}]
exec(compile(ast.Module(body=old_selected, type_ignores=[]), "old_selected_patch_code", "exec"), old_scope)
old = SimpleNamespace(**old_scope)
scope = {"torch": torch, "nn": nn, "np": np}
tree = ast.parse((OUT / "official_load_data.py").read_text())
selected = [n for n in tree.body if isinstance(n, ast.ClassDef)
            and n.name in {"NPSCalculator", "TotalVariation", "MaxProbExtractor"}]
exec(compile(ast.Module(body=selected, type_ignores=[]), "official_selected_classes", "exec"), scope)
tree = ast.parse((ROOT / "experiments/baselines/yolov5-advpatch-cap-20260911/detector_backends.py").read_text())
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_yolov2"]
exec(compile(ast.Module(body=selected, type_ignores=[]), "current_decode", "exec"), scope)

torch.manual_seed(1788272031)
patch = torch.rand(3, 23, 23, requires_grad=True)
official_nps = scope["NPSCalculator"](
    str(ROOT / "experiments/baselines/adversarial-yolo-master/non_printability/30values.txt"), 23)
formula_checks = {}
for name, local, reference in [
    ("nps", current.non_printability(patch), official_nps(patch)),
    ("tv", current.total_variation(patch), scope["TotalVariation"]()(patch)),
]:
    grad_local = torch.autograd.grad(local, patch, retain_graph=True)[0]
    grad_ref = torch.autograd.grad(reference, patch, retain_graph=True)[0]
    formula_checks[name] = {"current": float(local.detach()), "official": float(reference.detach()),
                            "max_gradient_error": float((grad_local - grad_ref).abs().max())}
    torch.testing.assert_close(local, reference)
    torch.testing.assert_close(grad_local, grad_ref)

raw = torch.randn(2, 425, 5, 5, requires_grad=True)
extractor = scope["MaxProbExtractor"](0, 80, SimpleNamespace(loss_target=lambda obj, cls: obj * cls))
reference = extractor(raw).mean()
decoded = scope["decode_yolov2"](raw, [1.0, 1.0] * 5, (640, 640))
local = current.attack_loss(decoded)
torch.testing.assert_close(local, reference)
torch.testing.assert_close(torch.autograd.grad(local, raw, retain_graph=True)[0],
                           torch.autograd.grad(reference, raw)[0])
formula_checks["obj_cls"] = {"current": float(local.detach()), "official": float(reference.detach()),
                              "outputs_and_gradients_match": True}

eot_args = dict(brightness=(-0.1, 0.1), noise=(-0.1, 0.1), scale=(1, 1),
                smooth=7, median=True, value_range=(1e-6, 0.99999))
images = torch.rand(2, 3, 100, 100)
targets = torch.tensor([[0, 0, 0.5, 0.5, 0.4, 0.6], [1, 0, 0.3, 0.4, 0.5, 0.8]])
outputs = []
for module in (old, current):
    np.random.seed(31)
    torch.manual_seed(31)
    texture = module.AdvPatch(31)
    assert bool((texture.values == 0.5).all()) and not texture.use_logits
    attacked = module.apply_patch(images, targets, texture, 0.25, module.EOT(**eot_args))
    attacked.square().mean().backward()
    assert torch.isfinite(texture.values.grad).all() and texture.values.grad.abs().sum() > 0
    outputs.append((attacked.detach(), texture.values.grad))
torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)

rotation = current.EOT(rotation=(20, 20), brightness=(0, 0), contrast=(1, 1),
                       noise=(0, 0), scale=(1, 1), smooth=1)
_, coverage = rotation(torch.ones(3, 300, 300))
angle = math.radians(20)
theta = torch.tensor([[[math.cos(angle), -math.sin(angle), 0],
                       [math.sin(angle), math.cos(angle), 0]]])
canvas = F.pad(torch.ones(1, 1, 300, 300), (150, 150, 150, 150))
grid = F.affine_grid(theta, canvas.shape, align_corners=False)
full_coverage = F.grid_sample(canvas, grid, align_corners=False)
rotation_check = {
    "angle_degrees": 20,
    "current_retained_area_fraction": float(coverage.sum() / 90000),
    "padded_canvas_retained_area_fraction": float(full_coverage.sum() / 90000),
    "scope": "Isolated geometric coverage check; not an AP comparison or a re-run of author training.",
}

manifest = json.loads((ROOT / "experiments/analysis/pac_patch_visualizations_20260910/current_20261110/manifest.json").read_text())
training = []
for item in manifest["patches"]:
    if item["method"] != "advpatch":
        continue
    checkpoint = Path(item["checkpoint"])
    rows = list(csv.DictReader((checkpoint.parent / "results.csv").open()))
    assert [int(r["epoch"]) for r in rows] == list(range(1, 201))
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == item["checkpoint_sha256"]
    best = min(rows, key=lambda r: float(r["loss"]))
    training.append({"source": item["source"], "epochs": len(rows), "best_epoch": int(best["epoch"]),
                     "first": rows[0], "best": best, "last": rows[-1],
                     "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                     "all_losses_finite": True, "checkpoint_matches_current_evaluation": True})

result = {"passed": True, "formula_checks": formula_checks,
          "old_s000_and_new_advpatch_placement_output_and_gradient_match": True,
          "gray_initialization_and_patch_gradient_checked": True,
          "rotation_difference": rotation_check, "training": training,
          "limitations": ["CPU formula and synthetic input checks; no new detector training or evaluation.",
                          "Finite losses and completed epochs do not establish convergence or optimality."]}
(OUT / "audit_checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "training"}, indent=2))
print("Verified complete 200-epoch logs and checkpoint hashes for", len(training), "sources.")
