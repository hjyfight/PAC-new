"""Focused numerical, gradient and batch-eight memory validation for the new methods."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PROJECT = ROOT / "yolov5-tsea-nap-20260912"
os.environ.update(json.loads((RUN / "runtime_environment.json").read_text()))
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from detector_backends import DetectorBackend
from utils.augmentations import TSEAImageAugment
from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.patch import AdvPatch, EOT, NAPBigGANPatch, TSEAEOT, apply_patch, load_patch, total_variation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
    if isinstance(a, dict):
        return max(compare(a[k], b[k]) for k in a)
    return max(compare(x, y) for x, y in zip(a, b, strict=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--method", choices=("tsea", "nap"), required=True)
    args = parser.parse_args()
    name, method = args.detector, args.method
    config = json.loads((ROOT / "experiments/advpatch_cap_completion_20260911/manifest.json").read_text())
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    started = time.monotonic()
    model = DetectorBackend(name, device=device, image_size=640, fused=False, tsea=method == "tsea")
    assert digest(model.weight_path) == config["detectors"][name]["sha256"]
    assert all(not parameter.requires_grad for parameter in model.parameters())
    loader, _ = create_dataloader(check_dataset(config["data_yaml"])["train"], 640, 8, 32, workers=4,
                                  shuffle=False, preprocess="advpatch", seed=config["seed"],
                                  source_transform=TSEAImageAugment() if method == "tsea" else None)
    assert len(loader.dataset) == 614 and len(loader) == 77
    image_u8, targets, paths, _ = next(iter(loader))
    assert image_u8.shape == (8, 3, 640, 640) and all("/images/train/" in path for path in paths)
    images = image_u8.to(device).float() / 255
    parity, stochastic = None, None
    if method == "tsea":
        model.set_tsea(False)
        reference = DetectorBackend(name, device=device, image_size=640, fused=False)
        with torch.no_grad():
            parity = compare(model(images[:1]), reference(images[:1]))
        del reference
        model.set_tsea(True)
        if name != "yolov2":
            assert model.tsea_layout
            with torch.no_grad():
                first = model.advreal_loss(images[:2])
                second = model.advreal_loss(images[:2])
            stochastic = float((first - second).abs())
            assert stochastic > 0, (name, "model ensemble has no observed effect")
        else:
            assert not model.tsea_layout
        module = AdvPatch(300).to(device)
        eot = TSEAEOT().to(device)
        optimizer = Adam(module.parameters(), lr=0.03, amsgrad=True)
        variable = module.values
    else:
        module = NAPBigGANPatch(RUN / "author/Naturalistic-Adversarial-Patch",
                                RUN / "weights/NAP_BigGAN_G_ema.pth", 300, 259).to(device)
        assert module.generator.dim_z == 120 and module.generator.training is False
        assert all(not parameter.requires_grad for parameter in module.generator.parameters())
        eot = EOT(rotation=(0, 0), brightness=(-0.1, 0.1), noise=(-0.1, 0.1),
                  scale=(1, 1), smooth=1, value_range=(1e-6, 0.99999)).to(device)
        optimizer = Adam([module.latent], lr=0.02 if name == "fasterrcnn" else 0.005,
                         betas=(0.5, 0.999), amsgrad=True)
        variable = module.latent
    buffers = {key: value.clone() for key, value in module.named_buffers()} if method == "nap" else {}
    steps = []
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        if method == "nap":
            with torch.no_grad():
                module.latent.mul_(10000).round_().mul_(1e-4).clamp_(-10, 10)
            native = module.native_patch()
            assert native.shape == (3, 128, 128) and native.min() >= 0 and native.max() <= 1
            patch = F.interpolate(native.unsqueeze(0), size=(300, 300), mode="bilinear", align_corners=False)[0]
            tv = 0.1 * total_variation(native)
        else:
            patch = module()
            tv = (2.5 * total_variation(patch)).clamp(min=0.1)
        before = variable.detach().clone()
        attacked = apply_patch(images, targets, patch, frac=0.25, eot=eot)
        det = model.nap_loss(attacked) if method == "nap" else model.advreal_loss(attacked)
        det_grad = torch.autograd.grad(det, variable, retain_graph=True)[0]
        assert det_grad.isfinite().all() and det_grad.abs().sum() > 0
        loss = det + tv
        assert torch.isfinite(loss)
        loss.backward()
        assert variable.grad.isfinite().all()
        assert all(parameter.grad is None for parameter in model.parameters())
        if method == "nap":
            assert all(parameter.grad is None for parameter in module.generator.parameters())
            assert all(torch.equal(value, dict(module.named_buffers())[key]) for key, value in buffers.items())
        optimizer.step()
        if method == "tsea":
            with torch.no_grad():
                variable.clamp_(0, 1)
        update = float((variable.detach() - before).abs().max())
        assert update > 0
        steps.append({"step": iteration + 1, "loss": float(loss.detach()), "detection": float(det.detach()),
                      "tv": float(tv.detach()), "detection_gradient_abs_sum": float(det_grad.abs().sum()),
                      "parameter_update_max": update})
        del patch, attacked, det, det_grad, loss, tv, before
        if method == "nap":
            del native
    output = RUN / "smoke" / f"{method}_{name}.pt"
    torch.save(module.checkpoint(), output)
    with torch.no_grad():
        compare(load_patch(output, device), module())
    torch.cuda.synchronize()
    record = {"passed": True, "method": method, "detector": name, "batch_size": 8, "images": 614,
              "batches_per_epoch": 77, "train_images_only": True, "detector_frozen": True,
              "finite_nonzero_detection_gradient": True, "steps": steps,
              "reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "disabled_ensemble_max_error": parity, "enabled_ensemble_loss_difference": stochastic,
              "tsea_residual_locations": model.tsea_layout,
              "nap_generator_frozen_and_buffers_unchanged": method == "nap",
              "seconds": time.monotonic() - started}
    (RUN / "smoke" / f"{method}_{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
