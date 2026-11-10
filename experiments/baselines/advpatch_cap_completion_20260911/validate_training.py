"""Focused batch-eight gradient and memory check before each AdvPatch source run."""
import argparse
import time

from prepare_recolor import PROJECT, RUN, read, sha, write
import numpy as np
import torch
from torch.optim import Adam
from detector_backends import DetectorBackend
from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.patch import AdvPatch, EOT, apply_patch, attack_loss, non_printability, total_variation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    args = parser.parse_args()
    name = args.detector
    config = read(RUN / "manifest.json")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    started = time.monotonic()
    model = DetectorBackend(name, device=device, image_size=640, fused=False)
    assert sha(model.weight_path) == config["detectors"][name]["sha256"]
    assert all(not p.requires_grad for p in model.parameters())
    loader, _ = create_dataloader(check_dataset(config["data_yaml"])["train"], 640, 8, 32,
                                   workers=4, shuffle=False, preprocess="advpatch", seed=config["seed"])
    assert len(loader.dataset) == 614 and len(loader) == 77
    image_u8, targets, paths, _ = next(iter(loader))
    assert image_u8.shape == (8, 3, 640, 640) and all("/images/train/" in p for p in paths)
    images = image_u8.to(device).float() / 255
    module = AdvPatch(300).to(device)
    eot = EOT(brightness=(-0.1, 0.1), noise=(-0.1, 0.1), scale=(1, 1),
              smooth=7, median=True, value_range=(1e-6, 0.99999)).to(device)
    optimizer = Adam(module.parameters(), lr=0.03, amsgrad=True)
    captured, handles = {}, []
    if name == "fasterrcnn":
        def capture_counts(layer, inputs):
            captured["counts"] = [len(proposals) for proposals in inputs[1]]
        def capture_logits(layer, inputs, output):
            captured["roi_scores"] = output[0].detach().softmax(-1)[:, 1]
        handles = [model.detector.roi_heads.register_forward_pre_hook(capture_counts),
                   model.detector.roi_heads.box_predictor.register_forward_hook(capture_logits)]
    else:
        def capture_output(layer, inputs, output):
            captured["per_image_scores"] = (output[..., 4] * output[..., 5]).detach().amax(dim=1)
        handles = [model.register_forward_hook(capture_output)]
    steps, equivalence = [], None
    try:
        for iteration in range(2):
            captured.clear()
            optimizer.zero_grad(set_to_none=True)
            before = module.values.detach().clone()
            patch = module()
            attacked = apply_patch(images, targets, module, frac=0.25, eot=eot)
            det = model.advpatch_loss(attacked)
            if name == "fasterrcnn":
                per_image = torch.stack([p.max() if p.numel() else p.sum()
                                         for p in captured["roi_scores"].split(captured["counts"])])
                assert model.roi_logits is None
                assert len(model.detector.roi_heads._forward_pre_hooks) == 1
            else:
                per_image = captured["per_image_scores"]
            assert per_image.shape == (8,)
            assert torch.allclose(det.detach(), per_image.mean(), atol=1e-7, rtol=1e-6)
            det_grad = torch.autograd.grad(det, module.values, retain_graph=True)[0]
            assert det_grad.isfinite().all() and det_grad.abs().sum() > 0
            if name == "yolov5s" and iteration == 0:
                original = attack_loss(model.detector(attacked)[0])
                original_grad = torch.autograd.grad(original, module.values, retain_graph=True)[0]
                assert torch.allclose(original, det, atol=1e-7, rtol=1e-6)
                assert torch.allclose(original_grad, det_grad, atol=1e-7, rtol=1e-5)
                equivalence = {"loss_max_absolute_error": float((original-det).abs()),
                               "gradient_max_absolute_error": float((original_grad-det_grad).abs().max())}
                del original, original_grad
            tv = (2.5 * total_variation(patch)).clamp(min=0.1)
            nps = 0.01 * non_printability(patch)
            loss = det + tv + nps
            assert torch.isfinite(loss)
            loss.backward()
            assert module.values.grad.isfinite().all()
            assert all(p.grad is None for p in model.parameters())
            optimizer.step()
            with torch.no_grad():
                module.values.clamp_(0, 1)
            update = float((module.values.detach() - before).abs().max())
            assert update > 0
            steps.append({"step": iteration + 1, "loss": float(loss.detach()),
                          "detection": float(det.detach()), "nps": float(nps.detach()), "tv": float(tv.detach()),
                          "detection_gradient_abs_sum": float(det_grad.abs().sum()),
                          "patch_update_max": update, "per_image_scores": per_image.cpu().tolist()})
            del patch, attacked, det, det_grad, loss, nps, tv, before, per_image
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()
    record = {"passed": True, "detector": name, "batch_size": 8, "images": 614,
              "batches_per_epoch": 77, "train_images_only": True, "detector_frozen": True,
              "finite_nonzero_detection_gradient": True, "steps": steps,
              "reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "selected_yolov5s_equivalence": equivalence, "seconds": time.monotonic() - started}
    write(RUN / "smoke" / f"{name}.json", record)
    print(record, flush=True)


if __name__ == "__main__":
    main()
