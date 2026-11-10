"""Check actual detector gradients and independently calculated regularizers."""
import argparse
import gc
from common import AUTHOR, MODELS, RUN, read, runtime, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    runtime()
    import torch
    from detector_backends import DetectorBackend
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset
    from utils.patch import apply_patch
    from pnap import DetectionLoss, regularizers
    torch.set_num_threads(2)
    torch.manual_seed(123)
    device = torch.device("cuda:0")
    cfg = read(RUN / "protocol.json")
    data = check_dataset(cfg["data_yaml"], autodownload=False)
    loader, _ = create_dataloader(data["train"], 640, 2, 32, workers=0, preprocess="advpatch")
    images, targets, *_ = next(iter(loader))
    images, targets = images.to(device).float()/255, targets.to(device)
    nps, tv = regularizers(device)
    patch = torch.rand(3, 512, 512, device=device, requires_grad=True)
    colors = torch.tensor([[float(v) for v in row.split(",")] for row in
                           (AUTHOR / "non_printability/30values.txt").read_text().splitlines()], device=device)
    pixels = patch.permute(1, 2, 0).reshape(-1, 3)
    expected_nps = (((pixels[:, None] - colors[None] + 1e-6)**2).sum(-1) + 1e-6).sqrt().min(1).values.mean()/3
    expected_tv = ((patch[:, :, 1:] - patch[:, :, :-1] + 1e-6).abs().sum() +
                   (patch[:, 1:] - patch[:, :-1] + 1e-6).abs().sum()) / patch.numel()
    assert torch.allclose(nps(patch), expected_nps, atol=1e-6)
    assert torch.allclose(tv(patch), expected_tv, atol=1e-6)
    del patch, pixels, expected_nps, expected_tv, nps, tv
    results = []
    for name in ((args.model,) if args.model else MODELS):
        backend = DetectorBackend(name, device=device, fused=False, conf_threshold=0.4,
                                   nms_iou=0.45, cache_dir=RUN / "runtime/torchhub")
        patch = torch.full((3, 300, 300), 0.5, device=device, requires_grad=True)
        loss = DetectionLoss(backend)(apply_patch(images, targets, patch, frac=0.25))
        loss.backward()
        assert torch.isfinite(loss) and patch.grad is not None and torch.isfinite(patch.grad).all()
        assert float(patch.grad.norm()) > 0, f"No patch gradient: {name}"
        assert all(p.grad is None for p in backend.parameters())
        result = dict(model=name, loss=float(loss), patch_grad_norm=float(patch.grad.norm()), passed=True)
        print(result, flush=True); results.append(result)
        del backend, patch, loss
        gc.collect(); torch.cuda.empty_cache()
    write(RUN / "evidence" / ("detector_validation_" + (args.model or "all") + ".json"),
          dict(passed=True, regularizers_match_independent_formulas=True, detectors=results))


if __name__ == "__main__":
    main()
