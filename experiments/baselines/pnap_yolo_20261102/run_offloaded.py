"""Run the frozen PNAP trainer with detector saved tensors stored in CPU memory.

Only tensor storage changes: the detector, batch, precision, objective and
gradient operations remain the same. Keep the original trainer intact so the
already-running YOLOv5s and Faster R-CNN checkpoints remain resumable.
"""
import argparse
import gc
import sys
from common import RUN, read, runtime, sha, write


def offloaded_loss(original):
    import torch

    def call(self, images):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return original(self, images)

    return call


def validate():
    import torch
    from detector_backends import DetectorBackend
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset
    from utils.patch import apply_patch
    from pnap import DetectionLoss

    torch.set_num_threads(2)
    torch.manual_seed(123)
    cfg = read(RUN / "protocol.json")
    data = check_dataset(cfg["data_yaml"], autodownload=False)
    loader, _ = create_dataloader(data["train"], 640, 2, 32, workers=0, preprocess=cfg["preprocess"])
    images, targets, *_ = next(iter(loader))
    images, targets = images.cuda().float() / 255, targets.cuda()
    original = DetectionLoss.__call__
    offloaded = offloaded_loss(original)
    results = []
    for name in ("yolov2", "yolov3", "yolov5m", "yolov4"):
        backend = DetectorBackend(name, device="cuda:0", fused=False, conf_threshold=0.4,
                                  nms_iou=0.45, max_det=300, cache_dir=RUN / "runtime/torchhub")
        values, gradients = [], []
        for function in (original, offloaded):
            patch = torch.full((3, 300, 300), 0.5, device="cuda:0", requires_grad=True)
            loss = function(DetectionLoss(backend), apply_patch(images, targets, patch, frac=0.25))
            loss.backward()
            assert torch.isfinite(loss) and patch.grad is not None
            assert torch.isfinite(patch.grad).all() and patch.grad.norm() > 0
            values.append(loss.detach().cpu())
            gradients.append(patch.grad.detach().cpu())
            assert all(p.grad is None for p in backend.parameters())
            del patch, loss
        result = dict(model=name, loss=float(values[0]),
                      loss_abs_error=float((values[0] - values[1]).abs()),
                      gradient_max_abs_error=float((gradients[0] - gradients[1]).abs().max()),
                      gradient_relative_l2_error=float((gradients[0] - gradients[1]).norm() / gradients[0].norm()),
                      exact_loss_match=torch.equal(*values), exact_gradient_match=torch.equal(*gradients))
        # CUDA reductions can differ by roundoff when saved tensors are copied
        # back with a different memory layout; require tight absolute AND relative bounds.
        result["passed"] = (result["exact_loss_match"] and result["gradient_max_abs_error"] <= 1e-8
                            and result["gradient_relative_l2_error"] <= 1e-5)
        print(result, flush=True)
        assert result["passed"], result
        results.append(result)
        del backend, values, gradients
        gc.collect()
        torch.cuda.empty_cache()
    write(RUN / "evidence/detector_offload_equivalence.json",
          dict(passed=True, wrapper_sha256=sha(__file__), batch_size=2,
               gradient_max_abs_tolerance=1e-8, gradient_relative_l2_tolerance=1e-5, results=results))


def main():
    runtime()
    if sys.argv[1:] == ["validate"]:
        validate()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train",))
    parser.add_argument("--model", choices=("yolov2", "yolov3", "yolov5m", "yolov4"), required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    evidence = read(RUN / "evidence/detector_offload_equivalence.json")
    if not evidence["passed"] or evidence["wrapper_sha256"] != sha(__file__):
        raise RuntimeError("Detector CPU offload equivalence must pass for this wrapper")
    import torch
    import pnap
    import run_experiments

    output = RUN / ("smoke" if args.smoke_steps else "sources") / args.model
    profile_path = output / "execution_profile.json"
    profile = dict(detector_saved_tensors="cpu_pinned", wrapper_sha256=sha(__file__),
                   protocol_sha256=sha(RUN / "protocol.json"), torch_version=torch.__version__,
                   equivalence_sha256=sha(RUN / "evidence/detector_offload_equivalence.json"))
    if profile_path.exists() and read(profile_path) != profile:
        raise RuntimeError("Refusing to mix execution profiles")
    if (output / "training_state.pt").exists() and not profile_path.exists():
        raise RuntimeError("Existing checkpoint has no CPU offload execution profile")
    write(profile_path, profile)
    pnap.DetectionLoss.__call__ = offloaded_loss(pnap.DetectionLoss.__call__)
    run_experiments.main()


if __name__ == "__main__":
    main()
