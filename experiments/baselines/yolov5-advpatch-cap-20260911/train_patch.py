# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train PAC / CAPGen adversarial patches against a YOLOv5 detector.

AdvPatch, CAPGen-T, AdvReal, and CAP run gradient descent. CAPGen-P and CAPGen-R
are analytic constructions that do not train a new patch.

Usage - AdvPatch (color-unrestricted):
    $ python train_patch.py --data data/inria.yaml --weights yolov5s.pt --method advpatch

Usage - CAPGen-T / R / P (Bc1 or Bc2 base colors):
    $ python train_patch.py --method capgen-t --base-colors bc1 --data data/inria.yaml
    $ python train_patch.py --method capgen-r --base-colors bc1
    $ python train_patch.py --method capgen-p --base-colors bc1 --source runs/patch/advpatch/best.pt

Evaluate any resulting patch with:
    $ python val.py --data data/inria.yaml --weights yolov5s.pt --patch runs/patch/advpatch/best.pt
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.general import LOGGER, TQDM, check_dataset, colorstr, increment_path, print_args, yaml_save
from utils.patch import (
    BASE_COLORS,
    EOT,
    AdvPatch,
    AdvRealEOT,
    CAPRotation,
    ColorProbPatch,
    advreal_2d_loss,
    apply_patch,
    attack_loss,
    load_patch,
    non_printability,
    recolor,
    total_variation,
)
from utils.torch_utils import select_device


def save_patch(ckpt, patch, path):
    """Writes a patch checkpoint plus a PNG preview next to it."""
    torch.save(ckpt, path)
    arr = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path.with_suffix(".png"))


def load_cap_isp(repo, weights, device):
    """Load the released CAP ISP proxy without importing its conflicting YOLO utilities."""
    package = Path(repo) / "ispNet"
    spec = importlib.util.spec_from_file_location(
        "cap_ispnet", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    network = module.UNet(n_channels=3, output_channels=3, bilinear=False)
    network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return network.to(device).eval().requires_grad_(False)


def cap_training_step(module, isp_net, isp_params, optimizers, confidence, images, targets, iteration,
                      frac, eot, tv_weight=20.0):
    """Run one released CAP phase: 20 ISP updates then 20 RGB-patch updates, globally."""
    isp_phase = iteration % 40 < 20
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    raw = module()
    rendered = isp_net((raw.detach() if isp_phase else raw).unsqueeze(0),
                       isp_params if isp_phase else isp_params.detach())[0]
    det = confidence(apply_patch(images, targets, rendered, frac=frac, eot=eot))
    tv = raw.new_zeros(()) if isp_phase else tv_weight * total_variation(raw)
    loss = 1 - det if isp_phase else 10 * det + tv
    if not torch.isfinite(loss):
        raise FloatingPointError("CAP objective is non-finite")
    loss.backward()
    optimizers[int(isp_phase)].step()  # tuple order: patch, ISP
    with torch.no_grad():
        (isp_params if isp_phase else module.values).clamp_(0, 1)
    return loss.detach(), det.detach(), tv.detach(), isp_phase


def run(
    data=ROOT / "data/inria.yaml",
    weights=ROOT / "yolov5s.pt",
    method="advpatch",
    detector=None,
    base_colors="bc1",
    source=None,
    patch_size=300,
    imgsz=640,
    frac=0.25,
    tau=0.1,
    epochs=None,
    batch_size=8,
    lr=None,
    cap_repo="/datacc05/huotianlin/code-hu/PAC/third_party/CAP_neurips2024",
    cap_isp_weights="/datacc05/huotianlin/code-hu/PAC/third_party/weights/CAP_ISPNet_author.pth",
    cap_isp_lr=0.0001,
    tv_weight=None,
    nps_weight=None,
    num_colors=3,
    no_eot=False,
    preprocess="letterbox",
    device="",
    workers=8,
    seed=0,
    project=ROOT / "runs/patch",
    name="exp",
    exist_ok=False,
):
    """Builds or trains one adversarial patch and saves it under `project/name`."""
    if epochs is None:
        epochs = 1000 if method == "cap" else 200
    if lr is None:
        lr = 0.01 if method == "cap" else 0.03
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    if tv_weight is None:
        tv_weight = 20.0 if method == "cap" else (2.5 if method in ("advpatch", "advreal") else 0.0)
    if nps_weight is None:
        nps_weight = 0.01 if method == "advpatch" else 0.0
    if method == "cap" and nps_weight:
        raise ValueError("Released CAP has no NPS term")
    options = dict(locals())
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    yaml_save(save_dir / "opt.yaml", options)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device, batch_size=batch_size)
    colors = BASE_COLORS[base_colors]

    # CAPGen-P: recolor a trained AdvPatch, no optimization (paper Eq.4).
    if method == "capgen-p":
        assert source, "--source is required for capgen-p (path to a trained AdvPatch)"
        patch = recolor(load_patch(source, device), colors)
        save_patch({"patch": patch.cpu(), "patch_size": patch.shape[-1]}, patch, save_dir / "best.pt")
        LOGGER.info(f"CAPGen-P built from {source} with {base_colors.upper()} -> {save_dir / 'best.pt'}")
        return save_dir

    # AdvReal's 2D branch reuses the free-pixel parameterization; only its augmentation and
    # loss differ, so it shares the AdvPatch module.
    module = (
        AdvPatch(patch_size).to(device)
        if method in ("advpatch", "advreal", "cap")
        else ColorProbPatch(colors[:num_colors], patch_size, tau).to(device)
    )

    if method == "cap":
        # CAP initializes the raw patch and six normalized camera parameters uniformly.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with torch.no_grad():
            module.values.uniform_(0, 1)
        isp_params = torch.nn.Parameter(torch.rand(1, 6, device=device))
        isp_net = load_cap_isp(cap_repo, cap_isp_weights, device)

    # CAPGen-R: the untrained random color-probability matrix. Sharing --seed with a
    # capgen-t run gives both methods the same initial matrix, as the paper requires.
    if method == "capgen-r":
        save_patch(module.checkpoint(), module(), save_dir / "best.pt")
        LOGGER.info(f"CAPGen-R (untrained, seed {seed}, {base_colors.upper()}) -> {save_dir / 'best.pt'}")
        return save_dir

    if detector is None:
        model = attempt_load(weights, device=device, fuse=False).eval().requires_grad_(False)
        stride = int(model.stride.max())
    else:
        if method not in ("advpatch", "advreal", "cap"):
            raise ValueError("--detector selects a native AdvPatch, AdvReal-2D or CAP training source")
        from detector_backends import DetectorBackend
        model = DetectorBackend(detector, device=device, image_size=imgsz, fused=False)
        if Path(weights).resolve() != model.weight_path.resolve():
            raise ValueError(f"--weights does not match the configured {detector} checkpoint: {model.weight_path}")
        stride = 32
    train_path = check_dataset(data)["train"]
    dataloader = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        stride,
        workers=workers,
        shuffle=True,
        seed=seed,
        prefix=colorstr("patch: "),
        preprocess=preprocess,
    )[0]

    eot = None
    if method == "cap":
        # ISP optimization is intrinsic to CAP; --no-eot disables only its random rotation.
        eot = CAPRotation(rotation=(0, 0) if no_eot else (-10, 10)).to(device)
    elif not no_eot:
        if method == "capgen-t":
            eot = EOT().to(device)
        elif method == "advpatch":
            eot = EOT(
                brightness=(-0.1, 0.1), noise=(-0.1, 0.1), scale=(1, 1),
                smooth=7, median=True, value_range=(1e-6, 0.99999),
            ).to(device)
        else:
            eot = AdvRealEOT().to(device)
    optimizer = Adam(module.parameters(), lr=lr, amsgrad=method in ("advpatch", "advreal", "cap"))
    if method == "cap":
        isp_optimizer = Adam([isp_params], lr=cap_isp_lr, amsgrad=True)
        confidence = model.cap_loss if detector is not None else lambda images: advreal_2d_loss(model(images)[0])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=50) if method == "advpatch" else None
    LOGGER.info(
        f"{colorstr('patch: ')}{method} | colors={'free' if method in ('advpatch', 'advreal', 'cap') else base_colors} | "
        f"size={patch_size} | area={frac} of box | resize={preprocess} | "
        f"epochs={epochs} | batch={batch_size} | lr={lr} | eot={not no_eot} | "
        f"tv={tv_weight} | nps={nps_weight}"
    )

    results_path = save_dir / "results.csv"
    with results_path.open("w", newline="") as f:
        columns = ["epoch", "loss", "det", "nps", "tv", "lr"]
        if method == "cap":
            columns += ["isp_loss", "isp_det", "isp_lr", "patch_steps", "isp_steps"]
        csv.writer(f).writerow(columns)
    best, best_epoch = float("inf"), 0
    total_iterations, patch_steps, isp_steps = 0, 0, 0
    for epoch in range(epochs):
        # Released AdvReal optimizer schedule; epoch is zero-based.
        if method == "advreal" and epoch % 100 == 90:
            for group in optimizer.param_groups:
                group["lr"] /= 1.1
        totals, seen = np.zeros(4), 0
        isp_totals, isp_seen = np.zeros(2), 0
        epoch_patch_steps, epoch_isp_steps = 0, 0
        pbar = TQDM(dataloader, desc=f"{method} {epoch + 1}/{epochs}")
        for im, targets, _, _ in pbar:
            im = im.to(device, non_blocking=True).float() / 255
            zero = torch.zeros((), device=device)
            isp_phase = False
            if method == "cap":
                loss, det, tv, isp_phase = cap_training_step(
                    module, isp_net, isp_params, (optimizer, isp_optimizer), confidence,
                    im, targets, total_iterations, frac, eot, tv_weight,
                )
                nps = zero
            else:
                patch = module()
                attacked = apply_patch(im, targets, module, frac, eot)
                # Retain each method's selected objective across native detectors.
                if detector is not None:
                    det = model.advpatch_loss(attacked) if method == "advpatch" else model.advreal_loss(attacked)
                else:
                    pred = model(attacked)[0]
                    det = advreal_2d_loss(pred) if method == "advreal" else attack_loss(pred)
                tv = torch.clamp(tv_weight * total_variation(patch), min=0.1) if tv_weight else zero
                nps = nps_weight * non_printability(patch) if nps_weight else zero
                loss = det + nps + tv
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if method in ("advpatch", "advreal"):
                    with torch.no_grad():
                        module.values.clamp_(0, 1)
            total_iterations += 1
            if isp_phase:
                isp_totals += np.array([loss.item(), det.item()]) * im.shape[0]
                isp_seen += im.shape[0]
                isp_steps += 1
                epoch_isp_steps += 1
            else:
                totals += np.array([loss.item(), det.item(), nps.item(), tv.item()]) * im.shape[0]
                seen += im.shape[0]
                patch_steps += 1
                epoch_patch_steps += 1
            d, n, t = totals[1:] / max(seen, 1)
            pbar.desc = f"{method} {epoch + 1}/{epochs} loss {totals[0] / max(seen, 1):.4f} det {d:.4f} nps {n:.4f} tv {t:.4f}"
            if method == "cap":
                pbar.desc += f" phase={'ISP' if isp_phase else 'patch'} ISP {isp_totals[0] / max(isp_seen, 1):.4f}"

        avg = totals[0] / max(seen, 1)
        with results_path.open("a", newline="") as f:
            row = [epoch + 1, *(totals / max(seen, 1)), optimizer.param_groups[0]["lr"]]
            if method == "cap":
                row += [*(isp_totals / max(isp_seen, 1)), isp_optimizer.param_groups[0]["lr"],
                        epoch_patch_steps, epoch_isp_steps]
            csv.writer(f).writerow(row)
        if scheduler is not None:
            scheduler.step(avg)
        if seen and avg < best:
            best, best_epoch = avg, epoch + 1
            save_patch(module.checkpoint(), module(), save_dir / "best.pt")
        # Save the optimizer along with the current patch so a long run has useful
        # recovery state. InfiniteDataLoader's prefetched batches are not serialized.
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            numpy_rng = np.random.get_state()
            state = {
                "completed_epoch": epoch + 1,
                "module": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_training_loss": float(best),
                "best_training_loss_epoch": best_epoch,
                "python_rng": random.getstate(),
                "numpy_rng": {"generator": numpy_rng[0], "keys": torch.tensor(numpy_rng[1].astype(np.int64)),
                              "position": numpy_rng[2], "has_gauss": numpy_rng[3], "cached_gaussian": numpy_rng[4]},
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "dataloader_generator": dataloader.generator.get_state(),
                "data_order_note": "Worker prefetch queues are not serialized; this is not a bitwise-resume guarantee.",
                "options": json.loads(json.dumps(options, default=str)),
            }
            if method == "cap":
                state["cap"] = {
                    "isp_parameters": isp_params.detach().cpu(),
                    "isp_optimizer": isp_optimizer.state_dict(),
                    "total_iterations": total_iterations,
                    "patch_optimizer_steps": patch_steps,
                    "isp_optimizer_steps": isp_steps,
                    "isp_weights": str(cap_isp_weights),
                }
            temporary = save_dir / "training_state.tmp"
            torch.save(state, temporary)
            temporary.replace(save_dir / "training_state.pt")

    save_patch(module.checkpoint(), module(), save_dir / "last.pt")
    summary = {
        "requested_epochs": epochs, "completed_epochs": epoch + 1,
        "optimizer_steps": (epoch + 1) * len(dataloader),
        "stop_reason": "epoch_limit",
        "checkpoint_selection": "minimum epoch-mean training total loss",
        "best_training_loss_epoch": best_epoch, "best_training_loss": float(best),
        "early_stopping": False,
        "detector": detector or "yolov5",
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20 if device.type == "cuda" else 0,
        "test_used_during_training": False,
    }
    if method == "advpatch":
        summary.update(
            method="AdvPatch (shared PAC protocol)",
            detector_confidence=("mean per-image peak pre-NMS ROI softmax person probability"
                                 if detector == "fasterrcnn" else "mean per-image peak objectness * person probability"),
            tv_weight=tv_weight, tv_floor=0.1 if tv_weight else None, nps_weight=nps_weight,
            optimizer="Adam AMSGrad", lr_schedule="ReduceLROnPlateau(mode=min, patience=50)",
        )
    if method == "cap":
        summary.update(
            method="CAP", patch_optimizer_steps=patch_steps, isp_optimizer_steps=isp_steps,
            checkpoint_selection="minimum epoch-mean patch-phase training loss (10*det + 20*TV)",
            patch_lr=lr, isp_lr=cap_isp_lr, lr_schedule="constant, as executed by the released code",
            source_repo=str(cap_repo), isp_weights=str(cap_isp_weights),
            detector_confidence="peak final box score" if detector == "fasterrcnn" else "peak raw objectness",
            eval_patch="raw RGB patch, without ISP or EOT",
        )
    (save_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info(f"Best total loss {best:.4f} at epoch {best_epoch}. Results saved to {colorstr('bold', save_dir)}")
    return save_dir


def parse_opt():
    """Parses command-line arguments for adversarial patch training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/inria.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="substitute model path")
    parser.add_argument(
        "--method",
        default="advpatch",
        choices=["advpatch", "capgen-t", "capgen-r", "capgen-p", "advreal", "cap"],
        help="patch family",
    )
    parser.add_argument("--detector", default=None,
                        choices=("yolov2", "yolov3", "yolov4", "yolov5s", "yolov5m", "fasterrcnn"),
                        help="native source for AdvPatch, AdvReal-2D or CAP; omitted keeps the existing YOLOv5 loader")
    parser.add_argument("--base-colors", default="bc1", choices=list(BASE_COLORS), help="paper base color set")
    parser.add_argument("--source", type=str, default=None, help="trained AdvPatch to recolor (capgen-p)")
    parser.add_argument("--patch-size", type=int, default=300, help="patch generation resolution (pixels)")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train image size (pixels)")
    parser.add_argument("--frac", type=float, default=0.25, help="patch area as a fraction of the box area")
    parser.add_argument("--tau", type=float, default=0.1, help="softmax temperature of Eq.3")
    parser.add_argument("--epochs", type=int, default=None, help="total epochs (CAP: 1000; others: 200)")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=None, help="Adam patch LR (CAP: 0.01; others: 0.03)")
    parser.add_argument("--cap-repo", default="/datacc05/huotianlin/code-hu/PAC/third_party/CAP_neurips2024")
    parser.add_argument("--cap-isp-weights", default="/datacc05/huotianlin/code-hu/PAC/third_party/weights/CAP_ISPNet_author.pth")
    parser.add_argument("--cap-isp-lr", type=float, default=0.0001, help="CAP ISP-parameter learning rate")
    parser.add_argument("--tv-weight", type=float, default=None, help="TV weight (CAP: 20; AdvPatch/AdvReal: 2.5; CAPGen: 0)")
    parser.add_argument(
        "--nps-weight", type=float, default=None, help="NPS weight (default: AdvPatch 0.01, CAPGen 0)"
    )
    parser.add_argument("--num-colors", type=int, default=3, help="number of base colors k")
    parser.add_argument("--no-eot", action="store_true", help="disable EOT augmentation (CAP: rotation only; ISP optimization is intrinsic)")
    preprocess_options = parser.add_mutually_exclusive_group()
    preprocess_options.add_argument(
        "--preprocess", choices=("advpatch", "letterbox", "squash"), default="letterbox",
        help="advpatch: pad with RGB 127 then PIL bilinear resize to a fixed square",
    )
    preprocess_options.add_argument(
        "--squash", dest="preprocess", action="store_const", const="squash", help="alias for --preprocess squash"
    )
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--seed", type=int, default=0, help="global training seed")
    parser.add_argument("--project", default=ROOT / "runs/patch", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Runs adversarial patch training with the parsed command-line options."""
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())
