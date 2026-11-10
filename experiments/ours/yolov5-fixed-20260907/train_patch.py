# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train PAC / CAPGen adversarial patches against a YOLOv5 detector.

AdvPatch, CAPGen-T, and AdvReal run gradient descent. CAPGen-P and CAPGen-R
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
    ColorProbPatch,
    apply_patch,
    attack_loss,
    load_patch,
    max_iou_loss,
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


def run(
    data=ROOT / "data/inria.yaml",
    weights=ROOT / "yolov5s.pt",
    method="advpatch",
    base_colors="bc1",
    source=None,
    patch_size=300,
    imgsz=640,
    frac=0.25,
    tau=0.1,
    epochs=200,
    batch_size=8,
    lr=0.03,
    tv_weight=None,
    nps_weight=None,
    num_colors=3,
    no_eot=False,
    squash=False,
    device="",
    workers=8,
    seed=0,
    project=ROOT / "runs/patch",
    name="exp",
    exist_ok=False,
):
    """Builds or trains one adversarial patch and saves it under `project/name`."""
    if tv_weight is None:
        tv_weight = 2.5 if method in ("advpatch", "advreal") else 0.0
    if nps_weight is None:
        nps_weight = 0.01 if method in ("advpatch", "advreal") else 0.0
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
        AdvPatch(patch_size, use_logits=method == "advreal").to(device)
        if method in ("advpatch", "advreal")
        else ColorProbPatch(colors[:num_colors], patch_size, tau).to(device)
    )

    # CAPGen-R: the untrained random color-probability matrix. Sharing --seed with a
    # capgen-t run gives both methods the same initial matrix, as the paper requires.
    if method == "capgen-r":
        save_patch(module.checkpoint(), module(), save_dir / "best.pt")
        LOGGER.info(f"CAPGen-R (untrained, seed {seed}, {base_colors.upper()}) -> {save_dir / 'best.pt'}")
        return save_dir

    model = attempt_load(weights, device=device, fuse=False).eval().requires_grad_(False)
    stride = int(model.stride.max())
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
        squash=squash,
    )[0]

    eot = None
    if not no_eot:
        if method == "capgen-t":
            eot = EOT().to(device)
        elif method == "advpatch":
            eot = EOT(
                brightness=(-0.1, 0.1), noise=(-0.1, 0.1), scale=(1, 1),
                smooth=7, median=True, value_range=(1e-6, 0.99999),
            ).to(device)
        else:
            eot = AdvRealEOT().to(device)
    optimizer = Adam(module.parameters(), lr=lr, amsgrad=method == "advpatch")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=50) if method == "advpatch" else None
    LOGGER.info(
        f"{colorstr('patch: ')}{method} | colors={'free' if method in ('advpatch', 'advreal') else base_colors} | "
        f"size={patch_size} | area={frac} of box | resize={'squash' if squash else 'letterbox'} | "
        f"epochs={epochs} | batch={batch_size} | lr={lr} | eot={not no_eot} | "
        f"tv={tv_weight} | nps={nps_weight}"
    )

    results_path = save_dir / "results.csv"
    with results_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "det", "nps", "tv", "lr"])
    best = float("inf")
    for epoch in range(epochs):
        totals, seen = np.zeros(4), 0
        pbar = TQDM(dataloader, desc=f"{method} {epoch + 1}/{epochs}")
        for im, targets, _, _ in pbar:
            im = im.to(device, non_blocking=True).float() / 255
            patch = module()
            pred = model(apply_patch(im, targets, module, frac, eot))[0]
            # AdvReal scores each labeled box through its max-IoU match and carries its own TV term;
            # the CAPGen/AdvPatch families take one peak score per image.
            if method == "advreal":
                det = max_iou_loss(pred, targets, im.shape[2:])
            else:
                det = attack_loss(pred)
            # Reference AdvPatch regularizers. PAC-T defaults to detection loss alone;
            # the PAC text does not specify explicit TV/NPS terms or their coefficients.
            zero = torch.zeros((), device=device)
            tv = torch.clamp(tv_weight * total_variation(patch), min=0.1) if tv_weight else zero
            nps = nps_weight * non_printability(patch) if nps_weight else zero
            loss = det + nps + tv
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if method == "advpatch":
                with torch.no_grad():
                    module.values.clamp_(0, 1)
            totals += np.array([loss.item(), det.item(), nps.item(), tv.item()]) * im.shape[0]
            seen += im.shape[0]
            d, n, t = totals[1:] / seen
            pbar.desc = f"{method} {epoch + 1}/{epochs} loss {totals[0] / seen:.4f} det {d:.4f} nps {n:.4f} tv {t:.4f}"

        avg = totals[0] / max(seen, 1)
        with results_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, *(totals / max(seen, 1)), optimizer.param_groups[0]["lr"]])
        if scheduler is not None:
            scheduler.step(avg)
        if avg < best:
            best = avg
            save_patch(module.checkpoint(), module(), save_dir / "best.pt")

    save_patch(module.checkpoint(), module(), save_dir / "last.pt")
    LOGGER.info(f"Best total loss {best:.4f}. Results saved to {colorstr('bold', save_dir)}")
    return save_dir


def parse_opt():
    """Parses command-line arguments for adversarial patch training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/inria.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="substitute model path")
    parser.add_argument(
        "--method",
        default="advpatch",
        choices=["advpatch", "capgen-t", "capgen-r", "capgen-p", "advreal"],
        help="patch family",
    )
    parser.add_argument("--base-colors", default="bc1", choices=list(BASE_COLORS), help="paper base color set")
    parser.add_argument("--source", type=str, default=None, help="trained AdvPatch to recolor (capgen-p)")
    parser.add_argument("--patch-size", type=int, default=300, help="patch generation resolution (pixels)")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train image size (pixels)")
    parser.add_argument("--frac", type=float, default=0.25, help="patch area as a fraction of the box area")
    parser.add_argument("--tau", type=float, default=0.1, help="softmax temperature of Eq.3")
    parser.add_argument("--epochs", type=int, default=200, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam learning rate")
    parser.add_argument("--tv-weight", type=float, default=None, help="TV weight (default: AdvPatch 2.5, CAPGen 0)")
    parser.add_argument(
        "--nps-weight", type=float, default=None, help="NPS weight (default: AdvPatch 0.01, CAPGen 0)"
    )
    parser.add_argument("--num-colors", type=int, default=3, help="number of base colors k")
    parser.add_argument("--no-eot", action="store_true", help="disable EOT augmentation")
    parser.add_argument("--squash", action="store_true", help="resize images to a square instead of letterboxing")
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
