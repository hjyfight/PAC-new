# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the INRIA Person dataset to YOLO format and write its dataset YAML.

The original distribution ships PASCAL-style annotation text files; this converts the
`PASperson` boxes to normalized YOLO labels and links the images into the layout the
dataloader expects. Class ids follow COCO so that COCO-pretrained weights can be
evaluated directly (person = 0).

Usage:
    $ python prepare_inria.py --source /path/to/INRIAPerson --out datasets/inria
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

BBOX_RE = re.compile(r'Bounding box for object \d+ "PASperson".*?\((\d+), *(\d+)\) *- *\((\d+), *(\d+)\)')


def convert_split(source, out, split):
    """Links images and writes YOLO labels for one INRIA split, returning the image count."""
    sub = "Train" if split == "train" else "Test"
    img_dir = next(d for d in (source / sub / "pos", source / sub / "images") if d.is_dir())
    ann_dir = source / sub / "annotations"
    out_img, out_lbl = out / "images" / split, out / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path in sorted(img_dir.iterdir()):
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg") or not ann_path.is_file():
            continue
        boxes = BBOX_RE.findall(ann_path.read_text(encoding="latin-1"))
        if not boxes:
            continue
        w, h = Image.open(img_path).size
        lines = []
        for x1, y1, x2, y2 in ((int(a) for a in b) for b in boxes):
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (out_lbl / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
        link = out_img / img_path.name
        link.unlink(missing_ok=True)
        link.symlink_to(img_path.resolve())
        n += 1
    return n


def main(opt):
    """Converts both INRIA splits and writes the dataset YAML."""
    source, out = Path(opt.source).resolve(), Path(opt.out).resolve()
    counts = {split: convert_split(source, out, split) for split in ("train", "val")}
    names = yaml.safe_load((ROOT / "data/coco.yaml").read_text())["names"]  # COCO ids, person = 0
    yaml_path = ROOT / "data/inria.yaml"
    yaml_path.write_text(
        f"# INRIA Person, person boxes only, COCO class ids\npath: {out}\ntrain: images/train\nval: images/val\n\n"
        + yaml.safe_dump({"names": names}, sort_keys=False)
    )
    print(f"train: {counts['train']} images | val: {counts['val']} images -> {out}\nwrote {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="path to the original INRIAPerson directory")
    parser.add_argument("--out", default=ROOT / "datasets/inria", help="output dataset root")
    main(parser.parse_args())
