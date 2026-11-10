"""Measure one aligned RGB image pair; this is not a dataset evaluation runner."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Clean image")
    parser.add_argument("--candidate", type=Path, required=True, help="Patched image")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Identical half-open pixel crop for both images")
    args = parser.parse_args()
    try:
        import numpy as np
        import PIL
        from PIL import Image
        import skimage
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as exc:
        parser.error(f"Missing dependency: {exc}. Install numpy, pillow, scikit-image in a dedicated environment.")

    def read_rgb(path):
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0

    reference = read_rgb(args.reference)
    candidate = read_rgb(args.candidate)
    if reference.shape != candidate.shape:
        parser.error("Image shapes differ; use the same rendering pipeline, not automatic resizing.")
    if args.roi:
        x0, y0, x1, y1 = args.roi
        height, width = reference.shape[:2]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            parser.error("ROI must lie inside both images and have positive area.")
        reference = reference[y0:y1, x0:x1]
        candidate = candidate[y0:y1, x0:x1]
    if min(reference.shape[:2]) < 11:
        parser.error("SSIM requires at least 11x11 pixels under this fixed protocol; do not silently change the window.")
    mse = float(np.mean((reference - candidate) ** 2))
    psnr = "inf" if mse == 0 else float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    ssim = float(structural_similarity(
        reference, candidate, data_range=1.0, channel_axis=-1,
        win_size=11, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, K1=0.01, K2=0.03,
    ))
    result = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "roi_xyxy": args.roi,
        "evaluated_shape_hwc": list(reference.shape),
        "mse": mse, "psnr_db": psnr, "ssim": ssim,
        "protocol": {
            "input": "PIL-decoded RGB / 255; no gamma linearization or further resizing",
            "data_range": 1.0, "ssim_window": 11, "gaussian_weights": True,
            "sigma": 1.5, "use_sample_covariance": False, "K1": 0.01, "K2": 0.03,
            "channels": "PSNR over all RGB entries; SSIM arithmetic mean of channel scores",
            "scope": "single image pair; no dataset aggregation or attack evaluation",
        },
        "versions": {"numpy": np.__version__, "pillow": PIL.__version__, "skimage": skimage.__version__},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
