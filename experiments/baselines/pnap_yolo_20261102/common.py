"""Paths, provenance, and project-local runtime for PNAP-YOLO."""
import hashlib
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
PROJECT = ROOT / "experiments/baselines/yolov5-advpatch-cap-20260911"
AUTHOR = RUN / "author"
MODELS = ("yolov2", "yolov3", "yolov4", "yolov5s", "yolov5m", "fasterrcnn")
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SD_PATH = RUN / "weights/sd15"


def runtime():
    for key, suffix in {
        "HF_HOME": "hf", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
        "MPLCONFIGDIR": "mpl", "TORCH_HOME": "torch", "YOLO_CONFIG_DIR": "yolo",
        "YOLOV5_CONFIG_DIR": "yolo", "NUMBA_CACHE_DIR": "numba",
    }.items():
        path = RUN / "runtime" / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    temporary = ROOT / ".tmp_pnap02"
    temporary.mkdir(exist_ok=True)
    os.environ.update({"TMPDIR": str(temporary), "PYTHONDONTWRITEBYTECODE": "1",
                       "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                       "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg",
                       "WANDB_MODE": "disabled", "YOLOv5_AUTOINSTALL": "false",
                       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
    sys.path.insert(0, str(PROJECT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_author():
    for name, digest in read(RUN / "evidence/source_provenance.json")["files"].items():
        if sha(AUTHOR / name) != digest:
            raise RuntimeError(f"Author source changed: {name}")
