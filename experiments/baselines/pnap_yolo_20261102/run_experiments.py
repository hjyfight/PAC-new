"""Prepare, train, evaluate and report PNAP under the common PAC protocol."""
import argparse
import csv
import importlib.metadata
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from common import (AUTHOR, MODELS, PROJECT, ROOT, RUN, SD_PATH, read, runtime,
                    sha, verify_author, write)

CODE_FILES = ("eval_transfer.py", "detector_backends.py", "utils/patch.py",
              "utils/dataloaders.py", "utils/general.py", "utils/metrics.py", "models/experimental.py")
PAC_PYTHON = Path("/datacc05/huotianlin/miniconda3/envs/pac/bin/python")


def prepare():
    verify_author()
    target = RUN / "protocol.json"
    if target.exists():
        raise RuntimeError("Protocol already frozen; use preflight/train/evaluate to resume.")
    previous = read(ROOT / "experiments/baselines/advpatch_cap_completion_20260911/manifest.json")
    keys = ("data_yaml", "train_images", "test_images", "test_targets", "batch_size", "patch_size",
            "patch_area_fraction", "placement", "imgsz", "preprocess", "pixel_pipeline", "conf_thres",
            "nms_iou", "max_det", "asr_conf", "asr_iou", "asr_definition", "detectors")
    config = {k: previous[k] for k in keys}
    config.update(method="PNAP-YOLO (PAC protocol adaptation)", seed=1176426343, epochs=500,
        epochs_reference="author/train_patch.py: n_epochs=500; user confirmed source default on 2026-11-02",
        learning_rate=0.001, latent_shape=[1, 4, 64, 64], native_patch_size=512,
        init_denoising_steps=50, optimization_timesteps=[601, 401, 201, 1], guidance_scale=7,
        prompt="1Pomeranian, portrait, extremely detailed, masterpiece, high quality, 8K",
        negative_prompt="low quality, bad anatomy, wrong anatomy, deformed iris, deformed pupils, "
                        "distorted, disfigured, watermark, text, signature",
        optimizer="Adam(amsgrad=True), betas=(0.9,0.999)",
        scheduler="ReduceLROnPlateau(mode=min, factor=0.1, patience=50), epoch mean total loss",
        objective="released detector-specific loss + 0.01*NPS + max(2.5*TV, 0.1)",
        checkpoint_selection="minimum epoch mean training total loss; final reported separately",
        author_commit=read(RUN / "evidence/source_provenance.json")["commit"],
        extensions=["yolov4: reconstructed native obj/class + released v5 extractor; missing author config",
                    "yolov5m: released v5s extractor with common v5m weights",
                    "fasterrcnn: released FRCNN mean-person extractor; no author training config"],
        source_code_sha256={n: sha(PROJECT / n) for n in CODE_FILES},
        test_used_during_training=False, models=list(MODELS))
    # Freeze both splits, independently of the old reports.
    import yaml
    dataset = yaml.safe_load(Path(config["data_yaml"]).read_text())
    records = {}
    for split, key, count in (("train", "train", 614), ("test", "val", 288)):
        directory = Path(dataset["path"]) / dataset[key]
        entries = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            label = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
            lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
            if any(len(row) != 5 or float(row[0]) != 0 for row in lines):
                raise RuntimeError(f"Unexpected person label: {label}")
            entries.append(dict(image=str(path), image_sha256=sha(path), label=str(label),
                                label_sha256=sha(label), targets=len(lines)))
        if len(entries) != count:
            raise RuntimeError(f"Unexpected {split} image count: {len(entries)}")
        records[split] = entries
    if sum(r["targets"] for r in records["test"]) != 589:
        raise RuntimeError("Unexpected test GT count")
    if {r["image_sha256"] for r in records["train"]} & {r["image_sha256"] for r in records["test"]}:
        raise RuntimeError("Train/test overlap")
    write(RUN / "dataset_audit.json", records)
    config["dataset_audit_sha256"] = sha(RUN / "dataset_audit.json")
    write(target, config)
    print("Frozen PNAP protocol: six sources, 500 epochs/source; no training started.")


def verify(config, weights=True):
    verify_author()
    if {name: sha(PROJECT / name) for name in CODE_FILES} != config["source_code_sha256"]:
        raise RuntimeError("Shared evaluation/training source changed")
    if sha(RUN / "dataset_audit.json") != config["dataset_audit_sha256"]:
        raise RuntimeError("Dataset audit changed")
    for rows in read(RUN / "dataset_audit.json").values():
        for row in rows:
            for kind in ("image", "label"):
                if sha(row[kind]) != row[kind + "_sha256"]:
                    raise RuntimeError(f"Dataset changed: {row[kind]}")
    for item in config["detectors"].values():
        if sha(item["weights"]) != item["sha256"]:
            raise RuntimeError(f"Detector weights changed: {item['weights']}")
    if weights:
        for name, digest in read(RUN / "evidence/sd15_provenance.json")["files"].items():
            if sha(SD_PATH / name) != digest:
                raise RuntimeError(f"SD weight changed: {name}")


def preflight():
    config = read(RUN / "protocol.json")
    versions = {}
    errors = []
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors",
                 "huggingface-hub", "numpy", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {name}")
    try:
        verify(config)
    except (OSError, RuntimeError) as exc:
        errors.append(str(exc))
    result = {"passed": not errors, "versions": versions, "errors": errors, "formal_results": False}
    write(RUN / "preflight.json", result)
    print(result)
    if errors:
        raise SystemExit(1)


def save_patch(path, patch, metadata):
    import torch
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    patch = patch.detach().cpu()
    torch.save({"patch": patch, **metadata}, path)
    Image.fromarray((patch.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype("uint8")).save(
        path.with_suffix(".png"))


def train(model_name, smoke_steps=0):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from detector_backends import DetectorBackend
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset
    from utils.patch import apply_patch
    from pnap import PNAP, DetectionLoss, make_eot, regularizers

    config = read(RUN / "protocol.json")
    verify(config)
    output = RUN / ("smoke" if smoke_steps else "sources") / model_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train_complete.json").exists():
        print("Already complete:", output)
        return
    if smoke_steps and (output / "training_state.pt").exists():
        raise RuntimeError("Smoke output already exists; inspect it before rerunning.")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pn = PNAP(config, device)
    pn.initialize()
    backend = DetectorBackend(model_name, device=device, image_size=640, fused=False,
        conf_threshold=0.4, nms_iou=0.45, max_det=300, cache_dir=RUN / "runtime/torchhub")
    det_loss = DetectionLoss(backend)
    nps, tv = regularizers(device)
    eot = make_eot()
    data = check_dataset(config["data_yaml"], autodownload=False)
    _, dataset = create_dataloader(data["train"], 640, 8, 32, workers=0,
                                    preprocess=config["preprocess"], seed=seed)
    assert len(dataset) == config["train_images"]
    optimizer = torch.optim.Adam([pn.latents], lr=config["learning_rate"], amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)
    best, start_epoch, step = math.inf, 0, 0
    grad_norm, detection_grad_norm = 0.0, 0.0
    state_path = output / "training_state.pt"
    adapter_hashes = {p.name: sha(p) for p in (RUN / "pnap.py", RUN / "run_experiments.py", RUN / "common.py")}
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["adapter_sha256"] != adapter_hashes or state["protocol_sha256"] != sha(RUN / "protocol.json"):
            raise RuntimeError("Refusing to resume with changed training code/protocol")
        with torch.no_grad(): pn.latents.copy_(state["latents"].to(device))
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        best, start_epoch, step = state["best"], state["epoch"] + 1, state["steps"]
        grad_norm, detection_grad_norm = state["grad_norm"], state["detection_grad_norm"]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state(state["cuda_rng"], device)
        pn.generator.set_state(state["generator_rng"])
    started = time.monotonic()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, 1 if smoke_steps else config["epochs"]):
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0,
            collate_fn=dataset.collate_fn, generator=torch.Generator().manual_seed(seed + epoch))
        sums, count = np.zeros(4), 0
        for images, targets, *_ in loader:
            images = images.to(device).float() / 255
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            native = pn.render()
            patch = F.interpolate(native[None], (300, 300), mode="bilinear", align_corners=False)[0]
            patched = apply_patch(images, targets, patch, frac=0.25, eot=eot)
            detection = det_loss(patched)
            nps_term, tv_term = 0.01 * nps(native), 2.5 * tv(native)
            loss = detection + nps_term + tv_term.clamp_min(0.1)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
            if step == 0:
                probe = torch.autograd.grad(detection, pn.latents, retain_graph=True)[0]
                detection_grad_norm = float(probe.norm())
                if not torch.isfinite(probe).all() or detection_grad_norm == 0:
                    raise RuntimeError("Detector loss does not reach diffusion latent")
                del probe
            loss.backward()
            gradient = pn.latents.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Missing/nonfinite diffusion latent gradient")
            grad_norm = float(gradient.norm())
            if step == 0 and grad_norm == 0: raise RuntimeError("Zero initial latent gradient")
            optimizer.step()
            values = [float(x.detach()) for x in (loss, detection, nps_term, tv_term)]
            sums += values; count += 1; step += 1
            if smoke_steps or count == 1 or count % 10 == 0:
                print(dict(epoch=epoch + 1, batch=count, step=step, loss=values[0],
                           detection=values[1], latent_grad_norm=grad_norm,
                           seconds=round(time.monotonic() - started, 2)), flush=True)
            del loss, native, patch, patched, detection, nps_term, tv_term
            if smoke_steps and count >= smoke_steps: break
        means = (sums / count).tolist()
        scheduler.step(means[0])
        improved = means[0] < best
        best = min(best, means[0])
        with torch.no_grad():
            patch = F.interpolate(pn.render()[None], (300, 300), mode="bilinear", align_corners=False)[0]
        metadata = dict(method=config["method"], optimization_source=model_name, epoch=epoch + 1,
                        steps=step, smoke=bool(smoke_steps), latent=pn.latents.detach().cpu(),
                        protocol_sha256=sha(RUN / "protocol.json"), adapter_sha256=adapter_hashes)
        save_patch(output / "patches/final.pt", patch, metadata)
        if improved: save_patch(output / "patches/best.pt", patch, metadata)
        state = dict(epoch=epoch, steps=step, best=best, latents=pn.latents.detach().cpu(),
            grad_norm=grad_norm, detection_grad_norm=detection_grad_norm,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device), generator_rng=pn.generator.get_state(),
            protocol_sha256=sha(RUN / "protocol.json"), adapter_sha256=adapter_hashes)
        temporary = output / "training_state.tmp"
        torch.save(state, temporary); temporary.replace(state_path)
        with (output / "history.jsonl").open("a") as log:
            import json
            log.write(json.dumps(dict(epoch=epoch + 1, steps=step, loss=means, best=best,
                                      seconds=time.monotonic() - started)) + "\n")
        write(output / "status.json", dict(epoch=epoch + 1, steps=step, best=best, smoke=bool(smoke_steps)))
    for module in (pn.pipeline.unet, pn.pipeline.vae, backend.detector):
        if any(p.grad is not None for p in module.parameters()):
            raise RuntimeError("Frozen model unexpectedly accumulated parameter gradients")
    result = dict(passed=True, epochs=epoch + 1, steps=step, smoke=bool(smoke_steps),
                  latent_grad_norm=grad_norm, seconds=time.monotonic() - started,
                  detection_latent_grad_norm=detection_grad_norm,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(), formal_results=False)
    write(output / ("smoke_complete.json" if smoke_steps else "train_complete.json"), result)


def evaluate(model_name, smoke=False):
    config = read(RUN / "protocol.json")
    verify(config, weights=False)
    source = RUN / ("smoke" if smoke else "sources") / model_name
    complete = read(source / ("smoke_complete.json" if smoke else "train_complete.json"))
    if not complete["passed"] or (not smoke and complete["epochs"] != config["epochs"]):
        raise RuntimeError("Training not complete")
    items = [dict(id="clean", label="Clean", path=None, optimization_source=None)]
    for version in ("best", "final"):
        path = source / "patches" / (version + ".pt")
        items.append(dict(id="pnap_" + version, label="PNAP-YOLO " + version, path=str(path),
                          sha256=sha(path), optimization_source=model_name))
    write(source / "manifest.json", {**config, "optimization_source": model_name, "patches": items,
                                    "smoke": smoke, "evaluation_batch_size": 8})
    for target in ((model_name,) if smoke else MODELS):
        logs = source / "logs"; logs.mkdir(exist_ok=True)
        command = [str(PAC_PYTHON), "-B", str(PROJECT / "eval_transfer.py"),
                   "--manifest", str(source / "manifest.json"), "--detector", target,
                   "--batch-size", "8", "--workers", "0"]
        with (logs / f"eval_{target}.log").open("a") as log:
            subprocess.run(command, cwd=PROJECT, env=os.environ, stdout=log, stderr=subprocess.STDOUT, check=True)
        clean = read(source / "eval" / target / "clean/metrics.json")
        old = read(ROOT / "experiments/NCQ/ncq_random3_20261025/sources" / target / "eval" / target / "clean/metrics.json")
        for key in ("map50", "conditional_asr_numerator", "conditional_asr_denominator", "images", "targets"):
            if clean[key] != old[key]: raise RuntimeError(f"Clean replay differs: {target} {key}")
        print("EVALUATED", model_name, target, "smoke" if smoke else "formal", flush=True)
    write(source / "evaluation_complete.json", dict(passed=True, smoke=smoke,
          evaluations=len(items) * (1 if smoke else len(MODELS))))


def summarize():
    from statistics import mean
    rows = []
    for source in MODELS:
        for target in MODELS:
            for version in ("best", "final"):
                path = RUN / "sources" / source / "eval" / target / f"pnap_{version}/metrics.json"
                item = read(path) if path.exists() else {}
                rows.append(dict(method="PNAP-YOLO", source=source, target=target, checkpoint=version,
                    relation="whitebox" if source == target else "blackbox",
                    map50_pct=item.get("map50_pct"), conditional_asr=item.get("conditional_asr"),
                    status="complete" if item else "pending", metrics_file=str(path)))
    table = RUN / "tables/comparison.csv"; table.parent.mkdir(exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(RUN / "results_status.json", dict(method="PNAP-YOLO", expected_attack_evaluations=72,
        completed_attack_evaluations=sum(r["status"] == "complete" for r in rows),
        formal_results_complete=all(r["status"] == "complete" for r in rows)))
    previous = ROOT / "experiments/baselines/tsea_nap_20260912/tables/comparison_best.csv"
    with previous.open(newline="") as stream:
        imported = list(csv.DictReader(stream))
    best = [r for r in rows if r["checkpoint"] == "best" and r["status"] == "complete"]
    wb = [r["map50_pct"] for r in best if r["relation"] == "whitebox"]
    bb = [r["map50_pct"] for r in best if r["relation"] == "blackbox"]
    imported.append(dict(method="PNAP-YOLO", wb_map50=mean(wb) if len(wb) == 6 else None,
                         bb_map50=mean(bb) if len(bb) == 30 else None, wb_cells=len(wb), bb_cells=len(bb)))
    with (table.parent / "comparison_best_with_pnap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(imported[0]))
        writer.writeheader(); writer.writerows(imported)
    write(RUN / "evidence/comparison_import.json", dict(path=str(previous), sha256=sha(previous),
          note="Previous shared-protocol rows imported unchanged; PNAP means require all 6 WB / 30 BB cells."))
    print("Saved", table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "train", "evaluate", "summarize"))
    parser.add_argument("--model", choices=MODELS, default="yolov5s")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.smoke_steps <= 77: parser.error("--smoke-steps must be between 0 and 77")
    if args.smoke and args.stage != "evaluate": parser.error("--smoke is for evaluate; use train --smoke-steps N")
    if args.smoke_steps and args.stage != "train": parser.error("--smoke-steps is for train")
    runtime()
    if args.stage == "train": train(args.model, args.smoke_steps)
    elif args.stage == "evaluate": evaluate(args.model, args.smoke)
    else: globals()[args.stage]()


if __name__ == "__main__":
    main()
