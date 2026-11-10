"""Evaluate selected PAC s000 and AdvReal-2D with shared preprocessing/AP code."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from detector_backends import DetectorBackend, MODEL_NAMES
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, scale_boxes, xywh2xyxy
from utils.metrics import ap_per_class, box_iou, process_batch
from utils.patch import apply_patch, load_patch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def detected_gt(predictions, labels, conf=0.5, iou_threshold=0.5):
    """One-to-one matches, using the same ordering as YOLO's AP matcher."""
    hit = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    pred = predictions[predictions[:, 4] >= conf]
    if not len(pred) or not len(labels):
        return hit.cpu().tolist()
    iou = box_iou(labels[:, 1:], pred[:, :4])
    pairs = torch.where((iou >= iou_threshold) & (labels[:, 0:1] == pred[:, 5]))
    if pairs[0].numel():
        matches = torch.cat((torch.stack(pairs, 1), iou[pairs[0], pairs[1]][:, None]), 1).cpu().numpy()
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        hit[torch.as_tensor(matches[:, 0].astype(int), device=labels.device)] = True
    return hit.cpu().tolist()


@torch.inference_mode()
def evaluate(model, loader, item, config, output_dir, weight_hash):
    started = time.monotonic()
    device = model.device
    patch = None
    patch_hash = None
    if item['id'] != 'clean':
        source = item['path']
        if source != 'gray':
            patch_hash = sha256(source)
            if item.get('sha256') and patch_hash != item['sha256']:
                raise RuntimeError(f"Patch hash changed: {source}")
        patch = load_patch(source, device=device, patch_size=config['patch_size'])
    stats, per_image = [], []
    seen, targets_seen = 0, 0
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    for batch_i, (images, targets, paths, shapes) in enumerate(loader):
        images = images.to(device, non_blocking=True).float() / 255
        targets = targets.to(device)
        assert images.shape[-2:] == (config['imgsz'], config['imgsz'])
        if patch is not None:
            images = apply_patch(images, targets, patch, frac=config['patch_area_fraction'], eot=None)
        predictions = model.predict(images)
        _, _, height, width = images.shape
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        for si, pred in enumerate(predictions):
            labels = targets[targets[:, 0] == si, 1:]
            seen += 1
            targets_seen += len(labels)
            pred_native = pred.clone()
            scale_boxes(images[si].shape[1:], pred_native[:, :4], shapes[si][0], shapes[si][1])
            boxes = xywh2xyxy(labels[:, 1:5])
            scale_boxes(images[si].shape[1:], boxes, shapes[si][0], shapes[si][1])
            labels_native = torch.cat((labels[:, :1], boxes), 1)
            correct = process_batch(pred_native, labels_native, iouv) if len(pred) and len(labels) else torch.zeros((len(pred), 10), dtype=torch.bool, device=device)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 0].cpu()))
            per_image.append({'image': str(paths[si]), 'detected_gt': detected_gt(
                pred_native, labels_native, conf=config['asr_conf'], iou_threshold=config['asr_iou'])})
        if batch_i % 5 == 0:
            print(f"[{model.name}/{item['id']}] {seen}/{len(loader.dataset)} images", flush=True)
    assert seen == config['test_images'], (seen, config['test_images'])
    assert targets_seen == config['test_targets'], (targets_seen, config['test_targets'])
    combined = [torch.cat(values).numpy() for values in zip(*stats)]
    precision = recall = map50 = map50_95 = 0.0
    if combined[0].any():
        _, _, p, r, _, ap, classes = ap_per_class(*combined, plot=False, names={0: 'person'})
        assert classes.tolist() == [0], classes
        precision, recall = float(p.mean()), float(r.mean())
        map50, map50_95 = float(ap[:, 0].mean()), float(ap.mean())
    clean_record = output_dir.parent / 'clean/detected_gt.json'
    if item['id'] == 'clean':
        clean = per_image
    else:
        clean = json.loads(clean_record.read_text())
    eligible = disappeared = 0
    for before, after in zip(clean, per_image, strict=True):
        assert before['image'] == after['image']
        a, b = np.asarray(before['detected_gt'], bool), np.asarray(after['detected_gt'], bool)
        assert a.shape == b.shape
        eligible += int(a.sum())
        disappeared += int((a & ~b).sum())
    relation = 'control' if item.get('optimization_source') is None else ('whitebox' if model.name == item['optimization_source'] else 'blackbox_transfer')
    record = {
        'detector': model.name, 'patch_id': item['id'], 'patch_label': item['label'],
        'relation': relation, 'optimization_source': item.get('optimization_source'),
        'precision': precision, 'recall': recall, 'map50': map50, 'map50_pct': map50 * 100,
        'map50_95': map50_95, 'conditional_asr': disappeared / eligible if eligible else None,
        'conditional_asr_numerator': disappeared, 'conditional_asr_denominator': eligible,
        'asr_conf': config['asr_conf'], 'asr_iou': config['asr_iou'],
        'images': seen, 'targets': targets_seen, 'patch': item.get('path'), 'patch_sha256': patch_hash,
        'weights': str(model.weight_path), 'weights_sha256': weight_hash, 'detector_source': model.source,
        'preprocess': config['preprocess'], 'imgsz': config['imgsz'], 'rect': False,
        'patch_size': config['patch_size'], 'patch_area_fraction': config['patch_area_fraction'],
        'eot': False, 'conf_thres': config['conf_thres'], 'nms_iou': config['nms_iou'],
        'max_det': config['max_det'], 'batch_size': loader.batch_size, 'seconds': time.monotonic() - started,
    }
    write_json(output_dir / 'detected_gt.json', per_image)
    write_json(output_dir / 'metrics.json', record)
    print('RESULT ' + json.dumps(record, ensure_ascii=False), flush=True)
    del stats, combined, patch
    gc.collect()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--detector', choices=MODEL_NAMES, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patches', default='reference')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.manual_seed(0)
    np.random.seed(0)
    config = json.loads(Path(args.manifest).read_text())
    run_dir = Path(args.manifest).resolve().parent
    requested = None if args.patches == 'reference' else set(args.patches.split(','))
    items = [item for item in config['patches']
             if (item['id'] != 'advreal2d' if requested is None else item['id'] in requested)]
    if not items:
        raise ValueError('No patches selected')
    if any(item['id'] == 'advreal2d' for item in items):
        complete = json.loads((run_dir / 'train_complete.json').read_text())
        assert complete['epochs'] == config['epochs'] and complete['passed']
    model = DetectorBackend(args.detector, device=device, image_size=config['imgsz'], fused=True,
                            conf_threshold=config['conf_thres'], nms_iou=config['nms_iou'],
                            max_det=config['max_det'], cache_dir=run_dir / 'torchhub')
    weight_hash = sha256(model.weight_path)
    assert weight_hash == config['detectors'][args.detector]['sha256'], args.detector
    data = check_dataset(config['data_yaml'])
    loader, _ = create_dataloader(data['val'], config['imgsz'], args.batch_size, 32,
                                 workers=args.workers, rect=False, pad=0, shuffle=False,
                                 preprocess=config['preprocess'])
    with torch.inference_mode():
        model.predict(torch.zeros(1, 3, config['imgsz'], config['imgsz'], device=device))
    for item in items:
        evaluate(model, loader, item, config, run_dir / 'eval' / args.detector / item['id'], weight_hash)


if __name__ == '__main__':
    main()
