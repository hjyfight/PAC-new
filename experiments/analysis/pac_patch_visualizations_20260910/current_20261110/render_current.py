"""Render the exact best patches referenced by the current comparison tables."""
import base64
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCES = ['yolov2', 'yolov3', 'yolov4', 'yolov5s', 'yolov5m', 'fasterrcnn']
SOURCE_LABELS = dict(zip(SOURCES, ['YOLOv2', 'YOLOv3', 'YOLOv4', 'YOLOv5s', 'YOLOv5m', 'Faster R-CNN']))
LABELS = {
    'advpatch': 'AdvPatch', 'p1': 'PAC-P1', 'p2': 'PAC-P2',
    'tsea': 'T-SEA', 'tsea_bc1': 'T-SEA + Bc1', 'tsea_bc2': 'T-SEA + Bc2',
    'advreal': 'AdvReal-2D', 'advreal_bc1': 'AdvReal-2D + Bc1', 'advreal_bc2': 'AdvReal-2D + Bc2',
    'cap': 'CAP', 'cap_bc1': 'CAP + Bc1', 'cap_bc2': 'CAP + Bc2',
    'nap': 'NAP', 'pnap': 'PNAP', 'gray': 'Gray',
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


legacy = module('previous_gallery', OUT.parent / 'render_patch_gallery.py')
utility = module('evaluated_patch_utils', ROOT / 'experiments/ours/yolov5-advreal-recolor-20260910/utils/patch.py')
read, sha, font, text = legacy.read, legacy.sha, legacy.font, legacy.text


def contact_sheet(entries, path, title, subtitle, columns=3):
    tile_w, tile_h, margin, header = 330, 374, 24, 110
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new('RGB', (margin * 2 + tile_w * columns, header + rows * tile_h + 20), 'white')
    draw = ImageDraw.Draw(canvas)
    text(draw, (margin, 20), title, 27, True)
    text(draw, (margin, 63), subtitle, 16)
    for i, entry in enumerate(entries):
        x = margin + (i % columns) * tile_w
        y = header + (i // columns) * tile_h
        text(draw, (x, y), entry['tile_label'], 20, True)
        canvas.paste(entry['image'], (x, y + 33))
        text(draw, (x, y + 341), 'White-box AP50: %.2f%%' % entry['ap50'], 16)
    canvas.save(path)
    return canvas


def make_payload(entries):
    for size, quality in [(200, 80), (192, 75), (176, 72), (160, 70), (144, 65)]:
        records = []
        for entry in entries:
            preview = entry['image'].resize((size, size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            preview.save(buffer, format='WEBP', quality=quality, method=6)
            records.append({
                'source': entry['source'], 'method': entry['method'], 'label': entry['label'],
                'ap50': round(entry['ap50'], 2),
                'image': 'data:image/webp;base64,' + base64.b64encode(buffer.getvalue()).decode(),
            })
        payload = json.dumps({'sources': SOURCE_LABELS, 'patches': records, 'preview_size': size},
                             ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
        if len(payload.encode()) < 850_000:
            return payload, size, quality
    raise RuntimeError('Inline image payload exceeds the size limit')


def main():
    torch.set_num_threads(2)
    index_path = ROOT / 'experiments/baselines/tsea_nap_20260912/matrix_results.json'
    matrix = read(index_path)
    entries = []
    (OUT / 'patches').mkdir(exist_ok=True)
    for source in SOURCES:
        for method, label in LABELS.items():
            if method == 'pnap':
                metric_path = ROOT / 'experiments/baselines/pnap_yolo_20261102/sources' / source / 'eval' / source / 'pnap_best/metrics.json'
            elif method == 'gray':
                metric_path = Path(matrix['gray_controls'][source]['metrics_path'])
            else:
                metric_path = Path(matrix['comparison_best'][method]['cells'][source][source]['metrics_path'])
            metric = read(metric_path)
            assert (metric['detector'], metric['images'], metric['targets']) == (source, 288, 589)
            if method == 'gray':
                checkpoint = None
                checkpoint_sha = None
                tensor = utility.load_patch('gray', device='cpu', patch_size=300)
            else:
                assert metric['optimization_source'] == source
                checkpoint = Path(metric['patch'])
                checkpoint_sha = sha(checkpoint)
                assert checkpoint_sha == metric['patch_sha256'], checkpoint
                tensor = utility.load_patch(checkpoint, device='cpu').float().contiguous()
            assert tuple(tensor.shape) == (3, 300, 300)
            assert torch.isfinite(tensor).all() and 0 <= tensor.min() <= tensor.max() <= 1
            raw = tensor.permute(1, 2, 0).numpy().astype(np.float64)
            rgb = np.rint(raw * 255).astype(np.uint8)
            im = Image.fromarray(rgb)
            png = Path('patches') / (source + '_' + method + '.png')
            im.save(OUT / png)
            assert np.array_equal(np.asarray(Image.open(OUT / png)), rgb)
            error = float(np.max(np.abs(rgb.astype(np.float64) / 255 - raw)))
            assert error <= 0.5 / 255 + 1e-12
            entries.append({
                'source': source, 'method': method, 'label': label, 'tile_label': label,
                'selection': 'control' if method == 'gray' else 'training-loss-selected best',
                'checkpoint': str(checkpoint) if checkpoint else None,
                'checkpoint_sha256': checkpoint_sha, 'metrics_path': str(metric_path),
                'metrics_sha256': sha(metric_path), 'png': str(png), 'png_sha256': sha(OUT / png),
                'ap50': metric['map50_pct'], 'asr_pct': metric['conditional_asr'] * 100,
                'max_png_quantization_error': error, 'image': im,
            })
    assert len(entries) == 90
    for source in SOURCES:
        items = [e for e in entries if e['source'] == source]
        sheet = contact_sheet(items, OUT / (source + '_all_methods_best.png'),
                              SOURCE_LABELS[source] + ': evaluated best patches',
                              '300 x 300 RGB | Same training source | Lower AP50 = stronger attack')
        preview = sheet.copy()
        preview.thumbnail((900, 2200), Image.Resampling.LANCZOS)
        preview.save(OUT / (source + '_preview.jpg'), quality=88, subsampling=0)
    advpatch = [dict(e, tile_label=SOURCE_LABELS[e['source']]) for e in entries if e['method'] == 'advpatch']
    contact_sheet(advpatch, OUT / 'advpatch_all_sources_best.png', 'AdvPatch: six training sources',
                  'Exact evaluated best patches | 300 x 300 RGB | 200 training epochs')
    main_methods = ['advpatch', 'p1', 'p2', 'tsea', 'advreal', 'cap', 'nap', 'pnap', 'gray']
    yolo_items = {e['method']: e for e in entries if e['source'] == 'yolov5s'}
    contact_sheet([yolo_items[m] for m in main_methods], OUT / 'yolov5s_main_methods_best.png',
                  'YOLOv5s: main methods and PAC-P',
                  'Exact evaluated best patches | 300 x 300 RGB | Different training budgets')
    payload, preview_size, quality = make_payload(entries)
    template = (OUT / 'gallery-template.html').read_text()
    assert template.count('__PATCH_PAYLOAD__') == 1
    fragment = template.replace('__PATCH_PAYLOAD__', payload)
    assert len(fragment.encode()) < 1_000_000
    (OUT / 'current-patches.html').write_text(fragment)
    manifest = {
        'passed': True, 'patch_count': len(entries), 'attack_patches': 84, 'gray_controls': 6,
        'comparison_index': str(index_path), 'comparison_index_sha256': sha(index_path),
        'display': 'PNG: exact RGB rounded to uint8; no contrast, brightness, or color normalization.',
        'inline_preview': {'size': preview_size, 'webp_quality': quality, 'bytes': len(fragment.encode())},
        'patches': [{k: v for k, v in e.items() if k != 'image'} for e in entries],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'patches'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
