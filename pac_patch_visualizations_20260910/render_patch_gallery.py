"""Export the evaluated patch tensors as RGB figures and a standalone gallery."""

import base64
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


OUT = Path(__file__).resolve().parent
PROJECT = Path('/datacc05/huotianlin/code-hu/PAC-yolo/yolov5-advreal-recolor-20260910')
RUN = PROJECT / 'runs/advreal_recolor_20260910'
CAP_RUN = Path('/datacc05/huotianlin/code-hu/PAC-yolo/yolov5-cap-rfatex-20260909/runs/cap_rfatex_whitebox_20260909')
SOURCES = ['yolov2', 'yolov3', 'yolov4', 'yolov5s', 'yolov5m', 'fasterrcnn']
SOURCE_LABELS = dict(zip(SOURCES, ['YOLOv2', 'YOLOv3', 'YOLOv4', 'YOLOv5s', 'YOLOv5m', 'Faster R-CNN']))
LABELS = {'advpatch': 'AdvPatch', 'p1': 'PAC-P1', 'p2': 'PAC-P2', 'cap': 'CAP (best)',
          'advreal2d': 'AdvReal', 'advreal_r1': 'PAC-R1 (AdvReal)', 'advreal_r2': 'PAC-R2 (AdvReal)'}
METHODS = list(LABELS)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text(draw, xy, value, size=18, bold=False, fill='#263345'):
    draw.text(xy, value, font=font(size, bold), fill=fill)


def save_preview(im, name, width):
    preview = im.copy()
    preview.thumbnail((width, 3000), Image.Resampling.LANCZOS)
    preview.save(OUT / name, quality=85, subsampling=0)


def draw_card(draw, canvas, entry, x, y):
    text(draw, (x, y), entry['label'], 22, True)
    text(draw, (x, y + 31), entry['description'], 15, fill='#657389')
    canvas.paste(entry['image'], (x, y + 58))
    draw.rectangle((x - 1, y + 57, x + 300, y + 358), outline='#cad2dd')
    text(draw, (x, y + 372), f"WB AP50: {entry['whitebox_ap50']:.2f}%", 18)


def render_figures(entries, colors):
    lookup = {(e['source'], e['method']): e for e in entries}
    canvas = Image.new('RGB', (1344, 994), 'white')
    draw = ImageDraw.Draw(canvas)
    text(draw, (28, 20), 'Evaluated patches: YOLOv5s source', 31, True)
    text(draw, (28, 65), '300 x 300 textures | original RGB | training-selected best checkpoints', 18)
    grid = [['advpatch', 'p1', 'p2', 'cap'], ['advreal2d', 'advreal_r1', 'advreal_r2']]
    for row, methods in enumerate(grid):
        for col, method in enumerate(methods):
            draw_card(draw, canvas, lookup['yolov5s', method], 28 + col * 328, 112 + row * 426)
    x, y = 1012, 538
    text(draw, (x, y), 'Fixed palettes', 22, True)
    for row, key in enumerate(('bc1', 'bc2')):
        sy = y + 55 + row * 116
        text(draw, (x, sy), key.upper(), 19, True)
        for col, color in enumerate(colors[key]):
            sx = x + 102 * col
            draw.rectangle((sx, sy + 32, sx + 90, sy + 73), fill=tuple(map(int, color)), outline='#c9d1db')
            text(draw, (sx, sy + 81), ','.join(str(int(v)) for v in color), 11)
    text(draw, (x, y + 308), 'Fixed RGB weights', 18)
    text(draw, (x, y + 338), 'No recolor optimization', 18)
    text(draw, (28, 961), 'Source budgets: AdvPatch 200 epochs; AdvReal 800; CAP 1000. Recoloring adds 0 epochs.', 16)
    canvas.save(OUT / 'yolov5s_all_methods.png')
    save_preview(canvas, 'yolov5s_all_methods_preview.jpg', 1200)

    canvas = Image.new('RGB', (1520, 2450), 'white')
    draw = ImageDraw.Draw(canvas)
    text(draw, (28, 20), 'CAP, AdvReal and AdvReal recolorings: all six sources', 29, True)
    text(draw, (28, 64), 'Each row uses the indicated training detector. Every texture is shown at its native 300 x 300 pixels.', 17)
    columns = ['cap', 'advreal2d', 'advreal_r1', 'advreal_r2']
    for col, method in enumerate(columns):
        text(draw, (194 + 328 * col, 109), LABELS[method], 22, True)
    for row, source in enumerate(SOURCES):
        sy = 157 + 372 * row
        text(draw, (20, sy + 119), SOURCE_LABELS[source], 19, True)
        for col, method in enumerate(columns):
            e = lookup[source, method]
            sx = 194 + col * 328
            canvas.paste(e['image'], (sx, sy))
            draw.rectangle((sx - 1, sy - 1, sx + 300, sy + 300), outline='#cad2dd')
            text(draw, (sx, sy + 311), f"WB AP50: {e['whitebox_ap50']:.2f}%", 18)
            text(draw, (sx, sy + 340), e['description'], 14, fill='#657389')
    text(draw, (28, 2404), 'CAP / Faster R-CNN: best is epoch 3 and nearly uniform; this is the evaluated checkpoint.', 17)
    canvas.save(OUT / 'all_sources_cap_advreal_recolors.png')
    save_preview(canvas, 'all_sources_preview.jpg', 1000)


def render_html(entries, colors):
    records = []
    for e in entries:
        record = {k: v for k, v in e.items() if k not in ('image', 'tensor')}
        record['data_url'] = 'data:image/png;base64,' + base64.b64encode((OUT / e['png']).read_bytes()).decode()
        records.append(record)
    payload = json.dumps({'patches': records, 'colors': colors, 'sources': SOURCE_LABELS}, ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>对抗补丁可视化 · best</title>
<style>
:root{font-family:system-ui,"Noto Sans CJK SC",sans-serif;color:#17243a;background:#f4f6f9}body{margin:0}main{max-width:1420px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 12px}.intro{color:#506079;line-height:1.8}.toolbar{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:22px 0}select,button,a.button{padding:10px 14px;border:1px solid #c9d2df;background:white;border-radius:7px;color:#17243a;font:inherit}a.button{text-decoration:none}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:20px}.card{background:white;border:1px solid #dce2eb;border-radius:10px;padding:16px}.card h3{font-size:19px;margin:0 0 6px}.meta{color:#52647c;font-size:14px;line-height:1.7}.texture{width:100%;aspect-ratio:1;object-fit:contain;image-rendering:pixelated;display:block;margin:12px 0;background:white;cursor:zoom-in}.score{font-size:15px;color:#263f60}section h2{font-size:21px;margin:28px 0 14px}.swatches{display:flex;gap:8px;flex-wrap:wrap}.swatch{display:inline-block;width:38px;height:22px;vertical-align:middle;border:1px solid #aaa;margin-right:5px}.notes{font-size:14px;line-height:1.8;margin-top:25px;background:white;padding:18px;border-radius:9px}.small{font-size:12px;color:#657389;overflow-wrap:anywhere}dialog{border:none;border-radius:10px;padding:18px;max-width:90vw;max-height:94vh}dialog::backdrop{background:#142032bb}dialog img{display:block;max-width:85vw;max-height:74vh;image-rendering:pixelated;margin:14px auto}dialog header{display:flex;justify-content:space-between;align-items:center;gap:25px}footer{margin-top:24px;font-size:13px;color:#657389}
</style><main>
<h1>AdvPatch、CAP、AdvReal 与 PAC 换色补丁</h1>
<div class="intro">展示实际评估的 best 补丁。默认比较共同 YOLOv5s 来源；可切换训练来源，点击补丁放大或下载原尺寸 PNG。</div>
<div class="toolbar"><label>训练来源：<select id="source"><option value="yolov5s">YOLOv5s：全部七种方法</option><option value="all">全部六个来源</option><option value="yolov2">YOLOv2</option><option value="yolov3">YOLOv3</option><option value="yolov4">YOLOv4</option><option value="yolov5m">YOLOv5m</option><option value="fasterrcnn">Faster R-CNN</option></select></label></div>
<div id="gallery"></div>
<div class="notes"><strong>配色与来源说明</strong><div id="palettes"></div>
<p>PAC-P1 / P2 来自 AdvPatch；PAC-R1 / R2 (AdvReal) 来自 AdvReal。这两个 R 版本与论文中的随机 PAC-R1 / R2 对照不同。</p>
<p>AdvPatch / PAC-P 的同协议参考产物仅有 YOLOv5s 训练来源。CAP、AdvReal 及其两种换色版本覆盖全部六个来源。</p>
<p>源训练预算：AdvPatch 200 轮，AdvReal 800 轮，CAP 1000 轮；换色不增加训练。各方法的 best 按训练损失选择，含义并非测试攻击效果最优。CAP 的 Faster R-CNN best 为第 3 轮，因此外观接近纯色。</p>
<p>白盒 mAP50 越低攻击越强。图片保留原始 RGB，未作亮度、对比度或颜色归一化调整；PNG 采用标准 8 位量化。</p></div>
<footer>本文件内嵌全部补丁图片，可离线打开。补丁哈希和原始路径见各图片的“来源记录”。</footer>
</main><dialog id="zoom"><header><strong id="zoom-title"></strong><button id="close">关闭</button></header><img id="zoom-img"><a class="button" id="download">下载原尺寸 PNG</a></dialog>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), gallery=document.getElementById('gallery'), source=document.getElementById('source'), dialog=document.getElementById('zoom');
function show(p){document.getElementById('zoom-title').textContent=p.label+' · '+data.sources[p.source];document.getElementById('zoom-img').src=p.data_url;const link=document.getElementById('download');link.href=p.data_url;link.download=p.png.split('/').pop();dialog.showModal();}
function render(){gallery.replaceChildren();for(const [key,label] of Object.entries(data.sources)){if(source.value!=='all'&&source.value!==key)continue;const section=document.createElement('section'),title=document.createElement('h2'),grid=document.createElement('div');title.textContent=label+' 训练来源';grid.className='grid';for(const p of data.patches.filter(p=>p.source===key)){const card=document.createElement('article');card.className='card';const h=document.createElement('h3');h.textContent=p.label;const meta=document.createElement('div');meta.className='meta';meta.textContent=p.description;const img=document.createElement('img');img.className='texture';img.src=p.data_url;img.alt=p.label+' '+label;img.onclick=()=>show(p);const score=document.createElement('div');score.className='score';score.textContent='白盒 mAP50：'+p.whitebox_ap50.toFixed(2)+'%';const details=document.createElement('details'),summary=document.createElement('summary'),provenance=document.createElement('p');summary.textContent='来源记录';provenance.className='small';provenance.textContent=p.checkpoint+' · SHA256 '+p.checkpoint_sha256;details.append(summary,provenance);card.append(h,meta,img,score,details);grid.append(card);}section.append(title,grid);gallery.append(section);}}
for(const [key,colors] of Object.entries(data.colors)){const row=document.createElement('p');row.className='swatches';row.append(key.toUpperCase()+'：');for(const color of colors){const span=document.createElement('span'),swatch=document.createElement('span');swatch.className='swatch';swatch.style.backgroundColor='rgb('+color.join(',')+')';span.append(swatch,color.join(', '));row.append(span);}document.getElementById('palettes').append(row);}
source.onchange=render;document.getElementById('close').onclick=()=>dialog.close();dialog.onclick=e=>{if(e.target===dialog)dialog.close();};render();
</script></html>'''
    (OUT / 'gallery.html').write_text(html.replace('__PAYLOAD__', payload))


def main():
    torch.set_num_threads(2)
    manifest = read(RUN / 'manifest.json')
    utility_path = PROJECT / 'utils/patch.py'
    assert sha(utility_path) == manifest['source_file_sha256']['utils/patch.py']
    spec = importlib.util.spec_from_file_location('evaluated_patch_utils', utility_path)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    (OUT / 'patches').mkdir(exist_ok=True)
    items = [item for item in manifest['patches'] if item['path'] is not None]
    cap_matrix = read(CAP_RUN / 'matrix_manifest.json')
    cap_epochs = {}
    for source in SOURCES:
        source_run = Path(cap_matrix['source_runs'][source])
        source_manifest = read(source_run / 'manifest.json')
        item = dict(next(p for p in source_manifest['patches'] if p['id'] == 'cap'))
        item['variant'] = 'cap'
        items.append(item)
        training = read(source_run / 'train_complete.json')
        assert item['sha256'] == training['checkpoints']['best']['sha256']
        cap_epochs[source] = training['best_training_loss_epoch']
    rec_metrics = {(r['patch_id'], r['target']): r for r in read(RUN / 'matrix_results.json')['rows']}
    with (RUN / 'tables/whitebox_map50.csv').open(newline='') as stream:
        cap_wb = next(row for row in csv.DictReader(stream) if row['method'] == 'cap_best')
    entries = []
    for item in items:
        source = item['optimization_source']
        method = item['variant'].removeprefix('reference_')
        path = Path(item['path'])
        assert sha(path) == item['sha256']
        tensor = utility.load_patch(path, device='cpu').float().contiguous()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert 'patch' in checkpoint and torch.equal(tensor, checkpoint['patch'].cpu())
        assert tuple(tensor.shape) == (3, 300, 300) and torch.isfinite(tensor).all()
        assert float(tensor.min()) >= 0 and float(tensor.max()) <= 1
        raw = tensor.permute(1, 2, 0).numpy().astype(np.float64)
        rgb = np.rint(raw * 255).astype(np.uint8)
        image = Image.fromarray(rgb)
        relative = f'patches/{source}_{method}.png'
        image.save(OUT / relative)
        assert np.array_equal(np.asarray(Image.open(OUT / relative)), rgb)
        error = float(np.abs(rgb.astype(np.float64) / 255 - raw).max())
        assert error <= 0.5 / 255 + 1e-12
        if method == 'cap':
            metric = float(cap_wb[source])
            description = f'best epoch {cap_epochs[source]} / 1000'
        else:
            metric_record = rec_metrics[item['id'], source]
            assert metric_record['patch_sha256'] == item['sha256']
            metric = metric_record['map50_pct']
            if method == 'advpatch':
                description = 'Original RGB | 200 epochs'
            elif method in ('p1', 'p2'):
                description = f"AdvPatch -> Bc{method[-1]} | 0 extra epochs"
            elif method == 'advreal2d':
                description = f"best epoch {item['source_best_epoch']} / 800"
            else:
                description = f"AdvReal -> Bc{method[-1]} | 0 extra epochs"
        entries.append({'source': source, 'method': method, 'label': LABELS[method],
                        'description': description, 'checkpoint': str(path),
                        'checkpoint_sha256': item['sha256'], 'png': relative, 'png_sha256': sha(OUT / relative),
                        'shape': list(tensor.shape), 'min': float(tensor.min()), 'max': float(tensor.max()),
                        'std': float(tensor.std()), 'max_png_quantization_error': error,
                        'whitebox_ap50': metric, 'tensor': tensor, 'image': image})
    entries.sort(key=lambda e: (SOURCES.index(e['source']), METHODS.index(e['method'])))
    assert len(entries) == 27
    lookup = {(e['source'], e['method']): e for e in entries}
    formula_errors = []
    for source in SOURCES:
        pairs = [('advreal2d', 'advreal_r1', 'bc1'), ('advreal2d', 'advreal_r2', 'bc2')]
        if source == 'yolov5s':
            pairs += [('advpatch', 'p1', 'bc1'), ('advpatch', 'p2', 'bc2')]
        for original, recolored, palette in pairs:
            raw = lookup[source, original]['tensor'].numpy().astype(np.float64)
            expected = np.einsum('khw,kc->chw', raw, np.asarray(manifest['base_colors_255'][palette]) / 255)
            actual = lookup[source, recolored]['tensor'].numpy().astype(np.float64)
            error = float(np.abs(expected - actual).max())
            assert error < 2e-7
            formula_errors.append({'source': source, 'method': recolored, 'max_error': error})
    render_figures(entries, manifest['base_colors_255'])
    render_html(entries, manifest['base_colors_255'])
    records = [{k: v for k, v in e.items() if k not in ('image', 'tensor')} for e in entries]
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
             'patch_count': len(records), 'checkpoint_version': 'training-selected best',
             'source_manifest': str(RUN / 'manifest.json'), 'source_manifest_sha256': sha(RUN / 'manifest.json'),
             'loader': str(utility_path), 'loader_sha256': sha(utility_path), 'recolor_checks': formula_errors,
             'display': 'Original RGB; uint8 rounding only; no color/brightness/contrast adjustment.',
             'patches': records}
    (OUT / 'manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    readme = '''# 已评估补丁可视化（best）

- `gallery.html`：可离线打开，按训练来源筛选，点击放大并下载原始 PNG。
- `yolov5s_all_methods.png`：共同 YOLOv5s 来源的全部七种方法。第一行是 AdvPatch、PAC-P1、PAC-P2、CAP；第二行是 AdvReal、PAC-R1 (AdvReal)、PAC-R2 (AdvReal) 及色板。
- `all_sources_cap_advreal_recolors.png`：六个来源，列依次为 CAP、AdvReal、PAC-R1 (AdvReal)、PAC-R2 (AdvReal)。
- `patches/`：27 张单独的 300×300 RGB PNG。
- `manifest.json`：原始 checkpoint 路径与哈希、白盒结果、颜色范围及数值核验。

原始 PAC-P1/P2 来自 AdvPatch；PAC-R1/R2 (AdvReal) 来自 AdvReal，不是论文中的随机 PAC-R 控制组。
CAP 和 AdvReal 使用既有训练损失选定的 best。CAP / Faster R-CNN 的 best 为第 3 轮，接近纯色；此图保留该实际评估版本。
AdvPatch、AdvReal、CAP 的源训练预算分别为 200、800、1000 轮。换色不重新训练，展示未应用 EOT 或 ISP。
图片来自评估使用的精确 RGB 张量。所有单图按 round(RGB×255) 导出，未归一化、调亮或调对比度。
总览 PNG 中补丁按原生 300×300 粘贴；预览 JPEG 仅供屏幕浏览，论文和后续分析请使用 PNG。
'''
    (OUT / 'README.md').write_text(readme)
    with zipfile.ZipFile(OUT / 'patch_visualizations_best.zip', 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip' and '__pycache__' not in path.parts:
                bundle.write(path, path.relative_to(OUT))
    print(json.dumps({'passed': True, 'patch_count': len(entries), 'max_quantization_error': max(e['max_png_quantization_error'] for e in entries),
                      'output': str(OUT), 'figures': ['yolov5s_all_methods.png', 'all_sources_cap_advreal_recolors.png'],
                      'cap_best_epochs': cap_epochs}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--preview-base64':
        image = Image.open(OUT / sys.argv[2]).convert('RGB')
        image.thumbnail((1000, 1650), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=72)
        print('data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode())
    else:
        main()
