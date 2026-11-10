"""Five preregistered column-sum-255 palettes, sharing the original NCQ evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
ORIGINAL = ROOT / 'experiments/NCQ/ncq_random3_20261025'
spec = importlib.util.spec_from_file_location('original_ncq_runner', ORIGINAL / 'run_experiments.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
# Reuse detector construction, code checks and evaluate() without modifying prior artifacts.
base.RUN = RUN
for key, relative in {
    'XDG_CACHE_HOME': 'runtime/cache', 'XDG_CONFIG_HOME': 'runtime/config',
    'MPLCONFIGDIR': 'runtime/mplconfig', 'TORCH_HOME': 'runtime/torch',
    'YOLO_CONFIG_DIR': 'runtime/yolo_config', 'YOLOV5_CONFIG_DIR': 'runtime/yolo_config',
    'NUMBA_CACHE_DIR': 'runtime/numba',
}.items():
    path = RUN / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
# Seed the existing local font before concurrent check_dataset() calls.
font = RUN / 'runtime/yolo_config/Arial.ttf'
if not font.exists():
    shutil.copy2(ORIGINAL / 'runtime/yolo_config/Arial.ttf', font)
base.temporary_directory = ROOT / '.tmp_ncq255'
base.temporary_directory.mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(base.temporary_directory)
MODELS, SEEDS = base.MODELS, base.SEEDS
read, write, sha = base.read, base.write, base.sha


def prepare():
    import numpy as np
    from PIL import Image
    import torch
    from utils.patch import load_patch, recolor

    if (RUN / 'protocol.json').exists():
        raise RuntimeError('Prepared protocol already exists; run or summarize instead.')
    torch.set_num_threads(2)
    original = read(ORIGINAL / 'protocol.json')
    base.check_code(original)
    # Uniform weak compositions of 255 into three ordered integer parts.
    # Two distinct separator positions among 257 give all C(257,2) triples once.
    palettes = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        columns = []
        for channel in range(3):
            a, b = sorted(rng.choice(257, size=2, replace=False).tolist())
            columns.append([a, b - a - 1, 256 - b])
        colors = np.array(columns, dtype=np.int64).T
        assert colors.shape == (3, 3) and (colors >= 0).all() and (colors <= 255).all()
        assert np.array_equal(colors.sum(axis=0), [255, 255, 255])
        palettes[f'sum255_s{seed}'] = colors.tolist()
    assert len({tuple(np.array(p).flatten()) for p in palettes.values()}) == 5
    # Freeze all palettes before creating patches or reading attack outcomes.
    write(RUN / 'palettes.json', palettes)
    dataset_audit = read(ORIGINAL / 'dataset_audit.json')
    assert sha(ORIGINAL / 'dataset_audit.json') == original['dataset_audit_sha256']
    for record in dataset_audit:
        assert sha(record['image']) == record['image_sha256']
        assert sha(record['label']) == record['label_sha256']
    write(RUN / 'dataset_audit.json', dataset_audit)
    audit, baseline_audit = [], []
    for model in MODELS:
        source = original['source_items'][model]
        assert sha(source['path']) == source['sha256']
        raw = load_patch(source['path'])
        assert raw.shape == (3, 300, 300) and torch.isfinite(raw).all()
        assert raw.min() >= 0 and raw.max() <= 1
        old = read(ORIGINAL / 'sources' / model / 'manifest.json')
        for item in old['patches']:
            record = ORIGINAL / 'sources' / model / 'eval' / model / item['id'] / 'metrics.json'
            baseline_audit.append({'model': model, 'variant': item['id'], 'path': str(record),
                                   'sha256': sha(record)})
        items = [{'id': 'clean', 'label': 'Clean', 'path': None, 'optimization_source': None}]
        directory = RUN / 'sources' / model
        for seed in SEEDS:
            colors = palettes[f'sum255_s{seed}']
            permutation = np.random.default_rng(10000 + seed).permutation(300 * 300)
            shuffled = raw.reshape(3, -1)[:, permutation].reshape_as(raw)
            retained_output = None
            for structure, donor in [('retained', raw), ('shuffled', shuffled)]:
                item_id = f'{structure}_sum255_s{seed}'
                unbounded64 = np.einsum('khw,kc->chw', donor.numpy().astype(np.float64),
                                       np.array(colors, dtype=np.float64) / 255)
                unbounded32 = (donor.permute(1, 2, 0) @ torch.tensor(
                    np.array(colors, dtype=np.float32) / 255)).permute(2, 0, 1)
                result = recolor(donor, colors).contiguous()
                assert (unbounded64 >= 0).all() and (unbounded64 <= 1).all()
                assert unbounded32.min() >= 0 and unbounded32.max() <= 1
                assert torch.equal(unbounded32, result), 'clamp must change no RGB component'
                if structure == 'retained':
                    retained_output = result
                else:
                    assert torch.equal(result, retained_output.reshape(3, -1)[:, permutation].reshape_as(result))
                path = directory / 'patches' / f'{item_id}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {'palette_seed': seed, 'structure': structure,
                            'permutation_seed': 10000 + seed if structure == 'shuffled' else None}
                torch.save({'patch': result, 'patch_size': 300, 'source_path': source['path'],
                            'source_sha256': source['sha256'], 'base_colors_255': colors,
                            'extra_optimization_steps': 0, **metadata}, path)
                assert torch.equal(load_patch(path), result)
                Image.fromarray((result.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
                    path.with_suffix('.png'))
                items.append({'id': item_id, 'label': item_id, 'path': str(path),
                              'sha256': sha(path), 'optimization_source': model, **metadata})
                audit.append({'model': model, 'id': item_id, 'column_sums': [255, 255, 255],
                              'clipped_fraction_float64': 0.0, 'clipped_fraction_float32': 0.0,
                              'clamp_changed_components': 0, 'unbounded_min': float(unbounded64.min()),
                              'unbounded_max': float(unbounded64.max()),
                              'paired_permutation_exact': structure == 'shuffled'})
        config = {**old, 'patches': items, 'palette_seeds': list(SEEDS), 'evaluation_batch_size': 8}
        write(directory / 'manifest.json', config)
    write(RUN / 'generation_audit.json', {'passed': True, 'patches': audit})
    write(RUN / 'baseline_audit.json', baseline_audit)
    protocol = {**original, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'parent_protocol': str(ORIGINAL / 'protocol.json'),
                'parent_protocol_sha256': sha(ORIGINAL / 'protocol.json'),
                'random_palette_sampling': 'Per seed 0..4, independently per RGB output channel: '
                    'sort default_rng(seed).choice(257,2,replace=False) as a,b; '
                    'parts=(a,b-a-1,256-b). Uniform over all ordered nonnegative integer triples summing to 255.',
                'constraint_axis': 'Sum over the THREE BASE COLORS for EACH output R/G/B channel; NOT each color RGB sum.',
                'structure_control': 'Same original donor and full RGB-vector permutation seeds 10000..10004 as parent.',
                'clipping_proof': '0<=donor_k<=1 and sum_k colors[k,c]=255 imply 0<=sum_k donor_k*colors[k,c]/255<=1.',
                'expected_evaluations': 66, 'new_attack_evaluations': 60, 'clean_replays': 6,
                'baseline_use': 'Prior metrics are referenced, not counted as new evaluations; SHA256 frozen in baseline_audit.json.',
                'baseline_audit_sha256': sha(RUN / 'baseline_audit.json'),
                'dataset_audit_sha256': sha(RUN / 'dataset_audit.json'),
                'script_sha256_at_prepare': sha(__file__),
                'shared_runner_sha256': sha(ORIGINAL / 'run_experiments.py')}
    write(RUN / 'protocol.json', protocol)
    print(json.dumps({'prepared': True, 'evaluations': 66, 'palettes': palettes, 'all_clipping_zero': True}))


def run(gpus):
    config = read(RUN / 'protocol.json')
    base.check_code(config)
    assert sha(ORIGINAL / 'run_experiments.py') == config['shared_runner_sha256']
    def lane(index, gpu):
        for model in MODELS[index::len(gpus)]:
            path = RUN / 'logs' / f'{model}.log'
            path.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--model', model]
            print(f'START GPU {gpu}: {model}', flush=True)
            with path.open('a') as log:
                result = subprocess.run(command, cwd=base.PROJECT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{model} exited {result.returncode}: {path}')
            print(f'DONE GPU {gpu}: {model}', flush=True)
    write(RUN / 'launch.json', {'gpus': gpus, 'pid': os.getpid(), 'command': sys.argv,
          'started_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__)})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs = [pool.submit(lane, i, gpu) for i, gpu in enumerate(gpus)]
        for job in jobs:
            job.result()


def export(name, rows):
    with (RUN / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    import numpy as np
    protocol = read(RUN / 'protocol.json')
    base.check_code(protocol)
    assert sha(ORIGINAL / 'protocol.json') == protocol['parent_protocol_sha256']
    assert sha(ORIGINAL / 'run_experiments.py') == protocol['shared_runner_sha256']
    assert sha(RUN / 'baseline_audit.json') == protocol['baseline_audit_sha256']
    for item in read(RUN / 'baseline_audit.json'):
        assert sha(item['path']) == item['sha256']
    assert sha(RUN / 'dataset_audit.json') == protocol['dataset_audit_sha256']
    for item in read(RUN / 'dataset_audit.json'):
        assert sha(item['image']) == item['image_sha256']
        assert sha(item['label']) == item['label_sha256']
    generation = read(RUN / 'generation_audit.json')
    assert generation['passed'] and len(generation['patches']) == 60
    assert all(p['clamp_changed_components'] == 0 for p in generation['patches'])
    rows, seed_rows, values, asrs, replays = [], [], {}, {}, []
    prior = read(ORIGINAL / 'summary.json')
    for model in MODELS:
        directory = RUN / 'sources' / model
        assert read(directory / 'complete.json') == {'passed': True, 'evaluations': 11}
        config = read(directory / 'manifest.json')
        values[model], asrs[model] = {}, {}
        for item in config['patches']:
            path = directory / 'eval' / model / item['id'] / 'metrics.json'
            record = read(path)
            assert record['images'] == 288 and record['targets'] == 589
            assert record['detector'] == model and record['patch_id'] == item['id']
            assert record['weights_sha256'] == config['detectors'][model]['sha256']
            assert record['eot'] is False and record['rect'] is False
            for key in ('patch_size','patch_area_fraction','imgsz','preprocess','batch_size',
                        'conf_thres','nms_iou','max_det','asr_conf','asr_iou'):
                assert record[key] == config[key], (model, key)
            if item['path'] is not None:
                assert record['patch_sha256'] == sha(item['path']) == item['sha256']
            else:
                old_clean = read(ORIGINAL / 'sources' / model / 'eval' / model / 'clean/metrics.json')
                for key in ('map50_pct','conditional_asr_numerator','conditional_asr_denominator'):
                    assert record[key] == old_clean[key], (model, key)
                assert read(path.with_name('detected_gt.json')) == read(
                    ORIGINAL / 'sources' / model / 'eval' / model / 'clean/detected_gt.json')
                replays.append({'model': model, 'map50_difference': record['map50_pct']-old_clean['map50_pct']})
            assert np.isfinite(record['map50_pct']) and 0 <= record['map50_pct'] <= 100
            values[model][item['id']] = record['map50_pct']
            asrs[model][item['id']] = record['conditional_asr'] * 100
            rows.append({'model': model, 'variant': item['id'], 'map50_pct': record['map50_pct'],
                         'asr_pct': record['conditional_asr'] * 100,
                         'asr_numerator': record['conditional_asr_numerator'],
                         'asr_denominator': record['conditional_asr_denominator'], 'metrics_path': str(path)})
    assert len(rows) == protocol['expected_evaluations'] == 66
    for data in (values, asrs):
        data['mean6'] = {key: float(np.mean([data[m][key] for m in MODELS])) for key in data[MODELS[0]]}
    comparisons = []
    for model in (*MODELS, 'mean6'):
        old = prior['values'][model]
        retained = np.array([values[model][f'retained_sum255_s{s}'] for s in SEEDS])
        shuffled = np.array([values[model][f'shuffled_sum255_s{s}'] for s in SEEDS])
        old_random = np.array([old[f'random_s{s}'] for s in SEEDS])
        old_shuffle = np.array([old[f'shuffle_bc1_s{s}'] for s in SEEDS])
        attack = np.array([asrs[model][f'retained_sum255_s{s}'] for s in SEEDS])
        shuffled_attack = np.array([asrs[model][f'shuffled_sum255_s{s}'] for s in SEEDS])
        comparisons.append({'model': model, 'bc1_map50': old['p1'], 'bc2_map50': old['p2'],
            'old_random_map50_mean': float(old_random.mean()), 'old_random_map50_sd': float(old_random.std(ddof=1)),
            'sum255_map50_mean': float(retained.mean()), 'sum255_map50_sd': float(retained.std(ddof=1)),
            'shuffled_bc1_map50_mean': float(old_shuffle.mean()),
            'shuffled_sum255_map50_mean': float(shuffled.mean()), 'shuffled_sum255_map50_sd': float(shuffled.std(ddof=1)),
            'sum255_minus_bc1': float(retained.mean()-old['p1']),
            'paired_structure_effect': float((shuffled-retained).mean()),
            'paired_structure_effect_sd': float((shuffled-retained).std(ddof=1)),
            'bc1_asr_pct': prior['asr_pct'][model]['p1'],
            'sum255_asr_pct_mean': float(attack.mean()), 'sum255_asr_pct_sd': float(attack.std(ddof=1)),
            'shuffled_sum255_asr_pct_mean': float(shuffled_attack.mean())})
        for seed in SEEDS:
            seed_rows.append({'model': model, 'palette_seed': seed, 'bc1_map50': old['p1'],
                 'sum255_map50': values[model][f'retained_sum255_s{seed}'],
                 'shuffled_sum255_map50': values[model][f'shuffled_sum255_s{seed}'],
                 'sum255_asr_pct': asrs[model][f'retained_sum255_s{seed}'],
                 'shuffled_sum255_asr_pct': asrs[model][f'shuffled_sum255_s{seed}']})
    export('all_metrics.csv', rows)
    export('comparison.csv', comparisons)
    export('per_seed.csv', seed_rows)
    summary = {'passed': True, 'evaluations': 66, 'new_attack_evaluations': 60,
        'clean_replays': replays, 'clipping_fraction': 0.0, 'comparison': comparisons,
        'values': values, 'asr_pct': asrs, 'palettes': read(RUN / 'palettes.json'),
        'script_sha256': sha(__file__), 'parent_summary_sha256': sha(ORIGINAL / 'summary.json')}
    write(RUN / 'summary.json', summary)
    lines = ['# 无截断随机三色色板补充实验', '',
        '六模型白盒；INRIA 完整测试集 288 张、589 人体框；60 项攻击评估 + 6 项 clean 重放均完成。',
        '复用原实验 donor、代码、检测器、贴图面积 25%、预处理和 ASR 定义，不新增攻击训练。', '',
        '## 色板抽样与零截断审计', '',
        '固定 seed=0…4，每个输出 RGB 通道独立均匀抽取三个非负整数，其和为 255。',
        '采用 stars-and-bars：从 0…256 无放回抽两个位置 a<b，取 (a,b−a−1,256−b)。',
        '约束是沿三个基色求和，即色板矩阵的列和为 (255,255,255)，不是每个颜色自身的 RGB 和。',
        '全部色板预先冻结，无测试结果筛选；输出沿用现有 recolor()，未改归一化规则。',
        '对 60 张补丁均核验：float64/float32 越界比例为 0，clamp 改变 RGB 分量数为 0。',
        '打乱版本使用原实验同一 RGB 向量置换，且逐像素验证与保留结构输出的置换完全相等。', '',
        '| seed | 基色 1 RGB | 基色 2 RGB | 基色 3 RGB | 通道和 |',
        '|---|---|---|---|---|']
    for seed in SEEDS:
        p = summary['palettes'][f'sum255_s{seed}']
        lines.append(f'| {seed} | {p[0]} | {p[1]} | {p[2]} | [255,255,255] |')
    lines += ['', '## mAP50（百分数，越低攻击越强）', '',
        '| 模型 | Bc1 | 原随机色板 | 无截断随机色板 | 打乱 / 无截断随机色板 |',
        '|---|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['model']} | {r['bc1_map50']:.2f} | {r['old_random_map50_mean']:.2f} ± {r['old_random_map50_sd']:.2f} | {r['sum255_map50_mean']:.2f} ± {r['sum255_map50_sd']:.2f} | {r['shuffled_sum255_map50_mean']:.2f} ± {r['shuffled_sum255_map50_sd']:.2f} |")
    m = comparisons[-1]
    lines += ['', '## 结果解释', '',
        f"六模型平均：无截断随机换色 mAP50={m['sum255_map50_mean']:.2f} ± {m['sum255_map50_sd']:.2f}，相对 Bc1 的变化为 {m['sum255_minus_bc1']:+.2f} 点。",
        f"同色板打乱结构后 mAP50={m['shuffled_sum255_map50_mean']:.2f} ± {m['shuffled_sum255_map50_sd']:.2f}，配对结构退化 {m['paired_structure_effect']:.2f} 点。",
        f"ASR：Bc1={m['bc1_asr_pct']:.2f}%，无截断随机色板={m['sum255_asr_pct_mean']:.2f} ± {m['sum255_asr_pct_sd']:.2f}%，打乱后={m['shuffled_sum255_asr_pct_mean']:.2f}%。", '',
        '这组补充消除了越界截断混杂；不同色板仍改变色调、亮度与对比度，不能理解为仅改变色相。',
        '与旧随机色板的对比是两种抽样分布的描述比较，不是只打开/关闭 clamp 的配对因果实验。',
        '± 是五组色板/置换的样本标准差；mean6 先每 seed 对六模型平均，再跨五个 seed 求 SD。',
        '同一模型只有一个已有训练 seed，不据此推断训练方差、统计显著性、黑盒或物理实验表现。',
        '原实验结果只引用，未覆盖；baseline_audit.json 冻结其 132 份指标文件 SHA256。',
        '六项新 clean 与原 clean 的 mAP50、ASR 及逐 GT 检出记录完全一致。', '',
        '复现：在 PAC-yolo 根目录使用 pac 环境运行本目录 run_experiments.py 的 prepare、run、summarize 阶段。',
        '已有 protocol.json 时勿重新 prepare；run 会跳过已有指标，summarize 会审计全量结果。', '']
    (RUN / 'report.md').write_text('\n'.join(lines))
    import matplotlib.pyplot as plt
    labels = ['Fixed Bc1', 'Random / unconstrained', 'Random / sum255', 'Shuffled / sum255']
    means = [m['bc1_map50'], m['old_random_map50_mean'], m['sum255_map50_mean'], m['shuffled_sum255_map50_mean']]
    errors = [0, m['old_random_map50_sd'], m['sum255_map50_sd'], m['shuffled_sum255_map50_sd']]
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.bar(labels, means, yerr=errors, capsize=5, color=['#54a24b','#e45756','#4c78a8','#f2cf5b'])
    ax.set(ylabel='mAP50 (%) - lower means stronger attack', ylim=(0,105), title='Six-model mean: five zero-clipping palettes')
    fig.savefig(RUN / 'comparison.png', dpi=180)
    fig.savefig(RUN / 'comparison.pdf')
    plt.close(fig)
    print(json.dumps({'passed': True, 'evaluations': 66, 'mean6': m}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','run','worker','summarize'))
    parser.add_argument('--gpus', default='3,6,7')
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'run':
        run([int(x) for x in args.gpus.split(',')])
    elif args.phase == 'worker':
        assert sha(ORIGINAL / 'run_experiments.py') == read(RUN / 'protocol.json')['shared_runner_sha256']
        base.worker(args.model)
    else:
        summarize()


if __name__ == '__main__':
    main()
