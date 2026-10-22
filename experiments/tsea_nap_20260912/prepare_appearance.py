"""Matched INRIA appearance panels and blinded pilot materials; no PSNR/SSIM calculation."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil

from prepare_recolor import MODELS, ROOT, RUN, read, sha, write
from finalize_results import LABELS, MODEL_LABELS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.patch import apply_patch, load_patch

OUT = RUN / "appearance_comparison"
ORDER = ("clean", "gray", "advpatch", "p1", "p2", "nap", "cap", "cap_bc1", "cap_bc2",
         "advreal", "advreal_bc1", "advreal_bc2", "tsea", "tsea_bc1", "tsea_bc2")
DISPLAY = {"clean": "Clean", "gray": "Gray", **LABELS}


def to_pil(tensor):
    return Image.fromarray(tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy())


def completed_methods(source):
    return [method for method in ("tsea", "nap") if (RUN / method / source / "source_complete.json").exists()]


def artifacts(source, comparison):
    result = {"clean": {"path": None, "sha256": None}, "gray": {"path": "gray", "sha256": None}}
    for key in ("advpatch", "p1", "p2", "cap", "cap_bc1", "cap_bc2", "advreal", "advreal_bc1", "advreal_bc2"):
        cell = comparison[key]["cells"][source][source]
        result[key] = {"path": cell["patch_path"], "sha256": cell["patch_sha256"], "wb_map50": cell["map50_pct"]}
    for method in completed_methods(source):
        config = read(RUN / method / source / "manifest.json")
        for item in config["patches"]:
            if item.get("selection") != "best":
                continue
            key = method if item["variant"] == "raw" else method + "_" + item["variant"]
            metric = read(RUN / method / source / "eval" / source / item["id"] / "metrics.json")
            result[key] = {"path": item["path"], "sha256": item["sha256"], "wb_map50": metric["map50_pct"]}
    return result


def select_samples(dataset, config):
    if (OUT / "samples.json").exists():
        samples = read(OUT / "samples.json")["samples"]
        for item in samples:
            assert dataset.im_files[item["dataset_index"]] == item["image"]
            assert sha(item["image"]) == item["image_sha256"]
        return samples
    areas = np.array([float(np.max(labels[:, 3] * labels[:, 4])) for labels in dataset.labels])
    rng = np.random.default_rng(config["seed"])
    samples = []
    for quartile, indices in enumerate(np.array_split(np.argsort(areas, kind="stable"), 4), 1):
        for index in rng.choice(indices, size=2, replace=False):
            source = dataset.im_files[int(index)]
            samples.append({"id": f"s{len(samples)+1:02}", "dataset_index": int(index), "image": source,
                            "image_sha256": sha(source), "largest_person_area_fraction": float(areas[index]), "quartile": quartile})
    write(OUT / "samples.json", {"seed": config["seed"], "selection": "Two images per largest-GT-area quartile; independent of attacks and visual outcomes",
          "split": "INRIA test 288", "samples": samples, "psnr_ssim_computed": False})
    return samples


@torch.inference_mode()
def render_source(source, dataset, samples, comparison, config):
    entries = artifacts(source, comparison)
    directory = OUT / "sources" / source
    directory.mkdir(parents=True, exist_ok=True)
    patches = {}
    for key, item in entries.items():
        if item["path"] not in (None, "gray"):
            assert sha(item["path"]) == item["sha256"]
        patches[key] = None if key == "clean" else load_patch(item["path"])
        if patches[key] is not None:
            target = directory / "textures" / (key + ".png")
            target.parent.mkdir(exist_ok=True)
            to_pil(patches[key]).save(target)
    rendered = []
    for item in samples:
        image, targets, image_path, _ = dataset[item["dataset_index"]]
        clean = image.float().div(255).unsqueeze(0)
        assert image_path == item["image"] and (targets[:, 1] == 0).all()
        coverage = apply_patch(torch.zeros_like(clean), targets, torch.ones(3, 300, 300), config["patch_area_fraction"])
        mask = coverage.bool()
        row = targets[torch.argmax(targets[:, 4] * targets[:, 5])]
        xc, yc, w, h = (row[2:6] * config["imgsz"]).tolist()
        roi = (max(0, math.floor(xc-w/2)), max(0, math.floor(yc-h/2)),
               min(640, math.ceil(xc+w/2)), min(640, math.ceil(yc+h/2)))
        views = {}
        for key, patch in patches.items():
            result = clean if patch is None else apply_patch(clean, targets, patch, config["patch_area_fraction"])
            assert torch.equal(result[~mask], clean[~mask]), (source, item["id"], key, "background changed outside common support")
            scene = directory / "scene" / item["id"] / (key + ".png")
            crop = directory / "roi" / item["id"] / (key + ".png")
            scene.parent.mkdir(parents=True, exist_ok=True)
            crop.parent.mkdir(parents=True, exist_ok=True)
            pil = to_pil(result[0])
            pil.save(scene)
            pil.crop(roi).save(crop)
            views[key] = {"scene": str(scene.relative_to(OUT)), "roi": str(crop.relative_to(OUT)),
                          "texture": str((directory / "textures" / (key + ".png")).relative_to(OUT)) if patch is not None else None,
                          "wb_map50": entries[key].get("wb_map50")}
        rendered.append({"id": item["id"], "roi_xyxy": list(roi), "image": image_path,
                         "common_patch_pixels": int(mask[0,0].sum()), "views": views})
    record = {"source": source, "new_methods_complete": completed_methods(source), "artifacts": entries, "samples": rendered,
              "same_placement_and_area": True, "outside_patch_support_unchanged": True, "test_eot": False,
              "preview_quantization": "round-to-nearest 8-bit RGB; detection evaluation uses exact float32 checkpoints",
              "psnr_ssim_computed": False}
    write(directory / "manifest.json", record)
    print("Appearance panels:", source, len(entries), "methods x", len(samples), "matched scenes", flush=True)


def export_gallery():
    data = {}
    for source in MODELS:
        path = OUT / "sources" / source / "manifest.json"
        if path.exists():
            data[source] = read(path)
    payload = {"sources": data, "order": list(ORDER), "labels": DISPLAY, "models": MODEL_LABELS}
    write(OUT / "gallery.json", payload)
    template = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAC 同场景外观对照</title><style>body{font:15px system-ui;margin:28px;background:#f4f5f7;color:#162333}h1{font-size:24px}header{max-width:1150px}select,button{font:inherit;padding:7px;margin-right:12px}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin-top:22px}.card{background:white;padding:12px;border-radius:8px;box-shadow:0 1px 4px #ddd}.card img{width:100%;height:320px;object-fit:contain;background:#f1f1f1}.missing{height:320px;display:grid;place-items:center;color:#667; background:#e9ecef}.meta{font-size:13px;color:#52616e}a{color:#075b92}</style>
<header><h1>同场景、同位置、同面积的外观对照</h1><p>样本按人体框面积分层固定抽取，未按攻击结果或外观挑图。这里只展示图片；PSNR/SSIM 仍保持暂停。新方法须完成正式训练与评估后才加入。</p><p><a href="blinded/index.html">匿名评分试用页</a> · <a href="blinded/ratings_template.csv">空白评分表</a> · <a href="../PROTOCOL.md">实验协议</a></p>
<label>补丁源模型 <select id="source"></select></label><label>场景 <select id="sample"></select></label><label>视图 <select id="view"><option value="scene">整图</option><option value="roi">同一人体区域</option><option value="texture">补丁纹理</option></select></label></header><main id="grid"></main>
<script>const data=__DATA__;const source=document.querySelector('#source'),sample=document.querySelector('#sample'),view=document.querySelector('#view');for(const name of Object.keys(data.sources)){source.add(new Option(data.models[name],name));}source.value=data.sources.yolov5s?'yolov5s':source.value;function choose(){sample.replaceChildren();for(const x of data.sources[source.value].samples)sample.add(new Option(x.id,x.id));render();}function render(){const scene=data.sources[source.value].samples.find(x=>x.id===sample.value);const grid=document.querySelector('#grid');grid.replaceChildren();for(const key of data.order){let card=document.createElement('article');card.className='card';let title=document.createElement('h3');title.textContent=data.labels[key];card.append(title);const item=scene.views[key];if(item&&item[view.value]){let img=document.createElement('img');img.src=item[view.value];img.alt=data.labels[key];card.append(img);let meta=document.createElement('p');meta.className='meta';meta.textContent=item.wb_map50==null?'共同场景控制':`白盒 mAP50：${item.wb_map50.toFixed(2)}（越低攻击越强）`;card.append(meta);}else{let box=document.createElement('div');box.className='missing';box.textContent=item?'无补丁纹理':'等待该源模型训练与评估完成';card.append(box);}grid.append(card);}}source.onchange=choose;sample.onchange=render;view.onchange=render;choose();</script></html>"""
    (OUT / "index.html").write_text(template.replace("__DATA__", json.dumps(payload, ensure_ascii=False)))
    if "yolov5s" in data:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        sample = data["yolov5s"]["samples"][0]
        panel = Image.new("RGB", (5*280, 3*310+45), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((12,12), "Matched INRIA scene / YOLOv5s-source patches / pending methods left blank", fill="black", font=font)
        for i,key in enumerate(ORDER):
            x,y = (i%5)*280, (i//5)*310+45
            draw.text((x+8,y+5), DISPLAY[key], fill="black", font=font)
            if key in sample["views"]:
                with Image.open(OUT / sample["views"][key]["scene"]) as image:
                    panel.paste(image.resize((264,264), Image.Resampling.LANCZOS),(x+8,y+32))
            else:
                draw.rectangle((x+8,y+32,x+272,y+296),fill="#eceff2")
                draw.text((x+65,y+150),"Training pending",fill="#666666",font=font)
        panel.save(OUT / "overview_yolov5s.png")
    return data


def export_blinded(data):
    if "yolov5s" not in data:
        return
    # A single fixed source is used for the pilot, avoiding duplicate clean controls across six sources.
    source = data["yolov5s"]
    directory = OUT / "blinded"
    directory.mkdir(exist_ok=True)
    cases, key_rows = [], []
    for sample in source["samples"]:
        for method,item in sample["views"].items():
            identifier = hashlib.sha256(f"PAC-pilot-1788272031:{sample['id']}:{method}".encode()).hexdigest()[:12]
            for view in ("scene","roi"):
                shutil.copyfile(OUT / item[view], directory / (identifier + "_" + view + ".png"))
            cases.append({"id":identifier,"scene":identifier+"_scene.png","roi":identifier+"_roi.png"})
            key_rows.append({"stimulus_id":identifier,"source":"yolov5s","sample_id":sample["id"],"method":DISPLAY[method],
                             "image":sample["image"],"patch_path":source["artifacts"][method]["path"]})
    cases.sort(key=lambda x:x["id"])
    with (directory / "ratings_template.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["participant_id","stimulus_id","view","environmental_consistency_1_low_7_high","noticeability_1_low_7_high","naturalness_1_low_7_high","notes"])
        writer.writerows([["",x["id"],"scene","","","",""] for x in cases])
    with (OUT / "blinding_key.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(key_rows[0]));writer.writeheader();writer.writerows(key_rows)
    template=r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>衣物/图案外观评分</title><style>body{font:17px system-ui;max-width:850px;margin:30px auto;padding:16px;color:#183040}img{width:100%;height:600px;object-fit:contain;background:#eee}button,select,input{font:inherit;padding:8px;margin:6px}label{display:block;margin:12px 0}.note{font-size:14px;color:#596778}</style>
<h1>衣物/图案外观评分</h1><p>请分别评价图中衣物或图案与周围环境的协调程度、显眼程度和自然程度。没有标准答案；未查看的图片请留空。</p>
<p class="note">这是试用材料。环境一致性沿用论文的 7 级尺度。正式评分请固定视图，并分配平衡的题目子集；环境一致性应使用保留场景背景的整图。所有分数初始为空。</p>
<p class="note">评分仅保留在当前页面中，关闭或刷新前请导出 CSV。此处新收集的数字图片评分应单独报告，不与原有实物实验评分合并。</p>
<label>参与者编号 <input id="participant" placeholder="自行填写匿名编号"></label><select id="view"><option value="scene">整图</option><option value="roi">人体区域</option></select><p id="counter"></p><img id="image">
<label>环境一致性（1 完全不一致 — 7 高度一致）<select id="consistency"><option value="">未评分</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option></select></label>
<label>显眼程度（1 很不显眼 — 7 非常显眼）<select id="notice"><option value="">未评分</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option></select></label>
<label>自然程度（1 很不自然 — 7 很自然）<select id="natural"><option value="">未评分</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option></select></label>
<button id="prev">上一张</button><button id="next">下一张</button><button id="save">导出本人评分 CSV</button>
<script>const cases=__CASES__;let i=0;const values={};
const image=document.querySelector('#image'),view=document.querySelector('#view'),c=document.querySelector('#consistency'),n=document.querySelector('#notice'),a=document.querySelector('#natural');
function store(){values[cases[i].id+':'+view.value]=[c.value,n.value,a.value];}
function show(){const x=cases[i];image.src=x[view.value];document.querySelector('#counter').textContent=`${i+1} / ${cases.length} · ${x.id}`;const s=values[x.id+':'+view.value]||['','',''];c.value=s[0];n.value=s[1];a.value=s[2];}
c.onchange=store;n.onchange=store;a.onchange=store;view.onchange=show;
document.querySelector('#prev').onclick=()=>{i=(i+cases.length-1)%cases.length;show();};document.querySelector('#next').onclick=()=>{i=(i+1)%cases.length;show();};
document.querySelector('#save').onclick=()=>{const q=x=>'"'+String(x).replaceAll('"','""')+'"';let rows=[['participant_id','stimulus_id','view','environmental_consistency_1_low_7_high','noticeability_1_low_7_high','naturalness_1_low_7_high','notes']];for(const [key,score] of Object.entries(values)){if(!score.some(x=>x!==''))continue;let [id,v]=key.split(':');rows.push([document.querySelector('#participant').value,id,v,...score,'']);}const url=URL.createObjectURL(new Blob(['\ufeff'+rows.map(x=>x.map(q).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download='appearance_ratings.csv';link.click();URL.revokeObjectURL(url);};show();</script></html>"""
    (directory / "index.html").write_text(template.replace("__CASES__",json.dumps(cases)))
    write(directory / "study_status.json", {"status":"materials_only_no_responses_collected","source":"yolov5s",
          "stimuli":len(cases),"views":["scene","roi"],"psnr_ssim_computed":False,"methods_in_key_only":True,
          "scale_points":7,"axes":["environmental_consistency","noticeability","naturalness"],
          "environmental_consistency_view":"scene","separate_from_existing_physical_study":True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sources",default=",".join(MODELS))
    args=parser.parse_args()
    sources=args.sources.split(",")
    assert set(sources)<=set(MODELS)
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    config=read(RUN / "manifest.json")
    comparison=read(RUN / "matrix_results.json")["comparison_best"]
    _,dataset=create_dataloader(check_dataset(config["data_yaml"])["val"],640,8,32,workers=0,shuffle=False,preprocess="advpatch")
    samples=select_samples(dataset,config)
    for source in sources:
        render_source(source,dataset,samples,comparison,config)
    data=export_gallery()
    export_blinded(data)
    write(OUT / "status.json", {"sources":list(data),"scene_count":len(samples),"psnr_ssim_computed":False,
          "human_ratings":"not collected","new_methods_complete":{s:data[s]["new_methods_complete"] for s in data}})


if __name__=="__main__":
    main()
