"""PNAP diffusion parameterization and released loss functions.

Author code remains byte-for-byte intact. Selected definitions are loaded by
AST to avoid importing their unrelated detector trees, CUDA setup, and W&B.
"""
import ast
import importlib.util
import time

from common import AUTHOR, SD_PATH
import numpy as np
import torch
from torch import nn
import torchvision


def author_definitions(relative_path, names, namespace):
    path = AUTHOR / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError(f"Missing author definitions: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def loss_namespace():
    from utils.general import xywh2xyxy
    ns = {"torch": torch, "nn": nn, "np": np, "time": time,
          "torchvision": torchvision, "xywh2xyxy": xywh2xyxy}
    author_definitions("utils/general.py", ["non_max_suppression_gradable"], ns)
    return author_definitions("load_data.py", ["NPSCalculator", "TotalVariation",
        "MaxProbExtractor_yolov5", "MeanProbExtractor_yolov5", "MeanProbExtractor_frcnn"], ns)


class DetectionLoss:
    def __init__(self, backend):
        self.backend = backend
        ns = loss_namespace()
        cls = ns["MaxProbExtractor_yolov5" if backend.name == "yolov3" else
                 "MeanProbExtractor_frcnn" if backend.name == "fasterrcnn" else "MeanProbExtractor_yolov5"]
        args = (0, 80, lambda obj, cls: obj)
        self.extractor = cls(*args) if backend.name == "yolov3" else cls(*args, 0.4, 0.45, 300)

    def __call__(self, images):
        backend = self.backend
        if backend.name == "yolov2":
            # Equivalent to released get_region_boxes: no NMS, filter by
            # argmax class and obj*class>0.4, optimize selected objectness.
            output = backend(images)
            probs, classes = output[..., 5:].detach().max(-1)
            selected = (classes == 0) & (output[..., 4].detach() * probs > 0.4)
            values = [x[m, 4].mean() for x, m in zip(output, selected) if m.any()]
            return torch.stack(values).mean() if values else output[..., 4].sum() * 0
        if backend.name == "yolov4":
            # Explicit extension: author registry names v4 but contains no v4
            # config class. Reconstruct native obj/class, then use v5 extractor.
            from tool.yolo_layer import YoloLayer
            heads = []
            def capture(layer, inputs):
                raw = inputs[0]
                b, _, h, w = raw.shape
                heads.append(raw.reshape(b, -1, 5 + layer.num_classes, h, w)[:, :, 4].sigmoid().flatten(1))
            handles = [m.register_forward_pre_hook(capture) for m in backend.detector.modules()
                       if isinstance(m, YoloLayer)]
            try:
                output = backend(images)
            finally:
                for handle in handles:
                    handle.remove()
            if len(heads) != 3:
                raise RuntimeError("Expected three YOLOv4 heads")
            obj = torch.cat(heads, 1).unsqueeze(-1)
            output = torch.cat((output[..., :4], obj, output[..., 5:] / obj.clamp_min(1e-12)), -1)
        else:
            output = backend(images)
        values = self.extractor(output)
        result = values.to(images.device).mean()
        backend.roi_logits = None
        # The released extractor returns a detached CPU zero for empty boxes.
        # Preserve zero value, while making an all-empty batch backward-safe.
        return result + images.sum() * 0


class PNAP:
    def __init__(self, config, device="cuda:0"):
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        from torch.utils.checkpoint import checkpoint
        self.config, self.device = config, torch.device(device)
        self.generator = torch.Generator(self.device).manual_seed(config["seed"])
        spec = importlib.util.spec_from_file_location("pnap_author_embed", AUTHOR / "patch_embed.py")
        self.author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.author)
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(SD_PATH), torch_dtype=torch.float32, local_files_only=True, use_safetensors=True,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        pipe = self.pipeline
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        for module in (pipe.unet, pipe.vae, pipe.text_encoder):
            module.eval().requires_grad_(False)
        # Activation recomputation leaves the author's function and weights
        # unchanged, and permits backpropagation through four diffusion steps.
        def recomputed(function):
            def call(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(function, *args, use_reentrant=False, **kwargs)
                return function(*args, **kwargs)
            return call
        pipe.unet.forward = recomputed(pipe.unet.forward)
        pipe.vae.decode = recomputed(pipe.vae.decode)
        with torch.no_grad():
            self.embeds = self.author.prompt_embedding(pipe, [config["prompt"]],
                [config["negative_prompt"]], True, self.device)
        pipe.text_encoder.to("cpu")
        self.timesteps = torch.linspace(601, 1, 4, dtype=torch.int, device=self.device)
        self.latents = None

    def initialize(self):
        pipe = self.pipeline
        with torch.no_grad():
            latents = torch.randn((1, pipe.unet.config.in_channels, 64, 64),
                                  generator=self.generator, device=self.device, dtype=torch.float32)
            latents *= pipe.scheduler.init_noise_sigma
            pipe.scheduler.set_timesteps(50, device=self.device)
            latents, initial = self.author.gen_patch(pipe, latents, self.embeds,
                pipe.scheduler.timesteps, 50, True, 7, self.generator, self.device)
            latents = pipe.scheduler.add_noise(latents, torch.randn_like(latents), self.timesteps[:1])
        self.latents = latents.detach().requires_grad_(True)
        return initial

    def render(self):
        return self.author.gen_patch(self.pipeline, self.latents, self.embeds, self.timesteps,
                                     4, True, 7, self.generator, self.device)[1]


def regularizers(device):
    ns = loss_namespace()
    return (ns["NPSCalculator"](str(AUTHOR / "non_printability/30values.txt"), 512).to(device),
            ns["TotalVariation"]().to(device))


def make_eot():
    from utils.patch import EOT
    return EOT(rotation=(-15, 15), brightness=(-0.2, 0.2), contrast=(0.8, 1.2),
               noise=(-0.1, 0.1), scale=(1, 1), smooth=7, median=True,
               value_range=(0.000001, 0.99999))
