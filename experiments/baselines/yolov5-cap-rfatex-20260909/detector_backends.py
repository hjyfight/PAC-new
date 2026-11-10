"""Native detectors for the fixed-source PAC/AdvReal 2D transfer comparison.

YOLOv2/3/4 use T-SEA's native models. YOLOv5 uses the same loader as the selected
PAC run; m and s have explicit, different official checkpoints. Faster R-CNN
uses torchvision COCO V1 with its FrozenBatchNorm settings and local weights.
"""
import importlib
import os
from pathlib import Path
import sys

import torch
from torch import nn
import torchvision

from models.experimental import attempt_load
from utils.general import coco80_to_coco91_class, non_max_suppression, xyxy2xywh

PAC = Path('/datacc05/huotianlin/code-hu/PAC')
REFERENCE = Path('/datacc05/huotianlin/code-hu/PAC-yolo/experiments/ours/yolov5-advpad-20260908')
TSEA = PAC / 'third_party/T-SEA-pac/detlib/HHDet'
WEIGHTS = PAC / 'third_party/weights'
MODEL_NAMES = ('yolov5s', 'yolov2', 'yolov3', 'yolov4', 'yolov5m', 'fasterrcnn')


def decode_yolov2(raw, anchors, image_size):
    """Darknet region head: sigmoid xy/obj, exp wh, softmax COCO classes."""
    batch, channels, gh, gw = raw.shape
    anchor = raw.new_tensor(anchors).reshape(-1, 2)
    count = len(anchor)
    classes = channels // count - 5
    value = raw.view(batch, count, classes + 5, gh, gw).permute(0, 1, 3, 4, 2)
    gy, gx = torch.meshgrid(torch.arange(gh, device=raw.device),
                            torch.arange(gw, device=raw.device), indexing='ij')
    grid = torch.stack((gx, gy), -1).to(raw.dtype)
    ih, iw = image_size
    stride = raw.new_tensor([iw / gw, ih / gh])
    xy = (value[..., :2].sigmoid() + grid) * stride
    wh = value[..., 2:4].exp() * anchor.view(1, count, 1, 1, 2) * stride
    return torch.cat((xy, wh, value[..., 4:5].sigmoid(), value[..., 5:].softmax(-1)), -1).reshape(batch, -1, classes + 5)


class DetectorBackend(nn.Module):
    def __init__(self, name, device='cuda:0', image_size=640, fused=False,
                 conf_threshold=0.001, nms_iou=0.6, max_det=300, cache_dir=None):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(name)
        self.name = name
        self.device = torch.device(device)
        self.image_size = image_size
        self.conf_threshold, self.nms_iou, self.max_det = conf_threshold, nms_iou, max_det
        self.roi_logits = None
        self.roi_hook = None
        self.names = {i: str(i) for i in range(80)}
        self.names[0] = 'person'
        if name in ('yolov5s', 'yolov5m'):
            self.weight_path = REFERENCE / 'yolov5s.pt' if name == 'yolov5s' else WEIGHTS / 'yolov5m-v6.2.pt'
            self.detector = attempt_load(str(self.weight_path), device=self.device, fuse=fused)
            self.names = self.detector.names
            self.source = 'PAC-yolo pinned YOLOv5 loader, official COCO v6.2 checkpoint'
        elif name == 'yolov2':
            root = TSEA / 'yolov2'
            sys.path.append(str(root))
            module = importlib.import_module('yolov2.darknet')
            self.detector = module.Darknet(str(root / 'yolov2/config/yolo.cfg'))
            self.weight_path = WEIGHTS / 'yolov2.weights'
            self.detector.load_weights(str(self.weight_path))
            self.source = str(Path(module.__file__))
        elif name == 'yolov3':
            root = TSEA / 'yolov3/PyTorch_YOLOv3'
            sys.path.append(str(root))
            module = importlib.import_module('pytorchyolo.models')
            self.detector = module.Darknet(str(root / 'config/yolov3.cfg'))
            self.weight_path = WEIGHTS / 'yolov3.weights'
            self.detector.load_darknet_weights(str(self.weight_path))
            self.source = str(Path(module.__file__))
        elif name == 'yolov4':
            root = TSEA / 'yolov4/Pytorch_YOLOv4'
            sys.path.append(str(root))
            module = importlib.import_module('tool.darknet2pytorch')
            self.detector = module.Darknet(str(root / 'cfg/yolov4.cfg'))
            self.weight_path = WEIGHTS / 'yolov4.weights'
            self.detector.load_weights(str(self.weight_path))
            self.source = str(Path(module.__file__))
        else:
            self.weight_path = WEIGHTS / 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
            hub = Path(cache_dir or (Path(__file__).resolve().parent / 'runs/detector_hub'))
            checkpoint = hub / 'checkpoints' / self.weight_path.name
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.exists() and not checkpoint.is_symlink():
                checkpoint.symlink_to(os.path.relpath(self.weight_path, checkpoint.parent))
            if checkpoint.resolve() != self.weight_path.resolve():
                raise RuntimeError(f'Unexpected checkpoint at {checkpoint}')
            old_hub = torch.hub.get_dir()
            try:
                torch.hub.set_dir(str(hub))
                self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                    weights_backbone=None, min_size=image_size, max_size=image_size,
                    box_score_thresh=conf_threshold, box_nms_thresh=nms_iou,
                    box_detections_per_img=max_det)
            finally:
                torch.hub.set_dir(old_hub)
            frozen_bn = [m for m in self.detector.modules() if isinstance(m, torchvision.ops.misc.FrozenBatchNorm2d)]
            assert frozen_bn and all(m.eps == 0 for m in frozen_bn)
            self.roi_hook = self.detector.roi_heads.box_predictor.register_forward_hook(self._capture_roi)
            mapping = torch.full((91,), -1, dtype=torch.long)
            mapping[torch.tensor(coco80_to_coco91_class())] = torch.arange(80)
            self.register_buffer('coco91_to_80', mapping)
            self.source = 'torchvision 0.20.1 fasterrcnn_resnet50_fpn COCO_V1 (FrozenBatchNorm eps=0)'
        self.detector.eval().requires_grad_(False)
        self.to(self.device)
        self.eval()

    def _capture_roi(self, module, inputs, output):
        self.roi_logits = output[0]

    def forward(self, images):
        """Return differentiable native output in the input image coordinate frame."""
        output = self.detector(images)
        if self.name == 'yolov2':
            return decode_yolov2(output, self.detector.anchors, images.shape[-2:])
        if self.name == 'yolov3':
            return output
        if self.name in ('yolov5s', 'yolov5m'):
            return output[0]
        if self.name == 'yolov4':
            boxes, confidence = output
            scale = images.new_tensor([images.shape[-1], images.shape[-2]] * 2)
            boxes = xyxy2xywh(boxes.squeeze(2) * scale)
            # v4 already returns objectness * class probability. Multiplication
            # by the synthetic one-valued obj column in common NMS is exact.
            return torch.cat((boxes, torch.ones_like(confidence[..., :1]), confidence), -1)
        return output

    def predict(self, images):
        output = self(images)
        if self.name != 'fasterrcnn':
            return non_max_suppression(output, self.conf_threshold, self.nms_iou,
                                       multi_label=True, max_det=self.max_det)
        predictions = []
        for item in output:
            labels = self.coco91_to_80[item['labels']]
            valid = labels >= 0
            predictions.append(torch.cat((item['boxes'][valid], item['scores'][valid, None],
                                          labels[valid, None].to(item['boxes'].dtype)), 1))
        self.roi_logits = None
        return predictions

    def advreal_loss(self, images):
        """Peak confidence as exposed by the released detlib API, averaged over images.

        v2/v3/v5 expose raw objectness. v4 exposes max(obj * class) over classes.
        Faster R-CNN's default score="bbox" exposes final box scores padded by zeros.
        """
        output = self(images)
        if self.name == "fasterrcnn":
            loss = torch.stack([item["scores"].max() if item["scores"].numel()
                                else item["scores"].sum() for item in output]).mean()
            self.roi_logits = None
            return loss
        if self.name == "yolov4":
            return output[..., 5:].amax(dim=(1, 2)).mean()
        return output[..., 4].amax(dim=1).mean()

    def cap_loss(self, images):
        """CAP's peak raw objectness; Faster R-CNN uses the existing final-score extension.

        T-SEA's v4 decoder multiplies objectness by class scores. Read its three
        raw head inputs so CAP optimizes objectness without that extra class factor.
        """
        if self.name != "yolov4":
            return self.advreal_loss(images)
        from tool.yolo_layer import YoloLayer

        confidence = []

        def capture(layer, inputs):
            raw = inputs[0]
            b, _, h, w = raw.shape
            confidence.append(raw.reshape(b, -1, 5 + layer.num_classes, h, w)[:, :, 4].sigmoid().flatten(1))

        handles = [layer.register_forward_pre_hook(capture)
                   for layer in self.detector.modules() if isinstance(layer, YoloLayer)]
        try:
            self.detector(images)
        finally:
            for handle in handles:
                handle.remove()
        if len(confidence) != 3:
            raise RuntimeError(f"Expected three YOLOv4 heads, got {len(confidence)}")
        return torch.cat(confidence, dim=1).amax(dim=1).mean()

    def person_gradient_probe(self, images):
        """Connectivity check only; this is not the AdvReal training objective."""
        output = self(images)
        if self.name == 'fasterrcnn':
            return self.roi_logits.softmax(-1)[:, 1].max()
        return (output[..., 4] * output[..., 5]).amax(dim=1).mean()
