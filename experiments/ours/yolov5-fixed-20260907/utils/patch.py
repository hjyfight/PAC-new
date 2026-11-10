# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PAC / CAPGen adversarial patch attack utilities.

Implements the patch parameterizations, EOT augmentation, differentiable placement
and attack loss used by `train_patch.py` and by `val.py --patch`.

Patch families (paper Table 1):
    AdvPatch  - color-unrestricted free-pixel patch (Thys et al. 2019)
    CAPGen-T  - color-probability matrix trained with base colors fixed (Eq.3)
    CAPGen-R  - the same matrix left at its random initialization (no training)
    CAPGen-P  - AdvPatch pattern recolored to new base colors, no retraining (Eq.4)
    AdvReal   - the 2D branch of AdvReal (ESWA 2026): same free-pixel patch, but with
                AdvReal's augmentation set and max-IoU objectness loss. Its 3D rendering
                branch and NRSM are not reproducible here (the release ships no meshes).

The loader accepts legacy `patch_logits`, exact RGB `patch` tensors, and
color-probability checkpoints carrying `logits` + `base_colors`.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.metrics import box_iou

# Paper base color sets Bc1 / Bc2 (RGB 0-255).
BC1 = np.array([[119, 49, 72], [2, 204, 1], [134, 2, 182]], dtype=np.float32)
BC2 = np.array([[199, 21, 131], [40, 165, 4], [16, 69, 120]], dtype=np.float32)
BASE_COLORS = {"bc1": BC1, "bc2": BC2}

# The 30 printable colors of Thys et al. (`non_printability/30values.txt`), used by the NPS loss.
PRINTABLE_COLORS = np.array(
    [
        [0.10588, 0.054902, 0.1098], [0.48235, 0.094118, 0.16863], [0.50196, 0.52549, 0.17647],
        [0.082353, 0.31765, 0.18431], [0.47843, 0.61176, 0.51765], [0.07451, 0.3098, 0.45882],
        [0.67843, 0.14902, 0.18039], [0.086275, 0.14118, 0.26275], [0.26667, 0.36863, 0.47843],
        [0.76078, 0.54118, 0.5451], [0.73333, 0.49412, 0.27451], [0.25882, 0.35294, 0.18039],
        [0.47843, 0.22353, 0.36471], [0.27059, 0.086275, 0.11765], [0.7098, 0.32157, 0.2],
        [0.27451, 0.13725, 0.29412], [0.75294, 0.75686, 0.63137], [0.28627, 0.54902, 0.41176],
        [0.47451, 0.2902, 0.15294], [0.74902, 0.70196, 0.28627], [0.098039, 0.42745, 0.44314],
        [0.50588, 0.65098, 0.65882], [0.12549, 0.42745, 0.23529], [0.4902, 0.58431, 0.33725],
        [0.26275, 0.49412, 0.26275], [0.07451, 0.14902, 0.12549], [0.090196, 0.20392, 0.36078],
        [0.68627, 0.15686, 0.30196], [0.30196, 0.5451, 0.57647], [0.71765, 0.32941, 0.40784],
    ],
    dtype=np.float32,
)  # fmt: skip


class AdvPatch(nn.Module):
    """Free RGB pixels for AdvPatch; AdvReal retains its existing sigmoid parameterization."""

    def __init__(self, patch_size=300, use_logits=False):
        """Initializes AdvPatch to gray, as in the Thys reference implementation."""
        super().__init__()
        self.patch_size = patch_size
        self.use_logits = use_logits
        initial = (
            torch.randn(3, patch_size, patch_size) * 0.1
            if use_logits else torch.full((3, patch_size, patch_size), 0.5)
        )
        self.values = nn.Parameter(initial)

    def forward(self, eot=None):
        """Renders RGB, optionally returning a transformed patch and its coverage mask."""
        patch = torch.sigmoid(self.values) if self.use_logits else self.values
        return eot(patch) if eot is not None else patch

    def checkpoint(self):
        """Saves exact RGB values, or the original AdvReal logits."""
        key = "patch_logits" if self.use_logits else "patch"
        return {key: self.values.detach().cpu(), "patch_size": self.patch_size}


def render_color_prob(logits, base_colors, tau=0.1, eot=None):
    """Applies PAC EOT to m before Eq.3 rendering, without clipping m to one.

    EOT returns m_f and its geometric coverage. Softmax removes the common
    coverage factor from each pixel's channels; multiply the resulting RGB by
    coverage exactly once so the caller can composite it over the background.
    Without EOT, the return value remains the checkpoint's RGB rendering.
    """
    m = torch.sigmoid(logits).permute(2, 0, 1)
    mask = None
    if eot is not None:
        m, mask = eot(m)
    r = F.softmax(torch.log(m + 1e-8) / tau, dim=0)
    patch = torch.einsum("khw,kc->chw", r, base_colors).clamp(0, 1)
    return (patch * mask, mask) if mask is not None else patch


class ColorProbPatch(nn.Module):
    """CAPGen color-probability matrix over k fixed base colors (paper Eq.3)."""

    def __init__(self, base_colors, patch_size=300, tau=0.1):
        """Initializes the (P, P, K) probability logits for the given base colors (RGB 0-255)."""
        super().__init__()
        colors = torch.as_tensor(np.asarray(base_colors, dtype=np.float32) / 255.0)
        self.register_buffer("base_colors", colors)
        self.patch_size = patch_size
        self.tau = tau
        self.logits = nn.Parameter(torch.randn(patch_size, patch_size, colors.shape[0]))

    def forward(self, eot=None):
        """Renders the matrix, applying training EOT before conversion to RGB."""
        return render_color_prob(self.logits, self.base_colors, self.tau, eot)

    def checkpoint(self):
        """Returns a serializable state dict for this patch."""
        return {
            "logits": self.logits.detach().cpu(),
            "base_colors": self.base_colors.detach().cpu(),
            "num_base_colors": self.base_colors.shape[0],
            "temperature": self.tau,
            "patch_size": self.patch_size,
        }


def recolor(patch, base_colors):
    """CAPGen-P (Eq.4): reuse an AdvPatch pattern with a new color set, no retraining.

    Each pixel's RGB channels are the (unnormalized) weights of the three source
    primaries, so the recolored pixel is sum_k c_k * (rgb_k / 255).

    Args:
        patch (torch.Tensor): source AdvPatch, (3, P, P) in [0, 1].
        base_colors (np.ndarray): target base colors, (3, 3) RGB in 0-255.

    Returns:
        (torch.Tensor): recolored patch, (3, P, P) in [0, 1].
    """
    colors = torch.as_tensor(np.asarray(base_colors, dtype=np.float32) / 255.0, device=patch.device)
    return (patch.permute(1, 2, 0) @ colors).clamp(0, 1).permute(2, 0, 1)


def load_patch(path, device="cpu", patch_size=300):
    """Loads any PAC patch artifact into a rendered (3, P, P) tensor in [0, 1].

    Accepts free-pixel checkpoints (`patch_logits`), color-probability checkpoints
    (`logits` + `base_colors`), or the literal string "gray" for the Table 1 baseline.
    """
    if str(path) == "gray":
        return torch.full((3, patch_size, patch_size), 127 / 255, device=device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "patch_logits" in ckpt:
        return torch.sigmoid(ckpt["patch_logits"].to(device)).detach()
    if "patch" in ckpt:  # already-rendered patch (CAPGen-P output)
        return ckpt["patch"].to(device).clamp(0, 1).detach()

    colors = torch.as_tensor(ckpt["base_colors"], dtype=torch.float32, device=device)
    return render_color_prob(ckpt["logits"].to(device), colors, float(ckpt.get("temperature", 0.1))).detach()


def median_pool(patch, kernel_size):
    """Median filter with the reflect padding used by the AdvPatch reference."""
    k = kernel_size
    patch = F.pad(patch, (k // 2, k - 1 - k // 2) * 2, mode="reflect")
    patch = patch.unfold(2, k, 1).unfold(3, k, 1)
    return patch.contiguous().view(patch.shape[:4] + (-1,)).median(dim=-1)[0]


class EOT(nn.Module):
    """Smooth, adjust photometry, then transform geometry and coverage together.

    Defaults implement the PAC appendix on m: mean smoothing, D*m + B + N,
    and S*R(theta). Sx and Sy are sampled independently. Matrix values above
    one are retained for Eq.3; only RGB callers supply a value_range.
    """

    def __init__(
        self,
        rotation=(-20, 20),
        brightness=(0.9, 1.1),
        contrast=(0.8, 1.2),
        noise=(0.0, 0.1),
        scale=(0.9, 1.1),
        smooth=3,
        median=False,
        value_range=None,
    ):
        """Configures matrix EOT or the reference RGB transformations."""
        super().__init__()
        self.rotation, self.brightness, self.contrast = rotation, brightness, contrast
        self.noise, self.scale, self.smooth = noise, scale, smooth
        self.median, self.value_range = median, value_range

    @staticmethod
    def _sample(rng):
        """Draws one uniform sample from a (low, high) range."""
        return float(np.random.uniform(*rng))

    def forward(self, patch):
        """Returns transformed channels and coverage, both in the same frame.

        Zero padding premultiplies channel values by coverage during geometric
        interpolation. RGB callers must not multiply by the mask a second time.
        """
        p = patch.unsqueeze(0)
        if self.smooth > 1:
            if self.median:
                p = median_pool(p, self.smooth)
            else:
                pad = self.smooth // 2
                p = F.avg_pool2d(F.pad(p, (pad, pad, pad, pad), mode="replicate"), self.smooth, stride=1)

        p = p * self._sample(self.contrast) + self._sample(self.brightness)
        if self.noise != (0, 0):
            p = p + torch.empty_like(p).uniform_(*self.noise)
        if self.value_range is not None:
            p = p.clamp(*self.value_range)

        rad = math.radians(self._sample(self.rotation))
        sx, sy = self._sample(self.scale), self._sample(self.scale)
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        # affine_grid maps output to input: inverse of the paper's S*R(theta).
        theta = torch.tensor(
            [[cos_t / sx, -sin_t / sy, 0.0], [sin_t / sx, cos_t / sy, 0.0]],
            dtype=p.dtype, device=p.device,
        )
        grid = F.affine_grid(theta.unsqueeze(0), p.shape, align_corners=False)
        sample = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}
        mask = F.grid_sample(torch.ones_like(p[:, :1]), grid, **sample)
        p = F.grid_sample(p, grid, **sample)
        return p.squeeze(0), mask.squeeze(0)


def apply_patch(im, targets, patch, frac=0.25, eot=None):
    """Pastes the patch on every labeled object at `frac` of the box area.

    The patch is centered on the box. Geometry is computed in the resized model input frame,
    so training and validation see exactly the same placement.

    Args:
        im (torch.Tensor): batch of images, (B, 3, H, W) in [0, 1].
        targets (torch.Tensor): labels, (n, 6) as [image_idx, class, xc, yc, w, h] normalized.
        patch (torch.Tensor | nn.Module): rendered RGB, or a patch module that applies EOT before rendering.
        frac (float, optional): patch area as a fraction of the box area. Default is 0.25.
        eot (EOT, optional): applied per placement, in training and validation alike. Default is None.

    Returns:
        (torch.Tensor): the patched batch, same shape and dtype as `im`.
    """
    if targets.numel() == 0:
        return im
    b, _, h, w = im.shape
    targets = targets.to(im.device)
    out = im.clone()
    scale = torch.tensor([w, h, w, h], device=im.device, dtype=torch.float32)

    for row in targets:
        idx = int(row[0])
        xc, yc, bw, bh = (row[2:6].float() * scale).tolist()
        side = int(round(math.sqrt(frac * bw * bh)))
        if side < 1:
            continue
        side = min(side, h, w)
        x0 = min(max(int(round(xc - side / 2)), 0), w - side)
        y0 = min(max(int(round(yc - side / 2)), 0), h - side)
        transformed = patch(eot) if isinstance(patch, nn.Module) else (eot(patch) if eot is not None else patch)
        p, mask = transformed if eot is not None else (transformed, None)
        p = F.interpolate(p.unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False)[0]
        if mask is not None:  # keep the image wherever the transform left the patch empty
            mask = F.interpolate(mask.unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False)[0]
            tile = out[idx, :, y0 : y0 + side, x0 : x0 + side].clone().float()
            p = p + tile * (1 - mask)
        out[idx, :, y0 : y0 + side, x0 : x0 + side] = p.to(out.dtype)
    return out


def total_variation(patch):
    """Total variation of a patch, normalized by element count (Thys et al.).

    Args:
        patch (torch.Tensor): the rendered patch, (3, P, P) in [0, 1].

    Returns:
        (torch.Tensor): scalar TV score, differentiable w.r.t. the patch.
    """
    dx = (patch[:, :, 1:] - patch[:, :, :-1] + 1e-6).abs().sum()
    dy = (patch[:, 1:, :] - patch[:, :-1, :] + 1e-6).abs().sum()
    return (dx + dy) / patch.numel()


def non_printability(patch):
    """Non-printability score: distance from each pixel to its closest printable color.

    Args:
        patch (torch.Tensor): the rendered patch, (3, P, P) in [0, 1].

    Returns:
        (torch.Tensor): scalar NPS, normalized like `total_variation` so the official 0.01
            weight carries over unchanged.
    """
    colors = torch.as_tensor(PRINTABLE_COLORS, device=patch.device, dtype=patch.dtype)
    pixels = patch.permute(1, 2, 0).reshape(-1, 1, 3)
    dist = ((pixels - colors + 1e-6) ** 2).sum(-1).add(1e-6).sqrt()  # (P*P, 30)
    return dist.amin(dim=1).sum() / patch.numel()


def attack_loss(pred, cls=0):
    """Objectness attack loss: mean over images of the peak target-class score.

    Args:
        pred (torch.Tensor): raw model output, (B, N, 5 + nc).
        cls (int, optional): class index to suppress. Default is 0 (person).

    Returns:
        (torch.Tensor): scalar loss, differentiable w.r.t. the patch.
    """
    return (pred[..., 4] * pred[..., 5 + cls]).amax(dim=1).mean()


class AdvRealEOT(nn.Module):
    """AdvReal's 2D-branch augmentation (ESWA 2026), in place of the CAPGen EOT.

    Applies, in AdvReal's order: median pool, rotation, jitter (contrast, additive
    brightness, uniform noise), then cutout. Parameters are the released defaults from
    `attack/uap/transformer.py` and `configs/baseline/v5.yaml`.

    AdvReal's 3D rendering branch and its NRSM (TPS) deformation are not included: the
    public release ships no human meshes or background renders, so they cannot be run.
    """

    def __init__(self, rotation=(-20, 20), contrast=(0.8, 1.2), brightness=(-0.1, 0.1),
                 noise=0.1, median=7, cutout_ratio=0.4, cutout_fill=0.5, cutout_p=0.9):
        """Initializes AdvReal's augmentation ranges."""
        super().__init__()
        self.rotation, self.contrast, self.brightness = rotation, contrast, brightness
        self.noise, self.median = noise, median
        self.cutout_ratio, self.cutout_fill, self.cutout_p = cutout_ratio, cutout_fill, cutout_p

    def forward(self, patch):
        """Applies one random AdvReal draw to a (3, P, P) patch.

        Returns:
            (torch.Tensor): the transformed patch, (3, P, P) in [0, 1].
            (torch.Tensor): its visibility mask, (1, P, P), zero where rotation moved
                content out of frame, so `apply_patch` keeps the image there.
        """
        p = patch.unsqueeze(0)
        if self.median > 1:
            p = median_pool(p, self.median)

        rad = math.radians(float(np.random.uniform(*self.rotation)))
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]], dtype=p.dtype, device=p.device)
        grid = F.affine_grid(theta.unsqueeze(0), p.shape, align_corners=False)
        sample = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}
        mask = F.grid_sample(torch.ones_like(p[:, :1]), grid, **sample)
        p = F.grid_sample(p, grid, **sample) / mask.clamp_min(1e-8)

        p = p * float(np.random.uniform(*self.contrast)) + float(np.random.uniform(*self.brightness))
        p = p + (torch.rand_like(p) * 2 - 1) * self.noise

        if np.random.rand() < self.cutout_p:  # square patch, so one ratio covers both axes
            side = int(round(p.shape[-1] * self.cutout_ratio))
            if side > 0:
                y = np.random.randint(0, p.shape[-2] - side + 1)
                x = np.random.randint(0, p.shape[-1] - side + 1)
                p[..., y : y + side, x : x + side] = self.cutout_fill
        return (p.clamp(0, 1) * mask).squeeze(0), mask.squeeze(0)


def max_iou_loss(pred, targets, imgsz, cls=0, iou_thresh=0.45):
    """AdvReal's `max_iou` detection loss.

    For every labeled box, the prediction of class `cls` with the highest IoU above
    `iou_thresh` contributes its objectness score; images whose targets are already
    missed contribute nothing. This is per-target, unlike `attack_loss`, which takes one
    peak score per image.

    Args:
        pred (torch.Tensor): raw model output, (B, N, 5 + nc), boxes in input-frame pixels.
        targets (torch.Tensor): labels, (n, 6) as [image_idx, class, xc, yc, w, h] normalized.
        imgsz (tuple[int, int]): model input (height, width), to put targets in the same frame.
        cls (int, optional): class index to suppress. Default is 0 (person).
        iou_thresh (float, optional): IoU gate for matching. Default is 0.45 (AdvReal --train_iou).

    AdvReal's TV term is `max(2.5 * tv, 0.1)`, which is what `--tv 2.5` already applies in
    `train_patch.py`, so it is not repeated here.

    Returns:
        (torch.Tensor): scalar loss, differentiable w.r.t. the patch.
    """
    boxes = xywh2xyxy(pred[..., :4])  # (B, N, 4) in pixels
    scores = pred[..., 4] * pred[..., 5 + cls]
    h, w = imgsz
    scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
    hits = []
    for i in range(pred.shape[0]):
        t = targets[targets[:, 0] == i]
        if not len(t):
            continue
        gt = xywh2xyxy(t[:, 2:6].to(boxes.device).to(boxes.dtype) * scale)
        ious = box_iou(gt, boxes[i])  # (n_gt, N)
        best, idx = ious.max(dim=1)
        keep = best >= iou_thresh
        if keep.any():
            hits.append(scores[i][idx[keep]])
    return torch.cat(hits).mean() if hits else scores.sum() * 0
