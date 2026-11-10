# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Numerical invariants for PAC matrix EOT, compositing, and raw RGB recoloring."""

import tempfile
import unittest
from unittest.mock import patch as mock_patch
from pathlib import Path

import numpy as np
import torch

from utils.patch import BC1, BC2, EOT, AdvPatch, AdvRealEOT, CAPRotation, ColorProbPatch, apply_patch, load_patch, recolor


class PatchTests(unittest.TestCase):
    def test_rotation_preserves_constant_color_at_empty_and_partial_coverage(self):
        """Rotating a patch onto an identical background cannot create a dark edge."""
        image = torch.full((1, 3, 64, 64), 0.37)
        patch = torch.full((3, 32, 32), 0.37)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 1, 1]])
        transformations = [
            EOT(rotation=(20, 20), brightness=(0, 0), contrast=(1, 1), noise=(0, 0), scale=(1, 1), smooth=1),
            AdvRealEOT(rotation=(20, 20), brightness=(0, 0), contrast=(1, 1), noise=0, median=1, cutout_p=0),
        ]
        for eot in transformations:
            with self.subTest(transform=type(eot).__name__):
                _, mask = eot(patch)
                assert (mask == 0).any() and ((mask > 0) & (mask < 1)).any()
                torch.testing.assert_close(apply_patch(image, targets, patch, eot=eot), image, atol=2e-7, rtol=0)


    def test_cap_rotation_keeps_area_and_background_after_resize(self):
        """Expansion must not shrink the patch or fill rotation corners with black."""
        image = torch.full((1, 3, 96, 96), 0.37)
        patch = torch.full((3, 300, 300), 0.37)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 1, 1]])
        eot = CAPRotation(rotation=(10, 10))
        with mock_patch.object(eot, "forward", wraps=eot.forward) as transform:
            result = apply_patch(image, targets, patch, frac=0.25, eot=eot)
        assert transform.call_args.args[0].shape[-2:] == (48, 48)
        torch.testing.assert_close(result, image, atol=2e-7, rtol=0)
        covered = apply_patch(torch.zeros_like(image), targets, torch.ones_like(patch), frac=0.25, eot=eot)
        assert abs(covered[0, 0].sum().item() - 48 * 48) < 48
        # A rotated patch as wide as the image is cropped safely at the image boundary.
        torch.testing.assert_close(apply_patch(image, targets, patch, frac=1, eot=eot), image, atol=2e-7, rtol=0)

    def test_pac_eot_keeps_values_above_one_and_renders_transformed_weights(self):
        """The paper's additive B acts on m, whose values may exceed one before Eq.3."""
        m = torch.tensor([0.2, 0.5, 0.8])
        module = ColorProbPatch(BC1, patch_size=8)
        with torch.no_grad():
            module.logits.copy_(torch.logit(m).expand_as(module.logits))
        eot = EOT(rotation=(0, 0), brightness=(1, 1), contrast=(1, 1), noise=(0, 0), scale=(1, 1), smooth=1)
        transformed_m, _ = eot(m[:, None, None].expand(3, 8, 8))
        assert transformed_m.min() > 1
        patch, mask = module(eot)
        weights = (m + 1).double().pow(10)
        expected_rgb = (weights / weights.sum()) @ torch.as_tensor(BC1, dtype=torch.float64).div(255)
        torch.testing.assert_close(patch.double(), expected_rgb[:, None, None].expand_as(patch), atol=2e-7, rtol=0)
        torch.testing.assert_close(mask, torch.ones_like(mask))

    def test_pac_noise_is_nonnegative_uniform_and_repeatable(self):
        """PAC noise has the specified support and both RNGs determine repeatability."""
        patch = torch.full((3, 64, 64), 0.5)
        eot = EOT(rotation=(0, 0), brightness=(0, 0), contrast=(1, 1), scale=(1, 1), smooth=1)
        np.random.seed(23)
        torch.manual_seed(23)
        first, _ = eot(patch)
        noise = first - patch
        assert noise.min() >= 0 and noise.max() <= 0.1 + 1e-7
        assert abs(noise.mean().item() - 0.05) < 0.001
        np.random.seed(23)
        torch.manual_seed(23)
        second, _ = eot(patch)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_gradient_reaches_patch_through_transform_and_compositing(self):
        """Every trained family must receive finite, nonzero gradients after placement."""
        for matrix in (False, True):
            with self.subTest(matrix=matrix):
                module = ColorProbPatch(BC1, patch_size=16) if matrix else AdvPatch(patch_size=16)
                eot = EOT() if matrix else EOT(brightness=(-0.1, 0.1), noise=(-0.1, 0.1), value_range=(0, 1))
                image = torch.full((1, 3, 64, 64), 0.37)
                targets = torch.tensor([[0, 0, 0.5, 0.5, 0.8, 0.9]])
                apply_patch(image, targets, module, eot=eot).square().mean().backward()
                gradient = next(module.parameters()).grad
                assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    def test_recolor_preserves_raw_rgb_weights_including_zero_and_one(self):
        """Direct RGB optimization and checkpoint loading must not renormalize P weights."""
        for palette in (BC1, BC2):
            with self.subTest(palette=palette.tolist()), tempfile.TemporaryDirectory() as directory:
                module = AdvPatch(patch_size=16)
                torch.manual_seed(19)
                pixels = torch.rand_like(module.values)
                pixels[:, 0, 0] = 0
                pixels[:, 1, 1] = 1
                with torch.no_grad():
                    module.values.copy_(pixels)
                checkpoint = Path(directory) / "advpatch.pt"
                torch.save(module.checkpoint(), checkpoint)
                restored = load_patch(checkpoint)
                torch.testing.assert_close(restored, pixels, atol=0, rtol=0)
                expected = sum(
                    pixels[i:i + 1].double() * torch.as_tensor(palette[i], dtype=torch.float64)[:, None, None] / 255
                    for i in range(3)
                )
                torch.testing.assert_close(recolor(restored, palette).double(), expected, atol=2e-7, rtol=0)
