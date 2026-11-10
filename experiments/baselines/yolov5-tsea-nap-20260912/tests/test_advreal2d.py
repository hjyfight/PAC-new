"""Method checks for the requested AdvReal 2D baseline."""
import importlib.util
from pathlib import Path
import unittest

import torch

from utils.patch import AdvRealEOT, advreal_2d_loss, total_variation
from detector_backends import DetectorBackend


class StubBackend(DetectorBackend):
    def __init__(self, name, output):
        torch.nn.Module.__init__(self)
        self.name = name
        self.output = output

    def forward(self, images):
        return self.output


class AdvReal2DTests(unittest.TestCase):
    def test_objectness_aggregation_ignores_class_probability(self):
        pred = torch.zeros(2, 3, 85, requires_grad=True)
        with torch.no_grad():
            pred[..., 4] = torch.tensor([[0.2, 0.8, 0.1], [0.4, 0.3, 0.2]])
            pred[..., 5:] = 0.001
        loss = advreal_2d_loss(pred)
        torch.testing.assert_close(loss, torch.tensor(0.6))
        loss.backward()
        self.assertEqual(pred.grad[..., 5:].abs().sum().item(), 0)
        self.assertEqual(pred.grad[..., 4].count_nonzero().item(), 2)
        with torch.no_grad():
            pred[..., 5:] = 1
        torch.testing.assert_close(advreal_2d_loss(pred), torch.tensor(0.6))

    def test_v4_uses_joint_confidence_not_synthetic_objectness(self):
        output = torch.zeros(2, 2, 85)
        output[..., 4] = 1
        output[0, 0, 5:7] = torch.tensor([0.2, 0.7])
        output[1, 1, 5:7] = torch.tensor([0.4, 0.3])
        output.requires_grad_()
        loss = StubBackend("yolov4", output).advreal_loss(None)
        torch.testing.assert_close(loss, torch.tensor(0.55))
        loss.backward()
        self.assertEqual(output.grad[..., 4].abs().sum().item(), 0)
        self.assertEqual(output.grad[..., 5:].count_nonzero().item(), 2)

    def test_frcnn_matches_zero_padded_box_score_maximum(self):
        scores = torch.tensor([0.2, 0.6], requires_grad=True)
        empty = torch.empty(0, requires_grad=True)
        loss = StubBackend("fasterrcnn", [{"scores": scores}, {"scores": empty}]).advreal_loss(None)
        torch.testing.assert_close(loss, torch.tensor(0.3))
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.tensor([0.0, 0.5]))

    def test_tv_matches_released_advreal_function(self):
        path = Path('/datacc05/huotianlin/code-hu/PAC/baseline/AdvReal-main/utils/solver/loss.py')
        spec = importlib.util.spec_from_file_location('advreal_reference_loss', path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        torch.manual_seed(712)
        patch = torch.rand(3, 37, 29, requires_grad=True)
        ours = total_variation(patch)
        theirs = reference.TVLoss.smooth(patch)
        torch.testing.assert_close(ours, theirs)
        g1 = torch.autograd.grad(ours, patch)[0]
        g2 = torch.autograd.grad(theirs, patch)[0]
        torch.testing.assert_close(g1, g2)

    def test_jitter_is_applied_before_coverage_without_clipping(self):
        eot = AdvRealEOT(rotation=(20, 20), contrast=(1, 1), brightness=(0.3, 0.3),
                         noise=0, median=1)
        self.assertEqual(eot.cutout_p, 0)
        pixels, mask = eot(torch.ones(3, 32, 32, dtype=torch.float64))
        self.assertTrue((mask == 0).any())
        self.assertTrue(((mask > 0) & (mask < 1)).any())
        torch.testing.assert_close(pixels, 1.3 * mask.expand_as(pixels), atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
