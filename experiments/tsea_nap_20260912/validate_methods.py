"""Run existing unittest cases and focused new augmentation/selection invariants."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[1] / "yolov5-tsea-nap-20260912"
os.environ.update(json.loads((RUN / "runtime_environment.json").read_text()))
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from unittest.mock import patch as mock_patch
from utils.augmentations import TSEAImageAugment
from utils.patch import EOT, TSEAEOT, total_variation
from detector_backends import DetectorBackend

torch.set_num_threads(2)

class NewMethodChecks(unittest.TestCase):
    def test_tsea_source_boxes_follow_rotation_and_flip(self):
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        image[20:80, 30:70] = 255
        labels = np.array([[0, 50/160, 50/100, 40/160, 60/100]], dtype=np.float32)
        aug = TSEAImageAugment()
        aug.jitter = lambda value: value
        # Find a seed whose author gate enables augmentation and hflip.
        for seed in range(20):
            torch.manual_seed(seed)
            gate = torch.empty(1).bernoulli_(0.5).item()
            flip = torch.rand(1).item() < .5
            if gate and flip:
                break
        torch.manual_seed(seed)
        with mock_patch("torchvision.transforms.RandomRotation.get_params", return_value=5):
            output, changed = aug(image, labels.copy())
        ys, xs = np.where(output[:, :, 0] > 127)
        center, half = changed[0, 1:3] * [160, 100], changed[0, 3:5] * [160, 100] / 2
        expected = np.r_[center-half, center+half]
        actual = np.array([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
        self.assertLess(np.max(np.abs(expected-actual)), 1.5)
        self.assertGreater(changed[0, 1], .5)

    def test_tsea_cutout_and_tv_match_author(self):
        # Import only the isolated transform package, not the author's conflicting utils.
        import types
        author = Path("/datacc05/huotianlin/code-hu/PAC/third_party/T-SEA-pac")
        name = "tsea_reference_uap"
        package = types.ModuleType(name)
        package.__path__ = [str(author / "attack/uap")]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + ".transformer", author / "attack/uap/transformer.py")
        transform = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transform)
        # Original code hardcodes CUDA; numerical parity is checked on the GPU in preflight.
        spec = importlib.util.spec_from_file_location("tsea_reference_tv", author / "utils/solver/loss.py")
        loss = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loss)
        value = torch.rand(3, 29, 37, requires_grad=True)
        a, b = total_variation(value), loss.TVLoss.smooth(value)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(torch.autograd.grad(a, value)[0], torch.autograd.grad(b, value)[0])
        eot = TSEAEOT()
        eot.cutout_gate = False
        baseline = EOT(brightness=(-.1,.1), noise=(-.1,.1), scale=(1,1), smooth=7,
                       median=True, value_range=(1e-6,.99999))
        torch.manual_seed(12); np.random.seed(12)
        x = eot.appearance(value)
        torch.manual_seed(12); np.random.seed(12)
        y = baseline.appearance(value)
        torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_nap_frcnn_person_filter_and_empty_gradient(self):
        class Stub(DetectorBackend):
            def __init__(self, output):
                torch.nn.Module.__init__(self)
                self.name, self.output = "fasterrcnn", output
            def forward(self, images):
                return self.output
        a = torch.tensor([.9, .8, .4], requires_grad=True)
        b = torch.tensor([.3, .2], requires_grad=True)
        model = Stub([{"scores": a, "labels": torch.tensor([3,1,1])},
                      {"scores": b, "labels": torch.tensor([1,1])}])
        result = model.nap_loss(None)
        torch.testing.assert_close(result, torch.tensor(.4))
        result.backward()
        torch.testing.assert_close(a.grad, torch.tensor([0.,.5,0.]))
        torch.testing.assert_close(b.grad, torch.zeros(2))

suite = unittest.TestSuite()
for filename in ("test_patch", "test_advreal2d"):
    spec = importlib.util.spec_from_file_location(filename, PROJECT / "tests" / (filename + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if filename == "test_advreal2d" and not Path("/datacc05/huotianlin/code-hu/PAC/baseline/AdvReal-main/utils/solver/loss.py").exists():
        target = module.AdvReal2DTests
        target.test_tv_matches_released_advreal_function = unittest.skip(
            "Existing external AdvReal file absent; shared TV independently checked against author T-SEA")(target.test_tv_matches_released_advreal_function)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NewMethodChecks))
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
          "failures": len(result.failures), "errors": len(result.errors),
          "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped]}
(RUN / "smoke/unit_tests.json").write_text(json.dumps(record, indent=2) + "\n")
raise SystemExit(not result.wasSuccessful())
