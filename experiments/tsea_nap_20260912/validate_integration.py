"""Verify actual one-epoch training, recovery state and author Cutout parity."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_recolor import CODE_FILES, PROJECT, PYTHON, ROOT, RUN, RUNTIME, read, sha, write
from controller import audit_training
import torch
from unittest.mock import patch as mock_patch
from utils.patch import EOT, TSEAEOT

base = read(ROOT / "experiments/advpatch_cap_completion_20260911/manifest.json")
files = {name: sha(PROJECT / name) for name in CODE_FILES}

def run_one(method, gpu):
    directory = RUN / "smoke/integration" / method
    directory.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **RUNTIME}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [PYTHON, "-B", "train_patch.py", "--method", method, "--detector", "yolov5s", "--weights",
               base["detectors"]["yolov5s"]["weights"], "--data", base["data_yaml"], "--preprocess", "advpatch",
               "--epochs", "1", "--batch-size", "8", "--workers", "4", "--device", str(gpu),
               "--seed", str(base["seed"]), "--project", str(directory / "patches"), "--name", method]
    with (directory / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f"One-epoch {method} failed; inspect {directory / 'train.log'}")
    config = {"method": method, "epochs": 1, "expected_optimizer_steps": 77, "optimization_source": "yolov5s",
              "initial_lr": .03 if method == "tsea" else .005, "source_code_sha256": files,
              "nap_weights": str(RUN / "weights/NAP_BigGAN_G_ema.pth"),
              "nap_weights_sha256": read(RUN / "weights/nap_weights.json")["sha256"]}
    record = audit_training(directory, config, gpu)
    print(method, "one-epoch checkpoint audit passed", flush=True)
    return method, record

with ThreadPoolExecutor(max_workers=2) as pool:
    results = dict(pool.map(lambda item: run_one(*item), [("tsea", 1), ("nap", 2)]))
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.cuda.set_device(0)
author = Path("/datacc05/huotianlin/code-hu/PAC/third_party/T-SEA-pac/attack/uap")
package = types.ModuleType("tsea_cutout_reference")
package.__path__ = [str(author)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location("tsea_cutout_reference.transformer", author / "transformer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
reference = module.PatchTransformer(torch.device("cuda:0"), types.SimpleNamespace(SCALE=.15))
reference.logger = False
value = torch.rand(3, 31, 31, device="cuda:0", requires_grad=True)
uniform = torch.Tensor.uniform_
centers = iter([.375, .625])
def fixed_uniform(self, low=0, high=1, **kwargs):
    if (low, high) == (-.05, 1.05):
        if self.numel() == 2:
            return self.copy_(self.new_tensor([.375, .625]))
        return self.fill_(next(centers))
    return uniform(self, low, high, **kwargs)
eot = TSEAEOT()
eot.cutout_gate = True
with mock_patch.object(torch.Tensor, "uniform_", fixed_uniform), mock_patch.object(EOT, "appearance", lambda self, x: x.unsqueeze(0)):
    actual = eot.appearance(value)[0]
    expected = reference.cutout(value[None, None], p_erase=1)[0, 0]
torch.testing.assert_close(actual, expected, atol=2e-7, rtol=0)
g1 = torch.autograd.grad(actual.sum(), value)[0]
g2 = torch.autograd.grad(expected.sum(), value)[0]
torch.testing.assert_close(g1, g2, atol=0, rtol=0)
record = {"passed": True, "cutout_max_error": float((actual - expected).abs().max()),
          "cutout_gradient_max_error": float((g1 - g2).abs().max()), "training": results,
          "source_code_sha256": files}
write(RUN / "smoke/integration_audit.json", record)
print("All integration checks passed", flush=True)
