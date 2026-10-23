"""
Tests for Utils/tf_device.py, the policy that keeps TensorFlow off the GPU.

The policy is tested in three ways. A fake `tf.config` checks the decision
itself on any machine. The real TensorFlow checks that it works end to end.
The two TensorFlow modules are imported for real to check that each applies
the policy before any TensorFlow op. This box has no GPU, so "a GPU is
present" is simulated with the fake.
"""

import ast
import importlib
import os
import sys
import types
import unittest
from unittest import mock

from Utils.tf_device import TF_GPU_ENV, keep_tensorflow_off_gpu, tf_gpu_opted_in

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TF_MODULES = {
    "Model_Selection.Sensitivity_robustness.GAN_test":
        os.path.join(_REPO, "Model_Selection", "Sensitivity_robustness", "GAN_test.py"),
    "Model_Selection.Sensitivity_robustness.Adversarial_testing":
        os.path.join(_REPO, "Model_Selection", "Sensitivity_robustness",
                     "Adversarial_testing.py"),
}


class _FakeConfig:
    """Records what the policy asks of tf.config."""

    def __init__(self, gpus=("GPU:0",), initialised=False):
        self.gpus = list(gpus)
        self.initialised = initialised
        self.visible_calls = []

    def list_physical_devices(self, device_type=None):
        return list(self.gpus) if device_type == "GPU" else []

    def set_visible_devices(self, devices, device_type=None):
        if self.initialised:
            # What real TensorFlow raises once its runtime is up.
            raise RuntimeError("Visible devices cannot be modified after being initialized")
        self.visible_calls.append((list(devices), device_type))


def _fake_tf(**kw):
    return types.SimpleNamespace(config=_FakeConfig(**kw))


class TestOptIn(unittest.TestCase):

    def test_only_explicit_truthy_values_opt_in(self):
        for value in ("1", "true", "TRUE", " yes ", "on"):
            self.assertTrue(tf_gpu_opted_in({TF_GPU_ENV: value}), value)
        for value in ("", "0", "false", "no", "off", "2", "gpu"):
            self.assertFalse(tf_gpu_opted_in({TF_GPU_ENV: value}), value)
        self.assertFalse(tf_gpu_opted_in({}))


class TestPolicy(unittest.TestCase):

    def test_default_hides_every_gpu_from_tensorflow(self):
        tf = _fake_tf(gpus=["GPU:0", "GPU:1"])
        self.assertTrue(keep_tensorflow_off_gpu(tf, environ={}))
        self.assertEqual(tf.config.visible_calls, [([], "GPU")])

    def test_opt_in_leaves_tensorflow_its_gpus(self):
        tf = _fake_tf()
        self.assertFalse(keep_tensorflow_off_gpu(tf, environ={TF_GPU_ENV: "1"}))
        self.assertEqual(tf.config.visible_calls, [])

    def test_opt_out_values_still_hide(self):
        tf = _fake_tf()
        self.assertTrue(keep_tensorflow_off_gpu(tf, environ={TF_GPU_ENV: "0"}))
        self.assertEqual(tf.config.visible_calls, [([], "GPU")])

    def test_no_gpu_is_harmless(self):
        tf = _fake_tf(gpus=[])
        self.assertTrue(keep_tensorflow_off_gpu(tf, environ={}))

    def test_already_initialised_runtime_warns_instead_of_raising(self):
        # Importing the GAN module must never fail because of the policy.
        tf = _fake_tf(initialised=True)
        self.assertFalse(keep_tensorflow_off_gpu(tf, environ={}))

    def test_reads_the_process_environment_by_default(self):
        tf = _fake_tf()
        with mock.patch.dict(os.environ, {TF_GPU_ENV: "1"}):
            self.assertFalse(keep_tensorflow_off_gpu(tf))
        self.assertEqual(tf.config.visible_calls, [])

    def test_torch_keeps_its_gpu(self):
        """Only TensorFlow's device list changes. CUDA_VISIBLE_DEVICES, which
        torch also reads, is never touched, nor is anything else in os.environ."""
        before = dict(os.environ)
        keep_tensorflow_off_gpu(_fake_tf(), environ={})
        self.assertEqual(dict(os.environ), before)


class TestRealTensorFlow(unittest.TestCase):

    def test_real_tensorflow_ends_with_no_visible_gpu_and_still_computes(self):
        import tensorflow as tf
        before = dict(os.environ)
        with mock.patch.dict(os.environ, {TF_GPU_ENV: ""}):
            self.assertTrue(keep_tensorflow_off_gpu(tf))
        self.assertEqual(tf.config.get_visible_devices("GPU"), [])
        x = tf.constant([[1.0, 2.0]])
        y = tf.matmul(x, tf.transpose(x))
        self.assertAlmostEqual(float(y.numpy()[0, 0]), 5.0)
        self.assertIn("CPU", y.device)
        self.assertEqual(dict(os.environ), before)


class TestModulesApplyThePolicyAtImport(unittest.TestCase):
    """Both TensorFlow modules apply the policy, and before any TensorFlow op."""

    def _import_fresh(self, name, recorder):
        import tensorflow as tf
        sys.modules.pop(name, None)
        with mock.patch.object(tf.config, "set_visible_devices", recorder):
            importlib.import_module(name)

    def test_each_module_hides_the_gpu_on_import(self):
        for name in _TF_MODULES:
            calls = []
            with mock.patch.dict(os.environ, {TF_GPU_ENV: ""}):
                self._import_fresh(name, lambda *a, **k: calls.append((a, k)))
            self.assertEqual(calls, [(([], "GPU"), {})], name)

    def test_each_module_honours_the_opt_in(self):
        for name in _TF_MODULES:
            calls = []
            with mock.patch.dict(os.environ, {TF_GPU_ENV: "1"}):
                self._import_fresh(name, lambda *a, **k: calls.append((a, k)))
            self.assertEqual(calls, [], name)

    def test_policy_runs_right_after_import_tensorflow(self):
        """No module-level statement may touch `tf` between `import tensorflow`
        and the policy call. The device list can only change before the
        runtime starts."""
        for name, path in _TF_MODULES.items():
            with open(path) as fh:
                body = ast.parse(fh.read()).body
            imp = next(i for i, n in enumerate(body)
                       if isinstance(n, ast.Import)
                       and any(a.name == "tensorflow" for a in n.names))
            call = next(i for i, n in enumerate(body)
                        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and getattr(n.value.func, "id", None) == "keep_tensorflow_off_gpu")
            self.assertLess(imp, call, name)
            between = body[imp + 1:call]
            self.assertTrue(all(isinstance(n, ast.ImportFrom)
                                and n.module == "Utils.tf_device" for n in between),
                            f"{name}: only the helper's import may sit between "
                            f"`import tensorflow` and the policy call")


if __name__ == "__main__":
    unittest.main()
