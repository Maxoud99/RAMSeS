"""Loading a checkpoint puts the whole detector on ONE device.

The bug: `app.load_trained_models` loaded with `map_location='cpu'` and then
repaired the pickled `device` attribute only when CUDA was absent. On a GPU
host a GPU-trained detector therefore came back with CPU weights and
`device='cuda'`, and its first forward pass moved the input to the GPU and
multiplied it by CPU weights. The other two load sites repaired nothing.

This box has no GPU, so the GPU host is simulated two ways. `choose_device` is
driven with `torch.cuda.is_available` / `device_count` / `current_device`
patched, and placement onto a CUDA device is observed through the one function
that moves things (`model_io._to`). Real movement is proven separately with the
`meta` device, which every torch build has: if everything a detector holds
lands on `meta`, the walk found everything, and the same walk is what runs for
`cuda:0`.
"""

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import dill
    import numpy as np
    import torch as t
    import torch.serialization as serialization
    from Utils import model_io
except Exception:                                          # pragma: no cover
    model_io = None

REAL_SKAB_1 = "/raid0_ssd2/maxoud/projects/ramses-data/Mononito/trained_models/skab/1"


if model_io is not None:

    class _Estimator:
        """A PyOD-style deep estimator: a plain object holding a network and
        the device it was trained on, NOT registered on the wrapper."""

        def __init__(self, device):
            self.device = device
            self.model = t.nn.Linear(2, 1)
            self.hparams = {"device": device, "lr": 1e-3}
            # A Parameter on a plain object is registered nowhere, so no
            # Module.to will ever reach it.
            self.center = t.nn.Parameter(t.zeros(2))

    class _FakeDeepDetector(t.nn.Module):
        """Everything a real detector in this repo keeps a device in."""

        def __init__(self, device="cuda", estimator_device=t.device("cuda:1")):
            super().__init__()
            self.window_size = 1
            self.window_step = 1
            self.device = device                         # MD/LSTMVAE/DGHL style
            self.net = t.nn.Linear(3, 2)                 # registered submodule
            self.blocks = [t.nn.Linear(2, 2)]            # AnomalyTransformer style
            self.scale = t.ones(2)                       # tensor held as attribute
            self.cache = {"mu": t.zeros(2)}
            self.hidden = (t.zeros(1), t.zeros(1))       # RNN-style state tuple
            self.estimator = _Estimator(estimator_device)  # TSB-AD picks cuda:k

        def forward(self, x):
            x = x.to(self.device)
            return self.blocks[0](self.net(x)) * self.scale

    class _NoDeviceDetector(t.nn.Module):
        """CUDA-trained, but records it ONLY in its storages."""

        def __init__(self):
            super().__init__()
            self.net = t.nn.Linear(3, 2)

    class _HalfInitialised(t.nn.Module):
        """Like Algorithms.nearest_neighbors.NearestNeighbors: an nn.Module
        subclass whose __init__ never calls super().__init__()."""

        def __init__(self):
            from sklearn.neighbors import NearestNeighbors
            self.window_size = 64
            self.device = None
            self.model = NearestNeighbors(n_neighbors=2).fit(np.eye(4))

    class _ClassicalDetector(t.nn.Module):
        """LOF/CBLOF style: a fitted sklearn estimator and device=None."""

        def __init__(self):
            super().__init__()
            from sklearn.neighbors import LocalOutlierFactor
            self.device = None
            self.model = LocalOutlierFactor(n_neighbors=3, novelty=True).fit(
                np.random.RandomState(0).normal(size=(20, 4)))


def _tensors(model):
    """Every tensor reachable from a _FakeDeepDetector, by name."""
    out = {f"net.{k}": v for k, v in model.net.named_parameters()}
    out.update({f"blocks0.{k}": v for k, v in model.blocks[0].named_parameters()})
    out.update({f"estimator.model.{k}": v
                for k, v in model.estimator.model.named_parameters()})
    out.update({"scale": model.scale, "estimator.center": model.estimator.center,
                "cache.mu": model.cache["mu"], "hidden0": model.hidden[0],
                "hidden1": model.hidden[1]})
    return out


def _gpu_host(count=1, current=0):
    """Patch torch into believing this host has `count` CUDA devices."""
    return mock.patch.multiple(t.cuda, is_available=lambda: True,
                               device_count=lambda: count,
                               current_device=lambda: current)


class _Recorder:
    """Stands in for `model_io._to`: records where things were sent."""

    def __init__(self):
        self.calls = []

    def __call__(self, obj, device):
        self.calls.append((obj, t.device(device)))
        return obj                                   # cannot really go to CUDA here


@unittest.skipIf(model_io is None, "torch not installed in this interpreter")
class TestChooseDevice(unittest.TestCase):

    def test_cpu_host_always_gets_cpu(self):
        with mock.patch.object(t.cuda, "is_available", lambda: False):
            self.assertEqual(model_io.choose_device({"cuda:0"}), t.device("cpu"))
            self.assertEqual(model_io.choose_device({"cuda"}), t.device("cpu"))

    def test_gpu_host_keeps_a_cuda_trained_checkpoint_on_cuda(self):
        with _gpu_host(count=2):
            self.assertEqual(model_io.choose_device({"cuda:0", "cpu"}), t.device("cuda:0"))
            self.assertEqual(model_io.choose_device({"cuda:1"}), t.device("cuda:1"))

    def test_cuda_1_on_a_single_gpu_box_falls_back_to_the_current_device(self):
        with _gpu_host(count=1, current=0):
            self.assertEqual(model_io.choose_device({"cuda:1"}), t.device("cuda:0"))
        with _gpu_host(count=1, current=0):
            self.assertEqual(model_io.choose_device({"cuda"}), t.device("cuda:0"))

    def test_gpu_host_leaves_a_cpu_trained_checkpoint_on_cpu(self):
        with _gpu_host():
            self.assertEqual(model_io.choose_device({"cpu"}), t.device("cpu"))
            self.assertEqual(model_io.choose_device(()), t.device("cpu"))

    def test_unparseable_locations_are_ignored(self):
        with _gpu_host():
            self.assertEqual(model_io.choose_device({"not-a-device"}), t.device("cpu"))


@unittest.skipIf(model_io is None, "torch not installed in this interpreter")
class TestPlaceDetectorOnAGpuHost(unittest.TestCase):
    """The reported bug, on a simulated GPU host."""

    def _place(self, model, **kw):
        recorder = _Recorder()
        with _gpu_host(count=1, current=0), \
                mock.patch.object(model_io, "_to", recorder):
            target = model_io.place_detector(model, **kw)
        return target, recorder

    def test_weights_and_device_attributes_agree(self):
        """The old repair skipped this case entirely: weights on CPU, device
        still 'cuda'. Now every module and tensor is sent to the same device
        the attributes name."""
        model = _FakeDeepDetector(device="cuda")
        target, recorder = self._place(model)
        self.assertEqual(target, t.device("cuda:0"))
        sent = {id(obj): dev for obj, dev in recorder.calls}
        for obj in (model, model.blocks[0], model.estimator.model):
            self.assertEqual(sent.get(id(obj)), target, type(obj).__name__)
        moved_tensors = [dev for obj, dev in recorder.calls if isinstance(obj, t.Tensor)]
        # scale, cache.mu, hidden[0], hidden[1], estimator.center
        self.assertEqual(len(moved_tensors), 5)
        self.assertTrue(all(d == target for d in moved_tensors))
        self.assertEqual({d for _, d in recorder.calls}, {target},
                         "something was sent to a second device")
        self.assertEqual(model.device, "cuda:0")                  # str stays str
        self.assertEqual(model.estimator.device, t.device("cuda:0"))
        self.assertEqual(model.estimator.hparams["device"], t.device("cuda:0"))

    def test_a_cuda_1_checkpoint_on_a_one_gpu_box_is_remapped(self):
        model = _FakeDeepDetector(device="cuda:1")
        target, _ = self._place(model, trained_on={"cuda:1"})
        self.assertEqual(target, t.device("cuda:0"))
        self.assertEqual(model.device, "cuda:0")
        self.assertEqual(model.estimator.device, t.device("cuda:0"))

    def test_the_storage_location_alone_is_enough_evidence(self):
        model = _NoDeviceDetector()
        target, recorder = self._place(model, trained_on={"cuda:0"})
        self.assertEqual(target, t.device("cuda:0"))
        self.assertIn((model, target), [(o, d) for o, d in recorder.calls])

    def test_a_cpu_trained_deep_detector_stays_on_cpu(self):
        model = _FakeDeepDetector(device="cpu", estimator_device=t.device("cpu"))
        target, recorder = self._place(model, trained_on={"cpu"})
        self.assertEqual(target, t.device("cpu"))
        self.assertTrue(all(d.type == "cpu" for _, d in recorder.calls))

    def test_classical_detectors_are_a_no_op(self):
        """sklearn/PyOD-based detectors keep device=None and no tensors: nothing
        in them may be rewritten or sent anywhere but the CPU."""
        model = _ClassicalDetector()
        estimator, fitted = model.model, model.model._fit_X
        snapshot = fitted.copy()
        target, recorder = self._place(model)
        self.assertEqual(target, t.device("cpu"))
        self.assertTrue(all(d.type == "cpu" for _, d in recorder.calls))
        self.assertIsNone(model.device)
        self.assertIs(model.model, estimator)
        self.assertIs(model.model._fit_X, fitted)
        self.assertTrue(np.array_equal(fitted, snapshot))

    def test_a_half_initialised_module_is_walked_not_moved(self):
        """NearestNeighbors never ran nn.Module.__init__, so `.to` raises on it."""
        model = _HalfInitialised()
        target, recorder = self._place(model)
        self.assertEqual(target, t.device("cpu"))
        self.assertEqual(recorder.calls, [])
        self.assertIsNone(model.device)


@unittest.skipIf(model_io is None, "torch not installed in this interpreter")
class TestPlaceDetectorReallyMoves(unittest.TestCase):
    """Real movement, onto the `meta` device, which needs no hardware."""

    def test_everything_reachable_lands_on_the_target(self):
        model = _FakeDeepDetector(device="cuda")
        model_io.place_detector(model, device="meta")
        for name, tensor in _tensors(model).items():
            self.assertEqual(tensor.device.type, "meta", name)
        self.assertIsInstance(model.estimator.center, t.nn.Parameter)
        self.assertEqual(model.device, "meta")
        self.assertEqual(model.estimator.device, t.device("meta"))
        self.assertIsInstance(model.hidden, tuple)

    def test_mixed_placement_is_what_crashed_and_placement_fixes_it(self):
        """The GPU-host state the old code left behind — weights on one device,
        `self.device` naming another — fails on the first forward pass. One
        `place_detector` call makes the same model run."""
        model = _FakeDeepDetector(device="meta")         # weights still on CPU
        with self.assertRaises((RuntimeError, NotImplementedError)):
            model(t.ones(1, 3))
        model_io.place_detector(model, device="meta")
        out = model(t.ones(1, 3))
        self.assertEqual(out.device.type, "meta")
        self.assertEqual(tuple(out.shape), (1, 2))

    def test_cpu_host_makes_a_gpu_trained_detector_runnable(self):
        """The half the old code did handle must keep working."""
        model = _FakeDeepDetector(device="cuda")
        with mock.patch.object(t.cuda, "is_available", lambda: False):
            target = model_io.place_detector(model, trained_on={"cuda:0"})
        self.assertEqual(target, t.device("cpu"))
        self.assertEqual(model.device, "cpu")
        self.assertEqual(model.estimator.device, t.device("cpu"))
        self.assertEqual(model(t.ones(1, 3)).device.type, "cpu")


@unittest.skipIf(model_io is None, "torch not installed in this interpreter")
class TestLoadDetector(unittest.TestCase):

    def _checkpoint(self, model, location):
        """Save as Logger.save_torch_model does, with every storage tagged as
        coming from `location` — how a GPU box writes it."""
        buffer = io.BytesIO()
        with mock.patch.object(serialization, "location_tag", lambda storage: location):
            t.save(model, buffer, pickle_module=dill)
        buffer.seek(0)
        return buffer

    def test_a_cuda_1_checkpoint_reads_on_this_cpu_box(self):
        buffer = self._checkpoint(_FakeDeepDetector(device="cuda:1"), "cuda:1")
        with self.assertRaises(RuntimeError):   # what restoring in place does here
            model_io.load_checkpoint(buffer)
        buffer.seek(0)
        model = model_io.load_detector(buffer)
        self.assertFalse(model.training)
        self.assertEqual(model.device, "cpu")
        for name, tensor in _tensors(model).items():
            self.assertEqual(tensor.device.type, "cpu", name)
        self.assertEqual(model(t.ones(1, 3)).device.type, "cpu")

    def test_on_a_gpu_host_the_storage_tag_decides(self):
        buffer = self._checkpoint(_NoDeviceDetector(), "cuda:1")
        recorder = _Recorder()
        with _gpu_host(count=1, current=0), \
                mock.patch.object(model_io, "_to", recorder):
            model = model_io.load_detector(buffer)
        self.assertEqual([d for o, d in recorder.calls if o is model],
                         [t.device("cuda:0")])

    def test_an_explicit_device_overrides_the_choice(self):
        buffer = self._checkpoint(_FakeDeepDetector(device="cuda"), "cuda:0")
        recorder = _Recorder()
        with _gpu_host(), mock.patch.object(model_io, "_to", recorder):
            model = model_io.load_detector(buffer, device="cpu")
        self.assertEqual({d for _, d in recorder.calls}, {t.device("cpu")})
        self.assertEqual(model.device, "cpu")


@unittest.skipIf(model_io is None or not os.path.isdir(REAL_SKAB_1),
                 "real SKAB checkpoints not on this machine")
class TestRealCheckpointsOnThisCpuHost(unittest.TestCase):
    """The CPU path on the checkpoints users actually have. Read-only: the files
    are opened 'rb' by load_detector and their mtimes are checked."""

    NAMES = ("MD_1", "DGHL_1", "RNN_1", "LSTMVAE_3", "NN_1", "LOF_1", "CBLOF_1")

    def test_every_family_loads_onto_the_cpu(self):
        for name in self.NAMES:
            path = os.path.join(REAL_SKAB_1, f"{name}.pth")
            if not os.path.exists(path):
                continue
            with self.subTest(detector=name):
                before = os.stat(path).st_mtime_ns
                with open(path, "rb") as fh:
                    model = model_io.load_detector(fh)
                self.assertEqual(os.stat(path).st_mtime_ns, before)
                device = getattr(model, "device", None)
                if device is not None:
                    self.assertEqual(str(device), "cpu")
                if "_parameters" in vars(model):
                    for pname, p in model.named_parameters():
                        self.assertEqual(p.device.type, "cpu", pname)


if __name__ == "__main__":
    unittest.main()
