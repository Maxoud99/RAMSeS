"""Training survives a detector that fails, and stops one that never finishes.

Two failures used to end a whole run at stage 3. A family that raised — an
AutoEncoder fitted on a series too short for one batch — escaped
`TrainModels.train_models`, which had no recovery at any level. And SOS on a
long entity trained for as long as it took: the only time limit in the
pipeline guards scoring, with an alarm that cannot interrupt a C call anyway.

The trainer tests run the REAL `TrainModels` — its constructor, dispatch,
per-instance isolation and checkpoint writing — on a synthetic entity, with
only the family methods replaced by small fakes shaped like the real ones
(number the grid, ask `_should_train`, fit, save). Each scenario runs both
in-process and with the forked deadline, because those are two different code
paths that must agree. The SOS tests use the real `train_sos`.
"""

import ast
import itertools
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import psutil
    import torch as t
    from loguru import logger
    from Datasets.dataset import Dataset, Entity
    from Model_Training import train as train_mod
    from Model_Training.deadline import (TrainingFailed, TrainingTimeout,
                                         run_with_deadline)
    from Utils.logger import Logger, PARTIAL_SUFFIX
    from Utils.model_io import load_checkpoint
    from WebUI import markers
except Exception:                                          # pragma: no cover
    train_mod = None

# A C-level call that never returns to the interpreter: builtin `sum` over
# `itertools.repeat` runs entirely in C without checking for signals, so it
# stands for the long numpy/BLAS call a SIGALRM handler cannot interrupt.
# ~10^12 additions is well over an hour.
def _stuck_in_c():
    sum(itertools.repeat(1, 10 ** 12))


def _dataset(n_time=300, n_features=2):
    rng = np.random.default_rng(0)
    entity = Entity(Y=rng.normal(size=(n_features, n_time)), name="e",
                    labels=np.zeros((1, n_time)), verbose=False)
    return Dataset(entities=[entity], name="d", verbose=False)


class _Tiny(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = t.nn.Parameter(t.ones(2))


class _Unpicklable(t.nn.Module):
    def __reduce_ex__(self, protocol):
        raise TypeError("cannot pickle this detector")


class _HangsWhilePickled(t.nn.Module):
    """Gets as far as an open checkpoint file, then never finishes writing it."""
    def __reduce_ex__(self, protocol):
        _stuck_in_c()


def _no_children_left(test):
    test.assertEqual([p.pid for p in psutil.Process().children() if p.is_running()
                      and p.status() != psutil.STATUS_ZOMBIE], [],
                     "a training child outlived run_with_deadline")


@unittest.skipIf(train_mod is None, "torch/numpy/loguru not installed in this interpreter")
class TestRunWithDeadline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_what_the_child_writes_to_disk_is_there_afterwards(self):
        """The child saves its own checkpoint; nothing else comes back."""
        path = os.path.join(self.tmp, "written_by_child")
        run_with_deadline(lambda: open(path, "w").write("ok"), seconds=30)
        with open(path) as f:
            self.assertEqual(f.read(), "ok")

    def test_a_call_stuck_in_c_is_stopped_at_the_deadline(self):
        t0 = time.monotonic()
        with self.assertRaises(TrainingTimeout) as ctx:
            run_with_deadline(_stuck_in_c, seconds=1)
        self.assertLess(time.monotonic() - t0, 10, "the deadline did not stop the child")
        self.assertEqual(ctx.exception.seconds, 1)
        _no_children_left(self)

    def test_an_exception_comes_back_as_its_type_and_message(self):
        def fit():
            raise ValueError("AutoEncoder was fitted on 29 window(s)")
        with self.assertRaises(TrainingFailed) as ctx:
            run_with_deadline(fit, seconds=30)
        self.assertEqual(str(ctx.exception),
                         "ValueError: AutoEncoder was fitted on 29 window(s)")

    def test_a_child_killed_by_a_signal_is_a_failure_not_a_crash(self):
        """What the OOM killer does to SOS on a long entity, in miniature."""
        with self.assertRaises(TrainingFailed) as ctx:
            run_with_deadline(lambda: os.kill(os.getpid(), signal.SIGKILL), seconds=30)
        self.assertIn("SIGKILL", str(ctx.exception))
        self.assertIn("out-of-memory", str(ctx.exception))


@unittest.skipIf(train_mod is None, "torch/numpy/loguru not installed in this interpreter")
class TestCheckpointWrite(unittest.TestCase):
    """A `.pth` on disk means "trained" to every reader, so it must never be partial."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log = Logger(save_dir=self.tmp, overwrite=True)
        self.path = os.path.join(self.tmp, "d", "e", "LOF_1.pth")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _save(self, obj):
        self.log.save(obj=obj, obj_name="LOF_1", obj_meta=None, obj_class=["d", "e"])

    def test_a_save_that_raises_leaves_nothing_that_reads_as_trained(self):
        with self.assertRaises(TypeError):
            self._save(_Unpicklable())
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.path + PARTIAL_SUFFIX))
        self.assertFalse(self.log.check_file_exists(["d", "e"], "LOF_1"))

    def test_a_failed_save_keeps_the_previous_checkpoint_intact(self):
        self._save(_Tiny())
        with open(self.path, "rb") as f:
            before = f.read()
        with self.assertRaises(TypeError):
            self._save(_Unpicklable())
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_a_completed_save_loads_back(self):
        self._save(_Tiny())
        self.assertTrue(t.equal(load_checkpoint(self.path).w, t.ones(2)))


@unittest.skipIf(train_mod is None, "torch/numpy/loguru not installed in this interpreter")
class TestTrainingContinuesPastAFailure(unittest.TestCase):
    """Each scenario runs in-process (timeout None) and forked (timeout set)."""

    TIMEOUT = None

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.warnings = []
        self._sink = logger.add(lambda m: self.warnings.append(m.record["message"]),
                                level="WARNING")

    def tearDown(self):
        logger.remove(self._sink)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def trainer(self, families, detectors, data=None, overwrite=True, timeout="class"):
        data = data or _dataset()
        timeout = self.TIMEOUT if timeout == "class" else timeout
        # Passed only when set, so the in-process scenarios also run against a
        # constructor that predates the keyword — which is how they were
        # checked to fail on the code before this fix.
        limit = {} if timeout is None else {"training_timeout": timeout}
        with mock.patch.object(train_mod, "load_data", lambda **kw: data):
            return train_mod.TrainModels(
                dataset="d", entity="e", algorithm_list=families,
                detectors=detectors, save_dir=self.tmp, overwrite=overwrite,
                verbose=False, **limit)

    def fake_family(self, trainer, family, n, behaviour=lambda name: _Tiny()):
        """Shaped like the real `train_*` methods: number the grid, ask
        `_should_train`, build, save. `behaviour(name)` returns the model to
        save, or raises / hangs to simulate a failing fit."""
        def train_family():
            for model_id in range(1, n + 1):
                name = f"{family}_{model_id}"
                if not trainer._should_train(name):
                    continue
                trainer.logging_obj.save(obj=behaviour(name), obj_name=name, obj_meta={},
                                         obj_class=trainer.logging_hierarchy)
        return train_family

    def pth(self, name):
        return os.path.join(self.tmp, "d", "e", f"{name}.pth")

    def test_a_family_that_raises_does_not_stop_the_families_after_it(self):
        """The reported case: AutoEncoder on a short series, then the rest."""
        trainer = self.trainer(["AutoEncoder", "LOF"],
                               ["AutoEncoder_1", "AutoEncoder_2", "LOF_1", "LOF_2"])

        def too_short(name):
            raise ValueError("AutoEncoder was fitted on 29 window(s) but its batch_size is 32")
        fake_ae = self.fake_family(trainer, "AutoEncoder", 3, too_short)
        trainer.train_pyod = lambda model_name: fake_ae()
        trainer.train_lof = self.fake_family(trainer, "LOF", 4)

        trainer.train_models()

        self.assertTrue(os.path.exists(self.pth("LOF_1")))
        self.assertTrue(os.path.exists(self.pth("LOF_2")))
        self.assertFalse(os.path.exists(self.pth("LOF_3")), "trained an unrequested instance")
        self.assertEqual(set(trainer.failed), {"AutoEncoder_1", "AutoEncoder_2"})
        self.assertIn("ValueError: AutoEncoder was fitted on 29 window(s)",
                      trainer.failed["AutoEncoder_1"])
        for name in ("AutoEncoder_1", "AutoEncoder_2"):
            line = next(w for w in self.warnings if name in w)
            self.assertIn("failed to train", line)
            self.assertIn("ValueError", line)
            # The real line, not a copy of it, must raise the run page's banner.
            self.assertEqual(markers.classify(line)["code"], "training_failed")

    def test_one_failing_instance_costs_only_that_instance(self):
        trainer = self.trainer(["LOF"], ["LOF_1", "LOF_2", "LOF_3"])

        def second_fails(name):
            if name == "LOF_2":
                raise RuntimeError("n_neighbors must be smaller than the sample count")
            return _Tiny()
        trainer.train_lof = self.fake_family(trainer, "LOF", 3, second_fails)

        trainer.train_models()

        self.assertEqual(set(trainer.failed), {"LOF_2"})
        self.assertTrue(os.path.exists(self.pth("LOF_1")))
        self.assertTrue(os.path.exists(self.pth("LOF_3")))

    def test_a_save_that_fails_leaves_no_checkpoint_behind(self):
        """The LOF_2 that failed must not be found as trained on the next run."""
        trainer = self.trainer(["LOF"], ["LOF_1", "LOF_2"])
        trainer.train_lof = self.fake_family(
            trainer, "LOF", 2, lambda name: _Unpicklable() if name == "LOF_2" else _Tiny())

        trainer.train_models()

        self.assertIn("TypeError: cannot pickle this detector", trainer.failed["LOF_2"])
        self.assertFalse(os.path.exists(self.pth("LOF_2")))
        self.assertFalse(os.path.exists(self.pth("LOF_2") + PARTIAL_SUFFIX))
        self.assertFalse(trainer.logging_obj.check_file_exists(["d", "e"], "LOF_2"))

    def test_a_family_that_fails_before_any_instance_names_its_detectors(self):
        trainer = self.trainer(["LOF", "CBLOF"], ["LOF_1", "LOF_2", "CBLOF_1"])

        def broken_family():
            raise KeyError("no grid for LOF")
        trainer.train_lof = broken_family
        trainer.train_cblof = self.fake_family(trainer, "CBLOF", 1)

        trainer.train_models()

        self.assertEqual(set(trainer.failed), {"LOF_1", "LOF_2"})
        self.assertTrue(os.path.exists(self.pth("CBLOF_1")))

    def save_checkpoint(self, name):
        """A complete checkpoint, as an earlier run would have left it."""
        Logger(save_dir=self.tmp, overwrite=True).save(
            obj=_Tiny(), obj_name=name, obj_meta={}, obj_class=["d", "e"])
        with open(self.pth(name), "rb") as f:
            return f.read()

    def test_a_family_failure_spares_its_instances_already_on_disk(self):
        """app.py leaves everything in `failed` out of loading, so with
        --overwrite false only the instances this run would have trained may
        be named there; LOF_1's checkpoint needed nothing from the family."""
        before = self.save_checkpoint("LOF_1")
        trainer = self.trainer(["LOF", "CBLOF"], ["LOF_1", "LOF_2", "CBLOF_1"],
                               overwrite=False)

        def broken_family():
            raise KeyError("no grid for LOF")
        trainer.train_lof = broken_family
        trainer.train_cblof = self.fake_family(trainer, "CBLOF", 1)

        trainer.train_models()

        self.assertEqual(set(trainer.failed), {"LOF_2"})
        self.assertFalse(any("LOF_1" in w and "failed to train" in w
                             for w in self.warnings), self.warnings)
        with open(self.pth("LOF_1"), "rb") as f:
            self.assertEqual(f.read(), before)

    def test_a_family_failure_with_every_instance_on_disk_fails_nothing(self):
        self.save_checkpoint("LOF_1")
        self.save_checkpoint("LOF_2")
        trainer = self.trainer(["LOF"], ["LOF_1", "LOF_2"], overwrite=False)

        def broken_family():
            raise KeyError("no grid for LOF")
        trainer.train_lof = broken_family

        trainer.train_models()

        self.assertEqual(trainer.failed, {})
        # Still said, since the family method is broken — but not as a training
        # failure, which would raise the run page's banner for a run that is
        # missing nothing.
        line = next(w for w in self.warnings if "LOF family raised" in w)
        self.assertIn("KeyError", line)
        self.assertIn("already trained", line)
        self.assertNotEqual((markers.classify(line) or {}).get("code"), "training_failed")

    def test_existing_checkpoints_are_skipped_without_training(self):
        trainer = self.trainer(["LOF"], ["LOF_1", "LOF_2"], overwrite=False)
        os.makedirs(os.path.dirname(self.pth("LOF_1")), exist_ok=True)
        with open(self.pth("LOF_1"), "wb") as f:
            f.write(b"already here")
        trained = []

        def record(name):
            trained.append(name)
            return _Tiny()
        trainer.train_lof = self.fake_family(trainer, "LOF", 2, record)
        forked, real = [], train_mod.run_with_deadline

        def spy(fn, seconds):
            forked.append(trainer._only)
            return real(fn, seconds)
        with mock.patch.object(train_mod, "run_with_deadline", spy):
            trainer.train_models()

        with open(self.pth("LOF_1"), "rb") as f:
            self.assertEqual(f.read(), b"already here")
        self.assertTrue(os.path.exists(self.pth("LOF_2")))
        if self.TIMEOUT is None:
            self.assertEqual((trained, forked), (["LOF_2"], []))
        else:
            # `trained` was appended to in the child; what the parent sees is
            # that no process was started for the instance already on disk.
            self.assertEqual(forked, ["LOF_2"])

    def test_the_real_sos_trains_on_a_short_entity(self):
        """Positive control for the size check: `train_sos` still trains."""
        trainer = self.trainer(["SOS"], ["SOS_1"])
        trainer.train_models()
        self.assertEqual(trainer.failed, {})
        self.assertTrue(os.path.exists(self.pth("SOS_1")))

    def test_the_real_sos_refuses_a_long_entity_before_fitting(self):
        from Algorithms.sos import SOS_MAX_ROWS
        trainer = self.trainer(["SOS", "LOF"], ["SOS_1", "SOS_2", "LOF_1"],
                               data=_dataset(n_time=SOS_MAX_ROWS + 1))
        trainer.train_lof = self.fake_family(trainer, "LOF", 1)
        with mock.patch("Algorithms.sos.TsadSOS.fit",
                        side_effect=AssertionError("SOS was fitted after all")):
            t0 = time.monotonic()
            trainer.train_models()
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(set(trainer.failed), {"SOS_1", "SOS_2"})
        # The count is the loader's windows — what SOS would actually be handed
        # — which right-padding makes one more than the timesteps.
        rows = re.search(r"for the ([\d,]+) training rows", trainer.failed["SOS_1"])
        self.assertIsNotNone(rows, trainer.failed["SOS_1"])
        self.assertGreater(int(rows.group(1).replace(",", "")), SOS_MAX_ROWS)
        self.assertIn("downsampling", trainer.failed["SOS_1"])
        self.assertNotIn("fitted after all", trainer.failed["SOS_1"])
        self.assertTrue(os.path.exists(self.pth("LOF_1")))


class TestTrainingContinuesPastAFailureForked(TestTrainingContinuesPastAFailure):
    """The same scenarios, each instance in a child process under a deadline."""

    TIMEOUT = 60

    def test_a_stuck_instance_is_stopped_and_its_siblings_not_attempted(self):
        """SOS on a long entity, in miniature: the first instance never returns.

        It is stopped at the limit; the rest of its family is not given the
        limit again; the next family trains as normal."""
        trainer = self.trainer(["SOS", "LOF"], ["SOS_1", "SOS_2", "SOS_3", "LOF_1"],
                               timeout=1)

        def stuck(name):
            _stuck_in_c()
        trainer.train_sos = self.fake_family(trainer, "SOS", 4, stuck)
        trainer.train_lof = self.fake_family(trainer, "LOF", 1)

        t0 = time.monotonic()
        trainer.train_models()
        self.assertLess(time.monotonic() - t0, 15)

        self.assertIn("exceeded the 1s training limit", trainer.failed["SOS_1"])
        for name in ("SOS_2", "SOS_3"):
            self.assertIn("not attempted, because SOS_1", trainer.failed[name])
        self.assertNotIn("SOS_4", trainer.failed, "an unrequested instance was reported")
        self.assertTrue(os.path.exists(self.pth("LOF_1")))
        line = next(w for w in self.warnings if "SOS_1 failed" in w)
        self.assertEqual(markers.classify(line)["code"], "training_failed")
        _no_children_left(self)

    def test_a_timeout_does_not_take_a_sibling_already_on_disk_with_it(self):
        """--overwrite false, LOF_2 trained by an earlier run, LOF_1 stuck.

        LOF_2 is used as it is, as it would be without the timeout; only LOF_3,
        which this run would actually have trained, is not attempted. Naming
        LOF_2 as failed would drop a valid checkpoint, since app.py loads
        nothing in `failed`, and could sink the run below MIN_DETECTORS."""
        before = self.save_checkpoint("LOF_2")
        trainer = self.trainer(["LOF"], ["LOF_1", "LOF_2", "LOF_3"],
                               overwrite=False, timeout=1)

        def first_hangs(name):
            if name == "LOF_1":
                _stuck_in_c()
            return _Tiny()
        trainer.train_lof = self.fake_family(trainer, "LOF", 3, first_hangs)

        trainer.train_models()

        self.assertIn("exceeded the 1s training limit", trainer.failed["LOF_1"])
        self.assertNotIn("LOF_2", trainer.failed)
        self.assertIn("not attempted, because LOF_1", trainer.failed["LOF_3"])
        self.assertFalse(any("LOF_2" in w and "failed to train" in w
                             for w in self.warnings), self.warnings)
        with open(self.pth("LOF_2"), "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertFalse(os.path.exists(self.pth("LOF_3")))
        _no_children_left(self)

    def test_an_instance_killed_mid_write_leaves_no_checkpoint(self):
        trainer = self.trainer(["LOF"], ["LOF_1"], timeout=1)
        trainer.train_lof = self.fake_family(trainer, "LOF", 1,
                                             lambda name: _HangsWhilePickled())
        trainer.train_models()
        self.assertIn("exceeded the 1s training limit", trainer.failed["LOF_1"])
        self.assertFalse(os.path.exists(self.pth("LOF_1")))
        self.assertFalse(os.path.exists(self.pth("LOF_1") + PARTIAL_SUFFIX),
                         "the killed write's .partial was not cleaned up")

    def test_with_cuda_initialised_it_trains_in_process_and_says_so(self):
        """A forked child cannot use CUDA once the parent has initialised it."""
        trainer = self.trainer(["LOF"], ["LOF_1"])
        trainer.train_lof = self.fake_family(trainer, "LOF", 1)
        with mock.patch.object(t.cuda, "is_initialized", return_value=True), \
                mock.patch.object(train_mod, "run_with_deadline",
                                  side_effect=AssertionError("forked anyway")):
            trainer.train_models()
        self.assertEqual(trainer.failed, {})
        self.assertTrue(os.path.exists(self.pth("LOF_1")))
        self.assertTrue(any("cannot be enforced" in w and "CUDA" in w for w in self.warnings))

    def test_checking_for_cuda_does_not_initialise_it(self):
        """`torch.cuda.is_available()` runs cuInit, which is what poisons a
        fork; the check that guards the fork must not be the thing that does."""
        from Model_Training import deadline
        with mock.patch.object(t.cuda, "is_available",
                               side_effect=AssertionError("is_available() was called")):
            deadline.fork_unsafe_reason()
        self.assertFalse(t.cuda.is_initialized())


@unittest.skipIf(train_mod is None, "torch/numpy/loguru not installed in this interpreter")
class TestSosSizeCheck(unittest.TestCase):

    def test_the_limit_itself_is_allowed(self):
        from Algorithms.sos import SOS_MAX_ROWS, check_tractable
        check_tractable(SOS_MAX_ROWS, "training")

    def test_a_full_length_smd_machine_is_refused_with_the_cost_and_the_way_out(self):
        from Algorithms.sos import check_tractable
        with self.assertRaises(ValueError) as ctx:
            check_tractable(28479, "training")
        message = str(ctx.exception)
        self.assertIn("about 26 GB", message)
        self.assertIn("28,479 training rows", message)
        self.assertIn("downsampling", message)


class _StopAtStage4(BaseException):
    """Raised from the stage-4 call. A BaseException, so run_app's
    `except Exception` lets it through to the test instead of logging it."""


def _draws():
    """One draw from each global RNG the pipeline uses; equal draws, equal state."""
    import random
    return (round(t.rand(1).item(), 12), round(float(np.random.rand()), 12),
            round(random.random(), 12))


@unittest.skipIf(train_mod is None, "torch/numpy/loguru not installed in this interpreter")
class TestRunAppSeeding(unittest.TestCase):
    """app.py's stage 3 -> 4 boundary, through the real `run_app` and the real
    `TrainModels`, with data, the family method and stage 4 replaced.

    The fake family does to the global RNGs what the DL families really do in
    the process they train in — torch.manual_seed(1) in the model constructor,
    np.random.seed in the Trainer — and then draws. In-process that lands in
    run_app's own RNGs; forked it does not. Stages 4-6 must not be able to tell.
    """

    SEED = 13

    @classmethod
    def setUpClass(cls):
        try:
            import app
        except Exception as e:                              # pragma: no cover
            raise unittest.SkipTest(f"app.py does not import here: {e}")
        cls.app = app

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.draw_dir = os.path.join(self.tmp, "draws")
        os.makedirs(self.draw_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, timeout, overwrite):
        return {
            "dataset": "d", "entity": "e", "detectors": ["LOF_1", "LOF_2"],
            "stages": ["ga"], "random_seed": self.SEED,
            "training_timeout": timeout, "overwrite": overwrite,
            "trained_model_path": os.path.join(self.tmp, "models"),
            "dataset_path": self.tmp, "downsampling": None, "min_length": 256,
            "training_size": 1, "verbose": False, "model_architectures": "all",
        }

    def run_to_stage_4(self, timeout, overwrite=True, entropy=0):
        """run_app as far as stage 4; returns what stage 4 drew, and what each
        trained instance drew on entry, in this process or the child."""
        # The state a fresh interpreter starts from: OS entropy, different
        # every run. Set here to a different value each call.
        np.random.seed(1000 + entropy)
        t.manual_seed(2000 + entropy)
        import random
        random.seed(3000 + entropy)
        draw_dir = self.draw_dir

        def build(**kw):
            with mock.patch.object(train_mod, "load_data", lambda **_: _dataset()):
                trainer = train_mod.TrainModels(**kw)

            def train_lof():
                for model_id in (1, 2):
                    name = f"LOF_{model_id}"
                    if not trainer._should_train(name):
                        continue
                    with open(os.path.join(draw_dir, name), "w") as f:
                        f.write(repr(_draws()))
                    t.manual_seed(1)
                    np.random.seed(1)
                    _draws()
                    trainer.logging_obj.save(obj=_Tiny(), obj_name=name, obj_meta={},
                                             obj_class=trainer.logging_hierarchy)
            trainer.train_lof = train_lof
            return trainer

        seen = {}

        def stage_4(data, *a, **kw):
            seen["stage_4"] = _draws()
            raise _StopAtStage4()

        app = self.app
        # The load-time compatibility probe scores each checkpoint on the data,
        # and `_Tiny` is a stand-in that cannot score anything — so it would be
        # excluded as incompatible and the run would stop before stage 4. What
        # is under test here is the RNG state stage 4 starts from, not
        # checkpoint compatibility (Utils/test_checkpoint_probe covers that).
        with mock.patch.object(app, "get_args_from_cmdline",
                               lambda: self.args(timeout, overwrite)), \
                mock.patch.object(app, "probe_detector", lambda *a, **k: None), \
                mock.patch.object(app, "load_data", lambda **_: _dataset()), \
                mock.patch.object(app, "TrainModels", build), \
                mock.patch.object(app, "Inject", stage_4), \
                self.assertRaises(_StopAtStage4):
            app.run_app(app.algorithm_list, app.algorithm_list_instances)
        for name in os.listdir(draw_dir):
            with open(os.path.join(draw_dir, name)) as f:
                seen[name] = f.read()
            os.remove(os.path.join(draw_dir, name))
        return seen

    def seeded_draws(self):
        from Utils.set_all_seeds import set_all_seeds
        set_all_seeds(self.SEED)
        return _draws()

    def test_stage_4_starts_from_the_seed_however_stage_3_ran(self):
        in_process = self.run_to_stage_4(timeout=0, entropy=1)
        forked = self.run_to_stage_4(timeout=60, entropy=2)
        cached = self.run_to_stage_4(timeout=60, overwrite=False, entropy=3)
        self.assertEqual(set(in_process) - {"stage_4"}, {"LOF_1", "LOF_2"})
        self.assertEqual(set(forked) - {"stage_4"}, {"LOF_1", "LOF_2"})
        self.assertEqual(set(cached), {"stage_4"}, "the cached run trained after all")
        expected = self.seeded_draws()
        self.assertEqual(in_process["stage_4"], expected)
        self.assertEqual(forked["stage_4"], expected)
        self.assertEqual(cached["stage_4"], expected)

    def test_a_forked_instance_trains_from_the_seed_not_from_os_entropy(self):
        """Each child inherits the RNGs at the fork. Unseeded, CBLOF's KMeans
        fitted different centres on every run; seeded, every instance starts
        from the configured seed, whatever the interpreter started from."""
        first = self.run_to_stage_4(timeout=60, entropy=1)
        second = self.run_to_stage_4(timeout=60, entropy=2)
        # torch and numpy only. CPython re-seeds the stdlib `random` from OS
        # entropy in every forked child (`os.register_at_fork` in random.py),
        # so no seed set before a fork reaches it there; MTAD-GAT, the only
        # training code that uses it, seeds it itself.
        expected = self.seeded_draws()[:2]
        for name in ("LOF_1", "LOF_2"):
            self.assertEqual(ast.literal_eval(first[name])[:2], expected, name)
            self.assertEqual(ast.literal_eval(second[name])[:2], expected, name)

    def test_seeding_does_not_initialise_cuda(self):
        """It runs before the trainer forks, and a parent that has initialised
        CUDA cannot hand a usable one to its children."""
        with mock.patch.object(t.cuda, "is_available",
                               side_effect=AssertionError("is_available() was called")):
            self.seeded_draws()
        self.assertFalse(t.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
