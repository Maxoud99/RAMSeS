"""The one way to read a detector checkpoint back.

`Utils.logger.Logger.save_torch_model` writes whole model objects with `dill`
rather than the stdlib pickle `torch.save` defaults to. That is not a stylistic
preference: PyOD 3 defines LSTMAD's network as a class inside a function
(`_LSTMModel.__init__.<locals>._Net`), pickle can only serialise classes
reachable at module level, and the raise landed *after* `fit` and after the
diagnostic plot — leaving a truncated `.pth` on disk that then read as a trained
model. No LSTMAD checkpoint has ever existed on any entity because of it.

A load must use the same pickler as the save, and there are three load sites in
this repo (`app.py`, `Model_Selection/model_selection.py` and the Flask
`Services/mmodel` inspector). Three copies of one rule is how the batch-size
rule drifted and cost LSTMAD its training; this module exists so there is one
copy. `Utils/test_pipeline_spec` asserts no load site calls `torch.load`
directly.

Reading old checkpoints still works. `dill` deserialises stdlib-pickle streams —
verified against the checkpoints already on disk — so nothing written before
this change needs regenerating.

`weights_only=False` is required and stated once here rather than at each call
site: these files hold whole estimator objects, not state dicts, so the safe
loader cannot read them. It is also the value torch is migrating away from as a
default, and pinning it in one place is what keeps that migration from silently
breaking two of the three sites.
"""

import dill
import torch as t


def load_checkpoint(f, map_location=None):
    """Load a detector checkpoint from an open binary file or a path.

    `map_location='cpu'` is worth passing wherever a checkpoint may have been
    trained on a GPU machine: LSTMVAE and DGHL used to pickle `device='cuda'`
    as state and allocated CUDA tensors on a CPU-only host.
    """
    return t.load(f, pickle_module=dill, weights_only=False,
                  map_location=map_location)
