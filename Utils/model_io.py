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
directly, and that each one goes through `load_detector`.

Reading old checkpoints still works. `dill` deserialises stdlib-pickle streams —
verified against the checkpoints already on disk — so nothing written before
this change needs regenerating.

`weights_only=False` is required and stated once here rather than at each call
site: these files hold whole estimator objects, not state dicts, so the safe
loader cannot read them. It is also the value torch is migrating away from as a
default, and pinning it in one place is what keeps that migration from silently
breaking two of the three sites.

Device placement is the second rule every load site has to share, and it is the
one that had drifted. A checkpoint is a whole detector, so where it runs is
recorded twice: in the storages its tensors were saved from, and in plain state
— MD, LSTMVAE and DGHL pickle `self.device`, PyOD's deep detectors keep a
`device` beside an `nn.Module` that is not registered on our wrapper, and the
vendored TSB-AD models pick `cuda:{least-used GPU}` at construction. `app.py`
loaded every checkpoint with `map_location='cpu'` and then repaired the device
attribute only when CUDA was ABSENT. On a GPU host that left the weights on the
CPU while `model.device` still said `cuda`, so the first forward pass moved its
input to the GPU and multiplied it with CPU weights: "Expected all tensors to be
on the same device". The other two sites did not repair anything. Both halves
now have one owner: `load_detector` always deserialises onto the CPU (which is
what makes a `cuda:1` checkpoint readable on a one-GPU or no-GPU host at all),
and `place_detector` then moves every tensor, module and device attribute it can
reach to ONE device, chosen by `choose_device`.
"""

import types

import dill
import torch as t


def load_checkpoint(f, map_location=None):
    """Load a detector checkpoint from an open binary file or a path.

    The raw read, with no placement. Load sites want `load_detector`, which
    calls this and then puts the whole detector on one device.
    """
    return t.load(f, pickle_module=dill, weights_only=False,
                  map_location=map_location)


def load_detector(f, device=None):
    """Load a detector checkpoint and place all of it on one device.

    `device=None` lets `choose_device` decide from what the checkpoint itself
    records; pass a device to override it (an inspector that only prints a
    model has no reason to touch a GPU).
    """
    trained_on = set()

    def _deserialise_on_cpu(storage, location):
        # torch calls this once per stored tensor with the device it was saved
        # from, and returning the CPU storage unchanged is exactly what
        # map_location='cpu' does. Going through the CPU first is not caution:
        # restoring straight to the saved device raises on any host where that
        # device does not exist — every GPU checkpoint on a CPU box, and a
        # `cuda:1` checkpoint on a one-GPU box. The location is kept because it
        # is the one record of CUDA training that survives a checkpoint whose
        # classes keep no `device` attribute.
        trained_on.add(str(location))
        return storage

    model = load_checkpoint(f, map_location=_deserialise_on_cpu)
    place_detector(model, device=device, trained_on=trained_on)
    try:
        model.eval()
    except AttributeError:
        # Not an nn.Module (a bare estimator); there is no train/eval mode.
        pass
    return model


def choose_device(trained_on=()):
    """The device a checkpoint recorded as living on `trained_on` runs on here.

    CUDA only when this host has it AND the checkpoint was trained there: a
    CPU-trained detector stays on the CPU even on a GPU box, because nothing in
    it was written to expect a GPU, and a CUDA-trained one falls back to the
    CPU wherever there is no GPU. The saved index is kept when this host has
    it, so a multi-GPU box restores what torch itself would; when it does not
    (`cuda:1` on a one-GPU box) the current device takes its place instead of
    the load failing.
    """
    cuda = []
    for location in trained_on:
        try:
            d = t.device(location)
        except (RuntimeError, TypeError, ValueError):
            continue
        if d.type == 'cuda':
            cuda.append(d)
    if not cuda or not t.cuda.is_available():
        return t.device('cpu')
    count = t.cuda.device_count()
    for index in sorted({d.index for d in cuda if d.index is not None}):
        if index < count:
            return t.device('cuda', index)
    return t.device('cuda', t.cuda.current_device())


def place_detector(model, device=None, trained_on=()):
    """Move everything in `model` that has a device onto one device.

    That means: every `nn.Module` reachable from it, registered or not (PyOD
    keeps its network on a plain estimator, AnomalyTransformer keeps blocks in
    a plain list); every tensor held as a plain attribute or inside a dict,
    list or tuple, which `Module.to` never sees; and every attribute that
    names a device, because the forward passes move their input to
    `self.device` and that has to agree with where the weights are. A
    detector with none of these — the sklearn- and PyOD-based classical ones —
    comes out untouched: nothing in it is rewritten, only walked.

    `device=None` asks `choose_device`, fed by `trained_on` (storage locations
    from the load) together with every device attribute found here. Returns
    the device used.
    """
    modules, slots, recorded = _scan(model)
    if device is None:
        target = choose_device(set(trained_on) | recorded)
    else:
        target = t.device(device)
    for module in modules:
        _to(module, target)
    for container, key, kind in slots:
        value = container[key]
        if kind == 'device':
            # Keep the type it had: some code compares `self.device == 'cpu'`,
            # which is False for a torch.device.
            container[key] = target if isinstance(value, t.device) else str(target)
        elif kind == 'tensor':
            container[key] = _moved_tensor(value, target)
        else:
            container[key] = _moved_tuple(value, target)
    return target


def _to(obj, device):
    """The single point where something actually changes device."""
    return obj.to(device)


def _moved_tensor(value, target):
    moved = _to(value, target)
    # `Parameter.to(other_device)` returns a plain Tensor, which would quietly
    # drop it out of anything that looks for parameters by type.
    if isinstance(value, t.nn.Parameter) and not isinstance(moved, t.nn.Parameter):
        moved = t.nn.Parameter(moved, requires_grad=value.requires_grad)
    return moved


def _moved_tuple(value, target):
    items = [_moved_tensor(x, target) if isinstance(x, t.Tensor)
             else target if isinstance(x, t.device)
             else _moved_tuple(x, target) if isinstance(x, tuple)
             else x
             for x in value]
    # A namedtuple takes its fields positionally; a plain tuple takes one iterable.
    return type(value)(*items) if hasattr(value, '_fields') else type(value)(items)


# Attribute names under which a STRING means a device. A torch.device value is
# recognised under any name; a string only here, so that a label that happens
# to read "cpu" is never rewritten.
_DEVICE_NAMES = frozenset({'device', '_device', 'device_'})

# Objects that have a __dict__ but can hold no detector state of their own.
_OPAQUE = (type, types.ModuleType, types.FunctionType,
           types.BuiltinFunctionType, types.MethodType)

# `Module.to` already moves these, and they must not be rebound one by one.
_MOVED_BY_MODULE_TO = frozenset({'_parameters', '_buffers'})


def _is_device(name, value):
    if isinstance(value, t.device):
        return True
    if name in _DEVICE_NAMES and isinstance(value, str):
        try:
            t.device(value)
        except (RuntimeError, TypeError, ValueError):
            return False
        return True
    return False


def _is_module(value):
    """An nn.Module whose own __init__ ran, so `.to` can work on it.

    `Algorithms.nearest_neighbors.NearestNeighbors` subclasses PyMADModel and
    never calls `super().__init__()`, so it has no `_modules` and `.to` (or
    `.eval`) raises AttributeError on it. It holds no registered tensors to
    move, so it is walked like any plain object instead.
    """
    return isinstance(value, t.nn.Module) and '_parameters' in vars(value)


def _holds_placeable(value):
    return any(isinstance(x, (t.Tensor, t.device))
               or (isinstance(x, tuple) and _holds_placeable(x))
               for x in value)


def _walkable(value):
    if isinstance(value, (dict, list, tuple)):
        return True
    if isinstance(value, _OPAQUE):
        return False
    try:
        # A class with __slots__ and a __getattr__ is asked for '__dict__' by
        # name, and such proxies do not always answer with AttributeError.
        return isinstance(getattr(value, '__dict__', None), dict)
    except Exception:
        return False


def _children(obj):
    """(container, key, value, registered) for everything directly inside obj.

    `container[key]` is where a replacement goes: the object's own __dict__
    for attributes, so nn.Module's __setattr__ bookkeeping is bypassed for
    names it never registered. Tuple elements get no container — the tuple is
    rebuilt from its parent instead.
    """
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            yield obj, key, value, False
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield obj, i, value, False
    elif isinstance(obj, tuple):
        for i, value in enumerate(obj):
            yield None, i, value, False
    else:
        attrs = vars(obj)
        is_module = _is_module(obj)
        for key, value in list(attrs.items()):
            if is_module and key in _MOVED_BY_MODULE_TO:
                continue
            if is_module and key == '_modules':
                # Registered children move with their parent's `.to`, but are
                # still walked for device attributes and plain tensors.
                for name, child in value.items():
                    if child is not None:
                        yield value, name, child, True
                continue
            yield attrs, key, value, False


def _scan(root):
    """What `place_detector` has to touch, and every device recorded on the way.

    Iterative with an identity set: module trees are deep and shared
    references are common (a wrapper and its estimator both pointing at one
    network), so plain recursion would either overflow or visit twice.
    """
    modules = [root] if _is_module(root) else []
    slots, recorded = [], set()
    seen = {id(root)}
    stack = [root] if _walkable(root) else []
    while stack:
        obj = stack.pop()
        for container, key, value, registered in _children(obj):
            if isinstance(value, t.Tensor):
                if container is not None:
                    slots.append((container, key, 'tensor'))
                continue
            if _is_device(key, value):
                recorded.add(str(value))
                if container is not None:
                    slots.append((container, key, 'device'))
                continue
            if isinstance(value, tuple) and container is not None and _holds_placeable(value):
                slots.append((container, key, 'tuple'))
                recorded.update(str(x) for x in value if isinstance(x, t.device))
            if id(value) in seen or not _walkable(value):
                continue
            seen.add(id(value))
            if _is_module(value) and not registered:
                modules.append(value)
            stack.append(value)
    return modules, slots, recorded
