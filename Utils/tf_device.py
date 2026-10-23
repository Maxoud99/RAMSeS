"""Keep TensorFlow off the GPU unless the environment says it is safe.

TensorFlow is used by one stage only, the GAN robustness test
(`Model_Selection/Sensitivity_robustness/GAN_test.py`, plus the currently
unused `Adversarial_testing.py`). It shares that process with torch, which
every detector runs on, and the two are built against different cuDNN
releases:

  * torch 2.5.1 carries cuDNN 9.1.0. The conda build `environment.full.yml`
    pins (pytorch 2.5.1 `py3.11_cuda11.8_cudnn9.1.0`) bundles it as
    `torch/lib/libcudnn.so.9`, and the PyPI wheel `requirements.txt` installs
    pins `nvidia-cudnn-cu12==9.1.0.70`.
  * tensorflow 2.18.1 is compiled against CUDA 12.5 and cuDNN 9.3.0. Its
    `[and-cuda]` extra pins `nvidia-cudnn-cu12==9.3.0.75`. When it first
    initialises cuDNN on a GPU, it requires the same major version and an
    equal or higher minor version (`IsSourceCompatibleWithCudnnLibrary` in
    XLA's cuda_dnn.cc), so it rejects 9.1 with "Loaded runtime CuDNN
    library: 9.1.0 but source was compiled with: 9.3.0".

app.py imports torch before anything imports TensorFlow, and `import torch`
maps its libcudnn.so.9 into the process straight away, even on a machine with
no GPU. When TensorFlow later asks for libcudnn.so.9, the dynamic loader
returns the copy that is already loaded and ignores TensorFlow's own search
path. So on a GPU host where TensorFlow's CUDA 12 libraries resolve, and
TensorFlow therefore registers the GPU, the GAN stage fails. Installing
`tensorflow[and-cuda]` does not fix it: pip cannot satisfy both exact
`nvidia-cudnn-cu12` pins, and the 9.3 library would still lose to the one
torch already loaded. `tensorflow-cpu` never sees a GPU and is not affected.

Hiding the GPUs from TensorFlow alone fixes this at little cost. The GAN is
two small dense networks (one 256-unit hidden layer each). On SKAB entity 1
the whole stage, 100 training epochs plus scoring four detectors, took 92 s on
a shared CPU-only server. `tf.config.set_visible_devices` changes TensorFlow's
device list and nothing else, so torch keeps its GPU. CUDA_VISIBLE_DEVICES
would hide the GPU from the detectors too.

An environment whose TensorFlow CUDA stack does match can opt back in with
RAMSES_TF_GPU=1. That means a separate TensorFlow install, or a torch built
against cuDNN 9.3 or later.
"""

import os

from loguru import logger

# The opt-in. Read from the environment rather than a CLI flag: it describes
# the installation, not the run, and the web UI spawns app.py with the server's
# environment, so one export covers both entry points.
TF_GPU_ENV = "RAMSES_TF_GPU"
_TRUTHY = {"1", "true", "yes", "on"}


def tf_gpu_opted_in(environ=None) -> bool:
    """True when RAMSES_TF_GPU asks for TensorFlow to keep its GPUs."""
    value = (os.environ if environ is None else environ).get(TF_GPU_ENV, "")
    return str(value).strip().lower() in _TRUTHY


def keep_tensorflow_off_gpu(tf, environ=None) -> bool:
    """Hide every GPU from TensorFlow unless RAMSES_TF_GPU opts in.

    Call it straight after `import tensorflow`, before any op runs. TensorFlow
    fixes its device list when its runtime first initialises and raises
    RuntimeError on any later change. Listing the physical devices does not
    initialise the runtime, so asking first is safe.

    Returns True when TensorFlow is left without GPUs and False when they stay
    visible: either the user opted in, or something ran a TensorFlow op before
    this call and the device list could no longer be changed.
    """
    if tf_gpu_opted_in(environ):
        logger.info(f"{TF_GPU_ENV} is set: TensorFlow may use the GPU for the "
                    f"GAN robustness stage.")
        return False
    try:
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as e:
        # Never fatal. This only means some code ran TensorFlow before the GAN
        # module was imported. The GAN stage fails soft in app.py if the GPU
        # stack then breaks, so a run still finishes.
        logger.warning(f"Could not keep TensorFlow off the GPU: it was already "
                       f"initialised ({e}). The GAN stage may hit the "
                       f"TensorFlow/torch cuDNN mismatch.")
        return False
    if gpus:
        logger.info(f"TensorFlow runs the GAN robustness stage on the CPU; "
                    f"{len(gpus)} GPU(s) stay with torch. Set {TF_GPU_ENV}=1 "
                    f"if this environment's TensorFlow matches its cuDNN.")
    return True
