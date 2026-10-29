"""m3env.py — import mamba_ssm's Python/Triton modules from a source snapshot without building the package.

mamba_ssm/__init__.py imports compiled CUDA extensions (selective_scan, causal_conv1d) that are not needed for
Mamba-3 SISO and do not build on Windows. We register an empty package object whose __path__ points at the
snapshot, so `import mamba_ssm.ops.triton.mamba3...` and `mamba_ssm.modules.mamba3` load directly.
"""
import os
import sys
import types


def use_source(root):
    root = os.path.abspath(root)
    pkg_dir = os.path.join(root, "mamba_ssm")
    assert os.path.isdir(pkg_dir), pkg_dir
    for k in [k for k in sys.modules if k == "mamba_ssm" or k.startswith("mamba_ssm.")]:
        del sys.modules[k]
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = [pkg_dir]
    pkg.__file__ = os.path.join(pkg_dir, "__init__.py")
    pkg.__version__ = "snapshot"
    sys.modules["mamba_ssm"] = pkg
    return root


def environment():
    import platform
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["torch_cuda"] = torch.version.cuda
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception as e:
        info["triton_error"] = repr(e)
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception as e:
        info["numpy_error"] = repr(e)
    return info
