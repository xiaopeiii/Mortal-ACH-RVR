import sys
import warnings
from typing import Optional, Union

import torch


def _is_cuda_device(device: Optional[Union[torch.device, str]]) -> bool:
    if device is None:
        return torch.cuda.is_available()
    try:
        d = device if isinstance(device, torch.device) else torch.device(device)
        return d.type == "cuda"
    except Exception:
        return False


def _has_working_triton() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def maybe_compile(
    module: torch.nn.Module,
    *,
    enable: bool,
    device: Optional[Union[torch.device, str]] = None,
    label: str = "model",
) -> torch.nn.Module:
    """Compile module if possible; otherwise fall back to eager with a warning.

    PyTorch's default torch.compile backend (inductor) requires Triton on CUDA.
    On Windows, CUDA inductor/triton support is commonly missing or unreliable.
    """
    if not enable:
        return module

    if not hasattr(torch, "compile") and not hasattr(module, "compile"):
        warnings.warn(f"{label}: torch.compile is not available in this PyTorch; skipping.")
        return module

    if _is_cuda_device(device):
        if sys.platform.startswith("win"):
            warnings.warn(
                f"{label}: torch.compile on CUDA is not supported/reliable on Windows; skipping."
            )
            return module
        if not _has_working_triton():
            warnings.warn(
                f"{label}: Triton is not available; skipping torch.compile. "
                "Install a working triton to enable compilation."
            )
            return module

    # If compilation fails later (often at first call), don't crash the whole run.
    try:
        import torch._dynamo as torch_dynamo  # type: ignore

        torch_dynamo.config.suppress_errors = True
    except Exception:
        pass

    try:
        # Prefer Module.compile() to preserve the original object when supported.
        if hasattr(module, "compile"):
            module.compile()  # type: ignore[attr-defined]
            return module
        return torch.compile(module)  # type: ignore[no-any-return]
    except Exception as exc:
        warnings.warn(f"{label}: torch.compile failed, falling back to eager. Error: {exc}")
        return module
