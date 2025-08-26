# Small runtime shims for cross-version compatibility.
# - Torch 2.2.x + Transformers ≥4.44: add torch.compiler.is_compiling()
# - Warn if NumPy ≥2 (common ABI mismatch with older wheels)
#
# Usage:
#   from mot_dinov3 import compat
#   compat.apply(strict_numpy=False, quiet=True)

from __future__ import annotations
import sys, os

def apply(strict_numpy: bool = False, quiet: bool = True) -> None:
    _torch_compiler_shim(quiet=quiet)
    _numpy_guard(strict=strict_numpy, quiet=quiet)

def _torch_compiler_shim(quiet: bool) -> None:
    try:
        import torch  # noqa: F401
        comp = getattr(torch, "compiler", None)
        created_ns = False
        if comp is None:
            class _C:  # minimal namespace
                pass
            torch.compiler = _C()  # type: ignore[attr-defined]
            created_ns = True
        if not hasattr(torch.compiler, "is_compiling"):  # type: ignore[attr-defined]
            torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
            if not quiet:
                _note("Applied shim: torch.compiler.is_compiling -> False")
        elif created_ns and not quiet:
            _note("Created torch.compiler namespace for transformers compat")
    except Exception as e:
        if not quiet:
            _note(f"torch shim skipped ({e})")

def _numpy_guard(strict: bool, quiet: bool) -> None:
    try:
        import numpy as np  # noqa: F401
        major = int(np.__version__.split(".")[0])
        if major >= 2:
            msg = (
                f"Detected NumPy {np.__version__}. Some wheels were built against NumPy 1.x.\n"
                "If you hit import errors, pin:\n"
                "    pip install 'numpy<2' 'scipy<1.13'\n"
            )
            if strict:
                raise RuntimeError(msg)
            if not quiet:
                _note(msg.strip())
    except Exception as e:
        if strict:
            raise
        if not quiet:
            _note(f"NumPy guard skipped ({e})")

def _note(msg: str) -> None:
    print(f"[compat] {msg}", file=sys.stderr)

