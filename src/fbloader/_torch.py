"""Peer-import torch. Never list torch in install_requires (CUDA wheel clobber)."""

from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "fbload needs PyTorch as a peer dependency. Install the wheel for "
            "this OS from https://pytorch.org (do not let pip pick a random CPU "
            "torch over an existing CUDA install). Python >= 3.10 required."
        ) from exc
    return torch
