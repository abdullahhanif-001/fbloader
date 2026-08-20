"""Tensor lifetime, GPU normalize, leak probes. No recycled views."""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Any, Iterator

from fbloader._torch import require_torch

log = logging.getLogger("fbloader")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resource_count() -> int:
    """Open FDs on Linux; process handle count on Windows; -1 if unknown."""
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        count = ctypes.c_ulong()
        if kernel32.GetProcessHandleCount(handle, ctypes.byref(count)):
            return int(count.value)
        return -1
    fd_dir = f"/proc/{os.getpid()}/fd"
    if os.path.isdir(fd_dir):
        return len(os.listdir(fd_dir))
    return -1


def leakcheck_enabled() -> bool:
    return os.environ.get("FBLOADER_LEAKCHECK", "") == "1"


def gpu_normalize(
    images: Any,
    device: Any | None = None,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    non_blocking: bool = True,
) -> Any:
    """uint8 NCHW host tensor -> fp32 on device. Does not mutate ``images``."""
    torch = require_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        non_blocking = bool(torch.cuda.is_available()) and non_blocking
    else:
        device = torch.device(device)
        if device.type != "cuda":
            non_blocking = False
    out = images.to(device, non_blocking=non_blocking)
    out = out.float().div_(255.0)
    m = torch.tensor(mean, dtype=out.dtype, device=out.device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=out.dtype, device=out.device).view(1, 3, 1, 1)
    out.sub_(m).div_(s)
    return out


class OwnStorageIterator:
    """Yield independent tensor storage so later ``next()`` cannot mutate held batches.

    WebDataset/PIL paths already copy in the pipeline. DALI GPU tensors are cloned
    (or fenced) here so a caller holding ``batch`` across ``next()`` stays valid.
    """

    def __init__(self, inner: Iterator[Any], *, clone: bool) -> None:
        self._inner = inner
        self._clone = clone
        self._n = 0
        self._start_res = resource_count() if leakcheck_enabled() else -1

    def __iter__(self) -> OwnStorageIterator:
        return self

    def __next__(self) -> Any:
        batch = next(self._inner)
        if self._clone:
            batch = _clone_batch(batch)
        self._n += 1
        if leakcheck_enabled() and self._n % 256 == 0:
            now = resource_count()
            log.info("fbloader leakcheck step=%s resources=%s start=%s", self._n, now, self._start_res)
        return batch

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()


def _clone_batch(batch: Any) -> Any:
    torch = require_torch()
    if isinstance(batch, torch.Tensor):
        return batch.detach().clone()
    if isinstance(batch, (tuple, list)):
        seq = [_clone_batch(x) for x in batch]
        return type(batch)(seq) if not isinstance(batch, tuple) else tuple(seq)
    if isinstance(batch, dict):
        return {k: _clone_batch(v) for k, v in batch.items()}
    return batch


class CudaEventFenceIterator:
    """Two-deep GPU buffer: recycle only after event + previous Python refs dropped."""

    def __init__(self, inner: Iterator[Any]) -> None:
        torch = require_torch()
        if not torch.cuda.is_available():
            self._inner = OwnStorageIterator(inner, clone=True)
            self._use_events = False
            return
        self._inner = inner
        self._use_events = True
        self._event = torch.cuda.Event()
        self._prev: Any = None
        self._n = 0

    def __iter__(self) -> CudaEventFenceIterator:
        return self

    def __next__(self) -> Any:
        if not self._use_events:
            return next(self._inner)
        torch = require_torch()
        if self._prev is not None:
            self._event.synchronize()
            self._prev = None
        batch = _clone_batch(next(self._inner))
        self._event.record(torch.cuda.current_stream())
        self._prev = batch
        self._n += 1
        return batch

    def close(self) -> None:
        if self._use_events:
            try:
                self._event.synchronize()
            except Exception:
                pass
            self._prev = None
            closer = getattr(self._inner, "close", None)
            if callable(closer):
                closer()
        else:
            self._inner.close()
