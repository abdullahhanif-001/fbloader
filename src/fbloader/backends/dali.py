"""NVIDIA DALI backend. Linux + CUDA only. Lazy import. Never used for s3/http."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Iterator

log = logging.getLogger("fbloader")

_DALI_LINUX_ONLY = "NVIDIA DALI is Linux+CUDA only. On Windows use WSL; macOS is not supported."


def dali_available() -> bool:
    if sys.platform != "linux":
        return False
    try:
        import nvidia.dali  # noqa: F401
    except Exception:
        return False
    return True


def require_dali_platform() -> None:
    if sys.platform != "linux":
        raise RuntimeError(_DALI_LINUX_ONLY)


class DaliLoader:
    """Iterator wrapping DALIGenericIterator. Clones GPU tensors for hold-across-next."""

    def __init__(
        self,
        source: str,
        *,
        batch_size: int,
        num_threads: int,
        crop: int | None,
        shuffle: bool,
        drop_last: bool,
        device_id: int,
        shard_id: int,
        num_shards: int,
        seed: int | None,
        steps: int | None,
        length: int | None,
    ) -> None:
        require_dali_platform()
        try:
            from nvidia.dali import fn, pipeline_def, types
            from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        except ImportError as exc:
            raise RuntimeError(
                "nvidia.dali not installed. On Linux+CUDA: "
                "python -m pip install nvidia-dali-cuda120 --extra-index-url https://pypi.nvidia.com"
            ) from exc

        file_root = os.path.abspath(source)
        if not os.path.isdir(file_root):
            raise ValueError(f"DALI file reader needs a local directory, got {source!r}")

        crop_hw = int(crop or 224)
        seed_v = 0 if seed is None else int(seed)

        @pipeline_def(
            batch_size=batch_size,
            num_threads=max(1, num_threads),
            device_id=device_id,
            seed=seed_v,
            prefetch_queue_depth=2,
            py_num_workers=0,
        )
        def _pipe():
            jpegs, labels = fn.readers.file(
                file_root=file_root,
                random_shuffle=shuffle,
                shard_id=shard_id,
                num_shards=num_shards,
                name="main",
                seed=seed_v,
            )
            # mixed: CPU reads, GPU nvJPEG decode. Do not crop_mirror_normalize
            # (that would fp32 on GPU inside DALI; we still clone outputs).
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(images, resize_x=crop_hw, resize_y=crop_hw)
            images = fn.cast(images, dtype=types.UINT8)
            return images, labels

        pipe = _pipe()
        policy = LastBatchPolicy.DROP if drop_last else LastBatchPolicy.PARTIAL
        self._it = DALIGenericIterator(
            [pipe],
            ["data", "label"],
            reader_name="main",
            last_batch_policy=policy,
            auto_reset=True,
        )
        self._steps = steps
        self._length = length
        self._n = 0
        self._closed = False

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        self._n = 0
        return self

    def __next__(self) -> tuple[Any, Any]:
        if self._steps is not None and self._n >= self._steps:
            raise StopIteration
        try:
            out = next(self._it)
        except StopIteration:
            self._n = 0
            raise
        from fbloader._torch import require_torch

        torch = require_torch()
        data = out[0]["data"].detach().clone()
        label = out[0]["label"].detach().clone().reshape(-1).to(torch.int64)
        if data.dtype != torch.uint8:
            data = data.to(torch.uint8)
        if data.ndim == 4 and data.shape[-1] in (1, 3):
            data = data.permute(0, 3, 1, 2).contiguous()
        self._n += 1
        return data, label

    def __len__(self) -> int:
        if self._length is not None:
            return int(self._length)
        if self._steps is not None:
            return int(self._steps)
        return len(self._it)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        reset = getattr(self._it, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
        try:
            del self._it
        except Exception:
            pass


def try_build_dali(**kwargs: Any) -> DaliLoader | None:
    if not dali_available():
        log.info("DALI unavailable; using webdataset/torch.")
        return None
    try:
        return DaliLoader(**kwargs)
    except Exception:
        log.exception("DALI pipeline failed; falling back.")
        return None
