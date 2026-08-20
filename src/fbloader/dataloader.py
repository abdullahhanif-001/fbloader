"""Drop-in DataLoader facade. Auto: DALI (Linux CUDA local) else webdataset else map."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Iterator

from fbloader._spawn import default_num_workers, default_pin_memory, require_main
from fbloader._torch import require_torch
from fbloader.backends import dali as dali_backend
from fbloader.backends import torch_map as map_backend
from fbloader.backends import wds as wds_backend
from fbloader.lifetime import OwnStorageIterator

log = logging.getLogger("fbloader")


def _is_map_dataset(obj: Any) -> bool:
    if obj is None or isinstance(obj, (str, bytes, os.PathLike)):
        return False
    if isinstance(obj, (list, tuple)):
        return False
    torch = require_torch()
    dataset_t = getattr(torch.utils.data, "Dataset", None)
    iterable_t = getattr(torch.utils.data, "IterableDataset", None)
    if iterable_t is not None and isinstance(obj, iterable_t):
        return False
    if dataset_t is not None and isinstance(obj, dataset_t):
        return True
    return callable(getattr(obj, "__getitem__", None)) and callable(getattr(obj, "__len__", None))


def _is_remote(source: Any) -> bool:
    if isinstance(source, (list, tuple)):
        return any(_is_remote(x) for x in source)
    text = os.fspath(source) if isinstance(source, (str, os.PathLike)) else str(source)
    return wds_backend.is_remote_url(text)


def _world_size() -> int:
    try:
        torch = require_torch()
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


def _rank() -> int:
    try:
        torch = require_torch()
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _cuda_ok() -> bool:
    try:
        return bool(require_torch().cuda.is_available())
    except ImportError:
        return False


def _pick_backend(
    source: Any,
    backend: str,
) -> str:
    if backend != "auto":
        return backend
    if _is_map_dataset(source):
        return "torch"
    if _is_remote(source):
        try:
            from fbloader.backends import streaming as st

            if st.streaming_available() and st.looks_like_mds(source):
                return "streaming"
        except Exception:
            pass
        return "wds"
    if (
        sys.platform == "linux"
        and dali_backend.dali_available()
        and _cuda_ok()
        and isinstance(source, (str, os.PathLike))
        and os.path.isdir(os.fspath(source))
    ):
        return "dali"
    return "wds"


class DataLoader:
    """PyTorch-shaped loader. Batches are uint8 NCHW until ``gpu_normalize``.

    Pass this object once (do not wrap again in torch DataLoader / Lightning).
    """

    def __init__(
        self,
        source: Any,
        batch_size: int = 256,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        drop_last: bool = True,
        shuffle: bool = True,
        persistent_workers: bool = False,
        *,
        backend: str = "auto",
        crop: int | None = 224,
        length: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        collate_fn: Callable[..., Any] | None = None,
        handler: str = "warn",
        device_id: int | None = None,
        prefetch_factor: int = 4,
    ) -> None:
        require_torch()
        if backend not in {"auto", "dali", "wds", "torch", "streaming"}:
            raise ValueError(f"unknown backend {backend!r}")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if num_workers is None:
            num_workers = default_num_workers()
        if int(num_workers) < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers!r}")
        if pin_memory is None:
            pin_memory = default_pin_memory()
        if persistent_workers and num_workers <= 0:
            persistent_workers = False

        require_main(num_workers=num_workers)

        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.crop = crop
        self._length = length
        self._inner: Any = None
        self._iter: Any = None
        self._backend_name = _pick_backend(source, backend)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        if steps is None:
            if length is not None:
                steps = max(1, int(length) // self.batch_size)
            else:
                steps = 1000
        self.steps = int(steps)

        if self._backend_name == "dali":
            if _is_remote(source):
                log.info("Remote source: DALI skipped (single-thread reader). Using webdataset.")
                self._backend_name = "wds"
            elif sys.platform != "linux":
                if backend == "dali":
                    dali_backend.require_dali_platform()
                log.info("DALI skipped on this OS.")
                self._backend_name = "wds"

        if self._backend_name == "dali":
            built = dali_backend.try_build_dali(
                source=os.fspath(source),
                batch_size=self.batch_size,
                num_threads=min(4, max(1, self.num_workers or 1)),
                crop=self.crop,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                device_id=int(device_id if device_id is not None else 0),
                shard_id=_rank(),
                num_shards=_world_size(),
                seed=seed,
                steps=self.steps,
                length=self._length,
            )
            if built is None:
                if backend == "dali":
                    raise RuntimeError("backend='dali' requested but DALI failed to start")
                self._backend_name = "wds"
            else:
                self._inner = built
                return

        if self._backend_name == "streaming":
            from fbloader.backends import streaming as st

            ds = st.build_streaming_dataset(source, batch_size=self.batch_size, shuffle=self.shuffle)
            self._inner = map_backend.build_map_loader(
                ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                shuffle=False,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                collate_fn=collate_fn,
                crop=self.crop,
            )
            return

        if self._backend_name == "torch":
            self._inner = map_backend.build_map_loader(
                source,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                collate_fn=collate_fn,
                crop=self.crop,
            )
            return

        n_shards = wds_backend.shard_count(source)
        wds_backend.maybe_warn_shards(n_shards, self.num_workers, _world_size())
        self.num_workers = wds_backend.cap_workers(self.num_workers, n_shards)
        dataset = wds_backend.build_wds_dataset(
            source,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            crop=self.crop,
            seed=seed,
            steps=self.steps,
            handler=handler,
        )
        torch = require_torch()
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": None,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": bool(persistent_workers) and self.num_workers > 0,
            "worker_init_fn": wds_backend.worker_init_fn,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = max(2, prefetch_factor or 4)
        self._inner = torch.utils.data.DataLoader(**kwargs)

    @property
    def backend(self) -> str:
        return self._backend_name

    def __iter__(self) -> Iterator[Any]:
        self.shutdown()
        raw = iter(self._inner)
        clone = self._backend_name == "dali"
        self._iter = OwnStorageIterator(raw, clone=clone)
        return self._iter

    def __len__(self) -> int:
        if self._length is not None:
            return int(self._length)
        if self._backend_name in {"wds", "dali"}:
            return int(self.steps)
        try:
            return len(self._inner)
        except TypeError:
            return int(self.steps)

    def __enter__(self) -> DataLoader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        it = self._iter
        self._iter = None
        if it is not None:
            closer = getattr(it, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        inner = self._inner
        if inner is None:
            return
        for name in ("shutdown", "close", "_shutdown_workers"):
            fn = getattr(inner, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
