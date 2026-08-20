"""WebDataset backend: resampled shards, equal epoch steps, uint8 CHW, skip corrupt JPEG."""

from __future__ import annotations

import io
import json
import logging
import os
from functools import partial
from typing import Any, Iterable

import numpy as np
from braceexpand import braceexpand

from fbloader._torch import require_torch

log = logging.getLogger("fbloader")


def expand_urls(source: str | os.PathLike[str] | Iterable[str]) -> list[str]:
    if isinstance(source, (list, tuple)):
        urls: list[str] = []
        for item in source:
            urls.extend(expand_urls(item))
        return urls
    text = os.fspath(source)
    if is_remote_url(text):
        return list(braceexpand(text))
    if os.path.exists(text):
        return [os.path.abspath(text)]
    if "{" in text:
        return [os.path.abspath(u) if not is_remote_url(u) else u for u in braceexpand(text)]
    return [os.path.abspath(text)]


def is_remote_url(url: str) -> bool:
    text = url.strip()
    lower = text.lower()
    if lower.startswith(("s3://", "gs://", "https://", "pipe:")):
        return True
    if "://" in lower:
        scheme = lower.split("://", 1)[0]
        if scheme and scheme not in {"https", "s3", "gs", "pipe"}:
            raise ValueError(
                "Unsupported or cleartext URL scheme; use HTTPS, s3://, gs://, or pipe:"
            )
    return False


def shard_count(source: str | os.PathLike[str] | Iterable[str]) -> int:
    return max(1, len(expand_urls(source)))


def _proxy_hint(exc: BaseException) -> str:
    msg = str(exc).lower()
    if any(k in msg for k in ("ssl", "certificate", "cert")):
        return " Set SSL_CERT_FILE or REQUESTS_CA_BUNDLE for corporate TLS."
    if any(k in msg for k in ("proxy", "407", "tunnel")):
        return " Set HTTPS_PROXY / HTTP_PROXY."
    if any(k in msg for k in ("403", "401", "access denied", "credentials", "forbidden")):
        return " Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or equivalent) for s3://."
    return ""


def _parse_label(value: Any) -> int:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore").strip()
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return 0
            if isinstance(parsed, dict):
                return int(parsed.get("label", parsed.get("cls", parsed.get("class", 0))))
            return int(parsed)
    return 0


def _pil_to_uint8_chw(image: Any, crop: int | None) -> Any:
    torch = require_torch()
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    else:
        image = image.convert("RGB")
    if crop:
        image = image.resize((int(crop), int(crop)), Image.BILINEAR)
    array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _decode_sample(sample: dict[str, Any], crop: int | None) -> tuple[Any, int]:
    image = None
    label: Any = 0
    for key, val in sample.items():
        if key in {"__key__", "__url__"}:
            continue
        lk = key.lower()
        ext = lk.split(".")[-1]
        if ext in {"jpg", "jpeg", "png", "webp"}:
            image = val
        elif ext in {"cls", "json", "txt"}:
            label = val
    if image is None:
        raise ValueError("sample missing image")
    from PIL import Image, UnidentifiedImageError

    if isinstance(image, Image.Image):
        pil = image
    else:
        raw = image if isinstance(image, (bytes, bytearray)) else bytes(image)
        try:
            pil = Image.open(io.BytesIO(raw))
            pil.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"corrupt jpeg skipped: {exc}") from exc
    return _pil_to_uint8_chw(pil, crop), _parse_label(label)


def _iter_tar_samples(path: str) -> Any:
    import tarfile

    with tarfile.open(path, "r") as tar:
        current: str | None = None
        sample: dict[str, Any] = {}
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            key, ext = os.path.splitext(base)
            ext = ext.lstrip(".").lower()
            if not ext:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            payload = handle.read()
            if current is None:
                current = key
            if key != current:
                if sample:
                    yield sample
                sample = {}
                current = key
            sample[ext] = payload
        if sample:
            yield sample


class LocalTarEpoch(require_torch().utils.data.IterableDataset):
    """Pickle-safe module-level IterableDataset for local .tar shards (Windows spawn)."""

    def __init__(
        self,
        urls: list[str],
        *,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        crop: int | None,
        seed: int | None,
        steps: int,
        skip_bad: bool,
    ) -> None:
        super().__init__()
        self.urls = urls
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.crop = crop
        self.seed = 0 if seed is None else seed
        self.steps = int(steps)
        self.skip_bad = skip_bad

    def __iter__(self) -> Any:
        torch = require_torch()
        worker = torch.utils.data.get_worker_info()
        urls = list(self.urls)
        if worker is not None:
            n_workers = worker.num_workers
            wid = worker.id
            base = self.steps // n_workers
            rem = self.steps % n_workers
            worker_steps = base + (1 if wid < rem else 0)
            urls = urls[wid :: n_workers] or list(self.urls)
        else:
            worker_steps = self.steps
        rng = np.random.default_rng(self.seed + (0 if worker is None else worker.id))
        produced = 0
        buf: list[tuple[Any, int]] = []
        empty_passes = 0
        while produced < worker_steps:
            order = list(urls)
            if self.shuffle:
                rng.shuffle(order)
            got = 0
            for path in order:
                for sample in _iter_tar_samples(path):
                    try:
                        item = _decode_sample(sample, self.crop)
                    except Exception:
                        if self.skip_bad:
                            log.warning("skipping corrupt sample in %s", path)
                            continue
                        raise
                    buf.append(item)
                    got += 1
                    if len(buf) >= self.batch_size:
                        images = torch.stack([x[0] for x in buf], dim=0)
                        labels = torch.tensor([x[1] for x in buf], dtype=torch.int64)
                        buf.clear()
                        yield images, labels
                        produced += 1
                        if produced >= worker_steps:
                            return
            if not order:
                raise RuntimeError("no local shards to read")
            empty_passes = 0 if got else empty_passes + 1
            if empty_passes > 8:
                raise RuntimeError("no valid samples in shards")
        if buf and not self.drop_last and produced < worker_steps:
            images = torch.stack([x[0] for x in buf], dim=0)
            labels = torch.tensor([x[1] for x in buf], dtype=torch.int64)
            yield images, labels


def _all_local(urls: list[str]) -> bool:
    return bool(urls) and all((not is_remote_url(u)) and os.path.isfile(u) for u in urls)


def _wds_sample_handler_warn(exn: BaseException) -> bool:
    log.warning("%s%s", exn, _proxy_hint(exn))
    return True


def _wds_sample_handler_strict(exn: BaseException) -> bool:
    raise exn


def _wds_open_handler(exn: BaseException) -> bool:
    raise RuntimeError(f"{exn}{_proxy_hint(exn)}") from exn


def _collate_wds_batch(batch: list[tuple[Any, int]]) -> tuple[Any, Any]:
    torch = require_torch()
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.int64)
    return images, labels


def _validate_local_urls(urls: list[str]) -> None:
    missing = [u for u in urls if not is_remote_url(u) and not os.path.isfile(u)]
    if missing:
        raise ValueError(f"No shard URLs. Missing or unreadable: {missing[0]}")


def build_wds_dataset(
    source: str | os.PathLike[str] | Iterable[str],
    *,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    crop: int | None,
    seed: int | None,
    steps: int,
    handler: str = "warn",
) -> Any:
    urls = [os.path.abspath(u) if not is_remote_url(u) else u for u in expand_urls(source)]
    if not urls:
        raise ValueError("No shard URLs. Check the path / brace expansion.")
    if not any(is_remote_url(u) for u in urls):
        _validate_local_urls(urls)

    skip_bad = handler == "warn"
    if _all_local(urls):
        return LocalTarEpoch(
            urls,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            crop=crop,
            seed=seed,
            steps=steps,
            skip_bad=skip_bad,
        )

    import webdataset as wds

    sample_handler = _wds_sample_handler_warn if skip_bad else _wds_sample_handler_strict

    kwargs: dict[str, Any] = {
        "resampled": True,
        "handler": _wds_open_handler,
        "shardshuffle": False,
        "nodesplitter": None,
    }
    if seed is not None:
        kwargs["seed"] = seed
    try:
        pipeline = wds.WebDataset(urls, empty_check=False, **kwargs)
    except TypeError:
        kwargs.pop("nodesplitter", None)
        pipeline = wds.WebDataset(urls, **kwargs)
    if shuffle:
        pipeline = pipeline.shuffle(1000)

    pipeline = pipeline.map(partial(_decode_sample, crop=crop), handler=sample_handler)

    pipeline = pipeline.batched(batch_size, collation_fn=_collate_wds_batch, partial=not drop_last)
    pipeline = pipeline.with_epoch(int(steps))
    return pipeline


def maybe_warn_shards(n_shards: int, num_workers: int, world_size: int) -> None:
    workers = max(1, num_workers)
    if world_size <= 1 and workers <= 1:
        return
    need = 2 * world_size * workers
    if n_shards < need:
        log.error(
            "Few shards (%s) vs world*workers (%s). resampled=True may duplicate. "
            "Use shards >> world_size * num_workers.",
            n_shards,
            world_size * workers,
        )


def cap_workers(num_workers: int, n_shards: int) -> int:
    if num_workers > n_shards:
        log.warning("num_workers %s > shard count %s; capping.", num_workers, n_shards)
        return max(0, n_shards)
    return num_workers


def worker_init_fn(_worker_id: int) -> None:
    torch = require_torch()
    torch.set_num_threads(1)
