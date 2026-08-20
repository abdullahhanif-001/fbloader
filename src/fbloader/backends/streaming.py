"""Optional MosaicML Streaming backend (install mosaicml-streaming separately)."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("fbloader")


def streaming_available() -> bool:
    try:
        import streaming  # noqa: F401
    except Exception:
        return False
    return True


def looks_like_mds(source: Any) -> bool:
    if isinstance(source, dict) and ("remote" in source or "local" in source):
        return True
    text = str(source)
    return text.endswith(".mds") or "/mds" in text.replace("\\", "/")


def build_streaming_dataset(source: Any, *, batch_size: int, shuffle: bool) -> Any:
    try:
        from streaming import StreamingDataset
    except ImportError as exc:
        raise RuntimeError(
            "mosaicml-streaming is not installed. python -m pip install mosaicml-streaming"
        ) from exc
    if isinstance(source, StreamingDataset):
        return source
    if isinstance(source, dict):
        kwargs = dict(source)
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("shuffle", shuffle)
        return StreamingDataset(**kwargs)
    return StreamingDataset(remote=str(source), batch_size=batch_size, shuffle=shuffle)
