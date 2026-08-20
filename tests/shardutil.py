"""Tar shard helpers for pytest fixtures."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from PIL import Image


def write_test_shard(path: Path, n: int, *, corrupt_at: int | None = None, size: int = 32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            buf = io.BytesIO()
            if corrupt_at is not None and i == corrupt_at:
                raw = b"not-a-jpeg-payload"
            else:
                Image.new("RGB", (size, size), (i % 255, 32, 200)).save(buf, format="JPEG")
                raw = buf.getvalue()
            name = f"{i:05d}.jpg"
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
            label = f"{i % 10}\n".encode("utf-8")
            info2 = tarfile.TarInfo(f"{i:05d}.cls")
            info2.size = len(label)
            tar.addfile(info2, io.BytesIO(label))
    return path
