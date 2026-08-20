from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

import fbloader


def _write_empty_tar(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w"):
        pass


def test_empty_tar_raises_no_valid_samples(tmp_path: Path) -> None:
    shard = tmp_path / "empty.tar"
    _write_empty_tar(shard)
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
        backend="wds",
        crop=8,
        steps=2,
        handler="warn",
    )
    with pytest.raises(RuntimeError, match="no valid samples"):
        list(loader)


def test_all_corrupt_shard_raises(tmp_path: Path) -> None:
    shard = tmp_path / "bad.tar"
    with tarfile.open(shard, "w") as tar:
        raw = b"not-a-jpeg"
        info = tarfile.TarInfo("00000.jpg")
        info.size = len(raw)
        tar.addfile(info, __import__("io").BytesIO(raw))
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
        backend="wds",
        crop=8,
        steps=2,
        handler="warn",
    )
    with pytest.raises(RuntimeError, match="no valid samples"):
        list(loader)
