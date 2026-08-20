from __future__ import annotations

from pathlib import Path

import torch

import fbloader
from tests.shardutil import write_test_shard


def test_corrupt_jpeg_skipped(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=20, corrupt_at=3)
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
        backend="wds",
        crop=8,
        steps=3,
        handler="warn",
    )
    n = 0
    with loader:
        for images, labels in loader:
            assert images.dtype == torch.uint8
            assert images.shape[0] == 4
            n += 1
    assert n == 3
