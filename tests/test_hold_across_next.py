from __future__ import annotations

from pathlib import Path

import torch

import fbloader
from tests.shardutil import write_test_shard


def test_hold_across_next(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=32)
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
        backend="wds",
        crop=16,
        steps=8,
        handler="warn",
    )
    it = iter(loader)
    images0, labels0 = next(it)
    saved = images0.clone()
    saved_y = labels0.clone()
    for _ in range(6):
        next(it)
    assert torch.equal(images0, saved)
    assert torch.equal(labels0, saved_y)
    assert images0.dtype == torch.uint8
    loader.shutdown()
