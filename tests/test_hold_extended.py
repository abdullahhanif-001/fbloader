from __future__ import annotations

from pathlib import Path

import torch

import fbloader
from tests.shardutil import write_test_shard


def test_hold_across_many_next_calls(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=128)
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
        backend="wds",
        crop=16,
        steps=32,
    )
    it = iter(loader)
    images0, labels0 = next(it)
    saved = images0.clone()
    saved_y = labels0.clone()
    for _ in range(30):
        next(it)
    assert torch.equal(images0, saved)
    assert torch.equal(labels0, saved_y)
    loader.shutdown()
