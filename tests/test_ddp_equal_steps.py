from __future__ import annotations

from pathlib import Path

import fbloader
from tests.shardutil import write_test_shard


def _count_steps(url: str, steps: int) -> int:
    loader = fbloader.DataLoader(
        url,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
        backend="wds",
        crop=8,
        steps=steps,
        seed=0,
    )
    n = 0
    with loader:
        for _ in loader:
            n += 1
    return n


def test_ddp_equal_steps(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=40)
    url = str(shard)
    a = _count_steps(url, steps=7)
    b = _count_steps(url, steps=7)
    assert a == 7
    assert b == 7
    assert a == b
