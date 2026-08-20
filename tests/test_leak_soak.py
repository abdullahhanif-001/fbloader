from __future__ import annotations

from pathlib import Path

import pytest

import fbloader
from fbloader.lifetime import resource_count
from tests.shardutil import write_test_shard


@pytest.mark.slow
def test_leak_soak(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=24)
    start = resource_count()
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
        backend="wds",
        crop=8,
        steps=200,
        seed=1,
    )
    n = 0
    with loader:
        for _ in loader:
            n += 1
    assert n == 200
    end = resource_count()
    if start >= 0 and end >= 0:
        assert end - start <= 32, f"resource leak start={start} end={end}"
