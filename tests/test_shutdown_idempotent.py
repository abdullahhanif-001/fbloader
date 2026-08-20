from __future__ import annotations

from pathlib import Path

import fbloader
from tests.shardutil import write_test_shard


def test_shutdown_idempotent(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=16)
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
    )
    with loader:
        for _ in loader:
            break
    loader.shutdown()
    loader.shutdown()
    loader.shutdown()


def test_context_manager_closes_cleanly(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=16)
    with fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
        backend="wds",
        crop=8,
        steps=2,
    ) as loader:
        n = sum(1 for _ in loader)
    assert n == 2
