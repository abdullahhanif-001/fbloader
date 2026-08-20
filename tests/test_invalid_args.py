from __future__ import annotations

from pathlib import Path

import pytest

import fbloader
from tests.shardutil import write_test_shard


def test_unknown_backend_rejected(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=4)
    with pytest.raises(ValueError, match="unknown backend"):
        fbloader.DataLoader(str(shard), backend="not-a-backend")


def test_nonpositive_batch_size_rejected(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=4)
    with pytest.raises(ValueError, match="batch_size"):
        fbloader.DataLoader(str(shard), batch_size=0, backend="wds")


def test_negative_num_workers_rejected(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=4)
    with pytest.raises(ValueError, match="num_workers"):
        fbloader.DataLoader(str(shard), num_workers=-1, backend="wds")


def test_missing_shard_path_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "shard.tar"
    with pytest.raises(ValueError, match="No shard URLs"):
        fbloader.DataLoader(str(missing), backend="wds")
