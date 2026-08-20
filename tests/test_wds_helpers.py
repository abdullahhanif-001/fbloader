from __future__ import annotations

import logging
from pathlib import Path

import pytest

from fbloader.backends import wds as wds_backend
from fbloader.lifetime import leakcheck_enabled, resource_count
from tests.shardutil import write_test_shard


def test_cap_workers() -> None:
    assert wds_backend.cap_workers(8, 2) == 2


def test_worker_init_fn() -> None:
    wds_backend.worker_init_fn(0)


def test_maybe_warn_shards(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="fbloader")
    wds_backend.maybe_warn_shards(n_shards=1, num_workers=4, world_size=1)
    assert "Few shards" in caplog.text


def test_resource_count_returns_int() -> None:
    count = resource_count()
    assert count == -1 or count >= 0


def test_leakcheck_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FBLOADER_LEAKCHECK", "1")
    assert leakcheck_enabled() is True


def test_strict_handler_raises_on_corrupt(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "bad.tar", n=4, corrupt_at=0)
    ds = wds_backend.build_wds_dataset(
        str(shard),
        batch_size=2,
        drop_last=True,
        shuffle=False,
        crop=8,
        seed=0,
        steps=1,
        handler="strict",
    )
    with pytest.raises(Exception):
        next(iter(ds))
