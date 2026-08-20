from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import fbloader
from tests.shardutil import write_test_shard


@pytest.mark.slow
def test_multiprocess_workers(tmp_path: Path) -> None:
    if sys.platform.startswith("win"):
        fbloader.mark_main()
    shards = [write_test_shard(tmp_path / f"s{i}.tar", n=24) for i in range(4)]
    loader = fbloader.DataLoader(
        [str(p) for p in shards],
        batch_size=4,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
        backend="wds",
        crop=8,
        steps=20,
        seed=7,
        persistent_workers=True,
        prefetch_factor=2,
    )
    n = 0
    with loader:
        for images, labels in loader:
            assert images.dtype == torch.uint8
            assert images.shape[0] == 4
            assert labels.shape[0] == 4
            n += 1
    assert n == 20
