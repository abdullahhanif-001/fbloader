from __future__ import annotations

from pathlib import Path

import torch

import fbloader
from fbloader.lifetime import resource_count
from tests.shardutil import write_test_shard


def test_concurrent_loaders_no_handle_explosion(tmp_path: Path) -> None:
    shards = [write_test_shard(tmp_path / f"s{i}.tar", n=12) for i in range(2)]
    start = resource_count()
    loaders = [
        fbloader.DataLoader(
            str(shards[i]),
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            backend="wds",
            crop=8,
            steps=3,
        )
        for i in range(2)
    ]
    try:
        for loader in loaders:
            for images, labels in loader:
                assert images.dtype == torch.uint8
                assert labels.dtype == torch.int64
    finally:
        for loader in loaders:
            loader.shutdown()
    end = resource_count()
    if start >= 0 and end >= 0:
        assert end - start <= 16
