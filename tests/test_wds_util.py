from __future__ import annotations

from pathlib import Path

import fbloader
from fbloader.backends import wds as wds_backend
from tests.shardutil import write_test_shard


def test_expand_urls_brace_pattern(tmp_path: Path) -> None:
    for i in range(3):
        write_test_shard(tmp_path / f"part{i}.tar", n=4)
    urls = wds_backend.expand_urls(str(tmp_path / "part{0..2}.tar"))
    assert len(urls) == 3


def test_is_remote_url() -> None:
    assert wds_backend.is_remote_url("s3://bucket/shard.tar")
    assert wds_backend.is_remote_url("https://example.com/shard.tar")
    assert not wds_backend.is_remote_url("/local/shard.tar")


def test_parse_label_variants() -> None:
    assert wds_backend._parse_label(b"7") == 7
    assert wds_backend._parse_label('{"label": 3}') == 3
    assert wds_backend._parse_label("bad") == 0


def test_dataloader_len_and_backend(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=8)
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        backend="wds",
        steps=5,
        crop=8,
    )
    assert loader.backend == "wds"
    assert len(loader) == 5
    loader.shutdown()


def test_map_backend_auto(tmp_path: Path) -> None:
    import torch
    from torch.utils.data import Dataset

    class Tiny(Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, idx: int):
            return torch.zeros(3, 8, 8, dtype=torch.uint8), idx

    loader = fbloader.DataLoader(Tiny(), batch_size=4, num_workers=0, steps=2, crop=8)
    assert loader.backend == "torch"
    n = sum(1 for _ in loader)
    assert n == 2
