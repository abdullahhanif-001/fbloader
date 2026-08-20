from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, IterableDataset

import fbloader
from fbloader import dataloader as dl
from fbloader.lifetime import _clone_batch
from tests.shardutil import write_test_shard


class _MapDs(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        return torch.zeros(3, 4, 4, dtype=torch.uint8), idx


class _IterDs(IterableDataset):
    def __iter__(self):
        yield torch.zeros(3, 4, 4, dtype=torch.uint8), 0


def test_is_map_dataset_variants() -> None:
    assert dl._is_map_dataset(_MapDs()) is True
    assert dl._is_map_dataset(_IterDs()) is False
    assert dl._is_map_dataset(["a.tar"]) is False


def test_is_remote_variants() -> None:
    assert dl._is_remote("s3://b/x.tar") is True
    assert dl._is_remote(["s3://a", "/local"]) is True
    assert dl._is_remote("/local/x.tar") is False


def test_pick_backend_local_directory(tmp_path: Path) -> None:
    write_test_shard(tmp_path / "s0.tar", n=4)
    assert dl._pick_backend(str(tmp_path), "auto") == "wds"


def test_pick_backend_map_dataset() -> None:
    assert dl._pick_backend(_MapDs(), "auto") == "torch"


def test_dataloader_context_manager(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=8)
    with fbloader.DataLoader(str(shard), batch_size=4, num_workers=0, steps=2, crop=8) as loader:
        assert sum(1 for _ in loader) == 2


def test_dataloader_len_from_length(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=8)
    loader = fbloader.DataLoader(
        str(shard),
        batch_size=4,
        num_workers=0,
        length=8,
        steps=2,
        crop=8,
    )
    assert len(loader) == 8
    loader.shutdown()


def test_streaming_backend_builds_map_loader() -> None:
    fake_ds = _MapDs()
    with patch("fbloader.backends.streaming.streaming_available", return_value=True):
        with patch("fbloader.backends.streaming.looks_like_mds", return_value=True):
            with patch("fbloader.backends.streaming.build_streaming_dataset", return_value=fake_ds):
                loader = fbloader.DataLoader(
                    "s3://bucket/data.mds",
                    batch_size=4,
                    num_workers=0,
                    backend="streaming",
                    steps=2,
                    crop=8,
                )
    assert loader.backend == "streaming"
    assert sum(1 for _ in loader) == 1


def test_clone_batch_structures() -> None:
    t = torch.tensor([1, 2], dtype=torch.uint8)
    out_t = _clone_batch(t)
    assert out_t is not t
    out_tuple = _clone_batch((t, 1))
    assert out_tuple[0] is not t
    out_dict = _clone_batch({"x": t})
    assert out_dict["x"] is not t
