from __future__ import annotations

import sys

import pytest


def test_import_without_dali() -> None:
    import fbloader

    assert fbloader.DataLoader is not None
    assert "nvidia.dali" not in sys.modules


def test_default_num_workers_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import fbloader._spawn as spawn

    monkeypatch.setattr(spawn.sys, "platform", "win32")
    assert spawn.default_num_workers() == 0


def test_default_num_workers_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    import fbloader._spawn as spawn

    monkeypatch.setattr(spawn.sys, "platform", "linux")
    monkeypatch.setattr(spawn.os, "cpu_count", lambda: 16)
    assert spawn.default_num_workers() == 4


def test_map_dataset_uint8() -> None:
    import torch
    from torch.utils.data import Dataset

    import fbloader

    class Tiny(Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, idx: int):
            from PIL import Image

            return Image.new("RGB", (20, 10), (idx, 1, 2)), idx

    loader = fbloader.DataLoader(
        Tiny(),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
        backend="torch",
        crop=8,
    )
    images, labels = next(iter(loader))
    assert images.dtype == torch.uint8
    assert images.shape == (4, 3, 8, 8)
    loader.shutdown()


def test_gpu_normalize_cpu() -> None:
    import torch

    import fbloader

    x = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8)
    saved = x.clone()
    y = fbloader.gpu_normalize(x, device="cpu")
    assert y.dtype == torch.float32
    assert y.shape == x.shape
    assert torch.equal(x, saved)


def test_dali_platform_message_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    from fbloader.backends import dali as dali_backend

    monkeypatch.setattr(dali_backend.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux\\+CUDA only"):
        dali_backend.require_dali_platform()

