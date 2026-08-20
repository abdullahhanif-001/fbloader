from __future__ import annotations

import torch

from fbloader.lifetime import OwnStorageIterator, gpu_normalize


def test_own_storage_iterator_clones() -> None:
    def gen():
        t = torch.arange(12, dtype=torch.uint8).reshape(3, 2, 2)
        yield t
        yield t + 1

    it = OwnStorageIterator(gen(), clone=True)
    first = next(it)
    saved = first.clone()
    second = next(it)
    assert not torch.equal(first, second)
    assert torch.equal(first, saved)


def test_gpu_normalize_cpu_tensor() -> None:
    images = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8)
    out = gpu_normalize(images, device="cpu", non_blocking=False)
    assert out.dtype == torch.float32
    assert out.shape == images.shape
