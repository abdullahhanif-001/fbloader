from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from fbloader.lifetime import CudaEventFenceIterator, OwnStorageIterator, gpu_normalize


def test_cuda_event_fence_cpu_fallback() -> None:
    def gen():
        yield torch.zeros(2, 3, 4, 4, dtype=torch.uint8)

    it = CudaEventFenceIterator(gen())
    batch = next(it)
    assert batch.dtype == torch.uint8
    it.close()


def test_own_storage_close() -> None:
    def gen():
        yield torch.zeros(2, 3, 4, 4, dtype=torch.uint8)

    it = OwnStorageIterator(gen(), clone=False)
    next(it)
    it.close()


def test_own_storage_leakcheck_logs(monkeypatch) -> None:
    monkeypatch.setenv("FBLOADER_LEAKCHECK", "1")

    def gen():
        for _ in range(300):
            yield torch.zeros(1, 3, 4, 4, dtype=torch.uint8)

    with patch("fbloader.lifetime.resource_count", side_effect=[10, 11, 12]):
        with patch("fbloader.lifetime.log.info") as info:
            it = OwnStorageIterator(gen(), clone=False)
            for _ in it:
                pass
            assert info.called


def test_cuda_event_fence_cuda_path() -> None:
    fake_torch = MagicMock()
    fake_torch.Tensor = torch.Tensor
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.Event.return_value = MagicMock()
    fake_torch.cuda.current_stream.return_value = MagicMock()

    def gen():
        yield torch.zeros(1, 3, 4, 4, dtype=torch.uint8)

    with patch("fbloader.lifetime.require_torch", return_value=fake_torch):
        it = CudaEventFenceIterator(gen())
        batch = next(it)
        assert batch is not None
        it.close()


def test_gpu_normalize_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    images = torch.randint(0, 255, (1, 3, 8, 8), dtype=torch.uint8)
    out = gpu_normalize(images, device="cuda")
    assert out.device.type == "cuda"
