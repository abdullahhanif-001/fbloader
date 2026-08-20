"""Map-style torch.utils.data.Dataset path (slow, documented)."""

from __future__ import annotations

from typing import Any, Callable

from fbloader._torch import require_torch


def worker_init_fn(_worker_id: int) -> None:
    torch = require_torch()
    torch.set_num_threads(1)


def build_map_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    collate_fn: Callable[..., Any] | None,
    crop: int | None,
) -> Any:
    torch = require_torch()
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "shuffle": shuffle,
        "persistent_workers": bool(persistent_workers) and num_workers > 0,
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor or 2
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    elif crop:
        kwargs["collate_fn"] = _resize_collate(crop)
    return torch.utils.data.DataLoader(**kwargs)


def _resize_collate(crop: int) -> Callable[[list[Any]], Any]:
    torch = require_torch()
    import numpy as np
    from PIL import Image

    def collate(batch: list[Any]) -> tuple[Any, Any]:
        images = []
        labels = []
        for item in batch:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                image, label = item[0], item[1]
            else:
                image, label = item, 0
            if isinstance(image, Image.Image):
                image = image.convert("RGB").resize((crop, crop), Image.BILINEAR)
                array = np.array(image, dtype=np.uint8, copy=True)
                image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            elif isinstance(image, torch.Tensor):
                if image.dtype != torch.uint8:
                    image = image.to(torch.uint8)
                if image.ndim == 3 and image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
                    image = image.permute(2, 0, 1).contiguous()
            images.append(image)
            labels.append(int(label) if not isinstance(label, torch.Tensor) else int(label.item()))
        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.int64)

    return collate
