"""fbload: cross-platform DataLoader wrapper. Import name ``fbloader``."""

from __future__ import annotations

from fbloader._spawn import default_num_workers, default_pin_memory, mark_main, require_main
from fbloader.dataloader import DataLoader
from fbloader.lifetime import gpu_normalize

__version__ = "0.2.0"
__all__ = [
    "DataLoader",
    "gpu_normalize",
    "mark_main",
    "require_main",
    "default_num_workers",
    "default_pin_memory",
]
