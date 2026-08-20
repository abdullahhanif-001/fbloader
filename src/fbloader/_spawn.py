"""Windows/macOS spawn guards. Default worker/pin policy."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

_SPAWN_PLATFORMS = {"win32", "cygwin", "darwin"}


def is_spawn_platform() -> bool:
    return sys.platform in _SPAWN_PLATFORMS


def default_num_workers() -> int:
    if sys.platform.startswith("win"):
        return 0
    cpu = os.cpu_count() or 1
    return min(4, max(1, cpu))


def default_pin_memory() -> bool:
    torch = None
    try:
        from fbloader._torch import require_torch

        torch = require_torch()
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def mark_main() -> None:
    """Call inside ``if __name__ == '__main__':`` before DataLoader(num_workers>0)."""
    main = sys.modules.get("__main__")
    if main is not None:
        setattr(main, "__fbloader_main_ok__", True)


def _pytest_running() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def require_main(*, num_workers: int = 1) -> None:
    """Raise if spawn workers would re-import an unguarded ``__main__``."""
    if os.environ.get("FBLOADER_SKIP_SPAWN_GUARD") == "1":
        return
    if _pytest_running():
        return
    if num_workers <= 0:
        return
    if not is_spawn_platform():
        return
    if mp.current_process().name != "MainProcess":
        return
    main = sys.modules.get("__main__")
    if main is None:
        return
    if getattr(main, "__file__", None) is None:
        return
    if getattr(main, "__fbloader_main_ok__", False):
        return
    raise RuntimeError(
        "On Windows/macOS, DataLoader(num_workers>0) uses spawn. "
        "Call fbloader.mark_main() inside `if __name__ == '__main__'` first, "
        "or keep num_workers=0 (the Windows default). "
        "Each worker re-imports torch (~RSS x N)."
    )
