"""Minimal entry. Windows/macOS: mark_main before workers."""

from __future__ import annotations

import fbloader


def main() -> None:
    fbloader.mark_main()
    print("fbload", fbloader.__version__, "workers", fbloader.default_num_workers())


if __name__ == "__main__":
    main()
