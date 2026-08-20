#!/usr/bin/env python3
"""
Soak benchmark: fbload vs native PyTorch DataLoader.

Materializes real JPEG tar shards and an ImageFolder tree, then measures
throughput and OS-level counters over a fixed step budget.

Metrics (best-effort per platform):
  - throughput (images/s)
  - context switches per batch (voluntary + involuntary)
  - page faults per batch
  - file descriptor / handle drift
  - tensor contiguity
  - Linux: hardware counters via ``perf stat`` when available

Usage:
  python -m pip install torch psutil pillow
  python tests/benchmark_soak.py --steps 10000 --batch-size 32 --num-workers 4

Smoke run:
  python tests/benchmark_soak.py --steps 200 --quick
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

if sys.platform != "win32":
    import resource
else:
    resource = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional deps — fail with actionable message
# ---------------------------------------------------------------------------
try:
    import psutil
except ImportError as exc:
    raise SystemExit("Install psutil: python -m pip install psutil") from exc

try:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset
except ImportError as exc:
    raise SystemExit("Install torch and pillow from pytorch.org / pip") from exc

try:
    import fbloader
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import fbloader

log = logging.getLogger("fbload.benchmark")


# ---------------------------------------------------------------------------
# Dataset materialization (JPEG on disk)
# ---------------------------------------------------------------------------
def _jpeg_bytes(width: int, height: int, seed: int) -> bytes:
    buf = io.BytesIO()
    r = (seed * 17 + 3) % 256
    g = (seed * 31 + 7) % 256
    b = (seed * 13 + 11) % 256
    Image.new("RGB", (width, height), (r, g, b)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def materialize_real_dataset(
    root: Path,
    *,
    num_shards: int,
    samples_per_shard: int,
    image_size: int,
) -> tuple[str, Path]:
    """Return (tar_glob_for_fbload, imagefolder_root_for_pytorch)."""
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    img_root = root / "imagefolder" / "class0"
    img_root.mkdir(parents=True, exist_ok=True)

    idx = 0
    for s in range(num_shards):
        tar_path = shard_dir / f"shard-{s:05d}.tar"
        with tarfile.open(tar_path, "w") as tar:
            for _ in range(samples_per_shard):
                raw = _jpeg_bytes(image_size, image_size, idx)
                jpg_name = f"{idx:06d}.jpg"
                info = tarfile.TarInfo(jpg_name)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
                cls = f"{idx % 10}\n".encode()
                cinfo = tarfile.TarInfo(f"{idx:06d}.cls")
                cinfo.size = len(cls)
                tar.addfile(cinfo, io.BytesIO(cls))
                (img_root / jpg_name).write_bytes(raw)
                idx += 1

    shard_paths = [shard_dir / f"shard-{s:05d}.tar" for s in range(num_shards)]
    return [str(p.resolve()) for p in shard_paths], img_root.parent


class _ImageFolderUint8(Dataset):
    """Map-style dataset mirroring fbload output layout (uint8 NCHW)."""

    def __init__(self, root: Path, crop: int) -> None:
        self.root = root
        self.crop = crop
        self.paths = sorted(p for p in root.rglob("*.jpg") if p.is_file())
        if not self.paths:
            raise RuntimeError(f"no JPEGs under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        p = self.paths[i % len(self.paths)]
        with Image.open(p) as im:
            im = im.convert("RGB")
            if self.crop:
                im = im.resize((self.crop, self.crop), Image.BILINEAR)
            import numpy as np

            arr = np.array(im, dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t, i % 10


def _collate_uint8(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    return images, labels


def _pytorch_worker_init_fn(_worker_id: int) -> None:
    """Pickle-safe worker_init_fn for PyTorch baseline (Windows spawn)."""
    torch.set_num_threads(1)


class _StepLimitedLoader:
    """Run exactly ``steps`` batches; restart epochs when map-style sampler exhausts."""

    def __init__(self, dl: TorchDataLoader, max_steps: int) -> None:
        self._dl = dl
        self._left = max_steps
        self._it: Iterator[Any] | None = None

    def __iter__(self) -> _StepLimitedLoader:
        return self

    def __next__(self) -> Any:
        while self._left > 0:
            if self._it is None:
                self._it = iter(self._dl)
            try:
                batch = next(self._it)
                self._left -= 1
                return batch
            except StopIteration:
                self._it = None
        raise StopIteration


# ---------------------------------------------------------------------------
# OS / kernel counter plumbing
# ---------------------------------------------------------------------------
@dataclass
class CounterSnapshot:
    ts: float = 0.0
    voluntary_ctx: int = 0
    involuntary_ctx: int = 0
    minor_faults: int = 0
    major_faults: int = 0
    ru_utime: float = 0.0
    ru_stime: float = 0.0
    rss_bytes: int = 0
    vms_bytes: int = 0
    num_threads: int = 0
    num_fds: int = -1
    num_handles: int = -1
    page_fault_count_win: int = 0
    # Linux /proc/status extras
    hw_cpu_mhz: float = 0.0


@dataclass
class CounterDelta:
    voluntary_ctx: int = 0
    involuntary_ctx: int = 0
    minor_faults: int = 0
    major_faults: int = 0
    ru_utime: float = 0.0
    ru_stime: float = 0.0
    rss_bytes_delta: int = 0
    num_fds_delta: int = 0
    num_handles_delta: int = 0
    page_fault_count_win: int = 0

    @staticmethod
    def between(a: CounterSnapshot, b: CounterSnapshot) -> CounterDelta:
        return CounterDelta(
            voluntary_ctx=max(0, b.voluntary_ctx - a.voluntary_ctx),
            involuntary_ctx=max(0, b.involuntary_ctx - a.involuntary_ctx),
            minor_faults=max(0, b.minor_faults - a.minor_faults),
            major_faults=max(0, b.major_faults - a.major_faults),
            ru_utime=max(0.0, b.ru_utime - a.ru_utime),
            ru_stime=max(0.0, b.ru_stime - a.ru_stime),
            rss_bytes_delta=b.rss_bytes - a.rss_bytes,
            num_fds_delta=b.num_fds - a.num_fds if a.num_fds >= 0 and b.num_fds >= 0 else 0,
            num_handles_delta=b.num_handles - a.num_handles if a.num_handles >= 0 and b.num_handles >= 0 else 0,
            page_fault_count_win=max(0, b.page_fault_count_win - a.page_fault_count_win),
        )


def _read_proc_status_int(key: str) -> int:
    path = Path("/proc/self/status")
    if not path.is_file():
        return 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(key + ":"):
            return int(line.split(":", 1)[1].strip().split()[0])
    return 0


def _count_fds_posix() -> int:
    fd_dir = Path(f"/proc/{os.getpid()}/fd")
    if fd_dir.is_dir():
        return len(list(fd_dir.iterdir()))
    try:
        return len(psutil.Process().open_files())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return -1


def _windows_handle_count() -> int:
    if os.name != "nt":
        return -1
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    count = ctypes.c_ulong()
    if kernel32.GetProcessHandleCount(handle, ctypes.byref(count)):
        return int(count.value)
    return -1


def _windows_page_fault_count() -> int:
    if os.name != "nt":
        return 0
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    psapi = ctypes.windll.psapi
    if psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        return int(counters.PageFaultCount)
    return 0


def snapshot_counters() -> CounterSnapshot:
    proc = psutil.Process()
    mem = proc.memory_info()
    ru_utime = 0.0
    ru_stime = 0.0
    minor_faults = 0
    major_faults = 0
    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        ru_utime = float(ru.ru_utime)
        ru_stime = float(ru.ru_stime)
        minor_faults = int(getattr(ru, "ru_minflt", 0))
        major_faults = int(getattr(ru, "ru_majflt", 0))
    else:
        try:
            ct = proc.cpu_times()
            ru_utime = float(ct.user)
            ru_stime = float(ct.system)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    snap = CounterSnapshot(
        ts=time.perf_counter(),
        ru_utime=ru_utime,
        ru_stime=ru_stime,
        minor_faults=minor_faults,
        major_faults=major_faults,
        rss_bytes=int(mem.rss),
        vms_bytes=int(mem.vms),
        num_threads=int(proc.num_threads()),
        num_fds=_count_fds_posix() if sys.platform != "win32" else -1,
        num_handles=_windows_handle_count(),
        page_fault_count_win=_windows_page_fault_count(),
    )
    try:
        ctx = proc.num_ctx_switches()
        snap.voluntary_ctx = int(ctx.voluntary)
        snap.involuntary_ctx = int(ctx.involuntary)
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        pass
    if sys.platform != "win32":
        snap.voluntary_ctx = max(snap.voluntary_ctx, _read_proc_status_int("voluntary_ctxt_switches"))
        snap.involuntary_ctx = max(snap.involuntary_ctx, _read_proc_status_int("nonvoluntary_ctxt_switches"))
    return snap


@dataclass
class PerfHwCounters:
    available: bool = False
    cycles: int = 0
    instructions: int = 0
    branch_misses: int = 0
    cache_references: int = 0
    cache_misses: int = 0
    llc_load_misses: int = 0
    context_switches: int = 0
    page_faults: int = 0
    raw_stderr: str = ""


def _parse_perf_stat(stderr: str) -> PerfHwCounters:
    out = PerfHwCounters(available=True, raw_stderr=stderr)
    mapping = {
        "cycles": "cycles",
        "instructions": "instructions",
        "br_misp_retired": "branch_misses",
        "branch-misses": "branch_misses",
        "cache-references": "cache_references",
        "cache-misses": "cache_misses",
        "LLC-load-misses": "llc_load_misses",
        "context-switches": "context_switches",
        "page-faults": "page_faults",
    }
    for line in stderr.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Performance"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            val = int(float(parts[0].replace(",", "")))
        except ValueError:
            continue
        name = parts[-1].strip(":").lower()
        for key, attr in mapping.items():
            if key in name or name.endswith(key):
                setattr(out, attr, val)
                break
    return out


class LinuxPerfMonitor:
    """Attach ``perf stat -p PID`` for hardware counters (Linux only)."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._proc: subprocess.Popen[str] | None = None
        self._stderr = ""

    def start(self) -> bool:
        if sys.platform != "win32" and shutil.which("perf"):
            events = (
                "cycles,instructions,br_misp_retired,"
                "cache-references,cache-misses,LLC-load-misses,"
                "context-switches,page-faults"
            )
            try:
                self._proc = subprocess.Popen(
                    ["perf", "stat", "-p", str(self.pid), "-e", events],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return True
            except OSError:
                return False
        return False

    def stop(self) -> PerfHwCounters:
        if self._proc is None:
            return PerfHwCounters(available=False)
        try:
            self._proc.terminate()
            _, err = self._proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            _, err = self._proc.communicate(timeout=2)
        self._stderr = err or ""
        return _parse_perf_stat(self._stderr)


# ---------------------------------------------------------------------------
# Tensor / batch instrumentation
# ---------------------------------------------------------------------------
@dataclass
class BatchTensorStats:
    contiguous: bool = True
    storage_offset: int = 0
    nbytes: int = 0
    stride_product_matches: bool = True
    numel: int = 0


def inspect_batch_tensors(batch: Any) -> BatchTensorStats:
    stats = BatchTensorStats()
    if not isinstance(batch, (tuple, list)) or not batch:
        return stats
    t = batch[0]
    if not isinstance(t, torch.Tensor):
        return stats
    stats.contiguous = bool(t.is_contiguous())
    stats.storage_offset = int(t.storage_offset())
    stats.nbytes = int(t.untyped_storage().nbytes())
    stats.numel = int(t.numel())
    expected = t.element_size() * t.numel()
    stats.stride_product_matches = stats.nbytes >= expected
    return stats


@dataclass
class SoakSample:
    step: int
    wall_s: float
    images_per_s: float
    fds: int
    handles: int
    voluntary_ctx: int
    involuntary_ctx: int
    contiguous: bool
    rss_mb: float


@dataclass
class SoakReport:
    name: str
    steps: int
    batch_size: int
    num_workers: int
    total_images: int
    wall_seconds: float
    throughput_ips: float
    counter_delta: CounterDelta = field(default_factory=CounterDelta)
    perf_hw: PerfHwCounters = field(default_factory=PerfHwCounters)
    tensor_contiguous_pct: float = 100.0
    fd_drift_max: int = 0
    handle_drift_max: int = 0
    ctx_switches_per_batch: float = 0.0
    page_faults_per_batch: float = 0.0
    efficiency_score: float = 0.0
    samples: list[SoakSample] = field(default_factory=list)
    cuda_profile_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["counter_delta"] = asdict(self.counter_delta)
        d["perf_hw"] = asdict(self.perf_hw)
        d["samples"] = [asdict(s) for s in self.samples]
        return d


def _efficiency_score(
    throughput_ips: float,
    perf: PerfHwCounters,
    delta: CounterDelta,
) -> float:
    """
    Throughput / cycles-per-image (higher is better).
    Falls back to throughput / (user+sys CPU seconds) when perf unavailable.
    """
    if perf.available and perf.cycles > 0 and throughput_ips > 0:
        cycles_per_image = perf.cycles / max(1.0, throughput_ips * 1.0)
        return throughput_ips / max(1.0, cycles_per_image / 1e6)
    cpu_s = delta.ru_utime + delta.ru_stime
    if cpu_s > 0 and throughput_ips > 0:
        return throughput_ips / cpu_s
    return throughput_ips


def run_soak(
    name: str,
    loader: Iterator[Any],
    *,
    steps: int,
    batch_size: int,
    num_workers: int,
    sample_interval: int,
    cuda_profile: bool,
    cuda_profile_steps: int,
) -> SoakReport:
    pid = os.getpid()
    perf_mon = LinuxPerfMonitor(pid)
    perf_started = perf_mon.start()
    if perf_started:
        log.info("[%s] perf stat attached to pid=%s", name, pid)

    t0 = time.perf_counter()
    snap0 = snapshot_counters()
    fd0 = snap0.num_fds
    h0 = snap0.num_handles

    contiguous_hits = 0
    total_batches = 0
    fd_series: list[int] = []
    handle_series: list[int] = []
    samples: list[SoakSample] = []
    cuda_path = ""

    prof = None
    if cuda_profile and torch.cuda.is_available():
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
        )
        prof.__enter__()

    images = 0
    try:
        for step in range(1, steps + 1):
            batch = next(loader)
            total_batches += 1
            images += batch_size

            ts = inspect_batch_tensors(batch)
            if ts.contiguous:
                contiguous_hits += 1

            if step % sample_interval == 0 or step == 1:
                snap = snapshot_counters()
                fd_series.append(snap.num_fds)
                handle_series.append(snap.num_handles)
                elapsed = time.perf_counter() - t0
                ips = images / max(elapsed, 1e-9)
                samples.append(
                    SoakSample(
                        step=step,
                        wall_s=elapsed,
                        images_per_s=ips,
                        fds=snap.num_fds,
                        handles=snap.num_handles,
                        voluntary_ctx=snap.voluntary_ctx,
                        involuntary_ctx=snap.involuntary_ctx,
                        contiguous=ts.contiguous,
                        rss_mb=snap.rss_bytes / (1024 * 1024),
                    )
                )

            if prof is not None and step == cuda_profile_steps:
                prof.step()

    finally:
        if prof is not None:
            prof.__exit__(None, None, None)
            cuda_path = str(Path(tempfile.gettempdir()) / f"fbload_{name}_cuda.json")
            prof.export_chrome_trace(cuda_path)

    wall = time.perf_counter() - t0
    snap1 = snapshot_counters()
    delta = CounterDelta.between(snap0, snap1)
    perf_hw = perf_mon.stop() if perf_started else PerfHwCounters(available=False)

    throughput = images / max(wall, 1e-9)
    ctx_per_batch = (delta.voluntary_ctx + delta.involuntary_ctx) / max(1, total_batches)
    if sys.platform == "win32" and delta.page_fault_count_win:
        pf_per_batch = delta.page_fault_count_win / max(1, total_batches)
    else:
        pf_per_batch = (delta.minor_faults + delta.major_faults) / max(1, total_batches)

    fd_drift = max((f - fd0 for f in fd_series if f >= 0), default=0)
    handle_drift = max((h - h0 for h in handle_series if h >= 0), default=0)

    return SoakReport(
        name=name,
        steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        total_images=images,
        wall_seconds=wall,
        throughput_ips=throughput,
        counter_delta=delta,
        perf_hw=perf_hw,
        tensor_contiguous_pct=100.0 * contiguous_hits / max(1, total_batches),
        fd_drift_max=fd_drift,
        handle_drift_max=handle_drift,
        ctx_switches_per_batch=ctx_per_batch,
        page_faults_per_batch=pf_per_batch,
        efficiency_score=_efficiency_score(throughput, perf_hw, delta),
        samples=samples,
        cuda_profile_path=cuda_path,
    )


def _build_fbload_loader(
    tar_source: str | list[str],
    *,
    batch_size: int,
    num_workers: int,
    steps: int,
    crop: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> tuple[Any, Callable[[], None]]:
    loader = fbloader.DataLoader(
        tar_source,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        backend="wds",
        crop=crop,
        steps=steps,
        seed=42,
        handler="warn",
    )

    def shutdown() -> None:
        loader.shutdown()

    return iter(loader), shutdown


def _build_pytorch_loader(
    imagefolder_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    crop: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    steps: int,
) -> tuple[Iterator[Any], Callable[[], None]]:
    ds = _ImageFolderUint8(imagefolder_root / "class0", crop=crop)
    kwargs: dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
        "collate_fn": _collate_uint8,
        "persistent_workers": bool(persistent_workers) and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(2, prefetch_factor)
        kwargs["worker_init_fn"] = _pytorch_worker_init_fn
    inner = TorchDataLoader(**kwargs)

    holder: dict[str, Any] = {"dl": inner}

    def shutdown() -> None:
        dl = holder["dl"]
        it = getattr(dl, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass

    return _StepLimitedLoader(inner, steps), shutdown


def print_report(r: SoakReport) -> None:
    d = r.counter_delta
    p = r.perf_hw
    print("\n" + "=" * 72)
    print(f"  {r.name}")
    print("=" * 72)
    print(f"  steps={r.steps}  batch={r.batch_size}  workers={r.num_workers}")
    print(f"  wall={r.wall_seconds:.2f}s  throughput={r.throughput_ips:.1f} img/s")
    print(f"  tensor contiguous={r.tensor_contiguous_pct:.2f}%")
    print(f"  ctx switches / batch (vol+invol): {r.ctx_switches_per_batch:.3f}")
    print(f"  page faults / batch: {r.page_faults_per_batch:.3f}")
    print(f"  FD drift max: {r.fd_drift_max}  handle drift max: {r.handle_drift_max}")
    print(f"  CPU user+sys: {d.ru_utime + d.ru_stime:.2f}s  RSS delta: {d.rss_bytes_delta / 1024 / 1024:.1f} MiB")
    print(f"  voluntary ctx: {d.voluntary_ctx}  involuntary ctx: {d.involuntary_ctx}")
    print(f"  minor faults: {d.minor_faults}  major faults: {d.major_faults}")
    if p.available:
        print("  --- perf hardware counters ---")
        print(f"  cycles: {p.cycles:,}  instructions: {p.instructions:,}")
        if p.instructions:
            print(f"  IPC: {p.instructions / max(1, p.cycles):.3f}")
        print(f"  branch_misses: {p.branch_misses:,}")
        print(f"  cache_refs: {p.cache_references:,}  cache_misses: {p.cache_misses:,}")
        if p.cache_references:
            print(f"  cache miss rate: {100.0 * p.cache_misses / p.cache_references:.3f}%")
        print(f"  LLC load misses: {p.llc_load_misses:,}")
    else:
        print("  perf HW counters: unavailable (install linux-tools / run on Linux with perf)")
    print(f"  efficiency score: {r.efficiency_score:.6f}")
    if r.cuda_profile_path:
        print(f"  CUDA chrome trace: {r.cuda_profile_path}")
    if r.samples:
        s0, s1 = r.samples[0], r.samples[-1]
        print(f"  sample[0] ips={s0.images_per_s:.1f} fds={s0.fds} handles={s0.handles} rss={s0.rss_mb:.1f}MB")
        print(f"  sample[-1] ips={s1.images_per_s:.1f} fds={s1.fds} handles={s1.handles} rss={s1.rss_mb:.1f}MB")


def compare_reports(fb: SoakReport, pt: SoakReport) -> None:
    print("\n" + "#" * 72)
    print("  COMPARISON (fbload vs PyTorch native)")
    print("#" * 72)
    def pct(a: float, b: float) -> str:
        if b == 0:
            return "n/a"
        return f"{100.0 * (a - b) / b:+.1f}%"

    tpct = pct(fb.throughput_ips, pt.throughput_ips)
    print(
        f"  throughput:     fbload {fb.throughput_ips:.1f}  "
        f"pytorch {pt.throughput_ips:.1f}  ({tpct})"
    )
    print(f"  ctx sw/batch:   fbload {fb.ctx_switches_per_batch:.3f}  pytorch {pt.ctx_switches_per_batch:.3f}")
    print(f"  faults/batch:   fbload {fb.page_faults_per_batch:.3f}  pytorch {pt.page_faults_per_batch:.3f}")
    print(f"  FD drift:       fbload {fb.fd_drift_max}  pytorch {pt.fd_drift_max}")
    print(f"  efficiency score: fbload {fb.efficiency_score:.6f}  pytorch {pt.efficiency_score:.6f}")
    if fb.perf_hw.available and pt.perf_hw.available:
        fb_cpi = fb.perf_hw.cycles / max(1, fb.total_images)
        pt_cpi = pt.perf_hw.cycles / max(1, pt.total_images)
        print(f"  cycles/image:   fbload {fb_cpi:.0f}  pytorch {pt_cpi:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="fbload vs PyTorch DataLoader soak benchmark")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=None, help="Reuse existing materialized dataset")
    parser.add_argument("--sample-interval", type=int, default=500)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-fbload", action="store_true")
    parser.add_argument("--cuda-profile", action="store_true")
    parser.add_argument("--cuda-profile-step", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--materialize-only", action="store_true", help="Write dataset to --data-dir and exit")
    parser.add_argument("--pin-memory", action="store_true", help="Force pin_memory=True (fair GPU baseline)")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent_workers")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="200 steps smoke")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.quick:
        args.steps = min(args.steps, 200)
        args.sample_interval = min(args.sample_interval, 50)

    fbloader.mark_main()
    num_workers = args.num_workers if args.num_workers is not None else fbloader.default_num_workers()
    if args.num_workers is not None and args.num_workers > 0:
        fbloader.require_main(num_workers=args.num_workers)
    pin = True if args.pin_memory else fbloader.default_pin_memory()
    persistent = (not args.no_persistent_workers) and num_workers > 0
    prefetch = max(2, args.prefetch_factor)

    if args.materialize_only:
        if not args.data_dir:
            raise SystemExit("--materialize-only requires --data-dir")
        root = Path(args.data_dir)
        root.mkdir(parents=True, exist_ok=True)
        materialize_real_dataset(
            root,
            num_shards=args.num_shards,
            samples_per_shard=args.samples_per_shard,
            image_size=args.crop,
        )
        print(f"Materialized dataset at {root}")
        return

    tmp_ctx: Any = (
        nullcontext(args.data_dir)
        if args.data_dir
        else tempfile.TemporaryDirectory(prefix="fbload_bench_", ignore_cleanup_errors=True)
    )
    with tmp_ctx as tmp:
        root = Path(args.data_dir) if args.data_dir else Path(tmp)
        if args.data_dir is None:
            log.info("Materializing real JPEG dataset under %s", root)
            tar_source, img_root = materialize_real_dataset(
                root,
                num_shards=args.num_shards,
                samples_per_shard=args.samples_per_shard,
                image_size=args.crop,
            )
        else:
            shard_dir = root / "shards"
            shards = sorted(shard_dir.glob("shard-*.tar"))
            if not shards:
                raise SystemExit(f"--data-dir missing shards in {shard_dir}")
            tar_source = [str(p.resolve()) for p in shards]
            img_root = root / "imagefolder"

        print(f"Platform: {platform.platform()}")
        print(f"Python: {sys.version.split()[0]}  Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
        print(f"Dataset: {args.num_shards}x{args.samples_per_shard} JPEGs @ {args.crop}px")
        print(f"Soak: {args.steps} steps  batch={args.batch_size}  workers={num_workers}")
        print(f"Tuning: pin_memory={pin}  persistent_workers={persistent}  prefetch_factor={prefetch}")

        reports: dict[str, SoakReport] = {}

        if not args.skip_fbload:
            it, shutdown = _build_fbload_loader(
                tar_source,
                batch_size=args.batch_size,
                num_workers=num_workers,
                steps=args.steps,
                crop=args.crop,
                pin_memory=pin,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
            )
            try:
                reports["fbload"] = run_soak(
                    "fbload",
                    it,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    num_workers=num_workers,
                    sample_interval=args.sample_interval,
                    cuda_profile=args.cuda_profile,
                    cuda_profile_steps=args.cuda_profile_step,
                )
            finally:
                shutdown()
            print_report(reports["fbload"])

        if not args.skip_pytorch:
            # PyTorch baseline uses same worker count for fair IPC comparison
            pw = num_workers if num_workers > 0 else 0
            it2, shutdown2 = _build_pytorch_loader(
                img_root,
                batch_size=args.batch_size,
                num_workers=pw,
                crop=args.crop,
                pin_memory=pin,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
                steps=args.steps,
            )
            try:
                reports["pytorch"] = run_soak(
                    "pytorch_native",
                    it2,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    num_workers=pw,
                    sample_interval=args.sample_interval,
                    cuda_profile=False,
                    cuda_profile_steps=0,
                )
            finally:
                shutdown2()
            print_report(reports["pytorch"])

        if "fbload" in reports and "pytorch" in reports:
            compare_reports(reports["fbload"], reports["pytorch"])

        if args.json_out:
            args.json_out.write_text(json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2))
            print(f"\nWrote JSON: {args.json_out}")

    # Windows: temp JPEG/tar handles may lag; ignore cleanup errors after explicit shutdown.


if __name__ == "__main__":
    main()
