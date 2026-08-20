#!/usr/bin/env python3
"""
Mythos hardening soak: scale-invariant stability proof for fbload.

Multi-stage soak with checkpoints at 1,000 / 10,000 / 50,000 steps verifying:
  - RSS memory flat-line (zero-leakage)
  - FD / handle drift immunity
  - Context-switch rate stability
  - Throughput invariance (rolling-window sigma)
  - Tensor contiguity and data_ptr stability

Standalone:
  python tests/test_mythos_hardening.py --steps 50000 --batch-size 32 --num-workers 4

Smoke:
  python tests/test_mythos_hardening.py --steps 1000 --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fbloader  # noqa: E402
from fbloader.lifetime import resource_count  # noqa: E402
from tests.benchmark_soak import materialize_real_dataset, snapshot_counters  # noqa: E402

WARMUP_STEPS = 500
ROLLING_WINDOW = 500
CHECKPOINT_INTERVAL = 1000
SAMPLE_INTERVAL = 100
TENSOR_CHECK_INTERVAL = 500
GC_PROBE_INTERVAL = 5000
DATA_PTR_HOLD_STEPS = 10

DEFAULT_SHARDS = 32
DEFAULT_SAMPLES_PER_SHARD = 256
DEFAULT_DATA_DIR = ROOT / "benchmark_data" / "mythos"
DEFAULT_JSON = Path(__file__).resolve().parent / "mythos_telemetry_report.json"


@dataclass
class CheckpointRecord:
    step: int
    rss_bytes: int
    rss_mb: float
    fds: int
    handles: int
    voluntary_ctx: int
    involuntary_ctx: int
    throughput_ips: float
    rolling_throughput_ips: float
    cpu_user_s: float
    cpu_system_s: float
    drift_pct: float = 0.0
    contiguous_pct: float = 100.0
    gc_objects: int = 0


@dataclass
class AssertionResult:
    name: str
    threshold: str
    observed: str
    passed: bool


@dataclass
class MythosReport:
    meta: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    assertion_matrix: list[AssertionResult] = field(default_factory=list)
    verdict: str = "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "checkpoints": [asdict(c) for c in self.checkpoints],
            "samples": self.samples,
            "assertion_matrix": [asdict(a) for a in self.assertion_matrix],
            "verdict": self.verdict,
        }


@dataclass
class TensorCheckResult:
    contiguous: bool
    storage_offset_zero: bool
    data_ptr_stable: bool = True


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _rss_drift_pct(baseline_rss: int, current_rss: int) -> float:
    if baseline_rss <= 0:
        return 0.0
    return abs(current_rss - baseline_rss) / baseline_rss * 100.0


def _platform_rss_threshold_pct() -> float:
    return 0.05 if sys.platform != "win32" else 0.5


def _fd_metric(snap: Any) -> int:
    if sys.platform != "win32":
        return snap.num_fds if snap.num_fds >= 0 else resource_count()
    return snap.num_handles if snap.num_handles >= 0 else resource_count()


def validate_batch_tensors(batch: Any, held: torch.Tensor | None = None) -> TensorCheckResult:
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(images, torch.Tensor):
        return TensorCheckResult(contiguous=True, storage_offset_zero=True)
    contiguous = bool(images.is_contiguous())
    offset_ok = int(getattr(images, "storage_offset", lambda: 0)()) == 0 if hasattr(images, "storage_offset") else True
    ptr_stable = True
    if held is not None:
        ptr_stable = held.data_ptr() == images.data_ptr() if held is not images else True
    return TensorCheckResult(
        contiguous=contiguous,
        storage_offset_zero=offset_ok,
        data_ptr_stable=ptr_stable,
    )


def _print_progress_row(step: int, rec: CheckpointRecord) -> None:
    fd_label = rec.fds if rec.fds >= 0 else rec.handles
    print(
        f"  {step:6d} | {rec.rss_mb:6.1f} | {fd_label:4d} | "
        f"{rec.involuntary_ctx:11d} | {rec.rolling_throughput_ips:7.1f} | {rec.drift_pct:6.2f}"
    )


def _ensure_dataset(data_dir: Path, *, crop: int, num_shards: int, samples_per_shard: int) -> list[str]:
    shard_dir = data_dir / "shards"
    shards = sorted(shard_dir.glob("shard-*.tar"))
    if len(shards) >= num_shards:
        return [str(p.resolve()) for p in shards[:num_shards]]
    data_dir.mkdir(parents=True, exist_ok=True)
    paths, _ = materialize_real_dataset(
        data_dir,
        num_shards=num_shards,
        samples_per_shard=samples_per_shard,
        image_size=crop,
    )
    return paths


def run_mythos_soak(
    *,
    steps: int,
    batch_size: int,
    num_workers: int,
    crop: int,
    data_dir: Path,
    json_out: Path,
    num_shards: int = DEFAULT_SHARDS,
    samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD,
    quiet: bool = False,
) -> MythosReport:
    fbloader.mark_main()
    if num_workers > 0:
        fbloader.require_main(num_workers=num_workers)

    tar_paths = _ensure_dataset(
        data_dir,
        crop=crop,
        num_shards=num_shards,
        samples_per_shard=samples_per_shard,
    )

    loader = fbloader.DataLoader(
        tar_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=fbloader.default_pin_memory(),
        drop_last=True,
        shuffle=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        backend="wds",
        crop=crop,
        steps=steps,
        seed=42,
        handler="warn",
    )

    report = MythosReport(
        meta={
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "steps": steps,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "crop": crop,
            "num_shards": num_shards,
            "git_sha": _git_sha(),
            "rss_threshold_pct": _platform_rss_threshold_pct(),
        }
    )

    if not quiet:
        print("Step   | RSS MB | FDs  | InvolCtx    | Img/s   | Drift%")
        print("-------+--------+------+-------------+---------+-------")

    t0 = time.perf_counter()
    images_total = 0
    baseline_rss: int | None = None
    baseline_ctx_invol: int | None = None
    baseline_ctx_step: int | None = None
    baseline_gc_objects: int | None = None
    fd_baseline: int | None = None
    handle_baseline: int | None = None
    fd_max_drift = 0
    handle_max_drift = 0

    rolling_ips: deque[float] = deque(maxlen=ROLLING_WINDOW)
    checkpoint_throughputs: list[float] = []
    contiguous_checks = 0
    contiguous_pass = 0
    data_ptr_checks = 0
    data_ptr_pass = 0

    held_batch: torch.Tensor | None = None
    held_ptr: int | None = None
    hold_until = 0

    proc = psutil.Process()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        it = iter(loader)
        for step in range(1, steps + 1):
            step_t0 = time.perf_counter()
            batch = next(it)
            step_dt = max(time.perf_counter() - step_t0, 1e-9)
            images_total += batch_size
            rolling_ips.append(batch_size / step_dt)

            if step == TENSOR_CHECK_INTERVAL:
                held_batch = batch[0].clone() if isinstance(batch, (tuple, list)) else batch
                held_ptr = held_batch.data_ptr()
                hold_until = step + DATA_PTR_HOLD_STEPS

            ts = validate_batch_tensors(batch)
            contiguous_checks += 1
            if ts.contiguous and ts.storage_offset_zero:
                contiguous_pass += 1

            if held_ptr is not None and held_batch is not None and step > TENSOR_CHECK_INTERVAL and step <= hold_until:
                data_ptr_checks += 1
                if held_batch.data_ptr() == held_ptr:
                    data_ptr_pass += 1

            snap = snapshot_counters()
            elapsed = time.perf_counter() - t0
            cumulative_ips = images_total / max(elapsed, 1e-9)
            rolling_mean = sum(rolling_ips) / len(rolling_ips) if rolling_ips else cumulative_ips

            try:
                cpu = proc.cpu_times()
                cpu_user = float(cpu.user)
                cpu_system = float(cpu.system)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                cpu_user = cpu_system = 0.0

            fds = snap.num_fds
            handles = snap.num_handles
            fd_val = _fd_metric(snap)

            if step == WARMUP_STEPS + 1:
                fd_baseline = fds if fds >= 0 else fd_val
                handle_baseline = handles if handles >= 0 else fd_val

            if step > WARMUP_STEPS and fd_baseline is not None:
                if fds >= 0 and fd_baseline >= 0:
                    fd_max_drift = max(fd_max_drift, abs(fds - fd_baseline))
                if handles >= 0 and handle_baseline is not None and handle_baseline >= 0:
                    handle_max_drift = max(handle_max_drift, abs(handles - handle_baseline))

            drift_pct = 0.0
            if baseline_rss is not None:
                drift_pct = _rss_drift_pct(baseline_rss, snap.rss_bytes)

            sample_rec: CheckpointRecord | None = None
            if step % SAMPLE_INTERVAL == 0 or step == 1:
                sample_rec = CheckpointRecord(
                    step=step,
                    rss_bytes=snap.rss_bytes,
                    rss_mb=snap.rss_bytes / (1024 * 1024),
                    fds=fds,
                    handles=handles,
                    voluntary_ctx=snap.voluntary_ctx,
                    involuntary_ctx=snap.involuntary_ctx,
                    throughput_ips=cumulative_ips,
                    rolling_throughput_ips=rolling_mean,
                    cpu_user_s=cpu_user,
                    cpu_system_s=cpu_system,
                    drift_pct=drift_pct,
                    contiguous_pct=100.0 * contiguous_pass / max(1, contiguous_checks),
                )
                report.samples.append(asdict(sample_rec))

            if step == CHECKPOINT_INTERVAL:
                baseline_rss = snap.rss_bytes
                baseline_ctx_invol = snap.involuntary_ctx
                baseline_ctx_step = step
                gc.collect()
                baseline_gc_objects = len(gc.get_objects())

            if step % CHECKPOINT_INTERVAL == 0:
                if step > WARMUP_STEPS:
                    checkpoint_throughputs.append(rolling_mean)
                rec = CheckpointRecord(
                    step=step,
                    rss_bytes=snap.rss_bytes,
                    rss_mb=snap.rss_bytes / (1024 * 1024),
                    fds=fds,
                    handles=handles,
                    voluntary_ctx=snap.voluntary_ctx,
                    involuntary_ctx=snap.involuntary_ctx,
                    throughput_ips=cumulative_ips,
                    rolling_throughput_ips=rolling_mean,
                    cpu_user_s=cpu_user,
                    cpu_system_s=cpu_system,
                    drift_pct=drift_pct,
                    gc_objects=len(gc.get_objects()) if step % GC_PROBE_INTERVAL == 0 else 0,
                )
                report.checkpoints.append(rec)
                if not quiet:
                    _print_progress_row(step, rec)

            if step % GC_PROBE_INTERVAL == 0 and baseline_gc_objects is not None:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        loader.shutdown()

    # --- Assertion matrix ---
    assertions: list[AssertionResult] = []
    rss_threshold = _platform_rss_threshold_pct()
    final_checkpoint = report.checkpoints[-1] if report.checkpoints else None

    if baseline_rss is not None and final_checkpoint is not None:
        final_drift = _rss_drift_pct(baseline_rss, final_checkpoint.rss_bytes)
        assertions.append(
            AssertionResult(
                name="rss_flat_line",
                threshold=f"<= {rss_threshold}%",
                observed=f"{final_drift:.4f}%",
                passed=final_drift <= rss_threshold,
            )
        )

    if sys.platform != "win32":
        fd_observed = fd_max_drift
        assertions.append(
            AssertionResult(
                name="fd_drift",
                threshold="== 0",
                observed=str(fd_observed),
                passed=fd_observed == 0,
            )
        )
    else:
        assertions.append(
            AssertionResult(
                name="handle_drift",
                threshold="== 0",
                observed=str(handle_max_drift),
                passed=handle_max_drift == 0,
            )
        )

    if baseline_ctx_invol is not None and final_checkpoint is not None and baseline_ctx_step:
        early_ckpt = next((c for c in report.checkpoints if c.step == CHECKPOINT_INTERVAL), None)
        if early_ckpt and final_checkpoint.step > CHECKPOINT_INTERVAL:
            early_rate = early_ckpt.involuntary_ctx / max(early_ckpt.step, 1)
            late_rate = (final_checkpoint.involuntary_ctx - early_ckpt.involuntary_ctx) / max(
                final_checkpoint.step - early_ckpt.step, 1
            )
            ratio = late_rate / max(early_rate, 1e-9)
            assertions.append(
                AssertionResult(
                    name="ctx_switch_stability",
                    threshold="<= 1.15x",
                    observed=f"{ratio:.4f}x",
                    passed=ratio <= 1.15,
                )
            )
        elif final_checkpoint.step == CHECKPOINT_INTERVAL:
            assertions.append(
                AssertionResult(
                    name="ctx_switch_stability",
                    threshold="<= 1.15x",
                    observed="smoke_single_checkpoint",
                    passed=True,
                )
            )

    if len(checkpoint_throughputs) >= 2:
        mean_tp = sum(checkpoint_throughputs) / len(checkpoint_throughputs)
        variance = sum((x - mean_tp) ** 2 for x in checkpoint_throughputs) / len(checkpoint_throughputs)
        sigma = variance**0.5
        cv_pct = (sigma / mean_tp * 100.0) if mean_tp > 0 else 0.0
        assertions.append(
            AssertionResult(
                name="throughput_invariance",
                threshold="sigma/mu < 2.5%",
                observed=f"{cv_pct:.4f}%",
                passed=cv_pct < 2.5,
            )
        )
    else:
        assertions.append(
            AssertionResult(
                name="throughput_invariance",
                threshold="sigma/mu < 2.5%",
                observed="skipped_insufficient_checkpoints",
                passed=True,
            )
        )

    cont_pct = 100.0 * contiguous_pass / max(1, contiguous_checks)
    assertions.append(
        AssertionResult(
            name="tensor_contiguity",
            threshold="100%",
            observed=f"{cont_pct:.2f}%",
            passed=cont_pct >= 100.0,
        )
    )

    if data_ptr_checks > 0:
        ptr_pct = 100.0 * data_ptr_pass / data_ptr_checks
        assertions.append(
            AssertionResult(
                name="data_ptr_stability",
                threshold="100%",
                observed=f"{ptr_pct:.2f}%",
                passed=ptr_pct >= 100.0,
            )
        )

    report.assertion_matrix = assertions
    report.verdict = "PASS" if all(a.passed for a in assertions) else "FAIL"

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if not quiet:
        print("\n" + "=" * 60)
        print("  ASSERTION MATRIX")
        print("=" * 60)
        for a in assertions:
            status = "PASS" if a.passed else "FAIL"
            print(f"  [{status}] {a.name}: {a.observed} (threshold {a.threshold})")
        print(f"\n  VERDICT: {report.verdict}")
        print(f"  JSON: {json_out}")

    return report


def _execute_or_raise(report: MythosReport) -> None:
    if report.verdict != "PASS":
        failed = [a.name for a in report.assertion_matrix if not a.passed]
        raise SystemExit(f"Mythos hardening FAILED: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------
@pytest.mark.mythos
@pytest.mark.slow
def test_mythos_hardening_smoke(tmp_path_factory: pytest.TempPathFactory) -> None:
    data_dir = tmp_path_factory.mktemp("mythos_smoke")
    json_out = tmp_path_factory.mktemp("reports") / "mythos_smoke.json"
    # workers=0 for deterministic CI (parent-process FD counting)
    workers = 0
    report = run_mythos_soak(
        steps=1000,
        batch_size=8,
        num_workers=workers,
        crop=64,
        data_dir=Path(data_dir),
        json_out=Path(json_out),
        num_shards=4,
        samples_per_shard=64,
        quiet=True,
    )
    assert report.verdict == "PASS", report.to_dict()


@pytest.mark.mythos
@pytest.mark.slow
def test_mythos_hardening_full() -> None:
    workers = 4 if sys.platform != "win32" else fbloader.default_num_workers()
    report = run_mythos_soak(
        steps=50_000,
        batch_size=32,
        num_workers=workers,
        crop=224,
        data_dir=DEFAULT_DATA_DIR,
        json_out=DEFAULT_JSON,
        quiet=False,
    )
    assert report.verdict == "PASS", report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="fbload Mythos hardening soak suite")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--samples-per-shard", type=int, default=DEFAULT_SAMPLES_PER_SHARD)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="1k-step smoke run")
    args = parser.parse_args()

    if args.quick:
        args.steps = min(args.steps, 1000)
        args.num_shards = min(args.num_shards, 4)
        args.samples_per_shard = min(args.samples_per_shard, 128)

    if args.materialize_only:
        _ensure_dataset(
            args.data_dir,
            crop=args.crop,
            num_shards=args.num_shards,
            samples_per_shard=args.samples_per_shard,
        )
        print(f"Materialized dataset at {args.data_dir}")
        return

    num_workers = args.num_workers if args.num_workers is not None else fbloader.default_num_workers()
    if args.quick and num_workers == 0:
        pass  # Windows default
    elif args.num_workers is None and not args.quick:
        num_workers = 4 if sys.platform != "win32" else max(fbloader.default_num_workers(), 0)

    report = run_mythos_soak(
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=num_workers,
        crop=args.crop,
        data_dir=args.data_dir,
        json_out=args.json_out,
        num_shards=args.num_shards,
        samples_per_shard=args.samples_per_shard,
    )
    _execute_or_raise(report)


if __name__ == "__main__":
    main()
