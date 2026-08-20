# fbload

Cross-platform PyTorch data loading with sequential tar shards (WebDataset-style),
optional NVIDIA DALI on Linux CUDA, and uint8 batches until GPU normalization.

## Features

- **Drop-in API** — `fbloader.DataLoader` wraps map-style datasets, local/remote tar shards, and optional MosaicML Streaming.
- **Equal steps per epoch** — iterable pipelines use `with_epoch(steps)` for reproducible training loops and DDP.
- **uint8 on host** — JPEG decode stays on CPU; normalize on device via `gpu_normalize()`.
- **Spawn-safe** — Windows multiprocessing guards and pickle-safe dataset classes.
- **Optional backends** — `wds` (default), `torch`, `dali` (Linux+CUDA), `streaming` (MosaicML).

## Installation

Install PyTorch from [pytorch.org](https://pytorch.org) first so pip does not replace a CUDA build with a CPU wheel.

```bash
python -m pip install torch
python -m pip install fbload
```

Optional backends (install separately; not in the locked core graph):

```bash
python -m pip install mosaicml-streaming   # MosaicML Streaming backend
python -m pip install nvidia-dali-cuda120 --extra-index-url https://pypi.nvidia.com  # Linux + CUDA only
```

Development (kept out of `uv.lock` so Sonar SCA scans production deps only):

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

## Quick start

```python
import fbloader

if __name__ == "__main__":
    fbloader.mark_main()
    loader = fbloader.DataLoader(
        "shards/{00000..00009}.tar",
        batch_size=256,
        crop=224,
        num_workers=4,
    )
    for images, labels in loader:
        images = fbloader.gpu_normalize(images, device="cuda")
        ...
    loader.shutdown()
```

On Windows, call `fbloader.mark_main()` before creating a loader with `num_workers > 0`.

## Backends

| Backend | Use case |
|---------|----------|
| `wds` | Local or remote `.tar` shards (default) |
| `torch` | Map-style `torch.utils.data.Dataset` |
| `dali` | NVIDIA DALI pipeline (Linux + CUDA) |
| `streaming` | MosaicML Streaming datasets |

Pass `backend="wds"` explicitly or rely on auto-selection from the source type.

## Benchmarking

`tests/benchmark_soak.py` runs a fixed-step soak comparing fbload to a tuned PyTorch
`ImageFolder` baseline (`pin_memory`, `persistent_workers`, `prefetch_factor=2`).

```bash
python tests/benchmark_soak.py \
  --steps 10000 \
  --batch-size 32 \
  --num-workers 4 \
  --crop 224 \
  --pin-memory \
  --json-out benchmark_report.json
```

On Linux, wrap with `perf stat` for hardware counters:

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
perf stat -e cycles,instructions,cache-misses,LLC-load-misses \
  python tests/benchmark_soak.py --steps 10000 --batch-size 32 --num-workers 4
```

Benchmark artifacts are gitignored; commit only the script, not local run outputs.

## Hardening / stability audit

`tests/test_mythos_hardening.py` is a multi-stage soak suite that verifies scale-invariant
stability under extended load:

| Assertion | What it proves |
|-----------|----------------|
| RSS flat-line | Memory does not drift after warmup (≤0.05% Linux, ≤0.5% Windows) |
| FD / handle drift | No file descriptor or handle leaks after warmup |
| Context-switch stability | Involuntary ctx rate does not grow over time |
| Throughput invariance | Rolling throughput σ/μ < 2.5% across checkpoints |
| Tensor contiguity | 100% contiguous uint8 batches; held `data_ptr` stable |

**Smoke (CI, ~1k steps):**

```bash
python tests/test_mythos_hardening.py --steps 1000 --quick
python -m pytest tests/test_mythos_hardening.py -k smoke
```

**Full audit (50k steps, nightly or manual):**

```bash
python tests/test_mythos_hardening.py --steps 50000 --batch-size 32 --num-workers 4 --crop 224
```

Telemetry is written to `tests/mythos_telemetry_report.json` (gitignored).
Nightly runs upload this artifact via `.github/workflows/mythos-nightly.yml`.

## Testing

```bash
python -m pytest -q --timeout=120 -m "not slow and not mythos"   # unit tests
python -m pytest -q --timeout=120 -m "slow and not mythos"       # leak soak
python -m pytest tests/test_mythos_hardening.py -k smoke         # hardening smoke
python -m ruff check src tests                                   # lint
python -m bandit -r src -ll -q                                   # security
python -m pytest --cov=fbloader --cov-fail-under=80 -m "not mythos"  # coverage gate
python scripts/perf_gate.py                                      # perf vs PyTorch baseline
```

## Audit readiness (reproducible proof pack)

Run this sequence before submitting to a repo audit platform:

```bash
python -m pip install -e ".[dev]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m ruff check src tests
python -m bandit -r src -ll -q
python -m pytest -q --cov=fbloader --cov-fail-under=80
python -m pytest -m slow -q
python -m pytest tests/test_mythos_hardening.py -k smoke -q

python tests/benchmark_soak.py --steps 10000 --batch-size 32 --num-workers 4 --crop 224 \
  --json-out benchmark_report.json
python scripts/perf_gate.py
```

Expected gates: all tests green, coverage ≥80%, fbload throughput ≥90% of tuned PyTorch baseline.

CI runs on Ubuntu, macOS, and Windows (see `.github/workflows/ci.yml`).

## Project layout

```text
src/fbloader/              Package source
tests/                     Pytest suite
tests/benchmark_soak.py    Comparative throughput benchmark
tests/test_mythos_hardening.py  Stability / hardening audit
examples/minimal.py        Minimal usage example
```

## License

MIT — see [LICENSE](LICENSE).
