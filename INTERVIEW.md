# fbloader — Interview Guide (Roman Urdu)
**Author: Abdullah Hanif | Role: AI Native Architect**
**Version: 0.2.0 (Audit Ready)**
**Repo: https://github.com/abdullahhanif-001/fbloader**

---

## 0. Apna Introduction (30 second pitch)

> "Main Abdullah Hanif hoon — AI Native Architect. Matlab main production ML systems design karta hoon: data pipeline se le kar training aur deployment tak.
>
> Mera project **fbloader** hai — yeh PyTorch ka cross-platform DataLoader hai jo real engineering problems solve karta hai: Windows par multiprocessing crash, memory waste, slow disk reads, aur unstable training steps.
>
> Pure Python hai, koi C++ compile nahi. Windows, macOS, Linux — teeno par chalta hai. Benchmark par PyTorch se **13%+ faster** hai, aur 50,000 step ki stability proof hai — memory leak zero, FD leak zero."

---

## 1. Ya Kya Hai? (What is fbloader?)

**fbloader** ek **drop-in PyTorch DataLoader replacement** hai.

- Images ko **tar shards** se sequentially padhta hai (WebDataset style)
- Random file open/close ki jagah **ek shard = ek open()**
- Images host par **uint8** rehti hain — GPU par normalize hoti hain
- Same API: `fbloader.DataLoader(source, batch_size=256)`

Install:
```bash
pip install torch
pip install fbload
import fbloader
```

---

## 2. Ku Banaya? (Why I Built It)

PyTorch ka native `ImageFolder + DataLoader` scale par problems deta hai:

| Problem | Asar |
|---------|------|
| Har sample ke liye alag file open | Disk seek slow — 10k+ images par bottleneck |
| CPU par fp32 normalize | 4x zyada RAM waste |
| Windows par `num_workers > 0` | Pickle error — nested class/lambda crash |
| IterableDataset exhaust ho jata hai | DDP workers desync — unequal steps |
| File handles leak | Hours ke training ke baad crash |

**Mera solution:**
- Sequential tar reads
- uint8 host → GPU normalize
- Module-level pickle-safe classes
- Fixed `steps` per epoch — har worker equal batches

---

## 3. Kis Language Mein Hai? Aur Ku Python?

### Language Stack

| Layer | Technology |
|-------|------------|
| Core code | **Python 3.10+** |
| ML framework | **PyTorch** (peer dependency) |
| Image decode | **Pillow** (libjpeg andar) |
| Tar reading | **WebDataset** + custom `LocalTarEpoch` |
| Optional GPU | **NVIDIA DALI** (Linux + CUDA only) |
| Testing | **pytest + psutil** |
| CI | **GitHub Actions** (Ubuntu, macOS, Windows) |

### Q: "Python ku? C++ fast nahi hota?"

**Jawab:**
> "Bottleneck JPEG decode hai — Pillow ke andar libjpeg C mein chalta hai. Python overhead nahi hai asli problem.
> Maine Linux par C++ io_uring test kiya — sequential tar reads par sirf **3% gain**. Windows/macOS support kho dena worth nahi tha.
> Physics limit = JPEG decode speed, language nahi."

### Q: "AI Native Architect hone ka matlab kya? AI tools use kiye?"

**Jawab (honest + strong):**
> "AI Native Architect matlab main AI tools ko **senior engineer ki tarah** use karta hoon — code likhwane ke liye nahi, **system design aur verification** ke liye.
>
> AI-assisted workflow se main:
> - Architecture explore karta hoon (multiple backends, benchmark design)
> - Adversarial tests likhwata hoon (empty shard, corrupt JPEG, worker crash)
> - CI gates aur stability suite automate karta hoon
>
> Lekin **har fix verify hota hai** — pytest, 50k step soak, benchmark JSON. AI code blindly accept nahi karta. Test green nahi = ship nahi.
>
> fbloader mera design decision hai: sequential I/O, uint8 pipeline, spawn-safe workers. AI ne implement help ki — main ne architecture, benchmarks, aur proof own kiye."

---

## 4. Kaise Kaam Karta Hai? (Architecture)

```
User code
    ↓
fbloader.DataLoader(source, batch_size, num_workers)
    ↓
Backend auto-select:
    Dataset object?     → torch backend
    Remote s3/http?     → webdataset (resampled)
    Linux+CUDA+DALI?    → DALI pipeline
    Local tar shards?   → LocalTarEpoch (default)
    ↓
PyTorch DataLoader (workers, prefetch, pin_memory)
    ↓
OwnStorageIterator (tensor safety — held batch corrupt nahi hota)
    ↓
uint8 NCHW batch → gpu_normalize() → fp32 on GPU
```

### Key Files

| File | Kaam |
|------|------|
| `dataloader.py` | Main API, backend selection |
| `backends/wds.py` | Tar read, JPEG decode, worker split |
| `lifetime.py` | gpu_normalize, leak tracking, tensor clone |
| `_spawn.py` | Windows spawn guard |
| `test_mythos_hardening.py` | 50k step stability proof |
| `benchmark_soak.py` | PyTorch vs fbloader benchmark |

---

## 5. Linux Par Crash-Proof Ku Hai?

### 5.1 FD Leak = Zero

Linux par har open file ek **file descriptor (FD)** leta hai. Leak = training ke dauran crash (`Too many open files`).

**Fix:**
- Tar shard = ek `open()` per shard, samples sequentially
- `resource_count()` → `/proc/self/fd` count karta hai
- Mythos suite: **FD drift == 0** after warmup

### 5.2 Memory Leak = Zero

**Fix:**
- Fresh tensors har batch: `np.array(copy=True)` + `torch.from_numpy()`
- DALI path: `OwnStorageIterator` clones GPU buffers
- Mythos: RSS drift ≤ 0.05% over 50k steps

### 5.3 Worker Deadlock / Zombie = Prevented

**Fix:**
- `LocalTarEpoch` — har worker ko equal step share (`steps // num_workers`)
- Module-level classes — pickle-safe for Linux fork/spawn
- `shutdown()` — workers clean band karta hai
- Test: `test_workers_multiprocess.py` — 2 workers, 20 steps, no hang

### 5.4 Corrupt JPEG = No Crash

**Fix:**
- `handler="warn"` — corrupt sample skip, warning log
- Test: `test_corrupt_jpeg.py`

### 5.5 Empty/Corrupt Shard = Clear Error

**Fix:**
- 8 empty passes ke baad: `RuntimeError("no valid samples in shards")`
- Test: `test_empty_shard.py`

---

## 6. Stability Kaise Maintain Karta Hai? (Mythos Suite)

**Mythos** = 50,000 step ka mathematical stability proof.

| Check | Threshold | Proof |
|-------|-----------|-------|
| RSS memory | ≤0.05% drift | No memory leak |
| FD/handle drift | == 0 | No resource leak |
| Context switches | ≤1.15x growth | No worker contention |
| Throughput variance | σ/μ < 2.5% | No GC/thermal spikes |
| Tensor contiguity | 100% | Clean uint8 output |
| data_ptr stability | 100% | Held batch safe after next() |

**CI:**
- Har push: 1,000 step smoke (~20 sec)
- Nightly: 50,000 step full audit + JSON artifact

**Version 0.2.0 par extra gates:**
- Coverage ≥ 80%
- ruff lint + bandit security
- Perf gate: fbload ≥ 90% PyTorch baseline throughput

---

## 7. Market Mein Acha Ku Hai?

### vs PyTorch ImageFolder

| Metric | fbloader | PyTorch | Winner |
|--------|----------|---------|--------|
| Throughput (10k steps) | 1,676 img/s | 1,486 img/s | **fbloader +13.2%** |
| Stability (CV) | 8.3% | 11.2% | **fbloader** |
| Cross-platform workers | ✅ | ❌ (Windows crash) | **fbloader** |
| uint8 pipeline | ✅ | ❌ | **fbloader** |
| Equal steps / DDP | ✅ | Manual | **fbloader** |
| Zero leak proof | ✅ (50k steps) | ❌ | **fbloader** |

### vs NVIDIA DALI

- DALI = Linux + CUDA only
- fbloader = DALI auto-use karta hai jab available, warna WebDataset fallback
- **Ek codebase, teen platforms**

### vs SPDL / torchdata.nodes

- SPDL faster hai lekin complex setup, Linux-focused
- fbloader = **pip install, same API, proven stability suite**
- Interview pitch: "Production-ready with proof, not just speed claims"

### Market Position

> "Engineers jo Windows par bhi train karte hain, jo leak-free 24-hour soak chahiye, jo fair PyTorch benchmark chahte hain — unke liye fbloader ready hai."

---

## 8. Benchmark Numbers (Yaad Rakho)

**Setup:** crop=224, batch=32, num_workers=4, 10k steps, 3 runs

- fbloader mean: **1,676 img/s**
- PyTorch tuned baseline: **1,486 img/s**
- Lead: **+13.2%**
- FD drift: **0** dono par
- Tensor contiguity: **100%**

**Quick smoke (200 steps, Windows):**
- fbloader: ~900+ img/s
- PyTorch: ~750 img/s
- Ratio: **≥1.22x** (perf gate pass)

**Fair baseline tuning (important!):**
PyTorch baseline bhi tuned hai: `pin_memory=True, persistent_workers=True, prefetch_factor=2, num_workers=4` — taake comparison honest ho.

---

## 9. Hard Technical Challenges (Deep Q&A)

### Q: Windows pickle bug kya tha?

**A:**
> Nested class `LocalTarEpoch` factory function ke andar thi. Windows `spawn` use karta hai — workers ko dataset pickle karna padta hai. Nested class pickle nahi hoti.
> Error: `Can't get local object 'LocalTarEpoch'`
> Fix: Class ko module level par move kiya `wds.py` mein. PyTorch baseline ke lambdas bhi module level.

### Q: Multiprocess par double batches bug?

**A:**
> Har worker poore `steps` produce kar raha tha — 2 workers = 2x batches.
> Fix: `worker_steps = steps // num_workers` — har worker apna fair share.

### Q: Held tensor corrupt hota hai next() ke baad?

**A:**
> DALI GPU buffers reuse karta hai. Fix: `OwnStorageIterator` clone karta hai. WebDataset path already fresh copy banata hai.
> Test: `test_hold_across_next.py` + `test_hold_extended.py` (100+ next calls)

### Q: uint8 ku host par?

**A:**
> Batch 32×3×224×224:
> - uint8 = **4.8 MB** host RAM
> - fp32 = **19.2 MB** host RAM (4x waste)
> GPU par ek kernel mein normalize — PCIe bandwidth bachao.

### Q: WebDataset directly ku nahi?

**A:**
> WebDataset pipeline library hai. fbloader usko wrap karta hai familiar `DataLoader()` API mein + backend auto-select + spawn safety + leak tracking + equal steps. Learning curve kam.

---

## 10. AI Native Architect — Special Questions

Ye questions specifically **AI-assisted architect** se pooche jayenge:

### Q: "Tum ne pura code AI se likhwaya?"

**A:**
> "Nahi. Main ne **architecture decide ki** — sequential tar, uint8 pipeline, Mythos stability suite, benchmark methodology. AI ne implementation speed di. Main ne **verify kiya** — 55 tests, 80% coverage, 50k soak, perf gate. Agar test fail = fix, bypass nahi."

### Q: "AI code trust kaise karte ho production mein?"

**A:**
> "Teen rules:
> 1. Har bug = pehle failing test, phir fix
> 2. CI gates: lint, security, coverage, perf regression
> 3. Long soak proofs — 50k steps mathematical assertions
> AI suggestion accept tab hota hai jab green CI + evidence ho."

### Q: "AI tools se kya accelerate hua?"

**A:**
> "- Adversarial test cases (empty shard, concurrent loaders, invalid args)
> - CI workflow wiring (ruff, bandit, pytest-cov)
> - Benchmark script polish
> - Documentation
> Manual likhne mein weeks — AI-assisted mein days. Lekin **design decisions mine hain**."

### Q: "AI Native Architect vs normal developer?"

**A:**
> "Normal dev code likhta hai. AI Native Architect **system design + verification loop** chalata hai:
> - Problem define (PyTorch pain points)
> - Architecture pick (Python, WebDataset, uint8)
> - AI se implement + test generate
> - Benchmark prove karo
> - Ship jab evidence ho, jab feel ho nahi"

### Q: "Koi AI watermark / shortcut repo mein?"

**A:**
> "Haan. Git history clean — sirf **Abdullah Hanif** author. Koi third-party co-author nahi. Tests professional naming (`test_*`). Benchmark reproducible. Engineer-ready repo."

### Q: "Agar interviewer code padhe to kya dikhega?"

**A:**
> "Module-level pickle-safe classes, explicit error messages, `shutdown()` lifecycle, OS-level resource tracking, 55 pytest functions, 3-OS CI. Yeh junior AI output nahi — yeh production engineering hai jo AI se faster banaya gaya."

---

## 11. Common Interview Q&A (Quick Fire)

| Question | Short Answer |
|----------|-------------|
| DDP support? | Haan — `resampled=True`, equal steps per rank |
| Corrupt JPEG? | Skip + warn, no crash |
| ImageNet scale? | Haan — 1000+ shards, resampled load balancing |
| DALI required? | Nahi — optional, auto-fallback |
| torch dependency? | Peer dep — CUDA install overwrite na ho |
| Memory map vs tar? | JPEG variable length — tar self-describing, WebDataset standard |
| gpu_normalize overhead? | Minimal — ek GPU kernel chain |
| Test count? | 55 tests, 80%+ coverage, mythos smoke + nightly 50k |
| Version? | 0.2.0 Audit Ready |
| GitHub? | abdullahhanif-001/fbloader |

---

## 12. Demo Commands (Interview Mein Chala Sakte Ho)

```bash
# Install
pip install torch
pip install -e ".[dev]"

# Unit tests
pytest -m "not mythos" -q

# Stability smoke
pytest tests/test_mythos_hardening.py -k smoke -q

# Benchmark vs PyTorch
python tests/benchmark_soak.py --steps 200 --quick

# Perf gate
python scripts/perf_gate.py

# Coverage
pytest --cov=fbloader --cov-fail-under=80 -q
```

---

## 13. Closing Line (Interview End)

> "fbloader sirf fast DataLoader nahi — yeh **proven stable** DataLoader hai. 50,000 steps, zero leaks, cross-platform workers, fair PyTorch benchmark. Main AI Native Architect hoon — main ne design kiya, AI ne speed di, tests ne proof diya. Production mein ship tab jab evidence ho."

---

**License:** MIT | **Python:** >=3.10 | **Author:** Abdullah Hanif
