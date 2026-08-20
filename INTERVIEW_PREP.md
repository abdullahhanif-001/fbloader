# fbloader — Interview Prep (Roman Urdu)
**Author: Abdullah Hanif | AI Native Architect | v0.2.0**

> **Poori detail ke liye:** [`INTERVIEW.md`](INTERVIEW.md) — wahan sab kuch Roman Urdu mein hai: architecture, stability, market comparison, AI Native Architect questions.

---

## Ek Line Mein

**fbloader** = cross-platform PyTorch DataLoader jo tar shards sequentially padhta hai, uint8 host par rakhta hai, GPU par normalize karta hai, Windows/Linux/macOS par crash-free workers deta hai.

---

## Ku Banaya?

PyTorch ImageFolder scale par slow hai (random seeks), memory zyada use karta hai (fp32 CPU par), Windows par workers crash karte hain, aur long training mein FD leak hota hai.

---

## Kis Language Mein? Ku?

- **Python 3.10+** + PyTorch + Pillow + WebDataset
- C++ nahi — bottleneck JPEG decode hai (libjpeg), Python overhead nahi
- Pure Python = Windows/macOS/Linux ek codebase

---

## Kaise Kaam Karta Hai?

`DataLoader(source)` → backend auto-select (wds / torch / dali / streaming) → workers + prefetch → uint8 batch → `gpu_normalize()` GPU par

---

## Linux Crash-Proof + Stability

- FD leak: 0 (Mythos 50k step proof)
- Memory drift: ≤0.05%
- Corrupt JPEG: skip, no crash
- Workers: pickle-safe, equal steps per worker
- Tests: 55 pytest, 80% coverage, 3-OS CI

---

## Market Mein Acha Ku?

| | fbloader | PyTorch |
|--|----------|---------|
| Speed | 1,676 img/s | 1,486 img/s (+13%) |
| Windows workers | ✅ | ❌ |
| Leak proof | 50k steps | ❌ |
| uint8 pipeline | ✅ | ❌ |

---

## AI Native Architect — Top 3 Jawab

1. **"AI se likhwaya?"** — Architecture mera, AI ne implement help ki, tests ne verify kiya. Bypass nahi.
2. **"Trust kaise?"** — Failing test pehle, phir fix. CI: lint + security + coverage + perf gate.
3. **"Role kya hai?"** — System design + verification loop. AI tool hai, engineer main hoon.

---

## Repo

https://github.com/abdullahhanif-001/fbloader

**Padh lo:** [`INTERVIEW.md`](INTERVIEW.md) — interview se pehle poori file ek bar.
