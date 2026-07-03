# Changelog

All notable changes to Nagato are documented here.

Versioning follows [Semantic Versioning](https://semver.org/):
- **MAJOR** — breaking UCI interface or weight format change
- **MINOR** — new feature, Elo-positive change confirmed on OpenBench
- **PATCH** — bug fix, refactor, or neutral change

Experimental branches use pre-release tags: `vMAJOR.MINOR.0-experiment-name.N`
(e.g. `v1.2.0-sphere-search.1`). These are never marked Latest.

---

## [Unreleased]

### Branch: `experiment/sphere-search`
- Endgame dispatcher refactored from O(S) sequential chain to O(1) packed
  material key dispatch — prepares the endgame library for KQK, KRK, KNNK
- `sphere-search` Cargo feature added: deterministic late-quiet-move
  perturbation (±60 cp, depth ≤ 6, history-heuristic tail only)
- OpenBench local setup: `Nagato` vs `Nagato-Sphere` A/B test configured

---

## [1.1.0] — 2026-03-09

### Summary
Training infrastructure release. Internal Rust training pipeline added;
NNUE architecture unchanged. Benchmark Elo vs reference engines unchanged
from v1.0.0 — this release is infrastructure, not a strength improvement.

### Added
- `nnue/trainer.rs` — full training pipeline: data loader, forward pass,
  WDL-aware loss (sigmoid MSE + cross-entropy), backpropagation, SGD with
  LR step decay
- `train` UCI command — train weights in-process:
  `train input <file> output <file> epochs <n> batch <n> lr <f> lambda <f>`
- King-bucket mapper for NNUE accumulator (merged from
  `experiment/nnue-p2-king-bucket-mapper`)
- KBN vs K endgame evaluator using Delétang's Triangle, A/B-gated behind
  `kbnk-scaling` Cargo feature
- Experience table (`nagato.exp`) — persisted search experience across games
- Syzygy tablebase probing (WDL + DTZ via `shakmaty-syzygy`)
- Lazy SMP multi-threading (merged from `experiment/threads-lazy-smp`)

### Changed
- Datagen: WDL encoding updated (2=white win, 1=draw, 0=white loss)
- Datagen: 16-position varied opening book
- Trainer: LR decay schedule (0.5× every `epochs/4`)

### Benchmark (tc=5+0.05, 180 games)
| Opponent | Score | Elo delta |
|---|---|---|
| Stockfish 18 (handicapped) | 0.0% | — |
| Lc0 v0.32.1 | 45.8% | +16 |
| Boychesser | ~10% | -264 |
| smol.cs | ~15% | -308 |

---

## [1.0.0] — 2026-02-18

### Summary
First stable release. Full UCI engine in Rust with NNUE evaluation.

### Features
- Bitboard representation with magic bitboards
- Alpha-beta with PVS, iterative deepening, aspiration windows
- Null move pruning, reverse futility pruning, late move reductions (LMR)
- Killer moves, history heuristic, counter-move heuristic
- Transposition table (4-entry buckets), Zobrist hashing
- NNUE evaluation (768→128→32→1 architecture) with HCE fallback
- Tapered evaluation with piece-square tables
- Pawn structure, mobility, and static exchange evaluation (SEE)
- UCI protocol — compatible with Arena, CuteChess, and any UCI GUI

---

[Unreleased]: https://github.com/ricemaster1/nagato-chessengine/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ricemaster1/nagato-chessengine/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ricemaster1/nagato-chessengine/releases/tag/v1.0.0
