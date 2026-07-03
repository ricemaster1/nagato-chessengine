# Acknowledgements

Nagato is original work, but it stands on tools, data, and ideas from others.
Thanks to all of the below.

## Libraries linked into the engine

- **[shakmaty](https://github.com/niklasf/shakmaty)** and
  **[shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy)** — Niklas Fiekas.
  Licensed GPL-3.0-or-later. Used for move/position handling and Syzygy tablebase
  probing ([`src/syzygy.rs`](src/syzygy.rs)). **Nagato's GPL-3.0-or-later license
  follows from linking these.**
- **[rand](https://github.com/rust-random/rand)** — The Rand Project Developers.
  MIT OR Apache-2.0.

## Tooling for training and testing

- **[bullet](https://github.com/jw1912/bullet)** (`bullet_lib`) — Jamie Whiting. MIT.
  The NNUE training framework wrapped by `tools/nagato-trainer`.
- **[sfbinpack](https://crates.io/crates/sfbinpack)** — GPL-3.0. Reader for the
  Stockfish binpack training-data format.
- **[OpenBench](https://github.com/AndyGrant/OpenBench)** — Andrew Grant.
  Distributed SPRT testing.
- **[Cute Chess](https://github.com/cutechess/cutechess)** — Ilari Pihlajisto,
  Arto Jonsson, and contributors. Engine match runner used for local SPRT.

## Data and formats

- **Syzygy endgame tablebases** — Ronald de Man. Probed at runtime for perfect
  endgame play.
- **Stockfish binpack** training-data format — the Stockfish project.

## Techniques

- **NNUE** (efficiently updatable neural network) — originated by Yu Nasu for shogi,
  popularized for chess by the Stockfish project.
- **Magic bitboards**, **Finny tables**, and the standard alpha-beta search
  enhancements (null-move pruning, late move reductions, ProbCut, SEE, history /
  killer heuristics) — the wider computer-chess community, much of it documented on
  the [Chess Programming Wiki](https://www.chessprogramming.org).
