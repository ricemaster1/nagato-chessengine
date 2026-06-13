<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/nagato-dark.png">
    <img src="logo/nagato.png" alt="Nagato" width="170">
  </picture>
</p>

# Nagato

chess engine written in rust

## building

```
cargo build --release
```

binary ends up in `target/release/nagato`

## usage

nagato speaks UCI so just point your gui at the binary (cutechess, arena, etc)

or run it in terminal:
```
./target/release/nagato
```
then type `uci` and go from there

## features

- bitboard board representation with magic bitboards for sliding pieces
- negamax with alpha-beta pruning and principal variation search
- iterative deepening with transposition table
- null move pruning and late move reductions
- killer moves + history heuristic
- quiescence search
- NNUE evaluation (HalfKP, 10 king buckets, horizontal mirroring)
- architecture: 6400→256→pairwise(128)→[4×(128→32→1)] + skip[8] + PSQT[4]
- quantized int16/int8 inference with NEON SIMD
- incremental accumulator updates with Finny cache
- lazy eval with DirtyPiece tracking
- dual network routing (HCE fallback for lopsided material)
- threat/attack feature encoding
- LEB128 weight compression
- architecture config for net parameter experiments
- self-play data generation with opening book and adjudication
- NNUE trainer with WDL-aware loss and LR decay
- experience table for search corrections
- tapered HCE with piece square tables, pawn structure, king safety
- uci protocol

## why "nagato"

named after the IJN Nagato, lead ship of the Nagato-class battleships of the Imperial Japanese Navy
