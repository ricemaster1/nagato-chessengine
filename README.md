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

- bitboard based board representation
- magic bitboards for sliding pieces
- negamax with alpha-beta pruning
- principal variation search
- iterative deepening
- transposition table
- null move pruning
- late move reductions
- killer moves + history heuristic
- quiescence search
- tapered eval with piece square tables
- pawn structure evaluation
- king safety
- uci protocol

## why "nagato"

named after the IJN Nagato, lead ship of the Nagato-class battleships of the Imperial Japanese Navy
