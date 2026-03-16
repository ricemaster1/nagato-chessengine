#!/usr/bin/env python3
"""
Full Lichess → Nagato training data pipeline.

Reads .pgn.zst files, filters for quality games (≥2000 Elo, ≥3min TC),
samples positions, evaluates with Stockfish, and writes Nagato's 40-byte
packed binary format compatible with src/nnue/trainer.rs parse_entry().

Binary entry format (40 bytes):
  [0..32]  packed board (nibble per square, 2 squares per byte)
  [32]     side to move (0=White, 1=Black)
  [33]     castling rights (KQkq = bits 0-3)
  [34]     en passant file (0-7, 255=none)
  [35]     padding (0)
  [36..38] score i16 LE (from White's perspective)
  [38]     wdl byte (0=loss, 1=draw, 2=win from side-to-move perspective)
  [39]     padding (0)
"""
import chess
import chess.pgn
import chess.engine
import zstandard
import io
import struct
import random
import time
import os
import sys
from pathlib import Path

import nnue_lab_common as c

# ────────────────────────── Configuration ──────────────────────────

GAMES_DIR = Path("games")
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
SF_DEPTH = 14
SF_THREADS = 4
SF_HASH_MB = 256
MIN_ELO = 2000
MIN_TIME_BASE = 180       # seconds (≥3 min base)
SKIP_PLIES = 16           # skip opening moves
POSITIONS_PER_GAME = 2    # positions to sample per game

# Piece encoding: must match Rust datagen::pack_board()
# White: P=1 N=2 B=3 R=4 Q=5 K=6
# Black: P=7 N=8 B=9 R=10 Q=11 K=12
PIECE_NIBBLE = {
    (chess.PAWN,   chess.WHITE): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK,   chess.WHITE): 4,
    (chess.QUEEN,  chess.WHITE): 5,
    (chess.KING,   chess.WHITE): 6,
    (chess.PAWN,   chess.BLACK): 7,
    (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK,   chess.BLACK): 10,
    (chess.QUEEN,  chess.BLACK): 11,
    (chess.KING,   chess.BLACK): 12,
}

# Castling bit encoding matching Rust: K=1, Q=2, k=4, q=8
CASTLING_MAP = {
    chess.BB_H1: 1,   # White kingside
    chess.BB_A1: 2,   # White queenside
    chess.BB_H8: 4,   # Black kingside
    chess.BB_A8: 8,   # Black queenside
}

ENTRY_SIZE = 40


# ────────────────────────── Packing ──────────────────────────

def pack_board(board: chess.Board) -> bytes:
    """Pack board into 32 bytes matching Rust nibble format.
    Square 0 = a1 in python-chess = sq 0 in our engine."""
    packed = bytearray(32)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            nibble = 0
        else:
            nibble = PIECE_NIBBLE[(piece.piece_type, piece.color)]
        byte_idx = sq // 2
        if sq % 2 == 0:
            packed[byte_idx] |= nibble
        else:
            packed[byte_idx] |= (nibble << 4)
    return bytes(packed)


def encode_castling(board: chess.Board) -> int:
    """Encode castling rights matching Rust: K=1 Q=2 k=4 q=8."""
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= 8
    return rights


def encode_ep_file(board: chess.Board) -> int:
    """En passant file (0-7) or 255 if none."""
    if board.ep_square is not None:
        return chess.square_file(board.ep_square)
    return 255


def write_entry(f, board: chess.Board, score_white: int, wdl: int):
    """Write a 40-byte training entry matching Rust datagen format."""
    packed = pack_board(board)
    side = 0 if board.turn == chess.WHITE else 1
    castling = encode_castling(board)
    ep_file = encode_ep_file(board)

    score_clamped = max(-32000, min(32000, score_white))

    entry = bytearray(ENTRY_SIZE)
    entry[0:32] = packed
    entry[32] = side
    entry[33] = castling
    entry[34] = ep_file
    entry[35] = 0  # padding
    struct.pack_into("<h", entry, 36, score_clamped)
    entry[38] = wdl
    entry[39] = 0  # padding
    f.write(entry)


# ────────────────────────── Filtering ──────────────────────────

def parse_time_control(tc_str):
    if not tc_str or tc_str == "-":
        return (0, 0)
    parts = tc_str.split("+")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return (0, 0)


def passes_filter(headers):
    try:
        w_elo = int(headers.get("WhiteElo", "0"))
        b_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if w_elo < MIN_ELO or b_elo < MIN_ELO:
        return False

    base, _ = parse_time_control(headers.get("TimeControl", ""))
    if base < MIN_TIME_BASE:
        return False

    if headers.get("Termination", "") != "Normal":
        return False
    if headers.get("Result", "*") not in ("1-0", "0-1", "1/2-1/2"):
        return False
    return True


def result_to_wdl(result_str, side_to_move):
    """WDL from side-to-move perspective: 2=win, 1=draw, 0=loss."""
    if result_str == "1/2-1/2":
        return 1
    white_wins = result_str == "1-0"
    stm_is_white = side_to_move == chess.WHITE
    if white_wins == stm_is_white:
        return 2
    return 0


# ────────────────────────── Sampling ──────────────────────────

def sample_positions(game, n=POSITIONS_PER_GAME, skip_plies=SKIP_PLIES):
    """Sample random non-terminal positions from the game, skipping opening."""
    board = game.board()
    positions = []
    ply = 0
    for move in game.mainline_moves():
        board.push(move)
        ply += 1
        if ply > skip_plies and not board.is_game_over():
            positions.append((ply, board.copy()))
    if not positions:
        return []
    k = min(n, len(positions))
    return random.sample(positions, k)


# ────────────────────────── Pipeline ──────────────────────────

def run_pipeline(output_path, max_games=None, progress_interval=100):
    """Main pipeline: filter → sample → eval → write."""
    game_files = c.iter_game_source_files()
    if not game_files:
        print("No .pgn or .pgn.zst files found in games/")
        return

    print(f"Found {len(game_files)} PGN source file(s)")
    print(f"Stockfish: {STOCKFISH_PATH}, depth={SF_DEPTH}, threads={SF_THREADS}")
    print(f"Filter: Elo≥{MIN_ELO}, TC≥{MIN_TIME_BASE}s, Normal termination")
    print(f"Output: {output_path}")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})

    t0 = time.time()
    games_seen = 0
    games_passed = 0
    positions_written = 0
    file_idx = 0

    with open(output_path, "wb") as out:
        for game_path in game_files:
            file_idx += 1
            print(f"\n[File {file_idx}/{len(game_files)}] {game_path.relative_to(Path.cwd())}")

            with c.open_pgn_text(game_path) as text_stream:
                while True:
                    if max_games is not None and games_passed >= max_games:
                        break

                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break

                    games_seen += 1

                    if not passes_filter(game.headers):
                        continue

                    games_passed += 1
                    result_str = game.headers["Result"]

                    sampled = sample_positions(game)
                    for ply, board in sampled:
                        # Stockfish eval
                        info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
                        score = info["score"].pov(chess.WHITE)

                        if score.is_mate():
                            mate_in = score.mate()
                            cp_white = 30000 if mate_in > 0 else -30000
                        else:
                            cp_white = score.score()

                        wdl = result_to_wdl(result_str, board.turn)
                        write_entry(out, board, cp_white, wdl)
                        positions_written += 1

                    if games_passed % progress_interval == 0:
                        elapsed = time.time() - t0
                        rate = positions_written / elapsed if elapsed > 0 else 0
                        print(f"  {games_passed} games ({games_seen} seen), "
                              f"{positions_written} positions, "
                              f"{rate:.1f} pos/s, {elapsed:.0f}s")

            if max_games is not None and games_passed >= max_games:
                break

    engine.quit()
    elapsed = time.time() - t0

    file_size = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Games: {games_seen} scanned, {games_passed} passed filter")
    print(f"  Positions: {positions_written}")
    print(f"  Time: {elapsed:.1f}s ({positions_written/elapsed:.1f} pos/s)")
    print(f"  Output: {output_path} ({file_size:,} bytes, "
          f"{file_size // ENTRY_SIZE} entries)")


def verify_output(path, show=5):
    """Read back and verify binary output."""
    file_size = os.path.getsize(path)
    n_entries = file_size // ENTRY_SIZE
    if file_size % ENTRY_SIZE != 0:
        print(f"WARNING: file size {file_size} not divisible by {ENTRY_SIZE}")

    print(f"\nVerification: {path} ({file_size:,} bytes, {n_entries} entries)")
    with open(path, "rb") as f:
        for i in range(min(show, n_entries)):
            data = f.read(ENTRY_SIZE)
            side = "w" if data[32] == 0 else "b"
            castling = data[33]
            ep = data[34]
            score = struct.unpack("<h", data[36:38])[0]
            wdl = data[38]
            # Reconstruct board for display
            board = chess.Board(fen=None)
            board.clear()
            for sq in range(64):
                byte_idx = sq // 2
                nibble = (data[byte_idx] & 0x0F) if sq % 2 == 0 else ((data[byte_idx] >> 4) & 0x0F)
                if nibble == 0:
                    continue
                for (pt, color), val in PIECE_NIBBLE.items():
                    if val == nibble:
                        board.set_piece_at(sq, chess.Piece(pt, color))
                        break
            board.turn = chess.WHITE if side == "w" else chess.BLACK
            print(f"  [{i+1}] score={score:+d} wdl={wdl} stm={side} "
                  f"castle={castling:04b} ep={ep} fen≈{board.board_fen()[:50]}")
    print(f"  Total: {n_entries} entries OK")


# ────────────────────────── Main ──────────────────────────

if __name__ == "__main__":
    # Parse args: pipeline.py [max_games] [output_path]
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    output_path = sys.argv[2] if len(sys.argv) > 2 else "lichess_test_data.bin"

    print(f"Running pipeline: max_games={max_games}, output={output_path}")
    run_pipeline(output_path, max_games=max_games, progress_interval=10)
    verify_output(output_path)
