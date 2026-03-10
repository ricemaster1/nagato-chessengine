#!/usr/bin/env python3
"""
Parallel Lichess → Nagato training data pipeline.

Uses multiprocessing to run multiple Stockfish instances in parallel.
Each worker gets a batch of (board, result) pairs, evaluates them, and
writes binary entries to a shared output file.

Strategy: 8 cores → 4 workers × 2 SF threads each = 8 threads total.
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
import multiprocessing as mp
from pathlib import Path
from queue import Empty

# ────────────────────────── Configuration ──────────────────────────

GAMES_DIR = Path("games")
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
SF_DEPTH = 10
SF_THREADS_PER_WORKER = 1
SF_HASH_PER_WORKER = 64
NUM_WORKERS = 8
MIN_ELO = 2000
MIN_TIME_BASE = 180
SKIP_PLIES = 16
POSITIONS_PER_GAME = 2
BATCH_SIZE = 100          # positions per worker batch
ENTRY_SIZE = 40

# Piece encoding matching Rust datagen::pack_board()
PIECE_NIBBLE = {
    (chess.PAWN,   chess.WHITE): 1,  (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,  (chess.ROOK,   chess.WHITE): 4,
    (chess.QUEEN,  chess.WHITE): 5,  (chess.KING,   chess.WHITE): 6,
    (chess.PAWN,   chess.BLACK): 7,  (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,  (chess.ROOK,   chess.BLACK): 10,
    (chess.QUEEN,  chess.BLACK): 11, (chess.KING,   chess.BLACK): 12,
}


# ────────────────────────── Pack/Write ──────────────────────────

def pack_board(board: chess.Board) -> bytes:
    packed = bytearray(32)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        nibble = PIECE_NIBBLE[(piece.piece_type, piece.color)]
        byte_idx = sq // 2
        if sq % 2 == 0:
            packed[byte_idx] |= nibble
        else:
            packed[byte_idx] |= (nibble << 4)
    return bytes(packed)


def encode_castling(board: chess.Board) -> int:
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):  rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE): rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):  rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK): rights |= 8
    return rights


def make_entry(board: chess.Board, score_white: int, wdl: int) -> bytes:
    packed = pack_board(board)
    entry = bytearray(ENTRY_SIZE)
    entry[0:32] = packed
    entry[32] = 0 if board.turn == chess.WHITE else 1
    entry[33] = encode_castling(board)
    entry[34] = chess.square_file(board.ep_square) if board.ep_square is not None else 255
    entry[35] = 0
    struct.pack_into("<h", entry, 36, max(-32000, min(32000, score_white)))
    entry[38] = wdl
    entry[39] = 0
    return bytes(entry)


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
    if result_str == "1/2-1/2":
        return 1
    white_wins = result_str == "1-0"
    stm_is_white = side_to_move == chess.WHITE
    return 2 if white_wins == stm_is_white else 0


def sample_positions(game, n=POSITIONS_PER_GAME, skip_plies=SKIP_PLIES):
    board = game.board()
    positions = []
    ply = 0
    for move in game.mainline_moves():
        board.push(move)
        ply += 1
        if ply > skip_plies and not board.is_game_over():
            positions.append((board.copy(), board.turn))
    if not positions:
        return []
    return random.sample(positions, min(n, len(positions)))


# ────────────────────────── Worker ──────────────────────────

def eval_worker(task_queue, result_queue, worker_id):
    """Worker process: pull batches of FENs, evaluate with SF, return entries."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": SF_THREADS_PER_WORKER, "Hash": SF_HASH_PER_WORKER})

    evaluated = 0
    while True:
        try:
            batch = task_queue.get(timeout=5)
        except Empty:
            break

        if batch is None:  # poison pill
            break

        entries = bytearray()
        for fen, result_str, turn in batch:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
            score = info["score"].pov(chess.WHITE)
            if score.is_mate():
                cp_white = 30000 if score.mate() > 0 else -30000
            else:
                cp_white = score.score()
            wdl = result_to_wdl(result_str, turn)
            entries.extend(make_entry(board, cp_white, wdl))
            evaluated += 1

        result_queue.put(bytes(entries))

    engine.quit()
    result_queue.put(None)  # signal done


# ────────────────────────── Main ──────────────────────────

def run_parallel_pipeline(output_path, max_games=None, progress_interval=500):
    zst_files = sorted(GAMES_DIR.glob("*.pgn.zst"))
    if not zst_files:
        print("No .pgn.zst files found in games/")
        return

    print(f"Pipeline: {NUM_WORKERS} workers × {SF_THREADS_PER_WORKER} SF threads, "
          f"depth={SF_DEPTH}")
    print(f"Filter: Elo≥{MIN_ELO}, TC≥{MIN_TIME_BASE}s, Normal termination")
    print(f"Files: {len(zst_files)}, Output: {output_path}")

    task_queue = mp.Queue(maxsize=NUM_WORKERS * 4)
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=eval_worker, args=(task_queue, result_queue, i))
        p.start()
        workers.append(p)

    t0 = time.time()
    games_seen = 0
    games_passed = 0
    positions_sent = 0
    positions_written = 0
    workers_done = 0
    current_batch = []

    def flush_batch():
        nonlocal positions_sent
        if current_batch:
            task_queue.put(list(current_batch))
            positions_sent += len(current_batch)
            current_batch.clear()

    # Producer: read games, filter, sample, send to workers
    with open(output_path, "wb") as out:
        for zst_path in zst_files:
            print(f"\n  Reading {zst_path.name}...")
            dctx = zstandard.ZstdDecompressor()
            with open(zst_path, "rb") as fh:
                reader = dctx.stream_reader(fh)
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

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

                    for board, turn in sample_positions(game):
                        current_batch.append((board.fen(), result_str, turn))
                        if len(current_batch) >= BATCH_SIZE:
                            flush_batch()

                    if games_passed % progress_interval == 0:
                        # Drain results so far
                        while not result_queue.empty():
                            data = result_queue.get_nowait()
                            if data is None:
                                workers_done += 1
                            else:
                                out.write(data)
                                positions_written += len(data) // ENTRY_SIZE

                        elapsed = time.time() - t0
                        rate = positions_written / elapsed if elapsed > 0 else 0
                        print(f"  {games_passed} games ({games_seen} scanned), "
                              f"{positions_written}/{positions_sent} pos written/sent, "
                              f"{rate:.1f} pos/s, {elapsed:.0f}s")

            if max_games is not None and games_passed >= max_games:
                break

        # Flush remaining
        flush_batch()

        # Send poison pills
        for _ in range(NUM_WORKERS):
            task_queue.put(None)

        # Drain all results
        while workers_done < NUM_WORKERS:
            data = result_queue.get(timeout=300)
            if data is None:
                workers_done += 1
            else:
                out.write(data)
                positions_written += len(data) // ENTRY_SIZE

    for p in workers:
        p.join()

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Games: {games_seen} scanned, {games_passed} passed")
    print(f"  Positions: {positions_written}")
    print(f"  Time: {elapsed:.1f}s ({positions_written/elapsed:.1f} pos/s)")
    print(f"  Output: {output_path} ({file_size:,} bytes)")


if __name__ == "__main__":
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    output_path = sys.argv[2] if len(sys.argv) > 2 else "lichess_train.bin"

    print(f"Parallel pipeline: max_games={max_games}, output={output_path}")
    run_parallel_pipeline(output_path, max_games=max_games, progress_interval=50)
