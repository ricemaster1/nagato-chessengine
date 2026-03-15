#!/usr/bin/env python3
"""Test the full pipeline: filter → sample → Stockfish eval → binary output."""
import chess, chess.pgn, chess.engine, zstandard, io, struct, random, time, os
from pathlib import Path

GAMES_DIR = Path("games")
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
SF_DEPTH = 14
SF_THREADS = 4
SF_HASH_MB = 256
MIN_ELO = 2000
MIN_TIME_BASE = 180
SKIP_PLIES = 16
POSITIONS_PER_GAME = 2
OUTPUT_FILE = "lichess_test_data.bin"

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
    if white_wins == stm_is_white:
        return 2
    return 0

def sample_positions(game, n=POSITIONS_PER_GAME, skip_plies=SKIP_PLIES):
    board = game.board()
    positions = []
    ply = 0
    for move in game.mainline_moves():
        board.push(move)
        ply += 1
        if ply > skip_plies and not board.is_game_over():
            positions.append((ply, board.fen(), board.turn))
    if not positions:
        return []
    k = min(n, len(positions))
    return random.sample(positions, k)

def write_entry(f, fen, cp_score, wdl):
    fen_bytes = fen.encode("ascii") + b"\x00"
    cp_clamped = max(-32000, min(32000, cp_score))
    f.write(fen_bytes)
    f.write(struct.pack("<h", cp_clamped))
    f.write(struct.pack("B", wdl))

def read_entry(f):
    fen_chars = []
    while True:
        b = f.read(1)
        if not b or b == b"\x00":
            break
        fen_chars.append(b)
    if not fen_chars:
        return None
    fen = b"".join(fen_chars).decode("ascii")
    data = f.read(3)
    if len(data) < 3:
        return None
    cp = struct.unpack("<h", data[:2])[0]
    wdl = data[2]
    return (fen, cp, wdl)

# Run on 10 quality games
zst_files = sorted(GAMES_DIR.glob("*.pgn.zst"))
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})

t0 = time.time()
games_seen = 0
games_passed = 0
positions_eval = 0

with open(OUTPUT_FILE, "wb") as out:
    dctx = zstandard.ZstdDecompressor()
    with open(zst_files[0], "rb") as fh:
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while games_passed < 10:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            games_seen += 1
            if not passes_filter(game.headers):
                continue
            games_passed += 1
            result_str = game.headers["Result"]
            sampled = sample_positions(game)
            for ply, fen, turn in sampled:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
                score = info["score"].pov(turn)
                if score.is_mate():
                    mate_in = score.mate()
                    cp = 30000 if mate_in > 0 else -30000
                else:
                    cp = score.score()
                wdl = result_to_wdl(result_str, turn)
                write_entry(out, fen, cp, wdl)
                positions_eval += 1
                print(f"  [{positions_eval}] ply={ply} cp={cp:+d} wdl={wdl} fen={fen[:50]}...")

engine.quit()
elapsed = time.time() - t0

print(f"\nDone: {games_seen} seen, {games_passed} passed, {positions_eval} positions in {elapsed:.1f}s")
print(f"Throughput: {positions_eval/elapsed:.1f} positions/sec")
print(f"Output: {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)} bytes)")

# Verify
print("\nVerification:")
with open(OUTPUT_FILE, "rb") as f:
    count = 0
    while True:
        e = read_entry(f)
        if e is None:
            break
        count += 1
        if count <= 3:
            print(f"  Entry {count}: cp={e[1]:+d} wdl={e[2]} fen={e[0][:50]}...")
print(f"Total entries verified: {count}")
