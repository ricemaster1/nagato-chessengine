#!/usr/bin/env python3
"""Quick test of the Lichess pipeline - scan 5000 games, check filter rate."""
import chess, chess.pgn, zstandard, io
from pathlib import Path

GAMES_DIR = Path("games")
MIN_ELO = 2000
MIN_TIME_BASE = 180

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

zst_files = sorted(GAMES_DIR.glob("*.pgn.zst"))
print(f"Found {len(zst_files)} ZST files")

dctx = zstandard.ZstdDecompressor()
seen = 0
passed = 0
with open(zst_files[0], "rb") as fh:
    reader = dctx.stream_reader(fh)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    while seen < 5000:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        seen += 1
        if passes_filter(game.headers):
            passed += 1
            if passed <= 3:
                print(f"  Pass #{passed}: {game.headers.get('WhiteElo')} vs {game.headers.get('BlackElo')}, "
                      f"TC={game.headers.get('TimeControl')}, Result={game.headers.get('Result')}")

print(f"\nScanned {seen} games, {passed} passed filter ({100*passed/seen:.1f}%)")
