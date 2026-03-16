from __future__ import annotations

import io
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zstandard

KING_BUCKETS = 10
PIECES_EX_KING = 5
SQUARES = 64
PER_COLOR_BUCKET = PIECES_EX_KING * SQUARES
PER_BUCKET_FEATS = PER_COLOR_BUCKET * 2
FT_SIZE = KING_BUCKETS * PER_BUCKET_FEATS
L1 = 256
L1_PAIR = L1 // 2
L2_INPUT = 2 * L1_PAIR
L2 = 32
NUM_STACKS = 4
SKIP = 8
NUM_PSQT = 4
ENTRY_SIZE = 40
SIGMOID_K = 400.0

NIBBLE_PIECE = {}
for i, (pt, col) in enumerate([
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
]):
    NIBBLE_PIECE[i + 1] = (pt, col)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def training_dir() -> Path:
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    return training_dir() / "data"


def games_dir() -> Path:
    return training_dir().parent / "games"


def elite_archive_dir() -> Path:
    return games_dir() / "Lichess Elite Database"


def iter_game_source_files() -> list[Path]:
    roots = [games_dir().glob("*.pgn.zst"), elite_archive_dir().glob("*.pgn")]
    files = [path for group in roots for path in group]
    return sorted(files)


def list_game_sources() -> pd.DataFrame:
    rows = []
    for path in iter_game_source_files():
        rows.append({
            "file": path.name,
            "relative_path": str(path.relative_to(training_dir().parent)),
            "format": "pgn.zst" if path.suffix == ".zst" else "pgn",
            "source": "elite-archive" if path.parent == elite_archive_dir() else "games-root",
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        })
    return pd.DataFrame(rows)


@contextmanager
def open_pgn_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".zst":
        dctx = zstandard.ZstdDecompressor()
        with path.open("rb") as fh:
            with dctx.stream_reader(fh) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as text_stream:
                    yield text_stream
        return

    with path.open("r", encoding="utf-8", errors="replace") as text_stream:
        yield text_stream


def list_training_bins() -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir().glob("*.bin")):
        size = path.stat().st_size
        rows.append({
            "file": path.name,
            "size_mb": round(size / (1024 * 1024), 2),
            "entries": size // ENTRY_SIZE,
        })
    return pd.DataFrame(rows)


def list_elite_pgns() -> pd.DataFrame:
    rows = []
    for path in sorted(elite_archive_dir().glob("*.pgn")):
        size = path.stat().st_size
        rows.append({
            "file": path.name,
            "month": path.stem.replace("lichess_elite_", ""),
            "size_mb": round(size / (1024 * 1024), 2),
        })
    return pd.DataFrame(rows)


def parse_time_control(value: str | None) -> tuple[int | None, int | None]:
    if not value or value in {"-", "?"}:
        return None, None
    if "+" not in value:
        return None, None
    base_str, inc_str = value.split("+", 1)
    try:
        return int(base_str), int(inc_str)
    except ValueError:
        return None, None


def classify_time_control(base_seconds: int | None) -> str:
    if base_seconds is None:
        return "unknown"
    if base_seconds < 180:
        return "bullet"
    if base_seconds < 600:
        return "blitz"
    if base_seconds < 1800:
        return "rapid"
    return "classical"


def _maybe_int(value: str | None) -> int | None:
    if not value or value in {"?", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _iter_pgn_headers(path: Path, max_games: int | None = None) -> Iterable[dict[str, str]]:
    headers: dict[str, str] = {}
    games = 0
    in_header = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if line.startswith("[") and line.endswith("]"):
                in_header = True
                key, _, remainder = line[1:-1].partition(" ")
                value = remainder.strip().strip('"')
                headers[key] = value
                continue

            if in_header and line == "":
                if headers:
                    yield headers
                    games += 1
                    if max_games is not None and games >= max_games:
                        return
                headers = {}
                in_header = False

    if headers and (max_games is None or games < max_games):
        yield headers


def sample_elite_headers(max_files: int | None = None, max_games_per_file: int = 250) -> pd.DataFrame:
    rows = []
    files = sorted(elite_archive_dir().glob("*.pgn"))
    if max_files is not None:
        files = files[:max_files]

    for path in files:
        month = path.stem.replace("lichess_elite_", "")
        for headers in _iter_pgn_headers(path, max_games=max_games_per_file):
            white_elo = _maybe_int(headers.get("WhiteElo"))
            black_elo = _maybe_int(headers.get("BlackElo"))
            avg_elo = None
            if white_elo is not None and black_elo is not None:
                avg_elo = (white_elo + black_elo) / 2.0

            base_seconds, increment = parse_time_control(headers.get("TimeControl"))
            rows.append({
                "file": path.name,
                "month": month,
                "event": headers.get("Event", "?"),
                "result": headers.get("Result", "?"),
                "termination": headers.get("Termination", "?"),
                "opening": headers.get("Opening", "Unknown"),
                "eco": headers.get("ECO", "?"),
                "white_elo": white_elo,
                "black_elo": black_elo,
                "avg_elo": avg_elo,
                "time_control": headers.get("TimeControl", "?"),
                "base_seconds": base_seconds,
                "increment": increment,
                "tc_class": classify_time_control(base_seconds),
            })

    return pd.DataFrame(rows)


def default_training_file() -> Path:
    bins = sorted(data_dir().glob("*.bin"))
    preferred = [
        data_dir() / "lichess_train_50k.bin",
        data_dir() / "training_data_large.bin",
        data_dir() / "training_data.bin",
    ]
    for path in preferred:
        if path.exists():
            return path
    if not bins:
        raise FileNotFoundError("No training .bin files found in training/data/")
    return bins[0]


def king_bucket_of(sq: int) -> int:
    file = sq & 7
    rank = sq >> 3
    fm = 7 - file if file >= 4 else file
    if 2 <= fm <= 3 and 2 <= rank <= 4:
        return 0
    if 1 <= fm <= 4 and 1 <= rank <= 6:
        return 1
    if fm >= 3 and 2 <= rank <= 5:
        return 2
    if rank == 0:
        return 3
    if rank == 1:
        return 4
    if rank == 6:
        return 5
    if rank == 7:
        return 6
    if fm <= 1 and (rank <= 2 or rank >= 5):
        return 7
    if fm >= 4 or rank <= 0 or rank >= 7:
        return 8
    return 9


def feat_white(pt: int, col: int, sq: int, ksq: int) -> int:
    bucket = king_bucket_of(ksq)
    color_offset = 0 if col == 0 else PER_COLOR_BUCKET
    return bucket * PER_BUCKET_FEATS + color_offset + pt * 64 + sq


def feat_black(pt: int, col: int, sq: int, ksq: int) -> int:
    fsq = sq ^ 56
    fksq = ksq ^ 56
    bucket = king_bucket_of(fksq)
    color_offset = 0 if col == 1 else PER_COLOR_BUCKET
    return bucket * PER_BUCKET_FEATS + color_offset + pt * 64 + fsq


def parse_entry(entry: bytes) -> dict:
    wk, bk = 255, 255
    pieces = []
    piece_counts = np.zeros(10, dtype=np.int32)
    for sq in range(64):
        byte = sq >> 1
        nibble = (entry[byte] & 0x0F) if (sq & 1) == 0 else (entry[byte] >> 4)
        if nibble == 0:
            continue
        pt, col = NIBBLE_PIECE[nibble]
        if pt == 5:
            if col == 0:
                wk = sq
            else:
                bk = sq
        else:
            pieces.append((pt, col, sq))
            piece_counts[pt + (5 if col else 0)] += 1

    wf = [feat_white(pt, col, sq, wk) for pt, col, sq in pieces]
    bf = [feat_black(pt, col, sq, bk) for pt, col, sq in pieces]
    score = int(struct.unpack_from("<h", entry, 36)[0])
    wdl_byte = int(entry[38])
    wdl = 1.0 if wdl_byte == 2 else (0.5 if wdl_byte == 1 else 0.0)
    stm = int(entry[32])
    return {
        "white_feats": wf,
        "black_feats": bf,
        "score": score,
        "wdl": wdl,
        "stm": stm,
        "pieces": len(pieces),
        "piece_counts": piece_counts,
    }


def sample_entries(path: Path | None = None, max_entries: int = 20000, stride: int | None = None) -> pd.DataFrame:
    path = path or default_training_file()
    total = path.stat().st_size // ENTRY_SIZE
    if total == 0:
        return pd.DataFrame()
    if stride is None:
        stride = max(1, total // max_entries)

    rows = []
    with path.open("rb") as f:
        idx = 0
        while True:
            entry = f.read(ENTRY_SIZE)
            if not entry or len(rows) >= max_entries:
                break
            if idx % stride == 0:
                parsed = parse_entry(entry)
                row = {
                    "idx": idx,
                    "score": parsed["score"],
                    "wdl": parsed["wdl"],
                    "stm": parsed["stm"],
                    "pieces": parsed["pieces"],
                    "white_feat_count": len(parsed["white_feats"]),
                    "black_feat_count": len(parsed["black_feats"]),
                }
                rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


def split_counts(total_entries: int, test_frac: float = 0.2) -> dict:
    test_count = round(total_entries * test_frac)
    return {
        "train_count": total_entries - test_count,
        "test_count": test_count,
        "test_frac": test_frac,
    }


class NagatoNNUE(nn.Module):
    def __init__(self, l1: int = 256, l2: int = 32, stacks: int = 4, skip: int = 8, psqt: int = 4):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.stacks = stacks
        self.skip = skip
        self.psqt = psqt
        self.ft_weight = nn.Parameter(torch.randn(FT_SIZE, l1) * 0.01)
        self.ft_bias = nn.Parameter(torch.zeros(l1))
        self.psqt_weight = nn.Parameter(torch.zeros(FT_SIZE, psqt))
        self.l2_weight = nn.Parameter(torch.randn(stacks, l1, l2) * 0.01)
        self.l2_bias = nn.Parameter(torch.zeros(stacks, l2))
        self.out_weight = nn.Parameter(torch.randn(stacks, l2) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(stacks))
        self.skip_weight = nn.Parameter(torch.zeros(stacks, skip))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def architecture_table() -> pd.DataFrame:
    configs = [
        {"name": "nagato-current", "l1": 256, "l2": 32, "stacks": 4, "skip": 8, "psqt": 4},
        {"name": "narrow", "l1": 192, "l2": 24, "stacks": 4, "skip": 8, "psqt": 4},
        {"name": "wide-l2", "l1": 256, "l2": 48, "stacks": 4, "skip": 8, "psqt": 4},
        {"name": "wide-l1", "l1": 320, "l2": 32, "stacks": 4, "skip": 8, "psqt": 4},
    ]
    rows = []
    for cfg in configs:
        model = NagatoNNUE(**{k: cfg[k] for k in ("l1", "l2", "stacks", "skip", "psqt")})
        rows.append({**cfg, "params": model.num_params()})
    return pd.DataFrame(rows)
