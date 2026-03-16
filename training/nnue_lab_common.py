from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
