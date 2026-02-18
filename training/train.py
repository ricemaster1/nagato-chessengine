#!/usr/bin/env python3
"""
NNUE Trainer for Nagato Chess Engine
=====================================

Reads binary training data from datagen and trains a neural network
with the architecture:
    768 → L1_SIZE (ClippedReLU, per-perspective) → concat(2*L1_SIZE) → L2_SIZE (ClippedReLU) → 1

Exports weights to nn.bin in the format expected by src/nnue.rs.

Supports ROCm (AMD GPUs) and CUDA (NVIDIA GPUs) via PyTorch.

Usage:
    python train.py --data training_data.bin --output nn.bin [--epochs 100] [--batch-size 16384] [--lr 0.001]
"""

import argparse
import math
import os
import struct
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch is required. Install with:")
    print("  pip install torch  (CUDA/CPU)")
    print("  pip install torch-rocm  (AMD ROCm)")
    sys.exit(1)

# ============================================================
# Constants — must match src/nnue.rs
# ============================================================

INPUT_SIZE = 768      # 2 colors × 6 pieces × 64 squares
L1_SIZE = 128         # First hidden layer (per perspective)
L2_SIZE = 32          # Second hidden layer
ENTRY_SIZE = 40       # Bytes per training entry

# Piece encoding in the binary format
# 0=empty, 1=WP,2=WN,3=WB,4=WR,5=WQ,6=WK,7=BP,8=BN,9=BB,10=BR,11=BQ,12=BK
PIECE_TO_INDEX = {
    1: (0, 0),   # White Pawn   → (color=0, piece=0)
    2: (0, 1),   # White Knight → (color=0, piece=1)
    3: (0, 2),   # White Bishop → (color=0, piece=2)
    4: (0, 3),   # White Rook   → (color=0, piece=3)
    5: (0, 4),   # White Queen  → (color=0, piece=4)
    6: (0, 5),   # White King   → (color=0, piece=5)
    7: (1, 0),   # Black Pawn   → (color=1, piece=0)
    8: (1, 1),   # Black Knight → (color=1, piece=1)
    9: (1, 2),   # Black Bishop → (color=1, piece=2)
    10: (1, 3),  # Black Rook   → (color=1, piece=3)
    11: (1, 4),  # Black Queen  → (color=1, piece=4)
    12: (1, 5),  # Black King   → (color=1, piece=5)
}


# ============================================================
# Feature encoding
# ============================================================

def feature_index_white(piece_idx, color, sq):
    """Feature index from white's perspective."""
    color_offset = 0 if color == 0 else 384
    return color_offset + piece_idx * 64 + sq


def feature_index_black(piece_idx, color, sq):
    """Feature index from black's perspective (mirror vertically)."""
    flipped = sq ^ 56
    color_offset = 0 if color == 1 else 384  # black's own pieces first
    return color_offset + piece_idx * 64 + flipped


# ============================================================
# Dataset
# ============================================================

class ChessDataset(Dataset):
    """Reads binary training data and produces (white_features, black_features, score, result) tuples."""

    def __init__(self, path, score_scale=400.0, result_weight=0.5):
        self.score_scale = score_scale
        self.result_weight = result_weight

        data = np.fromfile(path, dtype=np.uint8)
        if len(data) % ENTRY_SIZE != 0:
            raise ValueError(f"File size {len(data)} is not a multiple of {ENTRY_SIZE}")

        self.num_entries = len(data) // ENTRY_SIZE
        self.data = data.reshape(self.num_entries, ENTRY_SIZE)
        print(f"Loaded {self.num_entries:,} training positions from {path}")

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Parse packed board (32 bytes)
        white_features = []
        black_features = []

        for sq in range(64):
            byte_idx = sq // 2
            if sq % 2 == 0:
                nibble = entry[byte_idx] & 0x0F
            else:
                nibble = (entry[byte_idx] >> 4) & 0x0F

            if nibble == 0:
                continue

            color, piece_idx = PIECE_TO_INDEX[nibble]
            white_features.append(feature_index_white(piece_idx, color, sq))
            black_features.append(feature_index_black(piece_idx, color, sq))

        # Side to move
        side = entry[32]  # 0=white, 1=black

        # Score (i16, white's perspective)
        score = struct.unpack_from('<h', entry, 36)[0]

        # Result (i8: 1=white win, 0=draw, -1=black win)
        result = struct.unpack_from('<b', entry, 38)[0]

        # Build sparse feature tensors
        wf = torch.zeros(INPUT_SIZE, dtype=torch.float32)
        bf = torch.zeros(INPUT_SIZE, dtype=torch.float32)
        for fi in white_features:
            wf[fi] = 1.0
        for fi in black_features:
            bf[fi] = 1.0

        # Target: blend of sigmoid(score/scale) and WDL
        # This is the standard NNUE training target
        score_sigmoid = 1.0 / (1.0 + math.exp(-score / self.score_scale))
        wdl = (result + 1.0) / 2.0  # map {-1,0,1} to {0, 0.5, 1.0}
        target = (1.0 - self.result_weight) * score_sigmoid + self.result_weight * wdl

        return wf, bf, torch.tensor(side, dtype=torch.long), torch.tensor(target, dtype=torch.float32)


# ============================================================
# Network
# ============================================================

class NagotoNNUE(nn.Module):
    """
    Baby NNUE architecture.
    
    The feature transformer is a shared linear layer applied to both perspectives.
    The output head takes the concatenation [stm_features, opponent_features].
    """

    def __init__(self):
        super().__init__()
        # Feature transformer (shared weights for both perspectives)
        self.ft = nn.Linear(INPUT_SIZE, L1_SIZE)

        # Output head
        self.l2 = nn.Linear(2 * L1_SIZE, L2_SIZE)
        self.out = nn.Linear(L2_SIZE, 1)

    def forward(self, white_features, black_features, side):
        """
        Args:
            white_features: (batch, INPUT_SIZE) — white's perspective features
            black_features: (batch, INPUT_SIZE) — black's perspective features
            side: (batch,) — 0 for white to move, 1 for black to move
        """
        # Feature transform (shared weights, both perspectives)
        white_acc = self.ft(white_features)
        black_acc = self.ft(black_features)

        # ClippedReLU
        white_acc = torch.clamp(white_acc, 0.0, 1.0)
        black_acc = torch.clamp(black_acc, 0.0, 1.0)

        # Concat: [stm | opponent] — side determines the order
        # side=0: white to move → [white | black]
        # side=1: black to move → [black | white]
        white_to_move = (side == 0).unsqueeze(1).float()  # (batch, 1)
        stm = white_acc * white_to_move + black_acc * (1.0 - white_to_move)
        opp = black_acc * white_to_move + white_acc * (1.0 - white_to_move)
        concat = torch.cat([stm, opp], dim=1)  # (batch, 2*L1_SIZE)

        # Layer 2
        x = self.l2(concat)
        x = torch.clamp(x, 0.0, 1.0)  # ClippedReLU

        # Output (sigmoid for [0,1] target)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x.squeeze(1)


# ============================================================
# Export weights to nn.bin
# ============================================================

def export_weights(model, path):
    """Export model weights to the binary format expected by src/nnue.rs."""
    with open(path, 'wb') as f:
        # Magic
        f.write(b'NAGT')
        # Version
        f.write(struct.pack('<I', 1))

        # L1 weights: INPUT_SIZE × L1_SIZE (feature transformer)
        ft_weight = model.ft.weight.data.cpu().numpy()  # shape: (L1_SIZE, INPUT_SIZE)
        # We need [feature_idx][hidden_idx], so transpose
        ft_weight = ft_weight.T  # shape: (INPUT_SIZE, L1_SIZE)
        for i in range(INPUT_SIZE):
            for j in range(L1_SIZE):
                f.write(struct.pack('<f', float(ft_weight[i, j])))

        # L1 biases
        ft_bias = model.ft.bias.data.cpu().numpy()
        for j in range(L1_SIZE):
            f.write(struct.pack('<f', float(ft_bias[j])))

        # L2 weights: (2*L1_SIZE) × L2_SIZE
        l2_weight = model.l2.weight.data.cpu().numpy()  # shape: (L2_SIZE, 2*L1_SIZE)
        l2_weight = l2_weight.T  # shape: (2*L1_SIZE, L2_SIZE)
        for i in range(2 * L1_SIZE):
            for j in range(L2_SIZE):
                f.write(struct.pack('<f', float(l2_weight[i, j])))

        # L2 biases
        l2_bias = model.l2.bias.data.cpu().numpy()
        for j in range(L2_SIZE):
            f.write(struct.pack('<f', float(l2_bias[j])))

        # Output weights
        out_weight = model.out.weight.data.cpu().numpy()  # shape: (1, L2_SIZE)
        for j in range(L2_SIZE):
            f.write(struct.pack('<f', float(out_weight[0, j])))

        # Output bias
        f.write(struct.pack('<f', float(model.out.bias.data.cpu().numpy()[0])))

    file_size = os.path.getsize(path)
    expected = 8 + (INPUT_SIZE * L1_SIZE + L1_SIZE + 2 * L1_SIZE * L2_SIZE + L2_SIZE + L2_SIZE + 1) * 4
    print(f"Exported weights to {path} ({file_size:,} bytes, expected {expected:,})")
    assert file_size == expected, f"Size mismatch! Got {file_size}, expected {expected}"


# ============================================================
# Training loop
# ============================================================

def train(args):
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'hip') or (hasattr(torch.version, 'hip') and torch.version.hip):
        device = torch.device('cuda')  # ROCm uses 'cuda' device in PyTorch
        print(f"Using ROCm: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load data
    dataset = ChessDataset(args.data, score_scale=args.score_scale, result_weight=args.result_weight)

    # Split into train/val (95/5)
    val_size = max(1, len(dataset) // 20)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == 'cuda'))

    print(f"Train: {train_size:,} positions, Val: {val_size:,} positions")

    # Model
    model = NagotoNNUE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        t0 = time.time()

        for wf, bf, side, target in train_loader:
            wf = wf.to(device, non_blocking=True)
            bf = bf.to(device, non_blocking=True)
            side = side.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(wf, bf, side)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train = train_loss / max(1, train_batches)
        lr_now = optimizer.param_groups[0]['lr']

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for wf, bf, side, target in val_loader:
                wf = wf.to(device, non_blocking=True)
                bf = bf.to(device, non_blocking=True)
                side = side.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                pred = model(wf, bf, side)
                loss = criterion(pred, target)

                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(1, val_batches)
        elapsed = time.time() - t0

        # Save best
        improved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            export_weights(model, args.output)
            improved = " *BEST*"

        print(f"Epoch {epoch:3d}/{args.epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
              f"LR: {lr_now:.2e} | {elapsed:.1f}s{improved}")

        # Early stopping
        if epoch - best_epoch > args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Weights saved to: {args.output}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NNUE Trainer for Nagato Chess Engine")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to binary training data from datagen')
    parser.add_argument('--output', type=str, default='nn.bin',
                        help='Output path for weights file (default: nn.bin)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=16384,
                        help='Batch size (default: 16384)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--score-scale', type=float, default=400.0,
                        help='Score scaling for sigmoid target (default: 400.0)')
    parser.add_argument('--result-weight', type=float, default=0.5,
                        help='Weight for WDL result in target (0=pure score, 1=pure WDL, default: 0.5)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
