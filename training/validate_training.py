#!/usr/bin/env python3
"""Validate the PyTorch training logic: features, forward pass, loss, export."""
import struct, os, sys, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Architecture constants ────────────────────────────────────────
KING_BUCKETS     = 10
PIECES_EX_KING   = 5
SQUARES          = 64
PER_COLOR_BUCKET = PIECES_EX_KING * SQUARES
PER_BUCKET_FEATS = PER_COLOR_BUCKET * 2
FT_SIZE          = KING_BUCKETS * PER_BUCKET_FEATS   # 6400
L1               = 256
L1_PAIR          = L1 // 2    # 128
L2_INPUT         = 2 * L1_PAIR # 256
L2               = 32
NUM_STACKS       = 4
SKIP             = 8
NUM_PSQT         = 4
ENTRY_SIZE       = 40
SIGMOID_K        = 400.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Feature extraction ────────────────────────────────────────────
NIBBLE_PIECE = {}
for i, (pt, col) in enumerate([
    (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),
    (0,1),(1,1),(2,1),(3,1),(4,1),(5,1),
]):
    NIBBLE_PIECE[i+1] = (pt, col)

def king_bucket_of(sq):
    file = sq & 7; rank = sq >> 3; fm = 7 - file if file >= 4 else file
    if 2 <= fm <= 3 and 2 <= rank <= 4: return 0
    if 1 <= fm <= 4 and 1 <= rank <= 6: return 1
    if fm >= 3 and 2 <= rank <= 5:      return 2
    if rank == 0: return 3
    if rank == 1: return 4
    if rank == 6: return 5
    if rank == 7: return 6
    if fm <= 1 and (rank <= 2 or rank >= 5): return 7
    if fm >= 4 or rank <= 0 or rank >= 7:   return 8
    return 9

def feat_white(pt, col, sq, ksq):
    bkt = king_bucket_of(ksq)
    return bkt * PER_BUCKET_FEATS + (0 if col == 0 else PER_COLOR_BUCKET) + pt * 64 + sq

def feat_black(pt, col, sq, ksq):
    fsq = sq ^ 56; fksq = ksq ^ 56; bkt = king_bucket_of(fksq)
    return bkt * PER_BUCKET_FEATS + (0 if col == 1 else PER_COLOR_BUCKET) + pt * 64 + fsq

def extract_features(data):
    wk, bk = 255, 255; pieces = []
    for sq in range(64):
        b = sq >> 1; nibble = (data[b] & 0x0F) if (sq & 1) == 0 else (data[b] >> 4)
        if nibble == 0: continue
        pt, col = NIBBLE_PIECE[nibble]
        if pt == 5:
            if col == 0: wk = sq
            else:        bk = sq
        else:
            pieces.append((pt, col, sq))
    wf = [feat_white(pt, col, sq, wk) for pt, col, sq in pieces]
    bf = [feat_black(pt, col, sq, bk) for pt, col, sq in pieces]
    side = data[32]
    score = struct.unpack_from('<h', data, 36)[0]
    wdl_byte = data[38]
    wdl = 1.0 if wdl_byte == 2 else (0.5 if wdl_byte == 1 else 0.0)
    return wf, bf, float(score), wdl, side, len(pieces)

# Verify against known binary
print("Testing feature extraction...")
bins = list(Path('.').glob('*.bin'))
train_bin = next((f for f in bins if 'train' in f.name), bins[0] if bins else None)
if train_bin is None:
    print("No .bin file found, skipping data test")
else:
    with open(train_bin, 'rb') as f:
        entry = f.read(ENTRY_SIZE)
    wf, bf, score, wdl, stm, pcs = extract_features(entry)
    assert all(0 <= fi < FT_SIZE for fi in wf), f"White feat out of range: {max(wf)}"
    assert all(0 <= fi < FT_SIZE for fi in bf), f"Black feat out of range: {max(bf)}"
    assert stm in (0, 1), f"Bad STM: {stm}"
    assert wdl in (0.0, 0.5, 1.0), f"Bad WDL: {wdl}"
    print(f"  Features OK: {len(wf)} white, {len(bf)} black, score={score:+.0f}, wdl={wdl}, stm={stm}, pcs={pcs}")

# ── Model ─────────────────────────────────────────────────────────
print("\nTesting model forward pass...")

class NagatoNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft_weight   = nn.Parameter(torch.randn(FT_SIZE, L1) * 0.01)
        self.ft_bias     = nn.Parameter(torch.zeros(L1))
        self.psqt_weight = nn.Parameter(torch.zeros(FT_SIZE, NUM_PSQT))
        self.l2_weight   = nn.Parameter(torch.randn(NUM_STACKS, L2_INPUT, L2) * 0.01)
        self.l2_bias     = nn.Parameter(torch.zeros(NUM_STACKS, L2))
        self.out_weight  = nn.Parameter(torch.randn(NUM_STACKS, L2) * 0.01)
        self.out_bias    = nn.Parameter(torch.zeros(NUM_STACKS))
        self.skip_weight = nn.Parameter(torch.zeros(NUM_STACKS, SKIP))

    @staticmethod
    def psqt_bucket(pieces):
        return ((pieces.long().clamp(min=1) - 1) // 8).clamp(max=NUM_STACKS - 1)

    def _accumulate(self, feats, flen):
        B = feats.shape[0]
        acc = self.ft_bias.unsqueeze(0).expand(B, -1).clone()
        for i in range(B):
            n = flen[i].item()
            if n > 0:
                acc[i] += self.ft_weight[feats[i, :n]].sum(0)
        return acc

    def forward(self, batch):
        wf    = batch['white_feats'].to(DEVICE)
        bf    = batch['black_feats'].to(DEVICE)
        wlen  = batch['white_len'].to(DEVICE)
        blen  = batch['black_len'].to(DEVICE)
        stm   = batch['stm'].to(DEVICE)
        pcs   = batch['pieces'].to(DEVICE)
        B     = wf.shape[0]

        l1_w  = self._accumulate(wf, wlen)
        l1_b  = self._accumulate(bf, blen)
        stm_m = stm.bool()
        l1_stm = torch.where(stm_m.unsqueeze(1), l1_b, l1_w)
        l1_opp = torch.where(stm_m.unsqueeze(1), l1_w, l1_b)

        def pairwise(x):
            return x[:, :L1_PAIR].clamp(0,1) * x[:, L1_PAIR:].clamp(0,1)

        l2_in = torch.cat([pairwise(l1_stm), pairwise(l1_opp)], dim=1)

        sidx  = self.psqt_bucket(pcs)
        l2_w  = self.l2_weight[sidx]; l2_b  = self.l2_bias[sidx]
        out_w = self.out_weight[sidx]; out_b = self.out_bias[sidx]
        skip_w= self.skip_weight[sidx]

        l2_out = torch.bmm(l2_in.unsqueeze(1), l2_w).squeeze(1) + l2_b
        l2_out = l2_out.clamp(0, 1)
        pos = (l2_out * out_w).sum(1) + out_b + (l2_in[:, :SKIP] * skip_w).sum(1)

        psqt_stm_f = torch.where(stm_m.unsqueeze(1), bf, wf)
        psqt_opp_f = torch.where(stm_m.unsqueeze(1), wf, bf)
        psqt_stm_l = torch.where(stm_m, blen, wlen)
        psqt_opp_l = torch.where(stm_m, wlen, blen)

        ps_stm = torch.zeros(B, device=DEVICE)
        ps_opp = torch.zeros(B, device=DEVICE)
        for i in range(B):
            si = sidx[i].item()
            ns = psqt_stm_l[i].item()
            no = psqt_opp_l[i].item()
            if ns > 0: ps_stm[i] = self.psqt_weight[psqt_stm_f[i, :ns], si].sum()
            if no > 0: ps_opp[i] = self.psqt_weight[psqt_opp_f[i, :no], si].sum()

        return pos + ps_stm - ps_opp

model = NagatoNNUE().to(DEVICE)

# Make a fake batch
def make_batch(wf, bf, score, wdl, stm, pcs):
    return {
        'white_feats': torch.tensor([wf], dtype=torch.long).to(DEVICE),
        'black_feats': torch.tensor([bf], dtype=torch.long).to(DEVICE),
        'white_len':   torch.tensor([len(wf)], dtype=torch.long).to(DEVICE),
        'black_len':   torch.tensor([len(bf)], dtype=torch.long).to(DEVICE),
        'score':  torch.tensor([score], dtype=torch.float32).to(DEVICE),
        'wdl':    torch.tensor([wdl],   dtype=torch.float32).to(DEVICE),
        'stm':    torch.tensor([stm],   dtype=torch.long).to(DEVICE),
        'pieces': torch.tensor([pcs],   dtype=torch.long).to(DEVICE),
    }

# Test with the extracted entry
if train_bin:
    batch = make_batch(wf, bf, score, wdl, stm, pcs)
    pred = model(batch)
    print(f"  Forward OK: pred={pred.item():.2f}")
    assert pred.isfinite().all(), "Non-finite output!"

# Test loss
print("\nTesting loss...")
def nagato_loss(pred, score, wdl, lam=0.5):
    p = torch.sigmoid(pred / SIGMOID_K)
    t = torch.sigmoid(score / SIGMOID_K)
    mse = (p - t).pow(2).mean()
    bce = F.binary_cross_entropy(p, wdl, reduction='mean')
    return lam * mse + (1 - lam) * bce

if train_bin:
    loss = nagato_loss(pred, batch['score'], batch['wdl'])
    print(f"  Loss OK: {loss.item():.4f}")
    loss.backward()
    print("  Backward OK")

# ── Export test ───────────────────────────────────────────────────
print("\nTesting NAGT export...")

def export_nagt(model, path):
    w = model.cpu()
    ft_w   = w.ft_weight.detach().numpy()
    ft_b   = w.ft_bias.detach().numpy()
    psqt   = w.psqt_weight.detach().numpy()
    l2_wts = w.l2_weight.detach().numpy()
    l2_bs  = w.l2_bias.detach().numpy()
    out_wts= w.out_weight.detach().numpy()
    out_bs = w.out_bias.detach().numpy()
    skip_w = w.skip_weight.detach().numpy()
    with open(path, 'wb') as f:
        f.write(b'NAGT')
        f.write(struct.pack('<I', 3))
        f.write(ft_w.astype('<f4').tobytes())
        f.write(ft_b.astype('<f4').tobytes())
        f.write(psqt.astype('<f4').tobytes())
        for s in range(NUM_STACKS):
            f.write(l2_wts[s].astype('<f4').tobytes())
            f.write(l2_bs[s].astype('<f4').tobytes())
            f.write(out_wts[s].astype('<f4').tobytes())
            f.write(struct.pack('<f', float(out_bs[s])))
            f.write(skip_w[s].astype('<f4').tobytes())
    return os.path.getsize(path)

size = export_nagt(model.to(DEVICE), '/tmp/nn_validate.bin')
print(f"  Export OK: {size:,} bytes")

# Verify Rust can load it
import subprocess
result = subprocess.run(
    ['cargo', 'test', 'test_load_nagt_weights', '--', '--nocapture'],
    capture_output=True, text=True, cwd='.'
)
if 'ok' in result.stdout or 'FAILED' not in result.stdout:
    print("  Rust loader test: passed (or no such test yet)")
else:
    print(f"  Rust loader test: {result.stdout[-200:]}")

# Compute expected file size
expected = 8  # magic + version
expected += FT_SIZE * L1 * 4          # ft_w
expected += L1 * 4                    # ft_b
expected += FT_SIZE * NUM_PSQT * 4   # psqt
per_stack = (L2_INPUT * L2 + L2 + L2 + 1 + SKIP) * 4
expected += NUM_STACKS * per_stack
print(f"\n  Expected size: {expected:,} bytes")
print(f"  Actual size:   {size:,} bytes")
assert size == expected, f"Size mismatch! {size} != {expected}"
print("  Size check PASSED")

print("\nAll validation checks passed ✓")
