#!/usr/bin/env python3
"""Verify an nn.bin weights file has the correct format and dimensions."""

import struct
import sys
import os

INPUT_SIZE = 768
L1_SIZE = 128
L2_SIZE = 32

def verify(path):
    file_size = os.path.getsize(path)
    concat_size = 2 * L1_SIZE
    expected_floats = (INPUT_SIZE * L1_SIZE + L1_SIZE +
                       concat_size * L2_SIZE + L2_SIZE +
                       L2_SIZE + 1)
    expected_size = 8 + expected_floats * 4

    print(f"File: {path}")
    print(f"Size: {file_size:,} bytes (expected {expected_size:,})")

    if file_size != expected_size:
        print("ERROR: Size mismatch!")
        return False

    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'NAGT':
            print(f"ERROR: Bad magic: {magic}")
            return False

        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            print(f"ERROR: Unsupported version: {version}")
            return False

        # Read all weights and compute stats
        floats = []
        while True:
            data = f.read(4)
            if len(data) < 4:
                break
            floats.append(struct.unpack('<f', data)[0])

    print(f"Magic: NAGT")
    print(f"Version: {version}")
    print(f"Total weights: {len(floats):,}")

    idx = 0
    # L1 weights
    l1_w = floats[idx:idx + INPUT_SIZE * L1_SIZE]
    idx += INPUT_SIZE * L1_SIZE
    print(f"L1 weights ({INPUT_SIZE}x{L1_SIZE}): min={min(l1_w):.4f} max={max(l1_w):.4f} mean={sum(l1_w)/len(l1_w):.6f}")

    # L1 biases
    l1_b = floats[idx:idx + L1_SIZE]
    idx += L1_SIZE
    print(f"L1 biases ({L1_SIZE}): min={min(l1_b):.4f} max={max(l1_b):.4f}")

    # L2 weights
    l2_w = floats[idx:idx + concat_size * L2_SIZE]
    idx += concat_size * L2_SIZE
    print(f"L2 weights ({concat_size}x{L2_SIZE}): min={min(l2_w):.4f} max={max(l2_w):.4f} mean={sum(l2_w)/len(l2_w):.6f}")

    # L2 biases
    l2_b = floats[idx:idx + L2_SIZE]
    idx += L2_SIZE
    print(f"L2 biases ({L2_SIZE}): min={min(l2_b):.4f} max={max(l2_b):.4f}")

    # Output weights
    out_w = floats[idx:idx + L2_SIZE]
    idx += L2_SIZE
    print(f"Output weights ({L2_SIZE}): min={min(out_w):.4f} max={max(out_w):.4f}")

    # Output bias
    out_b = floats[idx]
    print(f"Output bias: {out_b:.6f}")

    print("\nOK — weights file is valid!")
    return True


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <nn.bin>")
        sys.exit(1)
    ok = verify(sys.argv[1])
    sys.exit(0 if ok else 1)
