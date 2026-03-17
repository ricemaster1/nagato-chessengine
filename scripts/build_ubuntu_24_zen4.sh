#!/usr/bin/env bash
set -euo pipefail

# Tuned release build for Ubuntu 24.04 x64 on Ryzen 9 7950X (Zen 4).
# Note: RX 7900 XTX GPUs do not accelerate this engine search path.

export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=znver4 -C target-feature=+avx2,+bmi2,+fma,+popcnt,+lzcnt"

cargo build --release --target x86_64-unknown-linux-gnu
