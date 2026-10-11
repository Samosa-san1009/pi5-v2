#!/usr/bin/env bash
# Build the PuReST engine -> native/purest/build/plr_purest
# Needs: sudo apt install -y cmake g++ libopencv-dev
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/native/purest"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build" -j"$(nproc)"
"$SRC/build/plr_purest" --version >/dev/null
echo "built: $SRC/build/plr_purest ($("$SRC/build/plr_purest" --version))"
