#!/usr/bin/env bash
# UCI engine with embedded net, generic x86-64-v2 (any 64-bit CPU since ~2009; no AVX requirement).
#   az/build_engine.sh [NET.bin]   (default: latest data/az_run1/net_XXXX.bin)
#   → dist/az-engine.exe (Windows, static) and build/az-engine (Linux)
# Needs: g++ (Linux) and g++-mingw-w64-x86-64-posix (Windows cross compile).
set -euo pipefail
cd "$(dirname "$0")/.."
NET=${1:-$(ls data/az_run1/net_[0-9][0-9][0-9][0-9].bin | sort | tail -1)}
mkdir -p build dist
cp "$NET" build/engine_net.bin
FLAGS=(-std=c++20 -O3 -march=x86-64-v2 -mtune=generic -fopenmp-simd -fno-math-errno -DNDEBUG -pthread
       "-DAZ_NET_FILE=\"$(pwd)/build/engine_net.bin\"")
g++ "${FLAGS[@]}" -o build/az-engine az/engine.cpp
x86_64-w64-mingw32-g++-posix "${FLAGS[@]}" -static -s -o dist/az-engine.exe az/engine.cpp
echo "net: $NET"
ls -la build/az-engine dist/az-engine.exe
