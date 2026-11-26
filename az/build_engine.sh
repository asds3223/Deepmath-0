#!/usr/bin/env bash
# UCI engine with embedded net.
#   az/build_engine.sh [NET.bin]   (default: latest data/az_run1/net_XXXX.bin)
#   → dist/az-engine.exe            Windows x64, static, generic x86-64-v2 (SSE4.2/POPCNT, no AVX requirement)
#   → dist/az-engine-android-arm64  Android arm64-v8a, static PIE, armv8-a (NEON, every 64-bit ARM phone)
#   → build/az-engine               Linux x64 (same source and flags as the .exe, used for testing)
# Needs: g++, g++-mingw-w64-x86-64-posix, g++-aarch64-linux-gnu (qemu-user-static to test the ARM build here).
set -euo pipefail
cd "$(dirname "$0")/.."
NET=${1:-$(ls data/az_run1/net_[0-9][0-9][0-9][0-9].bin | sort | tail -1)}
mkdir -p build dist
cp "$NET" build/engine_net.bin
COMMON=(-std=c++20 -O3 -fopenmp-simd -fno-math-errno -DNDEBUG -pthread "-DAZ_NET_FILE=\"$(pwd)/build/engine_net.bin\"")
X86=(-march=x86-64-v2 -mtune=generic)
g++ "${COMMON[@]}" "${X86[@]}" -o build/az-engine az/engine.cpp
x86_64-w64-mingw32-g++-posix "${COMMON[@]}" "${X86[@]}" -static -s -o dist/az-engine.exe az/engine.cpp
if command -v aarch64-linux-gnu-g++ >/dev/null; then
  aarch64-linux-gnu-g++ "${COMMON[@]}" -march=armv8-a -mtune=generic -static-pie -s -o dist/az-engine-android-arm64 az/engine.cpp
fi
echo "net: $NET"
ls -la build/az-engine dist/az-engine*
