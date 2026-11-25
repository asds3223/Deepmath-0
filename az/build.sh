#!/usr/bin/env bash
# Build the TA-NNUE AlphaZero chess binary → build/az
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
g++ -std=c++20 -O3 -march=native -fopenmp-simd -fno-math-errno -pthread -Wall -Wextra -o build/az az/main.cpp
echo "built build/az"
