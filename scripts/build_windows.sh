#!/usr/bin/env bash
# Windows x64 AVX2 전용 엔진 빌드 (MinGW-w64 크로스 컴파일, 정적 링크, 가중치 내장)
#   scripts/build_windows.sh [MODEL.dce]   → dist/pbrain-dce.exe
# 필요: apt-get install g++-mingw-w64-x86-64-posix
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-models/c3.dce}
mkdir -p build dist
python3 scripts/embed_model.py "$MODEL" build/model_data.h
x86_64-w64-mingw32-g++ -O3 -std=c++17 -march=haswell -mtune=generic -DNDEBUG -DDCE_EMBED_MODEL -Ibuild \
    -static -s engine/pbrain.cpp -o dist/pbrain-dce.exe
echo "built dist/pbrain-dce.exe (model: $MODEL)"
