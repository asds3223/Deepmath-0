#!/usr/bin/env bash
# 전체 빌드. ORACLE_ISA=AVX2 로 바꾸면 AVX512 없는 CPU 용 오라클.
set -euo pipefail
cd "$(dirname "$0")/.."
ISA=${ORACLE_ISA:-AVX512VNNI}
CXX=${CXX:-clang++}
FLAGS="-O3 -std=c++17 -march=native -DNDEBUG -pthread"
mkdir -p build
cmake -S oracle -B build/oracle -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DORACLE_ISA=$ISA >/dev/null
cmake --build build/oracle -j"$(nproc)" | tail -1
$CXX $FLAGS engine/test_alphabeta.cpp -o build/test_ab
$CXX $FLAGS engine/test_board.cpp     -o build/test_board
$CXX $FLAGS engine/test_incremental.cpp -o build/test_incremental
$CXX $FLAGS tools/train.cpp           -o build/train
$CXX $FLAGS tools/evaltool.cpp        -o build/evaltool
$CXX $FLAGS tools/match.cpp           -o build/match
$CXX $FLAGS engine/pbrain.cpp         -o build/pbrain-dce
echo "built: build/{oracle/oracle,train,evaltool,pbrain-dce,test_ab,test_board}"
