#!/usr/bin/env bash
# 학습 데이터 생성 레시피 (C 모델에 쓴 구성). 4코어 기준 생성 수 분, 리프 라벨 수 분.
#   1) 교사 롤아웃 3종 (온도·섭동·오프닝 길이 다양화, 형제 묶음 30%)
#   2) 현재 모델의 탐색 트리 리프를 수집해 NNUE 라벨 (평가가 실제로 쓰이는 분포)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MODEL=${MODEL:-models/c3.dce}
JOBS=4 SEED0=501 scripts/gen_parallel.sh data/r2_a.bin 16000 --sib-prob 0.3 --temp-late 0.6
JOBS=4 SEED0=601 scripts/gen_parallel.sh data/r2_b.bin 10000 --sib-prob 0.3 --temp-late 1.0 --eps 0.08
JOBS=4 SEED0=701 scripts/gen_parallel.sh data/r2_c.bin 10000 --sib-prob 0.3 --temp-late 0.4 --open 12
for f in r2_a r2_b r2_c; do
  ./build/evaltool leafdump --model "$MODEL" --data data/$f.bin --limit 8000 --nodes 5000 --prob 0.01 --seed $RANDOM --out data/leaf_$f.raw &
done; wait
cat data/leaf_*.raw > data/r2_leaf.raw && rm data/leaf_*.raw
JOBS=4 scripts/label_parallel.sh data/r2_leaf.raw data/r2_leaf.bin
