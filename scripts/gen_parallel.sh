#!/usr/bin/env bash
# 대량 생성: 코어 수만큼 oracle gen 프로세스를 띄우고 결과를 합친다.
#   scripts/gen_parallel.sh OUT.bin GAMES_PER_PROC [추가 oracle 옵션...]
#   예) scripts/gen_parallel.sh data/train1.bin 5000 --search-nodes 50000 --search-prob 0.02
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=$(realpath -m "$1"); GAMES=$2; shift 2
JOBS=${JOBS:-$(nproc)}
SEED0=${SEED0:-$RANDOM}
TMP=$(mktemp -d "${OUT}.parts.XXXX")
cd "$ROOT/data/net"     # config.toml 의 모델 상대경로 기준
for i in $(seq 0 $((JOBS-1))); do
  "$ROOT/build/oracle/oracle" gen --games "$GAMES" --seed $((SEED0*1000+i)) --out "$TMP/part$i.bin" "$@" 2> "$TMP/log$i.txt" &
done
wait
cat "$TMP"/part*.bin > "$OUT"
tail -n1 "$TMP"/log*.txt
echo "records: $(( $(stat -c %s "$OUT") / 136 )) -> $OUT"
rm -rf "$TMP"
