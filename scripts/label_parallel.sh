#!/usr/bin/env bash
# 대량 재라벨(예: DAgger 리프): 입력을 코어 수로 쪼개 oracle label 병렬 실행
#   scripts/label_parallel.sh IN.bin OUT.bin [--search-nodes N]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
JOBS=${JOBS:-$(nproc)}
TMP=$(mktemp -d "${OUT}.parts.XXXX")
N=$(( $(stat -c %s "$IN") / 136 )); PER=$(( (N + JOBS - 1) / JOBS ))
split -b $((PER*136)) -d -a 3 "$IN" "$TMP/in"
cd "$ROOT/data/net"
for f in "$TMP"/in*; do
  "$ROOT/build/oracle/oracle" label --in "$f" --out "$f.out" "$@" 2> "$f.log" &
done
wait
cat "$TMP"/in*.out > "$OUT"
echo "labelled: $(( $(stat -c %s "$OUT") / 136 )) / $N -> $OUT"
rm -rf "$TMP"
