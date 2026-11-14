#!/usr/bin/env bash
# 우리 엔진 vs Rapfi, 단일 프로세스 실시간 대국 (100쌍 = 200판).
#   scripts/vs_rapfi.sh 이름 우리수당ms [우리 탐색 옵션...]
# 환경: MODEL (기본 models/c3.dce), RNODES (Rapfi 수당 노드, 기본 4000), RCLS=1 (Rapfi 고전평가 모드), OPEN, PAIRS
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
name=$1; ms=$2; shift 2
MODEL=${MODEL:-$ROOT/models/c3.dce}; OPEN=${OPEN:-$ROOT/data/r1_val.bin}; PAIRS=${PAIRS:-100}
cd "$ROOT/data/net"
"$ROOT/build/oracle/oracle" match --model "$MODEL" --data "$OPEN" --ours search --ms "$ms" --rapfi search \
    --rapfi-nodes "${RNODES:-4000}" ${RCLS:+--rapfi-classical 1} --pairs "$PAIRS" --offset 17 --stride 3 "$@" 2>/dev/null \
  | grep -E "RESULT|ours:|rapfi:" | tee /dev/stderr | tr '\n' ' ' \
  | awk -v n="$name" '{split($2,a,"=");split($3,d,"=");split($4,b,"=");t=a[2]+d[2]+b[2];
      printf "%-30s %d-%d-%d %.1f%% (무승부 0.5점) | ours %s nodes %s ms | rapfi %s nodes %s ms\n",n,a[2],d[2],b[2],100*(a[2]+d[2]/2)/t,$9,$11,$14,$16}'
