#!/usr/bin/env bash
# 우리 엔진 vs Rapfi, 4프로세스 병렬 (기본 4×100쌍 = 800판, 오차 약 ±3.5%p). 셀프 대국 A/B 의 외부 검증용.
#   scripts/vs_rapfi_par.sh 이름 [우리 탐색 옵션...]      예) scripts/vs_rapfi_par.sh "defCost4" --defCost 4
# 환경: ONODES (우리 수당 노드, 기본 10000 — 노드 고정이라 CPU 경합에 무관), RNODES (Rapfi 수당 노드, 기본 8000),
#       RCLS=1 (Rapfi 고전평가 모드, 기본 켬; 빈 값이면 NNUE), MODEL, OPEN, JOBS, PAIRS, ORACLE (실행 파일)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
name=$1; shift
MODEL=${MODEL:-$ROOT/models/c3.dce}; OPEN=${OPEN:-$ROOT/data/r1_val.bin}
JOBS=${JOBS:-4}; PAIRS=${PAIRS:-100}; RCLS=${RCLS-1}
cd "$ROOT/data/net"
for i in $(seq 0 $((JOBS-1))); do
  "${ORACLE:-$ROOT/build/oracle/oracle}" match --model "$MODEL" --data "$OPEN" --ours search --nodes "${ONODES:-10000}" --rapfi search \
      --rapfi-nodes "${RNODES:-8000}" ${RCLS:+--rapfi-classical 1} --pairs "$PAIRS" --offset $((i*5000+17)) --stride 3 "$@" 2>/dev/null | grep RESULT &
done | tee /dev/stderr | awk -v n="$name" '{split($2,a,"=");split($3,d,"=");split($4,b,"=");w+=a[2];dr+=d[2];l+=b[2]}
  END{t=w+dr+l; s=(w+dr/2)/t; printf "%-36s ours %d-%d-%d  %.1f%% (±%.1f)\n",n,w,dr,l,100*s,100*1.96*sqrt(s*(1-s)/t)}'
