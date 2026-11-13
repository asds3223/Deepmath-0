#!/usr/bin/env bash
# 우리 엔진끼리 A/B 직접 대국 (양쪽 시간 제한이라 병렬 실행도 공정). 기본 4프로세스 × 125쌍 = 1,000판, 오차 ≈ ±3%p.
#   scripts/ab.sh 이름 수당ms "A옵션" "B옵션"
#   예) scripts/ab.sh "rfp3 vs 기본" 12 "--a-rfpDepth 3" ""
# 환경: MA/MB (모델, 기본 models/c3.dce), OPEN (오프닝 데이터, 기본 data/r1_val.bin), JOBS, PAIRS
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
name=$1; ms=$2; A=${3:-}; B=${4:-}
MA=${MA:-$ROOT/models/c3.dce}; MB=${MB:-$ROOT/models/c3.dce}; OPEN=${OPEN:-$ROOT/data/r1_val.bin}
JOBS=${JOBS:-4}; PAIRS=${PAIRS:-125}
for i in $(seq 0 $((JOBS-1))); do
  "$ROOT/build/match" --a "$MA" --b "$MB" --data "$OPEN" --pairs "$PAIRS" --ms "$ms" --offset $((i*5000+17)) --stride 3 $A $B &
done | awk -v n="$name" '/RESULT/{split($2,a,"=");split($3,d,"=");split($4,b,"=");w+=a[2];dr+=d[2];l+=b[2]}
  END{t=w+dr+l; s=(w+dr/2)/t; printf "%-36s A %d-%d-%d B  A %.1f%% (±%.1f)  ≈%+.0f Elo\n",n,w,dr,l,100*s,100*1.96*sqrt(s*(1-s)/t),-400*log(1/s-1)/log(10)}'
