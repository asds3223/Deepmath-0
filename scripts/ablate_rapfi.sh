#!/usr/bin/env bash
# Rapfi(고전평가 모드)의 탐색 기능을 하나씩 끄고 우리 승률 변화를 잰다 → 어떤 원리가 핵심인지.
# 양쪽 노드 고정이라 병렬 실행도 공정. 설정당 4프로세스 × 50쌍 = 400판.
# 환경: MODEL, ONODES (우리 노드, 기본 4500), RNODES (Rapfi 노드, 기본 8000), OPEN
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=${MODEL:-$ROOT/models/c3.dce}; OPEN=${OPEN:-$ROOT/data/r1_val.bin}
cd "$ROOT/data/net"
run() { name=$1; off=$2
  for i in 0 1 2 3; do "$ROOT/build/oracle/oracle" match --model "$MODEL" --data "$OPEN" --ours search --nodes "${ONODES:-4500}" \
      --rapfi search --rapfi-nodes "${RNODES:-8000}" --rapfi-classical 1 ${off:+--rapfi-off $off} --pairs 50 --offset $((i*5000+17)) --stride 3 2>/dev/null | grep RESULT & done \
  | awk -v n="$name" '{split($2,a,"=");split($4,b,"=");w+=a[2];l+=b[2]} END{t=w+l; s=w/t; printf "%-24s ours %3d-%3d  %.1f%% (±%.1f)\n",n,w,l,100*s,100*1.96*sqrt(s*(1-s)/t)}'
}
run "baseline (all on)" ""
for f in vcfleaf razor futility null lmp oppo5ext singular lmr failhigh defend iirtt; do run "off: $f" "$f"; done
