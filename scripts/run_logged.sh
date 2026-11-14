#!/usr/bin/env bash
# 모든 실험 실행의 기록 래퍼. 사용: scripts/run_logged.sh 이름 명령 [인자...]
# 기록(data/runs.log, 한 줄 = 한 실행): 시작·종료 시각(UTC), 걸린 초, 종료 코드·종료 사유, 소스 커밋(+더러움 표시),
#   쓰인 실행 파일 sha256(ORACLE/MATCH 환경변수와 build/ 아래 기본 파일), 명령 전체, 출력 파일, 출력의 RESULT 합계(승·무·패·판).
# 출력 전체는 data/runs/<시작시각>-<이름>.out 에 남는다. 예산 장부는 scripts/budget.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
name=$1; shift
mkdir -p "$ROOT/data/runs"
start=$(date -u +%Y-%m-%dT%H:%M:%SZ); t0=$(date +%s)
out="$ROOT/data/runs/${start//:/}-${name// /_}.out"
commit=$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet -- engine tools oracle scripts || echo "+dirty")
bins=""
for b in "${ORACLE:-$ROOT/build/oracle/oracle}" "${MATCH:-$ROOT/build/match}"; do [ -f "$b" ] && bins+="$(basename "$b")=$(sha256sum "$b" | cut -c1-16) "; done
reason=normal
trap 'reason=interrupted' INT TERM
"$@" > "$out" 2>&1; code=$?
[ "$reason" = normal ] && [ $code -ne 0 ] && reason="exit$code"
end=$(date -u +%Y-%m-%dT%H:%M:%SZ); secs=$(( $(date +%s) - t0 ))
wdl=$(grep -ho "RESULT.*" "$out" | awk '{for(i=2;i<=4;i++){split($i,x,"=");s[i]+=x[2]}} END{printf "W=%d D=%d L=%d games=%d", s[2],s[3],s[4],s[2]+s[3]+s[4]}')
printf '%s\t%s\t%ss\t%s\t%s\t%s\t%s\t%s\t%s\n' "$start" "$end" "$secs" "$reason" "$commit" "$bins" "$wdl" "$out" "$*" >> "$ROOT/data/runs.log"
cat "$out"
