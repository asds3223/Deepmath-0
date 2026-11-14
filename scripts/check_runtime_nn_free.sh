#!/usr/bin/env bash
# 배포(런타임) 바이너리에 신경망·Rapfi 가 없는지 검사한다. 실패하면 0 이 아닌 코드로 끝난다.
#   1) engine/pbrain.cpp 의 컴파일 의존 파일이 허용 목록(engine/ 아래 런타임 헤더)뿐인가
#   2) 그 파일들에 Rapfi/신경망 관련 include·식별자가 없는가
#   3) 빌드된 배포 바이너리 문자열에 rapfi|mix9|nnue|onnx|lz4|torch 가 없는가
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
allowed="engine/pbrain.cpp engine/eval.hpp engine/features.hpp engine/board.hpp engine/alphabeta.hpp engine/pattern.hpp engine/params.hpp"
deps=$(g++ -std=c++17 -MM engine/pbrain.cpp | tr ' \\' '\n\n' | grep -v -e '^$' -e ':$' | sort -u)
for d in $deps; do case " $allowed " in *" $d "*) ;; *) echo "FAIL dependency outside runtime allow-list: $d"; fail=1;; esac; done
if grep -nE '#include.*(rapfi|Rapfi|oracle|third_party)|Evaluation::|Search::Threads|mix9|onnx|torch' $deps; then echo "FAIL forbidden reference in runtime sources"; fail=1; fi
for b in dist/pbrain-dce.exe build/pbrain-dce; do
  [ -f "$b" ] || continue
  n=$(strings -n 5 "$b" | grep -icE 'rapfi|mix9|nnue|onnx|lz4|torch')
  [ "$n" -eq 0 ] && echo "ok   $b: no NN/Rapfi strings (sha256 $(sha256sum "$b" | cut -c1-16))" || { echo "FAIL $b: $n suspicious strings"; fail=1; }
done
[ $fail -eq 0 ] && echo "PASS runtime path is NN-free (deps: $(echo $deps | wc -w) files)"
exit $fail
