#!/usr/bin/env bash
# 실험 예산 장부: data/runs.log 의 판 수 합계와 상한(30,000판 / 7일, 5,000판마다 재평가) 대비 사용량
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
awk -F'\t' '{split($7,g,"games="); n+=g[2]; t+=$3} END{printf "games used %d / 30000 (next review at %d), wall %.1f h\n", n, (int(n/5000)+1)*5000, t/3600}' "$ROOT/data/runs.log" 2>/dev/null || echo "no runs logged"
