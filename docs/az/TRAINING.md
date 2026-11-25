# TA-NNUE AlphaZero 체스 — C++ 학습 시스템 소개

이론은 [THEORY.md](THEORY.md) 에 있다. 이 문서는 그 이론을 구현한 `az/` 의 사용법과 구조이다.
외부 의존성이 없다 (C++20 표준 라이브러리만, PyTorch·GPU 없음). 학습기·셀프플레이·대국·UCI 엔진이 바이너리 하나에 들어 있다.

## 1. 구성

```
az/
  chess.hpp     규칙: 비트보드, 합법수 생성(광선 방식), 복사-두기, FEN, Zobrist, 종국 판정
  nnue.hpp      TA-NNUE: 특징(772), 누산기 전체 계산/증분 갱신(P1), WDL 가치 + 합법수 희소 정책(P2)
  mcts.hpp      PUCT-MCTS: 노드별 누산기 저장(P1), 증명 전파(P5), Dirichlet 잡음, 온도 표집
  selfplay.hpp  셀프플레이 대국 → 학습 표본, 표본 파일 형식, 대국(arena)용 경기
  train.hpp     손실(정책 CE + WDL CE, P4 목표), 손으로 쓴 역전파, AdamW, 다중 스레드 배치
  main.cpp      명령: test / bench / selfplay / train / arena / loop / go / uci
  build.sh      → build/az
```

## 2. 빌드와 검증

```bash
az/build.sh          # g++ -O3 -march=native → build/az
build/az test        # 규칙·증분·역전파·증명 검증 (1초 이내)
build/az bench       # 평가 속도 (H2) + MCTS 속도
```

`test` 가 확인하는 것:

| 검사 | 내용 |
|---|---|
| perft | 표준 7개 국면 (시작, Kiwipete, 3·4·5번 국면) 노드 수 일치 → 합법수 생성·캐슬링·앙파상·승격 정확성 |
| FEN | 왕복 변환 일치 |
| 증분 누산기 (H1) | 무작위 대국 200판, 연쇄 증분 갱신 약 3.8만 번 vs 전체 계산: 최대 오차 1.4e-6 |
| 역전파 | 모든 매개변수 묶음(7개)에서 해석 기울기 vs 유한 차분, 5% 이내 비율 ≥ 90% (실측 334/336) |
| 증명 전파 | 메이트 1수 국면에서 무작위 망으로도 메이트 수를 고르고 뿌리를 "확정 승"으로 표시 |

## 3. 명령

| 명령 | 하는 일 | 주요 옵션 (기본값) |
|---|---|---|
| `loop` | **전체 학습 반복** (셀프플레이 → 학습 → 저장 → 주기적 대국) | `--dir run --gens 100 --games 200 --sims 1000 --threads 4 --window 8 --reuse 4 --arena-every 5 --arena-pairs 20` |
| `selfplay` | 망 하나로 셀프플레이, 표본 파일 저장 | `--net F --games 100 --out selfplay.bin` |
| `train` | 표본 파일로 학습 | `train a.bin b.bin --net F --out G --steps 1000 --report 100` |
| `arena` | 두 망 대국 (같은 오프닝을 색 바꿔 2판) | `--a F --b G|random --pairs 50 --sims 1000` |
| `go` | 한 국면 분석 (상위 10수 N/P/Q, 증명 상태) | `--net F --fen "..." --sims 1000` |
| `uci` | UCI 엔진 (cutechess, Arena 등 GUI 연결) | `--net F --sims 1000` / `go nodes N`, `go movetime`, `go wtime btime` |
| `bench` | 속도 측정 | `--net F` |

공통 탐색 옵션: `--sims --cinit --fpu --alpha --eps --ptemp`. 학습 옵션: `--batch 512 --lr 1e-3 --wd 1e-4 --lambda 0.3 --vw 1`.
셀프플레이 옵션: `--temp-plies 30 --max-plies 512 --resign 0 (끔) --no-noise`.

## 4. 학습 반복 (`loop`) 한 세대

```
net_{g-1} ──셀프플레이 (games 판, 수당 1,000 시뮬, 뿌리 잡음, 30 ply 온도 1)──▶ data_g
data_{g-W+1..g} (재생 창) ──AdamW, steps = reuse × |data_g| / batch──▶ net_g
g % arena-every == 0 이면: net_g vs net_{g-arena-every} 대국 → Elo (95% 구간)
```

- 관문 없음 (AlphaZero 방식). 대국은 측정만 한다.
- **재개:** 같은 `--dir` 로 다시 실행하면 마지막 `net_XXXX.bin`, `adam.bin`(옵티마이저 상태), 재생 창 데이터를 읽고 이어서 한다.
- 세대 수 g 의 데이터 `data_g` 는 `net_{g-1}` 이 만든 것이다.

### 출력 파일 (`--dir`)

| 파일 | 내용 |
|---|---|
| `net_XXXX.bin` | 망 가중치 (float32, 머리글 24바이트 + 약 25만 float ≈ 1 MB) |
| `data_XXXX.bin` | 그 세대 셀프플레이 표본 |
| `adam.bin` | AdamW 1·2차 모멘트 |
| `log.tsv` | 세대별 기록 (아래) |
| `live.log` | **실시간 로그** — 모든 줄에 시각, 쓰자마자 flush (아래) |

`log.tsv` 열: 세대, 대국 수, 백승/무/흑승, 메이트 수, 평균 ply, 표본 수, 셀프플레이 시간, 시뮬/초, 학습 걸음 수,
정책 CE, 정책 KL(= CE − 목표 엔트로피, 망 정책과 탐색 방문 분포의 거리), 가치 CE, top-1 일치율, 학습 시간,
대국 상대 세대, 점수, Elo, Elo 95% 반폭.

### 실시간 로그 (`live.log`, 표준 출력과 같은 내용)

```bash
tail -f data/az_run1/live.log
```

```
[2026-11-25 14:10:29] g11 selfplay 150/200 | 1-0 mate 15 ply | W/D/B 69/15/66 | 20.8s 7.21 g/s ETA 7s
[...] g11 train 50/212 steps | policy KL 0.6012  value CE 0.6050  top1 21.4% | 0.2s
[...] g15 arena 24/40 games | +15 =3 -6 | score 66.2% | 11.3s
[...]   arena net_0015 vs net_0010: +24 =5 -11  score 66.2%  Elo +117 ± 106
```

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--log-every N` | 10 (loop), 1 (selfplay) | 셀프플레이 N 판마다 한 줄: 결과, 끝난 이유, 길이, 누적 W/D/B, 속도, 남은 시간. 대국(arena)은 짝마다 한 줄 (0 이면 끔) |
| `--log-steps N` | 50 | 학습 N 걸음마다 한 줄: 그 구간 평균 정책 KL, 가치 CE, top-1 |

표준 출력은 줄 단위 버퍼라서 파일로 넘겨도(`> run.out`) 바로 보인다.

### 표본 형식 (`data_XXXX.bin`)

머리글 `AZD1`, 표본 수, 수 항목 수. 표본 하나 = 88바이트 고정부 + 합법수마다 4바이트:

| 필드 | 크기 | 뜻 |
|---|---|---|
| board | 64 | 칸별 기물 코드 (0–11, 빈칸 12) |
| stm, castle | 2 | 둘 차례, 캐슬링 권리 |
| result | 1 | 둘 차례 기준 결과 (+1/0/−1) |
| flags | 1 | bit0: 512 ply 상한 도달 |
| root[3] | 12 | 탐색 뿌리 WDL (P4 목표에 섞음) |
| off, n | 8 | 수 목록 위치와 개수 |
| 수 목록 | 4 × n | (정책 색인 16bit, 방문 수 16bit) — **모든 합법수**(방문 0 포함) |

학습 시에는 수 생성을 다시 하지 않는다. 저장된 정책 색인이 곧 softmax 대상이다.
특징은 같은 `featuresOf` 함수로 만든다 → 학습과 실전의 입력이 같다.

## 5. 속도 (이 컨테이너, 4코어, 무작위 초기 망)

| 항목 | 실측 |
|---|---|
| 평가 1회: 증분 + 희소 정책 (P1+P2) | 3.3 µs |
| 평가 1회: 전체 계산 + 4,168 출력층 | 18.1 µs (**5.4배** 느림, H2 확인) |
| MCTS (1 스레드) | 약 20만 시뮬/초 → 1,000 시뮬 = 5 ms/수 |
| 셀프플레이 (4 스레드) | 약 4.4 판/초, 약 70만 시뮬/초 |
| 학습 1 걸음 (배치 512, 4 스레드) | 약 3 ms |

셀프플레이가 시간의 대부분이다. 학습은 세대당 1초 안팎이다.

## 6. 실행 예

```bash
az/build.sh && build/az test
build/az loop --dir data/az_run1 --gens 40 --games 200      # 처음부터 (무작위 망)
build/az loop --dir data/az_run1 --gens 40 --games 200      # 같은 명령 = 이어서
build/az arena --a data/az_run1/net_0040.bin --b random --pairs 50
build/az go --net data/az_run1/net_0040.bin --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
build/az uci --net data/az_run1/net_0040.bin                 # GUI 연결
```

## 7. 첫 실행 결과

(아래 8절에 측정값을 적는다.)
