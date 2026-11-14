#!/usr/bin/env python3
"""가치 평가 표 묶음 배율 SPSA (적합도 = 대 Rapfi 고전, 같은 시간).

묶음 8개: CS CO J INIT CNT PSQ SH BIAS+PH. 반복 1,000, 반복당 8판
(θ+, θ− 가 같은 오프닝 2개를 색 바꿔 2판씩). 학습 오프닝은 짝수 번호, 최종 평가는 홀수 번호 400개(800판)로 SPSA 전/후를 같은 오프닝에서 비교한다.
사용: python3 tools/spsa_eval.py [--iters 1000] [--ms 100]
"""
import argparse, math, os, random, subprocess, sys, time, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--iters', type=int, default=1000)
ap.add_argument('--ms', type=int, default=100)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--eval-openings', type=int, default=400)
ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'spsa'))
ap.add_argument('--seed', type=int, default=1)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
NAMES = ['CS', 'CO', 'J', 'INIT', 'CNT', 'PSQ', 'SH', 'BIAS+PH']
K = len(NAMES)
C0, GAMMA, A0, ALPHA, ABIG = 0.2, 0.101, 0.055, 0.602, 100
LO, HI = 0.5, 1.6
N_OPEN = 1059

cmd = [os.path.join(ROOT, 'build', 'oracle2', 'oracle'), 'spsaworker', '--model', os.path.join(ROOT, 'models', 'c3.dce'),
       '--data', os.path.join(ROOT, 'data', 'r1_val.bin'), '--ms', str(args.ms), '--rapfi-ms', str(args.ms)]
workers = [subprocess.Popen(cmd, cwd=os.path.join(ROOT, 'data', 'net'), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1) for _ in range(args.workers)]

def run_jobs(jobs):
    """jobs: [(opening, theta)] → 점수 목록. 작업자 수만큼씩 나눠 보낸다."""
    res = [None] * len(jobs)
    for s in range(0, len(jobs), len(workers)):
        batch = jobs[s:s + len(workers)]
        for w, (oi, th) in zip(workers, batch):
            w.stdin.write('%d %s\n' % (oi, ' '.join('%.5f' % x for x in th)))
            w.stdin.flush()
        for j, w in enumerate(workers[:len(batch)]):
            line = w.stdout.readline().split()
            while line and line[0] == 'MESSAGE':          # Rapfi 시작 메시지는 건너뛴다
                line = w.stdout.readline().split()
            if not line or line[0] != 'R':
                sys.exit('worker failed: %r' % line)
            res[s + j] = (float(line[1]), int(line[2]))
    return res

rng = random.Random(args.seed)
theta = [1.0] * K
log = open(os.path.join(args.out, 'iter.log'), 'w')
t0 = time.time()
games = 0
for k in range(args.iters):
    ck = C0 / (k + 1) ** GAMMA
    ak = A0 / (k + 1 + ABIG) ** ALPHA
    delta = [rng.choice((-1, 1)) for _ in range(K)]
    tp = [min(HI, max(LO, t + ck * d)) for t, d in zip(theta, delta)]
    tm = [min(HI, max(LO, t - ck * d)) for t, d in zip(theta, delta)]
    o1, o2 = 2 * rng.randrange(N_OPEN // 2), 2 * rng.randrange(N_OPEN // 2)
    r = run_jobs([(o1, tp), (o2, tp), (o1, tm), (o2, tm)])
    sp = (r[0][0] + r[1][0]) / (r[0][1] + r[1][1])
    sm = (r[2][0] + r[3][0]) / (r[2][1] + r[3][1])
    games += sum(x[1] for x in r)
    for i in range(K):
        theta[i] = min(HI, max(LO, theta[i] + ak * (sp - sm) / (2 * ck * delta[i])))
    log.write(json.dumps({'k': k, 'sp': sp, 'sm': sm, 'theta': [round(x, 4) for x in theta], 'games': games,
                          'sec': round(time.time() - t0)}) + '\n')
    log.flush()
    if (k + 1) % 50 == 0:
        print('iter %d games %d  %.0fs  theta %s' % (k + 1, games, time.time() - t0,
              ' '.join('%s=%.3f' % (n, x) for n, x in zip(NAMES, theta))), flush=True)

json.dump({'names': NAMES, 'theta': theta}, open(os.path.join(args.out, 'theta.json'), 'w'))

# ── 최종 평가: 같은 홀수 오프닝에서 SPSA 전(θ=1)과 후 ──
odd = [2 * i + 1 for i in range(args.eval_openings)]
base = run_jobs([(o, [1.0] * K) for o in odd])
fin = run_jobs([(o, theta) for o in odd])
games += sum(x[1] for x in base) + sum(x[1] for x in fin)

def elo(s):
    s = min(max(s, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / s - 1)

def score(rs):
    return sum(x[0] for x in rs) / sum(x[1] for x in rs)

d = elo(score(fin)) - elo(score(base))
bs = []
br = random.Random(7)
n = len(odd)
for _ in range(2000):
    idx = [br.randrange(n) for _ in range(n)]
    bs.append(elo(score([fin[i] for i in idx])) - elo(score([base[i] for i in idx])))
bs.sort()
lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
summary = {'theta': dict(zip(NAMES, [round(x, 4) for x in theta])), 'base_score': score(base), 'final_score': score(fin),
           'delta_elo': d, 'ci95': [lo, hi], 'games_total': games, 'hours': (time.time() - t0) / 3600}
json.dump(summary, open(os.path.join(args.out, 'result.json'), 'w'), indent=1)
print('RESULT_SPSA delta=%+.1f Elo  ci95=[%+.1f, %+.1f]  base=%.3f final=%.3f  games=%d  %.2fh' %
      (d, lo, hi, score(base), score(fin), games, (time.time() - t0) / 3600), flush=True)
for w in workers:
    w.stdin.close()
