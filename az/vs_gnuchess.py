#!/usr/bin/env python3
"""Match: our UCI engine vs GNU Chess 5.x (xboard mode), python-chess adjudication.

  python3 az/vs_gnuchess.py --gnu PATH/gnuchess --engine build/az-engine --nodes 1000 --gnu-depth 2 --pairs 10

Each opening is played twice with colors swapped. Our engine: fixed nodes (UCI 'go nodes'), Threads 1.
GNU Chess: fixed depth ('depth N') or fixed seconds per move ('st N'), pondering off ('easy').
Both limits are independent of machine load. Prints W/D/L from our engine's view and an Elo estimate.
"""
import argparse, math, re, subprocess, sys
import chess, chess.engine

OPENINGS = [  # 10 common lines (UCI), balanced and varied
    "e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6", "e2e4 c7c5 g1f3 d7d6", "d2d4 g8f6 c2c4 g7g6",
    "e2e4 e7e6 d2d4 d7d5", "c2c4 e7e5 b1c3 g8f6", "g1f3 d7d5 g2g3 g8f6", "e2e4 c7c6 d2d4 d7d5",
    "d2d4 g8f6 c2c4 e7e6", "e2e4 e7e5 f1c4 g8f6",
]


class GnuChess:
    def __init__(self, path, depth=0, st=0):
        self.p = subprocess.Popen([path, "-x"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("xboard"); self.send("new"); self.send("easy")
        if depth: self.send(f"depth {depth}")
        if st: self.send(f"st {st}")

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def play(self, board):
        self.send("force"); self.send("setboard " + board.fen()); self.send("go")
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("gnuchess exited")
            m = re.search(r"(?:My move is\s*:\s*|^move\s+)(\S+)", line)
            if m:
                txt = m.group(1).strip()
                try:
                    mv = chess.Move.from_uci(txt.lower())
                    if mv in board.legal_moves:
                        return mv
                except ValueError:
                    pass
                return board.parse_san(txt)

    def quit(self):
        try:
            self.send("quit"); self.p.wait(5)
        except Exception:
            self.p.kill()


def play_game(az, gnu, opening, az_white, nodes, max_plies=300):
    board = chess.Board()
    for u in opening.split():
        board.push_uci(u)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == az_white:
            mv = az.play(board, chess.engine.Limit(nodes=nodes)).move
        else:
            mv = gnu.play(board)
        board.push(mv)
    res = board.result(claim_draw=True)
    if res == "*":
        res = "1/2-1/2"
    score = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    return score if az_white else 1.0 - score, board.ply()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnu", required=True)
    ap.add_argument("--engine", default="build/az-engine")
    ap.add_argument("--nodes", type=int, default=1000)
    ap.add_argument("--gnu-depth", type=int, default=0)
    ap.add_argument("--gnu-st", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--evalfile", default="")
    a = ap.parse_args()
    az = chess.engine.SimpleEngine.popen_uci([a.engine])
    opts = {"Threads": 1, "Hash": 128}
    if a.evalfile:
        opts["EvalFile"] = a.evalfile
    az.configure(opts)
    w = d = l = 0
    for i in range(a.pairs):
        op = OPENINGS[i % len(OPENINGS)]
        for az_white in (True, False):
            gnu = GnuChess(a.gnu, a.gnu_depth, a.gnu_st)
            s, plies = play_game(az, gnu, op, az_white, a.nodes)
            gnu.quit()
            w += s == 1.0; d += s == 0.5; l += s == 0.0
            print(f"game {w+d+l}: {'W' if s == 1 else 'D' if s == 0.5 else 'L'} ({'white' if az_white else 'black'}, {op}, {plies} ply)  total +{w} ={d} -{l}", flush=True)
    az.quit()
    n = w + d + l
    sc = (w + 0.5 * d) / n
    elo = -400 * math.log10(1 / min(max(sc, 1e-3), 1 - 1e-3) - 1)
    lim = f"depth {a.gnu_depth}" if a.gnu_depth else f"st {a.gnu_st}s"
    print(f"RESULT engine(nodes {a.nodes}) vs GNU Chess ({lim}): +{w} ={d} -{l}  score {100*sc:.1f}%  Elo {elo:+.0f}", flush=True)


if __name__ == "__main__":
    main()
