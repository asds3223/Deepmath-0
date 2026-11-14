// match.cpp — 우리 엔진끼리 직접 대국 (각 수 고정 시간, 엔진당 싱글 스레드, 폰더링 없음)
//   match --a A.dce --b B.dce --data openings.bin --pairs N --ms T [--offset k --stride s] [--a-키 값 ...] [--b-키 값 ...]
// 오프닝: 데이터의 일반 대국 국면(ply 4..10, NNUE 승률 0.3..0.7). 같은 오프닝을 색을 바꿔 2판.
// 양쪽 탐색 설정은 ab::Config 필드 이름에 a-/b- 를 붙여 따로 준다 (예: --a-rfpDepth 3).
#include "../engine/params.hpp"
#include "args.hpp"
#include "metrics.hpp"

using namespace gmk;

int main(int argc, char** argv) {
    const Args a(argc, argv);
    Model ma, mb;
    if (!ma.load(a.str("a")) || !mb.load(a.str("b"))) { std::fprintf(stderr, "cannot load models\n"); return 1; }
    std::vector<Record> recs, opens;
    readRecords(a.str("data"), recs);
    const int plyMin = int(a.num("ply-min", 4)), plyMax = int(a.num("ply-max", 10));
    const float band = float(a.num("band", 0.2));          // |NNUE 승률 - 0.5| < band
    for (auto& r : recs)
        if (!(r.flags & (REC_SIBLING | REC_LEAF | REC_SELF_FIVE | REC_OPPO_FIVE)) && r.ply >= plyMin && r.ply <= plyMax &&
            std::fabs(r.nnWinrate() - 0.5f) < band) opens.push_back(r);
    if (opens.empty()) { std::fprintf(stderr, "no openings\n"); return 1; }
    const int pairs = int(a.num("pairs", 20)), offset = int(a.num("offset", 0)), stride = int(a.num("stride", 1));
    const int64_t ms = int64_t(a.num("ms", 100));
    ab::Searcher<Board> sa(64), sb(64);
    ab::applyParams(sa.cfg, a.kv, "a-");
    ab::applyParams(sb.cfg, a.kv, "b-");
    int win = 0, draw = 0, loss = 0; double na = 0, ta = 0, nb = 0, tb = 0; long movesTotal = 0;
    int firstMoverWins = 0, pairSplit = 0, pairA2 = 0, pairB2 = 0;    // 진단: 선수(둘 차례) 승률, 쌍 결과 분포
    for (int p = 0; p < pairs; ++p) {
        const Record& o = opens[(size_t(offset) + size_t(p) * stride * 7919) % opens.size()];
        int pairScore = 0;
        for (int g = 0; g < 2; ++g) {
            Board ba, bb; ba.setModel(&ma); bb.setModel(&mb);
            o.toBoard(ba); o.toBoard(bb);
            const int colorA = g == 0 ? ba.side : opp(ba.side);
            sa.tt.clear(); sa.clearHeuristics(); sb.tt.clear(); sb.clearHeuristics();
            int result = 0;   // +1 A 승, -1 B 승
            for (;;) {
                const bool aMove = ba.side == colorA;
                Board& bd = aMove ? ba : bb;
                const ab::Result r = (aMove ? sa : sb).think(bd, ab::Limits{ab::MAX_PLY - 1, ms, 0});
                (aMove ? na : nb) += double(r.nodes); (aMove ? ta : tb) += double(r.timeMs);
                int m = r.bestMove;
                if (m < 0 || !ba.isEmptyMove(m)) { for (int q = 0; q < NMOVES; ++q) if (ba.isEmptyMove(q)) { m = q; break; } }
                ba.make(m); bb.make(m); ++movesTotal;
                if (ba.isLoss()) { result = aMove ? 1 : -1; break; }
                if (ba.isDraw()) break;
            }
            (result > 0 ? win : result < 0 ? loss : draw)++;
            pairScore += result;
            // 오프닝 국면에서 둘 차례였던 쪽이 이겼는가
            const bool aFirst = colorA == o.side;
            firstMoverWins += (result > 0 && aFirst) || (result < 0 && !aFirst);
        }
        if (pairScore == 2) ++pairA2; else if (pairScore == -2) ++pairB2; else ++pairSplit;
    }
    const int games = win + draw + loss;
    std::printf("RESULT A_win=%d draw=%d A_loss=%d games=%d | A nps=%.0f B nps=%.0f | avg moves/game=%.1f\n", win, draw, loss, games,
                ta > 0 ? na / (ta / 1000) : 0, tb > 0 ? nb / (tb / 1000) : 0, double(movesTotal) / std::max(1, games));
    std::printf("DIAG first-mover-wins=%d/%d pairs: A2=%d split=%d B2=%d\n", firstMoverWins, games, pairA2, pairSplit, pairB2);
    return 0;
}
