// match.cpp — 두 모델 직접 대국 (각 수 고정 시간, 엔진당 싱글 스레드, 폰더링 없음)
//   match --a A.dce --b B.dce --data openings.bin --pairs N --ms 100 [--offset k --stride s]
// 오프닝: 데이터에서 일반 대국 국면(ply 4..10, NNUE 승률 0.3..0.7)을 골라, 같은 오프닝을 색을 바꿔 2판 둔다.
#include "metrics.hpp"
#include <map>

using namespace gmk;

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
    auto num = [&](const char* k, double d) { return a.count(k) ? std::stod(a[k]) : d; };
    Model ma, mb;
    if (!ma.load(a["a"]) || !mb.load(a["b"])) { std::fprintf(stderr, "cannot load models\n"); return 1; }
    std::vector<Record> recs, opens;
    readRecords(a["data"], recs);
    for (auto& r : recs)
        if (!(r.flags & (REC_SIBLING | REC_LEAF | REC_SELF_FIVE | REC_OPPO_FIVE)) && r.ply >= 4 && r.ply <= 10 &&
            std::fabs(r.nnWinrate() - 0.5f) < 0.2f) opens.push_back(r);
    const int pairs = int(num("pairs", 20)), offset = int(num("offset", 0)), stride = int(num("stride", 1));
    const int64_t ms = int64_t(num("ms", 100));
    ab::Searcher<Board> sa(64), sb(64);
    // 양쪽 탐색 설정을 따로: --a-키 값 / --b-키 값
    auto apply = [&](ab::Config& c, const std::string& p) {
        auto g = [&](const char* k, int d) { const std::string key = p + k; return a.count(key) ? std::stoi(a[key]) : d; };
        c.useVCF = g("vcf", 0) != 0;
        c.lmrDiv100 = g("lmrdiv", c.lmrDiv100); c.lmrBase100 = g("lmrbase", c.lmrBase100);
        c.useIIR = g("iir", 1) != 0; c.useImproving = g("impr", 1) != 0; c.useCounter = g("cm", 1) != 0;
        c.qMaxPly = g("qmax", c.qMaxPly); c.cutGap = g("cutgap", c.cutGap); c.cutMinKeep = g("cutkeep", c.cutMinKeep);
        c.rpsPerLogit = g("rpsk", c.rpsPerLogit); c.keepBase = g("keepbase", c.keepBase); c.keepPerPly = g("keepply", c.keepPerPly);
        c.gateDepth = g("gate", c.gateDepth); c.gateMargin = g("gatemargin", c.gateMargin);
        c.rfpDepth = g("rfp", c.rfpDepth); c.rfpMargin = g("rfpmargin", c.rfpMargin);
        c.mcpBase = g("mcp", c.mcpBase); c.mcpExp100 = g("mcpexp", c.mcpExp100);
        c.lmrAdditive = g("lmradd", c.lmrAdditive) != 0;
    };
    apply(sa.cfg, "a-"); apply(sb.cfg, "b-");
    g_vcfBudget = int(num("vcf-budget", g_vcfBudget));
    int win = 0, draw = 0, loss = 0; double na = 0, ta = 0, nb = 0, tb = 0; long movesTotal = 0;
    for (int p = 0; p < pairs; ++p) {
        const Record& o = opens[(size_t(offset) + size_t(p) * stride * 7919) % opens.size()];
        for (int g = 0; g < 2; ++g) {
            Board ba, bb; ba.setModel(&ma); bb.setModel(&mb);
            o.toBoard(ba); o.toBoard(bb);
            const int colorA = g == 0 ? ba.side : opp(ba.side);
            sa.tt.clear(); sa.clearHeuristics(); sb.tt.clear(); sb.clearHeuristics();
            int result = 0;   // +1 A 승, -1 B 승, 0 무
            while (true) {
                const bool aMove = ba.side == colorA;
                Board& bd = aMove ? ba : bb;
                ab::Result r = (aMove ? sa : sb).think(bd, ab::Limits{ab::MAX_PLY - 1, ms, 0});
                (aMove ? na : nb) += double(r.nodes); (aMove ? ta : tb) += double(r.timeMs);
                int m = r.bestMove;
                if (m < 0 || !ba.isEmptyMove(m)) { for (int q = 0; q < NMOVES; ++q) if (ba.isEmptyMove(q)) { m = q; break; } }
                ba.make(m); bb.make(m); ++movesTotal;
                if (ba.isLoss()) { result = aMove ? 1 : -1; break; }
                if (ba.isDraw()) break;
            }
            (result > 0 ? win : result < 0 ? loss : draw)++;
        }
    }
    std::printf("RESULT A_win=%d draw=%d A_loss=%d games=%d | A nps=%.0f B nps=%.0f | avg moves/game=%.1f\n", win, draw, loss,
                win + draw + loss, ta > 0 ? na / (ta / 1000) : 0, tb > 0 ? nb / (tb / 1000) : 0,
                double(movesTotal) / std::max(1, win + draw + loss));
    return 0;
}
