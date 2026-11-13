// evaltool.cpp — 증류 결과 측정 / DAgger 리프 수집
//
//   static   --model m --data f                 주 지표: 정적 판단력 (구간별 가치 정확도, 형제 순위, Rapfi 고전평가 대비)
//   search   --model m --data f --nodes N|--ms T [--limit K]
//            보조 지표: 탐색 라벨이 있는 국면에서 우리 엔진 최선수 == Rapfi 최선수 비율, 형세 판정 일치
//   leafdump --model m --data f --nodes N --prob p --out leaves.bin
//            우리 탐색 트리의 리프 국면을 확률 p 로 저장 → oracle label 로 재라벨 → 재학습 (분포 이동 보정)
#include "metrics.hpp"
#include <chrono>
#include <map>
#include <random>

using namespace gmk;

namespace {
std::map<std::string, std::string> parseArgs(int argc, char** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a[k] = argv[++i]; else a[k] = "1";
    }
    return a;
}
double num(const std::map<std::string, std::string>& a, const char* k, double d) {
    auto it = a.find(k); return it == a.end() ? d : std::stod(it->second);
}

// 리프 덤프용 Position: evaluate() 를 가로채 일정 확률로 국면을 기록
struct DumpBoard : Board {
    mutable std::mt19937_64* rng = nullptr;
    mutable RecordWriter* out = nullptr;
    float prob = 0;
    mutable uint64_t dumped = 0;
    int evaluate() const {
        if (out && std::uniform_real_distribution<float>(0, 1)(*rng) < prob) {
            Record r; std::memset(&r, 0, sizeof r);
            r.setBoard(*this); r.flags = REC_LEAF; r.srchBest = 255;
            out->write(r); ++dumped;
        }
        return Board::evaluate();
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: evaltool <static|search|leafdump> --model m --data f ...\n"); return 2; }
    const std::string mode = argv[1];
    auto a = parseArgs(argc, argv, 2);
    Model mdl;
    if (a.count("model") && !mdl.load(a["model"])) { std::fprintf(stderr, "cannot load model\n"); return 1; }
    if (a.count("max-cands")) mdl.maxCands = int(num(a, "max-cands", 0));
    std::vector<Record> recs;
    if (!readRecords(a["data"], recs)) { std::fprintf(stderr, "cannot read data\n"); return 1; }
    const size_t limit = size_t(num(a, "limit", 1e18));

    if (mode == "static") {
        metrics::Report rep; metrics::evalSet(mdl, recs, rep); rep.print("[static]", true);
        return 0;
    }
    if (mode == "search") {
        const uint64_t nodes = uint64_t(num(a, "nodes", 0));
        const int64_t ms = int64_t(num(a, "ms", -1));
        ab::Searcher<Board> s(size_t(num(a, "tt-mb", 64)));
        s.cfg.useQuiesce = num(a, "q", 1) != 0; s.cfg.useRPS = num(a, "rps", 1) != 0; s.cfg.usePolicyCut = num(a, "cut", 1) != 0;
        s.cfg.useVCF = num(a, "vcf", 0) != 0;
        s.cfg.keepBase = int(num(a, "keepbase", s.cfg.keepBase)); s.cfg.keepPerPly = int(num(a, "keepply", s.cfg.keepPerPly));
        s.cfg.gateDepth = int(num(a, "gate", s.cfg.gateDepth)); s.cfg.gateMargin = int(num(a, "gatemargin", s.cfg.gateMargin));
        g_tacSelf = int(num(a, "tacself", g_tacSelf)); g_tacOppo = int(num(a, "tacoppo", g_tacOppo));
        g_vcfBudget = int(num(a, "vcfbudget", g_vcfBudget)); g_vcfMaxPly = int(num(a, "vcfply", g_vcfMaxPly));
        s.cfg.qThreePly = int(num(a, "qthree", s.cfg.qThreePly)); s.cfg.qMaxPly = int(num(a, "qmax", s.cfg.qMaxPly));
        s.cfg.cutGap = int(num(a, "cutgap", s.cfg.cutGap)); s.cfg.cutMinKeep = int(num(a, "cutkeep", s.cfg.cutMinKeep));
        s.cfg.rpsPerLogit = int(num(a, "rpsk", s.cfg.rpsPerLogit)); s.cfg.rpsMaxExtra = int(num(a, "rpsmax", s.cfg.rpsMaxExtra));
        size_t n = 0, agree = 0, verdictAgree = 0; double depthSum = 0, nodeSum = 0, msSum = 0;
        size_t rWin = 0, rWinFound = 0, rLoss = 0, rLossFound = 0; double intSum = 0, srchSum = 0;
        for (const Record& r : recs) {
            if (!(r.flags & REC_HAS_SEARCH) || r.srchBest == 255) continue;
            if (n >= limit) break;
            Board b; b.setModel(a.count("model") ? &mdl : nullptr); r.toBoard(b);
            s.tt.clear(); s.clearHeuristics();
            ab::Result res = s.think(b, ab::Limits{ab::MAX_PLY - 1, ms, nodes});
            ++n; agree += (res.bestMove == r.srchBest);
            const float ourWR = std::abs(res.score) >= ab::MATE_BOUND ? (res.score > 0 ? 1.f : 0.f)
                                                                     : metrics::sigmoid(res.score / mdl.scale);
            verdictAgree += ((ourWR - 0.5f) * (r.srchWR - 0.5f) >= 0);
            if (r.srchWR >= 0.999f) { ++rWin; rWinFound += res.score >= ab::MATE_BOUND; }
            if (r.srchWR <= 0.001f) { ++rLoss; rLossFound += res.score <= -ab::MATE_BOUND; }
            depthSum += res.depth; nodeSum += res.nodes; msSum += res.timeMs; intSum += res.interior; srchSum += res.searched;
        }
        if (!n) { std::printf("no search-labelled records\n"); return 0; }
        std::printf("[search] n=%zu bestmove-agree=%.3f verdict-agree=%.3f avg-depth=%.2f avg-nodes=%.0f avg-ms=%.1f nps=%.0f\n",
                    n, double(agree) / n, double(verdictAgree) / n, depthSum / n, nodeSum / n, msSum / n,
                    msSum > 0 ? nodeSum / (msSum / 1000) : 0);
        std::printf("[search] branching: searched/interior=%.2f\n", intSum > 0 ? srchSum / intSum : 0);
        std::printf("[search] Rapfi-says-WIN n=%zu ours-found-mate=%.3f | Rapfi-says-LOSS n=%zu ours-found-mated=%.3f\n",
                    rWin, rWin ? double(rWinFound) / rWin : 0, rLoss, rLoss ? double(rLossFound) / rLoss : 0);
        return 0;
    }
    if (mode == "bench") {
        // 평가 1회 시간 + 고정 노드 탐색 속도
        Board b; b.setModel(&mdl);
        std::vector<Record> pos;
        for (auto& r : recs) if (!(r.flags & (REC_SELF_FIVE | REC_SIBLING | REC_LEAF)) && r.ply >= 8) pos.push_back(r);
        if (pos.size() > 2000) pos.resize(2000);
        long long acc = 0; int mv[NMOVES];
        // 같은 보드에서 반복 호출해 순수 비용만 잰다
        double evalNs = 0, genNs = 0; long n = 0;
        for (auto& r : pos) {
            r.toBoard(b);
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 64; ++k) acc += b.evaluate();
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < 64; ++k) acc += b.generate(mv);
            auto t2 = std::chrono::steady_clock::now();
            evalNs += std::chrono::duration<double, std::nano>(t1 - t0).count();
            genNs += std::chrono::duration<double, std::nano>(t2 - t1).count();
            n += 64;
        }
        const double evalUs = evalNs / n / 1000, genUs = genNs / n / 1000;
        const uint64_t nodes = uint64_t(num(a, "nodes", 50000));
        ab::Searcher<Board> s(64); double tn = 0, tms = 0, dep = 0; int k = 0;
        for (size_t i = 0; i < pos.size() && k < int(num(a, "searches", 40)); i += 50, ++k) {
            pos[i].toBoard(b); s.tt.clear(); s.clearHeuristics();
            ab::Result r = s.think(b, ab::Limits{ab::MAX_PLY - 1, -1, nodes});
            tn += double(r.nodes); tms += double(r.timeMs); dep += r.depth;
        }
        std::printf("[bench] evaluate=%.2fus generate(+policy sort)=%.2fus | search %llu nodes x%d: nps=%.0f avg-depth=%.2f (acc=%lld)\n",
                    evalUs, genUs, (unsigned long long)nodes, k, tn / (tms / 1000), dep / k, acc);
        return 0;
    }
    if (mode == "leafdump") {
        const uint64_t nodes = uint64_t(num(a, "nodes", 20000));
        RecordWriter w(a.count("out") ? a["out"] : "leaves.bin");
        std::mt19937_64 rng(uint64_t(num(a, "seed", 5)));
        ab::Searcher<DumpBoard> s(32);
        DumpBoard b; b.setModel(&mdl); b.rng = &rng; b.out = &w; b.prob = float(num(a, "prob", 0.002));
        size_t n = 0;
        for (const Record& r : recs) {
            if (n++ >= limit) break;
            r.toBoard(b);
            s.think(b, ab::Limits{ab::MAX_PLY - 1, -1, nodes});
        }
        std::printf("[leafdump] roots=%zu leaves=%llu\n", n, (unsigned long long)b.dumped);
        return 0;
    }
    std::fprintf(stderr, "unknown mode\n");
    return 2;
}
