// evaltool.cpp — 모델·탐색 측정 도구
//
//   static      --model m --data f               정적 판단력 (구간별 가치 정확도, 형제 순위, 정책 top-k, Rapfi 고전평가 대비)
//   search      --model m --data f --nodes N|--ms T [--limit K]
//               탐색 라벨 국면: Rapfi 최선수 일치, 형세 일치, 필승/필패 발견률, 분기·정렬 통계
//   bench       --model m --data f [--nodes N]   평가·수 생성 비용, 고정 노드 탐색 nps
//   fingerprint --model m --data f [--nodes N] [--limit K]
//               고정 국면·고정 노드 탐색 결과 해시 — 리팩터링 전후 동작이 같은지 확인
//   leafdump    --model m --data f --nodes N --prob p --out leaves.bin
//               우리 탐색 트리의 리프를 확률 p 로 저장 (→ oracle label 로 재라벨해 학습 분포 보정)
//
// 탐색 설정은 ab::Config 필드 이름으로 덮어쓸 수 있다 (예: --rfpDepth 3 --qMaxPly 6).
#include "../engine/params.hpp"
#include "args.hpp"
#include "metrics.hpp"
#include <chrono>
#include <random>

using namespace gmk;

namespace {
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

double perSec(double n, double ms) { return ms > 0 ? n / (ms / 1000) : 0; }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: evaltool <static|search|bench|fingerprint|leafdump> --model m --data f ...\n"); return 2; }
    const std::string mode = argv[1];
    const Args a(argc, argv, 2);
    Model mdl;
    const bool hasModel = a.has("model");
    if (hasModel && !mdl.load(a.str("model"))) { std::fprintf(stderr, "cannot load model\n"); return 1; }
    std::vector<Record> recs;
    if (!readRecords(a.str("data"), recs)) { std::fprintf(stderr, "cannot read data\n"); return 1; }
    const size_t limit = size_t(a.num("limit", 1e18));
    ab::Config cfg;
    ab::applyParams(cfg, a.kv);
    auto searchLabelled = [](const Record& r) { return (r.flags & REC_HAS_SEARCH) && r.srchBest != 255; };

    if (mode == "static") {
        metrics::Report rep; metrics::evalSet(mdl, recs, rep); rep.print("[static]", true);
        return 0;
    }
    if (mode == "search") {
        const uint64_t nodes = uint64_t(a.num("nodes", 0));
        const int64_t ms = int64_t(a.num("ms", -1));
        ab::Searcher<Board> s(size_t(a.num("tt-mb", 64)));
        s.cfg = cfg;
        size_t n = 0, agree = 0, verdict = 0, rWin = 0, rWinFound = 0, rLoss = 0, rLossFound = 0;
        double depthSum = 0, nodeSum = 0, msSum = 0, intSum = 0, srchSum = 0, cf = 0, ct = 0, qn = 0, bt[2][3] = {};
        for (const Record& r : recs) {
            if (!searchLabelled(r)) continue;
            if (n >= limit) break;
            Board b; b.setModel(hasModel ? &mdl : nullptr); r.toBoard(b);
            s.tt.clear(); s.clearHeuristics();
            const ab::Result res = s.think(b, ab::Limits{ab::MAX_PLY - 1, ms, nodes});
            ++n; agree += res.bestMove == r.srchBest;
            const float wr = std::abs(res.score) >= ab::MATE_BOUND ? (res.score > 0 ? 1.f : 0.f) : metrics::sigmoid(res.score / mdl.scale);
            verdict += (wr - 0.5f) * (r.srchWR - 0.5f) >= 0;
            if (r.srchWR >= 0.999f) { ++rWin; rWinFound += res.score >= ab::MATE_BOUND; }
            if (r.srchWR <= 0.001f) { ++rLoss; rLossFound += res.score <= -ab::MATE_BOUND; }
            depthSum += res.depth; nodeSum += double(res.nodes); msSum += double(res.timeMs);
            for (int x = 0; x < 2; ++x) for (int y = 0; y < 3; ++y) bt[x][y] += double(res.byType[x][y]);
            intSum += double(res.interior); srchSum += double(res.searched); cf += double(res.cutFirst); ct += double(res.cutTotal); qn += double(res.qnodes);
        }
        if (!n) { std::printf("no search-labelled records\n"); return 0; }
        std::printf("[search] n=%zu bestmove-agree=%.3f verdict-agree=%.3f avg-depth=%.2f avg-nodes=%.0f avg-ms=%.1f nps=%.0f\n",
                    n, double(agree) / n, double(verdict) / n, depthSum / n, nodeSum / n, msSum / n, perSec(nodeSum, msSum));
        std::printf("[search] branching: searched/interior=%.2f first-move-cut=%.3f qnode-share=%.3f interior-share=%.3f\n",
                    intSum > 0 ? srchSum / intSum : 0, ct > 0 ? cf / ct : 0, nodeSum > 0 ? qn / nodeSum : 0, nodeSum > 0 ? intSum / nodeSum : 0);
        std::printf("[search] node share by entering move: root-side atk=%.3f def=%.3f quiet=%.3f | opp-side atk=%.3f def=%.3f quiet=%.3f\n",
                    bt[0][0] / nodeSum, bt[0][1] / nodeSum, bt[0][2] / nodeSum, bt[1][0] / nodeSum, bt[1][1] / nodeSum, bt[1][2] / nodeSum);
        std::printf("[search] Rapfi-says-WIN n=%zu ours-found-mate=%.3f | Rapfi-says-LOSS n=%zu ours-found-mated=%.3f\n",
                    rWin, rWin ? double(rWinFound) / rWin : 0, rLoss, rLoss ? double(rLossFound) / rLoss : 0);
        return 0;
    }
    if (mode == "bench") {
        Board b; b.setModel(&mdl);
        std::vector<Record> pos;
        for (auto& r : recs) if (!(r.flags & (REC_SELF_FIVE | REC_SIBLING | REC_LEAF)) && r.ply >= 8) pos.push_back(r);
        if (pos.size() > 2000) pos.resize(2000);
        long long acc = 0; int mv[NMOVES]; double evalNs = 0, genNs = 0; long cnt = 0;
        for (auto& r : pos) {                      // 같은 보드에서 반복 호출해 순수 비용만 잰다
            r.toBoard(b);
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 64; ++k) acc += b.evaluate();
            const auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < 64; ++k) acc += b.generate(mv);
            const auto t2 = std::chrono::steady_clock::now();
            evalNs += std::chrono::duration<double, std::nano>(t1 - t0).count();
            genNs += std::chrono::duration<double, std::nano>(t2 - t1).count();
            cnt += 64;
        }
        const uint64_t nodes = uint64_t(a.num("nodes", 50000));
        ab::Searcher<Board> s(64);
        s.cfg = cfg;
        double tn = 0, tms = 0, dep = 0; int k = 0;
        for (size_t i = 0; i < pos.size() && k < int(a.num("searches", 40)); i += 50, ++k) {
            pos[i].toBoard(b); s.tt.clear(); s.clearHeuristics();
            const ab::Result r = s.think(b, ab::Limits{ab::MAX_PLY - 1, -1, nodes});
            tn += double(r.nodes); tms += double(r.timeMs); dep += r.depth;
        }
        std::printf("[bench] evaluate=%.3fus generate=%.3fus | search %llu nodes x%d: nps=%.0f avg-depth=%.2f (acc=%lld)\n",
                    evalNs / cnt / 1000, genNs / cnt / 1000, (unsigned long long)nodes, k, perSec(tn, tms), k ? dep / k : 0, acc);
        return 0;
    }
    if (mode == "fingerprint") {
        const uint64_t nodes = uint64_t(a.num("nodes", 20000));
        ab::Searcher<Board> s(64);
        s.cfg = cfg;
        uint64_t h = 1469598103934665603ull; size_t n = 0; double nodeSum = 0;
        for (const Record& r : recs) {
            if (!searchLabelled(r)) continue;
            if (n >= limit) break;
            Board b; b.setModel(&mdl); r.toBoard(b);
            s.tt.clear(); s.clearHeuristics();
            const ab::Result res = s.think(b, ab::Limits{ab::MAX_PLY - 1, -1, nodes});
            for (uint64_t v : {uint64_t(res.bestMove + 1), uint64_t(int64_t(res.score)), res.nodes, uint64_t(res.depth)}) { h ^= v; h *= 1099511628211ull; }
            nodeSum += double(res.nodes); ++n;
        }
        std::printf("[fingerprint] positions=%zu nodes=%llu hash=%016llx avg-nodes=%.0f\n", n, (unsigned long long)nodes,
                    (unsigned long long)h, n ? nodeSum / n : 0);
        return 0;
    }
    if (mode == "leafdump") {
        const uint64_t nodes = uint64_t(a.num("nodes", 20000));
        RecordWriter w(a.str("out", "leaves.bin"));
        std::mt19937_64 rng(uint64_t(a.num("seed", 5)));
        ab::Searcher<DumpBoard> s(32);
        s.cfg = cfg;
        DumpBoard b; b.setModel(&mdl); b.rng = &rng; b.out = &w; b.prob = float(a.num("prob", 0.002));
        size_t n = 0;
        for (const Record& r : recs) {
            if (n++ >= limit) break;
            r.toBoard(b);
            s.think(b, ab::Limits{ab::MAX_PLY - 1, -1, nodes});
        }
        std::printf("[leafdump] roots=%zu leaves=%llu\n", n, (unsigned long long)b.dumped);
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
