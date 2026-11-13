// 증분 평가 검증: 무작위 make/undo 수순에서
//   (1) evaluate() (증분 정수 누적) == 전체 재계산 valueLogit×scale (반올림 오차 이내)
//   (2) generateScored 점수 == 전체 재계산 policyScore×256 (반올림 오차 이내), 후보 집합 == isCandidate
//   (3) undo 후 누적값이 처음과 정확히 같다
// 사용: test_incremental [model.dce]
#include "eval.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <set>

using namespace gmk;

int main(int argc, char** argv) {
    Model mdl;
    const char* path = argc > 1 ? argv[1] : "data/b.dce";
    if (!mdl.load(path)) {      // 모델이 없거나 형식이 다르면 무작위 가중치로 검증 (모든 표 칸을 건드림)
        std::mt19937 wr(11); std::uniform_real_distribution<float> U(-0.3f, 0.3f);
        for (auto& w : mdl.wv) w = U(wr);
        for (auto& w : mdl.wp) w = U(wr);
        mdl.quantize();
        std::printf("(random weights)\n");
    }
    std::mt19937 rng(3);
    long checks = 0, valBad = 0, polBad = 0, candBad = 0, restoreBad = 0; int maxVal = 0, maxPol = 0;
    for (int g = 0; g < 200; ++g) {
        Board b; b.setModel(&mdl);
        int v0[2][2]; std::memcpy(v0, b.vacc, sizeof v0);
        std::vector<int> seq;
        for (int ply = 0; ply < 120; ++ply) {
            int mv[NMOVES], n = b.generate(mv);
            if (!n || b.isLoss() || b.isDraw()) break;
            const int m = mv[rng() % std::min(n, 8)];
            b.make(m); seq.push_back(m);
            if (b.isLoss()) break;
            // (1) 가치
            const int inc = b.evaluate();
            const int full = int(std::lround(std::max(-29871.f, std::min(29871.f, mdl.valueLogit(b) * mdl.scale))));
            maxVal = std::max(maxVal, std::abs(inc - full)); valBad += std::abs(inc - full) > 2;
            // (2) 정책 / 후보
            int out[NMOVES], sc[NMOVES]; uint8_t tac[NMOVES];
            const int k = b.generateScored(out, sc, tac);
            if (!b.oppFive() && !b.oppThreat() && !b.clsCnt[b.side - 1][C_FIVE]) {
                std::set<int> a(out, out + k), e;
                for (int q = 0; q < NMOVES; ++q) if (b.isCandidate(toCell(q))) e.insert(q);
                candBad += a != e;
                const int tc = feat::threatClass(b);
                for (int i = 0; i < k; ++i) {
                    const int f = int(std::lround(mdl.policyScore(b, toCell(out[i]), tc) * 256));
                    maxPol = std::max(maxPol, std::abs(sc[i] - f)); polBad += std::abs(sc[i] - f) > 4;
                }
            }
            ++checks;
        }
        // (3) 전부 되돌리면 처음 상태
        while (!seq.empty()) { b.undo(seq.back()); seq.pop_back(); }
        restoreBad += std::memcmp(v0, b.vacc, sizeof v0) != 0;
    }
    std::printf("checks=%ld value-mismatch=%ld (max %d) policy-mismatch=%ld (max %d) cand-mismatch=%ld restore-mismatch=%ld\n",
                checks, valBad, maxVal, polBad, maxPol, candBad, restoreBad);
    // 속도
    Board b; b.setModel(&mdl);
    for (int m : {112, 113, 97, 127, 98, 96, 128, 82, 111, 126, 99, 100}) b.make(m);
    int out[NMOVES], sc[NMOVES]; uint8_t tac[NMOVES]; long long acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; ++i) acc += b.evaluate();
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 300000; ++i) acc += b.generateScored(out, sc, tac);
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 300000; ++i) { b.make(41); b.undo(41); }
    auto t3 = std::chrono::steady_clock::now();
    std::printf("evaluate=%.3fus generateScored=%.3fus make+undo=%.3fus (acc=%lld)\n",
                std::chrono::duration<double, std::micro>(t1 - t0).count() / 1e6,
                std::chrono::duration<double, std::micro>(t2 - t1).count() / 3e5,
                std::chrono::duration<double, std::micro>(t3 - t2).count() / 3e5, acc);
    const bool ok = !valBad && !polBad && !candBad && !restoreBad;
    std::printf(ok ? "ALL OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
