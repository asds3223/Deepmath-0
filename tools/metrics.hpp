// metrics.hpp — 증류 품질 지표 (학습기·평가도구 공용)
#pragma once
#include "../engine/eval.hpp"
#include "../engine/record.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

namespace gmk::metrics {

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float logitClip(float p) { p = std::min(0.999f, std::max(0.001f, p)); return std::log(p / (1 - p)); }

// 후보 칸 집합 위의 NNUE 정책 목표 분포 (상위 16 개를 후보로 제한·재정규화)
inline int policyTarget(const Board& b, const Record& r, int* moves, float* tgt, int maxN = NMOVES) {
    int n = 0;
    for (int m = 0; m < NMOVES && n < maxN; ++m) if (b.isCandidate(toCell(m))) moves[n++] = m;
    float tot = 0;
    for (int i = 0; i < n; ++i) tgt[i] = 0;
    for (int k = 0; k < r.polN; ++k) {
        for (int i = 0; i < n; ++i) if (moves[i] == r.polIdx[k]) { tgt[i] += r.polProb[k] / 65535.f; break; }
    }
    for (int i = 0; i < n; ++i) tot += tgt[i];
    if (tot <= 1e-6f) return 0;
    for (int i = 0; i < n; ++i) tgt[i] /= tot;
    return n;
}

struct Stats {
    double n = 0, bce = 0, mae = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sign = 0;
    double base_n = 0, base_mae = 0, base_sign = 0;       // Rapfi 고전평가 베이스라인
    double pn = 0, ptop1 = 0, ptop3 = 0, pce = 0;
    double srch_n = 0, srch_mae = 0;

    void addValue(float logit, const Record& r) {
        const float p = r.nnWinrate(), q = sigmoid(logit);
        bce += -(p * std::log(std::max(q, 1e-7f)) + (1 - p) * std::log(std::max(1 - q, 1e-7f)));
        mae += std::fabs(q - p);
        const float x = std::max(-8.f, std::min(8.f, logit)), y = logitClip(p);
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
        sign += ((q - 0.5f) * (p - 0.5f) >= 0);
        ++n;
        base_mae += std::fabs(r.rcls - p); base_sign += ((r.rcls - 0.5f) * (p - 0.5f) >= 0); ++base_n;
        if ((r.flags & REC_HAS_SEARCH) && !std::isnan(r.srchWR)) { srch_mae += std::fabs(q - r.srchWR); ++srch_n; }
    }
    void addPolicy(const float* score, const float* tgt, int n) {
        int ib = 0, it = 0;
        for (int i = 1; i < n; ++i) { if (score[i] > score[ib]) ib = i; if (tgt[i] > tgt[it]) it = i; }
        ptop1 += (ib == it);
        int rank = 0; for (int i = 0; i < n; ++i) if (score[i] > score[it]) ++rank;
        ptop3 += (rank < 3);
        float mx = score[0]; for (int i = 1; i < n; ++i) mx = std::max(mx, score[i]);
        double z = 0; for (int i = 0; i < n; ++i) z += std::exp(score[i] - mx);
        for (int i = 0; i < n; ++i) if (tgt[i] > 0) pce += -tgt[i] * (score[i] - mx - std::log(z));
        ++pn;
    }
    double corr() const {
        double vx = sxx / n - (sx / n) * (sx / n), vy = syy / n - (sy / n) * (sy / n);
        return (sxy / n - (sx / n) * (sy / n)) / std::sqrt(std::max(vx * vy, 1e-12));
    }
    void print(const char* tag) const {
        std::printf("%s value: n=%.0f bce=%.4f winrate-MAE=%.4f logit-corr=%.4f sign-agree=%.3f | "
                    "RapfiClassical MAE=%.4f sign=%.3f",
                    tag, n, bce / n, mae / n, corr(), sign / n, base_mae / std::max(base_n, 1.0), base_sign / std::max(base_n, 1.0));
        if (srch_n) std::printf(" | vs-search MAE=%.4f (n=%.0f)", srch_mae / srch_n, srch_n);
        std::printf("\n");
        if (pn) std::printf("%s policy: n=%.0f top1=%.3f nnTop1-in-our-top3=%.3f ce=%.4f\n", tag, pn, ptop1 / pn, ptop3 / pn, pce / pn);
        std::fflush(stdout);
    }
};

inline void evalSet(const Model& mdl, const std::vector<Record>& recs, Stats& st) {
    Board b; b.model = &mdl;
    int moves[NMOVES]; float tgt[NMOVES], sc[NMOVES];
    for (const Record& r : recs) {
        if (r.flags & REC_SELF_FIVE) continue;
        r.toBoard(b);
        st.addValue(mdl.valueLogit(b), r);
        if (r.flags & REC_OPPO_FIVE) continue;
        int n = policyTarget(b, r, moves, tgt);
        if (n < 2) continue;
        const int tc = feat::threatClass(b);
        for (int i = 0; i < n; ++i) sc[i] = mdl.policyScore(b, toCell(moves[i]), tc);
        st.addPolicy(sc, tgt, n);
    }
}

} // namespace gmk::metrics
