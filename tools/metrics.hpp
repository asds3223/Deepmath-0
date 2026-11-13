// metrics.hpp — 판단력 지표 (학습기·평가도구 공용)
//
// 주 지표는 "정적 평가가 모든 국면의 유불리를 NNUE 처럼 판별하는가" 이다.
//   1) 가치 정확도   : 승률 MAE, 로짓 상관, 우열 부호 일치 — 국면 유형/진행도/형세 구간별로 나눠 본다
//   2) 형제 순위     : 같은 부모의 자식 국면들 사이의 우열 순서를 NNUE 와 같게 매기는가 (쌍 일치율),
//                      그 순서로 고른 수가 NNUE 기준으로 얼마나 손해인가 (regret)
//   3) Rapfi 내장 고전평가를 같은 지표로 재서 하한 기준선으로 함께 표시
// 정책 지표(top1/top3)는 수 정렬 품질 확인용 보조 지표다.
#pragma once
#include "../engine/eval.hpp"
#include "../engine/record.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace gmk::metrics {

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float logitClip(float p) { p = std::min(0.999f, std::max(0.001f, p)); return std::log(p / (1 - p)); }

inline int policyTarget(const Board& b, const Record& r, int* moves, float* tgt) {
    int n = 0;
    for (int m = 0; m < NMOVES; ++m) if (b.isCandidate(toCell(m))) moves[n++] = m;
    for (int i = 0; i < n; ++i) tgt[i] = 0;
    for (int k = 0; k < r.polN; ++k)
        for (int i = 0; i < n; ++i) if (moves[i] == r.polIdx[k]) { tgt[i] += r.polProb[k] / 65535.f; break; }
    float tot = 0;
    for (int i = 0; i < n; ++i) tot += tgt[i];
    if (tot <= 1e-6f) return 0;
    for (int i = 0; i < n; ++i) tgt[i] /= tot;
    return n;
}

// 한 집단에 대한 가치 지표 (우리 모델과 Rapfi 고전평가를 같은 방식으로)
struct ValueStats {
    struct Acc {
        double n = 0, bce = 0, mae = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sign = 0, signN = 0;
        void add(float logit, float p) {
            const float q = sigmoid(logit);
            bce += -(p * std::log(std::max(q, 1e-7f)) + (1 - p) * std::log(std::max(1 - q, 1e-7f)));
            mae += std::fabs(q - p);
            const float x = std::max(-7.f, std::min(7.f, logit)), y = logitClip(p);
            sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
            if (std::fabs(p - 0.5f) >= 0.05f) { sign += ((q - 0.5f) * (p - 0.5f) > 0); ++signN; }   // 거의 균형은 제외
            ++n;
        }
        double corr() const {
            const double vx = sxx / n - (sx / n) * (sx / n), vy = syy / n - (sy / n) * (sy / n);
            return (sxy / n - (sx / n) * (sy / n)) / std::sqrt(std::max(vx * vy, 1e-12));
        }
    } ours, base;
    void add(float logit, const Record& r) {
        const float p = r.nnWinrate();
        ours.add(logit, p);
        base.add(logitClip(r.rcls), p);
    }
};

struct RankStats {
    double pairs = 0, agree = 0, baseAgree = 0;          // 쌍 순서 일치
    double groups = 0, bestAgree = 0, regret = 0, baseRegret = 0;
    // 한 형제 묶음: (우리 로짓, Rapfi 고전 승률, NNUE 승률) — 모두 "자식의 둘 차례" 관점
    void addGroup(const std::vector<float>& ours, const std::vector<float>& base, const std::vector<float>& nn) {
        const size_t k = nn.size();
        if (k < 2) return;
        for (size_t i = 0; i < k; ++i) for (size_t j = i + 1; j < k; ++j) {
            if (std::fabs(nn[i] - nn[j]) < 0.03f) continue;           // NNUE 도 구분 못 하는 쌍은 제외
            const bool truth = nn[i] < nn[j];
            agree += ((ours[i] < ours[j]) == truth);
            baseAgree += ((base[i] < base[j]) == truth);
            ++pairs;
        }
        // 부모 입장의 최선 = 자식(상대 차례) 승률이 가장 낮은 것
        auto argmin = [](const std::vector<float>& v) { size_t b = 0; for (size_t i = 1; i < v.size(); ++i) if (v[i] < v[b]) b = i; return b; };
        const size_t bn = argmin(nn), bo = argmin(ours), bb = argmin(base);
        bestAgree += (bo == bn);
        regret += nn[bo] - nn[bn];
        baseRegret += nn[bb] - nn[bn];
        ++groups;
    }
};

struct PolicyStats {
    double n = 0, top1 = 0, top3 = 0, ce = 0;
    void add(const float* score, const float* tgt, int n_) {
        int ib = 0, it = 0;
        for (int i = 1; i < n_; ++i) { if (score[i] > score[ib]) ib = i; if (tgt[i] > tgt[it]) it = i; }
        top1 += (ib == it);
        int rank = 0; for (int i = 0; i < n_; ++i) if (score[i] > score[it]) ++rank;
        top3 += (rank < 3);
        float mx = score[0]; for (int i = 1; i < n_; ++i) mx = std::max(mx, score[i]);
        double z = 0; for (int i = 0; i < n_; ++i) z += std::exp(score[i] - mx);
        for (int i = 0; i < n_; ++i) if (tgt[i] > 0) ce += -tgt[i] * (score[i] - mx - std::log(z));
        ++n;
    }
};

struct Report {
    enum { ALL, GAME, SIBLING, LEAF, EARLY, MID, LATE, BALANCED, LEANING, DECIDED, NSTRATA };
    static const char* name(int i) {
        static const char* n[] = {"all", "game", "sibling", "leaf", "ply<20", "ply20-59", "ply>=60",
                                  "nn 0.35-0.65", "nn 0.15-0.85", "nn decided"};
        return n[i];
    }
    ValueStats v[NSTRATA];
    RankStats rank;
    PolicyStats pol;

    void addValue(float logit, const Record& r) {
        v[ALL].add(logit, r);
        v[(r.flags & REC_LEAF) ? LEAF : (r.flags & REC_SIBLING) ? SIBLING : GAME].add(logit, r);
        v[r.ply < 20 ? EARLY : r.ply < 60 ? MID : LATE].add(logit, r);
        const float d = std::fabs(r.nnWinrate() - 0.5f);
        v[d < 0.15f ? BALANCED : d < 0.35f ? LEANING : DECIDED].add(logit, r);
    }

    void print(const char* tag, bool full) const {
        auto line = [&](int i) {
            const auto& o = v[i].ours; const auto& b = v[i].base;
            if (o.n < 1) return;
            std::printf("%s  %-13s n=%-7.0f MAE=%.4f corr=%.4f sign=%.3f bce=%.4f | Rapfi고전 MAE=%.4f corr=%.4f sign=%.3f\n",
                        tag, name(i), o.n, o.mae / o.n, o.corr(), o.sign / std::max(o.signN, 1.0), o.bce / o.n,
                        b.mae / b.n, b.corr(), b.sign / std::max(b.signN, 1.0));
        };
        line(ALL);
        if (full) for (int i = 1; i < NSTRATA; ++i) line(i);
        if (rank.groups)
            std::printf("%s  sibling-rank  groups=%.0f pairs=%.0f pair-agree=%.3f best-agree=%.3f regret=%.4f "
                        "| Rapfi고전 pair-agree=%.3f regret=%.4f\n",
                        tag, rank.groups, rank.pairs, rank.agree / std::max(rank.pairs, 1.0), rank.bestAgree / rank.groups,
                        rank.regret / rank.groups, rank.baseAgree / std::max(rank.pairs, 1.0), rank.baseRegret / rank.groups);
        if (pol.n)
            std::printf("%s  policy(보조)  n=%.0f top1=%.3f nnTop1-in-top3=%.3f ce=%.4f\n", tag, pol.n, pol.top1 / pol.n,
                        pol.top3 / pol.n, pol.ce / pol.n);
        std::fflush(stdout);
    }
};

inline void evalSet(const Model& mdl, const std::vector<Record>& recs, Report& rep, bool policy = true) {
    Board b; b.model = &mdl;
    int moves[NMOVES]; float tgt[NMOVES], sc[NMOVES];
    std::vector<float> go, gb, gn; uint32_t curGroup = 0;
    auto flush = [&] { rep.rank.addGroup(go, gb, gn); go.clear(); gb.clear(); gn.clear(); };
    for (const Record& r : recs) {
        if (r.flags & REC_SELF_FIVE) continue;
        r.toBoard(b);
        const float logit = mdl.valueLogit(b);
        rep.addValue(logit, r);
        const uint32_t g = (r.flags & REC_SIBLING) ? r.group() : 0;
        if (g != curGroup) { flush(); curGroup = g; }
        if (g) { go.push_back(logit); gb.push_back(r.rcls); gn.push_back(r.nnWinrate()); }
        if (!policy || (r.flags & REC_OPPO_FIVE)) continue;
        const int n = policyTarget(b, r, moves, tgt);
        if (n < 2) continue;
        const int tc = feat::threatClass(b);
        for (int i = 0; i < n; ++i) sc[i] = mdl.policyScore(b, toCell(moves[i]), tc);
        rep.pol.add(sc, tgt, n);
    }
    flush();
}

} // namespace gmk::metrics
