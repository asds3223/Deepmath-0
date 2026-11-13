// eval.hpp — 증류된 고전 평가 모델 (가치 + 정책) 및 Board::evaluate / Board::generate
#pragma once
#include "features.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace gmk {

class Model {
public:
    static constexpr uint32_t MAGIC = 0x44434531;   // "DCE1"
    std::vector<float> wv;      // 가치: [2*i] 초반, [2*i+1] 후반
    std::vector<float> wp;      // 정책
    float scale    = 200.f;     // 탐색 정수 점수 = 로짓 × scale
    int   maxCands = 0;         // >0 이면 정책 상위 N 개로 후보 제한 (기본 OFF)

    Model() { const auto& L = feat::Layout::get(); wv.assign(2 * size_t(L.NV), 0.f); wp.assign(L.NP, 0.f); }

    float valueLogit(const Board& b) const {
        float s0 = 0, s1 = 0;
        const float* w = wv.data();
        feat::valueFeatures(b, [&](int i) { s0 += w[2 * i]; s1 += w[2 * i + 1]; });
        const float t = feat::taper(b);
        return s0 * (1 - t) + s1 * t;
    }
    float policyScore(const Board& b, int c, int tclass) const {
        float s = 0;
        const float* w = wp.data();
        feat::policyFeatures(b, c, tclass, [&](int i) { s += w[i]; });
        return s;
    }

    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const auto& L = feat::Layout::get();
        uint32_t h[4] = {MAGIC, uint32_t(L.NV), uint32_t(L.NP), uint32_t(L.NC)};
        std::fwrite(h, sizeof h, 1, f);
        std::fwrite(&scale, sizeof scale, 1, f);
        std::fwrite(wv.data(), sizeof(float), wv.size(), f);
        std::fwrite(wp.data(), sizeof(float), wp.size(), f);
        std::fclose(f);
        return true;
    }
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        const auto& L = feat::Layout::get();
        uint32_t h[4];
        bool ok = std::fread(h, sizeof h, 1, f) == 1 && h[0] == MAGIC && h[1] == uint32_t(L.NV) &&
                  h[2] == uint32_t(L.NP) && h[3] == uint32_t(L.NC) &&
                  std::fread(&scale, sizeof scale, 1, f) == 1 &&
                  std::fread(wv.data(), sizeof(float), wv.size(), f) == wv.size() &&
                  std::fread(wp.data(), sizeof(float), wp.size(), f) == wp.size();
        std::fclose(f);
        return ok;
    }
};

// 모델이 없을 때의 단순 휴리스틱 (테스트/부트스트랩용)
inline int heuristicP4Score(int p4) {
    static const int v[PATTERN4_NB] = {0, 0, 2, 1, 6, 5, 12, 20, 40, 15, 30, 60, 200, 1000};
    return v[p4];
}

inline int Board::evaluate() const {
    if (model) {
        float v = model->valueLogit(*this) * model->scale;
        const float lim = float(ab::MATE_BOUND - 1);
        v = std::max(-lim, std::min(lim, v));
        return int(std::lround(v));
    }
    const int s = side - 1, o = 2 - side;
    int e = 0;
    for (int k = L_FLEX2; k <= B_FLEX4; ++k) e += heuristicP4Score(k) * (p4cnt[s][k] * 3 - p4cnt[o][k] * 2);
    return std::max(-2000, std::min(2000, e));
}

inline int Board::generate(int* out) const {
    if (ply == 0) { out[0] = (BS / 2) * BS + BS / 2; return 1; }
    const int s = side - 1, o = 2 - side;
    const int* cells = feat::boardCells();
    if (p4cnt[s][A_FIVE]) {                                   // 즉승: 한 수면 충분
        for (int m = 0; m < NMOVES; ++m) if (cell[cells[m]] == EMPTY && p4[s][cells[m]] == A_FIVE) { out[0] = m; return 1; }
    }
    int n = 0;
    if (p4cnt[o][A_FIVE]) {                                   // 상대 4: 막는 수만 합법적으로 의미 있음
        for (int m = 0; m < NMOVES; ++m) if (cell[cells[m]] == EMPTY && p4[o][cells[m]] == A_FIVE) out[n++] = m;
        return n;
    }
    float sc[NMOVES];
    const int tc = model ? feat::threatClass(*this) : 0;
    for (int m = 0; m < NMOVES; ++m) {
        const int c = cells[m];
        if (!isCandidate(c)) continue;
        sc[n] = model ? model->policyScore(*this, c, tc)
                      : float(heuristicP4Score(p4[s][c]) * 2 + heuristicP4Score(p4[o][c]));
        out[n++] = m;
    }
    // 점수 내림차순 정렬 (삽입 정렬, n ≤ ~120)
    for (int i = 1; i < n; ++i) {
        float x = sc[i]; int mv = out[i], j = i - 1;
        while (j >= 0 && sc[j] < x) { sc[j + 1] = sc[j]; out[j + 1] = out[j]; --j; }
        sc[j + 1] = x; out[j + 1] = mv;
    }
    if (model && model->maxCands > 0 && n > model->maxCands) n = model->maxCands;
    return n;
}

} // namespace gmk
