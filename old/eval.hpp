// eval.hpp — 증류된 고전 평가 모델 (가치 + 정책) 및 Board::evaluate / Board::generate
#pragma once
#include "features.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace gmk {

class Model {
public:
    static constexpr uint32_t MAGIC = 0x44434534;   // "DCE4"
    std::vector<float> wv;      // 가치: [2*i] 초반, [2*i+1] 후반
    std::vector<float> wp;      // 정책
    float scale    = 200.f;     // 탐색 정수 점수 = 로짓 × scale
    int   maxCands = 0;         // >0 이면 정책 상위 N 개로 후보 제한 (기본 OFF)

    template <class F> void forValue(const Board& b, F&& f) const { feat::valueFeatures(b, f); }
    template <class F> void forPolicy(const Board& b, int c, int tc, F&& f) const { feat::policyFeatures(b, c, tc, f); }

    Model() { const auto& L = feat::Layout::get(); wv.assign(2 * size_t(L.NV), 0.f); wp.assign(L.NP, 0.f); }

    float valueLogit(const Board& b) const {
        float s0 = 0, s1 = 0;
        const float* w = wv.data();
        forValue(b, [&](int i) { s0 += w[2 * i]; s1 += w[2 * i + 1]; });
        const float t = feat::taper(b);
        return s0 * (1 - t) + s1 * t;
    }
    float policyScore(const Board& b, int c, int tclass) const {
        float s = 0;
        const float* w = wp.data();
        forPolicy(b, c, tclass, [&](int i) { s += w[i]; });
        return s;
    }

    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const auto& L = feat::Layout::get();
        uint32_t h[3] = {MAGIC, uint32_t(L.NV), uint32_t(L.NP)};
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
        uint32_t h[3];
        bool ok = std::fread(h, sizeof h, 1, f) == 1 && h[0] == MAGIC && h[1] == uint32_t(L.NV) && h[2] == uint32_t(L.NP) &&
                  std::fread(&scale, sizeof scale, 1, f) == 1 &&
                  std::fread(wv.data(), sizeof(float), wv.size(), f) == wv.size() &&
                  std::fread(wp.data(), sizeof(float), wp.size(), f) == wp.size();
        std::fclose(f);
        return ok;
    }
};

// 모델이 없을 때의 단순 휴리스틱 (테스트/부트스트랩용)
inline int heuristicClassScore(int c) {
    //                         NONE TWO O2 D2 T3 T3+ O3 O3+ D3 F4 F4+ F43 WIN4 FIVE
    static const int v[CLASS_NB] = {0, 1, 2, 6, 3, 5, 12, 20, 40, 15, 30, 60, 200, 1000};
    return v[c];
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
    for (int k = C_TWO; k <= C_WINFOUR; ++k) e += heuristicClassScore(k) * (clsCnt[s][k] * 3 - clsCnt[o][k] * 2);
    return std::max(-2000, std::min(2000, e));
}

inline int Board::generate(int* out) const {
    if (ply == 0) { out[0] = (BS / 2) * BS + BS / 2; return 1; }
    const int s = side - 1, o = 2 - side;
    const int* cells = feat::boardCells();
    if (clsCnt[s][C_FIVE]) {                                  // 즉승: 한 수면 충분
        for (int m = 0; m < NMOVES; ++m) if (cell[cells[m]] == EMPTY && cls[s][cells[m]] == C_FIVE) { out[0] = m; return 1; }
    }
    int n = 0;
    if (clsCnt[o][C_FIVE]) {                                  // 상대 4: 막는 수만 의미 있음
        for (int m = 0; m < NMOVES; ++m) if (cell[cells[m]] == EMPTY && cls[o][cells[m]] == C_FIVE) out[n++] = m;
        return n;
    }
    float sc[NMOVES];
    const int tc = model ? feat::threatClass(*this) : 0;
    if (oppThreat()) {                                   // 위협 상태: 방어수만 (정렬 생략)
        const int nd = defenseMoves(out);
        if (nd > 0) return nd;
    }
    const int nd = 0; const int* defs = out; const bool onlyDef = false;
    for (int m = 0; m < NMOVES; ++m) {
        const int c = cells[m];
        if (!isCandidate(c)) continue;
        if (onlyDef && std::find(defs, defs + nd, m) == defs + nd) continue;
        sc[n] = model ? model->policyScore(*this, c, tc)
                      : float(heuristicClassScore(cls[s][c]) * 2 + heuristicClassScore(cls[o][c]));
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

// 탐색용: 정렬하지 않은 후보 + 정책 점수(로짓×256) + 전술 수 표시
//   전술 수 = 자기가 쌍삼 이상(4 포함)을 만들거나 상대의 쌍삼 이상 급소를 막는 수 → 감축·컷 대상에서 제외
inline int Board::generateScored(int* out, int* score, uint8_t* tactical) const {
    if (ply == 0) { out[0] = (BS / 2) * BS + BS / 2; score[0] = 0; tactical[0] = 1; return 1; }
    const int s = side - 1, o = 2 - side;
    const int* cells = feat::boardCells();
    if (clsCnt[s][C_FIVE]) {
        for (int m = 0; m < NMOVES; ++m) if (cell[cells[m]] == EMPTY && cls[s][cells[m]] == C_FIVE) { out[0] = m; score[0] = 0; tactical[0] = 1; return 1; }
    }
    int n = 0;
    if (clsCnt[o][C_FIVE]) {
        for (int m = 0; m < NMOVES; ++m)
            if (cell[cells[m]] == EMPTY && cls[o][cells[m]] == C_FIVE) { out[n] = m; score[n] = 0; tactical[n] = 1; ++n; }
        return n;
    }
    const int tc = model ? feat::threatClass(*this) : 0;
    int defs[NMOVES], nd = 0;
    if (oppThreat()) nd = defenseMoves(defs);           // 위협 상태: 방어수/맞받기 4 만
    for (int m = 0; m < NMOVES; ++m) {
        const int c = cells[m];
        if (!isCandidate(c)) continue;
        if (nd && std::find(defs, defs + nd, m) == defs + nd) continue;
        const float sc = model ? model->policyScore(*this, c, tc)
                               : float(heuristicClassScore(cls[s][c]) * 2 + heuristicClassScore(cls[o][c])) * 0.05f;
        out[n] = m;
        score[n] = int(sc * 256);
        tactical[n] = nd > 0 || cls[s][c] >= C_DOUBLE3 || cls[o][c] >= C_DOUBLE3;
        ++n;
    }
    return n;
}

// 위협 상태의 방어수: 실제로 두어 보고 상대의 4-3/승리 4 급소가 모두 사라지는 수 + 우리 4(맞받기)
inline int Board::defenseMoves(int* out) const {
    Board& b = const_cast<Board&>(*this);                // make/undo 는 상태를 정확히 복원한다
    const int s = side - 1, o = 2 - side;
    const int* cells = feat::boardCells();
    // 시험 후보: 상대 위협 칸(4-3/승리 4)을 지나는 4방향 줄 ±4 칸 (+ 우리 4 맞받기는 별도)
    bool mark[CELLS] = {};
    for (int m = 0; m < NMOVES; ++m) {
        const int t = cells[m];
        if (cell[t] != EMPTY || (cls[o][t] != C_FOUR3 && cls[o][t] != C_WINFOUR)) continue;
        mark[t] = true;
        for (int d = 0; d < 4; ++d)
            for (int k = -4; k <= 4; ++k) { const int q = t + k * DIRS[d]; if (cell[q] == EMPTY) mark[q] = true; }
    }
    int n = 0;
    for (int m = 0; m < NMOVES; ++m) {
        const int c = cells[m];
        if (cell[c] != EMPTY) continue;
        if (cls[s][c] >= C_FOUR) { out[n++] = m; continue; }       // 4 로 맞받기 (상대는 막아야 함)
        if (!mark[c]) continue;
        b.make(m);
        const bool ok = b.clsCnt[o][C_FOUR3] + b.clsCnt[o][C_WINFOUR] == 0;
        b.undo(m);
        if (ok) out[n++] = m;
    }
    return n;
}

// 정지 탐색 공격수: 4 를 만드는 수 (승리 4 → 4-3 → 4+ → 4), allowThree 면 열린 3 을 만드는 수도
inline int Board::forcingMoves(int* out, bool allowThree) const {
    const int s = side - 1;
    const int* S = clsCnt[s];
    const int lo = allowThree ? C_OPEN3 : C_FOUR;
    int any = 0; for (int k = lo; k <= C_WINFOUR; ++k) any += S[k];
    if (!any) return 0;
    const int* cells = feat::boardCells();
    int n = 0;
    for (int k = C_WINFOUR; k >= lo; --k) {
        if (!S[k] || k == C_THREE || k == C_THREE_PLUS) continue;
        for (int m = 0; m < NMOVES; ++m) if (cell[cells[m]] == EMPTY && cls[s][cells[m]] == k) out[n++] = m;
    }
    return n;
}

} // namespace gmk
