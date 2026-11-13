// eval.hpp — 고전 평가 모델(가치 + 정책)과 Board 의 평가·수 생성
//
//   Model           점수표(가치 2벌 테이퍼, 정책) — 학습은 float, 탐색은 정수(증분 누적이 정확히 복원되도록)
//   Board::evaluate 칸 단위 항은 증분 누적값(vacc), 전역 항만 그 자리에서 더한다
//   Board::generateScored  후보 + 정책 점수 + 전술 표시 (탐색용, 정렬 없음)
//   Board::generate        위를 정책 점수 순으로 정렬한 것
#pragma once
#include "features.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gmk {

class Model {
public:
    static constexpr uint32_t MAGIC = 0x44434537;   // "DCE7"
    std::vector<float>   wv;    // 가치: [2*i] 초반, [2*i+1] 후반
    std::vector<float>   wp;    // 정책
    std::vector<int32_t> qv;    // 가치 정수: 점수(로짓×scale) × 16
    std::vector<int32_t> qp;    // 정책 정수: 로짓 × 256
    float scale = 200.f;        // 탐색 점수 = 로짓 × scale

    Model() { const auto& L = feat::Layout::get(); wv.assign(2 * size_t(L.NV), 0.f); wp.assign(L.NP, 0.f); quantize(); }

    void quantize() {
        qv.resize(wv.size()); qp.resize(wp.size());
        for (size_t i = 0; i < wv.size(); ++i) qv[i] = int32_t(std::lround(wv[i] * scale * 16));
        for (size_t i = 0; i < wp.size(); ++i) qp[i] = int32_t(std::lround(wp[i] * 256));
    }

    // 전체 재계산 (학습·지표용). 탐색은 Board 의 증분 경로를 쓴다.
    template <class F> void forValue(const Board& b, F&& f) const { feat::valueFeatures(b, f); }
    template <class F> void forPolicy(const Board& b, int c, int tc, F&& f) const { feat::policyFeatures(b, c, tc, f); }
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

    // 파일 형식: [MAGIC, NV, NP] uint32 + scale float + wv + wp
    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const auto& L = feat::Layout::get();
        const uint32_t h[3] = {MAGIC, uint32_t(L.NV), uint32_t(L.NP)};
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
        std::vector<uint8_t> d;
        uint8_t buf[1 << 16];
        for (size_t k; (k = std::fread(buf, 1, sizeof buf, f)) > 0;) d.insert(d.end(), buf, buf + k);
        std::fclose(f);
        return loadMem(d.data(), d.size());
    }
    bool loadMem(const uint8_t* d, size_t n) {          // 실행 파일 내장 모델도 같은 형식
        const auto& L = feat::Layout::get();
        if (n != 16 + 4 * (wv.size() + wp.size())) return false;
        uint32_t h[3]; std::memcpy(h, d, 12);
        if (h[0] != MAGIC || h[1] != uint32_t(L.NV) || h[2] != uint32_t(L.NP)) return false;
        std::memcpy(&scale, d + 12, 4);
        std::memcpy(wv.data(), d + 16, 4 * wv.size());
        std::memcpy(wp.data(), d + 16 + 4 * wv.size(), 4 * wp.size());
        quantize();
        return true;
    }
};

// 모델이 없을 때의 단순 휴리스틱 (테스트·오라클의 후보 생성용)
inline int heuristicClassScore(int c) {
    //                              NONE TWO O2 D2 T3 T3+ O3 O3+ D3  F4  F4+ F43 WIN4  FIVE
    static const int v[CLASS_NB] = {0,   1,  2, 6, 3, 5,  12, 20, 40, 15, 30, 60, 200, 1000};
    return v[c];
}

// 전술 수: 이 등급 이상을 만들거나 막는 수 — 감축·컷·예산 대상에서 제외
constexpr int TACTICAL_CLASS = C_DOUBLE3;

inline const int* orbitTable() {
    static const int* t = [] { static int a[CELLS]; for (int c = 0; c < CELLS; ++c) a[c] = feat::orbitOf(c); return a; }();
    return t;
}

// ── 증분 가치 ──
// 칸 c 의 가치 기여 (두 관점 동시) — feat::valueFeatures 의 칸 단위 항과 같은 항
inline void Board::accCell(int c, int sign) {
    int32_t* cc = vcache[c];
    if (sign < 0) {                                       // 캐시된 기여를 뺀다
        vacc[0][0] -= cc[0]; vacc[0][1] -= cc[1]; vacc[1][0] -= cc[2]; vacc[1][1] -= cc[3];
        return;
    }
    const auto& L = feat::Layout::get();
    const int32_t* q = model->qv.data();
    const int col = cell[c];
    int32_t r[4] = {0, 0, 0, 0};
    if (col == EMPTY) {
        bool near = false;
        for (int d = 0; d < 4; ++d) near |= keyHasStone(lkey[d][c]);
        if (near) {
            const int32_t* cs0 = q + 2 * (L.CS + code[0][c]); const int32_t* cs1 = q + 2 * (L.CS + code[1][c]);
            const int32_t* co0 = q + 2 * (L.CO + code[1][c]); const int32_t* co1 = q + 2 * (L.CO + code[0][c]);
            r[0] = cs0[0] + co0[0]; r[1] = cs0[1] + co0[1];
            r[2] = cs1[0] + co1[0]; r[3] = cs1[1] + co1[1];
            if (keyHasStone(sh[c])) {
                const auto& SC = feat::ShapeCanon::get();
                const int32_t* s0 = q + 2 * (L.SH + SC.id[sh[c]]);
                const int32_t* s1 = q + 2 * (L.SH + SC.id[LineTables::get().swapKey[sh[c]]]);
                r[0] += s0[0]; r[1] += s0[1]; r[2] += s1[0]; r[3] += s1[1];
            }
            const int a = cls[0][c], bb = cls[1][c];
            if (a | bb) {
                const int32_t* j0 = q + 2 * (L.J + a * CLASS_NB + bb); const int32_t* j1 = q + 2 * (L.J + bb * CLASS_NB + a);
                r[0] += j0[0]; r[1] += j0[1]; r[2] += j1[0]; r[3] += j1[1];
            }
        }
    } else {
        const int orb = orbitTable()[c];
        const int32_t* pb = q + 2 * (L.PSQ + (col == BLACK ? 0 : 1) * feat::NORBIT + orb);   // 흑 관점
        const int32_t* pw = q + 2 * (L.PSQ + (col == WHITE ? 0 : 1) * feat::NORBIT + orb);   // 백 관점
        r[0] = pb[0]; r[1] = pb[1]; r[2] = pw[0]; r[3] = pw[1];
    }
    cc[0] = r[0]; cc[1] = r[1]; cc[2] = r[2]; cc[3] = r[3];
    vacc[0][0] += r[0]; vacc[0][1] += r[1]; vacc[1][0] += r[2]; vacc[1][1] += r[3];
}

inline void Board::resyncModel() {
    vacc[0][0] = vacc[0][1] = vacc[1][0] = vacc[1][1] = 0;
    std::memset(vcache, 0, sizeof(vcache));
    if (!model) return;
    const int* cells = feat::boardCells();
    for (int m = 0; m < NMOVES; ++m) accCell(cells[m], +1);
}

inline void Board::setModel(const Model* m) { model = m; resyncModel(); }

inline int Board::evaluate() const {
    if (model) {
        const auto& L = feat::Layout::get();
        const int32_t* q = model->qv.data();
        const int P = side - 1, s = P, o = 1 - P;
        int32_t s0 = vacc[P][0], s1 = vacc[P][1];
        auto add = [&](int i) { s0 += q[2 * i]; s1 += q[2 * i + 1]; };
        add(L.INIT + bestClass(side) * CLASS_NB + bestClass(opp(side)));
        for (int t = 0; t < 2; ++t) {
            const int* cnt = clsCnt[t == 0 ? s : o];
            for (int k = C_TWO; k <= C_FIVE; ++k)
                if (cnt[k]) add(L.CNT + (t * CLASS_NB + k) * 3 + std::min(cnt[k], 3) - 1);
        }
        add(L.BIAS + (side == WHITE));
        add(L.PH + std::min(ply / 16, 14));
        const float t = feat::taper(*this);
        float v = (float(s0) * (1 - t) + float(s1) * t) / 16.f;
        const float lim = float(ab::MATE_BOUND - 1);
        v = std::max(-lim, std::min(lim, v));
        return int(std::lround(v));
    }
    const int s = side - 1, o = 2 - side;
    int e = 0;
    for (int k = C_TWO; k <= C_WINFOUR; ++k) e += heuristicClassScore(k) * (clsCnt[s][k] * 3 - clsCnt[o][k] * 2);
    return std::max(-2000, std::min(2000, e));
}

// ── 수 생성 ──
// 즉승 → 그 한 수.  상대 4 → 막는 수만.  상대 4-3/승리 4 위협 → 방어수(+맞받기 4)만.  그 외 → 후보 칸 전체.
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
    int defs[NMOVES], nd = 0;
    if (oppThreat()) nd = defenseMoves(defs);
    if (nd) {
        for (int i = 0; i < nd; ++i) { out[i] = defs[i]; tactical[i] = 1; }
        n = nd;
    } else {
        for (int w = 0; w < 4; ++w)
            for (uint64_t bits = candBits[w]; bits; bits &= bits - 1) {
                const int m = w * 64 + __builtin_ctzll(bits);
                const int c = cells[m];
                out[n] = m;
                tactical[n] = cls[s][c] >= TACTICAL_CLASS || cls[o][c] >= TACTICAL_CLASS;
                ++n;
            }
    }
    if (!model) {
        for (int i = 0; i < n; ++i) {
            const int c = cells[out[i]];
            score[i] = (heuristicClassScore(cls[s][c]) * 2 + heuristicClassScore(cls[o][c])) * 13;
        }
        return n;
    }
    // 정책 점수 — feat::policyFeatures 와 같은 항
    const auto& L = feat::Layout::get();
    const auto& T = LineTables::get();
    const auto& LC = feat::LineCanon::get();
    const auto& SC = feat::ShapeCanon::get();
    const int32_t* q = model->qp.data();
    const int tc = feat::threatClass(*this);
    const int32_t* pj = q + L.PJ + tc * CLASS_NB * CLASS_NB;
    const int l1 = lastMove(0), l2 = lastMove(1);
    const int* orb = orbitTable();
    for (int i = 0; i < n; ++i) {
        const int c = cells[out[i]];
        int v = q[L.PCS + code[s][c]] + q[L.PCO + code[o][c]] + q[L.PPSQ + orb[c]]
              + q[L.PSH + SC.id[side == BLACK ? sh[c] : T.swapKey[sh[c]]]]
              + pj[cls[s][c] * CLASS_NB + cls[o][c]]
              + q[L.PD1 + feat::distBucket(c, l1)] + q[L.PD2 + feat::distBucket(c, l2)];
        for (int d = 0; d < 4; ++d) {
            const uint16_t k = lkey[d][c];
            if (keyHasStone(k)) v += q[L.PLS + LC.id[side == BLACK ? k : T.swapKey[k]]];
        }
        const int r1 = feat::relOffset(c, l1);
        v += q[L.PR1 + r1] + q[L.PR2 + feat::relOffset(c, l2)] + q[L.PRS + r1 * CLASS_NB + cls[s][c]] + q[L.PRO + r1 * CLASS_NB + cls[o][c]];
        v += q[L.PTC + tc * CODE_NB + code[s][c]] + q[L.PTO + tc * CODE_NB + code[o][c]];
        int d, k;
        feat::lineRel(c, l1, d, k);
        if (d >= 0) v += q[L.PL1 + ((k - 1) * LEVEL_NB + lv[s][d][c]) * LEVEL_NB + lv[o][d][c]];
        feat::lineRel(c, l2, d, k);
        if (d >= 0) v += q[L.PL2 + ((k - 1) * LEVEL_NB + lv[s][d][c]) * LEVEL_NB + lv[o][d][c]];
        score[i] = v;
    }
    return n;
}

inline int Board::generate(int* out) const {
    int sc[NMOVES]; uint8_t tac[NMOVES];
    const int n = generateScored(out, sc, tac);
    for (int i = 1; i < n; ++i) {                       // 점수 내림차순 (안정 삽입 정렬)
        const int x = sc[i], mv = out[i];
        int j = i - 1;
        while (j >= 0 && sc[j] < x) { sc[j + 1] = sc[j]; out[j + 1] = out[j]; --j; }
        sc[j + 1] = x; out[j + 1] = mv;
    }
    return n;
}

// 위협 상태의 방어수: 상대의 4-3/승리 4 급소를 모두 없애는 수 + 우리 4(맞받기).
// 우리 돌 하나가 급소 t 를 없애려면 t 를 차지하거나 t 를 지나는 줄(±4) 위에 놓여 그 줄의 단계를 떨어뜨려야 한다.
// 줄 키 하나만 고쳐 t 의 등급을 다시 찾아보면 실제로 둬 보는 것과 같은 판정이 된다.
inline int Board::defenseMoves(int* out) const {
    const auto& T = LineTables::get();
    const int s = side - 1, o = 2 - side, ocol = o + 1;
    const int* cells = feat::boardCells();
    int thr[NMOVES], nt = 0;
    for (int m = 0; m < NMOVES; ++m) {
        const int t = cells[m];
        if (cell[t] == EMPTY && (cls[o][t] == C_FOUR3 || cls[o][t] == C_WINFOUR)) thr[nt++] = t;
    }
    auto neutralizes = [&](int c) {
        for (int i = 0; i < nt; ++i) {
            const int t = thr[i];
            if (t == c) continue;
            const int dx = cellX(c) - cellX(t), dy = cellY(c) - cellY(t);
            int d = -1, k = 0;
            if (dy == 0) { d = 0; k = dx; } else if (dx == 0) { d = 1; k = dy; }
            else if (dx == dy) { d = 2; k = dx; } else if (dx == -dy) { d = 3; k = dy; }
            if (d < 0 || k < -4 || k > 4) return false;
            uint16_t key = uint16_t(lkey[d][t] | (side << (2 * slotOf(k))));
            if (ocol != BLACK) key = T.swapKey[key];
            uint8_t lv4[4] = {lv[o][0][t], lv[o][1][t], lv[o][2][t], lv[o][3][t]};
            lv4[d] = T.level[key];
            const int cl = T.classOf[T.code[lv4[0]][lv4[1]][lv4[2]][lv4[3]]];
            if (cl == C_FOUR3 || cl == C_WINFOUR) return false;
        }
        return true;
    };
    int n = 0;
    for (int m = 0; m < NMOVES; ++m) {
        const int c = cells[m];
        if (cell[c] != EMPTY) continue;
        if (cls[s][c] >= C_FOUR || neutralizes(c)) out[n++] = m;
    }
    return n;
}

// 정지 탐색 공격수: 4 를 만드는 수 (승리 4 → 4-3 → 4+ → 4 순)
inline int Board::forcingMoves(int* out) const {
    const int s = side - 1;
    const int* S = clsCnt[s];
    if (!(S[C_FOUR] + S[C_FOUR_PLUS] + S[C_FOUR3] + S[C_WINFOUR])) return 0;
    const int* cells = feat::boardCells();
    int n = 0;
    for (int k = C_WINFOUR; k >= C_FOUR; --k) {
        if (!S[k]) continue;
        for (int m = 0; m < NMOVES; ++m) if (cell[cells[m]] == EMPTY && cls[s][cells[m]] == k) out[n++] = m;
    }
    return n;
}

} // namespace gmk
