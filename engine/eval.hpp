// eval.hpp — 증류된 고전 평가 모델 (가치 + 정책) 및 Board::evaluate / Board::generate
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
    static constexpr uint32_t MAGIC = 0x44434537;   // "DCE7" (정책 2차 결합 표 추가)
    std::vector<float> wv;      // 가치: [2*i] 초반, [2*i+1] 후반
    std::vector<float> wp;      // 정책
    float scale    = 200.f;     // 탐색 정수 점수 = 로짓 × scale
    int   maxCands = 0;         // >0 이면 정책 상위 N 개로 후보 제한 (기본 OFF)

    template <class F> void forValue(const Board& b, F&& f) const { feat::valueFeatures(b, f); }
    template <class F> void forPolicy(const Board& b, int c, int tc, F&& f) const { feat::policyFeatures(b, c, tc, f); }

    // 탐색용 정수 가중치 (증분 누적의 정확한 복원을 위해 정수화)
    std::vector<int32_t> qv;    // 가치: 점수(로짓×scale) × 16
    std::vector<int32_t> qp;    // 정책: 로짓 × 256

    Model() { const auto& L = feat::Layout::get(); wv.assign(2 * size_t(L.NV), 0.f); wp.assign(L.NP, 0.f); quantize(); }
    void quantize() {
        qv.resize(wv.size()); qp.resize(wp.size());
        for (size_t i = 0; i < wv.size(); ++i) qv[i] = int32_t(std::lround(wv[i] * scale * 16));
        for (size_t i = 0; i < wp.size(); ++i) qp[i] = int32_t(std::lround(wp[i] * 256));
    }

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
    // 메모리에 담긴 모델(실행 파일 내장용) 로드 — 파일 형식과 같다
    bool loadMem(const uint8_t* d, size_t n) {
        const auto& L = feat::Layout::get();
        const size_t need = 12 + 4 + 4 * (wv.size() + wp.size());
        if (n < need) return false;
        uint32_t h[3]; std::memcpy(h, d, 12);
        if (h[0] != MAGIC || h[1] != uint32_t(L.NV) || h[2] != uint32_t(L.NP)) return false;
        std::memcpy(&scale, d + 12, 4);
        std::memcpy(wv.data(), d + 16, 4 * wv.size());
        std::memcpy(wp.data(), d + 16 + 4 * wv.size(), 4 * wp.size());
        quantize();
        return true;
    }
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        const auto& L = feat::Layout::get();
        uint32_t h[3];
        // DCE5(정책 확장 표 이전) 도 읽는다: 정책 새 표는 뒤에 붙었으므로 앞부분만 채우고 나머지는 0
        bool ok = std::fread(h, sizeof h, 1, f) == 1 && (h[0] == MAGIC || h[0] == 0x44434535u || h[0] == 0x44434536u) &&
                  h[1] == uint32_t(L.NV) && h[2] <= uint32_t(L.NP) && (h[0] != MAGIC || h[2] == uint32_t(L.NP)) &&
                  std::fread(&scale, sizeof scale, 1, f) == 1 &&
                  std::fread(wv.data(), sizeof(float), wv.size(), f) == wv.size() &&
                  std::fread(wp.data(), sizeof(float), h[2], f) == h[2];
        if (ok) std::fill(wp.begin() + h[2], wp.end(), 0.f);
        std::fclose(f);
        if (ok) quantize();
        return ok;
    }
};

// 모델이 없을 때의 단순 휴리스틱 (테스트/부트스트랩용)
inline int heuristicClassScore(int c) {
    //                         NONE TWO O2 D2 T3 T3+ O3 O3+ D3 F4 F4+ F43 WIN4 FIVE
    static const int v[CLASS_NB] = {0, 1, 2, 6, 3, 5, 12, 20, 40, 15, 30, 60, 200, 1000};
    return v[c];
}

inline const int* orbitTable() {
    static const int* t = [] { static int a[CELLS]; for (int c = 0; c < CELLS; ++c) a[c] = feat::orbitOf(c); return a; }();
    return t;
}

// 칸 c 의 가치 기여 (두 관점 동시) — feat::valueFeatures 의 칸 단위 항과 정확히 같은 항
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
            if (feat::g_useShape && keyHasStone(sh[c])) {
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
        // 칸 단위 항은 증분 누적값, 전역 항(주도권·개수·차례·진행)만 여기서 더한다
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

// 전술 수 판정 등급 (이 등급 이상을 만들거나 막는 수는 감축·컷·예산 대상에서 제외) — 튜닝용
inline int g_tacSelf = C_DOUBLE3;
inline int g_tacOppo = C_DOUBLE3;

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
    if (nd) {
        for (int i = 0; i < nd; ++i) { out[i] = defs[i]; tactical[i] = 1; }
        n = nd;
    } else {
        for (int w = 0; w < 4; ++w)
            for (uint64_t bits = candBits[w]; bits; bits &= bits - 1) {
                const int m = w * 64 + __builtin_ctzll(bits);
                const int c = cells[m];
                out[n] = m;
                tactical[n] = cls[s][c] >= g_tacSelf || cls[o][c] >= g_tacOppo;
                ++n;
            }
    }
    if (model) {
        const auto& L = feat::Layout::get();
        const int32_t* q = model->qp.data();
        const int* pj = q + L.PJ + tc * CLASS_NB * CLASS_NB;
        const int l1 = lastMove(0), l2 = lastMove(1);
        const int* orb = orbitTable();
        for (int i = 0; i < n; ++i) {
            const int c = cells[out[i]];
            const int shp = feat::g_useShape ? q[L.PSH + feat::ShapeCanon::get().id[side == BLACK ? sh[c] : LineTables::get().swapKey[sh[c]]]] : 0;
            int ext = 0;
            if (feat::g_policyExt) {
                const auto& LC = feat::LineCanon::get(); const auto& T = LineTables::get();
                for (int d = 0; d < 4; ++d) {
                    const uint16_t k = lkey[d][c];
                    if (keyHasStone(k)) ext += q[L.PLS + LC.id[side == BLACK ? k : T.swapKey[k]]];
                }
                const int r1 = feat::relOffset(c, l1);
                ext += q[L.PR1 + r1] + q[L.PR2 + feat::relOffset(c, l2)] + q[L.PRS + r1 * CLASS_NB + cls[s][c]] + q[L.PRO + r1 * CLASS_NB + cls[o][c]];
            }
            if (feat::g_policyExt2) {
                ext += q[L.PTC + tc * CODE_NB + code[s][c]] + q[L.PTO + tc * CODE_NB + code[o][c]];
                int d, k;
                feat::lineRel(c, l1, d, k);
                if (d >= 0) ext += q[L.PL1 + ((k - 1) * LEVEL_NB + lv[s][d][c]) * LEVEL_NB + lv[o][d][c]];
                feat::lineRel(c, l2, d, k);
                if (d >= 0) ext += q[L.PL2 + ((k - 1) * LEVEL_NB + lv[s][d][c]) * LEVEL_NB + lv[o][d][c]];
            }
            score[i] = ext + shp + q[L.PCS + code[s][c]] + q[L.PCO + code[o][c]] + q[L.PPSQ + orb[c]] + pj[cls[s][c] * CLASS_NB + cls[o][c]] + q[L.PD1 + feat::distBucket(c, l1)] + q[L.PD2 + feat::distBucket(c, l2)];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            const int c = cells[out[i]];
            score[i] = (heuristicClassScore(cls[s][c]) * 2 + heuristicClassScore(cls[o][c])) * 13;
        }
    }
    return n;
}

// 위협 상태의 방어수: 실제로 두어 보고 상대의 4-3/승리 4 급소가 모두 사라지는 수 + 우리 4(맞받기)
inline int Board::defenseMoves(int* out) const {
    // 우리 돌 하나가 상대의 4-3/승리 4 급소 t 를 없애려면 t 를 직접 차지하거나 t 를 지나는 줄 위(±4)에 놓여
    // 그 줄의 단계를 떨어뜨려야 한다. 줄 키 하나만 고쳐 t 의 등급을 다시 찾아보면 실제로 둬 보는 것과 같은 판정.
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
            if (t == c) continue;                                   // 급소를 직접 차지
            const int dx = cellX(c) - cellX(t), dy = cellY(c) - cellY(t);
            int d = -1, k = 0;
            if (dy == 0) { d = 0; k = dx; } else if (dx == 0) { d = 1; k = dy; }
            else if (dx == dy) { d = 2; k = dx; } else if (dx == -dy) { d = 3; k = dy; }
            if (d < 0 || k < -4 || k > 4) return false;             // 이 급소의 줄과 무관 → 위협이 남는다
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
        if (cls[s][c] >= C_FOUR || neutralizes(c)) out[n++] = m;   // 4 맞받기 또는 모든 급소 제거
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

// ── VCF 탐색기 ──
// 우리: 4 를 만드는 수만 둔다. 상대: 우리 5목 자리를 막는 것 외엔 선택이 없다.
// 상대가 막으면서 자기 4(5목 위협)를 만들면 이 단순 VCF 는 그 갈래를 실패로 본다.
// 비용 관리: 호출당 노드 한도 + "VCF 없음"으로 확인된 국면을 기억하는 실패 해시.
struct VcfCache {
    static constexpr int BITS = 18;
    uint64_t keys[1 << BITS] = {};
    static VcfCache& get() { static thread_local VcfCache c; return c; }
    bool knownFail(uint64_t k) const { return keys[k & ((1 << BITS) - 1)] == k; }
    void markFail(uint64_t k) { keys[k & ((1 << BITS) - 1)] = k; }
};

namespace detail {
inline bool vcfRec(Board& b, int left, int off, int& dist, int& budget) {
    const int s = b.side - 1, o = 2 - b.side;
    if (b.clsCnt[s][C_FIVE]) { dist = off + 1; return true; }
    if (b.clsCnt[o][C_FIVE]) return false;
    if (b.clsCnt[s][C_WINFOUR]) { dist = off + 3; return true; }   // 열린 4/쌍사: 막을 수 없다
    if (left < 2) return false;
    const int* S = b.clsCnt[s];
    if (!(S[C_FOUR] + S[C_FOUR_PLUS] + S[C_FOUR3])) return false;
    const uint64_t key = b.key() ^ (uint64_t(left) * 0x9E3779B97F4A7C15ull);
    auto& cache = VcfCache::get();
    if (cache.knownFail(key)) return false;
    const int* cells = feat::boardCells();
    int mv[64], n = 0;
    for (int w = 0; w < 4 && n < 64; ++w)
        for (uint64_t bits = b.candBits[w]; bits && n < 64; bits &= bits - 1) {
            const int m = w * 64 + __builtin_ctzll(bits);
            const int k = b.cls[s][cells[m]];
            if (k == C_FOUR3) mv[n++] = m | 0x10000;               // 4-3 을 먼저
            else if (k == C_FOUR || k == C_FOUR_PLUS) mv[n++] = m;
        }
    std::sort(mv, mv + n, [](int x, int y) { return x > y; });
    for (int i = 0; i < n; ++i) {
        if (--budget < 0) return false;                         // 한도 초과: 판정 보류 (실패로 기억하지 않음)
        const int m = mv[i] & 0xffff, c = toCell(m);
        b.make(m);
        int blk = -1;                                           // 상대의 유일한 응수: 우리 5목 자리
        for (int d = 0; d < 4 && blk < 0; ++d)
            for (int k = -4; k <= 4; ++k) {
                const int q = c + k * DIRS[d];
                if (k && b.cell[q] == EMPTY && b.cls[s][q] == C_FIVE) { blk = q; break; }
            }
        bool ok = false;
        if (blk >= 0 && !b.clsCnt[o][C_FIVE]) {
            const int bm = toMove(blk);
            b.make(bm);
            ok = vcfRec(b, left - 2, off + 2, dist, budget);
            b.undo(bm);
        }
        b.undo(m);
        if (ok) return true;
    }
    if (budget >= 0) cache.markFail(key);
    return false;
}
} // namespace detail

inline int g_vcfBudget = 200;   // VCF 1회 호출당 착수 한도 (튜닝용)
inline int g_vcfMaxPly = 20;

inline bool Board::vcfWin(int maxPly, int& dist) const {
    Board& b = const_cast<Board&>(*this);
    const bool l0 = b.light; b.light = true;                     // 패턴만 갱신
    int budget = g_vcfBudget;
    const bool r = detail::vcfRec(b, maxPly, 0, dist, budget);
    b.light = l0;
    return r;
}

inline bool Board::vcfProbe(int plyFromRoot, int& score) const {
    int dist;
    if (oppFive() || !vcfWin(g_vcfMaxPly, dist)) return false;
    score = ab::MATE - (plyFromRoot + dist);
    return true;
}

// ── VCT 탐색기 (위협 공간 AND/OR 탐색, 루트 전용) ──
// 공격: 4 또는 열린 3 을 만드는 수만.  방어: 막기 / 위협 제거(defenseMoves) / 맞받기 4.
// 상대가 4 로 맞받으면 공격자는 막아야 하고(강제), 그 뒤 공격을 이어간다.
// 반환: 공격 측 필승이면 true (bestMove = 첫 수). 한도 초과·불명은 false.
struct VctCache {
    static constexpr int BITS = 18;
    uint64_t keys[1 << BITS] = {};
    static VctCache& get() { static thread_local VctCache c; return c; }
    bool knownFail(uint64_t k) const { return keys[k & ((1 << BITS) - 1)] == k; }
    void markFail(uint64_t k) { keys[k & ((1 << BITS) - 1)] = k; }
};

namespace detail {
inline int fiveCellNear(const Board& b, int c, int colIdx) {      // c 의 4방향 ±4 안에서 colIdx 의 5목 자리
    for (int d = 0; d < 4; ++d)
        for (int k = -4; k <= 4; ++k) {
            const int q = c + k * DIRS[d];
            if (k && b.cell[q] == EMPTY && b.cls[colIdx][q] == C_FIVE) return q;
        }
    return -1;
}
inline int anyFiveCell(const Board& b, int colIdx) {
    const int* cells = feat::boardCells();
    for (int m = 0; m < NMOVES; ++m) if (b.cell[cells[m]] == EMPTY && b.cls[colIdx][cells[m]] == C_FIVE) return cells[m];
    return -1;
}

bool vctAttack(Board& b, int depth, int& budget, int* firstMove);

// 방어 측 차례: 모든 방어수에 대해 공격이 계속 성공해야 true
inline bool vctDefend(Board& b, int depth, int& budget) {
    const int s = b.side - 1, o = 2 - b.side;                 // s = 방어자, o = 공격자
    if (b.clsCnt[s][C_FIVE]) return false;                    // 방어자가 바로 5목
    if (b.clsCnt[o][C_FIVE] >= 2) return true;                // 막을 수 없는 쌍사
    int mv[NMOVES], n = 0;
    if (b.clsCnt[o][C_FIVE]) {                                // 공격자 4: 막기만
        const int q = anyFiveCell(b, o);
        if (q < 0) return false;
        mv[n++] = toMove(q);
    } else if (b.oppThreat()) {                               // 공격자 열린 3 류: 위협 제거 + 맞받기 4
        n = b.defenseMoves(mv);
        if (n == 0) return true;                              // 막을 길이 없다
    } else return false;                                      // 위협이 없으면 VCT 실패
    for (int i = 0; i < n; ++i) {
        if (--budget < 0) return false;
        b.make(mv[i]);
        bool ok;
        if (b.isLoss()) ok = false;
        else if (b.clsCnt[2 - b.side][C_FIVE]) {               // 방어자가 맞받기 4 → 공격자는 막아야 함
            const int q = anyFiveCell(b, 2 - b.side);
            ok = false;
            if (q >= 0 && !b.clsCnt[b.side - 1][C_FIVE]) {
                const int qm = toMove(q);
                b.make(qm);
                ok = vctDefend(b, depth, budget);              // 막은 뒤에도 공격자의 위협이 남아 있어야 한다
                b.undo(qm);
            } else if (b.clsCnt[b.side - 1][C_FIVE]) ok = true;   // 공격자가 먼저 5목
        } else ok = vctAttack(b, depth - 1, budget, nullptr);
        b.undo(mv[i]);
        if (!ok) return false;
    }
    return true;
}

// 공격 측 차례
inline bool vctAttack(Board& b, int depth, int& budget, int* firstMove) {
    const int s = b.side - 1, o = 2 - b.side;
    if (b.clsCnt[s][C_FIVE]) { if (firstMove) *firstMove = toMove(anyFiveCell(b, s)); return true; }
    if (b.clsCnt[o][C_FIVE]) return false;                    // 상대 4 를 먼저 막아야 함 → 이 단순 VCT 는 포기
    if (depth <= 0) return false;
    const uint64_t key = b.key() ^ (uint64_t(depth) * 0x9E3779B97F4A7C15ull);
    auto& cache = VctCache::get();
    if (!firstMove && cache.knownFail(key)) return false;
    const int* cells = feat::boardCells();
    int mv[96], n = 0;
    for (int k = C_WINFOUR; k >= C_OPEN3 && n < 96; --k) {
        if (!b.clsCnt[s][k] || k == C_THREE || k == C_THREE_PLUS) continue;
        for (int w = 0; w < 4 && n < 96; ++w)
            for (uint64_t bits = b.candBits[w]; bits && n < 96; bits &= bits - 1) {
                const int m = w * 64 + __builtin_ctzll(bits);
                if (b.cls[s][cells[m]] == k) mv[n++] = m;
            }
    }
    for (int i = 0; i < n; ++i) {
        if (--budget < 0) return false;
        b.make(mv[i]);
        const bool ok = !b.isLoss() && vctDefend(b, depth, budget);
        b.undo(mv[i]);
        if (ok) { if (firstMove) *firstMove = mv[i]; return true; }
    }
    if (budget >= 0) cache.markFail(key);
    return false;
}
} // namespace detail

// 루트 VCT: 둘 차례가 위협 연속으로 이기는 첫 수 (없거나 한도 초과면 -1)
inline int Board::rootVct(int maxAttack, int budget) const {
    Board& b = const_cast<Board&>(*this);
    const bool l0 = b.light; b.light = true;
    int first = -1, bud = budget;
    for (int d = 1; d <= maxAttack && bud > 0; ++d) {          // 짧은 승리부터 반복 심화
        int f = -1;
        if (detail::vctAttack(b, d, bud, &f)) { first = f; break; }
    }
    b.light = l0;
    return first;
}

} // namespace gmk
