// features.hpp — 고전 평가 특징 정의 (엔진·학습기·평가도구가 같은 코드를 공유 → train/serve skew 없음)
//
// 모든 특징은 "둘 차례(self)" 관점. 흑/백 색 정보는 BIAS 한 칸으로만 들어간다.
//
// [가치 특징]  v = Σ w_i  (로짓 단위, 가중치 2벌을 진행도 t 로 보간 — 초반/후반)
//   LS    라인 모양: 칸 상태(빈/자기/상대) × 8칸 라인 키(좌우 반전 동치)    — 국소 모양의 원재료
//   CS/CO 빈칸의 자기/상대 4방향 위협 단계 조합 코드(495)                 — 방향 간 결합
//   J     빈칸의 (자기 칸 등급 × 상대 칸 등급)                           — 공수 겸용 급소
//   INIT  (자기 최고 등급 × 상대 최고 등급)                               — 전역 주도권
//   CNT   칸 등급별 개수 버킷 (1, 2, 3+)                                — 위협의 양
//   PSQ   돌 위치(D4 대칭 36궤도) × 소유                                — 위치 가치
//   BIAS  둘 차례 색,  PH 진행 버킷(16수 단위)
//
// [정책 특징]  p(c) = Σ w_i  (후보 칸마다, NNUE 정책을 softmax 로 모사 → 수 정렬/가지치기용, 보조)
//   PLS 라인 키, PCS/PCO 조합 코드, PJ (전역 전술 등급 25 × 칸 등급 결합), PD1/PD2 직전 수 거리, PPSQ 위치
#pragma once
#include "board.hpp"
#include <algorithm>

namespace gmk::feat {

// ── 라인 키 정규화: 벽이 바깥쪽부터 연속인 키만 유효, 좌우 반전 동치류로 묶는다 ──
struct LineCanon {
    int id[KEY_NB];
    int count = 0;
    static const LineCanon& get() { static const LineCanon c; return c; }
private:
    static bool validHalf(int k, bool left) {
        bool seenNonWall = false;
        for (int j = 0; j < 4; ++j) {
            const int s = left ? j : 7 - j;     // 바깥 → 안쪽
            const int v = (k >> (2 * s)) & 3;
            if (v == 3) { if (seenNonWall) return false; }
            else seenNonWall = true;
        }
        return true;
    }
    LineCanon() {
        const auto& T = LineTables::get();
        std::fill(id, id + KEY_NB, -1);
        for (int k = 0; k < KEY_NB; ++k) {
            if (!validHalf(k, true) || !validHalf(k, false)) continue;
            const int c = std::min<int>(k, T.revKey[k]);
            if (id[c] < 0) id[c] = count++;
            id[k] = id[c];
        }
    }
};

inline int orbitOf(int c) {       // 15x15 D4 대칭 궤도 (36개)
    int x = cellX(c), y = cellY(c);
    x = std::min(x, BS - 1 - x); y = std::min(y, BS - 1 - y);
    const int a = std::min(x, y), b = std::max(x, y);
    return b * (b + 1) / 2 + a;
}
constexpr int NORBIT = 36;

inline int distBucket(int c, int m) {
    if (m < 0) return 4;
    const int q = toCell(m);
    const int d = std::max(std::abs(cellX(c) - cellX(q)), std::abs(cellY(c) - cellY(q)));
    return d <= 4 ? std::max(d, 1) - 1 : 4;
}

struct Layout {
    int NC;
    int LS, CS, CO, J, INIT, CNT, PSQ, BIAS, PH, NV;          // value
    int PLS, PCS, PCO, PJ, PD1, PD2, PPSQ, NP;                // policy
    static const Layout& get() { static const Layout l; return l; }
private:
    Layout() {
        NC = LineCanon::get().count;
        int o = 0;
        LS = o;   o += 3 * NC;
        CS = o;   o += CODE_NB;
        CO = o;   o += CODE_NB;
        J = o;    o += CLASS_NB * CLASS_NB;
        INIT = o; o += CLASS_NB * CLASS_NB;
        CNT = o;  o += 2 * CLASS_NB * 3;
        PSQ = o;  o += 2 * NORBIT;
        BIAS = o; o += 2;
        PH = o;   o += 15;
        NV = o;
        o = 0;
        PLS = o;  o += NC;
        PCS = o;  o += CODE_NB;
        PCO = o;  o += CODE_NB;
        PJ = o;   o += 25 * CLASS_NB * CLASS_NB;
        PD1 = o;  o += 5;
        PD2 = o;  o += 5;
        PPSQ = o; o += NORBIT;
        NP = o;
    }
};

inline const int* boardCells() {
    static const int* v = [] { static int a[NMOVES]; for (int m = 0; m < NMOVES; ++m) a[m] = toCell(m); return a; }();
    return v;
}

inline uint16_t relKey(const Board& b, int d, int c) {
    const uint16_t k = b.lkey[d][c];
    return b.side == BLACK ? k : LineTables::get().swapKey[k];
}

template <class F>
inline void valueFeatures(const Board& b, F&& emit) {
    const auto& L = Layout::get();
    const auto& C = LineCanon::get();
    const int self = b.side, s = self - 1, o = 2 - self;
    const int* cells = boardCells();
    for (int m = 0; m < NMOVES; ++m) {
        const int c = cells[m];
        const int col = b.cell[c];
        const int cs = col == EMPTY ? 0 : col == self ? 1 : 2;
        bool near = false;
        for (int d = 0; d < 4; ++d) {
            const bool hs = keyHasStone(b.lkey[d][c]);
            near |= hs;
            if (cs == 0 && !hs) continue;
            emit(L.LS + cs * L.NC + C.id[relKey(b, d, c)]);
        }
        if (cs == 0) {
            if (!near) continue;
            emit(L.CS + b.code[s][c]);
            emit(L.CO + b.code[o][c]);
            const int a = b.cls[s][c], bb = b.cls[o][c];
            if (a | bb) emit(L.J + a * CLASS_NB + bb);
        } else {
            emit(L.PSQ + (cs - 1) * NORBIT + orbitOf(c));
        }
    }
    emit(L.INIT + b.bestClass(self) * CLASS_NB + b.bestClass(opp(self)));
    for (int t = 0; t < 2; ++t) {
        const int* cnt = b.clsCnt[t == 0 ? s : o];
        for (int k = C_TWO; k <= C_FIVE; ++k)
            if (cnt[k]) emit(L.CNT + (t * CLASS_NB + k) * 3 + std::min(cnt[k], 3) - 1);
    }
    emit(L.BIAS + (self == WHITE));
    emit(L.PH + std::min(b.ply / 16, 14));
}

// 정책용 전역 전술 등급: (자기 최고 등급 tier) × (상대 최고 등급 tier) = 25
inline int threatClass(const Board& b) {
    return classTier(b.bestClass(b.side)) * 5 + classTier(b.bestClass(opp(b.side)));
}

template <class F>
inline void policyFeatures(const Board& b, int c, int tclass, F&& emit) {
    const auto& L = Layout::get();
    const auto& C = LineCanon::get();
    const int s = b.side - 1, o = 2 - b.side;
    for (int d = 0; d < 4; ++d)
        if (keyHasStone(b.lkey[d][c])) emit(L.PLS + C.id[relKey(b, d, c)]);
    emit(L.PCS + b.code[s][c]);
    emit(L.PCO + b.code[o][c]);
    emit(L.PJ + (tclass * CLASS_NB + b.cls[s][c]) * CLASS_NB + b.cls[o][c]);
    emit(L.PD1 + distBucket(c, b.lastMove(0)));
    emit(L.PD2 + distBucket(c, b.lastMove(1)));
    emit(L.PPSQ + orbitOf(c));
}

constexpr int TAPER_PLY = 120;
inline float taper(const Board& b) { return std::min(b.ply, TAPER_PLY) / float(TAPER_PLY); }

} // namespace gmk::feat
