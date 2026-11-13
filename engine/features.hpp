// features.hpp — 고전 평가 특징 정의 (엔진·학습기·평가도구가 같은 코드를 공유 → train/serve skew 없음)
//
// 모든 특징은 "둘 차례(self)" 관점. 흑/백 색 정보는 BIAS 한 칸으로만 들어간다
// (mix9svq freestyle 가중치도 흑백 공용이므로 같은 대칭을 갖는다).
//
// [가치 특징]  v = Σ w_i · x_i  (로짓 단위, 테이퍼 2벌: 초반/후반 가중치를 진행도 t 로 보간)
//   LS   라인 모양: 칸(빈/자기/상대) × 방향별 8칸 정규화 라인 키   ← NNUE 1층(라인 shape→codebook)의 대응물
//   PCS  빈칸의 자기 4방향 패턴 조합(pcode)                         ← 칸 단위 4방향 합성(비선형)의 대응물
//   PCO  빈칸의 상대 pcode
//   J    빈칸의 (자기 Pattern4 × 상대 Pattern4) 결합
//   TM   위협 마스크 11bit (Rapfi 정의)                             ← 전역 MLP 가 잡는 "누가 공격권을 쥐었나"
//   CNT  Pattern4 종류별 개수 버킷 (1, 2, 3+)
//   PSQ  돌 위치(D4 대칭 36궤도) × 소유                              ← NNUE 의 corner/edge/center 그룹 풀링 대응
//   BIAS 둘 차례 색,  PH 진행 버킷(16수 단위)
//
// [정책 특징]  p(c) = Σ w_i · x_i  (후보 칸 c 마다, softmax 로 NNUE 정책을 모사 → 수 정렬/가지치기)
//   PLS  c 의 방향별 라인 키,  PPS/PPO  c 의 자기/상대 pcode,
//   PJ   (전역 위협 클래스 32) × (c 의 자기 P4 × 상대 P4),
//   PD1/PD2  직전 상대수/직전 자기수까지의 체비셰프 거리,  PPSQ  c 의 위치 궤도
#pragma once
#include "board.hpp"
#include <algorithm>

namespace gmk::feat {

// ── 라인 키 정규화: 벽 연속성이 맞는 키만 남기고, 좌우 반전 동치류로 묶는다 ──
struct LineCanon {
    int id[KEY_NB];
    int count = 0;
    static const LineCanon& get() { static const LineCanon c; return c; }
private:
    static bool validHalf(int k, bool left) {
        // left: 슬롯 0(가장 먼 -4)..3(가장 가까운 -1).  벽은 바깥쪽부터 연속이어야 함
        bool seenNonWall = false;
        for (int j = 0; j < 4; ++j) {
            int s = left ? j : 7 - j;                 // 바깥 → 안쪽 순서
            int v = (k >> (2 * s)) & 3;
            if (v == 3) { if (seenNonWall) return false; }
            else seenNonWall = true;
        }
        return true;
    }
    LineCanon() {
        const auto& T = PatternTables::get();
        std::fill(id, id + KEY_NB, -1);
        for (int k = 0; k < KEY_NB; ++k) {
            if (!validHalf(k, true) || !validHalf(k, false)) continue;
            int c = std::min<int>(k, T.revKey[k]);
            if (id[c] < 0) id[c] = count++;
            id[k] = id[c];
        }
    }
};

// 15x15 D4 대칭 궤도 (36개)
inline int orbitOf(int c) {
    int x = cellX(c), y = cellY(c);
    x = std::min(x, BS - 1 - x); y = std::min(y, BS - 1 - y);
    int a = std::min(x, y), b = std::max(x, y);
    return b * (b + 1) / 2 + a;
}
constexpr int NORBIT = 36;

inline int distBucket(int c, int m) {
    if (m < 0) return 4;
    int q = toCell(m);
    int dx = std::abs(cellX(c) - cellX(q)), dy = std::abs(cellY(c) - cellY(q));
    int d = std::max(dx, dy);
    return d <= 4 ? std::max(d, 1) - 1 : 4;
}

// ── 레이아웃 ──
struct Layout {
    int NC;
    // value
    int LS, PCS, PCO, J, TM, CNT, PSQ, BIAS, PH, NV;
    // policy
    int PLS, PPS, PPO, PJ, PD1, PD2, PPSQ, NP;
    static const Layout& get() { static const Layout l; return l; }
private:
    Layout() {
        NC = LineCanon::get().count;
        int o = 0;
        LS = o;  o += 3 * NC;
        PCS = o; o += PCODE_NB;
        PCO = o; o += PCODE_NB;
        J = o;   o += PATTERN4_NB * PATTERN4_NB;
        TM = o;  o += 2048;
        CNT = o; o += 2 * PATTERN4_NB * 3;
        PSQ = o; o += 2 * NORBIT;
        BIAS = o; o += 2;
        PH = o;  o += 15;
        NV = o;
        o = 0;
        PLS = o; o += NC;
        PPS = o; o += PCODE_NB;
        PPO = o; o += PCODE_NB;
        PJ = o;  o += 32 * PATTERN4_NB * PATTERN4_NB;
        PD1 = o; o += 5;
        PD2 = o; o += 5;
        PPSQ = o; o += NORBIT;
        NP = o;
    }
};

inline const int* boardCells() {
    static const auto v = [] { static int a[NMOVES]; for (int m = 0; m < NMOVES; ++m) a[m] = toCell(m); return a; }();
    return v;
}

inline uint16_t relKey(const Board& b, int d, int c) {
    const uint16_t k = b.lkey[d][c];
    return b.side == BLACK ? k : PatternTables::get().swapKey[k];
}

// 가치 특징 열거: emit(int index)
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
            const uint16_t k = b.lkey[d][c];
            const bool hs = keyHasStone(k);
            near |= hs;
            if (cs == 0 && !hs) continue;
            emit(L.LS + cs * L.NC + C.id[relKey(b, d, c)]);
        }
        if (cs == 0) {
            if (!near) continue;
            emit(L.PCS + b.pc[s][c]);
            emit(L.PCO + b.pc[o][c]);
            const int a = b.p4[s][c], bb = b.p4[o][c];
            if (a | bb) emit(L.J + a * PATTERN4_NB + bb);
        } else {
            emit(L.PSQ + (cs - 1) * NORBIT + orbitOf(c));
        }
    }
    emit(L.TM + b.threatMask());
    for (int t = 0; t < 2; ++t) {
        const int* cnt = b.p4cnt[t == 0 ? s : o];
        for (int k = L_FLEX2; k <= A_FIVE; ++k)
            if (cnt[k]) emit(L.CNT + (t * PATTERN4_NB + k) * 3 + std::min(cnt[k], 3) - 1);
    }
    emit(L.BIAS + (self == WHITE));
    emit(L.PH + std::min(b.ply / 16, 14));
}

// 정책용 전역 위협 클래스 (5bit)
inline int threatClass(const Board& b) {
    const int* S = b.p4cnt[b.side - 1]; const int* O = b.p4cnt[2 - b.side];
    return (O[A_FIVE] > 0) | (O[B_FLEX4] > 0) << 1 | (O[C_BLOCK4_FLEX3] + O[D_BLOCK4_PLUS] + O[E_BLOCK4] > 0) << 2 |
           (S[B_FLEX4] > 0) << 3 | (S[C_BLOCK4_FLEX3] + S[D_BLOCK4_PLUS] + S[E_BLOCK4] > 0) << 4;
}

// 정책 특징 열거 (후보 칸 c): emit(int index)
template <class F>
inline void policyFeatures(const Board& b, int c, int tclass, F&& emit) {
    const auto& L = Layout::get();
    const auto& C = LineCanon::get();
    const int s = b.side - 1, o = 2 - b.side;
    for (int d = 0; d < 4; ++d)
        if (keyHasStone(b.lkey[d][c])) emit(L.PLS + C.id[relKey(b, d, c)]);
    emit(L.PPS + b.pc[s][c]);
    emit(L.PPO + b.pc[o][c]);
    emit(L.PJ + (tclass * PATTERN4_NB + b.p4[s][c]) * PATTERN4_NB + b.p4[o][c]);
    emit(L.PD1 + distBucket(c, b.lastMove(0)));
    emit(L.PD2 + distBucket(c, b.lastMove(1)));
    emit(L.PPSQ + orbitOf(c));
}

// 테이퍼 진행도
constexpr int TAPER_PLY = 120;
inline float taper(const Board& b) { return std::min(b.ply, TAPER_PLY) / float(TAPER_PLY); }

} // namespace gmk::feat
