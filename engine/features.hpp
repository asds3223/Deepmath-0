// features.hpp — 고전 평가 특징 정의 (엔진·학습기·평가도구 공용 → train/serve skew 없음)
//
// 평가함수는 하나: Yixin 식 압축 고전평가. 모든 특징은 "둘 차례(self)" 관점.
//
// [가치]  v = Σ w_i  (로짓 단위, 가중치 2벌을 진행도 t 로 보간 — 초반/후반)
//   CS/CO 빈칸의 자기/상대 4방향 위협 단계 조합 코드(495)       — 방향 간 결합
//   J     빈칸의 (자기 칸 등급 × 상대 칸 등급)                 — 공수 겸용 급소
//   INIT  (자기 최고 등급 × 상대 최고 등급)                     — 전역 주도권
//   CNT   칸 등급별 개수 버킷 (1, 2, 3+)                      — 위협의 양
//   PSQ   돌 위치(D4 대칭 36궤도) × 소유                      — 위치 가치
//   SH    빈칸의 3x3 국소 모양 (이웃 8칸, D4 대칭 동치)        — 줄로 안 이어진 형태(뭉침·날일자 등)
//   BIAS  둘 차례 색,  PH 진행 버킷(16수 단위)
//
// [정책]  p(c) = Σ w_i  (후보 칸마다; 수 정렬·실현확률 깊이·후보 컷에 사용)
//   PCS/PCO 조합 코드, PSH 3x3 국소 모양, PJ (전역 전술 등급 25 × 칸 등급 결합), PD1/PD2 직전 수 거리, PPSQ 위치
//   PLS  4방향 줄 모양 원형 (좌우 반전 동치)                    — 줄 위 미세한 모양 차이
//   PR1/PR2  상대 직전 수 / 자기 직전 수까지의 상대 좌표 (9x9, 그 밖 1칸)
//   PRS/PRO  상대 직전 수 상대 좌표 × 이 칸의 자기 / 상대 등급   — 응수 자리(막기·맞받기) 구분
//   PTC/PTO  전역 전술 등급(25) × 이 칸의 자기 / 상대 조합 코드(495) — 상황에 따른 급소의 무게
//   PL1/PL2  상대 / 자기 직전 수와 같은 줄(거리 1..4) 위: 거리 × 그 방향 자기 단계 × 상대 단계
#pragma once
#include "board.hpp"
#include <algorithm>

namespace gmk::feat {

// 줄 키 정규화(정책 PLS 용): 벽이 바깥쪽부터 연속인 키만 유효, 좌우 반전 동치
struct LineCanon {
    int id[KEY_NB];
    int count = 0;
    static const LineCanon& get() { static const LineCanon c; return c; }
private:
    static bool validHalf(int k, bool left) {
        bool seenNonWall = false;
        for (int j = 0; j < 4; ++j) {
            const int s = left ? j : 7 - j;
            const int v = (k >> (2 * s)) & 3;
            if (v == 3) { if (seenNonWall) return false; } else seenNonWall = true;
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
        for (int k = 0; k < KEY_NB; ++k) if (id[k] < 0) id[k] = 0;   // 실제로 나오지 않는 키
    }
};

// 직전 수 기준 상대 좌표 (9x9 = 0..80, 그 밖·없음 = 81)
inline int relOffset(int c, int m) {
    if (m < 0) return 81;
    const int q = toCell(m);
    const int dx = cellX(c) - cellX(q), dy = cellY(c) - cellY(q);
    if (dx < -4 || dx > 4 || dy < -4 || dy > 4) return 81;
    return (dy + 4) * 9 + (dx + 4);
}

// c 가 직전 수 m 과 같은 줄(거리 1..4) 위에 있으면 (방향, 거리) — 아니면 d = -1
inline void lineRel(int c, int m, int& d, int& dist) {
    d = -1; dist = 0;
    if (m < 0) return;
    const int q = toCell(m);
    const int dx = cellX(c) - cellX(q), dy = cellY(c) - cellY(q);
    int k = 0;
    if (dy == 0) { d = 0; k = dx; } else if (dx == 0) { d = 1; k = dy; }
    else if (dx == dy) { d = 2; k = dx; } else if (dx == -dy) { d = 3; k = dy; }
    if (d < 0) return;
    k = k < 0 ? -k : k;
    if (k < 1 || k > 4) { d = -1; return; }
    dist = k;
}

// 3x3 모양 정규화: 8 대칭 중 최소 키로 묶어 조밀한 id
struct ShapeCanon {
    int id[KEY_NB];
    int count = 0;
    static const ShapeCanon& get() { static const ShapeCanon c; return c; }
private:
    ShapeCanon() {
        int slotOf[3][3];
        for (int j = 0; j < 8; ++j) slotOf[SHAPE_DY[j] + 1][SHAPE_DX[j] + 1] = j;
        std::fill(id, id + KEY_NB, -1);
        for (int k = 0; k < KEY_NB; ++k) {
            int best = k;
            for (int t = 0; t < 8; ++t) {
                int v = 0;
                for (int j = 0; j < 8; ++j) {
                    int x = SHAPE_DX[j], y = SHAPE_DY[j];
                    if (t & 1) x = -x;
                    if (t & 2) y = -y;
                    if (t & 4) std::swap(x, y);
                    v |= ((k >> (2 * j)) & 3) << (2 * slotOf[y + 1][x + 1]);
                }
                best = std::min(best, v);
            }
            if (id[best] < 0) id[best] = count++;
            id[k] = id[best];
        }
    }
};
inline bool g_useShape = true;
inline bool g_policyExt = true;  // 정책 확장 표(PLS/PR*) 사용
inline bool g_policyExt2 = true; // 정책 2차 결합 표(PTC/PTO/PL*) 사용   // false 면 3x3 모양 항을 쓰지 않음 (B 와 같은 특징 집합, 비교용)

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
    int CS, CO, J, INIT, CNT, PSQ, SH, BIAS, PH, NV;      // value
    int PCS, PCO, PSH, PJ, PD1, PD2, PPSQ, PLS, PR1, PR2, PRS, PRO, PTC, PTO, PL1, PL2, NP;   // policy (새 표는 뒤에 붙인다)
    static const Layout& get() { static const Layout l; return l; }
private:
    Layout() {
        int o = 0;
        CS = o;   o += CODE_NB;
        CO = o;   o += CODE_NB;
        J = o;    o += CLASS_NB * CLASS_NB;
        INIT = o; o += CLASS_NB * CLASS_NB;
        CNT = o;  o += 2 * CLASS_NB * 3;
        PSQ = o;  o += 2 * NORBIT;
        SH = o;   o += ShapeCanon::get().count;
        BIAS = o; o += 2;
        PH = o;   o += 15;
        NV = o;
        o = 0;
        PCS = o;  o += CODE_NB;
        PCO = o;  o += CODE_NB;
        PSH = o;  o += ShapeCanon::get().count;
        PJ = o;   o += 25 * CLASS_NB * CLASS_NB;
        PD1 = o;  o += 5;
        PD2 = o;  o += 5;
        PPSQ = o; o += NORBIT;
        PLS = o;  o += LineCanon::get().count;
        PR1 = o;  o += 82;
        PR2 = o;  o += 82;
        PRS = o;  o += 82 * CLASS_NB;
        PRO = o;  o += 82 * CLASS_NB;
        PTC = o;  o += 25 * CODE_NB;
        PTO = o;  o += 25 * CODE_NB;
        PL1 = o;  o += 4 * LEVEL_NB * LEVEL_NB;
        PL2 = o;  o += 4 * LEVEL_NB * LEVEL_NB;
        NP = o;
    }
};

inline const int* boardCells() {
    static const int* v = [] { static int a[NMOVES]; for (int m = 0; m < NMOVES; ++m) a[m] = toCell(m); return a; }();
    return v;
}

template <bool Compact = false, class F>
inline void valueFeatures(const Board& b, F&& emit) {
    const auto& L = Layout::get();
    const int self = b.side, s = self - 1, o = 2 - self;
    const int* cells = boardCells();
    for (int m = 0; m < NMOVES; ++m) {
        const int c = cells[m];
        const int col = b.cell[c];
        const int cs = col == EMPTY ? 0 : col == self ? 1 : 2;
        if (cs == 0) {
            bool near = false;
            for (int d = 0; d < 4; ++d) near |= keyHasStone(b.lkey[d][c]);
            if (!near) continue;
            emit(L.CS + b.code[s][c]);
            emit(L.CO + b.code[o][c]);
            if (g_useShape && keyHasStone(b.sh[c]))
                emit(L.SH + ShapeCanon::get().id[self == BLACK ? b.sh[c] : LineTables::get().swapKey[b.sh[c]]]);
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
    const int s = b.side - 1, o = 2 - b.side;
    emit(L.PCS + b.code[s][c]);
    emit(L.PCO + b.code[o][c]);
    if (g_useShape) emit(L.PSH + ShapeCanon::get().id[b.side == BLACK ? b.sh[c] : LineTables::get().swapKey[b.sh[c]]]);
    emit(L.PJ + (tclass * CLASS_NB + b.cls[s][c]) * CLASS_NB + b.cls[o][c]);
    emit(L.PD1 + distBucket(c, b.lastMove(0)));
    emit(L.PD2 + distBucket(c, b.lastMove(1)));
    emit(L.PPSQ + orbitOf(c));
    if (g_policyExt) {
        const auto& LC = LineCanon::get(); const auto& T = LineTables::get();
        for (int d = 0; d < 4; ++d) {
            const uint16_t k = b.lkey[d][c];
            if (keyHasStone(k)) emit(L.PLS + LC.id[b.side == BLACK ? k : T.swapKey[k]]);
        }
        const int r1 = relOffset(c, b.lastMove(0));
        emit(L.PR1 + r1);
        emit(L.PR2 + relOffset(c, b.lastMove(1)));
        emit(L.PRS + r1 * CLASS_NB + b.cls[s][c]);
        emit(L.PRO + r1 * CLASS_NB + b.cls[o][c]);
    }
    if (g_policyExt2) {
        emit(L.PTC + tclass * CODE_NB + b.code[s][c]);
        emit(L.PTO + tclass * CODE_NB + b.code[o][c]);
        int d, k;
        lineRel(c, b.lastMove(0), d, k);
        if (d >= 0) emit(L.PL1 + ((k - 1) * LEVEL_NB + b.lv[s][d][c]) * LEVEL_NB + b.lv[o][d][c]);
        lineRel(c, b.lastMove(1), d, k);
        if (d >= 0) emit(L.PL2 + ((k - 1) * LEVEL_NB + b.lv[s][d][c]) * LEVEL_NB + b.lv[o][d][c]);
    }
}

constexpr int TAPER_PLY = 120;
inline float taper(const Board& b) { return std::min(b.ply, TAPER_PLY) / float(TAPER_PLY); }

} // namespace gmk::feat
