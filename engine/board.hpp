// board.hpp — 15x15 프리스타일 오목 Position (alphabeta.hpp 의 P 인터페이스 구현)
//
// - 패딩 4칸(벽)을 둔 23x23 배열. 라인 키(방향별 16bit)를 증분 갱신한다.
// - 각 빈칸의 방향별 패턴(Rapfi 동일 분류) / pcode / Pattern4 를 증분 유지하고,
//   Pattern4 개수(p4cnt)를 색별로 센다 → 즉승/강제 방어/위협 마스크가 O(1).
// - 평가/수 정렬은 외부 Model(eval.hpp) 포인터를 통해 연결된다(없으면 휴리스틱 정렬).
#pragma once
#include "pattern.hpp"
#include "alphabeta.hpp"
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace gmk {

constexpr int BS = 15, PAD = 4, W = BS + 2 * PAD;   // 23
constexpr int CELLS = W * W;                          // 529
constexpr int NMOVES = BS * BS;                       // 225
constexpr int DIRS[4] = {1, W, W + 1, W - 1};         // 가로, 세로, ↘, ↙
enum Color : uint8_t { EMPTY = 0, BLACK = 1, WHITE = 2, WALL = 3 };
inline int opp(int c) { return 3 - c; }

inline int toCell(int m) { return (m % BS + PAD) + (m / BS + PAD) * W; }
inline int toMove(int c) { return (c % W - PAD) + (c / W - PAD) * BS; }
inline int cellX(int c) { return c % W - PAD; }
inline int cellY(int c) { return c / W - PAD; }

// 키 슬롯: 오프셋 k(-4..-1 → 0..3, +1..+4 → 4..7)
inline int slotOf(int k) { return k < 0 ? k + 4 : k + 3; }

// 줄 키에 돌(1/2)이 하나라도 있는가 (벽=3 은 제외)
inline bool keyHasStone(uint16_t k) { return ((k ^ (k >> 1)) & 0x5555) != 0; }

class Model;   // eval.hpp

struct Zobrist {
    uint64_t z[3][CELLS];
    uint64_t side;
    Zobrist() {
        std::mt19937_64 r(0x9E3779B97F4A7C15ull);
        for (auto& a : z) for (auto& x : a) x = r();
        side = r();
    }
    static const Zobrist& get() { static const Zobrist z; return z; }
};

class Board {
public:
    static constexpr int NUM_MOVES = NMOVES;

    uint8_t  cell[CELLS];
    uint16_t lkey[4][CELLS];          // 절대 색 키 (0 빈,1 흑,2 백,3 벽)
    uint8_t  pat[2][4][CELLS];       // [색-1][방향][칸] 해당 색이 이 칸에 두었을 때 패턴 (빈칸만 유효)
    uint16_t pc[2][CELLS];           // pcode
    uint8_t  p4[2][CELLS];           // Pattern4
    uint8_t  nearCnt[CELLS];         // 후보 반경 내 돌 개수
    int      p4cnt[2][PATTERN4_NB];  // 빈칸 기준 Pattern4 개수 (색별)
    int      hist[NMOVES + 1];
    uint8_t  fiveMade[NMOVES + 1];   // hist[i] 가 5목을 만든 수인가
    int      ply  = 0;
    int      side = BLACK;           // 둘 차례
    uint64_t hash = 0;

    const Model* model = nullptr;    // 평가/정렬 모델 (eval.hpp)

    Board() { clear(); }

    void clear() {
        const auto& T = PatternTables::get();
        std::memset(cell, WALL, sizeof(cell));
        for (int y = 0; y < BS; ++y) for (int x = 0; x < BS; ++x) cell[(x + PAD) + (y + PAD) * W] = EMPTY;
        for (int d = 0; d < 4; ++d)
            for (int c = 0; c < CELLS; ++c) {
                uint16_t k = 0;
                for (int o = -4; o <= 4; ++o) {
                    if (!o) continue;
                    int q = c + o * DIRS[d];
                    int v = (q >= 0 && q < CELLS) ? cell[q] : WALL;
                    k |= uint16_t(v << (2 * slotOf(o)));
                }
                lkey[d][c] = k;
            }
        std::memset(nearCnt, 0, sizeof(nearCnt));
        std::memset(p4cnt, 0, sizeof(p4cnt));
        for (int c = 0; c < CELLS; ++c) {
            for (int s = 0; s < 2; ++s) { pc[s][c] = 0; p4[s][c] = NONE; for (int d = 0; d < 4; ++d) pat[s][d][c] = DEAD; }
            if (cell[c] != EMPTY) continue;
            for (int d = 0; d < 4; ++d) { pat[0][d][c] = T.patSelf[lkey[d][c]]; pat[1][d][c] = T.patSelf[T.swapKey[lkey[d][c]]]; }
            refreshCell(c);
            p4cnt[0][p4[0][c]]++; p4cnt[1][p4[1][c]]++;
        }
        ply = 0; side = BLACK; hash = 0;
        hist[0] = -1; fiveMade[0] = 0;
    }

    // ── alphabeta 인터페이스 ──
    uint64_t key() const { return hash ^ (side == WHITE ? Zobrist::get().side : 0); }

    void make(int m) {
        const int c = toCell(m);
        fiveMade[ply] = p4[side - 1][c] == A_FIVE;
        hist[ply++] = m;
        put(c, side);
        side = opp(side);
    }
    void undo(int /*m*/) {
        const int m = hist[--ply];
        side = opp(side);
        remove(toCell(m));
    }
    bool isLoss() const { return ply > 0 && fiveMade[ply - 1]; }
    bool isDraw() const { return ply >= NMOVES; }
    int  evaluate() const;                                  // eval.hpp 에서 정의
    int  generate(int* out) const;                          // eval.hpp 에서 정의 (정책 정렬)

    // 리프에서 정확히 판정 가능한 전술 상황 (alphabeta.hpp 의 leafProbe 훅)
    bool leafProbe(int plyFromRoot, int& score) const {
        const int s = side - 1, o = 2 - side;
        if (p4cnt[s][A_FIVE]) { score = ab::MATE - (plyFromRoot + 1); return true; }           // 즉승
        if (p4cnt[o][A_FIVE] >= 2) { score = -(ab::MATE - (plyFromRoot + 2)); return true; }   // 막을 수 없는 쌍사
        return false;
    }

    // ── 보조 ──
    int  stoneCount() const { return ply; }
    int  lastMove(int back = 0) const { return ply - 1 - back >= 0 ? hist[ply - 1 - back] : -1; }
    bool isEmptyMove(int m) const { return cell[toCell(m)] == EMPTY; }
    bool isCandidate(int c) const { return cell[c] == EMPTY && nearCnt[c] > 0; }
    int  colorAt(int m) const { return cell[toCell(m)]; }

    // 정적 배치(기록 복원용): 흑/백 돌을 차례 무관하게 놓는다. 이후 side/ply/hist 를 지정.
    void setPosition(const uint8_t* board225, int sideToMove, int last1, int last2) {
        clear();
        int n = 0;
        for (int m = 0; m < NMOVES; ++m) if (board225[m]) { put(toCell(m), board225[m]); ++n; }
        ply = n; side = sideToMove;
        for (int i = 0; i < ply; ++i) { hist[i] = -1; fiveMade[i] = 0; }
        if (ply >= 1) hist[ply - 1] = last1;
        if (ply >= 2) hist[ply - 2] = last2;
    }

    // 위협 마스크 (Rapfi eval.cpp makeThreatMask 와 동일한 11bit, self = 둘 차례)
    int threatMask() const {
        const int* S = p4cnt[side - 1]; const int* O = p4cnt[2 - side];
        int m = 0;
        m |= (O[A_FIVE] > 0) << 0;
        m |= (S[B_FLEX4] > 0) << 1;
        m |= (O[B_FLEX4] > 0) << 2;
        m |= (S[D_BLOCK4_PLUS] + S[C_BLOCK4_FLEX3] > 0) << 3;
        m |= (S[E_BLOCK4] > 0) << 4;
        m |= (S[G_FLEX3_PLUS] + S[F_FLEX3_2X] > 0) << 5;
        m |= (S[H_FLEX3] > 0) << 6;
        m |= (O[D_BLOCK4_PLUS] + O[C_BLOCK4_FLEX3] > 0) << 7;
        m |= (O[E_BLOCK4] > 0) << 8;
        m |= (O[G_FLEX3_PLUS] + O[F_FLEX3_2X] > 0) << 9;
        m |= (O[H_FLEX3] > 0) << 10;
        return m;
    }

    // 후보 반경: 5x5 정사각 + 8방향 거리 3 (Rapfi SQUARE2_LINE3 에 해당)
    static const std::vector<int>& nearOffsets() {
        static const std::vector<int> v = [] {
            std::vector<int> r;
            for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
                if (!dx && !dy) continue;
                int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
                bool sq = ax <= 2 && ay <= 2;
                bool ln = (ax == 3 || ay == 3) && (ax == 0 || ay == 0 || ax == ay);
                if (sq || ln) r.push_back(dx + dy * W);
            }
            return r;
        }();
        return v;
    }

private:
    void refreshCell(int c) {
        const auto& T = PatternTables::get();
        for (int s = 0; s < 2; ++s) {
            pc[s][c] = T.pcode[pat[s][0][c]][pat[s][1][c]][pat[s][2][c]][pat[s][3][c]];
            p4[s][c] = T.p4OfPcode[pc[s][c]];
        }
    }

    void put(int c, int col) {
        const auto& T = PatternTables::get();
        p4cnt[0][p4[0][c]]--; p4cnt[1][p4[1][c]]--;
        p4[0][c] = p4[1][c] = NONE;
        cell[c] = uint8_t(col);
        hash ^= Zobrist::get().z[col][c];
        for (int d = 0; d < 4; ++d)
            for (int k = -4; k <= 4; ++k) {
                if (!k) continue;
                const int q = c + k * DIRS[d];
                lkey[d][q] |= uint16_t(col << (2 * slotOf(-k)));
                if (cell[q] == EMPTY) updateLine(q, d, T);
            }
        for (int o : nearOffsets()) { int q = c + o; if (q >= 0 && q < CELLS) nearCnt[q]++; }
    }

    void remove(int c) {
        const auto& T = PatternTables::get();
        const int col = cell[c];
        cell[c] = EMPTY;
        hash ^= Zobrist::get().z[col][c];
        for (int d = 0; d < 4; ++d)
            for (int k = -4; k <= 4; ++k) {
                if (!k) continue;
                const int q = c + k * DIRS[d];
                lkey[d][q] &= uint16_t(~(3 << (2 * slotOf(-k))));
                if (cell[q] == EMPTY) updateLine(q, d, T);
            }
        for (int d = 0; d < 4; ++d) { pat[0][d][c] = T.patSelf[lkey[d][c]]; pat[1][d][c] = T.patSelf[T.swapKey[lkey[d][c]]]; }
        refreshCell(c);
        p4cnt[0][p4[0][c]]++; p4cnt[1][p4[1][c]]++;
        for (int o : nearOffsets()) { int q = c + o; if (q >= 0 && q < CELLS) nearCnt[q]--; }
    }

    void updateLine(int q, int d, const PatternTables& T) {
        p4cnt[0][p4[0][q]]--; p4cnt[1][p4[1][q]]--;
        pat[0][d][q] = T.patSelf[lkey[d][q]];
        pat[1][d][q] = T.patSelf[T.swapKey[lkey[d][q]]];
        refreshCell(q);
        p4cnt[0][p4[0][q]]++; p4cnt[1][p4[1][q]]++;
    }
};

} // namespace gmk
