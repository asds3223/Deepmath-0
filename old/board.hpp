// board.hpp — 15x15 프리스타일 오목 Position (alphabeta.hpp 의 P 인터페이스 구현)
//
// - 패딩 4칸(벽)을 둔 23x23 배열. 방향별 16bit 라인 키를 증분 갱신한다.
// - 빈칸마다 색별·방향별 위협 단계(Level), 4방향 조합 코드, 칸 등급(CellClass)을 증분 유지하고
//   등급별 개수(clsCnt)를 센다 → 즉승/강제 방어/전역 주도권 특징이 O(1).
// - 평가/수 정렬은 외부 Model(eval.hpp) 포인터로 연결(없으면 휴리스틱).
#pragma once
#include "alphabeta.hpp"
#include "pattern.hpp"
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
inline int slotOf(int k) { return k < 0 ? k + 4 : k + 3; }      // 오프셋 → 키 슬롯
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
    uint8_t  lv[2][4][CELLS];         // [색-1][방향][칸] 위협 단계 (빈칸만 유효)
    uint16_t code[2][CELLS];          // 4방향 조합 코드
    uint8_t  cls[2][CELLS];           // 칸 등급
    uint8_t  nearCnt[CELLS];          // 후보 반경 내 돌 수
    int      clsCnt[2][CLASS_NB];     // 빈칸 등급 개수 (색별)
    int      hist[NMOVES + 1];
    uint8_t  fiveMade[NMOVES + 1];
    int      ply  = 0;
    int      side = BLACK;
    uint64_t hash = 0;

    const Model* model = nullptr;

    Board() { clear(); }

    void clear() {
        const auto& T = LineTables::get();
        std::memset(cell, WALL, sizeof(cell));
        for (int y = 0; y < BS; ++y) for (int x = 0; x < BS; ++x) cell[(x + PAD) + (y + PAD) * W] = EMPTY;
        for (int d = 0; d < 4; ++d)
            for (int c = 0; c < CELLS; ++c) {
                uint16_t k = 0;
                for (int o = -4; o <= 4; ++o) {
                    if (!o) continue;
                    const int q = c + o * DIRS[d];
                    const int v = (q >= 0 && q < CELLS) ? cell[q] : WALL;
                    k |= uint16_t(v << (2 * slotOf(o)));
                }
                lkey[d][c] = k;
            }
        std::memset(nearCnt, 0, sizeof(nearCnt));
        std::memset(clsCnt, 0, sizeof(clsCnt));
        for (int c = 0; c < CELLS; ++c) {
            for (int s = 0; s < 2; ++s) { code[s][c] = 0; cls[s][c] = C_NONE; for (int d = 0; d < 4; ++d) lv[s][d][c] = DEAD; }
            if (cell[c] != EMPTY) continue;
            for (int d = 0; d < 4; ++d) setLevels(c, d, T);
            refreshCell(c, T);
            clsCnt[0][cls[0][c]]++; clsCnt[1][cls[1][c]]++;
        }
        ply = 0; side = BLACK; hash = 0;
        hist[0] = -1; fiveMade[0] = 0;
    }

    // ── alphabeta 인터페이스 ──
    uint64_t key() const { return hash ^ (side == WHITE ? Zobrist::get().side : 0); }
    void make(int m) {
        const int c = toCell(m);
        fiveMade[ply] = cls[side - 1][c] == C_FIVE;
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
    int  evaluate() const;            // eval.hpp
    int  generate(int* out) const;    // eval.hpp (정책 순 정렬)
    int  generateScored(int* out, int* score, uint8_t* tactical) const;   // eval.hpp (정렬 없음, 탐색용)
    // ── 위협 상태 ──
    // 상대가 다음 수에 5목(=4 를 막아야 함) 또는 4-3/승리 4 를 만들 수 있으면 "위협 상태".
    // 위협 상태에서는 위협을 실제로 없애는 수와 우리 4(맞받기)만 의미가 있다 (체스의 체크 회피와 같은 취급).
    bool oppFive() const   { return clsCnt[2 - side][C_FIVE] > 0; }
    bool oppThreat() const { const int* O = clsCnt[2 - side]; return O[C_FOUR3] + O[C_WINFOUR] > 0; }
    bool mustDefend() const { return oppFive(); }                           // 정지 탐색 강제 방어: 상대 4 만
    int  defenseMoves(int* out) const;                                     // eval.hpp
    int  forcingMoves(int* out, bool allowThree) const;                    // eval.hpp: 4 (+열린 3) 만드는 수

    // 리프에서 정확히 판정 가능한 전술 상황
    bool leafProbe(int plyFromRoot, int& score) const {
        const int s = side - 1, o = 2 - side;
        if (clsCnt[s][C_FIVE]) { score = ab::MATE - (plyFromRoot + 1); return true; }          // 즉승
        if (clsCnt[o][C_FIVE] >= 2) { score = -(ab::MATE - (plyFromRoot + 2)); return true; }  // 막을 수 없는 쌍사
        return false;
    }

    // ── 보조 ──
    int  lastMove(int back = 0) const { return ply - 1 - back >= 0 ? hist[ply - 1 - back] : -1; }
    bool isEmptyMove(int m) const { return cell[toCell(m)] == EMPTY; }
    bool isCandidate(int c) const { return cell[c] == EMPTY && nearCnt[c] > 0; }
    int  colorAt(int m) const { return cell[toCell(m)]; }
    int  bestClass(int color) const {
        const int* n = clsCnt[color - 1];
        for (int k = C_FIVE; k > C_NONE; --k) if (n[k]) return k;
        return C_NONE;
    }

    // 기록 복원: 돌을 차례 무관하게 배치한 뒤 side/ply/직전 수를 지정
    void setPosition(const uint8_t* board225, int sideToMove, int last1, int last2) {
        clear();
        int n = 0;
        for (int m = 0; m < NMOVES; ++m) if (board225[m]) { put(toCell(m), board225[m]); ++n; }
        ply = n; side = sideToMove;
        for (int i = 0; i < ply; ++i) { hist[i] = -1; fiveMade[i] = 0; }
        if (ply >= 1) hist[ply - 1] = last1;
        if (ply >= 2) hist[ply - 2] = last2;
    }

    // 후보 반경: 5x5 정사각 + 8방향 거리 3
    static const std::vector<int>& nearOffsets() {
        static const std::vector<int> v = [] {
            std::vector<int> r;
            for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
                if (!dx && !dy) continue;
                const int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
                if ((ax <= 2 && ay <= 2) || ((ax == 3 || ay == 3) && (ax == 0 || ay == 0 || ax == ay))) r.push_back(dx + dy * W);
            }
            return r;
        }();
        return v;
    }

private:
    void setLevels(int c, int d, const LineTables& T) {
        lv[0][d][c] = T.level[lkey[d][c]];
        lv[1][d][c] = T.level[T.swapKey[lkey[d][c]]];
    }
    void refreshCell(int c, const LineTables& T) {
        for (int s = 0; s < 2; ++s) {
            code[s][c] = T.code[lv[s][0][c]][lv[s][1][c]][lv[s][2][c]][lv[s][3][c]];
            cls[s][c]  = T.classOf[code[s][c]];
        }
    }
    void put(int c, int col) {
        const auto& T = LineTables::get();
        clsCnt[0][cls[0][c]]--; clsCnt[1][cls[1][c]]--;
        cls[0][c] = cls[1][c] = C_NONE;
        cell[c] = uint8_t(col);
        hash ^= Zobrist::get().z[col][c];
        for (int d = 0; d < 4; ++d)
            for (int k = -4; k <= 4; ++k) {
                if (!k) continue;
                const int q = c + k * DIRS[d];
                lkey[d][q] |= uint16_t(col << (2 * slotOf(-k)));
                if (cell[q] == EMPTY) updateLine(q, d, T);
            }
        for (int o : nearOffsets()) { const int q = c + o; if (q >= 0 && q < CELLS) nearCnt[q]++; }
    }
    void remove(int c) {
        const auto& T = LineTables::get();
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
        for (int d = 0; d < 4; ++d) setLevels(c, d, T);
        refreshCell(c, T);
        clsCnt[0][cls[0][c]]++; clsCnt[1][cls[1][c]]++;
        for (int o : nearOffsets()) { const int q = c + o; if (q >= 0 && q < CELLS) nearCnt[q]--; }
    }
    void updateLine(int q, int d, const LineTables& T) {
        clsCnt[0][cls[0][q]]--; clsCnt[1][cls[1][q]]--;
        setLevels(q, d, T);
        refreshCell(q, T);
        clsCnt[0][cls[0][q]]++; clsCnt[1][cls[1][q]]++;
    }
};

} // namespace gmk
