// Board 증분 갱신 검증: 무작위 대국에서 make/undo 후 상태가 처음부터 다시 만든 상태와 같아야 한다.
// 또한 5목 판정(isLoss)을 직접 세기 방식과 비교한다.
#include "eval.hpp"
#include <cstdio>
#include <random>

using namespace gmk;

static bool sameState(const Board& a, const Board& b) {
    for (int m = 0; m < NMOVES; ++m) {
        int c = toCell(m);
        if (a.cell[c] != b.cell[c]) return false;
        for (int d = 0; d < 4; ++d) if (a.lkey[d][c] != b.lkey[d][c]) return false;
        if (a.cell[c] == EMPTY) {
            for (int s = 0; s < 2; ++s) {
                if (a.code[s][c] != b.code[s][c] || a.cls[s][c] != b.cls[s][c]) return false;
                for (int d = 0; d < 4; ++d) if (a.lv[s][d][c] != b.lv[s][d][c]) return false;
            }
            if ((a.nearCnt[c] > 0) != (b.nearCnt[c] > 0)) return false;
        }
    }
    for (int s = 0; s < 2; ++s) for (int k = 0; k < CLASS_NB; ++k) if (a.clsCnt[s][k] != b.clsCnt[s][k]) return false;
    return a.hash == b.hash && a.side == b.side && a.ply == b.ply;
}

static bool fiveAt(const Board& b, int m) {
    int c = toCell(m), col = b.cell[c];
    for (int d = 0; d < 4; ++d) {
        int n = 1;
        for (int q = c + DIRS[d]; b.cell[q] == col; q += DIRS[d]) ++n;
        for (int q = c - DIRS[d]; b.cell[q] == col; q -= DIRS[d]) ++n;
        if (n >= 5) return true;
    }
    return false;
}

// 실제 보드에서 c 에 col 을 두었을 때 방향 d 로 c 를 지나는 연속 길이
static int runThrough(const Board& b, int c, int d, int col, int extra = -1) {
    auto is = [&](int q) { return q == c || q == extra || b.cell[q] == col; };
    int n = 1;
    for (int q = c + DIRS[d]; b.cell[q] != WALL && is(q); q += DIRS[d]) ++n;
    for (int q = c - DIRS[d]; b.cell[q] != WALL && is(q); q -= DIRS[d]) ++n;
    return n;
}
// 독자 정의 검증: FIVE/OPEN4/FOUR 를 보드 위 직접 시뮬레이션(완성점 개수)과 비교
static int checkLevels(const Board& b) {
    int bad = 0;
    for (int m = 0; m < NMOVES; ++m) {
        const int c = toCell(m);
        if (b.cell[c] != EMPTY) continue;
        for (int col = 1; col <= 2; ++col)
            for (int d = 0; d < 4; ++d) {
                const int lvl = b.lv[col - 1][d][c];
                const bool five = runThrough(b, c, d, col) >= 5;
                int comp = 0;
                for (int k = -4; k <= 4; ++k) {
                    const int q = c + k * DIRS[d];
                    if (!k || b.cell[q] != EMPTY) continue;
                    if (runThrough(b, c, d, col, q) >= 5) ++comp;
                }
                const int expect = five ? FIVE : comp >= 2 ? OPEN4 : comp == 1 ? FOUR : -1;
                if (expect >= 0 ? lvl != expect : lvl >= FOUR) ++bad;
            }
    }
    return bad;
}

int main() {
    std::mt19937 rng(1);
    int fail = 0, checks = 0, games = 0;
    long levelBad = 0;
    for (int g = 0; g < 300; ++g) {
        Board b;
        std::vector<int> moves;
        for (;;) {
            int mv[NMOVES], n = b.generate(mv);
            if (!n || b.isLoss() || b.isDraw()) break;
            int m = mv[rng() % std::min(n, 6)];
            b.make(m); moves.push_back(m);
            if (b.isLoss() != fiveAt(b, m)) { ++fail; std::printf("five mismatch g=%d ply=%d\n", g, b.ply); }
            if (b.isLoss()) break;
            levelBad += checkLevels(b);
        }
        ++games;
        // 중간 지점까지 undo 후 재구성 상태와 비교
        int back = int(moves.size()) / 2;
        for (int i = 0; i < back; ++i) { b.undo(moves.back()); moves.pop_back(); }
        uint8_t arr[NMOVES] = {0};
        for (size_t i = 0; i < moves.size(); ++i) arr[moves[i]] = (i % 2 == 0) ? BLACK : WHITE;
        Board r; r.setPosition(arr, b.side, b.lastMove(0), b.lastMove(1));
        ++checks;
        if (!sameState(b, r)) { ++fail; std::printf("state mismatch g=%d\n", g); }
    }
    std::printf("games=%d checks=%d fail=%d level-mismatch(FIVE/OPEN4/FOUR vs brute force)=%ld\n",
                games, checks, fail, levelBad);
    fail += levelBad != 0;
    // 등급 분포 (참고)
    {
        Board b; for (int m : {112, 113, 97, 127, 98, 96, 128, 82, 111, 126, 99, 100}) b.make(m);
        int hist[LEVEL_NB] = {0};
        for (int m = 0; m < NMOVES; ++m) if (b.isEmptyMove(m)) for (int d = 0; d < 4; ++d) hist[b.lv[0][d][toCell(m)]]++;
        std::printf("level histogram (black):");
        for (int l = 0; l < LEVEL_NB; ++l) std::printf(" %s=%d", levelName(l), hist[l]);
        std::printf("\n");
    }

    // 평가/생성 속도 (휴리스틱 모델 없음, 0 가중치 모델)
    Model mdl; Board b; b.setModel(&mdl);
    int seq[] = {112, 113, 97, 127, 98, 96, 128, 82, 111, 126};
    for (int m : seq) b.make(m);
    auto t0 = std::chrono::steady_clock::now();
    long long acc = 0; int mv[NMOVES];
    for (int i = 0; i < 200000; ++i) { acc += b.evaluate(); acc += b.generate(mv); }
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200000;
    std::printf("evaluate+generate: %.2f us/call (acc=%lld)\n", us, acc);
    std::printf(fail ? "FAILED\n" : "ALL OK\n");
    return fail != 0;
}
