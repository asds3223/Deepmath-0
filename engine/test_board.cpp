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
                if (a.pc[s][c] != b.pc[s][c] || a.p4[s][c] != b.p4[s][c]) return false;
                for (int d = 0; d < 4; ++d) if (a.pat[s][d][c] != b.pat[s][d][c]) return false;
            }
            if ((a.nearCnt[c] > 0) != (b.nearCnt[c] > 0)) return false;
        }
    }
    for (int s = 0; s < 2; ++s) for (int k = 0; k < PATTERN4_NB; ++k) if (a.p4cnt[s][k] != b.p4cnt[s][k]) return false;
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

int main() {
    std::mt19937 rng(1);
    int fail = 0, checks = 0, games = 0;
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
    std::printf("games=%d checks=%d fail=%d\n", games, checks, fail);

    // 평가/생성 속도 (휴리스틱 모델 없음, 0 가중치 모델)
    Model mdl; Board b; b.model = &mdl;
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
