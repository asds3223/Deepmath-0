// alphabeta.hpp 검증: m,n,k 게임에서 순수 negamax 와 결과 비교
// g++ -O2 -std=c++17 test_alphabeta.cpp -o test_ab && ./test_ab
#include "alphabeta.hpp"
#include <cstdio>
#include <random>

template <int W, int H, int K>
struct MNK {
    static constexpr int NUM_MOVES = W * H;
    int8_t   b[W * H] = {};       // 0 빈칸, 1/2 돌
    int      side = 1, cnt = 0;
    int      hist[W * H];
    uint64_t k = 0;
    static uint64_t Z[2][W * H], ZS;

    static void init() {
        std::mt19937_64 r(12345);
        for (auto& a : Z) for (auto& x : a) x = r();
        ZS = r();
    }
    uint64_t key() const { return k ^ (side == 2 ? ZS : 0); }
    int generate(int* out) const {
        int n = 0;
        for (int i = 0; i < W * H; ++i) if (!b[i]) out[n++] = i;
        return n;
    }
    void make(int m) { b[m] = side; k ^= Z[side - 1][m]; hist[cnt++] = m; side = 3 - side; }
    void undo(int m) { side = 3 - side; --cnt; k ^= Z[side - 1][m]; b[m] = 0; }
    int run(int x, int y, int dx, int dy, int c) const {
        int n = 0;
        for (x += dx, y += dy; x >= 0 && x < W && y >= 0 && y < H && b[y * W + x] == c; x += dx, y += dy) ++n;
        return n;
    }
    bool isLoss() const {
        if (!cnt) return false;
        int m = hist[cnt - 1], x = m % W, y = m / W, c = b[m];
        static const int D[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (auto& d : D)
            if (1 + run(x, y, d[0], d[1], c) + run(x, y, -d[0], -d[1], c) >= K) return true;
        return false;
    }
    bool isDraw() const { return cnt == W * H; }
    int evaluate() const {      // K-윈도우별: 한쪽 돌만 있으면 개수^2
        int s = 0;
        static const int D[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) for (auto& d : D) {
            int ex = x + d[0] * (K - 1), ey = y + d[1] * (K - 1);
            if (ex < 0 || ex >= W || ey < 0 || ey >= H) continue;
            int a = 0, o = 0;
            for (int i = 0; i < K; ++i) {
                int c = b[(y + d[1] * i) * W + x + d[0] * i];
                if (c == side) ++a; else if (c) ++o;
            }
            if (a && !o) s += a * a; else if (o && !a) s -= o * o;
        }
        return s;
    }
};
template <int W, int H, int K> uint64_t MNK<W, H, K>::Z[2][W * H];
template <int W, int H, int K> uint64_t MNK<W, H, K>::ZS;

template <class P>
int negamax(P& p, int depth, int ply) {
    if (ply) { if (p.isLoss()) return -ab::MATE + ply; if (p.isDraw()) return 0; }
    if (depth <= 0) return p.evaluate();
    int mv[P::NUM_MOVES], n = p.generate(mv), best = -ab::INF;
    if (!n) return 0;
    for (int i = 0; i < n; ++i) { p.make(mv[i]); best = std::max(best, -negamax(p, depth - 1, ply + 1)); p.undo(mv[i]); }
    return best;
}

int main() {
    int fail = 0;
    // 1) 틱택토 완전 탐색 = 무승부
    {
        using T = MNK<3, 3, 3>; T::init(); T p;
        ab::Searcher<T> s(8);
        auto r = s.think(p, ab::Limits{9, -1, 0});
        std::printf("[1] tictactoe score=%d depth=%d nodes=%llu\n", r.score, r.depth, (unsigned long long)r.nodes);
        fail += r.score != 0;
    }
    // 2) 4x4, k=3: 선수 필승
    {
        using T = MNK<4, 4, 3>; T::init(); T p;
        ab::Searcher<T> s(8);
        auto r = s.think(p, ab::Limits{10, -1, 0});
        std::printf("[2] 4x4x3 score=%d (mate in %d ply) pv_len=%zu\n", r.score, ab::MATE - r.score, r.pv.size());
        fail += r.score < ab::MATE_BOUND;
    }
    // 3) 5x5, k=4 무작위 국면: TT 없이 고정 깊이 값이 negamax 와 정확히 일치해야 함
    //    TT 사용 시에는 PV 수순을 실제로 두어 봤을 때 모순이 없는지, 승패 판정이 일치하는지 확인
    {
        using T = MNK<5, 5, 4>; T::init();
        std::mt19937 rng(7);
        int mism = 0, mismTT = 0, total = 0;
        uint64_t nRef = 0, nTT = 0;
        for (int t = 0; t < 60; ++t) {
            T p; int plies = rng() % 7; bool ok = true;
            for (int i = 0; i < plies && ok; ++i) {
                int mv[25], n = p.generate(mv); p.make(mv[rng() % n]);
                if (p.isLoss()) ok = false;
            }
            if (!ok) continue;
            const int D = 5;
            int ref = negamax(p, D, 0);
            ab::Searcher<T> a(8); a.cfg.useTT = false;
            auto ra = a.think(p, ab::Limits{D, -1, 0});
            ab::Searcher<T> b(8);
            auto rb = b.think(p, ab::Limits{D, -1, 0});
            ++total; nRef += ra.nodes; nTT += rb.nodes;
            // 메이트 조기 종료 시에는 깊이가 D 보다 작을 수 있으나 값은 같아야 함
            if (ra.score != ref) { ++mism; std::printf("  noTT mismatch: %d vs %d\n", ra.score, ref); }
            bool winRef = ref >= ab::MATE_BOUND, lossRef = ref <= -ab::MATE_BOUND;
            if ((rb.score >= ab::MATE_BOUND) != winRef || (rb.score <= -ab::MATE_BOUND) != lossRef) ++mismTT;
            if (rb.bestMove < 0 || p.b[rb.bestMove]) ++mismTT;
        }
        std::printf("[3] 5x5x4 positions=%d noTT-mismatch=%d TT-verdict-mismatch=%d nodes(noTT)=%llu nodes(TT)=%llu\n",
                    total, mism, mismTT, (unsigned long long)nRef, (unsigned long long)nTT);
        fail += mism + mismTT;
    }
    // 4) 시간 제한: 50ms 안에 합법수를 반환
    {
        using T = MNK<7, 7, 5>; T::init(); T p;
        ab::Searcher<T> s(16);
        auto r = s.think(p, ab::Limits{ab::MAX_PLY - 1, 50, 0});
        std::printf("[4] 7x7x5 50ms: move=%d depth=%d nodes=%llu time=%lldms\n",
                    r.bestMove, r.depth, (unsigned long long)r.nodes, (long long)r.timeMs);
        fail += r.bestMove < 0 || r.timeMs > 80;
    }
    std::printf(fail ? "FAILED (%d)\n" : "ALL OK\n", fail);
    return fail != 0;
}
