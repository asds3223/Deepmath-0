// alphabeta.hpp — 싱글코어 알파-베타 탐색 코어 (헤더 온리, C++17)
//
// 구성: 반복 심화 + 어스퍼레이션 윈도우 + PVS(negamax, fail-soft)
//       + 전치표(TT) + 킬러/히스토리 수 정렬 + 메이트 거리 가지치기 + 시간/노드 제한
//
// Position 타입 P 가 제공해야 하는 인터페이스 (수 = int, 0 <= m < P::NUM_MOVES):
//   static constexpr int NUM_MOVES;   // 수 인덱스 상한 (예: 15x15 → 225)
//   uint64_t key() const;             // 조브리스트 해시 (둘 차례 포함)
//   int  generate(int* out) const;    // 후보 수 생성, 개수 반환. 좋은 수가 앞에 오도록 정렬해 두면
//                                     // 그 순서가 히스토리 동점 시 타이브레이커로 쓰인다.
//   void make(int m);  void undo(int m);
//   bool isLoss() const;              // 둘 차례인 쪽이 이미 진 상태(직전 수로 상대가 승리)
//   bool isDraw() const;              // 무승부(판이 가득 참 등)
//   int  evaluate() const;            // 둘 차례 기준 정적 평가, |값| < ab::MATE_BOUND
//
// 선택 인터페이스 (있으면 자동 사용, 없으면 기존 동작과 완전히 동일):
//   bool leafProbe(int ply, int& score) const;
//       리프에서 정확히 판정 가능한 전술 상황(즉승, 막을 수 없는 쌍사 등)이면 true 와
//       루트 기준 메이트 점수를 돌려준다. VCF/위협 탐색 같은 정지 탐색도 이 자리에 붙인다.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

namespace ab {

constexpr int MAX_PLY    = 128;
constexpr int INF        = 32000;
constexpr int MATE       = 30000;
constexpr int MATE_BOUND = MATE - MAX_PLY;   // |score| >= MATE_BOUND 이면 강제승/패
constexpr int NO_MOVE    = -1;

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

namespace detail {
template <class P, class = void> struct HasLeafProbe : std::false_type {};
template <class P>
struct HasLeafProbe<P, std::void_t<decltype(std::declval<const P&>().leafProbe(0, std::declval<int&>()))>>
    : std::true_type {};
} // namespace detail

// ───────────────────────── 전치표 ─────────────────────────
struct TTEntry {            // 16 bytes
    uint64_t key;
    int16_t  score;
    int16_t  move;
    int8_t   depth;
    uint8_t  bound;
    uint8_t  gen;
    uint8_t  pad;
};

class TransTable {
public:
    explicit TransTable(size_t mb = 64) { resize(mb); }

    void resize(size_t mb) {
        size_t n = 1;
        while (n * 2 * sizeof(TTEntry) <= mb * 1024 * 1024) n *= 2;
        table_.assign(n, TTEntry{0, 0, int16_t(NO_MOVE), 0, BOUND_NONE, 0, 0});
        mask_ = n - 1;
    }
    void clear() { std::fill(table_.begin(), table_.end(), TTEntry{0, 0, int16_t(NO_MOVE), 0, BOUND_NONE, 0, 0}); }
    void newSearch() { ++gen_; }

    const TTEntry* probe(uint64_t key) const {
        const TTEntry& e = table_[key & mask_];
        return (e.bound != BOUND_NONE && e.key == key) ? &e : nullptr;
    }

    void store(uint64_t key, int score, int move, int depth, Bound bound) {
        TTEntry& e = table_[key & mask_];
        if (e.bound != BOUND_NONE && e.key == key) {
            if (move == NO_MOVE) move = e.move;                // 기존 최선수 보존
            if (bound != BOUND_EXACT && depth + 2 < e.depth) { // 훨씬 깊은 기존 정보는 유지
                e.move = int16_t(move); e.gen = gen_;
                return;
            }
        } else if (e.bound != BOUND_NONE && e.gen == gen_ && e.depth > depth + 2) {
            return;                                            // 같은 세대의 깊은 엔트리 우선
        }
        e.key = key; e.score = int16_t(score); e.move = int16_t(move);
        e.depth = int8_t(depth); e.bound = bound; e.gen = gen_;
    }

private:
    std::vector<TTEntry> table_;
    size_t  mask_ = 0;
    uint8_t gen_  = 0;
};

// 메이트 점수는 "루트로부터의 거리" 기준이므로 TT 에는 "현재 노드로부터의 거리"로 변환해 저장
inline int scoreToTT(int s, int ply)   { return s >= MATE_BOUND ? s + ply : s <= -MATE_BOUND ? s - ply : s; }
inline int scoreFromTT(int s, int ply) { return s >= MATE_BOUND ? s - ply : s <= -MATE_BOUND ? s + ply : s; }

// ───────────────────────── 설정/결과 ─────────────────────────
struct Limits {
    int      maxDepth = MAX_PLY - 1;
    int64_t  timeMs   = -1;     // <0: 시간 제한 없음 (하드 리밋)
    uint64_t maxNodes = 0;      // 0: 노드 제한 없음
};

struct Config {
    bool useTT         = true;
    bool useAspiration = true;
    int  aspDelta      = 40;    // 평가 스케일에 맞춰 조정
    // LMR 은 기본 OFF. 오목류에서 켤 때는 강제수(4, 열린 3 등)를 감축 대상에서 반드시 제외할 것.
    bool useLMR        = false;
};

struct Result {
    int      bestMove = NO_MOVE;
    int      score    = 0;
    int      depth    = 0;      // 완료된 마지막 반복 깊이
    uint64_t nodes    = 0;
    int64_t  timeMs   = 0;
    std::vector<int> pv;
};

// ───────────────────────── 탐색기 ─────────────────────────
template <class P>
class Searcher {
public:
    Config     cfg;
    TransTable tt;
    std::function<void(const Result&)> onIteration;   // 반복 완료마다 호출 (선택)

    explicit Searcher(size_t ttMb = 64) : tt(ttMb) { clearHeuristics(); }

    void stop() { stopFlag_.store(true, std::memory_order_relaxed); }  // 외부 중단
    void clearHeuristics() {
        std::memset(history_, 0, sizeof(history_));
        for (auto& k : killers_) k[0] = k[1] = NO_MOVE;
    }

    Result think(P& pos, const Limits& lim) {
        lim_ = lim; nodes_ = 0; stopped_ = false;
        stopFlag_.store(false, std::memory_order_relaxed);
        start_ = Clock::now();
        tt.newSearch();
        for (auto& k : killers_) k[0] = k[1] = NO_MOVE;
        for (auto& side : history_) for (int& h : side) h /= 2;   // 히스토리 에이징

        Result res;
        int rootMoves[P::NUM_MOVES];
        if (pos.isLoss() || pos.isDraw() || pos.generate(rootMoves) == 0) return res;
        rootBest_ = rootMoves[0];             // 어떤 경우에도 합법수 반환 보장
        res.bestMove = rootBest_;

        int prev = 0;
        const int maxD = std::min(lim.maxDepth, MAX_PLY - 1);
        for (int d = 1; d <= maxD; ++d) {
            int alpha = -INF, beta = INF, delta = cfg.aspDelta, s = 0;
            if (cfg.useAspiration && d >= 4 && std::abs(prev) < MATE_BOUND) {
                alpha = std::max(prev - delta, -INF);
                beta  = std::min(prev + delta, INF);
            }
            for (;;) {
                s = search(pos, alpha, beta, d, 0);
                if (stopped_) break;
                if      (s <= alpha) alpha = std::max(s - delta, -INF);   // fail-low → 아래로 확장
                else if (s >= beta)  beta  = std::min(s + delta, INF);    // fail-high → 위로 확장
                else break;
                delta *= 2;
                if (delta > 16 * cfg.aspDelta) { alpha = -INF; beta = INF; }
            }

            // 중단된 반복이라도 루트에서 alpha 를 올린 수는 유효 → rootBest_ 는 항상 신뢰 가능
            res.bestMove = rootBest_;
            if (stopped_) break;

            prev = s;
            res.score = s; res.depth = d;
            res.pv.assign(pv_[0], pv_[0] + pvLen_[0]);
            res.nodes = nodes_; res.timeMs = elapsedMs();
            if (onIteration) onIteration(res);

            // 강제승/패가 확정되고 그 수순 길이만큼 읽었으면 종료
            if (std::abs(s) >= MATE_BOUND && d >= MATE - std::abs(s)) break;
            // 소프트 리밋: 다음 반복을 끝낼 가망이 없으면 시작하지 않음
            if (lim.timeMs >= 0 && elapsedMs() * 2 > lim.timeMs) break;
        }
        res.nodes = nodes_; res.timeMs = elapsedMs();
        return res;
    }

private:
    using Clock = std::chrono::steady_clock;

    int64_t elapsedMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
    }

    bool checkStop() {
        if (stopped_) return true;
        if ((nodes_ & 2047) == 0) {
            if (stopFlag_.load(std::memory_order_relaxed) ||
                (lim_.timeMs >= 0 && elapsedMs() >= lim_.timeMs))
                stopped_ = true;
        }
        if (lim_.maxNodes && nodes_ >= lim_.maxNodes) stopped_ = true;
        return stopped_;
    }

    // negamax PVS, fail-soft. 반환값은 둘 차례 기준.
    int search(P& pos, int alpha, int beta, int depth, int ply) {
        pvLen_[ply] = 0;
        if (checkStop()) return 0;
        ++nodes_;

        const bool root   = (ply == 0);
        const bool pvNode = (beta - alpha > 1);

        if (!root) {
            if (pos.isLoss()) return -MATE + ply;
            if (pos.isDraw()) return 0;
            // 메이트 거리 가지치기: 이미 찾은 더 짧은 승리보다 나을 수 없으면 중단
            alpha = std::max(alpha, -MATE + ply);
            beta  = std::min(beta, MATE - ply - 1);
            if (alpha >= beta) return alpha;
        }

        if (depth <= 0 || ply >= MAX_PLY - 1) {          // ← 리프
            if constexpr (detail::HasLeafProbe<P>::value) {
                int s;
                if (pos.leafProbe(ply, s)) return s;     // 정확한 전술 판정 / 정지 탐색 자리
            }
            return pos.evaluate();
        }

        // ── 전치표 조회 ──
        const uint64_t key = pos.key();
        int ttMove = NO_MOVE;
        if (cfg.useTT) {
            if (const TTEntry* e = tt.probe(key)) {
                ttMove = e->move;
                if (!root && !pvNode && e->depth >= depth) {
                    const int s = scoreFromTT(e->score, ply);
                    if (e->bound == BOUND_EXACT ||
                        (e->bound == BOUND_LOWER && s >= beta) ||
                        (e->bound == BOUND_UPPER && s <= alpha))
                        return s;
                }
            }
        }

        // ── 수 생성 및 정렬 점수 ──
        int moves[P::NUM_MOVES];
        int order[P::NUM_MOVES];
        const int n = pos.generate(moves);
        if (n == 0) return 0;

        const int side = ply & 1;
        for (int i = 0; i < n; ++i) {
            const int m = moves[i];
            order[i] = m == ttMove            ? (1 << 30)
                     : m == killers_[ply][0]  ? (1 << 29)
                     : m == killers_[ply][1]  ? (1 << 29) - 1
                     : history_[side][m];
        }

        const int alpha0 = alpha;
        int best = -INF, bestMove = NO_MOVE;

        for (int i = 0; i < n; ++i) {
            // 지연 선택 정렬 (동점이면 생성 순서 유지)
            int bi = i;
            for (int j = i + 1; j < n; ++j) if (order[j] > order[bi]) bi = j;
            const int m = moves[bi], mOrder = order[bi];
            for (int j = bi; j > i; --j) { moves[j] = moves[j - 1]; order[j] = order[j - 1]; }
            moves[i] = m; order[i] = mOrder;

            pos.make(m);
            int s;
            if (i == 0) {
                s = -search(pos, -beta, -alpha, depth - 1, ply + 1);
            } else {
                int r = 0;
                if (cfg.useLMR && depth >= 3 && i >= 4 && mOrder < (1 << 29) - 1)
                    r = 1 + (i >= 12);
                s = -search(pos, -alpha - 1, -alpha, depth - 1 - r, ply + 1);      // 널 윈도우
                if (r && s > alpha)
                    s = -search(pos, -alpha - 1, -alpha, depth - 1, ply + 1);      // 감축 없이 재탐색
                if (s > alpha && s < beta)
                    s = -search(pos, -beta, -alpha, depth - 1, ply + 1);           // 풀 윈도우 재탐색
            }
            pos.undo(m);
            if (stopped_) return 0;

            if (s > best) {
                best = s; bestMove = m;
                if (s > alpha) {
                    alpha = s;
                    pv_[ply][0] = m;
                    std::memcpy(pv_[ply] + 1, pv_[ply + 1], pvLen_[ply + 1] * sizeof(int));
                    pvLen_[ply] = pvLen_[ply + 1] + 1;
                    if (root) rootBest_ = m;
                    if (s >= beta) {                     // 베타 컷
                        if (killers_[ply][0] != m) { killers_[ply][1] = killers_[ply][0]; killers_[ply][0] = m; }
                        int& h = history_[side][m];
                        h += depth * depth;
                        if (h > (1 << 28)) for (auto& sd : history_) for (int& x : sd) x /= 2;
                        break;
                    }
                }
            }
        }

        if (cfg.useTT) {
            const Bound b = best >= beta ? BOUND_LOWER : best > alpha0 ? BOUND_EXACT : BOUND_UPPER;
            tt.store(key, scoreToTT(best, ply), bestMove, depth, b);
        }
        return best;
    }

    Limits            lim_;
    uint64_t          nodes_   = 0;
    bool              stopped_ = false;
    std::atomic<bool> stopFlag_{false};
    Clock::time_point start_;
    int               rootBest_ = NO_MOVE;

    int killers_[MAX_PLY][2];
    int history_[2][P::NUM_MOVES];
    int pv_[MAX_PLY + 1][MAX_PLY + 1];
    int pvLen_[MAX_PLY + 1];
};

} // namespace ab
