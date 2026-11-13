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
//       리프에서 정확히 판정 가능한 전술 상황(즉승, 막을 수 없는 쌍사 등)이면 true 와 메이트 점수.
//   int  generateScored(int* out, int* score, uint8_t* tactical) const;
//       정렬하지 않은 후보 + 정책 점수(로짓×256) + 전술 수 표시. 있으면 다음을 쓴다:
//         · 수 정렬: TT/킬러 다음, 히스토리 + 정책 점수
//         · 실현 확률 깊이: 정책 1위와의 로짓 차이만큼 깊이를 더 소모 (전술 수는 감축 없음)
//         · 정책 간격 컷: 1위와 차이가 너무 큰 비전술 수는 탐색하지 않음 (루트 제외)
//   bool mustDefend() const;  int forcingMoves(int* out, bool allowThree) const;
//       있으면 리프에서 정지 탐색: 위협을 받지 않으면 현재 평가를 기준값(stand-pat)으로 두고
//       강제수(4 만들기, 앞쪽 몇 수는 열린 3 만들기 포함)만, 위협을 받으면 기준값 없이 방어수만 읽는다.
//
// 깊이는 내부적으로 1수 = ONE_PLY 단위(정수)로 다룬다.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

constexpr int ONE_PLY    = 4;

namespace detail {
template <class P, class = void> struct HasLeafProbe : std::false_type {};
template <class P>
struct HasLeafProbe<P, std::void_t<decltype(std::declval<const P&>().leafProbe(0, std::declval<int&>()))>>
    : std::true_type {};
template <class P, class = void> struct HasScored : std::false_type {};
template <class P>
struct HasScored<P, std::void_t<decltype(std::declval<const P&>().generateScored((int*)nullptr, (int*)nullptr, (uint8_t*)nullptr))>>
    : std::true_type {};
template <class P, class = void> struct HasVCF : std::false_type {};
template <class P>
struct HasVCF<P, std::void_t<decltype(std::declval<const P&>().vcfProbe(0, std::declval<int&>()))>> : std::true_type {};
template <class P, class = void> struct HasQuiesce : std::false_type {};
template <class P>
struct HasQuiesce<P, std::void_t<decltype(std::declval<const P&>().forcingMoves((int*)nullptr, false)),
                                 decltype(std::declval<const P&>().mustDefend())>> : std::true_type {};
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

    // ── generateScored 가 있을 때 (기본값: 15x15 프리스타일 튜닝 결과, 2026-11 1차) ──
    int  orderPolicyW  = 4;     // 정렬 점수 = 히스토리 + orderPolicyW × 정책(로짓×256) / 256 × 64
    bool useRPS        = true;  // 실현 확률 깊이
    int  rpsPerLogit   = 6;     // 로짓 1 차이당 추가 소모 깊이 (ONE_PLY 단위의 1/4 = 1)
    int  rpsMaxExtra   = 8;     // 추가 소모 상한 (단위)
    int  forcedCost    = 2;     // 수가 하나뿐일 때 소모 깊이 (ONE_PLY 보다 작으면 사실상 연장)
    bool usePolicyCut  = true;
    int  cutGap        = 1;     // 정책 1위와 로짓 차이가 이보다 크면 비전술 수는 생략
    int  cutMinKeep    = 3;     // 최소 이만큼은 탐색
    // ── 남은 깊이별 비전술 수 예산: 남은 깊이 d 에서 keepBase + keepPerPly·d 개까지 (0 이면 끔) ──
    int  keepBase      = 3;
    int  keepPerPly    = 2;
    // ── 정적 문턱: 비PV, 남은 깊이 ≤ gateDepth 에서 정적평가 + gateMargin·d ≤ alpha 면 첫 수 뒤 비전술 수 생략 ──
    int  gateDepth     = 2;     // 0 이면 끔
    int  gateMargin    = 150;
    // ── 로그형 늦은 수 감축: r = lmrBase + ln(d)·ln(i)/lmrDiv (×ONE_PLY, 0 이면 끔) ──
    int  lmrBase100    = 50;    // ×0.01 수
    int  lmrDiv100     = 250;   // ×0.01
    bool useIIR        = true;  // TT 수 없는 깊은 노드는 한 수 얕게
    bool useImproving  = true;  // 2수 전보다 정적평가가 나빠졌으면 더 줄이고 더 자른다
    bool useCounter    = true;  // 상대 직전 수에 대한 반박수 기억
    // ── mustDefend/forcingMoves 가 있을 때 ──
    bool useQuiesce    = true;
    int  qMaxPly       = 4;     // A/B 1000판: 4 가 6 보다 +24 Elo (12ms)
    int  softTimePct   = 50;    // 경과 시간이 한도의 이 비율을 넘으면 다음 반복을 시작하지 않음
    bool useVCF        = false; // (기본 끔: 같은 시간 대국에서 이득 없음, 비용 ~40%) vcfProbe 가 있으면 정지 탐색에서 기준값 전에 연속 4 강제승 판정
    int  qThreePly     = 0;     // 정지 탐색 앞쪽 몇 수까지 열린 3 공격을 포함할지
};

struct Result {
    int      bestMove = NO_MOVE;
    int      score    = 0;
    int      depth    = 0;      // 완료된 마지막 반복 깊이
    int      selDepth = 0;      // 실제로 도달한 최대 수 깊이 (정지 탐색 포함)
    uint64_t interior = 0;      // 수를 생성한 내부 노드 수
    uint64_t searched = 0;      // 내부 노드에서 실제로 탐색한 자식 수 (searched/interior = 실효 분기)
    uint64_t cutFirst = 0, cutTotal = 0;   // 베타 컷 중 첫 수에서 난 비율 = 정렬 품질
    uint64_t qnodes = 0;                   // 정지 탐색 노드 수
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
    void buildLmr() {
        for (int d = 0; d < 64; ++d) for (int i = 0; i < 64; ++i)
            lmrTab_[d][i] = (d < 1 || i < 1 || cfg.lmrDiv100 <= 0) ? 0
                : std::max(0, int(ONE_PLY * (cfg.lmrBase100 / 100.0 + std::log(double(d)) * std::log(double(i)) * 100.0 / cfg.lmrDiv100)));
    }

    void stop() { stopFlag_.store(true, std::memory_order_relaxed); }  // 외부 중단
    void clearHeuristics() {
        std::memset(history_, 0, sizeof(history_));
        for (auto& sd : counter_) for (int& x : sd) x = NO_MOVE;
        for (auto& k : killers_) k[0] = k[1] = NO_MOVE;
    }

    Result think(P& pos, const Limits& lim) {
        lim_ = lim; nodes_ = 0; stopped_ = false; selDepth_ = 0; interior_ = 0; searched_ = 0; cutFirst_ = cutTotal_ = 0; qnodes_ = 0;
        stopFlag_.store(false, std::memory_order_relaxed);
        start_ = Clock::now();
        tt.newSearch();
        buildLmr();
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
                s = search(pos, alpha, beta, d * ONE_PLY, 0);
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
            res.nodes = nodes_; res.timeMs = elapsedMs(); res.selDepth = selDepth_;
            res.interior = interior_; res.searched = searched_; res.cutFirst = cutFirst_; res.cutTotal = cutTotal_; res.qnodes = qnodes_;
            if (onIteration) onIteration(res);

            // 강제승/패가 확정되고 그 수순 길이만큼 읽었으면 종료
            if (std::abs(s) >= MATE_BOUND && d >= MATE - std::abs(s)) break;
            // 소프트 리밋: 다음 반복을 끝낼 가망이 없으면 시작하지 않음
            if (lim.timeMs >= 0 && elapsedMs() * 100 > lim.timeMs * cfg.softTimePct) break;
        }
        res.nodes = nodes_; res.timeMs = elapsedMs(); res.selDepth = std::max(res.selDepth, selDepth_);
        res.interior = interior_; res.searched = searched_; res.cutFirst = cutFirst_; res.cutTotal = cutTotal_; res.qnodes = qnodes_;
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
        if (ply > selDepth_) selDepth_ = ply;

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
            if constexpr (detail::HasQuiesce<P>::value)
                if (cfg.useQuiesce && ply < MAX_PLY - 1) return qsearch(pos, alpha, beta, ply, 0);
            if constexpr (detail::HasLeafProbe<P>::value) {
                int s;
                if (pos.leafProbe(ply, s)) return s;     // 정확한 전술 판정
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

        // 내부 반복 감축: 깊은 노드에서 TT 수가 없으면 정렬 정보가 부족 → 한 수 얕게 (싸게 TT 수 확보)
        if constexpr (detail::HasScored<P>::value)
            if (cfg.useIIR && cfg.useTT && !root && ttMove == NO_MOVE && depth >= 4 * ONE_PLY) depth -= ONE_PLY;

        // ── 수 생성 및 정렬 점수 ──
        int moves[P::NUM_MOVES];
        int order[P::NUM_MOVES];
        int pscore[P::NUM_MOVES];
        uint8_t tac[P::NUM_MOVES];
        constexpr bool scored = detail::HasScored<P>::value;
        int n, smax = 0;
        if constexpr (scored) {
            n = pos.generateScored(moves, pscore, tac);
            if (n > 0) { smax = pscore[0]; for (int i = 1; i < n; ++i) smax = std::max(smax, pscore[i]); }
        } else {
            n = pos.generate(moves);
        }
        if (n == 0) return 0;
        ++interior_;

        // 정적 문턱 (비PV, 얕은 깊이): 정적평가로도 alpha 에 못 미치면 조용한 수는 첫 수 뒤 생략
        bool gate = false, improving = true;
        if constexpr (scored) {
            const int dPly = depth / ONE_PLY;
            const int se = pos.evaluate();
            evalStack_[ply] = se;
            if (cfg.useImproving && ply >= 2) improving = se > evalStack_[ply - 2];
            if (!pvNode && !root && cfg.gateDepth > 0 && dPly <= cfg.gateDepth && std::abs(alpha) < MATE_BOUND)
                gate = se + cfg.gateMargin * std::max(dPly, 1) / (improving ? 1 : 2) <= alpha;
        }
        int quietSearched = 0;
        const int keepLimit = cfg.keepBase + cfg.keepPerPly * (depth / ONE_PLY) + (pvNode ? cfg.keepBase : 0) - (improving ? 0 : 1);
        const int prevMove = ply > 0 ? moveStack_[ply - 1] : NO_MOVE;
        const int counterMove = (cfg.useCounter && prevMove >= 0) ? counter_[ply & 1][prevMove] : NO_MOVE;

        const int side = ply & 1;
        for (int i = 0; i < n; ++i) {
            const int m = moves[i];
            int base = history_[side][m];
            if constexpr (scored) base += cfg.orderPolicyW * pscore[i] / 4;
            order[i] = m == ttMove            ? (1 << 30)
                     : m == killers_[ply][0]  ? (1 << 29)
                     : m == killers_[ply][1]  ? (1 << 29) - 1
                     : m == counterMove       ? (1 << 29) - 2
                     : std::min(base, (1 << 28));
        }

        const int alpha0 = alpha;
        int best = -INF, bestMove = NO_MOVE;

        for (int i = 0; i < n; ++i) {
            // 지연 선택 정렬 (동점이면 생성 순서 유지)
            int bi = i;
            for (int j = i + 1; j < n; ++j) if (order[j] > order[bi]) bi = j;
            const int m = moves[bi], mOrder = order[bi];
            int mScore = 0; uint8_t mTac = 0;
            if constexpr (scored) {
                mScore = pscore[bi]; mTac = tac[bi];
                for (int j = bi; j > i; --j) { pscore[j] = pscore[j - 1]; tac[j] = tac[j - 1]; }
                pscore[i] = mScore; tac[i] = mTac;
            }
            for (int j = bi; j > i; --j) { moves[j] = moves[j - 1]; order[j] = order[j - 1]; }
            moves[i] = m; order[i] = mOrder;

            // 이 수가 소모할 깊이 (기본 1수, 실현 확률 깊이면 정책 차이만큼 추가)
            int extra = 0, cost = ONE_PLY;
            if constexpr (scored) {
                const int gap = (smax - mScore) / 256;          // 로짓 차이 (정수부)
                if (n == 1) cost = cfg.forcedCost;
                else if (!mTac && mOrder < (1 << 29) - 1) {
                    if (cfg.usePolicyCut && !root && i >= cfg.cutMinKeep && gap > cfg.cutGap) continue;
                    if (!root && i > 0 && gate) continue;                                   // 정적 문턱
                    if (!root && cfg.keepBase > 0 && quietSearched >= keepLimit) continue;  // 깊이별 예산
                    ++quietSearched;
                    if (cfg.useRPS) extra = std::min(cfg.rpsMaxExtra, (smax - mScore) * cfg.rpsPerLogit / 256);
                    // 로그형 늦은 수 감축 (깊이 3 이상, 앞 2수 제외): PV 는 덜, 개선 없으면 더
                    if (i >= 2 && depth >= 3 * ONE_PLY) {
                        int r = lmrTab_[std::min(depth / ONE_PLY, 63)][std::min(i, 63)];
                        if (pvNode) r = r * 2 / 3;
                        if (!improving) r += ONE_PLY / 2;
                        extra = std::max(extra, std::min(r, depth - ONE_PLY));
                    }
                }
            }

            pos.make(m);
            ++searched_;
            moveStack_[ply] = m;
            int s;
            if (i == 0) {
                s = -search(pos, -beta, -alpha, depth - cost - extra, ply + 1);
                if (extra && s > alpha) s = -search(pos, -beta, -alpha, depth - cost, ply + 1);
            } else {
                int r = 0;
                if (cfg.useLMR && depth >= 3 * ONE_PLY && i >= 4 && mOrder < (1 << 29) - 1)
                    r = (1 + (i >= 12)) * ONE_PLY;
                r = std::max(r, extra);
                s = -search(pos, -alpha - 1, -alpha, depth - cost - r, ply + 1);      // 널 윈도우
                if (r && s > alpha)
                    s = -search(pos, -alpha - 1, -alpha, depth - cost, ply + 1);      // 감축 없이 재탐색
                if (s > alpha && s < beta)
                    s = -search(pos, -beta, -alpha, depth - cost, ply + 1);           // 풀 윈도우 재탐색
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
                        ++cutTotal_; cutFirst_ += (i == 0);
                        if (cfg.useCounter && prevMove >= 0) counter_[ply & 1][prevMove] = m;
                        if (killers_[ply][0] != m) { killers_[ply][1] = killers_[ply][0]; killers_[ply][0] = m; }
                        int& h = history_[side][m];
                        h += (depth * depth) / (ONE_PLY * ONE_PLY) + 1;
                        if (h > (1 << 28)) for (auto& sd : history_) for (int& x : sd) x /= 2;
                        break;
                    }
                }
            }
        }

        if (cfg.useTT) {
            const Bound b = best >= beta ? BOUND_LOWER : best > alpha0 ? BOUND_EXACT : BOUND_UPPER;
            tt.store(key, scoreToTT(best, ply), bestMove, std::min(depth, 127), b);
        }
        return best;
    }

    // 정지 탐색: stand-pat + 강제수(4)와 강제 방어만
    int qsearch(P& pos, int alpha, int beta, int ply, int qply) {
        pvLen_[ply] = 0;
        if (checkStop()) return 0;
        ++nodes_; ++qnodes_;
        if (ply > selDepth_) selDepth_ = ply;
        if (pos.isLoss()) return -MATE + ply;
        if (pos.isDraw()) return 0;
        if constexpr (detail::HasLeafProbe<P>::value) {
            int s;
            if (pos.leafProbe(ply, s)) return s;
        }
        if (ply >= MAX_PLY - 1) return pos.evaluate();
        int moves[P::NUM_MOVES], n, best;
        const bool defend = pos.mustDefend();
        if (defend) {
            if (qply >= cfg.qMaxPly) return pos.evaluate();
            n = pos.generate(moves);            // 방어수(와 맞받기 4)만 나온다
            best = -INF;
            if (n == 0) return -(MATE - ply - 3);
        } else {
            if constexpr (detail::HasVCF<P>::value) {
                int vs;
                if (cfg.useVCF && qply == 0 && pos.vcfProbe(ply, vs)) return vs;   // 정지 탐색 진입점에서만
            }
            const int stand = pos.evaluate();
            if (stand >= beta || qply >= cfg.qMaxPly) return stand;
            if (stand > alpha) alpha = stand;
            best = stand;
            n = pos.forcingMoves(moves, qply < cfg.qThreePly);
        }
        for (int i = 0; i < n; ++i) {
            pos.make(moves[i]);
            const int s = -qsearch(pos, -beta, -alpha, ply + 1, qply + 1);
            pos.undo(moves[i]);
            if (stopped_) return 0;
            if (s > best) {
                best = s;
                if (s > alpha) {
                    alpha = s;
                    pv_[ply][0] = moves[i];
                    std::memcpy(pv_[ply] + 1, pv_[ply + 1], pvLen_[ply + 1] * sizeof(int));
                    pvLen_[ply] = pvLen_[ply + 1] + 1;
                    if (s >= beta) break;
                }
            }
        }
        return best;
    }

    Limits            lim_;
    uint64_t          nodes_   = 0;
    int               selDepth_ = 0;
    uint64_t          interior_ = 0, searched_ = 0, cutFirst_ = 0, cutTotal_ = 0, qnodes_ = 0;
    int               moveStack_[MAX_PLY + 2];          // ply 에서 둔 수
    int               evalStack_[MAX_PLY + 2];          // ply 의 정적평가 (improving 판정)
    int               counter_[2][P::NUM_MOVES];        // [둘 차례][상대 직전 수] → 반박수
    int               lmrTab_[64][64];
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
