// alphabeta.hpp — 싱글코어 알파-베타 탐색 코어 (헤더 온리, C++17)
//
// 기본 구성: 반복 심화 + 어스퍼레이션 윈도우 + PVS(negamax, fail-soft) + 전치표
//            + 킬러/히스토리 수 정렬 + 메이트 거리 가지치기 + 시간/노드 제한
//
// Position 타입 P 의 필수 인터페이스 (수 = int, 0 <= m < P::NUM_MOVES):
//   static constexpr int NUM_MOVES;
//   uint64_t key() const;             // 조브리스트 해시 (둘 차례 포함)
//   int  generate(int* out) const;    // 후보 수 (좋은 수가 앞이면 히스토리 동점 타이브레이커가 된다)
//   void make(int m);  void undo(int m);
//   bool isLoss() const;              // 둘 차례가 이미 졌다 (직전 수로 상대가 승리)
//   bool isDraw() const;
//   int  evaluate() const;            // 둘 차례 기준 정적 평가, |값| < MATE_BOUND
//
// 선택 인터페이스 (있으면 자동 사용 — 없는 타입은 위 기본 구성 그대로 동작한다):
//   bool leafProbe(int ply, int& score) const;
//       정확히 판정되는 전술 상황(즉승, 막을 수 없는 쌍사)이면 true 와 루트 기준 메이트 점수.
//   int  generateScored(int* out, int* score, uint8_t* tactical) const;  bool quietNode() const;
//       정렬하지 않은 후보 + 정책 점수(로짓×256) + 전술 바이트(최상위 비트 = 전술 수, 하위 7비트 = 전술 등급).
//       있으면 선택 탐색을 켠다:
//         · 수 정렬: TT → 킬러 → 반박수 → 히스토리 + 정책
//         · 실현 확률 깊이: 정책 1위와의 로짓 차이만큼 깊이를 더 소모 + 로그형 늦은 수 감축 (가산)
//         · 정책 간격 컷, 남은 깊이별 조용한 수 예산, 정적 문턱 (비전술 수만)
//         · 역방향 정적 가지치기 (quietNode 일 때), 내부 반복 감축, improving 신호
//   bool mustDefend() const;  int forcingMoves(int* out) const;
//       있으면 리프에서 정지 탐색: 위협이 없으면 정적평가를 기준값으로 4 만들기만, 상대 4 가 있으면 막기만.
//
// 깊이는 내부적으로 1수 = ONE_PLY 단위의 정수다.
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
constexpr int ONE_PLY    = 4;

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

namespace detail {
template <class P, class = void> struct HasLeafProbe : std::false_type {};
template <class P>
struct HasLeafProbe<P, std::void_t<decltype(std::declval<const P&>().leafProbe(0, std::declval<int&>()))>> : std::true_type {};
template <class P, class = void> struct HasScored : std::false_type {};
template <class P>
struct HasScored<P, std::void_t<decltype(std::declval<const P&>().generateScored((int*)nullptr, (int*)nullptr, (uint8_t*)nullptr)),
                                decltype(std::declval<const P&>().quietNode())>> : std::true_type {};
template <class P, class = void> struct HasQuiesce : std::false_type {};
template <class P>
struct HasQuiesce<P, std::void_t<decltype(std::declval<const P&>().forcingMoves((int*)nullptr)),
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

    // 직접 사상. 같은 키면 갱신(훨씬 깊은 기존 정보는 최선수만 갱신), 다른 키면 같은 세대의 훨씬 깊은 엔트리만 지킨다
    void store(uint64_t key, int score, int move, int depth, Bound bound) {
        TTEntry& e = table_[key & mask_];
        if (e.bound != BOUND_NONE && e.key == key) {
            if (move == NO_MOVE) move = e.move;               // 기존 최선수 보존
            if (bound != BOUND_EXACT && depth + 2 < e.depth) { e.move = int16_t(move); e.gen = gen_; return; }
        } else if (e.bound != BOUND_NONE && e.gen == gen_ && e.depth > depth + 2) {
            return;
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
    int64_t  softMs   = -1;     // ≥0: 목표 시간. 반복 사이에 최선수 안정성·점수 하락으로 늘리고 줄인다 (<0: timeMs·softTimePct 고정 규칙)
};

// 기본값은 15x15 프리스타일 튜닝 결과 (1,000판 A/B 기준). 이름은 engine/params.hpp 의 키와 같다.
struct Config {
    // 기본
    bool useTT         = true;
    bool useAspiration = true;
    int  aspDelta      = 40;
    int  softTimePct   = 75;    // 경과 시간이 한도의 이 비율을 넘으면 다음 반복을 시작하지 않음
    // 실전 시간 배분 (planTime, Limits::softMs 를 쓸 때)
    bool tmDynamic     = false; // 남은 시간/tmDiv 를 목표로, 강제 수는 즉시·최선수 불안정하면 더·안정하면 덜 (끄면: min(수당, 남은/15) 고정)
    int  tmDiv         = 20;
    int  tmHardDiv     = 4;     // 한 수에 남은 시간의 1/tmHardDiv 이상은 쓰지 않음
    // 수 정렬 (generateScored 가 있을 때)
    int  orderPolicyW  = 4;     // 정렬 점수 = 히스토리 + orderPolicyW × 정책점수 / 4
    bool useCounter    = true;  // 상대 직전 수에 대한 반박수
    // 깊이 소모
    int  rpsPerLogit   = 4;     // 실현 확률 깊이: 정책 1위와 로짓 1 차이당 추가 소모 (ONE_PLY 의 1/4 단위) — 6→4: +13, +30 Elo (3, 5 는 +5)
    int  rpsMaxExtra   = 8;
    int  lmrDiv100     = 100;   // 로그형 늦은 수 감축 ln(d)·ln(i)·100/lmrDiv100 수 (0 = 끔), 실현 확률 감축에 더함
    int  tacCost       = 2;     // 전술 수(4·쌍삼 이상 만들기/막기)가 소모하는 깊이 — 반 수 연장 (A/B: +19, +22 Elo)
    int  forcedCost    = 0;     // 수가 하나뿐일 때(강제 응수) 소모 깊이 — 0 = 완전 연장 (A/B: 2→0 +23 Elo)
    bool useIIR        = true;  // TT 수 없는 깊은 노드는 한 수 얕게
    bool useImproving  = true;  // 2수 전보다 정적평가가 나빠졌으면 더 줄이고 더 자른다
    // 가지치기 (비전술 수만)
    bool usePolicyCut  = true;
    int  cutGap        = 1;     // 정책 1위와 로짓 차이가 이보다 크면 생략 (루트 제외)
    int  cutMinKeep    = 3;     // 최소 이만큼은 탐색
    int  keepBase      = 3;     // 남은 깊이 d 에서 조용한 수 keepBase + keepPerPly·d 개까지 (0 = 끔)
    int  keepPerPly    = 2;
    int  gateDepth     = 2;     // 정적 문턱: 비PV·d ≤ gateDepth 에서 정적평가 + gateMargin·d ≤ alpha 면 첫 수 뒤 생략 (0 = 끔)
    int  gateMargin    = 150;
    int  rfpDepth      = 5;     // 역방향 정적 가지치기: 비PV·위협 없음·d ≤ rfpDepth 에서
    int  rfpMargin     = 60;    //   정적평가 - rfpMargin·(d - improving) ≥ beta 면 즉시 반환 (0 = 끔)
    int  razorDepth    = 2;     // 레이저링 (A/B 2,000판: d2 ≈ +15 Elo): 비PV·d ≤ razorDepth 에서 정적평가 + razorMargin·d < alpha 면 정지 탐색으로 확인
    int  razorMargin   = 200;
    // 정지 탐색 (mustDefend/forcingMoves 가 있을 때)
    bool useQuiesce    = true;
    int  qMaxPly       = 4;
    bool qTT           = true;  // 정지 탐색에서도 전치표 조회·저장 (수가 있는 노드만, 깊이 0 으로) — A/B +18, +14 Elo
};

struct Result {
    int      bestMove = NO_MOVE;
    int      score    = 0;
    int      depth    = 0;      // 완료된 마지막 반복 깊이
    int      selDepth = 0;      // 실제로 도달한 최대 수 깊이 (정지 탐색 포함)
    uint64_t nodes    = 0;
    int64_t  timeMs   = 0;
    std::vector<int> pv;
    // 진단 통계
    uint64_t interior = 0;      // 수를 생성한 내부 노드
    uint64_t searched = 0;      // 내부 노드에서 탐색한 자식 (searched/interior = 실효 분기)
    uint64_t cutFirst = 0, cutTotal = 0;   // 베타 컷 중 첫 수에서 난 비율 = 정렬 품질
    uint64_t qnodes   = 0;      // 정지 탐색 노드
};

// 실전 시간 배분: 수당 제한 turnMs, 남은 대국 시간 leftMs(<0 이면 없음) → Limits 의 하드/소프트 시간
inline void planTime(const Config& c, int64_t turnMs, int64_t leftMs, Limits& lim) {
    if (turnMs <= 0) turnMs = 1000;
    const auto margin = [](int64_t t) { return std::max<int64_t>(20, t - std::min<int64_t>(60, t / 10)); };   // 통신 여유
    const auto cap = [&](int64_t t) { return leftMs >= 0 ? std::min<int64_t>(t, std::max<int64_t>(1, leftMs / 4)) : t; };  // 시간 부족 시 안전장치
    if (!c.tmDynamic) {
        lim.timeMs = cap(margin(leftMs >= 0 ? std::min<int64_t>(turnMs, leftMs / 15) : turnMs));
        lim.softMs = -1;
        return;
    }
    lim.timeMs = margin(leftMs >= 0 ? std::min<int64_t>(turnMs, leftMs / std::max(1, c.tmHardDiv)) : turnMs);
    lim.softMs = std::min<int64_t>(lim.timeMs, leftMs >= 0 ? leftMs / std::max(1, c.tmDiv) : lim.timeMs);
    lim.timeMs = cap(std::min<int64_t>(lim.timeMs, std::max<int64_t>(20, lim.softMs * 3)));   // 반복 중간이라도 목표의 3배에서 끊는다
}

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
        for (auto& sd : counter_) for (int& x : sd) x = NO_MOVE;
        for (auto& k : killers_) k[0] = k[1] = NO_MOVE;
    }

    Result think(P& pos, const Limits& lim) {
        lim_ = lim; nodes_ = 0; stopped_ = false; selDepth_ = 0;
        interior_ = searched_ = cutFirst_ = cutTotal_ = qnodes_ = 0;
        stopFlag_.store(false, std::memory_order_relaxed);
        start_ = Clock::now();
        tt.newSearch();
        buildLmr();
        for (auto& k : killers_) k[0] = k[1] = NO_MOVE;
        for (auto& side : history_) for (int& h : side) h /= 2;   // 히스토리 에이징

        Result res;
        int rootMoves[P::NUM_MOVES];
        const int rootN = (pos.isLoss() || pos.isDraw()) ? 0 : pos.generate(rootMoves);
        if (rootN == 0) return res;
        int lastBest = NO_MOVE, stable = 0;
        rootBest_ = rootMoves[0];             // 어떤 경우에도 합법수 반환 보장
        res.bestMove = rootBest_;

        int prev = 0, prevIter = 0;
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
            fillStats(res);
            if (onIteration) onIteration(res);

            // 강제승/패가 확정되고 그 수순 길이만큼 읽었으면 종료
            if (std::abs(s) >= MATE_BOUND && d >= MATE - std::abs(s)) break;
            // 소프트 리밋: 다음 반복을 끝낼 가망이 없으면 시작하지 않음
            if (lim.softMs >= 0) {
                if (rootN == 1) break;                                   // 강제 수
                stable = res.bestMove == lastBest ? stable + 1 : 0;
                lastBest = res.bestMove;
                int f = stable == 0 ? 150 : stable >= 3 ? 60 : 100;      // 최선수가 바뀌었으면 더, 오래 그대로면 덜
                if (d >= 4 && s < prevIter - 30) f = f * 13 / 10;        // 점수 하락 중이면 더
                if (elapsedMs() * 10000 > lim.softMs * f * cfg.softTimePct) break;
            } else if (lim.timeMs >= 0 && elapsedMs() * 100 > lim.timeMs * cfg.softTimePct) break;
            prevIter = s;
        }
        const int sel = res.selDepth;
        fillStats(res);
        res.selDepth = std::max(sel, selDepth_);
        return res;
    }

private:
    using Clock = std::chrono::steady_clock;

    int64_t elapsedMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
    }
    void fillStats(Result& r) const {
        r.nodes = nodes_; r.timeMs = elapsedMs(); r.selDepth = selDepth_;
        r.interior = interior_; r.searched = searched_; r.cutFirst = cutFirst_; r.cutTotal = cutTotal_; r.qnodes = qnodes_;
    }
    void buildLmr() {
        for (int d = 0; d < 64; ++d) for (int i = 0; i < 64; ++i)
            lmrTab_[d][i] = (d < 1 || i < 1 || cfg.lmrDiv100 <= 0) ? 0
                : std::max(0, int(ONE_PLY * (std::log(double(d)) * std::log(double(i)) * 100.0 / cfg.lmrDiv100)));
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
        constexpr bool scored = detail::HasScored<P>::value;
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
                if (pos.leafProbe(ply, s)) return s;
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

        // ── 선택 탐색 준비: 내부 반복 감축, 정적평가, improving, 역방향 정적 가지치기 ──
        bool improving = true;
        int se = 0;
        if constexpr (scored) {
            if (cfg.useIIR && cfg.useTT && !root && ttMove == NO_MOVE && depth >= 4 * ONE_PLY) depth -= ONE_PLY;
            se = pos.evaluate();
            evalStack_[ply] = se;
            if (cfg.useImproving && ply >= 2) improving = se > evalStack_[ply - 2];
            const int dPly = depth / ONE_PLY;
            if (cfg.rfpDepth > 0 && !pvNode && !root && pos.quietNode() && dPly <= cfg.rfpDepth &&
                std::abs(beta) < MATE_BOUND && se - cfg.rfpMargin * std::max(dPly - (improving ? 1 : 0), 0) >= beta)
                return se;
            if constexpr (detail::HasQuiesce<P>::value)
                if (cfg.razorDepth > 0 && !pvNode && !root && pos.quietNode() && dPly <= cfg.razorDepth &&
                    std::abs(alpha) < MATE_BOUND && se + cfg.razorMargin * std::max(dPly, 1) < alpha) {
                    const int v = qsearch(pos, alpha, alpha + 1, ply, 0);
                    if (v <= alpha) return v;
                }
        }

        // ── 수 생성 ──
        int moves[P::NUM_MOVES];
        int order[P::NUM_MOVES];
        int pscore[P::NUM_MOVES];
        uint8_t tac[P::NUM_MOVES];
        int n, smax = 0;
        if constexpr (scored) {
            n = pos.generateScored(moves, pscore, tac);
            if (n > 0) { smax = pscore[0]; for (int i = 1; i < n; ++i) smax = std::max(smax, pscore[i]); }
        } else {
            n = pos.generate(moves);
        }
        if (n == 0) return 0;
        ++interior_;

        // 노드 단위 가지치기 조건
        bool gate = false;                               // 정적 문턱
        int quietSearched = 0, keepLimit = 0;            // 남은 깊이별 조용한 수 예산
        if constexpr (scored) {
            const int dPly = depth / ONE_PLY;
            if (!pvNode && !root && cfg.gateDepth > 0 && dPly <= cfg.gateDepth && std::abs(alpha) < MATE_BOUND)
                gate = se + cfg.gateMargin * std::max(dPly, 1) / (improving ? 1 : 2) <= alpha;
            keepLimit = cfg.keepBase + cfg.keepPerPly * dPly + (pvNode ? cfg.keepBase : 0) - (improving ? 0 : 1);
        }
        const int prevMove    = ply > 0 ? moveStack_[ply - 1] : NO_MOVE;
        const int counterMove = (cfg.useCounter && prevMove >= 0) ? counter_[ply & 1][prevMove] : NO_MOVE;

        // ── 정렬 점수 ──
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

        bool restSorted = false;
        for (int i = 0; i < n; ++i) {
            // 앞 2수는 지연 선택, 그 뒤는 남은 수를 한 번에 안정 정렬 (어느 쪽이든 "점수 내림차순, 동점이면 생성 순서")
            if (!restSorted && i == 2 && n - i > 4) { sortRest(moves, order, pscore, tac, i, n); restSorted = true; }
            int bi = i;
            if (!restSorted) for (int j = i + 1; j < n; ++j) if (order[j] > order[bi]) bi = j;
            const int m = moves[bi], mOrder = order[bi];
            int mScore = 0; uint8_t mTac = 0;
            if constexpr (scored) {
                mScore = pscore[bi]; mTac = tac[bi];
                for (int j = bi; j > i; --j) { pscore[j] = pscore[j - 1]; tac[j] = tac[j - 1]; }
                pscore[i] = mScore; tac[i] = mTac;
            }
            for (int j = bi; j > i; --j) { moves[j] = moves[j - 1]; order[j] = order[j - 1]; }
            moves[i] = m; order[i] = mOrder;

            // 이 수가 소모할 깊이와 가지치기 (TT 수·킬러·전술 수는 제외)
            int extra = 0, cost = ONE_PLY;
            if constexpr (scored) {
                if (n == 1) cost = cfg.forcedCost;
                else if (mTac & 128) cost = cfg.tacCost;
                else if (mOrder < (1 << 29) - 1) {
                    if (!root) {
                        if (cfg.usePolicyCut && i >= cfg.cutMinKeep && (smax - mScore) / 256 > cfg.cutGap) continue;
                        if (i > 0 && gate) continue;
                        if (cfg.keepBase > 0 && quietSearched >= keepLimit) continue;
                    }
                    ++quietSearched;
                    extra = std::min(cfg.rpsMaxExtra, (smax - mScore) * cfg.rpsPerLogit / 256);
                    if (i >= 2 && depth >= 3 * ONE_PLY) {        // 로그형 감축: PV 는 덜, 개선 없으면 더
                        int r = lmrTab_[std::min(depth / ONE_PLY, 63)][std::min(i, 63)];
                        if (pvNode) r = r * 2 / 3;
                        if (!improving) r += ONE_PLY / 2;
                        extra = std::min(extra + r, std::max(depth - ONE_PLY, 0));
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
                s = -search(pos, -alpha - 1, -alpha, depth - cost - extra, ply + 1);
                if (extra && s > alpha)
                    s = -search(pos, -alpha - 1, -alpha, depth - cost, ply + 1);      // 감축 없이 재탐색
                if (s > alpha && s < beta)
                    s = -search(pos, -beta, -alpha, depth - cost, ply + 1);          // 풀 윈도우 재탐색
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
                        const int bonus = (depth * depth) / (ONE_PLY * ONE_PLY) + 1;
                        int& h = history_[side][m];
                        h += bonus;
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

    // [i, n) 구간을 order 내림차순으로 안정 정렬 (네 배열을 함께)
    static void sortRest(int* moves, int* order, int* pscore, uint8_t* tac, int i, int n) {
        int idx[P::NUM_MOVES], m2[P::NUM_MOVES], o2[P::NUM_MOVES], p2[P::NUM_MOVES]; uint8_t t2[P::NUM_MOVES];
        const int k = n - i;
        for (int j = 0; j < k; ++j) idx[j] = i + j;
        std::stable_sort(idx, idx + k, [&](int a, int b) { return order[a] > order[b]; });
        for (int j = 0; j < k; ++j) { m2[j] = moves[idx[j]]; o2[j] = order[idx[j]]; p2[j] = pscore[idx[j]]; t2[j] = tac[idx[j]]; }
        for (int j = 0; j < k; ++j) { moves[i + j] = m2[j]; order[i + j] = o2[j]; pscore[i + j] = p2[j]; tac[i + j] = t2[j]; }
    }

    // 정지 탐색: 위협 없음 → 기준값 + 4 만들기만, 상대 4 → 막기만
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
        const int alpha0 = alpha;
        const bool useQTT = cfg.qTT && cfg.useTT;
        auto probeQ = [&](int& out) {
            if (const TTEntry* e = tt.probe(pos.key())) {
                const int s = scoreFromTT(e->score, ply);
                if (e->bound == BOUND_EXACT || (e->bound == BOUND_LOWER && s >= beta) || (e->bound == BOUND_UPPER && s <= alpha)) {
                    out = s; return true;
                }
            }
            return false;
        };
        if (pos.mustDefend()) {
            if (qply >= cfg.qMaxPly) return pos.evaluate();
            if (useQTT) { int s; if (probeQ(s)) return s; }
            n = pos.generate(moves);            // 막는 수만 나온다
            best = -INF;
            if (n == 0) return -(MATE - ply - 3);
        } else {
            int stand = pos.evaluate();
            if (stand >= beta || qply >= cfg.qMaxPly) return stand;
            if (stand > alpha) alpha = stand;
            best = stand;
            n = pos.forcingMoves(moves);
            if (n && useQTT) { int s; if (probeQ(s)) return s; }
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
        if (useQTT && n > 0)
            tt.store(pos.key(), scoreToTT(best, ply), NO_MOVE, 0,
                     best >= beta ? BOUND_LOWER : best > alpha0 ? BOUND_EXACT : BOUND_UPPER);
        return best;
    }

    Limits            lim_;
    uint64_t          nodes_ = 0;
    int               selDepth_ = 0;
    uint64_t          interior_ = 0, searched_ = 0, cutFirst_ = 0, cutTotal_ = 0, qnodes_ = 0;
    bool              stopped_ = false;
    std::atomic<bool> stopFlag_{false};
    Clock::time_point start_;
    int               rootBest_ = NO_MOVE;

    int killers_[MAX_PLY][2];
    int history_[2][P::NUM_MOVES];
    int counter_[2][P::NUM_MOVES];          // [둘 차례][상대 직전 수] → 반박수
    int moveStack_[MAX_PLY + 2];            // ply 에서 둔 수
    int evalStack_[MAX_PLY + 2];            // ply 의 정적평가 (improving)
    int lmrTab_[64][64];
    int pv_[MAX_PLY + 1][MAX_PLY + 1];
    int pvLen_[MAX_PLY + 1];
};

} // namespace ab
