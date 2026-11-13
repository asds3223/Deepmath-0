// oracle.cpp — Rapfi(태그 250615) 소스를 직접 링크한 "정답지" 도구
//
// 모드
//   gen        Rapfi NNUE 정책 샘플링 롤아웃으로 국면을 만들고, NNUE 가치/정책(+선택적 탐색) 라벨을 붙인다.
//              (우리 엔진 자가대국이 아니라 교사 모델이 두는 국면 분포 + 무작위 섭동)
//              --sib-prob 로 일부 국면에서 형제 국면 묶음(상위 정책 수 + 무작위 수)을 함께 기록한다.
//              형제 묶음은 "같은 부모에서 어느 자식이 더 나은가"를 NNUE 와 같게 판단하는지 재는 데 쓴다.
//   label      기존 레코드(예: 우리 탐색 트리 리프 덤프)에 NNUE 가치/정책/탐색 라벨을 다시 붙인다.
//   bench      NNUE 정적 평가·정책·탐색 처리량 측정 (데이터 예산 산정용)
//   match      우리 고전평가 모델 vs Rapfi NNUE 단독(탐색 없음) 대국
//              --ours greedy(1수 정적평가) | search(우리 alphabeta, --ms)
//              --rapfi greedy(1수 NNUE 가치) | policy(NNUE 정책 1순위)
//                      | mixed(Rapfi 탐색 내부와 같은 고전+NNUE 혼합 평가, PVS 식 창으로 1수 읽기)
//              즉승·상대 4 막기·즉패 회피 규칙은 양쪽에 동일 적용
//
// 병렬: 한 프로세스 = 한 코어. 대량 작업은 scripts/ 의 셸 스크립트가 코어 수만큼 프로세스를 띄운다.
#include "../engine/eval.hpp"
#include "../engine/record.hpp"

#include "command/command.h"
#include "config.h"
#include "core/pos.h"
#include "eval/eval.h"
#include "eval/evaluator.h"
#include "game/board.h"
#include "search/searchthread.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double secondsSince(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::map<std::string, std::string> parseArgs(int argc, char** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a[k] = argv[++i];
        else a[k] = "1";
    }
    return a;
}
template <class T> T get(const std::map<std::string, std::string>& a, const char* k, T def) {
    auto it = a.find(k);
    if (it == a.end()) return def;
    if constexpr (std::is_same_v<T, std::string>) return it->second;
    else if constexpr (std::is_floating_point_v<T>) return T(std::stod(it->second));
    else return T(std::stoll(it->second));
}

inline Pos toPos(int m) { return Pos(m % gmk::BS, m / gmk::BS); }

// ── Rapfi 초기화 ──
bool initRapfi(const std::string& configPath, int ttMb) {
    Config::MessageMode = MsgMode::NONE;
    Command::configPath = configPath;
    Command::allowInternalConfig = false;
    if (!Command::loadConfig()) { std::fprintf(stderr, "config load failed: %s\n", configPath.c_str()); return false; }
    Config::MessageMode = MsgMode::NONE;
    Search::Threads.setNumThreads(1);
    Search::Threads.searcher()->setMemoryLimit(size_t(ttMb) * 1024);
    return true;
}

// NNUE 가 붙은 Rapfi 보드를 쥐고, 우리 gmk::Board 와 같이 움직이는 쌍
struct Pair {
    Search::SearchThread* th = nullptr;
    Board*                rb = nullptr;      // 평가자 연결된 Rapfi 보드 (th->board)
    std::unique_ptr<Board> plain;            // 평가자 없는 Rapfi 보드 (Rapfi 고전평가 베이스라인용)
    gmk::Board            gb;

    void reset() {
        th = Search::Threads.main();
        Board empty(gmk::BS, CandidateRange::SQUARE3_LINE4);
        empty.newGame(Rule::FREESTYLE);
        th->setBoardAndEvaluator(empty);
        rb = th->board.get();
        plain = std::make_unique<Board>(gmk::BS, CandidateRange::SQUARE3_LINE4);
        plain->newGame(Rule::FREESTYLE);
        gb.clear();
    }
    void move(int m) {
        rb->move(Rule::FREESTYLE, toPos(m));
        plain->move(Rule::FREESTYLE, toPos(m));
        gb.make(m);
    }
    void undo() {
        rb->undo(Rule::FREESTYLE);
        plain->undo(Rule::FREESTYLE);
        gb.undo(gb.lastMove());
    }
    // 탐색은 스레드 보드를 교체하므로, 끝난 뒤 평가자 보드를 다시 동기화
    void resyncAfterSearch() {
        Board snap(*plain, nullptr);
        th->setBoardAndEvaluator(snap);
        rb = th->board.get();
    }
};

struct NNOut {
    float w, l, d;
    std::vector<std::pair<float, int>> pol;   // (prob, move), 내림차순
};

NNOut nnEval(Pair& p) {
    NNOut o;
    Evaluation::Evaluator* ev = p.rb->evaluator();
    Evaluation::ValueType v = ev->evaluateValue(*p.rb);
    o.w = v.win(); o.l = v.loss(); o.d = v.draw();
    Evaluation::PolicyBuffer pb(gmk::BS);
    pb.setComputeFlagForAllEmptyCell(*p.rb);
    ev->evaluatePolicy(*p.rb, pb);
    pb.applySoftmax(1.0f);
    for (int m = 0; m < gmk::NMOVES; ++m)
        if (p.gb.isEmptyMove(m)) o.pol.push_back({pb(m % gmk::BS, m / gmk::BS), m});
    std::sort(o.pol.begin(), o.pol.end(), [](auto& a, auto& b) { return a.first > b.first; });
    return o;
}

float rapfiClassicalWR(Pair& p) {
    Value v = Evaluation::evaluate(*p.plain, Rule::FREESTYLE);
    return Config::valueToWinRate(v);
}

struct SearchOut { int best; float wr; int depth; };

SearchOut rapfiSearch(Pair& p, uint64_t nodes) {
    Search::SearchOptions opt;
    opt.rule = {Rule::FREESTYLE, GameRule::FREEOPEN};
    opt.infoMode = Search::SearchOptions::INFO_NONE;
    opt.disableOpeningQuery = true;
    opt.maxNodes = nodes;
    opt.maxDepth = 99;
    Search::Threads.clear(false);
    Board snap(*p.plain, nullptr);
    Search::Threads.startThinking(snap, opt, false);
    Search::Threads.waitForIdle();
    const auto& rm = Search::Threads.main()->rootMoves;
    SearchOut s{-1, 0.5f, 0};
    if (!rm.empty()) {
        Pos b = rm[0].pv[0];
        s.best  = b.x() + b.y() * gmk::BS;
        s.wr    = Config::valueToWinRate(rm[0].value);
        s.depth = rm[0].selDepth;
    }
    p.resyncAfterSearch();
    return s;
}

void fillRecord(gmk::Record& r, Pair& p, const NNOut& nn) {
    std::memset(&r, 0, sizeof r);
    r.setBoard(p.gb);
    r.nnW = nn.w; r.nnL = nn.l; r.nnD = nn.d;
    r.rcls = rapfiClassicalWR(p);
    r.srchWR = std::nanf(""); r.srchBest = 255; r.srchDepth = 0;
    const int n = std::min<int>(16, int(nn.pol.size()));
    r.polN = uint8_t(n);
    for (int i = 0; i < n; ++i) { r.polIdx[i] = uint8_t(nn.pol[i].second); r.polProb[i] = uint16_t(std::lround(nn.pol[i].first * 65535)); }
    const int s = p.gb.side - 1, o = 2 - p.gb.side;
    if (p.gb.clsCnt[s][gmk::C_FIVE]) r.flags |= gmk::REC_SELF_FIVE;
    if (p.gb.clsCnt[o][gmk::C_FIVE]) r.flags |= gmk::REC_OPPO_FIVE;
}

// ── gen ──
int cmdGen(const std::map<std::string, std::string>& a) {
    const int      games   = get(a, "games", 100);
    const uint64_t seed    = get<uint64_t>(a, "seed", 1);
    const float    temp    = get(a, "temp", 1.0);
    const float    eps     = get(a, "eps", 0.03);         // 무작위 섭동 수 비율
    const int      openMax = get(a, "open", 6);           // 시작 무작위 수 0..openMax
    const float    sample  = get(a, "sample", 1.0);       // 국면 기록 확률
    const uint64_t nodes   = get<uint64_t>(a, "search-nodes", 0);
    const float    sprob   = get(a, "search-prob", 0.0);
    const int      maxPly  = get(a, "max-ply", 180);
    const std::string out  = get<std::string>(a, "out", "gen.bin");
    const float    sibProb = get(a, "sib-prob", 0.0);     // 형제 묶음 기록 확률
    const int      sibTop  = get(a, "sib-top", 4);        // 정책 상위 수 개수
    const int      sibRand = get(a, "sib-rand", 4);       // 무작위 후보 수 개수
    const float    tempLate = get(a, "temp-late", 0.0);   // >0 이면 오프닝 이후 온도
    uint32_t groupId = uint32_t((seed * 2654435761ull) & 0xffff) << 8;   // 프로세스별 시작값

    gmk::RecordWriter w(out);
    if (!w.ok()) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U(0, 1);
    Pair p;
    uint64_t nrec = 0, nsrch = 0;
    auto t0 = Clock::now();
    for (int g = 0; g < games; ++g) {
        p.reset();
        const int nOpen = int(rng() % (openMax + 1));
        for (int ply = 0; ply < maxPly; ++ply) {
            if (p.gb.isLoss() || p.gb.isDraw()) break;
            int cand[gmk::NMOVES], nc = 0;
            if (ply == 0) {
                for (int y = 5; y <= 9; ++y) for (int x = 5; x <= 9; ++x) cand[nc++] = x + y * gmk::BS;
            } else {
                for (int m = 0; m < gmk::NMOVES; ++m) if (p.gb.isCandidate(gmk::toCell(m))) cand[nc++] = m;
            }
            if (!nc) break;
            int mv;
            if (ply < nOpen) {
                mv = cand[rng() % nc];
            } else {
                NNOut nn = nnEval(p);
                if (U(rng) < sample) {
                    gmk::Record r; fillRecord(r, p, nn);
                    if (!(r.flags & gmk::REC_SELF_FIVE)) {
                        if (nodes && U(rng) < sprob) {
                            SearchOut s = rapfiSearch(p, nodes);
                            r.srchBest = uint8_t(s.best); r.srchWR = s.wr; r.srchDepth = int8_t(std::min(s.depth, 127));
                            r.flags |= gmk::REC_HAS_SEARCH; ++nsrch;
                        }
                        w.write(r); ++nrec;
                    }
                }
                if (sibProb > 0 && U(rng) < sibProb && !(p.gb.clsCnt[p.gb.side - 1][gmk::C_FIVE])) {
                    // 형제 묶음: 정책 상위 sibTop + 무작위 후보 sibRand (중복 제거)
                    std::vector<int> kids;
                    for (int i = 0; i < int(nn.pol.size()) && int(kids.size()) < sibTop; ++i) kids.push_back(nn.pol[i].second);
                    for (int t = 0; t < 4 * sibRand && int(kids.size()) < sibTop + sibRand; ++t) {
                        int m = cand[rng() % nc];
                        if (std::find(kids.begin(), kids.end(), m) == kids.end()) kids.push_back(m);
                    }
                    groupId = (groupId + 1) & 0xffffff; if (!groupId) groupId = 1;
                    for (int m : kids) {
                        p.move(m);
                        if (!p.gb.isLoss()) {
                            NNOut cn = nnEval(p);
                            gmk::Record cr; fillRecord(cr, p, cn);
                            if (!(cr.flags & gmk::REC_SELF_FIVE)) {
                                cr.flags |= gmk::REC_SIBLING; cr.setGroup(groupId);
                                w.write(cr); ++nrec;
                            }
                        }
                        p.undo();
                    }
                }
                if (U(rng) < eps) mv = cand[rng() % nc];
                else {
                    // 온도 적용 정책 샘플링
                    double tot = 0; std::vector<double> pr(nn.pol.size());
                    const double tt = (tempLate > 0 && ply >= nOpen + 4) ? tempLate : temp;
                    for (size_t i = 0; i < nn.pol.size(); ++i) tot += pr[i] = std::pow(std::max(nn.pol[i].first, 1e-12f), 1.0 / tt);
                    double r = U(rng) * tot; mv = nn.pol[0].second;
                    for (size_t i = 0; i < pr.size(); ++i) { r -= pr[i]; if (r <= 0) { mv = nn.pol[i].second; break; } }
                }
            }
            p.move(mv);
        }
        if ((g + 1) % 20 == 0 || g + 1 == games)
            std::fprintf(stderr, "[gen] games %d/%d records %llu search %llu  %.1fs\n", g + 1, games,
                         (unsigned long long)nrec, (unsigned long long)nsrch, secondsSince(t0));
    }
    return 0;
}

// ── label ──
int cmdLabel(const std::map<std::string, std::string>& a) {
    const std::string in = get<std::string>(a, "in", ""), out = get<std::string>(a, "out", "label.bin");
    const uint64_t nodes = get<uint64_t>(a, "search-nodes", 0);
    std::vector<gmk::Record> recs;
    if (!gmk::readRecords(in, recs)) { std::fprintf(stderr, "cannot read %s\n", in.c_str()); return 1; }
    gmk::RecordWriter w(out);
    Pair p;
    auto t0 = Clock::now();
    for (size_t i = 0; i < recs.size(); ++i) {
        const gmk::Record& src = recs[i];
        uint8_t arr[gmk::NMOVES]; src.unpack(arr);
        // 흑/백 돌을 번갈아 놓아 같은 배치를 만든다. 직전 2수는 맨 끝에 둔다.
        std::vector<int> bl, wh;
        int l1 = src.last1 == 255 ? -1 : src.last1, l2 = src.last2 == 255 ? -1 : src.last2;
        for (int m = 0; m < gmk::NMOVES; ++m) {
            if (m == l1 || m == l2) continue;
            if (arr[m] == gmk::BLACK) bl.push_back(m); else if (arr[m] == gmk::WHITE) wh.push_back(m);
        }
        auto pushLast = [&](int m) { if (m >= 0) (arr[m] == gmk::BLACK ? bl : wh).push_back(m); };
        pushLast(l2); pushLast(l1);
        if (!(bl.size() == wh.size() || bl.size() == wh.size() + 1)) continue;
        p.reset();
        bool okSeq = true;
        for (size_t k = 0; k < bl.size() + wh.size(); ++k) {
            int m = (k % 2 == 0) ? bl[k / 2] : wh[k / 2];
            if (p.gb.isLoss()) { okSeq = false; break; }
            p.move(m);
        }
        if (!okSeq || int(p.gb.side) != src.side) continue;
        NNOut nn = nnEval(p);
        gmk::Record r; fillRecord(r, p, nn);
        r.flags |= (src.flags & gmk::REC_LEAF);
        if (r.flags & gmk::REC_SELF_FIVE) continue;
        if (nodes) {
            SearchOut s = rapfiSearch(p, nodes);
            r.srchBest = uint8_t(s.best); r.srchWR = s.wr; r.srchDepth = int8_t(std::min(s.depth, 127));
            r.flags |= gmk::REC_HAS_SEARCH;
        }
        w.write(r);
        if ((i + 1) % 1000 == 0) std::fprintf(stderr, "[label] %zu/%zu %.1fs\n", i + 1, recs.size(), secondsSince(t0));
    }
    return 0;
}

// ── bench ──
int cmdBench(const std::map<std::string, std::string>& a) {
    const uint64_t nodes = get<uint64_t>(a, "search-nodes", 50000);
    const int searches = get(a, "searches", 5);
    std::mt19937_64 rng(3);
    Pair p; p.reset();
    // 중반 국면 하나를 만든다
    int seq[] = {112, 113, 97, 127, 98, 96, 128, 82, 111, 126, 99, 100, 142, 84};
    for (int m : seq) p.move(m);
    auto t0 = Clock::now(); int n = 0; float acc = 0;
    while (secondsSince(t0) < 2.0) {
        for (int i = 0; i < 100; ++i) acc += p.rb->evaluator()->evaluateValue(*p.rb).win();
        n += 100;
    }
    const double vps = n / secondsSince(t0);
    t0 = Clock::now(); n = 0;
    while (secondsSince(t0) < 2.0) { NNOut o = nnEval(p); acc += o.w; ++n; }
    const double pps = n / secondsSince(t0);
    t0 = Clock::now(); n = 0;
    while (secondsSince(t0) < 2.0) { acc += rapfiClassicalWR(p); ++n; }
    const double cps = n / secondsSince(t0);
    t0 = Clock::now();
    for (int i = 0; i < searches; ++i) rapfiSearch(p, nodes);
    const double sps = searches / secondsSince(t0);
    std::printf("NNUE value only : %.0f /s\n", vps);
    std::printf("NNUE value+policy (record label): %.0f /s\n", pps);
    std::printf("Rapfi classical eval: %.0f /s\n", cps);
    std::printf("Rapfi search %llu nodes: %.2f searches/s (%.0f ms each)\n", (unsigned long long)nodes, sps, 1000 / sps);
    (void)acc;
    return 0;
}

// ── match ──
std::vector<int> replaySeq(const gmk::Record& src) {
    uint8_t arr[gmk::NMOVES]; src.unpack(arr);
    std::vector<int> bl, wh;
    const int l1 = src.last1 == 255 ? -1 : src.last1, l2 = src.last2 == 255 ? -1 : src.last2;
    for (int m = 0; m < gmk::NMOVES; ++m) {
        if (m == l1 || m == l2) continue;
        if (arr[m] == gmk::BLACK) bl.push_back(m); else if (arr[m] == gmk::WHITE) wh.push_back(m);
    }
    auto pushLast = [&](int m) { if (m >= 0) (arr[m] == gmk::BLACK ? bl : wh).push_back(m); };
    pushLast(l2); pushLast(l1);
    std::vector<int> seq;
    if (!(bl.size() == wh.size() || bl.size() == wh.size() + 1)) return seq;
    for (size_t k = 0; k < bl.size() + wh.size(); ++k) seq.push_back(k % 2 == 0 ? bl[k / 2] : wh[k / 2]);
    return seq;
}

// 자식 국면 판정 공통: 내가 5목 → 승, 상대 즉승/쌍사 → 패
// 반환: +1 확정 승, -1 확정 패, 0 미확정
int childVerdict(const gmk::Board& child) {
    if (child.isLoss()) return 1;
    int ls;
    if (child.leafProbe(0, ls)) return ls > 0 ? -1 : 1;
    return 0;
}

int pickOursGreedy(gmk::Board& b) {
    int mv[gmk::NMOVES], n = b.generate(mv);
    if (n == 1) return mv[0];
    int best = mv[0]; double bs = -1e18;
    for (int i = 0; i < n; ++i) {
        b.make(mv[i]);
        const int v = childVerdict(b);
        const double sc = v ? v * 1e9 : -double(b.evaluate());
        b.undo(mv[i]);
        if (sc > bs) { bs = sc; best = mv[i]; }
    }
    return best;
}

// Rapfi 혼합 평가(탐색 내부와 동일): Evaluation::evaluate<FREESTYLE>(board, alpha, beta)
//   고전평가가 창 ± 여유폭 안이면 NNUE, 밖이면 고전평가 값을 그대로 쓴다.
uint64_t g_mixCalls = 0, g_mixNN = 0;
int pickRapfiMixed(Pair& p) {
    const gmk::Model* keep = p.gb.model; p.gb.model = nullptr;
    int mv[gmk::NMOVES], n = p.gb.generate(mv);
    p.gb.model = keep;
    if (n == 1) return mv[0];
    int best = mv[0]; double bs = -1e18;
    Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
    for (int i = 0; i < n; ++i) {
        p.move(mv[i]);
        const int v = childVerdict(p.gb);
        double sc;
        if (v) sc = v * 1e9;
        else {
            // 첫 후보는 전체 창, 이후는 PVS 처럼 [alpha, alpha+1] 널 창 (자식 관점으로 뒤집어 전달)
            const Value a = alpha, b = (i == 0) ? beta : Value(alpha + 1);
            const Value cv = Evaluation::evaluate<Rule::FREESTYLE>(*p.rb, Value(-b), Value(-a));
            const Value cls = Evaluation::evaluate<Rule::FREESTYLE>(*p.plain, Value(-b), Value(-a));
            ++g_mixCalls; g_mixNN += (cv != cls);
            sc = -double(cv);
            if (Value(-cv) > alpha) alpha = Value(-cv);
        }
        p.undo();
        if (sc > bs) { bs = sc; best = mv[i]; }
    }
    return best;
}

int pickRapfi(Pair& p, bool policy) {
    const gmk::Model* keep = p.gb.model; p.gb.model = nullptr;
    int mv[gmk::NMOVES], n = p.gb.generate(mv);
    p.gb.model = keep;
    if (n == 1) return mv[0];
    if (policy) {
        NNOut nn = nnEval(p);
        for (auto& pr : nn.pol) for (int i = 0; i < n; ++i) if (mv[i] == pr.second) return pr.second;
        return mv[0];
    }
    int best = mv[0]; double bs = -1e18;
    for (int i = 0; i < n; ++i) {
        p.move(mv[i]);
        const int v = childVerdict(p.gb);
        double sc;
        if (v) sc = v * 1e9;
        else { auto val = p.rb->evaluator()->evaluateValue(*p.rb); sc = 1.0 - (val.win() + 0.5 * val.draw()); }
        p.undo();
        if (sc > bs) { bs = sc; best = mv[i]; }
    }
    return best;
}

int cmdMatch(const std::map<std::string, std::string>& a) {
    gmk::Model mdl;
    if (!mdl.load(get<std::string>(a, "model", ""))) { std::fprintf(stderr, "cannot load model\n"); return 1; }
    const bool oursSearch = get<std::string>(a, "ours", "greedy") == "search";
    const std::string rmode = get<std::string>(a, "rapfi", "greedy");
    const bool rapfiPolicy = rmode == "policy", rapfiMixed = rmode == "mixed";
    const int64_t ms = get<int64_t>(a, "ms", 100);
    const int pairs = get(a, "pairs", 20), offset = get(a, "offset", 0), stride = get(a, "stride", 1);
    std::vector<gmk::Record> recs, opens;
    gmk::readRecords(get<std::string>(a, "data", ""), recs);
    for (auto& r : recs)
        if (!(r.flags & (gmk::REC_SIBLING | gmk::REC_LEAF | gmk::REC_SELF_FIVE | gmk::REC_OPPO_FIVE)) && r.ply >= 4 &&
            r.ply <= 10 && std::fabs(r.nnWinrate() - 0.5f) < 0.2f) opens.push_back(r);
    ab::Searcher<gmk::Board> srch(64);
    int win = 0, draw = 0, loss = 0; long moves = 0;
    for (int pi = 0; pi < pairs; ++pi) {
        const gmk::Record& o = opens[(size_t(offset) + size_t(pi) * stride * 7919) % opens.size()];
        const std::vector<int> seq = replaySeq(o);
        if (seq.empty()) continue;
        for (int g = 0; g < 2; ++g) {
            Pair p; p.reset();
            gmk::Board ob; ob.model = &mdl;
            for (int m : seq) { p.move(m); ob.make(m); }
            const int oursColor = g == 0 ? ob.side : gmk::opp(ob.side);
            srch.tt.clear(); srch.clearHeuristics();
            int result = 0;
            while (true) {
                const bool oursTurn = ob.side == oursColor;
                int m;
                if (oursTurn) {
                    if (oursSearch) m = srch.think(ob, ab::Limits{ab::MAX_PLY - 1, ms, 0}).bestMove;
                    else m = pickOursGreedy(ob);
                } else m = rapfiMixed ? pickRapfiMixed(p) : pickRapfi(p, rapfiPolicy);
                if (m < 0 || !ob.isEmptyMove(m)) { for (int q = 0; q < gmk::NMOVES; ++q) if (ob.isEmptyMove(q)) { m = q; break; } }
                p.move(m); ob.make(m); ++moves;
                if (ob.isLoss()) { result = oursTurn ? 1 : -1; break; }
                if (ob.isDraw()) break;
            }
            (result > 0 ? win : result < 0 ? loss : draw)++;
        }
    }
    std::printf("RESULT ours_win=%d draw=%d ours_loss=%d games=%d avg-moves=%.1f nnue-used=%.3f\n", win, draw, loss,
                win + draw + loss, double(moves) / std::max(1, win + draw + loss),
                g_mixCalls ? double(g_mixNN) / g_mixCalls : -1.0);
    return 0;
}

} // namespace


int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: oracle <gen|label|bench> [--config path] [options]\n");
        return 2;
    }
    Command::CommandLine::init(argc, argv);
    const std::string mode = argv[1];
    auto a = parseArgs(argc, argv, 2);
    if (!initRapfi(get<std::string>(a, "config", "config.toml"), get(a, "tt-mb", 64))) return 1;
    if (mode == "gen") return cmdGen(a);
    if (mode == "label") return cmdLabel(a);
    if (mode == "bench") return cmdBench(a);
    if (mode == "match") return cmdMatch(a);
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
