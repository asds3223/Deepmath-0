// oracle.cpp — Rapfi(태그 250615) 소스를 직접 링크한 "정답지" 도구
//
// 모드
//   gen        Rapfi NNUE 정책 샘플링 롤아웃으로 국면을 만들고, NNUE 가치/정책(+선택적 탐색) 라벨을 붙인다.
//              (우리 엔진 자가대국이 아니라 교사 모델이 두는 국면 분포)
//   label      기존 레코드(예: 우리 탐색 트리 리프 덤프)에 NNUE 가치/정책/탐색 라벨을 다시 붙인다.
//   crosscheck 우리 pattern.hpp/board.hpp 의 패턴·Pattern4 가 Rapfi 와 완전히 같은지 무작위 국면에서 검증.
//   bench      NNUE 정적 평가·정책·탐색 처리량 측정 (데이터 예산 산정용)
//
// 병렬: 한 프로세스 = 한 코어. 대량 작업은 scripts/ 의 셸 스크립트가 코어 수만큼 프로세스를 띄운다.
#include "../engine/record.hpp"

#include "command/command.h"
#include "config.h"
#include "core/pos.h"
#include "eval/eval.h"
#include "eval/evaluator.h"
#include "game/board.h"
#include "search/searchthread.h"

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
    if (p.gb.p4cnt[s][gmk::A_FIVE]) r.flags |= gmk::REC_SELF_FIVE;
    if (p.gb.p4cnt[o][gmk::A_FIVE]) r.flags |= gmk::REC_OPPO_FIVE;
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
                if (U(rng) < eps) mv = cand[rng() % nc];
                else {
                    // 온도 적용 정책 샘플링
                    double tot = 0; std::vector<double> pr(nn.pol.size());
                    for (size_t i = 0; i < nn.pol.size(); ++i) tot += pr[i] = std::pow(std::max(nn.pol[i].first, 1e-12f), 1.0 / temp);
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

// ── crosscheck ──
int cmdCrosscheck(const std::map<std::string, std::string>& a) {
    const int games = get(a, "games", 200);
    std::mt19937_64 rng(get<uint64_t>(a, "seed", 7));
    static const int dirMap[4] = {0, 1, 3, 2};   // gmk 방향 → Rapfi 방향 (↘/↙ 순서가 반대)
    uint64_t cells = 0, patMis = 0, p4Mis = 0;
    for (int g = 0; g < games; ++g) {
        auto rb = std::make_unique<Board>(gmk::BS, CandidateRange::SQUARE3_LINE4);
        rb->newGame(Rule::FREESTYLE);
        gmk::Board gb;
        for (int ply = 0; ply < 160; ++ply) {
            int cand[gmk::NMOVES], nc = 0;
            for (int m = 0; m < gmk::NMOVES; ++m) if (gb.isEmptyMove(m) && (ply == 0 || gb.isCandidate(gmk::toCell(m)))) cand[nc++] = m;
            if (!nc || gb.isLoss()) break;
            int m = cand[rng() % nc];
            rb->move(Rule::FREESTYLE, toPos(m)); gb.make(m);
            if (gb.isLoss()) break;
            for (int q = 0; q < gmk::NMOVES; ++q) {
                if (!gb.isEmptyMove(q)) continue;
                const Cell& rc = rb->cell(toPos(q));
                const int c = gmk::toCell(q);
                ++cells;
                for (int col = 0; col < 2; ++col) {
                    Color rcol = col == 0 ? BLACK : WHITE;
                    for (int d = 0; d < 4; ++d)
                        if (int(rc.pattern(rcol, dirMap[d])) != int(gb.pat[col][d][c])) ++patMis;
                    if (int(rc.pattern4[rcol]) != int(gb.p4[col][c])) ++p4Mis;
                }
            }
        }
    }
    std::printf("crosscheck: empty-cells=%llu pattern-mismatch=%llu pattern4-mismatch=%llu\n",
                (unsigned long long)cells, (unsigned long long)patMis, (unsigned long long)p4Mis);
    return (patMis || p4Mis) ? 1 : 0;
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

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: oracle <gen|label|crosscheck|bench> [--config path] [options]\n");
        return 2;
    }
    Command::CommandLine::init(argc, argv);
    const std::string mode = argv[1];
    auto a = parseArgs(argc, argv, 2);
    if (!initRapfi(get<std::string>(a, "config", "config.toml"), get(a, "tt-mb", 64))) return 1;
    if (mode == "gen") return cmdGen(a);
    if (mode == "label") return cmdLabel(a);
    if (mode == "crosscheck") return cmdCrosscheck(a);
    if (mode == "bench") return cmdBench(a);
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
