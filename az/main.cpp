// main.cpp — TA-NNUE AlphaZero-style chess: test / bench / selfplay / train / arena / loop / go / uci.
// Usage: az <command> [--key value ...]   (see docs/az/TRAINING.md)
#include "train.hpp"
#include <atomic>
#include <deque>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>

using namespace az;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double secsSince(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Args {
    std::map<std::string, std::string> kv;
    std::vector<std::string> pos;
    Args(int argc, char** argv, int start) {
        for (int i = start; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                std::string k = a.substr(2);
                if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv[k] = argv[++i];
                else kv[k] = "1";
            } else pos.push_back(a);
        }
    }
    std::string s(const std::string& k, const std::string& d) const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    long long i(const std::string& k, long long d) const { auto it = kv.find(k); return it == kv.end() ? d : std::stoll(it->second); }
    double f(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::stod(it->second); }
    bool has(const std::string& k) const { return kv.count(k) > 0; }
};

static SearchParams searchParams(const Args& a) {
    SearchParams sp;
    sp.sims = int(a.i("sims", 1000));
    sp.cInit = float(a.f("cinit", sp.cInit));
    sp.fpuRed = float(a.f("fpu", sp.fpuRed));
    sp.dirAlpha = float(a.f("alpha", sp.dirAlpha));
    sp.dirEps = float(a.f("eps", sp.dirEps));
    sp.policyTemp = float(a.f("ptemp", sp.policyTemp));
    return sp;
}
static TrainConfig trainConfig(const Args& a) {
    TrainConfig c;
    c.batch = int(a.i("batch", c.batch));
    c.lr = float(a.f("lr", c.lr));
    c.wd = float(a.f("wd", c.wd));
    c.lambda = float(a.f("lambda", c.lambda));
    c.valueW = float(a.f("vw", c.valueW));
    c.threads = int(a.i("threads", 4));
    return c;
}
static bool loadNet(Net& n, const std::string& path, uint64_t seed) {
    if (path.empty() || path == "random-init") { n.init(seed); return true; }
    if (!n.load(path)) { std::fprintf(stderr, "cannot load net %s\n", path.c_str()); return false; }
    return true;
}

// ─────────────────────────── self-play / arena drivers ───────────────────────────
struct SelfplaySummary {
    int games = 0, white = 0, draws = 0, black = 0, capped = 0, resigned = 0;
    int mates = 0, stalemates = 0, fifty = 0, rep = 0, material = 0;
    int64_t plies = 0, sims = 0;
    double secs = 0;
};

static SelfplaySummary runSelfplay(const Net& net, const SelfplayConfig& cfg, int games, int threads, uint64_t seed,
                                   Dataset& out, bool progress) {
    SelfplaySummary sum;
    std::atomic<int> next{0};
    std::mutex mu;
    auto t0 = Clock::now();
    std::vector<std::thread> th;
    for (int ti = 0; ti < threads; ++ti) {
        th.emplace_back([&, ti] {
            Searcher S(&net, seed * 7919 + uint64_t(ti) * 104729 + 1);
            Dataset local;
            for (;;) {
                int gi = next.fetch_add(1);
                if (gi >= games) break;
                GameStats gs = playSelfplayGame(S, cfg, local);
                std::lock_guard<std::mutex> lk(mu);
                sum.games++;
                sum.plies += gs.plies;
                sum.sims += gs.sims;
                (gs.result > 0 ? sum.white : gs.result < 0 ? sum.black : sum.draws)++;
                sum.capped += gs.capped;
                sum.resigned += gs.resigned;
                switch (gs.how) {
                    case Outcome::CHECKMATE: sum.mates++; break;
                    case Outcome::STALEMATE: sum.stalemates++; break;
                    case Outcome::FIFTY: sum.fifty++; break;
                    case Outcome::REPETITION: sum.rep++; break;
                    case Outcome::MATERIAL: sum.material++; break;
                    default: break;
                }
                if (progress && (sum.games % std::max(1, games / 10) == 0 || sum.games == games))
                    std::fprintf(stderr, "\r  selfplay %d/%d  (%.1fs)   ", sum.games, games, secsSince(t0));
            }
            std::lock_guard<std::mutex> lk(mu);
            out.append(local);
        });
    }
    for (auto& x : th) x.join();
    if (progress) std::fprintf(stderr, "\n");
    sum.secs = secsSince(t0);
    return sum;
}

struct ArenaResult {
    int w = 0, d = 0, l = 0;   // from A's view
    int n() const { return w + d + l; }
    double score() const { return n() ? (w + 0.5 * d) / n() : 0.5; }
    static double elo(double s) { s = std::clamp(s, 1e-4, 1 - 1e-4); return -400.0 * std::log10(1.0 / s - 1.0); }
    double eloCI95() const {   // half-width, delta method on the per-game score variance
        if (!n()) return 0;
        double s = score(), var = (w * (1 - s) * (1 - s) + d * (0.5 - s) * (0.5 - s) + l * s * s) / n();
        double se = std::sqrt(var / n());
        double sc = std::clamp(s, 0.02, 0.98);
        return 1.96 * se * 400.0 / (std::log(10.0) * sc * (1 - sc));
    }
};

// `pairs` openings, each played twice with colors swapped. nullptr = random mover.
static ArenaResult runArena(const Net* A, const Net* B, const SearchParams& sp, int pairs, int threads, uint64_t seed,
                            int openingPlies, bool progress) {
    ArenaResult r;
    std::atomic<int> next{0};
    std::mutex mu;
    std::vector<std::thread> th;
    auto t0 = Clock::now();
    for (int ti = 0; ti < threads; ++ti) {
        th.emplace_back([&, ti] {
            Searcher sa(A, seed + 11 * ti + 1), sb(B, seed + 13 * ti + 2);
            for (;;) {
                int pi = next.fetch_add(1);
                if (pi >= pairs) break;
                std::mt19937_64 rng(seed * 1000003 + uint64_t(pi));
                auto op = randomOpening(openingPlies, rng);
                int g1 = playMatchGame(sa, sb, sp, op, rng);    // A white
                int g2 = -playMatchGame(sb, sa, sp, op, rng);   // A black
                std::lock_guard<std::mutex> lk(mu);
                for (int g : {g1, g2}) (g > 0 ? r.w : g < 0 ? r.l : r.d)++;
                if (progress) std::fprintf(stderr, "\r  arena %d/%d games  +%d =%d -%d  (%.1fs)   ", r.n(), 2 * pairs, r.w, r.d, r.l, secsSince(t0));
            }
        });
    }
    for (auto& x : th) x.join();
    if (progress) std::fprintf(stderr, "\n");
    return r;
}

// ─────────────────────────── commands ───────────────────────────
static int cmdTest(const Args&) {
    int fails = 0;
    struct P { const char* fen; int d; uint64_t n; };
    const P suite[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
    };
    for (auto& t : suite) {
        Position p;
        p.setFen(t.fen);
        auto t0 = Clock::now();
        uint64_t n = perft(p, t.d);
        bool ok = n == t.n;
        fails += !ok;
        std::printf("perft %-70s d%d %10llu %s (%.2fs)\n", t.fen, t.d, (unsigned long long)n, ok ? "OK" : "FAIL", secsSince(t0));
    }
    // FEN round trip
    {
        Position p;
        const char* f = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        p.setFen(f);
        bool ok = p.fen() == f;
        fails += !ok;
        std::printf("fen round trip %s\n", ok ? "OK" : "FAIL");
    }
    // H1: incremental accumulator == full refresh along random games
    {
        Net net;
        net.init(7);
        std::mt19937_64 rng(3);
        float maxErr = 0;
        int steps = 0;
        for (int g = 0; g < 200; ++g) {
            Position p;
            p.setStart();
            Acc a, b, c;
            refreshAcc(net, p, a);
            for (int ply = 0; ply < 200; ++ply) {
                Move ms[MAX_MOVES];
                int n = genLegal(p, ms);
                if (!n) break;
                Position q = p;
                q.make(ms[std::uniform_int_distribution<int>(0, n - 1)(rng)]);
                updateAcc(net, a, p, q, b);
                refreshAcc(net, q, c);
                for (int k = 0; k < 2; ++k) for (int j = 0; j < H; ++j) maxErr = std::max(maxErr, std::fabs(b.v[k][j] - c.v[k][j]));
                a = b;
                p = q;
                ++steps;
            }
        }
        bool ok = maxErr < 1e-3f;
        fails += !ok;
        std::printf("incremental accumulator: %d chained updates, max |diff| %.2e %s\n", steps, maxErr, ok ? "OK" : "FAIL");
    }
    // Backprop vs finite differences
    {
        Net net;
        net.init(11);
        for (size_t i = L::V2B; i < L::V2B + 3; ++i) net.w[i] = 0.1f * float(i - L::V2B);
        std::mt19937_64 rng(5);
        TrainConfig cfg;
        cfg.lambda = 0.3f;
        cfg.valueW = 1.f;
        int checked = 0, good = 0;
        for (int trial = 0; trial < 6; ++trial) {
            Position p;
            p.setStart();
            for (int k = 0; k < 10 + 7 * trial; ++k) {
                Move ms[MAX_MOVES];
                int n = genLegal(p, ms);
                if (!n) break;
                p.make(ms[std::uniform_int_distribution<int>(0, n - 1)(rng)]);
            }
            Move ms[MAX_MOVES];
            int n = genLegal(p, ms);
            if (!n) continue;
            Dataset d;
            SampleHdr h{};
            std::memcpy(h.board, p.sq, 64);
            h.stm = p.stm; h.castle = p.castle; h.result = int8_t(trial % 3 - 1);
            h.root[0] = 0.5f; h.root[1] = 0.3f; h.root[2] = 0.2f;
            h.n = uint16_t(n);
            for (int i = 0; i < n; ++i) d.mv.push_back(uint32_t(policyIndex(ms[i], p.stm)) | (uint32_t(rng() % 50) << 16));
            d.s.push_back(h);
            auto loss = [&]() {
                LossStats st;
                std::vector<float> g(L::TOTAL, 0.f);
                sampleGrad(net, d.s[0], d.mv.data(), g.data(), st, cfg);
                return st.pol + cfg.valueW * st.val;
            };
            std::vector<float> g(L::TOTAL, 0.f);
            LossStats st;
            sampleGrad(net, d.s[0], d.mv.data(), g.data(), st, cfg);
            const size_t regions[][2] = {{L::FTW, L::FTB}, {L::FTB, L::V1W}, {L::V1W, L::V1B}, {L::V2W, L::V2B},
                                         {L::P1W, L::P1B}, {L::PM, L::PB}, {L::PB, L::TOTAL}};
            for (auto& rg : regions) {
                std::vector<size_t> idx;
                for (size_t i = rg[0]; i < rg[1]; ++i) if (std::fabs(g[i]) > 1e-3f) idx.push_back(i);
                std::shuffle(idx.begin(), idx.end(), rng);
                for (size_t k = 0; k < std::min<size_t>(idx.size(), 8); ++k) {
                    size_t i = idx[k];
                    float w0 = net.w[i], e = 1e-2f;
                    net.w[i] = w0 + e; double lp = loss();
                    net.w[i] = w0 - e; double lm = loss();
                    net.w[i] = w0;
                    double num = (lp - lm) / (2 * e);
                    double rel = std::fabs(num - g[i]) / std::max(1e-3, std::fabs(num) + std::fabs(g[i]));
                    ++checked;
                    good += rel < 0.05;
                }
            }
        }
        bool ok = checked > 50 && good >= 0.9 * checked;
        fails += !ok;
        std::printf("gradient check: %d/%d params within 5%% of finite differences %s\n", good, checked, ok ? "OK" : "FAIL");
    }
    // Solver: mate in one must be found and proven
    {
        Net net;
        net.init(1);
        Searcher S(&net);
        S.sp.sims = 400;
        Position p;
        p.setFen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");   // Ra8#
        S.search(p, {});
        Move m = S.pickMove(0);
        bool ok = moveStr(m) == "a1a8" && S.rootNode().proven == LOSS;
        fails += !ok;
        std::printf("solver mate-in-1: picked %s, root proven=%d %s\n", moveStr(m).c_str(), S.rootNode().proven, ok ? "OK" : "FAIL");
    }
    std::printf("%s (%d failures)\n", fails ? "TESTS FAILED" : "ALL TESTS PASSED", fails);
    return fails ? 1 : 0;
}

static int cmdBench(const Args& a) {
    Net net;
    if (!loadNet(net, a.s("net", ""), 1)) return 1;
    // H2 micro-benchmark: (incremental + legal-sparse) vs (full refresh + dense 4,168 output layer)
    std::mt19937_64 rng(9);
    struct Pair { Position before, after; std::vector<Move> moves; };
    std::vector<Pair> pairs;
    while (pairs.size() < 4000) {
        Position p;
        p.setStart();
        for (int ply = 0; ply < 80; ++ply) {
            Move ms[MAX_MOVES];
            int n = genLegal(p, ms);
            if (!n) break;
            Position q = p;
            q.make(ms[std::uniform_int_distribution<int>(0, n - 1)(rng)]);
            Move qs[MAX_MOVES];
            int qn = genLegal(q, qs);
            if (!qn) break;
            pairs.push_back({p, q, std::vector<Move>(qs, qs + qn)});
            p = q;
        }
    }
    Acc parent, child;
    float pri[MAX_MOVES], wdl[3], sink = 0;
    const int reps = 5;
    auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
        for (auto& pr : pairs) {
            refreshAcc(net, pr.before, parent);   // parent acc exists in the tree; excluded below by subtracting
        }
    double tParent = secsSince(t0);
    t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
        for (auto& pr : pairs) {
            refreshAcc(net, pr.before, parent);
            updateAcc(net, parent, pr.before, pr.after, child);
            evaluate(net, child, pr.after.stm, pr.moves.data(), int(pr.moves.size()), pri, wdl);
            sink += pri[0] + wdl[0];
        }
    double tInc = secsSince(t0) - tParent;
    std::vector<float> dense(NPOL);
    t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
        for (auto& pr : pairs) {
            refreshAcc(net, pr.after, child);
            evaluate(net, child, pr.after.stm, nullptr, 0, pri, wdl);   // value + (policy hidden computed below)
            const float* w = net.w.data();
            float z[2 * H], hp[HP];
            for (int j = 0; j < H; ++j) { z[j] = clamp01(child.v[pr.after.stm][j]); z[H + j] = clamp01(child.v[pr.after.stm ^ 1][j]); }
            for (int o = 0; o < HP; ++o) hp[o] = clamp01(w[L::P1B + o] + dot(&w[L::P1W + size_t(o) * 2 * H], z, 2 * H));
            for (int k = 0; k < NPOL; ++k) dense[k] = w[L::PB + k] + dot(&w[L::PM + size_t(k) * HP], hp, HP);
            for (size_t i = 0; i < pr.moves.size(); ++i) pri[i] = dense[policyIndex(pr.moves[i], pr.after.stm)];
            softmaxInPlace(pri, int(pr.moves.size()));
            sink += pri[0] + wdl[0];
        }
    double tFull = secsSince(t0);
    double nEv = double(pairs.size()) * reps;
    std::printf("eval (P1 incremental + P2 legal-sparse): %.2f us/eval\n", 1e6 * tInc / nEv);
    std::printf("eval (full refresh + dense output layer): %.2f us/eval\n", 1e6 * tFull / nEv);
    std::printf("speedup: %.2fx   (sink %.3f)\n", tFull / std::max(1e-9, tInc), sink);
    // End-to-end MCTS speed
    Searcher S(&net, 1);
    S.sp = searchParams(a);
    const char* fens[] = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                          "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8"};
    int64_t sims = 0;
    t0 = Clock::now();
    for (auto f : fens) {
        Position p;
        p.setFen(f);
        for (int r = 0; r < 5; ++r) sims += S.search(p, {});
    }
    double t = secsSince(t0);
    std::printf("mcts: %lld sims in %.2fs = %.0f sims/s (1 thread); %d sims/move = %.1f ms/move\n", (long long)sims, t,
                sims / t, S.sp.sims, 1e3 * t * S.sp.sims / sims);
    return 0;
}

static SelfplayConfig selfplayConfig(const Args& a) {
    SelfplayConfig c;
    c.sp = searchParams(a);
    c.sp.noise = !a.has("no-noise");
    c.tempPlies = int(a.i("temp-plies", c.tempPlies));
    c.maxPlies = int(a.i("max-plies", c.maxPlies));
    c.resignQ = float(a.f("resign", c.resignQ));
    return c;
}

static void printSelfplay(const SelfplaySummary& s, size_t samples) {
    std::printf("  games %d (W %d / D %d / B %d)  mate %d stalemate %d 50-move %d rep %d material %d cap %d resign %d\n", s.games,
                s.white, s.draws, s.black, s.mates, s.stalemates, s.fifty, s.rep, s.material, s.capped, s.resigned);
    std::printf("  avg plies %.1f  samples %zu  %.1fs  (%.2f games/s, %.0f sims/s)\n", s.games ? double(s.plies) / s.games : 0.0,
                samples, s.secs, s.games / std::max(1e-9, s.secs), s.sims / std::max(1e-9, s.secs));
}

static int cmdSelfplay(const Args& a) {
    Net net;
    if (!loadNet(net, a.s("net", ""), uint64_t(a.i("seed", 1)))) return 1;
    Dataset d;
    auto s = runSelfplay(net, selfplayConfig(a), int(a.i("games", 100)), int(a.i("threads", 4)), uint64_t(a.i("seed", 1)), d, true);
    printSelfplay(s, d.size());
    std::string out = a.s("out", "selfplay.bin");
    if (!d.save(out)) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    std::printf("wrote %s\n", out.c_str());
    return 0;
}

static void printLoss(const char* tag, const LossStats& st) {
    double n = std::max<int64_t>(1, st.n);
    std::printf("  %s policy CE %.4f (target entropy %.4f, KL %.4f)  value CE %.4f  top1 %.1f%%\n", tag, st.pol / n,
                st.polTargetEnt / n, (st.pol - st.polTargetEnt) / n, st.val / n, 100.0 * st.top1 / n);
}

static int cmdTrain(const Args& a) {
    Net net;
    if (!loadNet(net, a.s("net", ""), uint64_t(a.i("seed", 1)))) return 1;
    std::vector<Dataset> ds(a.pos.size());
    std::vector<const Dataset*> win;
    for (size_t i = 0; i < a.pos.size(); ++i) {
        if (!ds[i].load(a.pos[i])) { std::fprintf(stderr, "cannot load %s\n", a.pos[i].c_str()); return 1; }
        win.push_back(&ds[i]);
    }
    if (win.empty()) { std::fprintf(stderr, "usage: az train data1.bin [data2.bin ...] --net in --out out --steps N\n"); return 1; }
    Trainer tr(trainConfig(a));
    std::mt19937_64 rng(uint64_t(a.i("seed", 1)));
    int steps = int(a.i("steps", 1000)), every = int(a.i("report", 100));
    for (int s = 0; s < steps; s += every) {
        auto st = tr.train(net, win, std::min(every, steps - s), rng);
        std::printf("step %d\n", s + std::min(every, steps - s));
        printLoss("train", st);
    }
    std::string out = a.s("out", "net.bin");
    if (!net.save(out)) return 1;
    std::printf("wrote %s\n", out.c_str());
    return 0;
}

static int cmdArena(const Args& a) {
    Net na, nb;
    if (!loadNet(na, a.s("a", ""), 1)) return 1;
    bool bRandom = a.s("b", "") == "random";
    if (!bRandom && !loadNet(nb, a.s("b", ""), 2)) return 1;
    SearchParams sp = searchParams(a);
    auto r = runArena(&na, bRandom ? nullptr : &nb, sp, int(a.i("pairs", 50)), int(a.i("threads", 4)), uint64_t(a.i("seed", 1)),
                      int(a.i("opening-plies", 4)), true);
    std::printf("A vs B: +%d =%d -%d  score %.1f%%  Elo %+.0f ± %.0f\n", r.w, r.d, r.l, 100 * r.score(), ArenaResult::elo(r.score()),
                r.eloCI95());
    return 0;
}

static std::string genPath(const std::string& dir, const char* kind, int g) {
    char b[64];
    std::snprintf(b, sizeof(b), "%s_%04d.bin", kind, g);
    return (fs::path(dir) / b).string();
}

static int cmdLoop(const Args& a) {
    const std::string dir = a.s("dir", "run");
    fs::create_directories(dir);
    const int gens = int(a.i("gens", 100)), games = int(a.i("games", 200)), threads = int(a.i("threads", 4));
    const int window = int(a.i("window", 8)), arenaEvery = int(a.i("arena-every", 5)), arenaPairs = int(a.i("arena-pairs", 20));
    const double reuse = a.f("reuse", 4.0);
    SelfplayConfig sc = selfplayConfig(a);
    TrainConfig tc = trainConfig(a);
    SearchParams arenaSp = searchParams(a);
    arenaSp.sims = int(a.i("arena-sims", sc.sp.sims));

    int g = 0;
    while (fs::exists(genPath(dir, "net", g + 1))) ++g;
    Net net;
    if (!fs::exists(genPath(dir, "net", 0))) {
        net.init(uint64_t(a.i("seed", 1)));
        net.save(genPath(dir, "net", 0));
        std::printf("initialized random net %s\n", genPath(dir, "net", 0).c_str());
    }
    if (!net.load(genPath(dir, "net", g))) { std::fprintf(stderr, "cannot load %s\n", genPath(dir, "net", g).c_str()); return 1; }
    Trainer tr(tc);
    std::string adamPath = (fs::path(dir) / "adam.bin").string();
    if (g > 0 && !tr.load(adamPath)) std::printf("note: no optimizer state found, starting AdamW fresh\n");
    std::deque<Dataset> win;
    for (int k = std::max(1, g - window + 1); k <= g; ++k) {
        Dataset d;
        if (d.load(genPath(dir, "data", k))) win.push_back(std::move(d));
    }
    std::printf("resume at generation %d (window %zu datasets)\n", g, win.size());

    std::string logPath = (fs::path(dir) / "log.tsv").string();
    bool newLog = !fs::exists(logPath);
    FILE* log = std::fopen(logPath.c_str(), "a");
    if (newLog && log)
        std::fprintf(log, "gen\tgames\tW\tD\tB\tmates\tavg_plies\tsamples\tsp_secs\tsims_per_s\tsteps\tpol_ce\tpol_kl\tval_ce\ttop1\ttrain_secs\tarena_vs\tarena_score\tarena_elo\tarena_ci\n");

    for (int gen = g + 1; gen <= g + gens; ++gen) {
        std::printf("── generation %d ──\n", gen);
        Dataset d;
        auto s = runSelfplay(net, sc, games, threads, uint64_t(gen), d, true);
        printSelfplay(s, d.size());
        d.save(genPath(dir, "data", gen));
        win.push_back(std::move(d));
        while (int(win.size()) > window) win.pop_front();
        std::vector<const Dataset*> wp;
        for (auto& x : win) wp.push_back(&x);
        int steps = std::max(1, int(reuse * double(win.back().size()) / tc.batch));
        std::mt19937_64 rng(uint64_t(gen) * 31337);
        auto t0 = Clock::now();
        auto st = tr.train(net, wp, steps, rng);
        double trainSecs = secsSince(t0);
        std::printf("  train %d steps x %d (%.1fs)\n", steps, tc.batch, trainSecs);
        printLoss("train", st);
        net.save(genPath(dir, "net", gen));
        tr.save(adamPath);

        int vs = -1;
        ArenaResult ar;
        if (arenaEvery > 0 && gen % arenaEvery == 0) {
            vs = gen - arenaEvery;
            Net old;
            if (old.load(genPath(dir, "net", vs))) {
                ar = runArena(&net, &old, arenaSp, arenaPairs, threads, uint64_t(gen) * 77, 4, true);
                std::printf("  arena net_%04d vs net_%04d: +%d =%d -%d  score %.1f%%  Elo %+.0f ± %.0f\n", gen, vs, ar.w, ar.d, ar.l,
                            100 * ar.score(), ArenaResult::elo(ar.score()), ar.eloCI95());
            } else vs = -1;
        }
        double n = std::max<int64_t>(1, st.n);
        if (log) {
            std::fprintf(log, "%d\t%d\t%d\t%d\t%d\t%d\t%.1f\t%zu\t%.1f\t%.0f\t%d\t%.4f\t%.4f\t%.4f\t%.3f\t%.1f\t%d\t%.3f\t%.0f\t%.0f\n",
                         gen, s.games, s.white, s.draws, s.black, s.mates, s.games ? double(s.plies) / s.games : 0.0, win.back().size(),
                         s.secs, s.sims / std::max(1e-9, s.secs), steps, st.pol / n, (st.pol - st.polTargetEnt) / n, st.val / n,
                         st.top1 / n, trainSecs, vs, vs >= 0 ? ar.score() : 0.0, vs >= 0 ? ArenaResult::elo(ar.score()) : 0.0,
                         vs >= 0 ? ar.eloCI95() : 0.0);
            std::fflush(log);
        }
        std::fflush(stdout);
    }
    if (log) std::fclose(log);
    return 0;
}

static void printSearchInfo(Searcher& S, int sims, double secs, std::ostream& os) {
    float wdl[3];
    S.rootWDL(wdl);
    double q = std::clamp(double(wdl[0] - wdl[2]), -0.999, 0.999);
    int cp = int(std::lround(400.0 * std::log10((1 + q) / (1 - q))));
    if (S.rootNode().proven == LOSS) cp = 20000;
    if (S.rootNode().proven == WIN) cp = -20000;
    os << "info depth " << S.pv().size() << " nodes " << sims << " nps " << int(sims / std::max(1e-6, secs)) << " wdl "
       << int(1000 * wdl[0]) << ' ' << int(1000 * wdl[1]) << ' ' << int(1000 * wdl[2]) << " score cp " << cp << " pv";
    for (Move m : S.pv()) os << ' ' << moveStr(m);
    os << std::endl;
}

static int cmdGo(const Args& a) {
    Net net;
    if (!loadNet(net, a.s("net", ""), 1)) return 1;
    Position p;
    if (!p.setFen(a.s("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))) { std::fprintf(stderr, "bad fen\n"); return 1; }
    Searcher S(&net, 1);
    S.sp = searchParams(a);
    auto t0 = Clock::now();
    int sims = S.search(p, {});
    printSearchInfo(S, sims, secsSince(t0), std::cout);
    const Node& r = S.rootNode();
    std::vector<int> order(r.nch);
    for (int k = 0; k < r.nch; ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](int x, int y) { return S.child(r, x).N > S.child(r, y).N; });
    for (int k = 0; k < std::min<int>(r.nch, 10); ++k) {
        const Node& c = S.child(r, order[k]);
        std::printf("  %-6s N %5d  P %.3f  Q %+.3f  %s\n", moveStr(c.move).c_str(), c.N, c.prior, c.N ? c.W / c.N : 0.f,
                    c.proven == WIN ? "proven win" : c.proven == LOSS ? "proven loss" : c.proven == DRAW ? "proven draw" : "");
    }
    std::cout << "bestmove " << moveStr(S.pickMove(0)) << std::endl;
    return 0;
}

static int cmdUci(const Args& a) {
    Net net;
    if (!loadNet(net, a.s("net", ""), 1)) return 1;
    Searcher S(&net, 1);
    SearchParams base = searchParams(a);
    Position pos;
    pos.setStart();
    std::vector<u64> hist;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if (cmd == "uci") {
            std::cout << "id name TA-NNUE-AZ\nid author Deepmath-0\noption name Nodes type spin default " << base.sims
                      << " min 1 max 100000000\nuciok" << std::endl;
        } else if (cmd == "isready") std::cout << "readyok" << std::endl;
        else if (cmd == "setoption") {
            std::string tok, name, value;
            while (is >> tok) {
                if (tok == "name") is >> name;
                else if (tok == "value") is >> value;
            }
            if (name == "Nodes" && !value.empty()) base.sims = std::stoi(value);
        } else if (cmd == "ucinewgame") { pos.setStart(); hist.clear(); }
        else if (cmd == "position") {
            std::string tok;
            is >> tok;
            hist.clear();
            if (tok == "startpos") { pos.setStart(); is >> tok; }
            else if (tok == "fen") {
                std::string fen, part;
                while (is >> part && part != "moves") fen += part + " ";
                pos.setFen(fen);
                tok = part;
            }
            if (tok == "moves") {
                std::string ms;
                while (is >> ms) {
                    Move m = parseMove(pos, ms);
                    if (!m) break;
                    hist.push_back(pos.key);
                    pos.make(m);
                }
            }
        } else if (cmd == "go") {
            SearchParams sp = base;
            std::string tok;
            long long wt = -1, bt = -1, wi = 0, bi = 0, mt = -1;
            while (is >> tok) {
                long long v = 0;
                if (tok == "nodes" && is >> v) sp.sims = int(v);
                else if (tok == "movetime" && is >> v) mt = v;
                else if (tok == "wtime" && is >> v) wt = v;
                else if (tok == "btime" && is >> v) bt = v;
                else if (tok == "winc" && is >> v) wi = v;
                else if (tok == "binc" && is >> v) bi = v;
            }
            long long my = pos.stm == WHITE ? wt : bt, inc = pos.stm == WHITE ? wi : bi;
            if (mt > 0) { sp.maxMs = mt; sp.sims = 100000000; }
            else if (my >= 0) { sp.maxMs = std::max(10LL, my / 30 + inc / 2); }
            Move ms[MAX_MOVES];
            if (!genLegal(pos, ms)) { std::cout << "bestmove 0000" << std::endl; continue; }
            S.sp = sp;
            auto t0 = Clock::now();
            int sims = S.search(pos, hist);
            printSearchInfo(S, sims, secsSince(t0), std::cout);
            std::cout << "bestmove " << moveStr(S.pickMove(0)) << std::endl;
        } else if (cmd == "quit") break;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: az <test|bench|selfplay|train|arena|loop|go|uci> [--key value ...]\n"
                     "  see docs/az/TRAINING.md\n");
        return 1;
    }
    std::string c = argv[1];
    Args a(argc, argv, 2);
    if (c == "test") return cmdTest(a);
    if (c == "bench") return cmdBench(a);
    if (c == "selfplay") return cmdSelfplay(a);
    if (c == "train") return cmdTrain(a);
    if (c == "arena") return cmdArena(a);
    if (c == "loop") return cmdLoop(a);
    if (c == "go") return cmdGo(a);
    if (c == "uci") return cmdUci(a);
    std::fprintf(stderr, "unknown command %s\n", c.c_str());
    return 1;
}
