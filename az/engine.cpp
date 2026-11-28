// engine.cpp — TA-NNUE UCI engine (single file, weights embedded). No pondering.
//   UCI options: Hash (MB, tree memory), Threads (shared-tree MCTS with virtual loss), MoveOverhead,
//                MoveTime (fixed ms per move, 0 = use clock), Nodes (0 = unlimited), EvalFile.
//   go: wtime/btime/winc/binc/movestogo, movetime, nodes, infinite (+ stop).
// Build: az/build_engine.sh  → dist/az-engine.exe (Windows, generic x86-64-v2) and build/az-engine (Linux)
#include "mcts.hpp"   // SearchParams, Node proof constants (WIN/LOSS/DRAW), Net, chess
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>

#ifndef AZ_NET_FILE
#error "define AZ_NET_FILE (path of the net to embed)"
#endif
#if defined(_WIN32)
#define AZ_RODATA ".section .rdata,\"dr\"\n"
#else
#define AZ_RODATA ".section .rodata\n"
#endif
__asm__(AZ_RODATA ".balign 64\n.globl az_net_blob\naz_net_blob:\n.incbin \"" AZ_NET_FILE "\"\n.globl az_net_blob_end\naz_net_blob_end:\n.byte 0\n.text\n");
extern "C" const char az_net_blob[], az_net_blob_end[];

using namespace az;
using Clock = std::chrono::steady_clock;

static bool loadNetMem(Net& net, const char* p, size_t n) {
    uint32_t h[7];
    if (n < 28) return false;
    std::memcpy(h, p, 28);
    if (h[0] != 0x454E4E41u || h[2] != NF) return false;
    Dims d;
    size_t off;
    if (h[1] == 1) { d = {int(h[3]), int(h[4]), 32}; off = 24; }
    else if (h[1] == 2) { d = {int(h[3]), int(h[4]), int(h[5])}; off = 28; if (h[6] != NPOL) return false; }
    else return false;
    if (d.H < 1 || d.H > HMAX || d.HV < 1 || d.HV > HVMAX || d.HP < 1 || d.HP > HPMAX) return false;
    net.setDims(d);
    if (n < off + net.w.size() * 4) return false;
    std::memcpy(net.w.data(), p + off, net.w.size() * 4);
    return true;
}

// Shared-tree parallel PUCT: selection/backup under one mutex with virtual loss; move generation and
// network evaluation (the expensive part) run outside the lock. Tree memory is bounded by Hash.
struct PNode {
    float prior = 0.f, W = 0.f, D = 0.f;
    int32_t N = 0, first = -1, acc = -1;
    Move move = 0;
    uint8_t nch = 0;
    int8_t proven = UNKNOWN;
    uint8_t expanding = 0;
};

class ParallelSearch {
public:
    const Net* net = nullptr;
    float cInit = 1.25f, cBase = 19652.f, fpuRed = 0.25f;
    float policyTemp = 1.6f;   // tuned 2026-11-28 with net_1300: +127 ± 47 Elo vs 1.0 (data/tune/search_1300.log)
    std::vector<PNode> nodes;
    std::vector<Acc> accs;
    size_t nNodes = 0, nAccs = 0;
    std::mutex mu;
    Position root;
    std::vector<u64> gameHist;
    std::atomic<bool> stop{false};
    std::atomic<int64_t> sims{0};

    void setHash(size_t mb) {
        size_t bytes = std::max<size_t>(mb, 16) << 20;
        // Accumulators are a small win at large widths (the hidden layers dominate), so most memory goes to nodes;
        // when the accumulator pool is full, new leaves refresh from scratch. Search stops when nodes run out.
        size_t accN = std::max<size_t>(1024, bytes / 10 / sizeof(Acc));
        size_t nodeN = std::max<size_t>(65536, bytes * 9 / 10 / sizeof(PNode));
        std::vector<Acc>().swap(accs);
        std::vector<PNode>().swap(nodes);
        accs.resize(accN);
        nodes.resize(nodeN);
    }

    bool haveTree = false, allowReuse = true;
    int64_t reusedVisits = 0;

    // Tree reuse between moves: if p is reached from the previous root by 1 or 2 plies that are in the tree,
    // keep that subtree (compacted). Returns true when the tree was kept.
    bool reuse(const Position& p) {
        if (!haveTree || nodes[0].first < 0) return false;
        int cur = 0;
        Position q = root;
        for (int depth = 0; depth < 2 && q.key != p.key; ++depth) {
            const PNode& nd = nodes[cur];
            if (nd.first < 0) return false;
            int found = -1;
            for (int k = 0; k < nd.nch && found < 0; ++k) {
                Position t = q;
                t.make(nodes[nd.first + k].move);
                if (t.key == p.key || (depth == 0 && nodes[nd.first + k].first >= 0)) {
                    // depth 0: try each expanded child for a 2-ply match
                    if (t.key == p.key) { found = nd.first + k; q = t; }
                    else {
                        const PNode& c = nodes[nd.first + k];
                        for (int j = 0; j < c.nch; ++j) {
                            Position u = t;
                            u.make(nodes[c.first + j].move);
                            if (u.key == p.key) { found = c.first + j; q = u; break; }
                        }
                    }
                }
            }
            if (found < 0) return false;
            cur = found;
        }
        if (q.key != p.key || cur == 0 || nodes[cur].first < 0 || nodes[cur].proven != UNKNOWN) return false;
        std::vector<PNode> nn;
        std::vector<Acc> na;
        std::vector<int> from;
        nn.push_back(nodes[cur]);
        from.push_back(cur);
        for (size_t i = 0; i < nn.size(); ++i) {
            const PNode& o = nodes[from[i]];
            if (o.acc >= 0) { na.push_back(accs[o.acc]); nn[i].acc = int32_t(na.size() - 1); }
            if (o.first >= 0) {
                nn[i].first = int32_t(nn.size());
                for (int k = 0; k < o.nch; ++k) { nn.push_back(nodes[o.first + k]); from.push_back(o.first + k); }
            }
        }
        if (nn.size() > nodes.size() || na.size() > accs.size()) return false;
        std::copy(nn.begin(), nn.end(), nodes.begin());
        std::copy(na.begin(), na.end(), accs.begin());
        nNodes = nn.size();
        nAccs = na.size();
        return true;
    }

    void prepare(const Position& p, const std::vector<u64>& hist) {
        bool kept = allowReuse && reuse(p);
        haveTree = true;
        root = p;
        gameHist = hist;
        sims = 0;
        stop = false;
        reusedVisits = kept ? nodes[0].N - 1 : 0;
        if (kept) return;
        nNodes = 1;
        nAccs = 0;
        nodes[0] = PNode{};
        Move ms[MAX_MOVES];
        int n = genLegal(p, ms);
        int ai = int(nAccs++);
        refreshAcc(*net, p, accs[ai]);
        float pri[MAX_MOVES], wdl[3];
        evaluate(*net, accs[ai], p.stm, ms, n, pri, wdl, policyTemp);
        for (int i = 0; i < n; ++i) { nodes[nNodes + i] = PNode{}; nodes[nNodes + i].move = ms[i]; nodes[nNodes + i].prior = pri[i]; }
        nodes[0].first = int32_t(nNodes);
        nodes[0].nch = uint8_t(n);
        nodes[0].acc = ai;
        nodes[0].N = 1;
        nodes[0].W = -(wdl[0] - wdl[2]);
        nodes[0].D = wdl[1];
        nNodes += size_t(n);
    }

    struct Ctx {
        std::vector<u64> hist;
        std::vector<int> path;
        Acc tmp;
    };

    void worker(int64_t maxSims, Clock::time_point deadline, bool timed) {
        Ctx c;
        c.hist.reserve(gameHist.size() + 256);
        int k = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (maxSims > 0 && sims.load(std::memory_order_relaxed) >= maxSims) break;
            if (timed && (++k & 7) == 0 && Clock::now() >= deadline) break;
            if (simulate(c)) sims.fetch_add(1, std::memory_order_relaxed);
        }
        stop = true;
    }

    Move bestMove() const {
        const PNode& r = nodes[0];
        int best = -1;
        for (int k = 0; k < r.nch; ++k) {
            const PNode& ch = nodes[r.first + k];
            if (ch.proven == WIN && (best < 0 || ch.N > nodes[r.first + best].N)) best = k;
        }
        if (best >= 0) return nodes[r.first + best].move;
        bool anyOk = false;
        for (int k = 0; k < r.nch; ++k) anyOk |= nodes[r.first + k].proven != LOSS;
        for (int k = 0; k < r.nch; ++k) {
            const PNode& ch = nodes[r.first + k];
            if (anyOk && ch.proven == LOSS) continue;
            if (best < 0 || ch.N > nodes[r.first + best].N || (ch.N == nodes[r.first + best].N && ch.prior > nodes[r.first + best].prior)) best = k;
        }
        return nodes[r.first + best].move;
    }

    std::vector<Move> pv(int maxLen = 16) const {
        std::vector<Move> line;
        int cur = 0;
        while (int(line.size()) < maxLen && nodes[cur].first >= 0 && nodes[cur].nch) {
            const PNode& n = nodes[cur];
            int b = n.first;
            for (int k = 1; k < n.nch; ++k) if (nodes[n.first + k].N > nodes[b].N) b = n.first + k;
            if (nodes[b].N == 0) break;
            line.push_back(nodes[b].move);
            cur = b;
        }
        if (!line.empty()) line[0] = bestMove();
        return line;
    }

    void rootWDL(float out[3]) const {
        const PNode& r = nodes[0];
        if (r.proven == LOSS) { out[0] = 1; out[1] = 0; out[2] = 0; return; }
        if (r.proven == WIN) { out[0] = 0; out[1] = 0; out[2] = 1; return; }
        if (r.proven == DRAW) { out[0] = 0; out[1] = 1; out[2] = 0; return; }
        float q = -r.W / r.N, d = std::clamp(r.D / r.N, 0.f, 1.f);
        float w = std::max(0.f, (1 + q - d) * 0.5f), l = std::max(0.f, (1 - q - d) * 0.5f), s = w + d + l;
        out[0] = w / s; out[1] = d / s; out[2] = l / s;
    }

private:
    int select(int ni) const {
        const PNode& nd = nodes[ni];
        float sqrtN = std::sqrt(float(nd.N));
        float c = cInit + std::log((float(nd.N) + cBase + 1.f) / cBase);
        float visitedP = 0.f;
        for (int k = 0; k < nd.nch; ++k) if (nodes[nd.first + k].N > 0) visitedP += nodes[nd.first + k].prior;
        float fpu = -nd.W / std::max(1, nd.N) - fpuRed * std::sqrt(visitedP);
        int best = nd.first;
        float bs = -1e30f;
        for (int k = 0; k < nd.nch; ++k) {
            int idx = nd.first + k;
            const PNode& ch = nodes[idx];
            if (ch.proven == WIN) return idx;
            float q = ch.proven == LOSS ? -10.f : ch.proven == DRAW ? 0.f : ch.N > 0 ? ch.W / ch.N : fpu;
            float s = q + c * ch.prior * sqrtN / (1.f + ch.N);
            if (s > bs) { bs = s; best = idx; }
        }
        return best;
    }

    bool updateProven(int ni) {
        PNode& nd = nodes[ni];
        if (nd.proven != UNKNOWN || nd.first < 0) return false;
        bool all = true, anyDraw = false;
        for (int k = 0; k < nd.nch; ++k) {
            int8_t p = nodes[nd.first + k].proven;
            if (p == WIN) { nd.proven = LOSS; return true; }
            if (p == UNKNOWN) all = false;
            else if (p == DRAW) anyDraw = true;
        }
        if (!all) return false;
        nd.proven = anyDraw ? DRAW : WIN;
        return true;
    }

    bool repeated(const Position& p, const std::vector<u64>& hist) const {
        int S = int(hist.size());
        for (int i = S - 2; i >= 0 && i >= S - p.halfmove; i -= 2) if (hist[i] == p.key) return true;
        return false;
    }

    // path nodes (excluding root) carry a virtual loss: N += 1, W -= 1. Root carries N += 1.
    void backup(Ctx& c, float v, float d, bool newlyProven) {
        float val = -v;
        for (int i = int(c.path.size()) - 1; i >= 0; --i) {
            PNode& nd = nodes[c.path[i]];
            nd.W += (i > 0 ? 1.f : 0.f) + val;
            nd.D += d;
            val = -val;
            if (newlyProven && i > 0) newlyProven = updateProven(c.path[i - 1]);
        }
    }

    bool simulate(Ctx& c) {
        Position pos = root, prev;
        c.hist.assign(gameHist.begin(), gameHist.end());
        c.path.clear();
        c.path.push_back(0);
        int cur = 0, ai = -1, parentAcc = -1;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (nodes[0].proven != UNKNOWN) { stop = true; return false; }
            nodes[0].N += 1;
            while (nodes[cur].proven == UNKNOWN && nodes[cur].first >= 0) {
                int ch = select(cur);
                nodes[ch].N += 1;
                nodes[ch].W -= 1.f;
                c.hist.push_back(pos.key);
                prev = pos;
                pos.make(nodes[ch].move);
                cur = ch;
                c.path.push_back(ch);
            }
            if (nodes[cur].proven != UNKNOWN) {
                int8_t p = nodes[cur].proven;
                backup(c, p == WIN ? -1.f : p == LOSS ? 1.f : 0.f, p == DRAW ? 1.f : 0.f, false);
                return true;
            }
            if (nodes[cur].expanding) {   // another thread is evaluating this leaf: undo virtual loss
                for (size_t i = 0; i < c.path.size(); ++i) {
                    nodes[c.path[i]].N -= 1;
                    if (i > 0) nodes[c.path[i]].W += 1.f;
                }
                return false;
            }
            nodes[cur].expanding = 1;
            parentAcc = nodes[c.path[c.path.size() - 2]].acc;
            if (nAccs < accs.size()) ai = int(nAccs++);
        }
        // ── outside the lock: rules, accumulator, network ──
        Move ms[MAX_MOVES];
        int n = genLegal(pos, ms);
        int8_t term = UNKNOWN;
        float v = 0, d = 0;
        if (n == 0) { term = pos.inCheck() ? WIN : DRAW; v = pos.inCheck() ? -1.f : 0.f; d = term == DRAW; }
        else if (pos.halfmove >= 100 || pos.insufficientMaterial() || repeated(pos, c.hist)) { term = DRAW; v = 0; d = 1; }
        float pri[MAX_MOVES], wdl[3];
        if (term == UNKNOWN) {
            Acc& dst = ai >= 0 ? accs[ai] : c.tmp;
            if (parentAcc >= 0) updateAcc(*net, accs[parentAcc], prev, pos, dst);
            else refreshAcc(*net, pos, dst);
            evaluate(*net, dst, pos.stm, ms, n, pri, wdl, policyTemp);
            v = wdl[0] - wdl[2];
            d = wdl[1];
        }
        std::lock_guard<std::mutex> lk(mu);
        PNode& leaf = nodes[cur];
        leaf.expanding = 0;
        bool newlyProven = false;
        if (term != UNKNOWN) { leaf.proven = term; newlyProven = true; }
        else if (nNodes + size_t(n) <= nodes.size()) {
            size_t first = nNodes;
            nNodes += size_t(n);
            for (int i = 0; i < n; ++i) { nodes[first + i] = PNode{}; nodes[first + i].move = ms[i]; nodes[first + i].prior = pri[i]; }
            leaf.first = int32_t(first);
            leaf.nch = uint8_t(n);
            leaf.acc = ai;
        } else stop = true;   // tree memory (Hash) exhausted
        backup(c, v, d, newlyProven);
        return true;
    }
};

struct Engine {
    Net net;
    ParallelSearch S;
    int threads = 1, hashMB = 512;
    int64_t overhead = 50, fixedMoveTime = 0, fixedNodes = 0;
    std::string evalFile = "<embedded>";
    Position pos;
    std::vector<u64> hist;
    std::thread searchThread;

    Engine() {
        if (!loadNetMem(net, az_net_blob, size_t(az_net_blob_end - az_net_blob))) { std::cout << "info string embedded net invalid" << std::endl; std::exit(1); }
        net.prepareInference();
        S.net = &net;
        S.setHash(size_t(hashMB));
        pos.setStart();
    }

    void waitSearch() { if (searchThread.joinable()) { S.stop = true; searchThread.join(); } }

    static int cpOf(float q) {
        double x = std::clamp(double(q), -0.999, 0.999);
        return int(std::lround(400.0 * std::log10((1 + x) / (1 - x))));
    }

    void info(Clock::time_point t0) {
        float wdl[3];
        S.rootWDL(wdl);
        int64_t ms = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count());
        int cp = cpOf(wdl[0] - wdl[2]);
        if (S.nodes[0].proven == LOSS) cp = 20000;
        if (S.nodes[0].proven == WIN) cp = -20000;
        auto line = S.pv();
        std::string out = "info depth " + std::to_string(std::max<size_t>(1, line.size())) + " nodes " + std::to_string(S.sims.load()) +
                          " nps " + std::to_string(S.sims.load() * 1000 / ms) + " time " + std::to_string(ms) + " hashfull " +
                          std::to_string(S.nNodes * 1000 / S.nodes.size()) + " score cp " + std::to_string(cp) + " wdl " +
                          std::to_string(int(1000 * wdl[0])) + " " + std::to_string(int(1000 * wdl[1])) + " " + std::to_string(int(1000 * wdl[2])) + " pv";
        for (Move m : line) out += " " + moveStr(m);
        std::cout << out << std::endl;
    }

    void go(std::istringstream& is) {
        waitSearch();
        int64_t wt = -1, bt = -1, wi = 0, bi = 0, mtg = 0, mt = -1, nodes = -1;
        bool infinite = false;
        std::string tok;
        while (is >> tok) {
            if (tok == "infinite") infinite = true;
            else if (tok == "ponder") {}   // pondering not supported: searched as a normal move
            else {
                int64_t v;
                if (!(is >> v)) break;
                if (tok == "wtime") wt = v; else if (tok == "btime") bt = v; else if (tok == "winc") wi = v; else if (tok == "binc") bi = v;
                else if (tok == "movestogo") mtg = v; else if (tok == "movetime") mt = v; else if (tok == "nodes") nodes = v;
            }
        }
        Move ms[MAX_MOVES];
        if (!genLegal(pos, ms)) { std::cout << "bestmove 0000" << std::endl; return; }
        int64_t my = pos.stm == WHITE ? wt : bt, inc = pos.stm == WHITE ? wi : bi;
        int64_t budget = -1;   // ms
        if (!infinite) {
            if (mt > 0) budget = mt - overhead;
            else if (my >= 0) {
                int64_t moves = mtg > 0 ? std::min<int64_t>(mtg, 40) : 30;
                budget = my / moves + inc * 3 / 4 - overhead;
                budget = std::min(budget, my / 3 - overhead);
            } else if (fixedMoveTime > 0) budget = fixedMoveTime - overhead;
            if (budget >= 0) budget = std::max<int64_t>(budget, 5);
        }
        int64_t maxSims = nodes > 0 ? nodes : fixedNodes;
        if (!infinite && budget < 0 && maxSims <= 0) maxSims = 1000;   // bare "go": 1,000 playouts
        S.prepare(pos, hist);
        if (S.reusedVisits > 0) std::cout << "info string reused " << S.reusedVisits << " visits" << std::endl;
        searchThread = std::thread([this, budget, maxSims, infinite] {
            auto t0 = Clock::now();
            auto deadline = t0 + std::chrono::milliseconds(std::max<int64_t>(budget, 0));
            bool timed = budget >= 0;
            std::vector<std::thread> ws;
            for (int i = 1; i < threads; ++i) ws.emplace_back([&] { S.worker(maxSims, deadline, timed); });
            auto lastInfo = t0;
            std::thread main0([&] { S.worker(maxSims, deadline, timed); });
            while (!S.stop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (Clock::now() - lastInfo > std::chrono::seconds(1)) { std::lock_guard<std::mutex> lk(S.mu); info(t0); lastInfo = Clock::now(); }
            }
            main0.join();
            for (auto& w : ws) w.join();
            (void)infinite;
            info(t0);
            std::cout << "bestmove " << moveStr(S.bestMove()) << std::endl;
        });
    }

    void loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream is(line);
            std::string cmd;
            is >> cmd;
            if (cmd == "uci") {
                std::cout << "id name TA-NNUE-AZ (" << net.d.str() << ")\nid author Deepmath-0\n"
                          << "option name Hash type spin default 512 min 16 max 65536\n"
                          << "option name Threads type spin default 1 min 1 max 256\n"
                          << "option name MoveOverhead type spin default 50 min 0 max 5000\n"
                          << "option name MoveTime type spin default 0 min 0 max 3600000\n"
                          << "option name Nodes type spin default 0 min 0 max 2000000000\n"
                          << "option name EvalFile type string default <embedded>\n"
                          << "option name TreeReuse type check default true\n"
                          << "option name PolicyTemp type string default 1.6\n"
                          << "option name CPuct type string default 1.25\n"
                          << "uciok" << std::endl;
            } else if (cmd == "isready") std::cout << "readyok" << std::endl;
            else if (cmd == "setoption") {
                waitSearch();
                std::string tok, name, value;
                bool inName = false, inValue = false;
                while (is >> tok) {
                    if (tok == "name") { inName = true; inValue = false; continue; }
                    if (tok == "value") { inValue = true; inName = false; continue; }
                    if (inName) name += (name.empty() ? "" : " ") + tok;
                    else if (inValue) value += (value.empty() ? "" : " ") + tok;
                }
                try {
                    if (name == "Hash") { hashMB = std::stoi(value); S.setHash(size_t(hashMB)); S.haveTree = false; }
                    else if (name == "Threads") threads = std::clamp(std::stoi(value), 1, 256);
                    else if (name == "MoveOverhead") overhead = std::stoll(value);
                    else if (name == "MoveTime") fixedMoveTime = std::stoll(value);
                    else if (name == "Nodes") fixedNodes = std::stoll(value);
                    else if (name == "TreeReuse") { S.allowReuse = value == "true"; S.haveTree = false; }
                    else if (name == "PolicyTemp") { S.policyTemp = std::clamp(std::stof(value), 0.1f, 10.f); S.haveTree = false; }
                    else if (name == "CPuct") { S.cInit = std::clamp(std::stof(value), 0.1f, 20.f); S.haveTree = false; }
                    else if (name == "EvalFile") {
                        Net n2;
                        if (value == "<embedded>" ? loadNetMem(n2, az_net_blob, size_t(az_net_blob_end - az_net_blob)) : n2.load(value)) {
                            net = std::move(n2);
                            net.prepareInference();
                            S.haveTree = false;
                            evalFile = value;
                            std::cout << "info string loaded net " << net.d.str() << std::endl;
                        } else std::cout << "info string cannot load " << value << std::endl;
                    }
                } catch (...) { std::cout << "info string bad value for " << name << std::endl; }
            } else if (cmd == "ucinewgame") { waitSearch(); pos.setStart(); hist.clear(); }
            else if (cmd == "position") {
                waitSearch();
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
                    std::string m;
                    while (is >> m) {
                        Move mv = parseMove(pos, m);
                        if (!mv) break;
                        hist.push_back(pos.key);
                        pos.make(mv);
                    }
                }
            } else if (cmd == "go") go(is);
            else if (cmd == "stop") waitSearch();
            else if (cmd == "ponderhit") {}
            else if (cmd == "quit") break;
        }
        waitSearch();
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    static Engine e;
    e.loop();
    return 0;
}
