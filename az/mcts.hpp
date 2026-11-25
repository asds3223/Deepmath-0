// mcts.hpp — PUCT-MCTS with tree-stored accumulators (P1) and MCTS-solver proof propagation (P5).
#pragma once
#include "nnue.hpp"
#include <chrono>

namespace az {

struct SearchParams {
    int sims = 1000;
    float cInit = 1.25f, cBase = 19652.f;   // AlphaZero pseudocode
    float fpuRed = 0.25f;                   // first-play-urgency reduction
    bool noise = false;
    float dirAlpha = 0.3f, dirEps = 0.25f;  // AlphaZero chess root noise
    float policyTemp = 1.f;
    int64_t maxMs = 0;                      // optional wall-clock limit (0 = none)
};

// Proof status, from the view of the player who made the move INTO the node.
enum : int8_t { UNKNOWN = 0, WIN = 1, LOSS = -1, DRAW = 2 };

struct Node {
    float prior = 0.f, W = 0.f, D = 0.f;    // W: value sum (mover's view), D: draw-probability sum
    int32_t N = 0, first = -1, acc = -1;
    Move move = 0;
    uint8_t nch = 0;
    int8_t proven = UNKNOWN;
};

class Searcher {
public:
    const Net* net;
    SearchParams sp;
    std::mt19937_64 rng;
    std::vector<Node> nodes;
    std::vector<Acc> accs;
    std::vector<u64> hist;
    std::vector<int> path;
    Position root;
    int64_t evals = 0;

    explicit Searcher(const Net* n, uint64_t seed = 1) : net(n), rng(seed) {}

    // Caller guarantees the root is not terminal (has legal moves, game not over).
    int search(const Position& rootPos, const std::vector<u64>& gameHist) {
        auto t0 = std::chrono::steady_clock::now();
        root = rootPos;
        hist = gameHist;
        nodes.clear();
        accs.clear();
        nodes.reserve(size_t(sp.sims) * 40 + 256);
        accs.reserve(size_t(sp.sims) + 2);   // every simulation expands at most one node → no reallocation
        nodes.emplace_back();
        float v, d;
        expand(0, root, nullptr, -1, v, d);
        nodes[0].N = 1; nodes[0].W = -v; nodes[0].D = d;
        if (sp.noise) addNoise();
        int done = 0;
        for (; done < sp.sims && nodes[0].proven == UNKNOWN; ++done) {
            simulate();
            if (sp.maxMs && (done & 31) == 31 &&
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() >= sp.maxMs)
                break;
        }
        return done;
    }

    const Node& rootNode() const { return nodes[0]; }
    const Node& child(const Node& n, int k) const { return nodes[size_t(n.first) + k]; }

    // Root WDL from the side to move.
    void rootWDL(float out[3]) const {
        const Node& r = nodes[0];
        if (r.proven == LOSS) { out[0] = 1; out[1] = 0; out[2] = 0; return; }
        if (r.proven == WIN)  { out[0] = 0; out[1] = 0; out[2] = 1; return; }
        if (r.proven == DRAW) { out[0] = 0; out[1] = 1; out[2] = 0; return; }
        float q = -r.W / r.N, d = r.D / r.N;
        float w = std::max(0.f, (1 + q - d) * 0.5f), l = std::max(0.f, (1 - q - d) * 0.5f);
        d = std::max(0.f, d);
        float s = w + d + l;
        out[0] = w / s; out[1] = d / s; out[2] = l / s;
    }

    // temp > 0: sample ∝ N^(1/temp); temp == 0: most visits. Proven wins first, proven losses last.
    Move pickMove(float temp) {
        const Node& r = nodes[0];
        int best = -1;
        for (int k = 0; k < r.nch; ++k) {
            const Node& c = child(r, k);
            if (c.proven == WIN && (best < 0 || c.N > child(r, best).N)) best = k;
        }
        if (best >= 0) return child(r, best).move;
        bool anyOk = false;
        for (int k = 0; k < r.nch; ++k) anyOk |= child(r, k).proven != LOSS;
        auto ok = [&](const Node& c) { return !anyOk || c.proven != LOSS; };
        if (temp > 0) {
            double wsum = 0, wts[MAX_MOVES];
            for (int k = 0; k < r.nch; ++k) {
                const Node& c = child(r, k);
                wts[k] = ok(c) && c.N > 0 ? std::pow(double(c.N), 1.0 / temp) : 0.0;
                wsum += wts[k];
            }
            if (wsum > 0) {
                double x = std::uniform_real_distribution<double>(0, wsum)(rng);
                for (int k = 0; k < r.nch; ++k) { if ((x -= wts[k]) <= 0 && wts[k] > 0) return child(r, k).move; }
            }
        }
        for (int k = 0; k < r.nch; ++k) {
            const Node& c = child(r, k);
            if (!ok(c)) continue;
            if (best < 0 || c.N > child(r, best).N || (c.N == child(r, best).N && c.prior > child(r, best).prior)) best = k;
        }
        return child(r, best).move;
    }

    std::vector<Move> pv(int maxLen = 12) const {
        std::vector<Move> line;
        int cur = 0;
        while (int(line.size()) < maxLen && nodes[cur].first >= 0 && nodes[cur].nch) {
            const Node& n = nodes[cur];
            int b = n.first;
            for (int k = 1; k < n.nch; ++k) if (nodes[n.first + k].N > nodes[b].N) b = n.first + k;
            if (nodes[b].N == 0) break;
            line.push_back(nodes[b].move);
            cur = b;
        }
        return line;
    }

private:
    bool repeated(const Position& p) const {   // twofold inside the tree / against game history
        int S = int(hist.size());
        for (int i = S - 2; i >= 0 && i >= S - p.halfmove; i -= 2) if (hist[i] == p.key) return true;
        return false;
    }

    // Returns true when `pos` is terminal (node is then proven and gets no children).
    bool expand(int ni, const Position& pos, const Position* parentPos, int parentAcc, float& v, float& d) {
        Move ms[MAX_MOVES];
        int n = genLegal(pos, ms);
        if (ni != 0) {
            if (n == 0) {
                if (pos.inCheck()) { v = -1; d = 0; nodes[ni].proven = WIN; }
                else { v = 0; d = 1; nodes[ni].proven = DRAW; }
                return true;
            }
            if (pos.halfmove >= 100 || pos.insufficientMaterial() || repeated(pos)) {
                v = 0; d = 1; nodes[ni].proven = DRAW;
                return true;
            }
        }
        int ai = int(accs.size());
        accs.emplace_back();
        if (parentPos) updateAcc(*net, accs[parentAcc], *parentPos, pos, accs[ai]);
        else refreshAcc(*net, pos, accs[ai]);
        float pri[MAX_MOVES], wdl[3];
        evaluate(*net, accs[ai], pos.stm, ms, n, pri, wdl, sp.policyTemp);
        ++evals;
        v = wdl[0] - wdl[2];
        d = wdl[1];
        int first = int(nodes.size());
        for (int i = 0; i < n; ++i) {
            Node c;
            c.move = ms[i];
            c.prior = pri[i];
            nodes.push_back(c);
        }
        Node& nd = nodes[ni];
        nd.first = first; nd.nch = uint8_t(n); nd.acc = ai;
        return false;
    }

    int select(int ni) const {
        const Node& nd = nodes[ni];
        float sqrtN = std::sqrt(float(nd.N));
        float c = sp.cInit + std::log((float(nd.N) + sp.cBase + 1.f) / sp.cBase);
        float visitedP = 0.f;
        for (int k = 0; k < nd.nch; ++k) if (nodes[nd.first + k].N > 0) visitedP += nodes[nd.first + k].prior;
        float fpu = -nd.W / nd.N - sp.fpuRed * std::sqrt(visitedP);
        int best = nd.first;
        float bs = -1e30f;
        for (int k = 0; k < nd.nch; ++k) {
            int idx = nd.first + k;
            const Node& ch = nodes[idx];
            if (ch.proven == WIN) return idx;
            float q = ch.proven == LOSS ? -10.f : ch.proven == DRAW ? 0.f : ch.N > 0 ? ch.W / ch.N : fpu;
            float s = q + c * ch.prior * sqrtN / (1.f + ch.N);
            if (s > bs) { bs = s; best = idx; }
        }
        return best;
    }

    bool updateProven(int ni) {
        Node& nd = nodes[ni];
        if (nd.proven != UNKNOWN) return false;
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

    void simulate() {
        Position pos = root, prev;
        size_t hb = hist.size();
        path.clear();
        path.push_back(0);
        int cur = 0;
        while (nodes[cur].proven == UNKNOWN && nodes[cur].first >= 0) {
            int c = select(cur);
            hist.push_back(pos.key);
            prev = pos;
            pos.make(nodes[c].move);
            cur = c;
            path.push_back(c);
        }
        float v, d;
        bool newlyProven = false;
        if (nodes[cur].proven != UNKNOWN) {
            int8_t p = nodes[cur].proven;
            v = p == WIN ? -1.f : p == LOSS ? 1.f : 0.f;
            d = p == DRAW ? 1.f : 0.f;
        } else {
            newlyProven = expand(cur, pos, &prev, nodes[path[path.size() - 2]].acc, v, d);
        }
        hist.resize(hb);
        float val = -v;   // value for the player who moved into the leaf
        for (int i = int(path.size()) - 1; i >= 0; --i) {
            Node& nd = nodes[path[i]];
            nd.N += 1; nd.W += val; nd.D += d;
            val = -val;
            if (newlyProven && i > 0) newlyProven = updateProven(path[i - 1]);
        }
    }

    void addNoise() {
        Node& r = nodes[0];
        std::gamma_distribution<float> g(sp.dirAlpha, 1.f);
        float buf[MAX_MOVES], s = 0.f;
        for (int k = 0; k < r.nch; ++k) { buf[k] = g(rng); s += buf[k]; }
        if (s <= 0.f) return;
        for (int k = 0; k < r.nch; ++k) {
            Node& c = nodes[r.first + k];
            c.prior = (1.f - sp.dirEps) * c.prior + sp.dirEps * buf[k] / s;
        }
    }
};

}  // namespace az
