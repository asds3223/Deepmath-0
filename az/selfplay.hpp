// selfplay.hpp — self-play game generation, training-sample storage, match games.
#pragma once
#include "mcts.hpp"
#include <cstdio>

namespace az {

// One training position. Policy targets live in Dataset::mv[off .. off+n) as (policyIndex | visits << 16).
struct SampleHdr {
    uint8_t board[64];
    uint8_t stm, castle;
    int8_t result;      // game result from stm's view: +1 win, 0 draw, -1 loss
    uint8_t flags;      // bit0: game hit the ply cap; bit1: value-only sample (fast search, no policy target)
    float root[3];      // root WDL from search (stm view)
    uint32_t off;
    uint16_t n, pad;
};
static_assert(sizeof(SampleHdr) == 88, "record layout");

struct Dataset {
    std::vector<SampleHdr> s;
    std::vector<uint32_t> mv;
    size_t size() const { return s.size(); }
    void append(const Dataset& o) {
        uint32_t base = uint32_t(mv.size());
        for (auto h : o.s) { h.off += base; s.push_back(h); }
        mv.insert(mv.end(), o.mv.begin(), o.mv.end());
    }
    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        uint64_t hdr[3] = {0x31445A41ull /*"AZD1"*/, s.size(), mv.size()};
        bool ok = std::fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
                  std::fwrite(s.data(), sizeof(SampleHdr), s.size(), f) == s.size() &&
                  std::fwrite(mv.data(), sizeof(uint32_t), mv.size(), f) == mv.size();
        return std::fclose(f) == 0 && ok;
    }
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        uint64_t hdr[3];
        bool ok = std::fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == 0x31445A41ull;
        if (ok) {
            s.resize(hdr[1]);
            mv.resize(hdr[2]);
            ok = std::fread(s.data(), sizeof(SampleHdr), s.size(), f) == s.size() &&
                 std::fread(mv.data(), sizeof(uint32_t), mv.size(), f) == mv.size();
        }
        std::fclose(f);
        return ok;
    }
};

inline Position fromSample(const SampleHdr& h) {
    Position p;
    p.clear();
    for (int s = 0; s < 64; ++s) if (h.board[s] != NO_PIECE) p.put(h.board[s], s);
    p.stm = h.stm;
    p.castle = h.castle;
    return p;
}

struct SelfplayConfig {
    SearchParams sp;
    int tempPlies = 30;        // AlphaZero: sample by visit count for the first 30 plies
    int maxPlies = 512;        // AlphaZero pseudocode max_moves
    float resignQ = 0.f;       // 0 = never resign; otherwise resign when q < -resignQ twice in a row
    float noResignFrac = 0.1f; // fraction of games where resignation is disabled (calibration)
    // P7 playout cap randomization (KataGo): with probability pFull a move gets the full search (sp.sims, root noise)
    // and is recorded; otherwise a fast search (fastSims, no noise) only plays the move. pFull = 1 → plain AlphaZero.
    float pFull = 1.f;
    int fastSims = 200;
    bool reuseTree = false;   // P7 tree reuse between moves (Lc0/KataGo); sims = total root visits
    bool fastValue = false;   // P7: record fast-search positions as value-only samples (z + fast root WDL)
};

struct GameStats {
    int plies = 0, result = 0;   // result from White's view
    Outcome how = Outcome::ONGOING;
    bool capped = false, resigned = false;
    int64_t sims = 0;
};

inline void recordSample(const Searcher& S, const Position& pos, Dataset& out) {
    SampleHdr h{};
    std::memcpy(h.board, pos.sq, 64);
    h.stm = pos.stm;
    h.castle = pos.castle;
    S.rootWDL(h.root);
    h.off = uint32_t(out.mv.size());
    const Node& r = S.rootNode();
    h.n = r.nch;
    for (int k = 0; k < r.nch; ++k) {
        const Node& c = S.child(r, k);
        uint32_t vis = uint32_t(std::min(c.N, 65535));
        out.mv.push_back(uint32_t(policyIndex(c.move, pos.stm)) | (vis << 16));
    }
    out.s.push_back(h);
}

inline GameStats playSelfplayGame(Searcher& S, const SelfplayConfig& cfg, Dataset& out) {
    GameStats gs;
    Position pos;
    pos.setStart();
    std::vector<u64> hist;
    size_t first = out.s.size();
    bool allowResign = cfg.resignQ > 0 && std::uniform_real_distribution<float>(0, 1)(S.rng) >= cfg.noResignFrac;
    int low[2] = {0, 0};
    S.sp = cfg.sp;
    int ply = 0;
    for (;; ++ply) {
        Move ms[MAX_MOVES];
        int n = genLegal(pos, ms);
        Outcome o = outcome(pos, hist, n);
        if (o != Outcome::ONGOING) {
            gs.how = o;
            gs.result = o == Outcome::CHECKMATE ? (pos.stm == WHITE ? -1 : 1) : 0;
            break;
        }
        if (ply >= cfg.maxPlies) { gs.capped = true; gs.result = 0; break; }
        Move m;
        if (n == 1) m = ms[0];   // forced: no search, no sample
        else {
            bool full = cfg.pFull >= 1.f || std::uniform_real_distribution<float>(0, 1)(S.rng) < cfg.pFull;
            S.sp = cfg.sp;
            if (!full) { S.sp.sims = cfg.fastSims; S.sp.noise = false; }
            gs.sims += S.search(pos, hist, cfg.reuseTree);
            if (full) recordSample(S, pos, out);
            else if (cfg.fastValue) {
                SampleHdr h{};
                std::memcpy(h.board, pos.sq, 64);
                h.stm = pos.stm;
                h.castle = pos.castle;
                S.rootWDL(h.root);
                h.off = uint32_t(out.mv.size());
                h.n = 0;
                h.flags = 2;
                out.s.push_back(h);
            }
            m = S.pickMove(ply < cfg.tempPlies ? 1.f : 0.f);
            if (allowResign) {
                float wdl[3];
                S.rootWDL(wdl);
                if (wdl[0] - wdl[2] < -cfg.resignQ) {
                    if (++low[pos.stm] >= 2) { gs.resigned = true; gs.result = pos.stm == WHITE ? -1 : 1; break; }
                } else low[pos.stm] = 0;
            }
        }
        if (cfg.reuseTree) S.advance(m);
        hist.push_back(pos.key);
        pos.make(m);
    }
    gs.plies = ply;
    for (size_t i = first; i < out.s.size(); ++i) {
        out.s[i].result = int8_t(out.s[i].stm == WHITE ? gs.result : -gs.result);
        out.s[i].flags = uint8_t((out.s[i].flags & 2) | (gs.capped ? 1 : 0));
    }
    return gs;
}

// Match game. net == nullptr → uniformly random legal mover. Returns result from White's view.
inline int playMatchGame(Searcher& sw, Searcher& sb, const SearchParams& sp, const std::vector<Move>& opening,
                         std::mt19937_64& rng, int maxPlies = 512, int* pliesOut = nullptr) {
    Position pos;
    pos.setStart();
    std::vector<u64> hist;
    for (Move m : opening) { hist.push_back(pos.key); pos.make(m); }
    int ply = int(opening.size()), res = 0;
    for (;; ++ply) {
        Move ms[MAX_MOVES];
        int n = genLegal(pos, ms);
        Outcome o = outcome(pos, hist, n);
        if (o != Outcome::ONGOING) { res = o == Outcome::CHECKMATE ? (pos.stm == WHITE ? -1 : 1) : 0; break; }
        if (ply >= maxPlies) break;
        Searcher& S = pos.stm == WHITE ? sw : sb;
        Move m;
        if (n == 1) m = ms[0];
        else if (!S.net) m = ms[std::uniform_int_distribution<int>(0, n - 1)(rng)];
        else {
            S.sp = sp;
            S.sp.noise = false;
            S.search(pos, hist);
            m = S.pickMove(0.f);
        }
        hist.push_back(pos.key);
        pos.make(m);
    }
    if (pliesOut) *pliesOut = ply;
    return res;
}

// Random opening of `plies` uniformly random legal moves that does not end the game.
inline std::vector<Move> randomOpening(int plies, std::mt19937_64& rng) {
    for (;;) {
        Position pos;
        pos.setStart();
        std::vector<Move> line;
        bool ok = true;
        for (int i = 0; i < plies && ok; ++i) {
            Move ms[MAX_MOVES];
            int n = genLegal(pos, ms);
            if (!n) { ok = false; break; }
            Move m = ms[std::uniform_int_distribution<int>(0, n - 1)(rng)];
            line.push_back(m);
            pos.make(m);
        }
        Move ms[MAX_MOVES];
        if (ok && genLegal(pos, ms) > 0) return line;
    }
}

}  // namespace az
