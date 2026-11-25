// nnue.hpp — TA-NNUE: sparse feature transformer (shared by both perspectives) + WDL value head + legal-sparse policy head.
// See docs/az/THEORY.md (P1 tree-accumulator, P2 legal-sparse readout, P3 perspective normalization, P4 WDL).
#pragma once
#include "chess.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace az {

constexpr int NF = 772;            // 2 (own/opp) x 6 pieces x 64 squares + 4 castling rights
constexpr int NPOL = 4096 + 72;    // policy move indices
constexpr int MAX_FEAT = 40;
constexpr int HMAX = 512, HVMAX = 128, HPMAX = 128;   // largest widths a stage may grow to (P6)

// Widths are runtime values so one lineage can grow in stages (P6). Stage 1 = 128 / 32 / 32.
struct Dims {
    int H = 128, HV = 32, HP = 32;
    bool operator==(const Dims& o) const { return H == o.H && HV == o.HV && HP == o.HP; }
    std::string str() const { return std::to_string(H) + "x" + std::to_string(HV) + "x" + std::to_string(HP); }
};

struct Layout {
    size_t FTW, FTB, V1W, V1B, V2W, V2B, P1W, P1B, PM, PB, TOTAL;
    explicit Layout(Dims d = {}) {
        FTW = 0;
        FTB = FTW + size_t(NF) * d.H;
        V1W = FTB + d.H;                       // [HV][2H]
        V1B = V1W + size_t(d.HV) * 2 * d.H;
        V2W = V1B + d.HV;                      // [3][HV]
        V2B = V2W + 3 * size_t(d.HV);
        P1W = V2B + 3;                         // [HP][2H]
        P1B = P1W + size_t(d.HP) * 2 * d.H;
        PM  = P1B + d.HP;                      // [NPOL][HP]
        PB  = PM + size_t(NPOL) * d.HP;        // [NPOL]
        TOTAL = PB + NPOL;
    }
};

inline float clamp01(float x) { return x < 0.f ? 0.f : x > 1.f ? 1.f : x; }

// Perspective-relative features, written in increasing order (needed by the diff in updateAcc).
inline int featuresOf(const Position& p, int persp, int* out) {
    int n = 0;
    for (int rel = 0; rel < 2; ++rel) {
        int c = rel == 0 ? persp : persp ^ 1;
        for (int t = 0; t < 6; ++t) {
            u64 b = p.bb[pieceOf(c, t)];
            if (persp == BLACK) b = flipRanks(b);
            int base = (rel * 6 + t) * 64;
            while (b) out[n++] = base + popLsb(b);
        }
    }
    const int ownK = persp == WHITE ? WK : BK, ownQ = persp == WHITE ? WQ : BQ;
    const int oppK = persp == WHITE ? BK : WK, oppQ = persp == WHITE ? BQ : WQ;
    if (p.castle & ownK) out[n++] = 768;
    if (p.castle & ownQ) out[n++] = 769;
    if (p.castle & oppK) out[n++] = 770;
    if (p.castle & oppQ) out[n++] = 771;
    return n;
}

// Side-to-move-relative policy index (P2/P3).
inline int policyIndex(Move m, int stm) {
    int f = mFrom(m), t = mTo(m), pr = mPromo(m);
    if (stm == BLACK) { f ^= 56; t ^= 56; }
    if (pr && pr != QUEEN) {
        int dir = (t & 7) - (f & 7) + 1;   // 0 capture left, 1 push, 2 capture right
        return 4096 + (pr - KNIGHT) * 24 + dir * 8 + (t & 7);
    }
    return f * 64 + t;
}

struct Net {
    Dims d;
    Layout L{d};
    std::vector<float> w = std::vector<float>(L.TOTAL, 0.f);

    void setDims(Dims nd) { d = nd; L = Layout(nd); w.assign(L.TOTAL, 0.f); }
    void init(uint64_t seed) {
        std::mt19937_64 rng(seed);
        auto fill = [&](size_t off, size_t n, float sd) {
            std::normal_distribution<float> nd(0.f, sd);
            for (size_t i = 0; i < n; ++i) w[off + i] = nd(rng);
        };
        std::fill(w.begin(), w.end(), 0.f);
        fill(L.FTW, size_t(NF) * d.H, 0.1f);
        fill(L.V1W, size_t(d.HV) * 2 * d.H, std::sqrt(2.f / (2 * d.H)));
        fill(L.V2W, 3 * size_t(d.HV), 0.05f);
        fill(L.P1W, size_t(d.HP) * 2 * d.H, std::sqrt(2.f / (2 * d.H)));
        fill(L.PM, size_t(NPOL) * d.HP, 0.05f);
    }
    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t hdr[7] = {0x454E4E41u /*"ANNE"*/, 2, NF, uint32_t(d.H), uint32_t(d.HV), uint32_t(d.HP), NPOL};
        f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        f.write(reinterpret_cast<const char*>(w.data()), std::streamsize(w.size() * sizeof(float)));
        return bool(f);
    }
    bool load(const std::string& path) {   // reads v2 and the stage-1 v1 format (H/HV in header, HP = 32)
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t h[6];
        f.read(reinterpret_cast<char*>(h), sizeof(h));
        if (!f || h[0] != 0x454E4E41u || h[2] != NF) return false;
        Dims nd;
        if (h[1] == 1) { if (h[5] != NPOL) return false; nd = {int(h[3]), int(h[4]), 32}; }
        else if (h[1] == 2) {
            uint32_t npol;
            f.read(reinterpret_cast<char*>(&npol), 4);
            if (!f || npol != NPOL) return false;
            nd = {int(h[3]), int(h[4]), int(h[5])};
        } else return false;
        if (nd.H < 1 || nd.H > HMAX || nd.HV < 1 || nd.HV > HVMAX || nd.HP < 1 || nd.HP > HPMAX) return false;
        setDims(nd);
        f.read(reinterpret_cast<char*>(w.data()), std::streamsize(w.size() * sizeof(float)));
        return bool(f);
    }
};

// P6: function-preserving widening. New units get random incoming weights and ZERO outgoing weights,
// so the widened net computes exactly the same value and policy. Returns old→new parameter index map
// (used to carry the optimizer state across the stage boundary).
inline std::vector<size_t> widenNet(const Net& o, Dims nd, Net& n, uint64_t seed) {
    const Dims od = o.d;
    n.setDims(nd);
    n.init(seed);   // random values everywhere; outgoing weights of new units are zeroed below
    const Layout &A = o.L, &B = n.L;
    std::vector<size_t> map(A.TOTAL);
    auto put = [&](size_t oi, size_t ni) { n.w[ni] = o.w[oi]; map[oi] = ni; };
    auto inIdx = [&](int i) { return i < od.H ? i : i - od.H + nd.H; };   // z = [stm(H), opp(H)]
    for (int f = 0; f < NF; ++f) for (int j = 0; j < od.H; ++j) put(A.FTW + size_t(f) * od.H + j, B.FTW + size_t(f) * nd.H + j);
    for (int j = 0; j < nd.H; ++j) n.w[B.FTB + j] = 0.f;
    for (int j = 0; j < od.H; ++j) put(A.FTB + j, B.FTB + j);
    auto hidden = [&](size_t aW, size_t aB, size_t bW, size_t bB, int oh, int nh) {
        for (int r = 0; r < nh; ++r) {
            n.w[bB + r] = 0.f;
            for (int i = 0; i < 2 * nd.H; ++i) {
                bool newInput = (i % nd.H) >= od.H;   // input from a new accumulator unit
                if (r < oh && newInput) n.w[bW + size_t(r) * 2 * nd.H + i] = 0.f;   // old unit must ignore new inputs
            }
        }
        for (int r = 0; r < oh; ++r) {
            for (int i = 0; i < 2 * od.H; ++i) put(aW + size_t(r) * 2 * od.H + i, bW + size_t(r) * 2 * nd.H + inIdx(i));
            put(aB + r, bB + r);
        }
    };
    hidden(A.V1W, A.V1B, B.V1W, B.V1B, od.HV, nd.HV);
    hidden(A.P1W, A.P1B, B.P1W, B.P1B, od.HP, nd.HP);
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < nd.HV; ++j) n.w[B.V2W + size_t(k) * nd.HV + j] = 0.f;   // new value units: zero outgoing
        for (int j = 0; j < od.HV; ++j) put(A.V2W + size_t(k) * od.HV + j, B.V2W + size_t(k) * nd.HV + j);
        put(A.V2B + k, B.V2B + k);
    }
    for (int m = 0; m < NPOL; ++m) {
        for (int j = 0; j < nd.HP; ++j) n.w[B.PM + size_t(m) * nd.HP + j] = 0.f;    // new policy units: zero outgoing
        for (int j = 0; j < od.HP; ++j) put(A.PM + size_t(m) * od.HP + j, B.PM + size_t(m) * nd.HP + j);
        put(A.PB + m, B.PB + m);
    }
    return map;
}

struct alignas(64) Acc { float v[2][HMAX]; };   // indexed by absolute perspective (WHITE / BLACK); first H used

inline void refreshAcc(const Net& net, const Position& p, Acc& a) {
    const int H = net.d.H;
    int fs[MAX_FEAT];
    for (int c = 0; c < 2; ++c) {
        float* v = a.v[c];
        std::copy_n(&net.w[net.L.FTB], H, v);
        int n = featuresOf(p, c, fs);
        for (int i = 0; i < n; ++i) {
            const float* row = &net.w[net.L.FTW + size_t(fs[i]) * H];
            for (int j = 0; j < H; ++j) v[j] += row[j];
        }
    }
}

// P1: child accumulator = parent accumulator + feature difference (no unmake needed).
inline void updateAcc(const Net& net, const Acc& parent, const Position& before, const Position& after, Acc& out) {
    const int H = net.d.H;
    int fa[MAX_FEAT], fb[MAX_FEAT];
    for (int c = 0; c < 2; ++c) {
        int na = featuresOf(before, c, fa), nb = featuresOf(after, c, fb);
        float* v = out.v[c];
        std::copy_n(parent.v[c], H, v);
        int i = 0, k = 0;
        while (i < na || k < nb) {
            if (k >= nb || (i < na && fa[i] < fb[k])) {          // removed feature
                const float* row = &net.w[net.L.FTW + size_t(fa[i++]) * H];
                for (int j = 0; j < H; ++j) v[j] -= row[j];
            } else if (i >= na || fb[k] < fa[i]) {               // added feature
                const float* row = &net.w[net.L.FTW + size_t(fb[k++]) * H];
                for (int j = 0; j < H; ++j) v[j] += row[j];
            } else { ++i; ++k; }
        }
    }
}

inline float dot(const float* a, const float* b, int n) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline void softmaxInPlace(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    float s = 0.f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; ++i) x[i] /= s;
}

// Forward pass from an accumulator. Writes WDL (stm view) and softmax priors over the given legal moves.
inline void evaluate(const Net& net, const Acc& a, int stm, const Move* moves, int n, float* prior, float wdl[3],
                     float policyTemp = 1.f) {
    const float* w = net.w.data();
    const Layout& L = net.L;
    const int H = net.d.H, HV = net.d.HV, HP = net.d.HP;
    alignas(64) float z[2 * HMAX];
    for (int j = 0; j < H; ++j) { z[j] = clamp01(a.v[stm][j]); z[H + j] = clamp01(a.v[stm ^ 1][j]); }
    float hv[HVMAX];
    for (int o = 0; o < HV; ++o) hv[o] = clamp01(w[L.V1B + o] + dot(&w[L.V1W + size_t(o) * 2 * H], z, 2 * H));
    for (int k = 0; k < 3; ++k) wdl[k] = w[L.V2B + k] + dot(&w[L.V2W + size_t(k) * HV], hv, HV);
    softmaxInPlace(wdl, 3);
    if (!n) return;
    float hp[HPMAX];
    for (int o = 0; o < HP; ++o) hp[o] = clamp01(w[L.P1B + o] + dot(&w[L.P1W + size_t(o) * 2 * H], z, 2 * H));
    for (int i = 0; i < n; ++i) {
        int k = policyIndex(moves[i], stm);
        prior[i] = (w[L.PB + k] + dot(&w[L.PM + size_t(k) * HP], hp, HP)) / policyTemp;
    }
    softmaxInPlace(prior, n);
}

}  // namespace az
