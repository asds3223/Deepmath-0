// nnue.hpp — TA-NNUE: sparse feature transformer (shared by both perspectives) + WDL value head + legal-sparse policy head.
// See docs/az/THEORY.md (P1 tree-accumulator, P2 legal-sparse readout, P3 perspective normalization, P4 WDL).
#pragma once
#include "chess.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace az {

constexpr int NF = 772;            // 2 (own/opp) x 6 pieces x 64 squares + 4 castling rights
constexpr int H = 128;             // accumulator width per perspective
constexpr int HV = 32, HP = 32;    // value / policy hidden widths
constexpr int NPOL = 4096 + 72;    // policy move indices
constexpr int MAX_FEAT = 40;

namespace L {
constexpr size_t FTW = 0;
constexpr size_t FTB = FTW + size_t(NF) * H;
constexpr size_t V1W = FTB + H;                 // [HV][2H]
constexpr size_t V1B = V1W + size_t(HV) * 2 * H;
constexpr size_t V2W = V1B + HV;                // [3][HV]
constexpr size_t V2B = V2W + 3 * HV;
constexpr size_t P1W = V2B + 3;                 // [HP][2H]
constexpr size_t P1B = P1W + size_t(HP) * 2 * H;
constexpr size_t PM  = P1B + HP;                // [NPOL][HP]
constexpr size_t PB  = PM + size_t(NPOL) * HP;  // [NPOL]
constexpr size_t TOTAL = PB + NPOL;
}  // namespace L

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
    std::vector<float> w = std::vector<float>(L::TOTAL, 0.f);

    void init(uint64_t seed) {
        std::mt19937_64 rng(seed);
        auto fill = [&](size_t off, size_t n, float sd) {
            std::normal_distribution<float> nd(0.f, sd);
            for (size_t i = 0; i < n; ++i) w[off + i] = nd(rng);
        };
        std::fill(w.begin(), w.end(), 0.f);
        fill(L::FTW, size_t(NF) * H, 0.1f);
        fill(L::V1W, size_t(HV) * 2 * H, std::sqrt(2.f / (2 * H)));
        fill(L::V2W, 3 * HV, 0.05f);
        fill(L::P1W, size_t(HP) * 2 * H, std::sqrt(2.f / (2 * H)));
        fill(L::PM, size_t(NPOL) * HP, 0.05f);
    }
    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t hdr[6] = {0x454E4E41u /*"ANNE"*/, 1, NF, H, HV, NPOL};
        f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        f.write(reinterpret_cast<const char*>(w.data()), std::streamsize(w.size() * sizeof(float)));
        return bool(f);
    }
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t hdr[6];
        f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        if (!f || hdr[0] != 0x454E4E41u || hdr[1] != 1 || hdr[2] != NF || hdr[3] != H || hdr[4] != HV || hdr[5] != NPOL) return false;
        f.read(reinterpret_cast<char*>(w.data()), std::streamsize(w.size() * sizeof(float)));
        return bool(f);
    }
};

struct alignas(64) Acc { float v[2][H]; };   // indexed by absolute perspective (WHITE / BLACK)

inline void refreshAcc(const Net& net, const Position& p, Acc& a) {
    int fs[MAX_FEAT];
    for (int c = 0; c < 2; ++c) {
        float* v = a.v[c];
        std::copy_n(&net.w[L::FTB], H, v);
        int n = featuresOf(p, c, fs);
        for (int i = 0; i < n; ++i) {
            const float* row = &net.w[L::FTW + size_t(fs[i]) * H];
            for (int j = 0; j < H; ++j) v[j] += row[j];
        }
    }
}

// P1: child accumulator = parent accumulator + feature difference (no unmake needed).
inline void updateAcc(const Net& net, const Acc& parent, const Position& before, const Position& after, Acc& out) {
    int fa[MAX_FEAT], fb[MAX_FEAT];
    for (int c = 0; c < 2; ++c) {
        int na = featuresOf(before, c, fa), nb = featuresOf(after, c, fb);
        float* v = out.v[c];
        std::copy_n(parent.v[c], H, v);
        int i = 0, k = 0;
        while (i < na || k < nb) {
            if (k >= nb || (i < na && fa[i] < fb[k])) {          // removed feature
                const float* row = &net.w[L::FTW + size_t(fa[i++]) * H];
                for (int j = 0; j < H; ++j) v[j] -= row[j];
            } else if (i >= na || fb[k] < fa[i]) {               // added feature
                const float* row = &net.w[L::FTW + size_t(fb[k++]) * H];
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
    alignas(64) float z[2 * H];
    for (int j = 0; j < H; ++j) { z[j] = clamp01(a.v[stm][j]); z[H + j] = clamp01(a.v[stm ^ 1][j]); }
    float hv[HV];
    for (int o = 0; o < HV; ++o) hv[o] = clamp01(w[L::V1B + o] + dot(&w[L::V1W + size_t(o) * 2 * H], z, 2 * H));
    for (int k = 0; k < 3; ++k) wdl[k] = w[L::V2B + k] + dot(&w[L::V2W + size_t(k) * HV], hv, HV);
    softmaxInPlace(wdl, 3);
    if (!n) return;
    float hp[HP];
    for (int o = 0; o < HP; ++o) hp[o] = clamp01(w[L::P1B + o] + dot(&w[L::P1W + size_t(o) * 2 * H], z, 2 * H));
    for (int i = 0; i < n; ++i) {
        int k = policyIndex(moves[i], stm);
        prior[i] = (w[L::PB + k] + dot(&w[L::PM + size_t(k) * HP], hp, HP)) / policyTemp;
    }
    softmaxInPlace(prior, n);
}

}  // namespace az
