// train.hpp — loss, hand-written backprop and AdamW for TA-NNUE (docs/az/THEORY.md §4).
#pragma once
#include "selfplay.hpp"
#include <functional>
#include <thread>

namespace az {

struct TrainConfig {
    int batch = 512;
    float lr = 1e-3f, wd = 1e-4f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float valueW = 1.f;
    float lambda = 0.3f;   // P4: value target = (1-λ)·result + λ·root WDL
    int threads = 4;
};

struct LossStats {
    double pol = 0, val = 0, polTargetEnt = 0, top1 = 0;
    int64_t n = 0;
    void add(const LossStats& o) { pol += o.pol; val += o.val; polTargetEnt += o.polTargetEnt; top1 += o.top1; n += o.n; }
};

// Accumulates d(loss)/d(w) for one sample into g (not divided by batch). Returns per-sample stats in st.
inline void sampleGrad(const Net& net, const SampleHdr& h, const uint32_t* mv, float* g, LossStats& st,
                       const TrainConfig& cfg) {
    const float* w = net.w.data();
    const Layout& L = net.L;
    const int H = net.d.H, HV = net.d.HV, HP = net.d.HP;
    Position p = fromSample(h);
    int fs[2][MAX_FEAT], nf[2];
    nf[0] = featuresOf(p, WHITE, fs[0]);
    nf[1] = featuresOf(p, BLACK, fs[1]);
    const int stm = h.stm;

    alignas(64) float acc[2][HMAX], z[2 * HMAX], dz[2 * HMAX];
    for (int c = 0; c < 2; ++c) {
        std::copy_n(&w[L.FTB], H, acc[c]);
        for (int i = 0; i < nf[c]; ++i) {
            const float* row = &w[L.FTW + size_t(fs[c][i]) * H];
            for (int j = 0; j < H; ++j) acc[c][j] += row[j];
        }
    }
    for (int j = 0; j < H; ++j) { z[j] = clamp01(acc[stm][j]); z[H + j] = clamp01(acc[stm ^ 1][j]); }
    std::fill_n(dz, 2 * H, 0.f);

    // ── value head (WDL cross-entropy) ──
    float av[HVMAX], hv[HVMAX], out[3], dhv[HVMAX] = {};
    for (int o = 0; o < HV; ++o) { av[o] = w[L.V1B + o] + dot(&w[L.V1W + size_t(o) * 2 * H], z, 2 * H); hv[o] = clamp01(av[o]); }
    for (int k = 0; k < 3; ++k) out[k] = w[L.V2B + k] + dot(&w[L.V2W + size_t(k) * HV], hv, HV);
    softmaxInPlace(out, 3);
    float res[3] = {h.result > 0 ? 1.f : 0.f, h.result == 0 ? 1.f : 0.f, h.result < 0 ? 1.f : 0.f};
    float tv[3], dout[3];
    for (int k = 0; k < 3; ++k) {
        tv[k] = (1 - cfg.lambda) * res[k] + cfg.lambda * h.root[k];
        st.val -= tv[k] > 0 ? tv[k] * std::log(std::max(out[k], 1e-12f)) : 0.f;
        dout[k] = (out[k] - tv[k]) * cfg.valueW;
        g[L.V2B + k] += dout[k];
        for (int j = 0; j < HV; ++j) {
            g[L.V2W + size_t(k) * HV + j] += dout[k] * hv[j];
            dhv[j] += dout[k] * w[L.V2W + size_t(k) * HV + j];
        }
    }
    for (int o = 0; o < HV; ++o) {
        float da = (av[o] > 0.f && av[o] < 1.f) ? dhv[o] : 0.f;
        if (da == 0.f) continue;
        g[L.V1B + o] += da;
        float* gw = &g[L.V1W + size_t(o) * 2 * H];
        const float* ww = &w[L.V1W + size_t(o) * 2 * H];
        for (int i = 0; i < 2 * H; ++i) { gw[i] += da * z[i]; dz[i] += da * ww[i]; }
    }

    // ── policy head (cross-entropy vs visit distribution over legal moves) ──
    float vsum = 0.f;
    for (int i = 0; i < h.n; ++i) vsum += float(mv[i] >> 16);
    if (h.n > 0 && vsum > 0.f) {
        float ap[HPMAX], hp[HPMAX], dhp[HPMAX] = {};
        for (int o = 0; o < HP; ++o) { ap[o] = w[L.P1B + o] + dot(&w[L.P1W + size_t(o) * 2 * H], z, 2 * H); hp[o] = clamp01(ap[o]); }
        float q[MAX_MOVES];
        int bestT = 0, bestQ = 0;
        for (int i = 0; i < h.n; ++i) {
            int k = int(mv[i] & 0xFFFF);
            q[i] = w[L.PB + k] + dot(&w[L.PM + size_t(k) * HP], hp, HP);
        }
        softmaxInPlace(q, h.n);
        for (int i = 0; i < h.n; ++i) {
            int k = int(mv[i] & 0xFFFF);
            float pi = float(mv[i] >> 16) / vsum;
            if (pi > 0.f) { st.pol -= pi * std::log(std::max(q[i], 1e-12f)); st.polTargetEnt -= pi * std::log(pi); }
            if ((mv[i] >> 16) > (mv[bestT] >> 16)) bestT = i;
            if (q[i] > q[bestQ]) bestQ = i;
            float dl = q[i] - pi;
            g[L.PB + k] += dl;
            float* gm = &g[L.PM + size_t(k) * HP];
            const float* wm = &w[L.PM + size_t(k) * HP];
            for (int j = 0; j < HP; ++j) { gm[j] += dl * hp[j]; dhp[j] += dl * wm[j]; }
        }
        st.top1 += bestT == bestQ;
        for (int o = 0; o < HP; ++o) {
            float da = (ap[o] > 0.f && ap[o] < 1.f) ? dhp[o] : 0.f;
            if (da == 0.f) continue;
            g[L.P1B + o] += da;
            float* gw = &g[L.P1W + size_t(o) * 2 * H];
            const float* ww = &w[L.P1W + size_t(o) * 2 * H];
            for (int i = 0; i < 2 * H; ++i) { gw[i] += da * z[i]; dz[i] += da * ww[i]; }
        }
    }

    // ── feature transformer (sparse rows) ──
    for (int c = 0; c < 2; ++c) {
        const float* dzc = c == stm ? dz : dz + H;
        float da[HMAX];
        for (int j = 0; j < H; ++j) da[j] = (acc[c][j] > 0.f && acc[c][j] < 1.f) ? dzc[j] : 0.f;
        for (int j = 0; j < H; ++j) g[L.FTB + j] += da[j];
        for (int i = 0; i < nf[c]; ++i) {
            float* gr = &g[L.FTW + size_t(fs[c][i]) * H];
            for (int j = 0; j < H; ++j) gr[j] += da[j];
        }
    }
    st.n += 1;
}

struct Trainer {
    TrainConfig cfg;
    std::vector<float> m, v;
    int64_t t = 0;
    std::vector<std::vector<float>> grads;

    Trainer(const TrainConfig& c, size_t total) : cfg(c), m(total, 0.f), v(total, 0.f) {}

    // v2: magic, parameter count, t, m, v.  Also reads the stage-1 format (t, m, v) when the size matches.
    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        uint64_t hdr[2] = {0x324D4441ull /*"ADM2"*/, m.size()};
        bool ok = std::fwrite(hdr, sizeof(hdr), 1, f) == 1 && std::fwrite(&t, sizeof(t), 1, f) == 1 &&
                  std::fwrite(m.data(), 4, m.size(), f) == m.size() && std::fwrite(v.data(), 4, v.size(), f) == v.size();
        return std::fclose(f) == 0 && ok;
    }
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        uint64_t hdr[2] = {0, 0};
        bool ok = std::fread(hdr, sizeof(hdr), 1, f) == 1;
        if (ok && hdr[0] == 0x324D4441ull) ok = hdr[1] == m.size();
        else if (ok) {   // stage-1 format: no header
            std::fseek(f, 0, SEEK_END);
            ok = size_t(std::ftell(f)) == sizeof(t) + 8 * m.size();
            std::fseek(f, 0, SEEK_SET);
        }
        ok = ok && std::fread(&t, sizeof(t), 1, f) == 1 && std::fread(m.data(), 4, m.size(), f) == m.size() &&
             std::fread(v.data(), 4, v.size(), f) == v.size();
        std::fclose(f);
        return ok;
    }
    // Carry the optimizer state across a widening (new parameters start with zero moments).
    void remap(const std::vector<size_t>& map, size_t newTotal) {
        std::vector<float> m2(newTotal, 0.f), v2(newTotal, 0.f);
        for (size_t i = 0; i < map.size(); ++i) { m2[map[i]] = m[i]; v2[map[i]] = v[i]; }
        m.swap(m2);
        v.swap(v2);
    }

    // One optimizer step on a batch of (dataset, sample) indices.
    LossStats step(Net& net, const std::vector<std::pair<const Dataset*, size_t>>& batch) {
        int T = std::max(1, cfg.threads);
        grads.resize(T);
        std::vector<LossStats> st(T);
        std::vector<std::thread> th;
        for (int ti = 0; ti < T; ++ti) {
            th.emplace_back([&, ti] {
                auto& g = grads[ti];
                g.assign(net.L.TOTAL, 0.f);
                for (size_t b = ti; b < batch.size(); b += T) {
                    const Dataset& d = *batch[b].first;
                    const SampleHdr& h = d.s[batch[b].second];
                    sampleGrad(net, h, &d.mv[h.off], g.data(), st[ti], cfg);
                }
            });
        }
        for (auto& x : th) x.join();
        LossStats tot;
        for (auto& s : st) tot.add(s);
        // AdamW (decoupled weight decay)
        ++t;
        const float inv = 1.f / float(batch.size());
        const float bc1 = 1.f - std::pow(cfg.beta1, float(t)), bc2 = 1.f - std::pow(cfg.beta2, float(t));
        const float lr = cfg.lr;
        float* w = net.w.data();
        for (size_t i = 0; i < net.L.TOTAL; ++i) {
            float gi = 0.f;
            for (int ti = 0; ti < T; ++ti) gi += grads[ti][i];
            gi *= inv;
            m[i] = cfg.beta1 * m[i] + (1 - cfg.beta1) * gi;
            v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * gi * gi;
            float mh = m[i] / bc1, vh = v[i] / bc2;
            w[i] -= lr * (mh / (std::sqrt(vh) + cfg.eps) + cfg.wd * w[i]);
        }
        return tot;
    }

    // `steps` steps with uniform sampling over all positions of the window.
    // onLog(stepsDone, statsSinceLastLog) is called every `logEvery` steps and at the end (if logEvery > 0).
    LossStats train(Net& net, const std::vector<const Dataset*>& window, int steps, std::mt19937_64& rng, int logEvery = 0,
                    const std::function<void(int, const LossStats&)>& onLog = {}) {
        std::vector<size_t> cum;
        size_t total = 0;
        for (auto* d : window) { total += d->size(); cum.push_back(total); }
        LossStats all;
        if (!total) return all;
        std::uniform_int_distribution<size_t> U(0, total - 1);
        std::vector<std::pair<const Dataset*, size_t>> batch(size_t(cfg.batch));
        LossStats recent;
        for (int s = 0; s < steps; ++s) {
            for (auto& b : batch) {
                size_t x = U(rng);
                size_t di = size_t(std::upper_bound(cum.begin(), cum.end(), x) - cum.begin());
                b = {window[di], x - (di ? cum[di - 1] : 0)};
            }
            LossStats st = step(net, batch);
            all.add(st);
            recent.add(st);
            if (onLog && logEvery > 0 && ((s + 1) % logEvery == 0 || s + 1 == steps)) { onLog(s + 1, recent); recent = LossStats{}; }
        }
        return all;
    }
};

}  // namespace az
