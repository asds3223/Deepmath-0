// train.cpp — Rapfi NNUE → 고전 평가 증류 학습기 (희소 선형 모델, 동기식 데이터 병렬 + Adam, 코사인 감쇠)
//
//   가치: BCE( sigmoid(v), NNUE 승률 )
//   정책: softmax CE( 후보 칸 점수, NNUE 정책(후보로 제한·재정규화) ),  --pbal k 면 균형 국면 가중 1 + k·4p(1-p)
//
//   train --data a.bin,b.bin [--val v.bin] --out model.dce [--init old.dce] [--value 0|1] [--policy 0|1]
//         [--epochs 6] [--threads N] [--batch 1024] [--lr 0.02] [--l2 1e-6] [--pbal 0] [--seed 1]
#include "args.hpp"
#include "metrics.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>

using namespace gmk;

namespace {

struct Grad {
    std::vector<float> gv, gp;
    double vloss = 0, ploss = 0; size_t vn = 0, pn = 0;
    void reset(size_t nv, size_t np) { gv.assign(nv, 0.f); gp.assign(np, 0.f); vloss = ploss = 0; vn = pn = 0; }
    void add(const Grad& o) {
        for (size_t i = 0; i < gv.size(); ++i) gv[i] += o.gv[i];
        for (size_t i = 0; i < gp.size(); ++i) gp[i] += o.gp[i];
        vloss += o.vloss; ploss += o.ploss; vn += o.vn; pn += o.pn;
    }
};

struct Opts { bool value, policy; float pbal; };

void accumulate(const Model& mdl, const Record* recs, const uint32_t* idx, size_t n, const Opts& o, Grad& g) {
    Board b;                          // 학습은 모델을 직접 호출 (증분 상태 불필요)
    int moves[NMOVES]; float tgt[NMOVES], sc[NMOVES];
    for (size_t k = 0; k < n; ++k) {
        const Record& r = recs[idx[k]];
        r.toBoard(b);
        const float p = r.nnWinrate();
        if (o.value) {
            const float q = metrics::sigmoid(mdl.valueLogit(b));
            const float t = feat::taper(b), g0 = (q - p) * (1 - t), g1 = (q - p) * t;
            mdl.forValue(b, [&](int f) { g.gv[2 * f] += g0; g.gv[2 * f + 1] += g1; });
            g.vloss += -(p * std::log(std::max(q, 1e-7f)) + (1 - p) * std::log(std::max(1 - q, 1e-7f)));
            ++g.vn;
        }
        if (o.policy && !(r.flags & REC_OPPO_FIVE)) {
            const int nc = metrics::policyTarget(b, r, moves, tgt);
            if (nc < 2) continue;
            const int tc = feat::threatClass(b);
            float mx = -1e30f;
            for (int c = 0; c < nc; ++c) { sc[c] = mdl.policyScore(b, toCell(moves[c]), tc); mx = std::max(mx, sc[c]); }
            double z = 0; for (int c = 0; c < nc; ++c) z += std::exp(sc[c] - mx);
            const float w = 1 + o.pbal * 4 * p * (1 - p);
            for (int c = 0; c < nc; ++c) {
                const float pr = float(std::exp(sc[c] - mx) / z), gp = w * (pr - tgt[c]);
                if (tgt[c] > 0) g.ploss += -tgt[c] * std::log(std::max(pr, 1e-9f));
                if (gp != 0) mdl.forPolicy(b, toCell(moves[c]), tc, [&](int f) { g.gp[f] += gp; });
            }
            ++g.pn;
        }
    }
}

struct Adam {
    std::vector<float> m, v; float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f; long t = 0;
    void init(size_t n) { m.assign(n, 0.f); v.assign(n, 0.f); }
    void step(std::vector<float>& w, const std::vector<float>& g, float lr, float l2, float scale) {
        ++t;
        const float c1 = 1 - std::pow(b1, float(t)), c2 = 1 - std::pow(b2, float(t));
        for (size_t i = 0; i < w.size(); ++i) {
            const float gi = g[i] * scale + l2 * w[i];
            if (gi == 0 && m[i] == 0) continue;
            m[i] = b1 * m[i] + (1 - b1) * gi;
            v[i] = b2 * v[i] + (1 - b2) * gi * gi;
            w[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
        }
    }
};

} // namespace

int main(int argc, char** argv) {
    const Args a(argc, argv);
    const auto dataFiles = a.list("data");
    if (dataFiles.empty()) { std::fprintf(stderr, "usage: train --data f1,f2 [--val v] --out model.dce\n"); return 2; }
    const std::string out = a.str("out", "model.dce");
    const int    epochs  = int(a.num("epochs", 6));
    const int    threads = int(a.num("threads", std::max(1u, std::thread::hardware_concurrency())));
    const size_t batch   = size_t(a.num("batch", 1024));
    const float  lr0     = float(a.num("lr", 0.02));
    const float  l2      = float(a.num("l2", 1e-6));
    const Opts   o{a.num("value", 1) != 0, a.num("policy", 1) != 0, float(a.num("pbal", 0))};

    std::vector<Record> recs, val;
    for (auto& f : dataFiles) if (!readRecords(f, recs)) { std::fprintf(stderr, "cannot read %s\n", f.c_str()); return 1; }
    recs.erase(std::remove_if(recs.begin(), recs.end(), [](const Record& r) { return r.flags & REC_SELF_FIVE; }), recs.end());
    if (a.has("val")) { for (auto& f : a.list("val")) readRecords(f, val); }
    else {
        // 같은 대국의 국면이 연속으로 저장되므로 무작위 분할은 누설 → 파일 끝 부분을 검증셋으로
        const size_t nv = size_t(recs.size() * a.num("val-frac", 0.05));
        val.assign(recs.end() - nv, recs.end()); recs.resize(recs.size() - nv);
    }
    const auto& L = feat::Layout::get();
    std::printf("train=%zu val=%zu | params value=%d(x2) policy=%d | threads=%d batch=%zu lr=%.4f l2=%.1e pbal=%.2f\n",
                recs.size(), val.size(), L.NV, L.NP, threads, batch, lr0, l2, o.pbal);

    Model mdl;
    mdl.scale = float(a.num("scale", 200));
    if (a.has("init") && !mdl.load(a.str("init"))) { std::fprintf(stderr, "cannot load init model\n"); return 1; }
    Adam av, ap; av.init(mdl.wv.size()); ap.init(mdl.wp.size());
    std::vector<Grad> grads(threads);
    std::vector<uint32_t> order(recs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = uint32_t(i);
    std::mt19937_64 rng(uint64_t(a.num("seed", 1)));

    { metrics::Report rep; metrics::evalSet(mdl, val, rep); rep.print("[init]", false); }
    const size_t stepsPerEpoch = (order.size() + batch - 1) / batch, totalSteps = stepsPerEpoch * epochs;
    size_t step = 0;
    for (int ep = 0; ep < epochs; ++ep) {
        std::shuffle(order.begin(), order.end(), rng);
        const auto t0 = std::chrono::steady_clock::now();
        double vl = 0, pl = 0; size_t vn = 0, pn = 0;
        for (size_t s = 0; s < stepsPerEpoch; ++s, ++step) {
            const size_t lo = s * batch, n = std::min(order.size(), lo + batch) - lo;
            std::vector<std::thread> th;
            for (int t = 0; t < threads; ++t) {
                const size_t a0 = lo + n * t / threads, a1 = lo + n * (t + 1) / threads;
                th.emplace_back([&, t, a0, a1] {
                    grads[t].reset(mdl.wv.size(), mdl.wp.size());
                    accumulate(mdl, recs.data(), order.data() + a0, a1 - a0, o, grads[t]);
                });
            }
            for (auto& x : th) x.join();
            for (int t = 1; t < threads; ++t) grads[0].add(grads[t]);
            const float lr = lr0 * 0.5f * (1 + std::cos(3.14159265f * float(step) / float(totalSteps)));
            if (o.value && grads[0].vn) av.step(mdl.wv, grads[0].gv, lr, l2, 1.f / grads[0].vn);
            if (o.policy && grads[0].pn) ap.step(mdl.wp, grads[0].gp, lr, l2, 1.f / grads[0].pn);
            vl += grads[0].vloss; vn += grads[0].vn; pl += grads[0].ploss; pn += grads[0].pn;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("epoch %d: train value-bce=%.4f policy-ce=%.4f  (%.1fs)\n", ep + 1, vn ? vl / vn : 0, pn ? pl / pn : 0, secs);
        metrics::Report rep; metrics::evalSet(mdl, val, rep); rep.print("  [val]", ep + 1 == epochs);
        mdl.save(out);
    }
    std::printf("saved %s\n", out.c_str());
    return 0;
}
