// train.cpp — Rapfi NNUE → 고전 평가 증류 학습기 (희소 선형 모델, 동기식 데이터 병렬 + Adam)
//
//   가치: BCE( sigmoid(v), target ),  target = (1-λ)·NNUE 승률 + λ·Rapfi 탐색 승률(있을 때)
//   정책: softmax CE( p(c) over 후보, NNUE 정책(후보로 제한·재정규화) )
//
// 사용 예:
//   train --data a.bin,b.bin --val v.bin --out model.dce --epochs 8 --threads 4
#include "metrics.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <thread>

using namespace gmk;

namespace {

std::map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a[k] = argv[++i]; else a[k] = "1";
    }
    return a;
}
std::string arg(const std::map<std::string, std::string>& a, const char* k, const std::string& d) {
    auto it = a.find(k); return it == a.end() ? d : it->second;
}
double argf(const std::map<std::string, std::string>& a, const char* k, double d) {
    auto it = a.find(k); return it == a.end() ? d : std::stod(it->second);
}
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(x);
    return v;
}

struct Grad {
    std::vector<float> gv, gp;
    double vloss = 0, ploss = 0; size_t vn = 0, pn = 0;
    void reset(size_t nv, size_t np) {
        gv.assign(nv, 0.f); gp.assign(np, 0.f); vloss = ploss = 0; vn = pn = 0;
    }
};

struct Opts {
    float lambda; bool value, policy;
    float wbal;    // 균형 국면 가중치: w = 1 + wbal * 4p(1-p)
    float pairW;   // 형제 순위 손실 가중치 (같은 부모의 자식 간 NNUE 우열 순서를 직접 맞춤)
};
using Item = std::pair<uint32_t, uint32_t>;   // (시작 인덱스, 개수) — 형제 묶음은 한 아이템

void accumulate(const Model& mdl, const Record* recs, const Item* items, size_t n, const Opts& o, Grad& g) {
    thread_local std::vector<Board> boards(16);
    int moves[NMOVES]; float tgt[NMOVES], sc[NMOVES];
    float v[16], pt[16], gr[16];
    for (size_t it = 0; it < n; ++it) {
        const uint32_t base = items[it].first, k = std::min<uint32_t>(items[it].second, 16);
        for (uint32_t i = 0; i < k; ++i) {
            const Record& r = recs[base + i];
            Board& b = boards[i]; r.toBoard(b);   // 학습은 모델 직접 호출(증분 상태 불필요)
            float p = r.nnWinrate();
            if (o.lambda > 0 && (r.flags & REC_HAS_SEARCH) && !std::isnan(r.srchWR)) p = (1 - o.lambda) * p + o.lambda * r.srchWR;
            pt[i] = p;
            v[i] = o.value ? mdl.valueLogit(b) : 0.f;
            const float q = metrics::sigmoid(v[i]);
            gr[i] = (1 + o.wbal * 4 * p * (1 - p)) * (q - p);
            if (o.value) {
                g.vloss += -(p * std::log(std::max(q, 1e-7f)) + (1 - p) * std::log(std::max(1 - q, 1e-7f)));
                ++g.vn;
            }
        }
        if (o.value && o.pairW > 0 && k >= 2) {
            int np = 0;
            for (uint32_t i = 0; i < k; ++i) for (uint32_t j = i + 1; j < k; ++j) np += std::fabs(pt[i] - pt[j]) >= 0.03f;
            if (np) {
                const float w = o.pairW * float(k) / float(np);   // 묶음 크기에 비례, 쌍 수로 정규화
                for (uint32_t i = 0; i < k; ++i) for (uint32_t j = i + 1; j < k; ++j) {
                    if (std::fabs(pt[i] - pt[j]) < 0.03f) continue;
                    const uint32_t hi = pt[i] > pt[j] ? i : j, lo = hi == i ? j : i;
                    const float d = v[hi] - v[lo], sg = metrics::sigmoid(-d);
                    gr[hi] -= w * sg; gr[lo] += w * sg;
                    g.ploss += 0;   // (순위 손실은 보고에서 제외)
                }
            }
        }
        for (uint32_t i = 0; i < k; ++i) {
            const Record& r = recs[base + i];
            Board& b = boards[i];
            if (o.value) {
                const float t = feat::taper(b), g0 = gr[i] * (1 - t), g1 = gr[i] * t;
                mdl.forValue(b, [&](int f) { g.gv[2 * f] += g0; g.gv[2 * f + 1] += g1; });
            }
            if (o.policy && !(r.flags & REC_OPPO_FIVE)) {
                const int nc = metrics::policyTarget(b, r, moves, tgt);
                if (nc >= 2) {
                    const int tc = feat::threatClass(b);
                    float mx = -1e30f;
                    for (int c = 0; c < nc; ++c) { sc[c] = mdl.policyScore(b, toCell(moves[c]), tc); mx = std::max(mx, sc[c]); }
                    double z = 0; for (int c = 0; c < nc; ++c) z += std::exp(sc[c] - mx);
                    for (int c = 0; c < nc; ++c) {
                        const float pr = float(std::exp(sc[c] - mx) / z), gp = pr - tgt[c];
                        if (tgt[c] > 0) g.ploss += -tgt[c] * std::log(std::max(pr, 1e-9f));
                        if (gp != 0) mdl.forPolicy(b, toCell(moves[c]), tc, [&](int f) { g.gp[f] += gp; });
                    }
                    ++g.pn;
                }
            }
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
    auto a = parseArgs(argc, argv);
    const auto dataFiles = split(arg(a, "data", ""));
    if (dataFiles.empty()) { std::fprintf(stderr, "usage: train --data f1,f2 [--val v] --out model.dce\n"); return 2; }
    const std::string out = arg(a, "out", "model.dce");
    const int    epochs   = int(argf(a, "epochs", 6));
    const int    threads  = int(argf(a, "threads", std::max(1u, std::thread::hardware_concurrency())));
    const size_t batch    = size_t(argf(a, "batch", 1024));
    const float  lr0      = float(argf(a, "lr", 0.02));
    const float  l2       = float(argf(a, "l2", 1e-6));
    const float  valFrac  = float(argf(a, "val-frac", 0.05));
    Opts o{float(argf(a, "lambda", 0.0)), argf(a, "value", 1) != 0, argf(a, "policy", 1) != 0,
           float(argf(a, "wbal", 0.0)), float(argf(a, "pair", 0.0))};

    std::vector<Record> recs, val;
    for (auto& f : dataFiles) if (!readRecords(f, recs)) { std::fprintf(stderr, "cannot read %s\n", f.c_str()); return 1; }
    recs.erase(std::remove_if(recs.begin(), recs.end(), [](const Record& r) { return r.flags & REC_SELF_FIVE; }), recs.end());
    std::mt19937_64 rng(uint64_t(argf(a, "seed", 1)));
    if (a.count("val")) { for (auto& f : split(a["val"])) readRecords(f, val); }
    else {
        // 국면이 같은 대국에서 연속으로 나오므로 무작위 분할은 누설이 있다 → 파일 끝 부분을 검증셋으로 사용
        size_t nv = size_t(recs.size() * valFrac);
        val.assign(recs.end() - nv, recs.end()); recs.resize(recs.size() - nv);
    }
    const auto& L = feat::Layout::get();
    std::printf("train=%zu val=%zu | params value=%d(x2 taper) policy=%d | threads=%d batch=%zu\n",
                recs.size(), val.size(), L.NV, L.NP, threads, batch);

    Model mdl;
    mdl.scale = float(argf(a, "scale", 200));
    if (a.count("init") && !mdl.load(a["init"])) { std::fprintf(stderr, "cannot load init model\n"); return 1; }
    Adam av, ap; av.init(mdl.wv.size()); ap.init(mdl.wp.size());
    std::vector<Grad> grads(threads);
    // 아이템 구성: 연속된 같은 그룹의 형제 국면은 한 묶음으로 (셔플해도 함께 다님)
    std::vector<Item> order;
    for (size_t i = 0; i < recs.size();) {
        size_t j = i + 1;
        const uint32_t gid = (recs[i].flags & REC_SIBLING) ? recs[i].group() : 0;
        if (gid) while (j < recs.size() && j - i < 16 && (recs[j].flags & REC_SIBLING) && recs[j].group() == gid) ++j;
        order.push_back({uint32_t(i), uint32_t(j - i)});
        i = j;
    }
    std::printf("items=%zu (sibling groups merged) wbal=%.2f pair=%.2f lr=%.4f l2=%.1e\n", order.size(), o.wbal, o.pairW, lr0, l2);

    { metrics::Report rep; metrics::evalSet(mdl, val, rep); rep.print("[init]", false); }
    const size_t stepsPerEpoch = (order.size() + batch - 1) / batch;
    const size_t totalSteps = stepsPerEpoch * epochs;
    size_t step = 0;
    for (int ep = 0; ep < epochs; ++ep) {
        std::shuffle(order.begin(), order.end(), rng);
        auto t0 = std::chrono::steady_clock::now();
        double vl = 0, pl = 0; size_t vn = 0, pn = 0;
        for (size_t s = 0; s < stepsPerEpoch; ++s, ++step) {
            const size_t lo = s * batch, hi = std::min(order.size(), lo + batch), n = hi - lo;
            std::vector<std::thread> th;
            for (int t = 0; t < threads; ++t) {
                const size_t a0 = lo + n * t / threads, a1 = lo + n * (t + 1) / threads;
                th.emplace_back([&, t, a0, a1] {
                    grads[t].reset(mdl.wv.size(), mdl.wp.size());
                    accumulate(mdl, recs.data(), order.data() + a0, a1 - a0, o, grads[t]);
                });
            }
            for (auto& x : th) x.join();
            for (int t = 1; t < threads; ++t) {
                for (size_t i = 0; i < grads[0].gv.size(); ++i) grads[0].gv[i] += grads[t].gv[i];
                for (size_t i = 0; i < grads[0].gp.size(); ++i) grads[0].gp[i] += grads[t].gp[i];
                grads[0].vloss += grads[t].vloss; grads[0].ploss += grads[t].ploss;
                grads[0].vn += grads[t].vn; grads[0].pn += grads[t].pn;
            }
            // 코사인 감쇠
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
