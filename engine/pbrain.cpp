// pbrain.cpp — Gomocup(piskvork) 프로토콜 엔진 "DCE"
//   실전 규칙: 싱글코어, 폰더링 없음 (명령 사이에는 아무 계산도 하지 않는다).
//   15x15 프리스타일 전용. 가중치는 실행 파일에 내장(DCE_EMBED_MODEL) — 파일 하나로 동작.
//
//   pbrain-dce [--model 파일.dce] [--tt-mb N] [--vcf 0|1] [--vcf-budget N]   프로토콜 모드
//   pbrain-dce bench                              속도 측정 (고정 국면 탐색 nps)
//
// 정보 출력 (반복 완료마다 MESSAGE 한 줄, 탐색 중 추가 비용 없음):
//   depth  완료 반복 깊이 / sel 선택 깊이(정지 탐색·VCF 포함 실제 최대 수 깊이)
//   ev 평가(점수) wr 승률(둘 차례 관점) nodes 노드 nps 초당 노드 tm 경과 ms pv 예상 수순
#include "eval.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef DCE_EMBED_MODEL
#include "model_data.h"     // static const unsigned char kModelData[]; kModelSize
#endif

using namespace gmk;

namespace {

const char* VERSION = "0.5";

std::string coord(int m) {     // a1 ~ o15 (열 문자 + 행 번호)
    if (m < 0) return "--";
    return std::string(1, char('a' + m % BS)) + std::to_string(m / BS + 1);
}
std::string humanNum(double v) {
    if (v < 0) return "-";
    char b[32];
    if (v >= 1e9) std::snprintf(b, sizeof b, "%.2fG", v / 1e9);
    else if (v >= 1e6) std::snprintf(b, sizeof b, "%.2fM", v / 1e6);
    else if (v >= 1e3) std::snprintf(b, sizeof b, "%.1fk", v / 1e3);
    else std::snprintf(b, sizeof b, "%.0f", v);
    return b;
}
std::string scoreStr(int s, float scale) {
    char b[48];
    if (s >= ab::MATE_BOUND) std::snprintf(b, sizeof b, "ev +M%d wr 100.0%%", ab::MATE - s);
    else if (s <= -ab::MATE_BOUND) std::snprintf(b, sizeof b, "ev -M%d wr 0.0%%", ab::MATE + s);
    else std::snprintf(b, sizeof b, "ev %d wr %.1f%%", s, 100.f / (1.f + std::exp(-float(s) / scale)));
    return b;
}

struct Engine {
    Model model; bool hasModel = false; std::string modelSrc = "none";
    Board board;
    ab::Searcher<Board> searcher;
    int64_t timeoutTurn = 5000, timeoutMatch = 0, timeLeft = -1;
    size_t ttMb;
    explicit Engine(size_t mb) : searcher(mb), ttMb(mb) {
        searcher.onIteration = [this](const ab::Result& r) { report(r, false); };
    }

    void report(const ab::Result& r, bool final) {
        std::string pv;
        for (size_t i = 0; i < r.pv.size() && i < 12; ++i) { pv += ' '; pv += coord(r.pv[i]); }
        const double nps = r.timeMs > 0 ? double(r.nodes) * 1000.0 / double(r.timeMs) : -1;
        std::printf("MESSAGE %sdepth %d sel %d %s nodes %s nps %s tm %lld pv%s\n", final ? "[best " : "",
                    r.depth, r.selDepth, scoreStr(r.score, model.scale).c_str(), humanNum(double(r.nodes)).c_str(),
                    humanNum(nps).c_str(), (long long)r.timeMs, pv.c_str());
    }
    int64_t budgetMs() const {
        int64_t t = timeoutTurn > 0 ? timeoutTurn : 1000;
        if (timeLeft >= 0) t = std::min<int64_t>(t, timeLeft / 15);
        return std::max<int64_t>(20, t - std::min<int64_t>(60, t / 10));   // 통신 여유
    }
    void reset() { board.clear(); board.setModel(hasModel ? &model : nullptr); searcher.tt.clear(); searcher.clearHeuristics(); }
    std::string think() {
        ab::Result r = searcher.think(board, ab::Limits{ab::MAX_PLY - 1, budgetMs(), 0});
        int m = r.bestMove;
        if (m < 0 || !board.isEmptyMove(m)) { for (int q = 0; q < NMOVES; ++q) if (board.isEmptyMove(q)) { m = q; break; } }
        std::printf("MESSAGE [move %s] depth %d sel %d %s nodes %s nps %s tm %lldms\n", coord(m).c_str(), r.depth, r.selDepth,
                    scoreStr(r.score, model.scale).c_str(), humanNum(double(r.nodes)).c_str(),
                    humanNum(r.timeMs > 0 ? double(r.nodes) * 1000.0 / double(r.timeMs) : 0).c_str(), (long long)r.timeMs);
        board.make(m);
        return std::to_string(m % BS) + "," + std::to_string(m / BS);
    }
};

bool parseXY(const std::string& s, int& x, int& y) { return std::sscanf(s.c_str(), "%d,%d", &x, &y) == 2; }

int bench(Engine& e) {
    // 고정 중반 국면들에서 고정 노드 탐색 → 평균 nps
    static const std::vector<std::vector<int>> pos = {
        {112},
        {112, 128},
        {112, 113, 127},
        {112, 96, 98, 126},
        {112, 113, 97, 127, 98, 96, 128, 82, 111, 126},
        {112, 96, 128, 144, 97, 82, 113, 98, 127, 142, 111, 110},
    };
    const uint64_t nodes = 2000000;
    double tn = 0, tms = 0;
    std::printf("bench: %zu positions x %s nodes, single thread\n", pos.size(), humanNum(double(nodes)).c_str());
    for (size_t i = 0; i < pos.size(); ++i) {
        e.reset();
        for (int m : pos[i]) e.board.make(m);
        e.searcher.onIteration = nullptr;
        ab::Result r = e.searcher.think(e.board, ab::Limits{ab::MAX_PLY - 1, -1, nodes});
        tn += double(r.nodes); tms += double(r.timeMs);
        std::printf("  #%zu best %-4s depth %2d sel %2d %s nodes %s tm %lldms nps %s\n", i + 1, coord(r.bestMove).c_str(),
                    r.depth, r.selDepth, scoreStr(r.score, e.model.scale).c_str(), humanNum(double(r.nodes)).c_str(),
                    (long long)r.timeMs, humanNum(r.timeMs > 0 ? double(r.nodes) * 1000.0 / double(r.timeMs) : 0).c_str());
    }
    std::printf("total nodes %s time %.0fms  =>  nps %s\n", humanNum(tn).c_str(), tms, humanNum(tms > 0 ? tn * 1000 / tms : 0).c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string modelPath; size_t ttMb = 128; bool doBench = false, vcfOn = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (a == "--tt-mb" && i + 1 < argc) ttMb = std::stoul(argv[++i]);
        else if (a == "bench") doBench = true;
        else if (a == "--vcf" && i + 1 < argc) vcfOn = std::stoi(argv[++i]) != 0;
        else if (a == "--vcf-budget" && i + 1 < argc) g_vcfBudget = std::stoi(argv[++i]);
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Engine e(ttMb);
    if (!modelPath.empty()) { e.hasModel = e.model.load(modelPath); e.modelSrc = modelPath; }
#ifdef DCE_EMBED_MODEL
    if (!e.hasModel) { e.hasModel = e.model.loadMem(kModelData, kModelSize); e.modelSrc = "embedded"; }
#endif
    e.searcher.cfg.useVCF = vcfOn;
    e.reset();
    if (doBench) return bench(e);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream is(line); std::string cmd; is >> cmd;
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);
        if (cmd.empty()) continue;
        if (cmd == "START") {
            int n = 0; is >> n;
            if (n != BS) { std::printf("ERROR only 15x15 is supported\n"); continue; }
            e.reset();
            std::printf("OK\n");
            std::printf("MESSAGE DCE %s | model %s | single thread, no pondering | tt %zuMB\n", VERSION,
                        e.hasModel ? e.modelSrc.c_str() : "MISSING (heuristic eval)", e.ttMb);
        } else if (cmd == "RESTART") { e.reset(); std::printf("OK\n"); }
        else if (cmd == "BEGIN") { std::printf("%s\n", e.think().c_str()); }
        else if (cmd == "TURN") {
            std::string xy; is >> xy; int x, y;
            if (!parseXY(xy, x, y) || x < 0 || y < 0 || x >= BS || y >= BS || !e.board.isEmptyMove(x + y * BS)) {
                std::printf("ERROR bad move %s\n", xy.c_str()); continue;
            }
            e.board.make(x + y * BS);
            std::printf("%s\n", e.think().c_str());
        } else if (cmd == "TAKEBACK") {
            std::string xy; is >> xy; int x, y;
            const int m = (parseXY(xy, x, y) && x >= 0 && y >= 0 && x < BS && y < BS) ? x + y * BS : -1;
            if (m < 0 || e.board.isEmptyMove(m)) { std::printf("ERROR cannot take back %s\n", xy.c_str()); continue; }
            if (e.board.lastMove() == m) e.board.undo(m);          // 마지막 수면 그대로 무르기
            else {                                                  // 아니면 그 돌만 빼고 국면 재구성
                uint8_t arr[NMOVES]; const int col = e.board.colorAt(m);
                for (int q = 0; q < NMOVES; ++q) arr[q] = uint8_t(e.board.colorAt(q));
                arr[m] = 0;
                e.board.setPosition(arr, col, -1, -1);
            }
            std::printf("OK\n");
        } else if (cmd == "BOARD") {
            std::vector<std::pair<int, int>> st;   // (move, who 1=me 2=opp 3=continuous)
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::string u = line; std::transform(u.begin(), u.end(), u.begin(), ::toupper);
                if (u == "DONE") break;
                int x, y, w; if (std::sscanf(line.c_str(), "%d,%d,%d", &x, &y, &w) == 3 && x >= 0 && y >= 0 && x < BS && y < BS)
                    st.push_back({x + y * BS, w == 3 ? 2 : w});
            }
            e.reset();
            int mine = 0, theirs = 0; for (auto& s : st) (s.second == 1 ? mine : theirs)++;
            const int myColor = mine == theirs ? BLACK : WHITE;
            bool alt = !st.empty();
            const int first = st.empty() ? 1 : st[0].second;
            for (size_t i = 0; i < st.size(); ++i) if (st[i].second != (i % 2 == 0 ? first : 3 - first)) alt = false;
            if (alt && ((first == 1) == (myColor == BLACK))) for (auto& s : st) e.board.make(s.first);
            else {
                uint8_t arr[NMOVES] = {0}; int l1 = -1, l2 = -1;
                for (auto& s : st) {
                    arr[s.first] = uint8_t(s.second == 1 ? myColor : opp(myColor));
                    if (s.second == 1) l2 = s.first; else l1 = s.first;
                }
                e.board.setPosition(arr, myColor, l1, l2);
            }
            std::printf("%s\n", e.think().c_str());
        } else if (cmd == "INFO") {
            std::string k; std::string v; is >> k >> v;
            std::transform(k.begin(), k.end(), k.begin(), ::tolower);
            long long n = 0; try { n = std::stoll(v); } catch (...) {}
            if (k == "timeout_turn") e.timeoutTurn = n;
            else if (k == "timeout_match") e.timeoutMatch = n;
            else if (k == "time_left") e.timeLeft = n;
            else if (k == "max_memory" && n > 0) {
                const size_t mb = size_t(std::max<long long>(8, n / (1024 * 1024) / 2));
                if (mb < e.ttMb) { e.ttMb = mb; e.searcher.tt.resize(mb); }
            } else if (k == "rule" && n != 0) std::printf("MESSAGE warning: only freestyle (rule 0) is supported\n");
            // pondering 등 나머지 키는 무시 (폰더링 하지 않음)
        } else if (cmd == "ABOUT") {
            std::printf("name=\"DCE\", version=\"%s\", author=\"Deepmath-0\", country=\"KR\"\n", VERSION);
        } else if (cmd == "END") break;
        else std::printf("UNKNOWN %s\n", cmd.c_str());
    }
    return 0;
}
