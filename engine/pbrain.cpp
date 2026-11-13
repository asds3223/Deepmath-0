// pbrain.cpp — Gomocup(piskvork) 프로토콜 엔진. 실전 규칙: 싱글코어, 폰더링 없음.
//   pbrain-dce [--model model.dce] [--tt-mb 128]
#include "eval.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace gmk;

namespace {
struct Engine {
    Model model; bool hasModel = false;
    Board board;
    ab::Searcher<Board> searcher;
    int64_t timeoutTurn = 5000, timeoutMatch = 0, timeLeft = -1;
    explicit Engine(size_t ttMb) : searcher(ttMb) {}

    int64_t budgetMs() const {
        int64_t t = timeoutTurn > 0 ? timeoutTurn : 5000;
        if (timeLeft >= 0) t = std::min<int64_t>(t, timeLeft / 15);
        return std::max<int64_t>(30, t - 60);   // 통신/여유 마진
    }
    void reset() { board.clear(); board.setModel(hasModel ? &model : nullptr); searcher.tt.clear(); searcher.clearHeuristics(); }
    std::string think() {
        ab::Result r = searcher.think(board, ab::Limits{ab::MAX_PLY - 1, budgetMs(), 0});
        int m = r.bestMove;
        if (m < 0) { for (int q = 0; q < NMOVES; ++q) if (board.isEmptyMove(q)) { m = q; break; } }
        std::printf("MESSAGE depth %d score %d nodes %llu time %lldms\n", r.depth, r.score,
                    (unsigned long long)r.nodes, (long long)r.timeMs);
        board.make(m);
        return std::to_string(m % BS) + "," + std::to_string(m / BS);
    }
};

bool parseXY(const std::string& s, int& x, int& y) { return std::sscanf(s.c_str(), "%d,%d", &x, &y) == 2; }
} // namespace

int main(int argc, char** argv) {
    std::string modelPath = "model.dce"; size_t ttMb = 128;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--model") modelPath = argv[++i];
        else if (std::string(argv[i]) == "--tt-mb") ttMb = std::stoul(argv[++i]);
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Engine e(ttMb);
    e.hasModel = e.model.load(modelPath);
    e.reset();
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream is(line); std::string cmd; is >> cmd;
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);
        if (cmd == "START") {
            int n = 0; is >> n;
            if (n != BS) { std::printf("ERROR only 15x15 supported\n"); continue; }
            e.reset();
            if (!e.hasModel) std::printf("MESSAGE warning: model not found (%s), heuristic eval\n", modelPath.c_str());
            std::printf("OK\n");
        } else if (cmd == "RESTART") { e.reset(); std::printf("OK\n"); }
        else if (cmd == "BEGIN") { std::printf("%s\n", e.think().c_str()); }
        else if (cmd == "TURN") {
            std::string xy; is >> xy; int x, y;
            if (!parseXY(xy, x, y) || x < 0 || y < 0 || x >= BS || y >= BS || !e.board.isEmptyMove(x + y * BS)) {
                std::printf("ERROR bad move\n"); continue;
            }
            e.board.make(x + y * BS);
            std::printf("%s\n", e.think().c_str());
        } else if (cmd == "BOARD") {
            std::vector<std::pair<int, int>> st;   // (move, who 1=me 2=opp)
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::string u = line; std::transform(u.begin(), u.end(), u.begin(), ::toupper);
                if (u == "DONE") break;
                int x, y, w; if (std::sscanf(line.c_str(), "%d,%d,%d", &x, &y, &w) == 3) st.push_back({x + y * BS, w});
            }
            e.reset();
            // 순서대로 번갈아 두어졌으면 그대로 재생, 아니면 배치로 복원
            bool alt = true; int first = st.empty() ? 1 : st[0].second;
            for (size_t i = 0; i < st.size(); ++i) if (st[i].second != (i % 2 == 0 ? first : 3 - first)) alt = false;
            int mine = 0, theirs = 0; for (auto& s : st) (s.second == 1 ? mine : theirs)++;
            const int myColor = mine == theirs ? BLACK : WHITE;
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
            std::string k; int64_t v = 0; is >> k >> v;
            std::transform(k.begin(), k.end(), k.begin(), ::tolower);
            if (k == "timeout_turn") e.timeoutTurn = v;
            else if (k == "timeout_match") e.timeoutMatch = v;
            else if (k == "time_left") e.timeLeft = v;
        } else if (cmd == "ABOUT") {
            std::printf("name=\"DCE\", version=\"0.1\", author=\"distilled-classical-eval\", country=\"KR\"\n");
        } else if (cmd == "END") break;
    }
    return 0;
}
