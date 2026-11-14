// params.hpp — 탐색 설정(ab::Config)을 이름으로 읽고 쓰기. 모든 도구가 같은 이름을 쓴다.
//   키 = Config 필드 이름 (예: rfpDepth, cutGap, qMaxPly).  bool 은 0/1.
#pragma once
#include "alphabeta.hpp"
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace ab {

struct ParamDef {
    const char* name;
    int  Config::*ip = nullptr;
    bool Config::*bp = nullptr;
};

inline const std::vector<ParamDef>& paramDefs() {
    static const std::vector<ParamDef> d = {
        {"useTT", nullptr, &Config::useTT},           {"useAspiration", nullptr, &Config::useAspiration},
        {"aspDelta", &Config::aspDelta},              {"softTimePct", &Config::softTimePct},
        {"orderPolicyW", &Config::orderPolicyW},      {"useCounter", nullptr, &Config::useCounter},
        {"rpsPerLogit", &Config::rpsPerLogit},        {"rpsMaxExtra", &Config::rpsMaxExtra},
        {"lmrDiv100", &Config::lmrDiv100},            {"forcedCost", &Config::forcedCost},
        {"tacCost", &Config::tacCost},
        {"useIIR", nullptr, &Config::useIIR},         {"useImproving", nullptr, &Config::useImproving},
        {"usePolicyCut", nullptr, &Config::usePolicyCut},
        {"cutGap", &Config::cutGap},                  {"cutMinKeep", &Config::cutMinKeep},
        {"keepBase", &Config::keepBase},              {"keepPerPly", &Config::keepPerPly},
        {"gateDepth", &Config::gateDepth},            {"gateMargin", &Config::gateMargin},
        {"rfpDepth", &Config::rfpDepth},              {"rfpMargin", &Config::rfpMargin},
        {"useQuiesce", nullptr, &Config::useQuiesce}, {"qMaxPly", &Config::qMaxPly},
        {"ttWays", &Config::ttWays},                  {"contW", &Config::contW},
        {"histMalus", nullptr, &Config::histMalus},   {"razorDepth", &Config::razorDepth},
        {"razorMargin", &Config::razorMargin},        {"mcpBase", &Config::mcpBase},
        {"tacW", &Config::tacW},                      {"histDiv", &Config::histDiv},
        {"qThreatPen", &Config::qThreatPen},          {"cutNodeRed", &Config::cutNodeRed},
        {"seDepth", &Config::seDepth},                {"seMargin", &Config::seMargin},
        {"seCost", &Config::seCost},
        {"nmpDepth", &Config::nmpDepth},              {"nmpR", &Config::nmpR},
        {"nmpDiv", &Config::nmpDiv},                  {"qTT", nullptr, &Config::qTT},                  {"seMultiCut", nullptr, &Config::seMultiCut},
    };
    return d;
}

inline bool setParam(Config& c, const std::string& name, int value) {
    for (const auto& p : paramDefs())
        if (name == p.name) { if (p.ip) c.*p.ip = value; else c.*p.bp = value != 0; return true; }
    return false;
}

// args 에서 prefix+키 로 시작하는 항목을 모두 적용 (예: prefix "a-" → "a-rfpDepth"). 모르는 키는 무시.
inline void applyParams(Config& c, const std::map<std::string, std::string>& args, const std::string& prefix = "") {
    for (const auto& p : paramDefs()) {
        auto it = args.find(prefix + p.name);
        if (it != args.end()) setParam(c, p.name, std::stoi(it->second));
    }
}

// "키=값" 문자열 (pbrain --set 용)
inline bool setParamKV(Config& c, const std::string& kv) {
    const auto eq = kv.find('=');
    return eq != std::string::npos && setParam(c, kv.substr(0, eq), std::stoi(kv.substr(eq + 1)));
}

inline void printParams(const Config& c, FILE* f = stdout) {
    for (const auto& p : paramDefs()) std::fprintf(f, "%s=%d ", p.name, p.ip ? c.*p.ip : int(c.*p.bp));
    std::fprintf(f, "\n");
}

} // namespace ab
