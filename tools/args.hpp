// args.hpp — 도구 공용 인자 파서: --키 값 / --스위치 (값 없으면 "1")
#pragma once
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Args {
    std::map<std::string, std::string> kv;
    Args(int argc, char** argv, int from = 1) {
        for (int i = from; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0) continue;
            k = k.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv[k] = argv[++i]; else kv[k] = "1";
        }
    }
    bool has(const std::string& k) const { return kv.count(k) != 0; }
    std::string str(const std::string& k, const std::string& d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    double num(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::stod(it->second); }
    std::vector<std::string> list(const std::string& k) const {
        std::vector<std::string> v; std::stringstream ss(str(k)); std::string x;
        while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(x);
        return v;
    }
};
