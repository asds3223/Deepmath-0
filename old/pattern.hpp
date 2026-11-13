// pattern.hpp — 라인 위협 단계 분류 (독자 정의, 헤더 온리, C++17)
//
// 한 칸 c 에 "자기 돌을 놓는다"고 가정하고, c 를 지나는 한 방향 9칸 창(±4)만 보고
// "c 를 포함하는 5목"까지 얼마나 가까운지를 완성점(completion point) 집합으로 정의한다.
//
//   FIVE   c 를 두면 c 를 포함한 5목 이상 (프리스타일: 장목도 승리)
//   OPEN4  c 를 둔 뒤 c 를 포함한 5목 완성점이 2곳 이상   (막을 수 없는 4)
//   FOUR   완성점 1곳
//   OPEN3  한 칸을 더 두면 OPEN4 가 되는 칸이 있다
//   THREE  한 칸을 더 두면 FOUR 가 되는 칸만 있다
//   OPEN2  한 칸을 더 두면 OPEN3 가 되는 칸이 있다
//   TWO    한 칸을 더 두면 THREE 가 된다
//   ONE    c 를 포함하는 막히지 않은 5칸 창이 있다
//   DEAD   c 를 포함해 5목을 만들 공간이 없다
//
// 칸 단위 조합(4방향): 4개 단계의 중복조합 코드(495종)와, 그로부터 정한 칸 등급 CellClass.
// 라인 키(16bit): 중심을 뺀 8칸 × 2bit. 슬롯 0..3 = 오프셋 -4..-1, 4..7 = +1..+4
// 상대 인코딩: 0 빈칸, 1 자기, 2 상대, 3 벽
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>

namespace gmk {

enum Level : uint8_t { DEAD, ONE, TWO, OPEN2, THREE, OPEN3, FOUR, OPEN4, FIVE, LEVEL_NB };

enum CellClass : uint8_t {
    C_NONE, C_TWO, C_OPEN2, C_DOUBLE2, C_THREE, C_THREE_PLUS, C_OPEN3, C_OPEN3_PLUS,
    C_DOUBLE3, C_FOUR, C_FOUR_PLUS, C_FOUR3, C_WINFOUR, C_FIVE, CLASS_NB
};

constexpr int KEY_NB  = 1 << 16;
constexpr int CODE_NB = 495;        // C(9+4-1, 4)

inline const char* levelName(int l) {
    static const char* n[] = {"DEAD", "ONE", "TWO", "OPEN2", "THREE", "OPEN3", "FOUR", "OPEN4", "FIVE"};
    return l >= 0 && l < LEVEL_NB ? n[l] : "?";
}
inline const char* className(int c) {
    static const char* n[] = {"NONE", "TWO", "OPEN2", "DOUBLE2", "THREE", "THREE+", "OPEN3", "OPEN3+",
                              "DOUBLE3", "FOUR", "FOUR+", "FOUR3", "WINFOUR", "FIVE"};
    return c >= 0 && c < CLASS_NB ? n[c] : "?";
}

// 위협 단계 → 전술 등급 (정책/전역 특징의 거친 버킷)
inline int classTier(int c) {
    return c >= C_DOUBLE3 ? (c >= C_FOUR3 || c == C_DOUBLE3 ? 4 : 3) : c >= C_OPEN3 ? 2 : c >= C_THREE ? 1 : 0;
}

class LineTables {
public:
    uint8_t  level[KEY_NB];          // 상대 키 → 자기 관점 단계
    uint16_t swapKey[KEY_NB];        // 자기/상대 교환
    uint16_t revKey[KEY_NB];         // 방향 반전
    uint16_t code[LEVEL_NB][LEVEL_NB][LEVEL_NB][LEVEL_NB];
    uint8_t  classOf[CODE_NB];
    uint8_t  codeLevels[CODE_NB][4];

    static const LineTables& get() { static const LineTables t; return t; }

private:
    // 9칸 창: bit i (0..8) — 자기 돌 마스크 / 막힘(상대·벽) 마스크
    struct Win { uint16_t self, block; };
    static constexpr int C = 4;   // 중심 인덱스

    // c 를 포함하는 5칸 창(시작 0..4) 중 막힘 없는 것에 대해, 자기 돌로 채워진 경우 5목
    static bool fiveThroughCenter(uint16_t self) {
        // 연속 길이로 판정 (장목 포함)
        int run = 1;
        for (int i = C - 1; i >= 0 && (self >> i & 1); --i) ++run;
        for (int i = C + 1; i < 9 && (self >> i & 1); ++i) ++run;
        return run >= 5;
    }
    static int completions(Win w) {
        int n = 0;
        for (int i = 0; i < 9; ++i) {
            if (i == C || (w.self >> i & 1) || (w.block >> i & 1)) continue;
            if (fiveThroughCenter(w.self | uint16_t(1 << i))) ++n;
        }
        return n;
    }
    static bool hasSpace(Win w) {
        for (int s = 0; s <= 4; ++s) {
            uint16_t m = uint16_t(0x1f << s);
            if (!(w.block & m)) return true;
        }
        return false;
    }
    // 단계 계산 (중심에 자기 돌이 이미 놓인 창). depth: 남은 선행 수
    static int levelOf(Win w, int depth) {
        if (fiveThroughCenter(w.self)) return FIVE;
        if (!hasSpace(w)) return DEAD;
        const int k = completions(w);
        if (k >= 2) return OPEN4;
        if (k == 1) return FOUR;
        if (depth == 0) return ONE;
        int best = ONE;
        for (int i = 0; i < 9; ++i) {
            if (i == C || (w.self >> i & 1) || (w.block >> i & 1)) continue;
            const int l = levelOf(Win{uint16_t(w.self | (1 << i)), w.block}, depth - 1);
            // 한 수 더 두었을 때의 단계 → 현재 단계로 한 칸 내림
            const int down = l == OPEN4 ? OPEN3 : l == FOUR ? THREE : l == OPEN3 ? OPEN2 : l == THREE ? TWO : ONE;
            best = std::max(best, down);
        }
        return best;
    }

    LineTables() {
        for (int k = 0; k < KEY_NB; ++k) {
            Win w{uint16_t(1 << C), 0};
            int sw = 0, rv = 0;
            for (int s = 0; s < 8; ++s) {
                const int v = (k >> (2 * s)) & 3;
                const int i = s < 4 ? s : s + 1;
                if (v == 1) w.self |= uint16_t(1 << i);
                else if (v >= 2) w.block |= uint16_t(1 << i);
                sw |= (v == 1 ? 2 : v == 2 ? 1 : v) << (2 * s);
                rv |= v << (2 * (7 - s));
            }
            level[k]   = uint8_t(levelOf(w, 2));
            swapKey[k] = uint16_t(sw);
            revKey[k]  = uint16_t(rv);
        }
        int idx = 0, tmp[LEVEL_NB][LEVEL_NB][LEVEL_NB][LEVEL_NB];
        for (int a = 0; a < LEVEL_NB; ++a) for (int b = a; b < LEVEL_NB; ++b)
        for (int c = b; c < LEVEL_NB; ++c) for (int d = c; d < LEVEL_NB; ++d) {
            tmp[a][b][c][d] = idx;
            codeLevels[idx][0] = uint8_t(a); codeLevels[idx][1] = uint8_t(b);
            codeLevels[idx][2] = uint8_t(c); codeLevels[idx][3] = uint8_t(d);
            classOf[idx] = classify(a, b, c, d);
            ++idx;
        }
        for (int a = 0; a < LEVEL_NB; ++a) for (int b = 0; b < LEVEL_NB; ++b)
        for (int c = 0; c < LEVEL_NB; ++c) for (int d = 0; d < LEVEL_NB; ++d) {
            int v[4] = {a, b, c, d}; std::sort(v, v + 4);
            code[a][b][c][d] = uint16_t(tmp[v[0]][v[1]][v[2]][v[3]]);
        }
    }

    static uint8_t classify(int a, int b, int c, int d) {
        int n[LEVEL_NB] = {0};
        n[a]++; n[b]++; n[c]++; n[d]++;
        const int fours = n[FOUR], open3 = n[OPEN3], minor = n[THREE] + n[OPEN2];
        if (n[FIVE])                    return C_FIVE;
        if (n[OPEN4] || fours >= 2)     return C_WINFOUR;
        if (fours && open3)             return C_FOUR3;
        if (fours)                      return minor ? C_FOUR_PLUS : C_FOUR;
        if (open3 >= 2)                 return C_DOUBLE3;
        if (open3)                      return minor ? C_OPEN3_PLUS : C_OPEN3;
        if (n[THREE])                   return (n[THREE] >= 2 || n[OPEN2]) ? C_THREE_PLUS : C_THREE;
        if (n[OPEN2] >= 2)              return C_DOUBLE2;
        if (n[OPEN2])                   return C_OPEN2;
        if (n[TWO])                     return C_TWO;
        return C_NONE;
    }
};

} // namespace gmk
