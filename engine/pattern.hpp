// pattern.hpp — 15x15 프리스타일 라인 패턴 분류기 (헤더 온리, C++17)
//
// Rapfi(GPL-3) game/pattern.cpp 의 FREESTYLE 분류 알고리즘을 그대로 옮긴 것이다.
// 목적: 우리 고전 평가의 "패턴 어휘"를 Rapfi 와 1:1 로 맞춰서, NNUE 판단을 증류할 때
//       같은 개념 단위(B4/F3/...)로 비교·분석할 수 있게 한다. (oracle crosscheck 로 검증)
//
// 라인 키(16bit): 중심 칸을 뺀 8칸 × 2bit.  슬롯 0..3 = 오프셋 -4..-1, 슬롯 4..7 = +1..+4
// 상대 인코딩: 0 빈칸, 1 자기(self), 2 상대(oppo), 3 벽
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>

namespace gmk {

enum Pattern : uint8_t { DEAD, OL, B1, F1, B2, F2, F2A, F2B, B3, F3, F3S, B4, F4, F5, PATTERN_NB };

enum Pattern4 : uint8_t {
    NONE, FORBID, L_FLEX2, K_BLOCK3, J_FLEX2_2X, I_BLOCK3_PLUS, H_FLEX3, G_FLEX3_PLUS,
    F_FLEX3_2X, E_BLOCK4, D_BLOCK4_PLUS, C_BLOCK4_FLEX3, B_FLEX4, A_FIVE, PATTERN4_NB
};

constexpr int KEY_NB   = 1 << 16;
constexpr int PCODE_NB = 2380;          // C(14+4-1, 4): 4방향 패턴 중복조합 수

inline const char* patternName(int p) {
    static const char* n[] = {"DEAD","OL","B1","F1","B2","F2","F2A","F2B","B3","F3","F3S","B4","F4","F5"};
    return p >= 0 && p < PATTERN_NB ? n[p] : "?";
}
inline const char* pattern4Name(int p) {
    static const char* n[] = {"NONE","FORBID","L_FLEX2","K_BLOCK3","J_FLEX2_2X","I_BLOCK3_PLUS","H_FLEX3",
                              "G_FLEX3_PLUS","F_FLEX3_2X","E_BLOCK4","D_BLOCK4_PLUS","C_BLOCK4_FLEX3","B_FLEX4","A_FIVE"};
    return p >= 0 && p < PATTERN4_NB ? n[p] : "?";
}

class PatternTables {
public:
    uint8_t  patSelf[KEY_NB];           // 상대 키 → 자기 돌을 중심에 놓았을 때의 패턴
    uint16_t swapKey[KEY_NB];           // self/oppo 교환 (1<->2)
    uint16_t revKey[KEY_NB];            // 라인 방향 반전
    uint16_t pcode[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB];
    uint8_t  p4OfPcode[PCODE_NB];
    uint8_t  pcodePats[PCODE_NB][4];    // pcode → 정렬된 4패턴

    static const PatternTables& get() { static const PatternTables t; return t; }

private:
    enum CF : uint8_t { SELF, OPPO, EMPT };
    using Line = std::array<uint8_t, 9>;
    static constexpr int Mid = 4;

    std::array<uint8_t, 19683> memo_;   // 3^9

    PatternTables() {
        memo_.fill(0xff);
        for (int k = 0; k < KEY_NB; ++k) {
            Line l;
            for (int i = 0; i < 9; ++i) {
                if (i == Mid) { l[i] = SELF; continue; }
                const int s = i < Mid ? i : i - 1;
                const int v = (k >> (2 * s)) & 3;
                l[i] = v == 0 ? EMPT : v == 1 ? SELF : OPPO;
            }
            patSelf[k] = classify(l);
            int sw = 0, rv = 0;
            for (int s = 0; s < 8; ++s) {
                int v = (k >> (2 * s)) & 3;
                sw |= (v == 1 ? 2 : v == 2 ? 1 : v) << (2 * s);
                rv |= v << (2 * (7 - s));
            }
            swapKey[k] = uint16_t(sw);
            revKey[k]  = uint16_t(rv);
        }
        int idx = 0;
        int tmp[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB];
        for (auto& a : tmp) for (auto& b : a) for (auto& c : b) for (auto& d : c) d = -1;
        for (int a = 0; a < PATTERN_NB; ++a) for (int b = a; b < PATTERN_NB; ++b)
        for (int c = b; c < PATTERN_NB; ++c) for (int d = c; d < PATTERN_NB; ++d) {
            tmp[a][b][c][d] = idx;
            pcodePats[idx][0] = uint8_t(a); pcodePats[idx][1] = uint8_t(b);
            pcodePats[idx][2] = uint8_t(c); pcodePats[idx][3] = uint8_t(d);
            p4OfPcode[idx] = combine(Pattern(a), Pattern(b), Pattern(c), Pattern(d));
            ++idx;
        }
        for (int a = 0; a < PATTERN_NB; ++a) for (int b = 0; b < PATTERN_NB; ++b)
        for (int c = 0; c < PATTERN_NB; ++c) for (int d = 0; d < PATTERN_NB; ++d) {
            int v[4] = {a, b, c, d};
            std::sort(v, v + 4);
            pcode[a][b][c][d] = uint16_t(tmp[v[0]][v[1]][v[2]][v[3]]);
        }
    }

    static int code(const Line& l) { int c = 0; for (int i = 0; i < 9; ++i) c = c * 3 + l[i]; return c; }

    uint8_t classify(const Line& line) {
        uint8_t& m = memo_[code(line)];
        if (m != 0xff) return m;
        int realLen = 1, fullLen = 1, inc = 1, start = Mid, end = Mid;
        for (int i = Mid - 1; i >= 0; --i) {
            if (line[i] == SELF) realLen += inc; else if (line[i] == OPPO) break; else inc = 0;
            ++fullLen; start = i;
        }
        inc = 1;
        for (int i = Mid + 1; i < 9; ++i) {
            if (line[i] == SELF) realLen += inc; else if (line[i] == OPPO) break; else inc = 0;
            ++fullLen; end = i;
        }
        Pattern p = DEAD;
        if (realLen >= 5) p = F5;                     // 프리스타일: 장목도 승리
        else if (fullLen < 5) p = DEAD;
        else {
            int cnt[PATTERN_NB] = {0};
            for (int i = start; i <= end; ++i) {
                if (line[i] != EMPT) continue;
                Line sl;
                for (int j = 0; j < 9; ++j) { int id = j + i - Mid; sl[j] = (id >= 0 && id < 9) ? line[id] : OPPO; }
                sl[Mid] = SELF;
                cnt[classify(sl)]++;
            }
            if      (cnt[F5] >= 2)               p = F4;
            else if (cnt[F5])                    p = B4;
            else if (cnt[F4] >= 2)               p = F3S;
            else if (cnt[F4])                    p = F3;
            else if (cnt[B4])                    p = B3;
            else if (cnt[F3S] + cnt[F3] >= 4)    p = F2B;
            else if (cnt[F3S] + cnt[F3] >= 3)    p = F2A;
            else if (cnt[F3S] + cnt[F3])         p = F2;
            else if (cnt[B3])                    p = B2;
            else if (cnt[F2] + cnt[F2A] + cnt[F2B]) p = F1;
            else if (cnt[B2])                    p = B1;
        }
        return m = uint8_t(p);
    }

    static uint8_t combine(Pattern p1, Pattern p2, Pattern p3, Pattern p4) {
        int n[PATTERN_NB] = {0};
        n[p1]++; n[p2]++; n[p3]++; n[p4]++;
        const int f2 = n[F2] + n[F2A] + n[F2B], f3 = n[F3] + n[F3S];
        if (n[F5] >= 1) return A_FIVE;
        if (n[B4] >= 2 || n[F4] >= 1) return B_FLEX4;
        if (n[B4] >= 1) return f3 ? C_BLOCK4_FLEX3 : (n[B3] || f2) ? D_BLOCK4_PLUS : E_BLOCK4;
        if (f3 >= 1)    return f3 >= 2 ? F_FLEX3_2X : (n[B3] || f2) ? G_FLEX3_PLUS : H_FLEX3;
        if (n[B3] >= 2 || (n[B3] >= 1 && f2 >= 1)) return I_BLOCK3_PLUS;
        if (f2 >= 2) return J_FLEX2_2X;
        if (n[B3] >= 1) return K_BLOCK3;
        if (f2 >= 1) return L_FLEX2;
        return NONE;
    }
};

} // namespace gmk
