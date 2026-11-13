// record.hpp — 학습/평가 데이터 레코드 (고정 136 바이트, 리틀엔디언)
//
// 국면은 "돌 배치 + 둘 차례 + 직전 2수"로 저장한다(정책 특징이 직전 수 거리를 쓰므로).
// 라벨은 모두 Rapfi(오라클)가 만든다:
//   nn*   : mix9svq NNUE 정적 가치 (둘 차례 기준 승/패/무 확률)
//   pol*  : NNUE 정책 상위 16개 (확률 × 65535)
//   srch* : Rapfi PVS+NNUE 고정 노드 탐색 결과 (선택)
//   rcls  : Rapfi 내장 고전 평가의 승률 (비교 베이스라인)
#pragma once
#include "board.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gmk {

enum RecFlags : uint8_t {
    REC_SELF_FIVE  = 1,   // 둘 차례가 즉승 가능 (탐색이 정확히 처리 → 가치 학습에서 제외)
    REC_OPPO_FIVE  = 2,   // 상대 4 를 막아야 함 (강제수)
    REC_HAS_SEARCH = 4,
    REC_LEAF       = 8,   // 우리 탐색 트리의 리프에서 수집된 국면 (DAgger)
};

struct Record {
    float    nnW, nnL, nnD;
    float    srchWR;          // 둘 차례 승률 (Rapfi 탐색)
    float    rcls;            // Rapfi 고전평가 승률
    uint16_t polProb[16];
    uint8_t  polIdx[16];
    uint8_t  board[57];       // 225칸 × 2bit (0 빈, 1 흑, 2 백)
    uint8_t  side;            // 1 흑, 2 백
    uint8_t  ply;
    uint8_t  last1, last2;    // 255 = 없음
    uint8_t  flags;
    uint8_t  srchBest;        // 255 = 없음
    int8_t   srchDepth;
    uint8_t  polN;
    uint8_t  pad[3];

    float nnWinrate() const { return nnW + 0.5f * nnD; }

    void setBoard(const Board& b) {
        std::memset(board, 0, sizeof board);
        for (int m = 0; m < NMOVES; ++m) {
            int v = b.colorAt(m);
            board[m >> 2] |= uint8_t(v << (2 * (m & 3)));
        }
        side = uint8_t(b.side); ply = uint8_t(b.ply);
        int l1 = b.lastMove(0), l2 = b.lastMove(1);
        last1 = uint8_t(l1 < 0 ? 255 : l1); last2 = uint8_t(l2 < 0 ? 255 : l2);
    }
    void unpack(uint8_t out[NMOVES]) const {
        for (int m = 0; m < NMOVES; ++m) out[m] = (board[m >> 2] >> (2 * (m & 3))) & 3;
    }
    void toBoard(Board& b) const {
        uint8_t a[NMOVES]; unpack(a);
        b.setPosition(a, side, last1 == 255 ? -1 : last1, last2 == 255 ? -1 : last2);
    }
};
static_assert(sizeof(Record) == 136, "record size");

inline bool readRecords(const std::string& path, std::vector<Record>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    size_t n = size_t(sz) / sizeof(Record), old = out.size();
    out.resize(old + n);
    size_t got = std::fread(out.data() + old, sizeof(Record), n, f);
    out.resize(old + got);
    std::fclose(f);
    return true;
}

class RecordWriter {
public:
    explicit RecordWriter(const std::string& path) : f_(std::fopen(path.c_str(), "wb")) {}
    ~RecordWriter() { if (f_) std::fclose(f_); }
    bool ok() const { return f_ != nullptr; }
    void write(const Record& r) { std::fwrite(&r, sizeof r, 1, f_); }
    void flush() { std::fflush(f_); }
private:
    FILE* f_;
};

} // namespace gmk
