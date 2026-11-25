// chess.hpp — chess rules: bitboards, legal move generation, copy-make, FEN, Zobrist.
// Squares: 0 = a1 ... 63 = h8. Sliding attacks use the classical ray method (no magics).
#pragma once
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace az {

using u64 = uint64_t;
enum : int { WHITE = 0, BLACK = 1 };
enum : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr int NO_PIECE = 12;
inline int pieceOf(int c, int t) { return c * 6 + t; }
inline int colorOf(int p) { return p / 6; }
inline int typeOf(int p) { return p % 6; }

inline int lsb(u64 b) { return std::countr_zero(b); }
inline int msb(u64 b) { return 63 - std::countl_zero(b); }
inline int popLsb(u64& b) { int s = lsb(b); b &= b - 1; return s; }
inline u64 bit(int s) { return 1ull << s; }
inline u64 flipRanks(u64 b) { return __builtin_bswap64(b); }

// Move: from | to << 6 | promo << 12   (promo: 0 = none, otherwise KNIGHT..QUEEN)
using Move = uint16_t;
inline int mFrom(Move m) { return m & 63; }
inline int mTo(Move m) { return (m >> 6) & 63; }
inline int mPromo(Move m) { return m >> 12; }
inline Move mkMove(int f, int t, int p = 0) { return Move(f | (t << 6) | (p << 12)); }
constexpr int MAX_MOVES = 320;

// Castling rights bits
enum : int { WK = 1, WQ = 2, BK = 4, BQ = 8 };

struct Tables {
    u64 knight[64], king[64], pawn[2][64], ray[8][64];
    u64 zPiece[12][64], zCastle[16], zEp[8], zSide;
    uint8_t castleMask[64];
    // ray directions 0..3 increase the square index (first blocker = lsb), 4..7 decrease (msb)
    static constexpr int DF[8] = {0, 1, 1, -1, 0, -1, -1, 1};
    static constexpr int DR[8] = {1, 0, 1, 1, -1, 0, -1, -1};
    Tables() {
        auto on = [](int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; };
        for (int s = 0; s < 64; ++s) {
            int f = s & 7, r = s >> 3;
            knight[s] = king[s] = pawn[0][s] = pawn[1][s] = 0;
            const int nf[8] = {1, 2, 2, 1, -1, -2, -2, -1}, nr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
            for (int i = 0; i < 8; ++i) if (on(f + nf[i], r + nr[i])) knight[s] |= bit((r + nr[i]) * 8 + f + nf[i]);
            for (int df = -1; df <= 1; ++df)
                for (int dr = -1; dr <= 1; ++dr)
                    if ((df || dr) && on(f + df, r + dr)) king[s] |= bit((r + dr) * 8 + f + df);
            for (int df : {-1, 1}) {
                if (on(f + df, r + 1)) pawn[WHITE][s] |= bit((r + 1) * 8 + f + df);
                if (on(f + df, r - 1)) pawn[BLACK][s] |= bit((r - 1) * 8 + f + df);
            }
            for (int d = 0; d < 8; ++d) {
                ray[d][s] = 0;
                for (int k = 1; on(f + k * DF[d], r + k * DR[d]); ++k) ray[d][s] |= bit((r + k * DR[d]) * 8 + f + k * DF[d]);
            }
            castleMask[s] = 15;
        }
        castleMask[0] = 15 & ~WQ; castleMask[7] = 15 & ~WK; castleMask[4] = 15 & ~(WK | WQ);
        castleMask[56] = 15 & ~BQ; castleMask[63] = 15 & ~BK; castleMask[60] = 15 & ~(BK | BQ);
        std::mt19937_64 rng(0xC0FFEE123456789ull);
        for (auto& a : zPiece) for (auto& x : a) x = rng();
        for (auto& x : zCastle) x = rng();
        for (auto& x : zEp) x = rng();
        zSide = rng();
    }
};
inline const Tables TB{};

inline u64 slide(int d, int s, u64 occ) {
    u64 r = TB.ray[d][s], b = r & occ;
    if (b) r ^= TB.ray[d][d < 4 ? lsb(b) : msb(b)];
    return r;
}
inline u64 rookAtt(int s, u64 o) { return slide(0, s, o) | slide(1, s, o) | slide(4, s, o) | slide(5, s, o); }
inline u64 bishopAtt(int s, u64 o) { return slide(2, s, o) | slide(3, s, o) | slide(6, s, o) | slide(7, s, o); }

struct Position {
    u64 bb[12];
    u64 occ[2];
    uint8_t sq[64];
    uint8_t stm = WHITE, castle = 0;
    int8_t ep = -1;          // en-passant target square, set only when an enemy pawn can capture there
    int halfmove = 0, fullmove = 1;
    u64 key = 0;

    void clear() {
        std::memset(bb, 0, sizeof(bb));
        occ[0] = occ[1] = 0;
        std::memset(sq, NO_PIECE, sizeof(sq));
        stm = WHITE; castle = 0; ep = -1; halfmove = 0; fullmove = 1; key = 0;
    }
    void put(int p, int s) { bb[p] |= bit(s); occ[colorOf(p)] |= bit(s); sq[s] = uint8_t(p); key ^= TB.zPiece[p][s]; }
    void remove(int s) { int p = sq[s]; bb[p] ^= bit(s); occ[colorOf(p)] ^= bit(s); sq[s] = NO_PIECE; key ^= TB.zPiece[p][s]; }
    u64 all() const { return occ[0] | occ[1]; }
    int king(int c) const { return lsb(bb[pieceOf(c, KING)]); }

    bool attacked(int s, int by) const {
        u64 o = all();
        if (TB.pawn[by ^ 1][s] & bb[pieceOf(by, PAWN)]) return true;
        if (TB.knight[s] & bb[pieceOf(by, KNIGHT)]) return true;
        if (TB.king[s] & bb[pieceOf(by, KING)]) return true;
        u64 bq = bb[pieceOf(by, BISHOP)] | bb[pieceOf(by, QUEEN)];
        if (bq && (bishopAtt(s, o) & bq)) return true;
        u64 rq = bb[pieceOf(by, ROOK)] | bb[pieceOf(by, QUEEN)];
        if (rq && (rookAtt(s, o) & rq)) return true;
        return false;
    }
    bool inCheck() const { return attacked(king(stm), stm ^ 1); }

    void make(Move m) {
        int f = mFrom(m), t = mTo(m), pr = mPromo(m);
        int p = sq[f], us = stm, them = us ^ 1, ty = typeOf(p);
        if (ep >= 0) key ^= TB.zEp[ep & 7];
        int newEp = -1;
        ++halfmove;
        if (sq[t] != NO_PIECE) { remove(t); halfmove = 0; }
        if (ty == PAWN) {
            halfmove = 0;
            if (t == ep) remove(us == WHITE ? t - 8 : t + 8);
            if ((t ^ f) == 16) {
                int e = (f + t) / 2;
                if (TB.pawn[us][e] & bb[pieceOf(them, PAWN)]) newEp = e;
            }
        }
        remove(f);
        put(pr ? pieceOf(us, pr) : p, t);
        if (ty == KING && (t - f == 2 || f - t == 2)) {
            int rf = t > f ? f + 3 : f - 4, rt = t > f ? f + 1 : f - 1;
            int r = sq[rf];
            remove(rf);
            put(r, rt);
        }
        key ^= TB.zCastle[castle];
        castle &= TB.castleMask[f] & TB.castleMask[t];
        key ^= TB.zCastle[castle];
        ep = int8_t(newEp);
        if (ep >= 0) key ^= TB.zEp[ep & 7];
        stm = uint8_t(them);
        key ^= TB.zSide;
        if (us == BLACK) ++fullmove;
    }

    bool insufficientMaterial() const {
        if (bb[pieceOf(0, PAWN)] | bb[pieceOf(1, PAWN)] | bb[pieceOf(0, ROOK)] | bb[pieceOf(1, ROOK)] |
            bb[pieceOf(0, QUEEN)] | bb[pieceOf(1, QUEEN)]) return false;
        int minors = std::popcount(bb[pieceOf(0, KNIGHT)] | bb[pieceOf(1, KNIGHT)] | bb[pieceOf(0, BISHOP)] | bb[pieceOf(1, BISHOP)]);
        return minors <= 1;
    }

    bool setFen(const std::string& fen) {
        clear();
        std::istringstream is(fen);
        std::string board, side, cast, eps;
        is >> board >> side >> cast >> eps;
        int r = 7, f = 0;
        const std::string pcs = "PNBRQKpnbrqk";
        for (char ch : board) {
            if (ch == '/') { --r; f = 0; }
            else if (ch >= '1' && ch <= '8') f += ch - '0';
            else {
                auto i = pcs.find(ch);
                if (i == std::string::npos || r < 0 || f > 7) return false;
                put(int(i), r * 8 + f++);
            }
        }
        stm = side == "b" ? BLACK : WHITE;
        for (char ch : cast) castle |= ch == 'K' ? WK : ch == 'Q' ? WQ : ch == 'k' ? BK : ch == 'q' ? BQ : 0;
        if (eps.size() == 2) {
            int e = (eps[1] - '1') * 8 + (eps[0] - 'a');
            if (TB.pawn[stm ^ 1][e] & bb[pieceOf(stm, PAWN)]) ep = int8_t(e);
        }
        if (!(is >> halfmove)) halfmove = 0;
        if (!(is >> fullmove)) fullmove = 1;
        key ^= TB.zCastle[castle];
        if (ep >= 0) key ^= TB.zEp[ep & 7];
        if (stm == BLACK) key ^= TB.zSide;
        return std::popcount(bb[pieceOf(0, KING)]) == 1 && std::popcount(bb[pieceOf(1, KING)]) == 1;
    }
    void setStart() { setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

    std::string fen() const {
        const char* pcs = "PNBRQKpnbrqk";
        std::string s;
        for (int r = 7; r >= 0; --r) {
            int e = 0;
            for (int f = 0; f < 8; ++f) {
                int p = sq[r * 8 + f];
                if (p == NO_PIECE) { ++e; continue; }
                if (e) { s += char('0' + e); e = 0; }
                s += pcs[p];
            }
            if (e) s += char('0' + e);
            if (r) s += '/';
        }
        s += stm == WHITE ? " w " : " b ";
        std::string c;
        if (castle & WK) c += 'K';
        if (castle & WQ) c += 'Q';
        if (castle & BK) c += 'k';
        if (castle & BQ) c += 'q';
        s += c.empty() ? "-" : c;
        s += ' ';
        if (ep >= 0) { s += char('a' + (ep & 7)); s += char('1' + (ep >> 3)); }
        else s += '-';
        s += ' ' + std::to_string(halfmove) + ' ' + std::to_string(fullmove);
        return s;
    }
};

inline int genPseudo(const Position& p, Move* out) {
    int n = 0, us = p.stm, them = us ^ 1;
    u64 own = p.occ[us], opp = p.occ[them], all = own | opp;
    int fwd = us == WHITE ? 8 : -8, startRank = us == WHITE ? 1 : 6, lastRank = us == WHITE ? 7 : 0;
    auto addPawn = [&](int f, int t) {
        if ((t >> 3) == lastRank)
            for (int pr = QUEEN; pr >= KNIGHT; --pr) out[n++] = mkMove(f, t, pr);
        else out[n++] = mkMove(f, t);
    };
    u64 pawns = p.bb[pieceOf(us, PAWN)];
    u64 capMask = opp | (p.ep >= 0 ? bit(p.ep) : 0);
    while (pawns) {
        int f = popLsb(pawns), t = f + fwd;
        if (!(all & bit(t))) {
            addPawn(f, t);
            if ((f >> 3) == startRank && !(all & bit(t + fwd))) out[n++] = mkMove(f, t + fwd);
        }
        u64 caps = TB.pawn[us][f] & capMask;
        while (caps) addPawn(f, popLsb(caps));
    }
    for (int ty = KNIGHT; ty <= KING; ++ty) {
        u64 b = p.bb[pieceOf(us, ty)];
        while (b) {
            int f = popLsb(b);
            u64 a = ty == KNIGHT ? TB.knight[f] : ty == BISHOP ? bishopAtt(f, all) : ty == ROOK ? rookAtt(f, all)
                  : ty == QUEEN ? (bishopAtt(f, all) | rookAtt(f, all)) : TB.king[f];
            a &= ~own;
            while (a) out[n++] = mkMove(f, popLsb(a));
        }
    }
    if (p.castle & (us == WHITE ? (WK | WQ) : (BK | BQ))) {
        int k = us == WHITE ? 4 : 60;
        int rK = us == WHITE ? WK : BK, rQ = us == WHITE ? WQ : BQ;
        if (!p.attacked(k, them)) {
            if ((p.castle & rK) && !(all & (bit(k + 1) | bit(k + 2))) && !p.attacked(k + 1, them)) out[n++] = mkMove(k, k + 2);
            if ((p.castle & rQ) && !(all & (bit(k - 1) | bit(k - 2) | bit(k - 3))) && !p.attacked(k - 1, them)) out[n++] = mkMove(k, k - 2);
        }
    }
    return n;
}

inline int genLegal(const Position& p, Move* out) {
    Move tmp[MAX_MOVES];
    int n = genPseudo(p, tmp), k = 0;
    for (int i = 0; i < n; ++i) {
        Position q = p;
        q.make(tmp[i]);
        if (!q.attacked(q.king(p.stm), q.stm)) out[k++] = tmp[i];
    }
    return k;
}

inline std::string moveStr(Move m) {
    std::string s;
    s += char('a' + (mFrom(m) & 7)); s += char('1' + (mFrom(m) >> 3));
    s += char('a' + (mTo(m) & 7));   s += char('1' + (mTo(m) >> 3));
    if (mPromo(m)) s += " nbrq"[mPromo(m)];
    return s;
}
inline Move parseMove(const Position& p, const std::string& s) {
    Move ms[MAX_MOVES];
    int n = genLegal(p, ms);
    for (int i = 0; i < n; ++i) if (moveStr(ms[i]) == s) return ms[i];
    return 0;
}

inline uint64_t perft(const Position& p, int d) {
    Move ms[MAX_MOVES];
    int n = genLegal(p, ms);
    if (d == 1) return uint64_t(n);
    uint64_t s = 0;
    for (int i = 0; i < n; ++i) { Position q = p; q.make(ms[i]); s += perft(q, d - 1); }
    return s;
}

// Game-level outcome from the rules only (threefold repetition, 50 moves, material, mate, stalemate).
// `hist` holds the keys of all positions before `p` in the game.
enum class Outcome { ONGOING, CHECKMATE, STALEMATE, FIFTY, REPETITION, MATERIAL };
inline int repetitions(const Position& p, const std::vector<u64>& hist) {
    int cnt = 0, S = int(hist.size());
    for (int i = S - 2; i >= 0 && i >= S - p.halfmove; i -= 2) cnt += hist[i] == p.key;
    return cnt;
}
inline Outcome outcome(const Position& p, const std::vector<u64>& hist, int nLegal) {
    if (nLegal == 0) return p.inCheck() ? Outcome::CHECKMATE : Outcome::STALEMATE;
    if (p.halfmove >= 100) return Outcome::FIFTY;
    if (p.insufficientMaterial()) return Outcome::MATERIAL;
    if (repetitions(p, hist) >= 2) return Outcome::REPETITION;
    return Outcome::ONGOING;
}

}  // namespace az
