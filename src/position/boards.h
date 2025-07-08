/*
 * Stormphrax, a UCI chess engine
 * Copyright (C) 2025 Ciekce
 *
 * Stormphrax is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Stormphrax is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Stormphrax. If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "../types.h"

#include <array>
#include <bit>
#include <immintrin.h>

#include "../bitboard.h"

namespace stormphrax {
    class BitboardSet {
    public:
        [[nodiscard]] inline Bitboard& forColor(Color color) {
            return m_colors[static_cast<i32>(color)];
        }

        [[nodiscard]] inline Bitboard forColor(Color color) const {
            return m_colors[static_cast<i32>(color)];
        }

        [[nodiscard]] inline Bitboard& forPiece(PieceType piece) {
            return m_pieces[static_cast<i32>(piece)];
        }

        [[nodiscard]] inline Bitboard forPiece(PieceType piece) const {
            return m_pieces[static_cast<i32>(piece)];
        }

        [[nodiscard]] inline Bitboard forPiece(PieceType piece, Color c) const {
            return m_pieces[static_cast<i32>(piece)] & forColor(c);
        }

        [[nodiscard]] inline Bitboard forPiece(Piece piece) const {
            return forPiece(pieceType(piece), pieceColor(piece));
        }

        [[nodiscard]] inline Bitboard blackOccupancy() const {
            return m_colors[0];
        }

        [[nodiscard]] inline Bitboard whiteOccupancy() const {
            return m_colors[1];
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard occupancy() const {
            return m_colors[static_cast<i32>(kC)];
        }

        [[nodiscard]] inline Bitboard occupancy(Color c) const {
            return m_colors[static_cast<i32>(c)];
        }

        [[nodiscard]] inline Bitboard occupancy() const {
            return m_colors[0] | m_colors[1];
        }

        [[nodiscard]] inline Bitboard pawns() const {
            return forPiece(PieceType::kPawn);
        }

        [[nodiscard]] inline Bitboard knights() const {
            return forPiece(PieceType::kKnight);
        }

        [[nodiscard]] inline Bitboard bishops() const {
            return forPiece(PieceType::kBishop);
        }

        [[nodiscard]] inline Bitboard rooks() const {
            return forPiece(PieceType::kRook);
        }

        [[nodiscard]] inline Bitboard queens() const {
            return forPiece(PieceType::kQueen);
        }

        [[nodiscard]] inline Bitboard kings() const {
            return forPiece(PieceType::kKing);
        }

        [[nodiscard]] inline Bitboard blackPawns() const {
            return pawns() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whitePawns() const {
            return pawns() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard blackKnights() const {
            return knights() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteKnights() const {
            return knights() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard blackBishops() const {
            return bishops() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteBishops() const {
            return bishops() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard blackRooks() const {
            return rooks() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteRooks() const {
            return rooks() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard blackQueens() const {
            return queens() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteQueens() const {
            return queens() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard blackKings() const {
            return kings() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteKings() const {
            return kings() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard minors() const {
            return knights() | bishops();
        }

        [[nodiscard]] inline Bitboard blackMinors() const {
            return minors() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteMinors() const {
            return minors() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard majors() const {
            return rooks() | queens();
        }

        [[nodiscard]] inline Bitboard blackMajors() const {
            return majors() & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteMajors() const {
            return majors() & whiteOccupancy();
        }

        [[nodiscard]] inline Bitboard nonPk() const {
            return occupancy() ^ pawns() ^ kings();
        }

        [[nodiscard]] inline Bitboard blackNonPk() const {
            return blackOccupancy() ^ (pawns() | kings()) & blackOccupancy();
        }

        [[nodiscard]] inline Bitboard whiteNonPk() const {
            return whiteOccupancy() ^ (pawns() | kings()) & whiteOccupancy();
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard pawns() const {
            if constexpr (kC == Color::kBlack) {
                return blackPawns();
            } else {
                return whitePawns();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard knights() const {
            if constexpr (kC == Color::kBlack) {
                return blackKnights();
            } else {
                return whiteKnights();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard bishops() const {
            if constexpr (kC == Color::kBlack) {
                return blackBishops();
            } else {
                return whiteBishops();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard rooks() const {
            if constexpr (kC == Color::kBlack) {
                return blackRooks();
            } else {
                return whiteRooks();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard queens() const {
            if constexpr (kC == Color::kBlack) {
                return blackQueens();
            } else {
                return whiteQueens();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard kings() const {
            if constexpr (kC == Color::kBlack) {
                return blackKings();
            } else {
                return whiteKings();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard minors() const {
            if constexpr (kC == Color::kBlack) {
                return blackMinors();
            } else {
                return whiteMinors();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard majors() const {
            if constexpr (kC == Color::kBlack) {
                return blackMajors();
            } else {
                return whiteMajors();
            }
        }

        template <Color kC>
        [[nodiscard]] inline Bitboard nonPk() const {
            if constexpr (kC == Color::kBlack) {
                return blackNonPk();
            } else {
                return whiteNonPk();
            }
        }

        [[nodiscard]] inline Bitboard pawns(Color color) const {
            return forPiece(PieceType::kPawn, color);
        }

        [[nodiscard]] inline Bitboard knights(Color color) const {
            return forPiece(PieceType::kKnight, color);
        }

        [[nodiscard]] inline Bitboard bishops(Color color) const {
            return forPiece(PieceType::kBishop, color);
        }

        [[nodiscard]] inline Bitboard rooks(Color color) const {
            return forPiece(PieceType::kRook, color);
        }

        [[nodiscard]] inline Bitboard queens(Color color) const {
            return forPiece(PieceType::kQueen, color);
        }

        [[nodiscard]] inline Bitboard kings(Color color) const {
            return forPiece(PieceType::kKing, color);
        }

        [[nodiscard]] inline Bitboard minors(Color color) const {
            return color == Color::kBlack ? blackMinors() : whiteMinors();
        }

        [[nodiscard]] inline Bitboard majors(Color color) const {
            return color == Color::kBlack ? blackMajors() : whiteMajors();
        }

        [[nodiscard]] inline Bitboard nonPk(Color color) const {
            return color == Color::kBlack ? blackNonPk() : whiteNonPk();
        }

        [[nodiscard]] inline bool operator==(const BitboardSet& other) const = default;

    private:
        std::array<Bitboard, 6> m_pieces{};
        std::array<Bitboard, 2> m_colors{};
    };

    class PositionBoards {
    public:
        [[nodiscard]] inline const BitboardSet& bbs() const {
            return m_bbs;
        }

        [[nodiscard]] inline BitboardSet& bbs() {
            return m_bbs;
        }

        [[nodiscard]] inline PieceType pieceTypeAt(Square square) const {
            assert(square != Square::kNone);

            const auto p = squareLookup(square);

            if (p == 0) {
                return PieceType::kNone;
            }

            return static_cast<PieceType>(std::countr_zero(p));
        }

        [[nodiscard]] inline Piece pieceOn(Square square) const {
            assert(square != Square::kNone);

            const auto p = squareLookup(square);

            if (p == 0) {
                return Piece::kNone;
            }

            const auto color = static_cast<Color>(p >> 7);
            const auto piece = static_cast<PieceType>(std::countr_zero(p));

            return colorPiece(piece, color);
        }

        [[nodiscard]] inline Piece pieceAt(u32 rank, u32 file) const {
            return pieceOn(toSquare(rank, file));
        }

        inline void setPiece(Square square, Piece piece) {
            assert(square != Square::kNone);
            assert(piece != Piece::kNone);

            assert(pieceOn(square) == Piece::kNone);

            const auto mask = Bitboard::fromSquare(square);

            m_bbs.forPiece(pieceType(piece)) ^= mask;
            m_bbs.forColor(pieceColor(piece)) ^= mask;
        }

        inline void movePiece(Square src, Square dst, Piece piece) {
            assert(src != Square::kNone);
            assert(dst != Square::kNone);

            const auto mask = Bitboard::fromSquare(src) ^ Bitboard::fromSquare(dst);

            m_bbs.forPiece(pieceType(piece)) ^= mask;
            m_bbs.forColor(pieceColor(piece)) ^= mask;
        }

        inline void moveAndChangePiece(Square src, Square dst, Piece moving, PieceType promo) {
            assert(src != Square::kNone);
            assert(dst != Square::kNone);
            assert(src != dst);

            assert(moving != Piece::kNone);
            assert(promo != PieceType::kNone);

            assert(pieceOn(src) == moving);

            m_bbs.forPiece(pieceType(moving))[src] = false;
            m_bbs.forPiece(promo)[dst] = true;

            const auto mask = Bitboard::fromSquare(src) ^ Bitboard::fromSquare(dst);
            m_bbs.forColor(pieceColor(moving)) ^= mask;
        }

        inline void removePiece(Square square, Piece piece) {
            assert(square != Square::kNone);
            assert(piece != Piece::kNone);

            assert(pieceOn(square) == piece);

            m_bbs.forPiece(pieceType(piece))[square] = false;
            m_bbs.forColor(pieceColor(piece))[square] = false;
        }

        inline void regenFromBbs() {}

        [[nodiscard]] inline bool operator==(const PositionBoards& other) const = default;

    private:
        [[nodiscard]] inline u8 squareLookup(Square square) const {
            static_assert(sizeof(BitboardSet) == sizeof(__m512i));
            const auto bit = _mm512_set1_epi64(static_cast<i64>(squareBit(square)));
            return _mm512_test_epi64_mask(std::bit_cast<__m512i>(m_bbs), bit);
        }

        BitboardSet m_bbs{};
    };
} // namespace stormphrax
