/*
 * Stormphrax, a UCI chess engine
 * Copyright (C) 2026 Ciekce
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

#include "../../../../types.h"

#include <array>

#include "../../../nnue.h"
#include "../psq.h"
#include "geometry.h"

namespace stormphrax::eval::nnue::features::threats::push {
    using UpdatedThreat = nnue::features::psq::ThreatDescriptor;

#if SP_HAS_VBMI2
    static_assert(sizeof(UpdatedThreat) == sizeof(u32));
    static_assert(offsetof(UpdatedThreat, attacker) == 0 * sizeof(u8));
    static_assert(offsetof(UpdatedThreat, attackerSq) == 1 * sizeof(u8));
    static_assert(offsetof(UpdatedThreat, attacked) == 2 * sizeof(u8));
    static_assert(offsetof(UpdatedThreat, attackedSq) == 3 * sizeof(u8));

    static inline __m512i flip(__m512i x) {
        return _mm512_shuffle_i64x2(x, x, 0b01001110);
    }

    inline void pushFocusThreatFeatures(
        StaticVector<UpdatedThreat, 128>& features,
        bool outgoing,
        std::array<Square, 64> otherSqs, // List of square indexes
        std::array<Piece, 64> others,    // List of pieces on those squares as indexed by indexes
        geometry::Bitrays br,            // Bitrays where bit set is a piece attacked/being attacked by focus square
        Piece piece,                     // Piece on the focus square
        Square sq                        // The focus square
    ) {
        const auto indexes = _mm512_loadu_si512(otherSqs.data());
        const auto rays = _mm512_loadu_si512(others.data());

        // Create the (piecea, squarea, pieceb, squareb) tuples.
        // Whether pair1 or pair2 is paira or pairb is determined by kOutgoing.

        // clang-format off
            const auto pair2Shuffle = _mm512_set_epi8(
                79, 15, 79, 15, 78, 14, 78, 14, 77, 13, 77, 13, 76, 12, 76, 12, 75, 11, 75, 11,
                74, 10, 74, 10, 73, 9, 73, 9, 72, 8, 72, 8, 71, 7, 71, 7, 70, 6, 70, 6, 69, 5,
                69, 5, 68, 4, 68, 4, 67, 3, 67, 3, 66, 2, 66, 2, 65, 1, 65, 1, 64, 0, 64, 0
            );
        // clang-format on

        // Focus pair
        const auto pair1 = _mm512_set1_epi16(static_cast<i16>(piece.idx() | (sq.idx() << 8)));

        // Non-focus pair
        const auto pair2Sq = _mm512_maskz_compress_epi8(br, indexes);
        const auto pair2Piece = _mm512_maskz_compress_epi8(br, rays);
        const auto pair2 = _mm512_permutex2var_epi8(pair2Piece, pair2Shuffle, pair2Sq);

        // Select which is the attacker and which is the victim.
        const u64 mask = outgoing ? 0xCCCCCCCCCCCCCCCC : 0x3333333333333333;
        const auto vector = _mm512_mask_mov_epi8(pair1, mask, pair2);

        features.unsafeWrite([&](UpdatedThreat* ptr) {
            _mm512_storeu_si512(ptr, vector);
            return std::popcount(br);
        });
    }

    inline void pushDiscoveredThreatFeatures(
        StaticVector<UpdatedThreat, 128>& features,
        std::array<Square, 64> squares,
        std::array<Piece, 64> pieces,
        geometry::Bitrays sliders,
        geometry::Bitrays victims
    ) {
        const auto indexes = _mm512_loadu_si512(squares.data());
        const auto rays = _mm512_loadu_si512(pieces.data());

        const auto count = std::popcount(victims);
        assert(std::popcount(victims) == std::popcount(sliders));

        // Create the (piece1, square1, piece2, square2) tuples.

        const auto p1 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders, rays));
        const auto sq1 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders, indexes));
        const auto p2 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(victims, flip(rays)));
        const auto sq2 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(victims, flip(indexes)));

        const auto pair1 = _mm_unpacklo_epi8(p1, sq1);
        const auto pair2 = _mm_unpacklo_epi8(p2, sq2);

        const auto tuple1 = _mm_unpacklo_epi16(pair1, pair2);
        const auto tuple2 = _mm_unpackhi_epi16(pair1, pair2);

        features.unsafeWrite([&](UpdatedThreat* ptr) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr) + 0, tuple1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr) + 1, tuple2);
            return count;
        });
    }
#else
    inline void pushFocusThreatFeatures(
        StaticVector<UpdatedThreat, 128>& features,
        bool outgoing,
        std::array<Square, 64> otherSqs, // List of square indexes
        std::array<Piece, 64> others,    // List of pieces on those squares as indexed by indexes
        geometry::Bitrays br,            // Bitrays where bit set is a piece attacked/being attacked by focus square
        Piece piece,                     // Piece on the focus square
        Square sq                        // The focus square
    ) {
        for (; br; br = util::resetLsb(br)) {
            const auto i = util::ctz(br);

            const auto other = others[i];
            const auto otherSq = otherSqs[i];

            const auto attacker = outgoing ? piece : other;
            const auto attackerSq = outgoing ? sq : otherSq;
            const auto attacked = outgoing ? other : piece;
            const auto attackedSq = outgoing ? otherSq : sq;

            features.push({
                .attacker = attacker,
                .attackerSq = attackerSq,
                .attacked = attacked,
                .attackedSq = attackedSq,
            });
        }
    }

    inline void pushDiscoveredThreatFeatures(
        StaticVector<UpdatedThreat, 128>& features,
        std::array<Square, 64> squares,
        std::array<Piece, 64> pieces,
        geometry::Bitrays sliders,
        geometry::Bitrays victims
    ) {
        for (; sliders; sliders = util::resetLsb(sliders), victims = util::resetLsb(victims)) {
            const auto slider = util::ctz(sliders);
            const auto victim = util::ctz(victims);

            const auto attacker = pieces[slider];
            const auto attackerSq = squares[slider];
            const auto attacked = pieces[(victim + 32) % 64];
            const auto attackedSq = squares[(victim + 32) % 64];

            features.push({
                .attacker = attacker,
                .attackerSq = attackerSq,
                .attacked = attacked,
                .attackedSq = attackedSq,
            });
        }

        assert(!sliders && !victims);
    }
#endif
} // namespace stormphrax::eval::nnue::features::threats::push
