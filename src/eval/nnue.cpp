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

#include "nnue.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <string_view>

#include "../../3rdparty/zstd/zstd.h"

#include "../attacks/attacks.h"
#include "../util/align.h"
#include "../util/memstream.h"
#include "../util/numa/numa.h"
#include "header.h"
#include "nnue/features/threats/geometry.h"
#include "nnue/loader.h"

#ifdef _MSC_VER
    #define SP_MSVC
    #pragma push_macro("_MSC_VER")
    #undef _MSC_VER
#endif

#define INCBIN_PREFIX g_
#include "../../3rdparty/incbin.h"

#ifdef SP_MSVC
    #pragma pop_macro("_MSC_VER")
    #undef SP_MSVC
#endif

namespace {
    INCBIN(std::byte, defaultNet, SP_NETWORK_FILE);
} // namespace

namespace stormphrax::eval {
    namespace {
        inline std::string_view archName(u8 arch) {
            static constexpr std::array kNetworkArchNames = {
                "basic",
                "perspective",
                "perspective_multilayer",
                "perspective_multilayer_dual_act",
                "perspective_multilayer_skip_l2",
                "perspective_multilayer_dual_act_skip_l2",
            };

            if (arch < kNetworkArchNames.size()) {
                return kNetworkArchNames[arch];
            }

            return "<unknown>";
        }

        inline std::string_view activationFuncName(u8 func) {
            static constexpr std::array kActivationFunctionNames = {"crelu", "screlu", "relu"};

            if (func < kActivationFunctionNames.size()) {
                return kActivationFunctionNames[func];
            }

            return "<unknown>";
        }

        //TODO better error messages
        bool validate(const NetworkHeader& header) {
            if (header.magic != std::array{'C', 'B', 'N', 'F'}) {
                println("invalid magic bytes in network header");
                return false;
            }

            if (header.version != kExpectedHeaderVersion) {
                eprintln(
                    "unsupported network format version {} (expected: {})",
                    header.version,
                    kExpectedHeaderVersion
                );
                return false;
            }

            if (header.arch != LayeredArch::kArchId) {
                eprintln(
                    "wrong network architecture {} (expected: {})",
                    archName(header.arch),
                    archName(LayeredArch::kArchId)
                );
                return false;
            }

            if (testFlags(header.flags, NetworkFlags::kHorizontallyMirrored) != InputFeatureSet::kIsMirrored) {
                if constexpr (InputFeatureSet::kIsMirrored) {
                    eprintln("unmirrored network, expected horizontally mirrored");
                } else {
                    eprintln("horizontally mirrored network, expected unmirrored");
                }

                return false;
            }

            if (testFlags(header.flags, NetworkFlags::kMergedKings) != InputFeatureSet::kMergedKings) {
                if constexpr (InputFeatureSet::kMergedKings) {
                    eprintln("network does not have merged king planes, expected merged");
                } else {
                    eprintln("network has merged king planes, expected unmerged");
                }

                return false;
            }

            if (testFlags(header.flags, NetworkFlags::kPairwiseMul) != LayeredArch::kPairwise) {
                if constexpr (LayeredArch::kPairwise) {
                    eprintln("network L1 does not require pairwise multiplication, expected paired");
                } else {
                    eprintln("network L1 requires pairwise multiplication, expected unpaired");
                }

                return false;
            }

            if (header.activation != L1Activation::kId) {
                eprintln(
                    "wrong l1 activation function {} (expected: {})",
                    activationFuncName(header.activation),
                    activationFuncName(L1Activation::kId)
                );
                return false;
            }

            if (header.hiddenSize != kL1Size) {
                eprintln("wrong number of l1 neurons {} (expected: {})", header.hiddenSize, kL1Size);
                return false;
            }

            const bool headerThreatInputs = (header.inputBuckets & 0b10000000) != 0;
            const auto headerInputBuckets = header.inputBuckets & ~0b10000000;

            if (headerThreatInputs != InputFeatureSet::kThreatInputs) {
                if constexpr (InputFeatureSet::kThreatInputs) {
                    eprintln("network does not have the expected threat inputs");
                } else {
                    eprintln("network unexpectedly has threat inputs");
                }

                return false;
            }

            if (headerInputBuckets != InputFeatureSet::kBucketCount) {
                eprintln(
                    "wrong number of input buckets {} (expected: {})",
                    header.inputBuckets,
                    InputFeatureSet::kBucketCount
                );
                return false;
            }

            if (header.outputBuckets != OutputBucketing::kBucketCount) {
                eprintln(
                    "wrong number of output buckets {} (expected: {})",
                    header.outputBuckets,
                    OutputBucketing::kBucketCount
                );
                return false;
            }

            return true;
        }

        // must manually allocate for alignment
        std::byte* s_loadedNetworkData{nullptr};

#ifdef SP_USE_LIBNUMA
        std::unique_ptr<numa::NumaUniqueAllocation<std::byte>> s_networkData{};
        std::unique_ptr<numa::NumaUniqueAllocation<Network>> s_networks{};
#else
        Network s_network{};
#endif

        bool s_networkLoaded{false};
    } // namespace

    void init() {
        if (g_defaultNetSize < sizeof(NetworkHeader)) {
            eprintln("Missing default network?");
            return;
        }

        const auto& header = *reinterpret_cast<const NetworkHeader*>(g_defaultNetData);

        if (!validate(header)) {
            eprintln("Failed to validate default network header");
            return;
        }

        const auto networkSize = Network::byteSize();

        const bool compressed = testFlags(header.flags, NetworkFlags::kZstdCompressed);

        const std::byte* ptr;

        if (compressed) {
#ifndef SP_USE_LIBNUMA
            eprintln("Warning: default network is compressed and will not be shared between running instances");
#endif

            s_networkLoaded = false;

            if (!s_loadedNetworkData) {
                s_loadedNetworkData = util::alignedAlloc<std::byte>(util::simd::kAlignment, networkSize);
            }

            const auto decompressedSize = ZSTD_decompress(
                s_loadedNetworkData,
                networkSize,
                g_defaultNetData + sizeof(NetworkHeader),
                g_defaultNetSize - sizeof(NetworkHeader)
            );

            if (ZSTD_isError(decompressedSize)) {
                eprintln("Failed to decompress default network: {}", ZSTD_getErrorName(decompressedSize));
                return;
            }

            if (decompressedSize < networkSize) {
                eprintln("Decompressed default network too small? {} < {}", decompressedSize, networkSize);
                return;
            }

            ptr = s_loadedNetworkData;
        } else {
            const auto defaultNetSize = g_defaultNetSize - sizeof(NetworkHeader);

            if (defaultNetSize < networkSize) {
                eprintln("Default network too small? {} < {}", defaultNetSize, networkSize);
                return;
            }

            s_networkLoaded = false;

            if (s_loadedNetworkData) {
                util::alignedFree(s_loadedNetworkData);
                s_loadedNetworkData = nullptr;
            }

            ptr = g_defaultNetData + sizeof(NetworkHeader);
        }

#ifdef SP_USE_LIBNUMA
        s_networks = std::make_unique<numa::NumaUniqueAllocation<Network>>();
        s_networkData = std::make_unique<numa::NumaUniqueAllocation<std::byte>>(networkSize);

        const auto nodeCount = numa::nodeCount();
        for (i32 node = 0; node < nodeCount; ++node) {
            auto* target = s_networkData->get(node);
            std::memcpy(target, ptr, networkSize);
            nnue::NetworkLoader loader{target, networkSize};
            if (!s_networks->get(node)->loadFrom(loader, !compressed)) {
                eprintln("Failed to load default network on NUMA node {}", node);
                return;
            }
        }

        if (s_loadedNetworkData) {
            util::alignedFree(s_loadedNetworkData);
            s_loadedNetworkData = nullptr;
        }
#else
        nnue::NetworkLoader loader{ptr, networkSize};
        if (!s_network.loadFrom(loader, !compressed)) {
            eprintln("Failed to load default network");
            return;
        }
#endif

        s_networkLoaded = true;
    }

    void shutdown() {
        if (s_loadedNetworkData) {
            util::alignedFree(s_loadedNetworkData);
            s_loadedNetworkData = nullptr;
        }

#ifdef SP_USE_LIBNUMA
        s_networkData = nullptr;
        s_networks = nullptr;
#endif

        s_networkLoaded = false;
    }

    bool isNetworkLoaded() {
        return s_networkLoaded;
    }

    const Network* getNetwork(u32 numaId) {
#ifdef SP_USE_LIBNUMA
        return s_networks->get(numaId);
#else
        SP_UNUSED(numaId);
        return &s_network;
#endif
    }

    std::string_view defaultNetworkName() {
        const auto& header = *reinterpret_cast<const NetworkHeader*>(g_defaultNetData);
        return {header.name.data(), header.nameLen};
    }

    template void updatePieceThreatsOnChange<false>(NnueUpdates&, const Position&, Piece, Square);
    template void updatePieceThreatsOnChange<true>(NnueUpdates&, const Position&, Piece, Square);

    template <bool kAdd>
    void updatePieceThreatsOnChange(NnueUpdates& updates, const Position& pos, Piece piece, Square sq) {
        using namespace nnue::features::threats;

        // Generate a list of indexes and ray array for a given focus square sq.
        const auto permutation = geometry::permutationFor(sq);
        const auto [rays, bits] = geometry::permuteMailbox(permutation, pos.mailbox());

        // Determine all threats relative to the focus square sq.
        const auto closest = geometry::closestOccupied(bits);
        const auto outgoingThreats = geometry::outgoingThreats(piece, closest);
        const auto incomingAttackers = geometry::incomingAttackers(bits, closest);
        const auto incomingSliders = geometry::incomingSliders(bits, closest);

        // Push all focus square relative threats.
        if constexpr (kAdd) {
            updates.addFocusThreatFeatures(
                true,
                permutation.indexes.toArray<Square>(),
                rays.toArray<Piece>(),
                outgoingThreats,
                piece,
                sq
            );
            updates.addFocusThreatFeatures(
                false,
                permutation.indexes.toArray<Square>(),
                rays.toArray<Piece>(),
                incomingAttackers,
                piece,
                sq
            );
        } else {
            updates.removeFocusThreatFeatures(
                true,
                permutation.indexes.toArray<Square>(),
                rays.toArray<Piece>(),
                outgoingThreats,
                piece,
                sq
            );
            updates.removeFocusThreatFeatures(
                false,
                permutation.indexes.toArray<Square>(),
                rays.toArray<Piece>(),
                incomingAttackers,
                piece,
                sq
            );
        }

        // Discover threat updates from sliders whose threats needs to be extended (if focus piece removed)
        // or retracted (if focus piece added).
        // A valid threat is when one ray has a slider on it, with a vicitim on the opposite ray.
        // For example, a bishop on NW ray and blocker on the SE ray.
        // Here we just detect all valid discovered threats.
        const auto victimMask = std::rotr(closest & 0xFEFEFEFEFEFEFEFE, 32);
        const auto valid = geometry::rayFill(victimMask) & geometry::rayFill(incomingSliders);

        if constexpr (kAdd) {
            updates.removeDiscoveredThreatFeatures(
                permutation.indexes.toArray<Square>(),
                rays.toArray<Piece>(),
                incomingSliders & valid,
                victimMask & valid
            );
        } else {
            updates.addDiscoveredThreatFeatures(
                permutation.indexes.toArray<Square>(),
                rays.toArray<Piece>(),
                incomingSliders & valid,
                victimMask & valid
            );
        }
    }

    void updatePieceThreatsOnMutate(
        NnueUpdates& updates,
        const Position& pos,
        Piece oldPiece,
        Piece newPiece,
        Square sq
    ) {
        using namespace nnue::features::threats;

        // Generate a list of indexes and ray array for a given focus square sq.
        const auto permutation = geometry::permutationFor(sq);
        const auto [rays, bits] = geometry::permuteMailbox(permutation, pos.mailbox());

        // Determine all threats relative to the focus square sq.
        const auto closest = geometry::closestOccupied(bits);
        const auto oldOutgoingThreats = geometry::outgoingThreats(oldPiece, closest);
        const auto newOutgoingThreats = geometry::outgoingThreats(newPiece, closest);
        const auto incomingAttackers = geometry::incomingAttackers(bits, closest);

        // Push all focus square relative threats.
        updates.removeFocusThreatFeatures(
            true,
            permutation.indexes.toArray<Square>(),
            rays.toArray<Piece>(),
            oldOutgoingThreats,
            oldPiece,
            sq
        );
        updates.addFocusThreatFeatures(
            true,
            permutation.indexes.toArray<Square>(),
            rays.toArray<Piece>(),
            newOutgoingThreats,
            newPiece,
            sq
        );
        updates.removeFocusThreatFeatures(
            false,
            permutation.indexes.toArray<Square>(),
            rays.toArray<Piece>(),
            incomingAttackers,
            oldPiece,
            sq
        );
        updates.addFocusThreatFeatures(
            false,
            permutation.indexes.toArray<Square>(),
            rays.toArray<Piece>(),
            incomingAttackers,
            newPiece,
            sq
        );
    }

    void updatePieceThreatsOnMove(
        NnueUpdates& updates,
        const Position& pos,
        Piece oldPiece,
        Square src,
        Piece newPiece,
        Square dst
    ) {
        using namespace nnue::features::threats;

        const auto srcPerm = geometry::permutationFor(src);
        const auto dstPerm = geometry::permutationFor(dst);
        const auto [srcRays, srcBits] = geometry::permuteMailbox(srcPerm, pos.mailbox(), dst);
        const auto [dstRays, dstBits] = geometry::permuteMailbox(dstPerm, pos.mailbox());

        const auto srcClosest = geometry::closestOccupied(srcBits);
        const auto dstClosest = geometry::closestOccupied(dstBits);
        const auto srcOutgoingThreats = geometry::outgoingThreats(oldPiece, srcClosest);
        const auto dstOutgoingThreats = geometry::outgoingThreats(newPiece, dstClosest);
        const auto srcIncomingAttackers = geometry::incomingAttackers(srcBits, srcClosest);
        const auto dstIncomingAttackers = geometry::incomingAttackers(dstBits, dstClosest);
        const auto srcIncomingSliders = geometry::incomingSliders(srcBits, srcClosest);
        const auto dstIncomingSliders = geometry::incomingSliders(dstBits, dstClosest);

        updates.removeFocusThreatFeatures(
            true,
            srcPerm.indexes.toArray<Square>(),
            srcRays.toArray<Piece>(),
            srcOutgoingThreats,
            oldPiece,
            src
        );
        updates.addFocusThreatFeatures(
            true,
            dstPerm.indexes.toArray<Square>(),
            dstRays.toArray<Piece>(),
            dstOutgoingThreats,
            newPiece,
            dst
        );
        updates.removeFocusThreatFeatures(
            false,
            srcPerm.indexes.toArray<Square>(),
            srcRays.toArray<Piece>(),
            srcIncomingAttackers,
            oldPiece,
            src
        );
        updates.addFocusThreatFeatures(
            false,
            dstPerm.indexes.toArray<Square>(),
            dstRays.toArray<Piece>(),
            dstIncomingAttackers,
            newPiece,
            dst
        );

        const auto srcVictimMask = std::rotr(srcClosest & 0xFEFEFEFEFEFEFEFE, 32);
        const auto dstVictimMask = std::rotr(dstClosest & 0xFEFEFEFEFEFEFEFE, 32);
        const auto srcValid = geometry::rayFill(srcVictimMask) & geometry::rayFill(srcIncomingSliders);
        const auto dstValid = geometry::rayFill(dstVictimMask) & geometry::rayFill(dstIncomingSliders);

        updates.addDiscoveredThreatFeatures(
            srcPerm.indexes.toArray<Square>(),
            srcRays.toArray<Piece>(),
            srcIncomingSliders & srcValid,
            srcVictimMask & srcValid
        );
        updates.removeDiscoveredThreatFeatures(
            dstPerm.indexes.toArray<Square>(),
            dstRays.toArray<Piece>(),
            dstIncomingSliders & dstValid,
            dstVictimMask & dstValid
        );
    }
} // namespace stormphrax::eval
