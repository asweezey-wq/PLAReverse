#pragma once
#include "pokemonVerif.cuh"
#include "deviceConstants.cuh"
#include "xoroshiro.cuh"
#include <iostream>

constexpr uint64_t XOROSHIRO_ROTL_24 = 0x75229d6a5b82a2b1;
#define GENERATOR_OUTPUT_BUFFER_SIZE 4

__device__ __forceinline__ uint64_t x0_from_x1(uint64_t x1) {
    x1 = rotl(x1, 27);
    return XOROSHIRO_ROTL_24 ^ x1 ^ (x1 << 16) ^ rotl(x1, 24);
}

__device__ __forceinline__ uint64_t seed_from_x1(uint64_t x1) {
    return rotl(x1, 27) ^ XOROSHIRO_CONSTANT;
}

__device__ __forceinline__ uint64_t getBaseSeed(uint8_t ivs[6]) {
    uint8_t threadIVs[6] = {
        (uint8_t)(threadIdx.x),
        (uint8_t)(threadIdx.y),
        (uint8_t)((blockIdx.x >> 0) & 31),
        (uint8_t)((blockIdx.x >> 5) & 31),
        (uint8_t)((blockIdx.y >> 0) & 31),
        (uint8_t)((blockIdx.y >> 5) & 31)
    };
    uint64_t packedIvInput = 0;
    for (int i = 0; i < 6; i++) {
        uint8_t sub = ivs[i] - threadIVs[i];
        packedIvInput |= ((uint64_t)(sub & 31) << (10 * i));
        packedIvInput |= ((uint64_t)threadIVs[i] << (10 * i + 5));
    }
    packedIvInput ^= g_seedReversalCtx.pokemonReversalCtx.xoroshiroBits;
    uint64_t baseSeed = 0;
    for (int i = 0; i < 60; i++) {
        if (packedIvInput & 1) {
            baseSeed ^= g_seedReversalCtx.pokemonReversalCtx.inverseMatrix[i];
        }
        packedIvInput >>= 1;
        if (!packedIvInput) {
            break;
        }
    }
    return baseSeed;
}

__device__ __forceinline__ void findGroupSeeds(uint64_t seed, uint32_t groupIndex, uint32_t* outputCount, uint64_t* outputBuffer) {
    // Expected 256 blockDim.x
    const GroupSeedReversalContext& groupReversalCtx = g_seedReversalCtx.groupReversalCtx;
    uint64_t index = (blockIdx.x << 8) | (threadIdx.x);
    uint32_t seedSlice = (uint32_t)((seed >> groupReversalCtx.shiftConst[groupIndex]) & 0xFFFFFFFF);
    uint32_t seedSub = (seedSlice - index) - threadIdx.y;
    uint64_t seedBits = seedSub | (index << 32);

    seedBits ^= groupReversalCtx.xoroshiroBits[groupIndex];
    uint64_t groupSeed = 0;
    for (int i = 0; i < 64; i++) {
        if (seedBits & 1) {
            groupSeed ^= groupReversalCtx.inverseMatrix[groupIndex][i];
        }
        seedBits >>= 1;
    }
    Xoroshiro rng = {groupSeed};
    // It is implied that index 0 is the second generated pokemon
    // The first generated pokemon can easily have its group seed reversed
    for (int i = 0; i <= groupIndex; i++) {
        xoroshiroNext(&rng);
        xoroshiroNext(&rng);
    }
    if (xoroshiroNext(&rng) == seed) {
        if (verifyGroupSeed(groupSeed)) {
            printf("Found group seed! Index:%u Seed: 0x%llx %llu\n", groupIndex+1, groupSeed, groupSeed);
            outputBuffer[atomicAdd(outputCount, 1)] = groupSeed;
        }
    }
}


__device__ __forceinline__ uint32_t findGeneratorSeeds(uint64_t seed, uint64_t* outputBuffer, const uint64_t slices[256]) {
    uint32_t count = 0;
    for (int i = 0; i < 32768; i++) {
        const unsigned int x = ((i >> 0) & 31) * 8 + threadIdx.x;
        const unsigned int y = ((i >> 5) & 31) * 8 + threadIdx.y;
        const unsigned int z = ((i >> 10) & 31) * 8 + threadIdx.z;
        // printf("%d %d %d\n", x, y, z);

        const uint64_t x1_slice = slices[x] | (slices[y] << 1) | (slices[z] << 2);
        const uint64_t x0_slice = x0_from_x1(x1_slice) & 0x3838383838383838ULL;
        const uint64_t sub = seed - x0_slice - x1_slice;

        const uint64_t base_x1_slice_3 = sub & 0x808080808080808ULL;
        const uint64_t sub_carry = sub - 0xc0c0c0c0c0c0c0c0ULL;
        const uint64_t changed = (sub_carry ^ sub) & 0x808080808080808ULL;

        for (uint64_t i0 = 0; i0 <= ((changed >> 11) & 1); i0++) {
            uint64_t part_0 = i0 << 11;
            for (uint64_t i1 = 0; i1 <= ((changed >> 19) & 1); i1++) {
                uint64_t part_1 = part_0 | (i1 << 19);
                for (uint64_t i2 = 0; i2 <= ((changed >> 27) & 1); i2++) {
                    uint64_t part_2 = part_1 | (i2 << 27);
                    for (uint64_t i3 = 0; i3 <= ((changed >> 35) & 1); i3++) {
                        uint64_t part_3 = part_2 | (i3 << 35);
                        for (uint64_t i4 = 0; i4 <= ((changed >> 43) & 1); i4++) {
                            uint64_t part_4 = part_3 | (i4 << 43);
                            for (uint64_t i5 = 0; i5 <= ((changed >> 51) & 1); i5++) {
                                uint64_t part_5 = part_4 | (i5 << 51);
                                for (uint64_t i6 = 0; i6 <= ((changed >> 59) & 1); i6++) {
                                    uint64_t part_6 = part_5 | (i6 << 59);
                                    uint64_t x1_slice_3 = base_x1_slice_3 ^ part_6;
                                    uint64_t x0_slice_3 = x0_from_x1(x1_slice_3) & 0x4040404040404040ULL;

                                    uint64_t x0_slice_ = x0_slice | x0_slice_3;
                                    uint64_t x1_slice_ = x1_slice | x1_slice_3;

                                    uint64_t x1_slice_4 = (seed - x0_slice_ - x1_slice_) & 0x1010101010101010ULL;
                                    uint64_t x0_slice_4 = x0_from_x1(x1_slice_4) & 0x8080808080808080ULL;

                                    x0_slice_ |= x0_slice_4;
                                    x1_slice_ |= x1_slice_4;

                                    uint64_t x1_slice_567 = (seed - x0_slice_ - x1_slice_) & 0xe0e0e0e0e0e0e0e0ULL;
                                    uint64_t x0_slice_567 = x0_from_x1(x1_slice_567) & 0x707070707070707ULL;

                                    uint64_t x0 = x0_slice_ | x0_slice_567;
                                    uint64_t x1 = x1_slice_ | x1_slice_567;

                                    if ((x0 + x1) == seed) {
                                        uint64_t seed = seed_from_x1(x1);
                                        outputBuffer[count++] = seed;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return count;
}