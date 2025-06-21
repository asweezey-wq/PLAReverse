#include "pokemon.cuh"
#include "xoroshiro.cuh"
#include "pokemon_cuda.h"
#include "ivReversal.cuh"
#include "generatorReversal.cuh"
#include <iostream>

// __constant__ SeedReverseConstantsFlexible g_reversalConstants;
__constant__ SeedVerifyConstantsFlexible g_multiVerifyConstants[4];
__constant__ uint64_t g_slices[256];

__device__ bool verifyGeneratorSeed(uint64_t seed, const SeedVerifyConstantsFlexible& verifyConst) {
    Xoroshiro rng = {seed};
    xoroshiroNext(&rng);
    uint64_t pokeSeed = xoroshiroNext(&rng);
    bool valid = verifySeed(pokeSeed, g_reversalConstants, verifyConst);
    return valid;
}

__device__ uint32_t findGeneratorSeeds2(uint64_t seed, uint64_t outputBuffer[GENERATOR_OUTPUT_BUFFER_SIZE], const uint64_t slices[256]) {
    const unsigned int x = ((blockIdx.x >> 0) & 31) * 8 + threadIdx.x;
    const unsigned int y = ((blockIdx.x >> 5) & 31) * 8 + threadIdx.y;
    const unsigned int z = ((blockIdx.x >> 10) & 31) * 8 + threadIdx.z;

    const uint64_t x1_slice = slices[x] | (slices[y] << 1) | (slices[z] << 2);
    const uint64_t x0_slice = x0_from_x1(x1_slice) & 0x3838383838383838ULL;
    const uint64_t sub = seed - x0_slice - x1_slice;

    const uint64_t base_x1_slice_3 = sub & 0x808080808080808ULL;
    const uint64_t sub_carry = sub - 0xc0c0c0c0c0c0c0c0ULL;
    const uint64_t changed = (sub_carry ^ sub) & 0x808080808080808ULL;

    uint32_t count = 0;

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
    return count;
}

__device__ uint32_t findGeneratorSeeds3(uint64_t seed, uint64_t outputBuffer[GENERATOR_OUTPUT_BUFFER_SIZE], const uint64_t slices[256]) {
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

        // if (!changed) {
        //     uint64_t x1_slice_3 = base_x1_slice_3;
        //     uint64_t x0_slice_3 = x0_from_x1(x1_slice_3) & 0x4040404040404040ULL;

        //     uint64_t x0_slice_ = x0_slice | x0_slice_3;
        //     uint64_t x1_slice_ = x1_slice | x1_slice_3;

        //     uint64_t x1_slice_4 = (seed - x0_slice_ - x1_slice_) & 0x1010101010101010ULL;
        //     uint64_t x0_slice_4 = x0_from_x1(x1_slice_4) & 0x8080808080808080ULL;

        //     x0_slice_ |= x0_slice_4;
        //     x1_slice_ |= x1_slice_4;

        //     uint64_t x1_slice_567 = (seed - x0_slice_ - x1_slice_) & 0xe0e0e0e0e0e0e0e0ULL;
        //     uint64_t x0_slice_567 = x0_from_x1(x1_slice_567) & 0x707070707070707ULL;

        //     uint64_t x0 = x0_slice_ | x0_slice_567;
        //     uint64_t x1 = x1_slice_ | x1_slice_567;

        //     if ((x0 + x1) == seed) {
        //         uint64_t seed = seed_from_x1(x1);
        //         outputBuffer[0] = seed;
        //         return 1;
        //     }
        //     return 0;
        // }


        // uint32_t numPossibleSeeds = 1 << (((changed >> 11) & 1) + ((changed >> 19) & 1) + ((changed >> 27) & 1) + ((changed >> 35) & 1) + ((changed >> 43) & 1) + ((changed >> 51) & 1) + ((changed >> 59) & 1));
        

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

__global__
void kernel_reverseIVSetMulti(uint32_t* numFoundSeeds, uint64_t* outputBuffer) {
    uint8_t ivs[6];
    for (int i = 0; i < 6; i++) {
        ivs[i] = g_multiVerifyConstants[blockIdx.z].ivs[i][0];
    }
    uint64_t baseSeed = getBaseSeed(ivs);
    for (int i = 0; i < 16; i++) {
        uint64_t permutedSeed = baseSeed ^ g_reversalConstants.nullspace[i];
        if (verifySeedNoIVs(permutedSeed, g_reversalConstants, g_multiVerifyConstants[blockIdx.z])) {
            outputBuffer[atomicAdd(numFoundSeeds, 1)] = permutedSeed;
        }
    }
}

__global__
void kernel_reverseIVSetOffset(uint32_t* numFoundSeeds, int index, uint64_t* outputBuffer) {
    uint8_t threadIVAdded[6] = {0};
    uint8_t threadIVRanges[6] = {0};
    for (int i = 0; i < 6; i++) {
        threadIVRanges[i] = 1 + g_multiVerifyConstants[index].ivs[i][1] - g_multiVerifyConstants[index].ivs[i][0];
    }
    uint8_t ivs[6];
    bool exit = false;
    while (!exit) {
        for (int i = 0; i < 6; i++) {
            ivs[i] = g_multiVerifyConstants[index].ivs[i][0] + threadIVAdded[i];
        }
        uint64_t baseSeed = getBaseSeed(ivs);
        for (int i = 0; i < 16; i++) {
            uint64_t permutedSeed = baseSeed ^ g_reversalConstants.nullspace[i];
            if (verifySeedNoIVs(permutedSeed, g_reversalConstants, g_multiVerifyConstants[index])) {
                outputBuffer[atomicAdd(numFoundSeeds, 1)] = permutedSeed;
            }
        }
        exit = true;
        for (int i = 0; i < 6; i++) {
            threadIVAdded[i] += 1;
            if (threadIVAdded[i] < threadIVRanges[i]) {
                exit = false;
                break;
            } else {
                threadIVAdded[i] = 0;
            }
        }
    }
}

__device__ bool verifyGroupSeed(uint64_t groupSeed) {
    Xoroshiro rng = {groupSeed};
    bool valid = true;
    for (int k = 0; k < 4; k++) {
        uint64_t genSeed = xoroshiroNext(&rng);
        if (!verifyGeneratorSeed(genSeed, g_multiVerifyConstants[k])) {
            valid = false;
            break;
        }
        xoroshiroNext(&rng);
    }
    return valid;
}

__device__ bool verifyGroupSeedSmart(uint64_t groupSeed) {
    Xoroshiro rng = {groupSeed};
    xoroshiroNext(&rng);
    xoroshiroNext(&rng);
    for (int k = 1; k < 4; k++) {
        uint64_t genSeed = xoroshiroNext(&rng);
        bool found = false;
        for (int j = 1; j < 4; j++) {
            if (verifyGeneratorSeed(genSeed, g_multiVerifyConstants[j])) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
        xoroshiroNext(&rng);
    }
    return true;
}

__global__ void kernel_reversePokemonSeeds(uint64_t* allSeeds, uint32_t* numGroupSeeds, uint64_t* outputBuffer) {
    uint32_t index = blockIdx.y;
    uint64_t seed = allSeeds[index];
    uint64_t genOutputBuffer[GENERATOR_OUTPUT_BUFFER_SIZE];
    uint32_t seedCount = findGeneratorSeeds2(seed, genOutputBuffer, g_slices);
    for (int j = 0; j < seedCount; j++) {
        uint64_t groupSeed = genOutputBuffer[j] - XOROSHIRO_CONSTANT;
        if (verifyGroupSeed(groupSeed)) {
            uint32_t bufIndex = atomicAdd(numGroupSeeds, 1);
            outputBuffer[bufIndex] = groupSeed;
        }
    }
}

__global__ void kernel_reversePokemonSeedsSmart(uint64_t* allSeeds, uint32_t* numGroupSeeds, uint64_t* outputBuffer) {
    uint32_t index = blockIdx.x;
    uint64_t seed = allSeeds[index];
    uint64_t genOutputBuffer[GENERATOR_OUTPUT_BUFFER_SIZE];
    uint32_t seedCount = findGeneratorSeeds3(seed, genOutputBuffer, g_slices);
    for (int j = 0; j < seedCount; j++) {
        uint64_t groupSeed = genOutputBuffer[j] - XOROSHIRO_CONSTANT;
        if (verifyGroupSeedSmart(groupSeed)) {
            uint32_t bufIndex = atomicAdd(numGroupSeeds, 1);
            outputBuffer[bufIndex] = groupSeed;
            printf("Found group seed %llx %llu\n", groupSeed, groupSeed);
        }
    }
}

int reversePokemon(const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible pokemonVerify[4], uint64_t* outputBuffer) {
    uint64_t slices[256] = {0};
    for (uint64_t i = 0; i < 256; i++) {
        for (uint64_t j = 0; j < 8; j++) {
            if ((i >> j) & 1) {
                slices[i] |= (1ull << (j * 8));
            }
        }
    }
    cudaMemcpyToSymbol(g_slices, slices, sizeof(slices));
    cudaMemcpyToSymbol(g_reversalConstants, &reversalConst, sizeof(SeedReverseConstantsFlexible));
    cudaMemcpyToSymbol(g_multiVerifyConstants, pokemonVerify, 4 * sizeof(SeedVerifyConstantsFlexible));
    uint32_t foundSeeds = 0;
    uint32_t* deviceFoundSeeds;
    cudaMalloc(&deviceFoundSeeds, sizeof(uint32_t));
    cudaMemcpy(deviceFoundSeeds, &foundSeeds, sizeof(uint32_t), cudaMemcpyHostToDevice);
    uint64_t* deviceOutputBuffer;
    constexpr uint32_t outputBufferSize = OUTPUT_BUFFER_SIZE;
    cudaMalloc(&deviceOutputBuffer, outputBufferSize * sizeof(uint64_t));
    printf("Starting reversal\n");
    dim3 ivBlock(32, 32);
    dim3 ivGrid(1024, 1024, 4);
    kernel_reverseIVSetMulti <<< ivGrid, ivBlock >>> (deviceFoundSeeds, deviceOutputBuffer);
    cudaDeviceSynchronize();
    cudaMemcpy(&foundSeeds, deviceFoundSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Found %u pokemon seeds\n", foundSeeds);
    if (foundSeeds >= outputBufferSize) {
        printf("Potential output buffer overflow!\n");
        foundSeeds = outputBufferSize;
    }

    uint32_t foundGroupSeeds = 0;
    uint32_t* deviceFoundGroupSeeds;
    cudaMalloc(&deviceFoundGroupSeeds, sizeof(uint32_t));
    cudaMemcpy(deviceFoundGroupSeeds, &foundGroupSeeds, sizeof(uint32_t), cudaMemcpyHostToDevice);
    uint64_t* deviceGenOutputBuffer;
    cudaMalloc(&deviceGenOutputBuffer, 128 * sizeof(uint64_t));
    dim3 block(8, 8, 8);
    dim3 grid(32768, foundSeeds);
    kernel_reversePokemonSeeds<<<grid, block>>>(deviceOutputBuffer, deviceFoundGroupSeeds, deviceGenOutputBuffer);
    cudaDeviceSynchronize();
    cudaMemcpy(&foundGroupSeeds, deviceFoundGroupSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Found %u group seeds\n", foundGroupSeeds);
    cudaMemcpy(outputBuffer, deviceGenOutputBuffer, foundGroupSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(deviceOutputBuffer);
    cudaFree(deviceFoundSeeds);
    cudaFree(deviceGenOutputBuffer);
    cudaFree(deviceFoundGroupSeeds);
    return foundGroupSeeds;
}

int reversePokemonFromStartingMon(const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible pokemonVerify[4], int index, uint64_t* outputBuffer) {
    uint64_t slices[256] = {0};
    for (uint64_t i = 0; i < 256; i++) {
        for (uint64_t j = 0; j < 8; j++) {
            if ((i >> j) & 1) {
                slices[i] |= (1ull << (j * 8));
            }
        }
    }
    cudaMemcpyToSymbol(g_slices, slices, sizeof(slices));
    cudaMemcpyToSymbol(g_reversalConstants, &reversalConst, sizeof(SeedReverseConstantsFlexible));
    cudaMemcpyToSymbol(g_multiVerifyConstants, pokemonVerify, 4 * sizeof(SeedVerifyConstantsFlexible));
    uint32_t* deviceFoundSeeds;
    cudaMalloc(&deviceFoundSeeds, sizeof(uint32_t));
    cudaMemset(deviceFoundSeeds, 0, sizeof(uint32_t));
    uint64_t* deviceOutputBuffer;
    constexpr uint32_t outputBufferSize = OUTPUT_BUFFER_SIZE;
    cudaMalloc(&deviceOutputBuffer, outputBufferSize * sizeof(uint64_t));
    printf("Starting reversal\n");
    dim3 ivBlock(32, 32);
    dim3 ivGrid(1024, 1024);
    kernel_reverseIVSetOffset <<< ivGrid, ivBlock >>> (deviceFoundSeeds, index, deviceOutputBuffer);
    cudaDeviceSynchronize();
    uint32_t foundSeeds = 0;
    cudaMemcpy(&foundSeeds, deviceFoundSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Found %u pokemon seeds\n", foundSeeds);
    if (foundSeeds >= outputBufferSize) {
        printf("Potential output buffer overflow!\n");
        foundSeeds = outputBufferSize;
    }

    uint32_t* deviceFoundGroupSeeds;
    cudaMalloc(&deviceFoundGroupSeeds, sizeof(uint32_t));
    cudaMemset(deviceFoundGroupSeeds, 0, sizeof(uint32_t));
    uint64_t* deviceGenOutputBuffer;
    cudaMalloc(&deviceGenOutputBuffer, 10 * 1024 * 1024 * sizeof(uint64_t));
    dim3 block(8, 8, 8);
    dim3 grid(foundSeeds, 1);
    kernel_reversePokemonSeedsSmart<<<grid, block>>>(deviceOutputBuffer, deviceFoundGroupSeeds, deviceGenOutputBuffer);
    cudaDeviceSynchronize();
    uint32_t foundGroupSeeds = 0;
    cudaMemcpy(&foundGroupSeeds, deviceFoundGroupSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Found %u group seeds\n", foundGroupSeeds);
    cudaMemcpy(outputBuffer, deviceGenOutputBuffer, foundGroupSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(deviceOutputBuffer);
    cudaFree(deviceFoundSeeds);
    cudaFree(deviceGenOutputBuffer);
    cudaFree(deviceFoundGroupSeeds);
    return foundGroupSeeds;
}