#include <math.h>
#include <iostream>
#include "pokemon_cuda.h"

__constant__ SeedReverseConstantsFlexible g_reversalConstants;
__constant__ SeedVerifyConstantsFlexible g_verifyConstants;

__device__ int seedCount = 0;

struct Xoroshiro {
    uint64_t seed0;
    uint64_t seed1 = 0x82a2b175229d6a5b;
};

__device__
uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__device__
uint64_t xoroshiroNext(Xoroshiro* rng) {
    uint64_t s0 = rng->seed0;
    uint64_t s1 = rng->seed1;
    uint64_t result = s0 + s1;
    s1 ^= s0;
    s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s1 = rotl(s1, 37);
    rng->seed0 = s0;
    rng->seed1 = s1;
    return result;
}

__device__
uint64_t xoroshiroRand(Xoroshiro* rng, uint64_t max, uint64_t mask) {
    while (true) {
        uint64_t number = xoroshiroNext(rng);
        uint64_t maskedNumber = number & mask;
        if (maskedNumber < max) {
            return maskedNumber;
        }
    }
}

__device__
bool verifySeed(uint64_t seed) {
    Xoroshiro rng = {seed};
    for (int i = 0; i < 8 + g_reversalConstants.shinyRolls; i++) {
        xoroshiroNext(&rng);
    }
    uint64_t ability = xoroshiroRand(&rng, 2, 1);
    if (ability != g_verifyConstants.ability[0]) {
        return false;
    }
    uint8_t genderRatio = g_verifyConstants.genderData[0];
    uint8_t targetGender = g_verifyConstants.genderData[1];
    uint64_t genderRng = xoroshiroRand(&rng, 253, 255) + 1;
    uint8_t gender = genderRng < genderRatio;
    if (gender != targetGender) {
        return false;
    }
    uint8_t nature = xoroshiroRand(&rng, 25, 31);
    if (nature != g_verifyConstants.nature) {
        return false;
    }

    uint64_t height = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (height != g_verifyConstants.height[0]) {
        return false;
    }
    uint64_t weight = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (weight != g_verifyConstants.weight[0]) {
        return false;
    }
    return true;
}

__device__
bool verifySeedFlexible(uint64_t seed) {
    Xoroshiro rng = {seed};
    for (int i = 0; i < 8 + g_reversalConstants.shinyRolls; i++) {
        xoroshiroNext(&rng);
    }
    uint64_t ability = xoroshiroRand(&rng, 2, 1);
    if (ability < g_verifyConstants.ability[0] || ability >= g_verifyConstants.ability[1]) {
        return false;
    }
    uint8_t genderRatio = g_verifyConstants.genderData[0];
    uint8_t targetGender = g_verifyConstants.genderData[1];
    uint64_t genderRng = xoroshiroRand(&rng, 253, 255);
    uint8_t gender = genderRng < genderRatio;
    if (gender != targetGender) {
        return false;
    }
    uint8_t nature = xoroshiroRand(&rng, 25, 31);
    if (nature != g_verifyConstants.nature) {
        return false;
    }

    uint64_t height = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (height < g_verifyConstants.height[0] || height >= g_verifyConstants.height[1]) {
        return false;
    }
    uint64_t weight = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (weight < g_verifyConstants.weight[0] || weight >= g_verifyConstants.weight[1]) {
        return false;
    }
    return true;
}

__device__
uint64_t getBaseSeed(uint8_t ivs[6]) {
    uint8_t threadIVs[6] = {
        (uint8_t)(threadIdx.x & 31),
        (uint8_t)((threadIdx.x >> 5) & 31),
        (uint8_t)((blockIdx.x >> 0) & 31),
        (uint8_t)((blockIdx.x >> 5) & 31),
        (uint8_t)((blockIdx.x >> 10) & 31),
        (uint8_t)((blockIdx.x >> 15) & 31)
    };
    uint64_t packedIvInput = 0;
    for (int i = 0; i < 6; i++) {
        uint8_t sub = ivs[i] - threadIVs[i];
        packedIvInput |= ((uint64_t)(sub & 31) << (10 * i));
        packedIvInput |= ((uint64_t)threadIVs[i] << (10 * i + 5));
    }
    packedIvInput ^= g_reversalConstants.ivConst;
    uint64_t baseSeed = 0;
    for (int i = 0; i < 60; i++) {
        if (packedIvInput & 1) {
            baseSeed ^= g_reversalConstants.seedVector[i];
        }
        packedIvInput >>= 1;
        if (!packedIvInput) {
            break;
        }
    }
    return baseSeed;
}

__global__
void reverseIVSet(uint64_t* outputBuffer) {
    uint8_t ivs[6];
    for (int i = 0; i < 6; i++) {
        ivs[i] = g_reversalConstants.ivs[i][0];
    }
    uint64_t baseSeed = getBaseSeed(ivs);
    for (int i = 0; i < 16; i++) {
        uint64_t permutedSeed = baseSeed ^ g_reversalConstants.nullspace[i];
        if (verifySeed(permutedSeed) && seedCount < OUTPUT_BUFFER_SIZE - 1) {
            outputBuffer[atomicAdd(&seedCount, 1)] = permutedSeed;
        }
    }
}

__global__
void reverseIVSetFlexible(uint64_t* outputBuffer) {
    uint8_t threadIVAdded[6] = {0};
    uint8_t threadIVRanges[6] = {0};
    for (int i = 0; i < 6; i++) {
        threadIVRanges[i] = g_reversalConstants.ivs[i][1] - g_reversalConstants.ivs[i][0];
    }
    uint8_t ivs[6];
    bool exit = false;
    while (!exit && seedCount < OUTPUT_BUFFER_SIZE) {
        for (int i = 0; i < 6; i++) {
            ivs[i] = g_reversalConstants.ivs[i][0] + threadIVAdded[i];
        }
        uint64_t baseSeed = getBaseSeed(ivs);
        for (int i = 0; i < 16; i++) {
            uint64_t permutedSeed = baseSeed ^ g_reversalConstants.nullspace[i];
            if (verifySeedFlexible(permutedSeed) && seedCount < OUTPUT_BUFFER_SIZE) {
                outputBuffer[atomicAdd(&seedCount, 1)] = permutedSeed;
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

int ivWrapper(const SeedReverseConstantsFlexible reverseConst, const SeedVerifyConstantsFlexible verifyConst, uint64_t* outputBuffer) {
    cudaMemcpyToSymbol(g_reversalConstants, &reverseConst, sizeof(SeedReverseConstantsFlexible));
    cudaMemcpyToSymbol(g_verifyConstants, &verifyConst, sizeof(SeedVerifyConstantsFlexible));
    uint64_t* deviceOutputBuffer;
    cudaMalloc(&deviceOutputBuffer, OUTPUT_BUFFER_SIZE * sizeof(uint64_t));
    reverseIVSet <<< 1024 * 1024, 1024 >>> (deviceOutputBuffer);
    cudaDeviceSynchronize();
    int seedsFound = 0;
    cudaMemcpyFromSymbol(&seedsFound, seedCount, sizeof(int));
    printf("Found %d seeds\n", seedsFound);
    if (seedsFound >= OUTPUT_BUFFER_SIZE) {
        printf("Potential output buffer overflow!\n");
        seedsFound = OUTPUT_BUFFER_SIZE;
    }
    cudaMemcpy(outputBuffer, deviceOutputBuffer, seedsFound * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(deviceOutputBuffer);
    return seedsFound;
}

int ivWrapperFlexible(const SeedReverseConstantsFlexible reverseConst, const SeedVerifyConstantsFlexible verifyConst, uint64_t* outputBuffer) {
    cudaMemcpyToSymbol(g_reversalConstants, &reverseConst, sizeof(SeedReverseConstantsFlexible));
    cudaMemcpyToSymbol(g_verifyConstants, &verifyConst, sizeof(SeedVerifyConstantsFlexible));
    uint64_t* deviceOutputBuffer;
    cudaMalloc(&deviceOutputBuffer, OUTPUT_BUFFER_SIZE * sizeof(uint64_t));
    reverseIVSetFlexible <<< 1024 * 1024, 1024 >>> (deviceOutputBuffer);
    cudaDeviceSynchronize();
    int seedsFound = 0;
    cudaMemcpyFromSymbol(&seedsFound, seedCount, sizeof(int));
    printf("Found %d seeds\n", seedsFound);
    if (seedsFound >= OUTPUT_BUFFER_SIZE) {
        printf("Potential output buffer overflow!\n");
        seedsFound = OUTPUT_BUFFER_SIZE;
    }
    cudaMemcpy(outputBuffer, deviceOutputBuffer, seedsFound * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(deviceOutputBuffer);
    return seedsFound;
}