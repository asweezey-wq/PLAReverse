#include <iostream>
#include "pokemon.cuh"
#include "xoroshiro.cuh"
#include "ivReversal.cuh"

__constant__ SeedReverseConstantsFlexible g_reversalConstants;
__constant__ SeedVerifyConstantsFlexible g_verifyConstants;

__device__ uint64_t getBaseSeed(uint8_t ivs[6]) {
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

__global__ void kernel_reverseIVSet(uint32_t* numFoundSeeds, uint64_t* outputBuffer) {
    uint8_t threadIVAdded[6] = {0};
    uint8_t threadIVRanges[6] = {0};
    for (int i = 0; i < 6; i++) {
        threadIVRanges[i] = 1 + g_verifyConstants.ivs[i][1] - g_verifyConstants.ivs[i][0];
    }
    uint8_t ivs[6];
    bool exit = false;
    while (!exit && *numFoundSeeds < OUTPUT_BUFFER_SIZE) {
        for (int i = 0; i < 6; i++) {
            ivs[i] = g_verifyConstants.ivs[i][0] + threadIVAdded[i];
        }
        uint64_t baseSeed = getBaseSeed(ivs);
        for (int i = 0; i < 16; i++) {
            uint64_t permutedSeed = baseSeed ^ g_reversalConstants.nullspace[i];
            if (verifySeedNoIVs(permutedSeed, g_reversalConstants, g_verifyConstants) && *numFoundSeeds < OUTPUT_BUFFER_SIZE) {
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

int ivWrapper(const SeedReverseConstantsFlexible reverseConst, const SeedVerifyConstantsFlexible verifyConst, uint64_t* outputBuffer) {
    cudaMemcpyToSymbol(g_reversalConstants, &reverseConst, sizeof(SeedReverseConstantsFlexible));
    cudaMemcpyToSymbol(g_verifyConstants, &verifyConst, sizeof(SeedVerifyConstantsFlexible));
    uint32_t foundSeeds = 0;
    uint32_t* deviceFoundSeeds;
    cudaMalloc(&deviceFoundSeeds, sizeof(uint32_t));
    cudaMemcpy(deviceFoundSeeds, &foundSeeds, sizeof(uint32_t), cudaMemcpyHostToDevice);
    uint64_t* deviceOutputBuffer;
    cudaMalloc(&deviceOutputBuffer, OUTPUT_BUFFER_SIZE * sizeof(uint64_t));
    dim3 ivBlock(32, 32);
    dim3 ivGrid(1024, 1024);
    kernel_reverseIVSet <<< ivGrid, ivBlock >>> (deviceFoundSeeds, deviceOutputBuffer);
    cudaDeviceSynchronize();
    cudaMemcpy(&foundSeeds, deviceFoundSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Found %u seeds\n", foundSeeds);
    if (foundSeeds >= OUTPUT_BUFFER_SIZE) {
        printf("Potential output buffer overflow!\n");
        foundSeeds = OUTPUT_BUFFER_SIZE;
    }
    cudaMemcpy(outputBuffer, deviceOutputBuffer, foundSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(deviceOutputBuffer);
    cudaFree(deviceFoundSeeds);
    return foundSeeds;
}