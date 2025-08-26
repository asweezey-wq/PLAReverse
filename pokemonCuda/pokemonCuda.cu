#include "pokemonCuda.hpp"
#include "deviceConstants.cuh"
#include "pokemonVerif.cuh"
#include "seedReversal.cuh"
#include "xoroshiro.cuh"
#include <iostream>

__constant__ SeedReversalContext g_seedReversalCtx;
__constant__ uint64_t g_slices[256];

__global__ void kernel_reversePokemonSeeds(uint32_t index, uint32_t* numFoundSeeds, uint64_t* outputBuffer) {
    const PokemonVerificationContext& verifCtx = g_seedReversalCtx.pokemonVerifCtx[index];
    uint8_t threadIVAdded[6] = {0};
    uint8_t threadIVRanges[6] = {0};
    for (int i = 0; i < 6; i++) {
        threadIVRanges[i] = 1 + verifCtx.ivs[i][1] - verifCtx.ivs[i][0];
    }
    uint8_t ivs[6];
    bool exit = false;
    while (!exit) {
        for (int i = 0; i < 6; i++) {
            ivs[i] = verifCtx.ivs[i][0] + threadIVAdded[i];
        }
        uint64_t baseSeed = getBaseSeed(ivs);
        for (int i = 0; i < 16; i++) {
            uint64_t permutedSeed = baseSeed ^ g_seedReversalCtx.pokemonReversalCtx.nullspace[i];
            if (permutedSeed == 0x766ed5dd282c4ca3) {
                printf("Found true seed verify:%d\n", verifySeedNoIVs(permutedSeed, verifCtx, g_seedReversalCtx.primarySizePairs));
            }
            if (verifySeedNoIVs(permutedSeed, verifCtx, g_seedReversalCtx.primarySizePairs)) {
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

__global__ void kernel_reverseGeneratorSeeds(uint64_t* pokemonSeeds, uint32_t pokemonIndex, uint32_t* numGenSeeds, uint64_t* outputBuffer) {
    uint32_t index = blockIdx.x;
    uint64_t seed = pokemonSeeds[index];
    uint64_t tmpBuffer[4];
    uint32_t foundSeeds = findGeneratorSeeds(seed, tmpBuffer, g_slices);
    for (int i = 0; i < foundSeeds; i++) {
        if (verifyGeneratorSlotSeed(tmpBuffer[i], g_seedReversalCtx.pokemonVerifCtx[pokemonIndex])) {
            uint32_t index = atomicAdd(numGenSeeds, 1);
            outputBuffer[index] = tmpBuffer[i];
        }
    }
}

__global__ void kernel_reverseGroupSeedsFirstPokemon(uint64_t* genSeeds, uint32_t* numGroupSeeds, uint64_t* outputBuffer) {
    uint32_t index = (blockIdx.x << 8) | threadIdx.x;
    uint64_t seed = genSeeds[index];
    uint64_t groupSeed = seed - XOROSHIRO_CONSTANT;
    if (verifyGroupSeed(groupSeed)) {
        printf("Found group seed! Index:0 Seed: 0x%llx %llu\n", groupSeed, groupSeed);
        outputBuffer[atomicAdd(numGroupSeeds, 1)] = groupSeed;
    }
}

__global__ void kernel_reverseGroupSeeds(uint64_t* genSeeds, uint32_t groupIndex, uint32_t* numGroupSeeds, uint64_t* outputBuffer) {
    uint32_t index = blockIdx.y;
    uint64_t seed = genSeeds[index];
    findGroupSeeds(seed, groupIndex, numGroupSeeds, outputBuffer);
}

int reversePokemonFromSingleMon(const SeedReversalContext& reversalCtx, int index, uint64_t* outputBuffer) {
    uint64_t slices[256] = {0};
    for (uint64_t i = 0; i < 256; i++) {
        for (uint64_t j = 0; j < 8; j++) {
            if ((i >> j) & 1) {
                slices[i] |= (1ull << (j * 8));
            }
        }
    }
    cudaMemcpyToSymbol(g_slices, slices, sizeof(slices));
    cudaMemcpyToSymbol(g_seedReversalCtx, &reversalCtx, sizeof(SeedReversalContext));

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t* device_numPokemonSeeds;
    cudaMalloc(&device_numPokemonSeeds, sizeof(uint32_t));
    cudaMemset(device_numPokemonSeeds, 0, sizeof(uint32_t));
    uint64_t* device_pokemonSeedsBuf;
    constexpr uint32_t outputBufferSize = 1024 * 1024;
    cudaMalloc(&device_pokemonSeedsBuf, outputBufferSize * sizeof(uint64_t));
    printf("Starting Pokemon reversal\n");
    dim3 ivBlock(32, 32);
    dim3 ivGrid(1024, 1024);
    cudaEventRecord(start, 0);
    kernel_reversePokemonSeeds <<< ivGrid, ivBlock >>> ((uint32_t)index, device_numPokemonSeeds, device_pokemonSeedsBuf);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    uint32_t numPokemonSeeds = 0;
    cudaMemcpy(&numPokemonSeeds, device_numPokemonSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Found %u pokemon seeds (took %.1f s)\n", numPokemonSeeds, time / 1000.0f);
    if (numPokemonSeeds >= outputBufferSize) {
        printf("Potential output buffer overflow!\n");
        numPokemonSeeds = outputBufferSize;
    }

    uint32_t* device_numGenSeeds;
    cudaMalloc(&device_numGenSeeds, sizeof(uint32_t));
    cudaMemset(device_numGenSeeds, 0, sizeof(uint32_t));
    uint64_t* device_genSeedsBuf;
    uint32_t genOutputBufferSize = 2 * numPokemonSeeds;
    cudaMalloc(&device_genSeedsBuf, genOutputBufferSize * sizeof(uint64_t));
    dim3 genBlock(8,8,8);
    dim3 genGrid(numPokemonSeeds, 1);
    cudaEventRecord(start, 0);
    kernel_reverseGeneratorSeeds<<<genGrid, genBlock>>>(device_pokemonSeedsBuf, index, device_numGenSeeds, device_genSeedsBuf);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    uint32_t numGenSeeds = 0;
    cudaMemcpy(&numGenSeeds, device_numGenSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Found %u generator seeds (took %.1f s)\n", numGenSeeds, time / 1000.0f);
    if (numGenSeeds >= genOutputBufferSize) {
        printf("Potential output buffer overflow!\n");
        numGenSeeds = genOutputBufferSize;
    }
    uint32_t numGroupSeeds = 0;
    if (numGenSeeds) {
        uint32_t* device_numGroupSeeds;
        cudaMalloc(&device_numGroupSeeds, sizeof(uint32_t));
        cudaMemset(device_numGroupSeeds, 0, sizeof(uint32_t));
        uint64_t* deivce_groupSeedsBuf;
        constexpr uint32_t groupOutputBufferSize = 4;
        cudaMalloc(&deivce_groupSeedsBuf, groupOutputBufferSize * sizeof(uint64_t));
        uint32_t numThreads = 256;
        uint32_t roundedNumSeeds = (uint32_t)(numThreads * std::ceil((float)numGenSeeds / numThreads));
        uint32_t numBlocks = roundedNumSeeds / numThreads;
        printf("Checking group index 0\n");
        kernel_reverseGroupSeedsFirstPokemon<<<numBlocks, numThreads>>>(device_genSeedsBuf, device_numGroupSeeds, deivce_groupSeedsBuf);
        cudaDeviceSynchronize();
        cudaMemcpy(&numGroupSeeds, device_numGroupSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (!numGroupSeeds) {
            if (numGenSeeds > 65535) {
                printf("Too many group seeds!\n");
                exit(1);
            }
            for (int i = 0; i < 3; i++) {
                bool needsCarry = reversalCtx.groupReversalCtx.shiftConst[i] != 0;
                dim3 groupBlock(256, needsCarry ? 2 : 1);
                dim3 groupGrid(1 << 24, numGenSeeds);
                printf("Checking group index %d (estimated %0.0f s)\n", i+1, 0.2f * numGenSeeds * (needsCarry ? 2 : 1));
                cudaEventRecord(start, 0);
                kernel_reverseGroupSeeds<<<groupGrid, groupBlock>>>(device_genSeedsBuf, i, device_numGroupSeeds, deivce_groupSeedsBuf);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                // About 0.4s per generator seed
                printf("  Took %.1f s\n", time / 1000.0f);
                cudaMemcpy(&numGroupSeeds, device_numGroupSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                if (numGroupSeeds) {
                    break;
                }
            }
        }
        printf("Found %u group seeds!\n", numGroupSeeds);
        cudaMemcpy(outputBuffer, deivce_groupSeedsBuf, numGroupSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(deivce_groupSeedsBuf);
        cudaFree(device_numGroupSeeds);
    }
    cudaFree(device_pokemonSeedsBuf);
    cudaFree(device_numPokemonSeeds);
    cudaFree(device_genSeedsBuf);
    cudaFree(device_numGenSeeds);

    return numGroupSeeds;
}

HostReversalManager::HostReversalManager(const SeedReversalContext& reversalCtx, int index, uint64_t* outputBuffer)
    : m_reversalCtx(reversalCtx), m_index(index), m_outputBuffer(outputBuffer) {
    
}

void HostReversalManager::setupReversal() {
    uint64_t slices[256] = {0};
    for (uint64_t i = 0; i < 256; i++) {
        for (uint64_t j = 0; j < 8; j++) {
            if ((i >> j) & 1) {
                slices[i] |= (1ull << (j * 8));
            }
        }
    }
    cudaMemcpyToSymbol(g_slices, slices, sizeof(slices));
    cudaMemcpyToSymbol(g_seedReversalCtx, &m_reversalCtx, sizeof(SeedReversalContext));

    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);

    constexpr uint32_t groupOutputBufferSize = 4;
    cudaMalloc(&m_deviceGroupSeedsBuf, groupOutputBufferSize * sizeof(uint64_t));

    m_numPokemonSeeds = 0;
    m_numGenSeeds = 0;
    m_numGroupSeeds = 0;
}

uint32_t HostReversalManager::reversePokemonSeeds() {
    uint32_t* device_numPokemonSeeds;
    cudaMalloc(&device_numPokemonSeeds, sizeof(uint32_t));
    cudaMemset(device_numPokemonSeeds, 0, sizeof(uint32_t));
    constexpr uint32_t outputBufferSize = 1024 * 1024;
    cudaMalloc(&m_devicePokemonSeedsBuf, outputBufferSize * sizeof(uint64_t));
    printf("Starting Pokemon reversal\n");
    dim3 ivBlock(32, 32);
    dim3 ivGrid(1024, 1024);
    cudaEventRecord(m_start, 0);
    kernel_reversePokemonSeeds <<< ivGrid, ivBlock >>> ((uint32_t)m_index, device_numPokemonSeeds, m_devicePokemonSeedsBuf);
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
    float time;
    cudaEventElapsedTime(&time, m_start, m_stop);
    cudaMemcpy(&m_numPokemonSeeds, device_numPokemonSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(device_numPokemonSeeds);
    printf("Found %u pokemon seeds (took %.1f s)\n", m_numPokemonSeeds, time / 1000.0f);
    if (m_numPokemonSeeds >= outputBufferSize) {
        printf("Potential output buffer overflow!\n");
        m_numPokemonSeeds = outputBufferSize;
    }
    return m_numPokemonSeeds;
}

uint32_t HostReversalManager::reverseGeneratorSeeds() {
    uint32_t* device_numGenSeeds;
    cudaMalloc(&device_numGenSeeds, sizeof(uint32_t));
    cudaMemset(device_numGenSeeds, 0, sizeof(uint32_t));
    uint32_t genOutputBufferSize = 2 * m_numPokemonSeeds;
    cudaMalloc(&m_deviceGenSeedsBuf, genOutputBufferSize * sizeof(uint64_t));
    dim3 genBlock(8,8,8);
    dim3 genGrid(m_numPokemonSeeds, 1);
    cudaEventRecord(m_start, 0);
    kernel_reverseGeneratorSeeds<<<genGrid, genBlock>>>(m_devicePokemonSeedsBuf, m_index, device_numGenSeeds, m_deviceGenSeedsBuf);
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
    float time;
    cudaEventElapsedTime(&time, m_start, m_stop);
    cudaMemcpy(&m_numGenSeeds, device_numGenSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(device_numGenSeeds);
    printf("Found %u generator seeds (took %.1f s)\n", m_numGenSeeds, time / 1000.0f);
    if (m_numGenSeeds >= genOutputBufferSize) {
        printf("Potential output buffer overflow!\n");
        m_numGenSeeds = genOutputBufferSize;
    }
    return m_numGenSeeds;
}

bool HostReversalManager::reverseGroupSeedsFirstIndex() {
    uint32_t* device_numGroupSeeds;
    cudaMalloc(&device_numGroupSeeds, sizeof(uint32_t));
    cudaMemset(device_numGroupSeeds, 0, sizeof(uint32_t));
    uint32_t numThreads = 256;
    uint32_t roundedNumSeeds = (uint32_t)(numThreads * std::ceil((float)m_numGenSeeds / numThreads));
    uint32_t numBlocks = roundedNumSeeds / numThreads;
    printf("Checking group index 0\n");
    kernel_reverseGroupSeedsFirstPokemon<<<numBlocks, numThreads>>>(m_deviceGenSeedsBuf, device_numGroupSeeds, m_deviceGroupSeedsBuf);
    cudaDeviceSynchronize();
    cudaMemcpy(&m_numGroupSeeds, device_numGroupSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(device_numGroupSeeds);
    return m_numGroupSeeds != 0;
}

bool HostReversalManager::reverseGroupSeeds(int index) {
    if (m_numGenSeeds > 65535) {
        return false;
    }
    uint32_t* device_numGroupSeeds;
    cudaMalloc(&device_numGroupSeeds, sizeof(uint32_t));
    cudaMemset(device_numGroupSeeds, 0, sizeof(uint32_t));
    bool needsCarry = m_reversalCtx.groupReversalCtx.shiftConst[index] != 0;
    dim3 groupBlock(256, needsCarry ? 2 : 1);
    dim3 groupGrid(1 << 24, m_numGenSeeds);
    float estimatedTime = 0.2f * m_numGenSeeds * (needsCarry ? 2 : 1);
    printf("Checking group index %d (estimated %.1f min)\n", index+1, estimatedTime / 60);
    cudaEventRecord(m_start, 0);
    kernel_reverseGroupSeeds<<<groupGrid, groupBlock>>>(m_deviceGenSeedsBuf, index, device_numGroupSeeds, m_deviceGroupSeedsBuf);
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
    float time;
    cudaEventElapsedTime(&time, m_start, m_stop);
    // About 0.4s per generator seed
    printf("  Took %.1f s\n", time / 1000.0f);
    cudaMemcpy(&m_numGroupSeeds, device_numGroupSeeds, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(device_numGroupSeeds);
    return m_numGroupSeeds != 0;
}

uint32_t HostReversalManager::finishReversal() {
    printf("Found %u group seeds!\n", m_numGroupSeeds);
    if (m_numGroupSeeds) {
        cudaMemcpy(m_outputBuffer, m_deviceGroupSeedsBuf, m_numGroupSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    cudaFree(m_devicePokemonSeedsBuf);
    cudaFree(m_deviceGenSeedsBuf);
    cudaFree(m_deviceGroupSeedsBuf);

    return m_numGroupSeeds;
}
