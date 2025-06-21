#pragma once
#include "xoroshiro.cuh"

constexpr uint64_t XOROSHIRO_ROTL_24 = 0x75229d6a5b82a2b1;
#define GENERATOR_OUTPUT_BUFFER_SIZE 4

__device__ __forceinline__ uint64_t x0_from_x1(uint64_t x1) {
    x1 = rotl(x1, 27);
    return XOROSHIRO_ROTL_24 ^ x1 ^ (x1 << 16) ^ rotl(x1, 24);
}

__device__ __forceinline__ uint64_t seed_from_x1(uint64_t x1) {
    return rotl(x1, 27) ^ XOROSHIRO_CONSTANT;
}

__device__ void findGeneratorSeeds(uint64_t seed, uint32_t* seedCount, uint64_t outputBuffer[GENERATOR_OUTPUT_BUFFER_SIZE], const uint64_t slices[256]);