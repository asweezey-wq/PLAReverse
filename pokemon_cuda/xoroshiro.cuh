#pragma once
constexpr uint64_t XOROSHIRO_CONSTANT = 0x82a2b175229d6a5b;

struct Xoroshiro {
    uint64_t seed0;
    uint64_t seed1 = XOROSHIRO_CONSTANT;
};

__forceinline__ __device__ uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__forceinline__ __device__ uint64_t xoroshiroNext(Xoroshiro* rng) {
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

__forceinline__ __device__ uint64_t xoroshiroRand(Xoroshiro* rng, uint64_t max, uint64_t mask) {
    while (true) {
        uint64_t number = xoroshiroNext(rng);
        uint64_t maskedNumber = number & mask;
        if (maskedNumber < max) {
            return maskedNumber;
        }
    }
}