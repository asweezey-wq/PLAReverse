#pragma once
#include <cstdint>
#include <bit>
#include <iostream>

class Xoroshiro128PlusRNG {
public:
    Xoroshiro128PlusRNG(uint64_t s0 = 0x0f4b17a579f18960, uint64_t s1 = 0x82a2b175229d6a5b) : m_seed0(s0), m_seed1(s1) {}

    uint64_t next() {
        const uint64_t s0 = m_seed0;
        uint64_t s1 = m_seed1;
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        m_seed0 = std::rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
        m_seed1 = std::rotl(s1, 37); // c

        return result;
    }

    void long_jump(void) {
        static const uint64_t LONG_JUMP[] = { 0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1 };
    
        uint64_t s0 = 0;
        uint64_t s1 = 0;
        for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
            for(int b = 0; b < 64; b++) {
                if (LONG_JUMP[i] & UINT64_C(1) << b) {
                    s0 ^= m_seed0;
                    s1 ^= m_seed1;
                }
                next();
            }
    
        m_seed0 = s0;
        m_seed1 = s1;
    }

    uint64_t nextWithMax(uint64_t max) {
        uint64_t mask = (1ull << std::bit_width(max - 1)) - 1;
        while (true) {
            uint64_t result = next() & mask;
            if (result < max) {
                return result;
            }
        }
    }

    uint8_t nextBool() {
        return next() & 1;
    }

    uint32_t nextUint32() {
        return next() & UINT32_MAX;
    }

    float nextFloat(float range, float bias = 0) {
        const float inv_64_f = 5.421e-20f;
        uint64_t nextValue = next();
        return (range * (nextValue * inv_64_f)) + bias;
    }

    uint64_t getSeed0() { return m_seed0; }
    uint64_t getSeed1() { return m_seed1; }

private:
    uint64_t m_seed0, m_seed1;
};