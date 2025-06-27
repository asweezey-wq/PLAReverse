#pragma once
#include "pokemonCuda.h"
#include "xoroshiro.cuh"

__device__ __forceinline__ bool verifySeed(uint64_t seed, const PokemonVerificationContext& verifCtx) {
    Xoroshiro rng = {seed, XOROSHIRO_CONSTANT};
    for (int i = 0; i < 2 + g_seedReversalCtx.shinyRolls; i++) {
        xoroshiroNext(&rng);
    }
    for (int i = 0; i < 6; i++) {
        uint8_t ivValue = xoroshiroNext(&rng) & 31;
        if (ivValue < verifCtx.ivs[i][0] || ivValue > verifCtx.ivs[i][1]) {
            return false;
        }
    }
    uint64_t ability = xoroshiroRand(&rng, 2, 1);
    if (ability < verifCtx.ability[0] || ability > verifCtx.ability[1]) {
        return false;
    }
    uint8_t genderRatio = verifCtx.genderData[0];
    if (genderRatio != 255) {
        uint8_t targetGender = verifCtx.genderData[1];
        uint64_t genderRng = xoroshiroRand(&rng, 253, 255) + 1;
        uint8_t gender = genderRng < genderRatio;
        if (gender != targetGender) {
            return false;
        }
    }
    uint8_t nature = xoroshiroRand(&rng, 25, 31);
    if (nature != verifCtx.nature) {
        return false;
    }

    uint64_t height = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (height < verifCtx.height[0] || height > verifCtx.height[1]) {
        return false;
    }
    uint64_t weight = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (weight < verifCtx.weight[0] || weight > verifCtx.weight[1]) {
        return false;
    }
    return true;
}

__device__ __forceinline__ bool verifySeedNoIVs(uint64_t seed, const PokemonVerificationContext& verifCtx) {
    Xoroshiro rng = {seed};
    for (int i = 0; i < 8 + g_seedReversalCtx.shinyRolls; i++) {
        xoroshiroNext(&rng);
    }
    uint64_t ability = xoroshiroRand(&rng, 2, 1);
    if (ability < verifCtx.ability[0] || ability > verifCtx.ability[1]) {
        return false;
    }
    uint8_t genderRatio = verifCtx.genderData[0];
    if (genderRatio != 255) {
        uint8_t targetGender = verifCtx.genderData[1];
        uint64_t genderRng = xoroshiroRand(&rng, 253, 255) + 1;
        uint8_t gender = genderRng < genderRatio;
        if (gender != targetGender) {
            return false;
        }
    }
    uint8_t nature = xoroshiroRand(&rng, 25, 31);
    if (nature != verifCtx.nature) {
        return false;
    }

    uint64_t height = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (height < verifCtx.height[0] || height > verifCtx.height[1]) {
        return false;
    }
    uint64_t weight = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (weight < verifCtx.weight[0] || weight > verifCtx.weight[1]) {
        return false;
    }
    return true;
}

__device__ __forceinline__ bool verifyGeneratorSeed(uint64_t seed, const PokemonVerificationContext& verifCtx) {
    Xoroshiro rng = {seed};
    xoroshiroNext(&rng);
    uint64_t pokeSeed = xoroshiroNext(&rng);
    bool valid = verifySeed(pokeSeed, verifCtx);
    return valid;
}

__device__ __forceinline__ bool verifyGeneratorSlotSeed(uint64_t seed, const PokemonVerificationContext& verifCtx) {
    Xoroshiro rng = {seed};
    const float inv_64_f = 5.421e-20f;
    uint64_t nextValue = xoroshiroNext(&rng);
    float slotRate = ((float)g_seedReversalCtx.generatorSlotSum * (nextValue * inv_64_f));
    if (slotRate < verifCtx.slotThresholds[0] || slotRate > verifCtx.slotThresholds[1]) {
        return false;
    }
    if (verifCtx.levelRange[1] != 0) {
        uint64_t pokeSeed = xoroshiroNext(&rng);
        uint64_t level = verifCtx.levelRange[0] + xoroshiroRand(&rng, verifCtx.levelRange[1], verifCtx.levelRange[2]);
        return level == verifCtx.level;
    }
    return true;
}

__device__ __forceinline__ bool verifyGroupSeed(uint64_t groupSeed) {
    Xoroshiro rng = {groupSeed};
    for (int k = 0; k < 4; k++) {
        uint64_t genSeed = xoroshiroNext(&rng);
        bool found = false;
        for (int j = 0; j < 4; j++) {
            if (verifyGeneratorSeed(genSeed, g_seedReversalCtx.pokemonVerifCtx[j])) {
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