#include "pokemon.cuh"
#include "xoroshiro.cuh"
#include <iostream>

__device__ bool verifySeed(uint64_t seed, const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible& verifyConst) {
    Xoroshiro rng = {seed, XOROSHIRO_CONSTANT};
    for (int i = 0; i < 2 + reversalConst.shinyRolls; i++) {
        xoroshiroNext(&rng);
    }
    for (int i = 0; i < 6; i++) {
        uint8_t ivValue = xoroshiroNext(&rng) & 31;
        if (ivValue < verifyConst.ivs[i][0] || ivValue > verifyConst.ivs[i][1]) {
            return false;
        }
    }
    uint64_t ability = xoroshiroRand(&rng, 2, 1);
    if (ability < verifyConst.ability[0] || ability > verifyConst.ability[1]) {
        return false;
    }
    uint8_t genderRatio = verifyConst.genderData[0];
    uint8_t targetGender = verifyConst.genderData[1];
    uint64_t genderRng = xoroshiroRand(&rng, 253, 255);
    uint8_t gender = genderRng < genderRatio;
    if (gender != targetGender) {
        return false;
    }
    uint8_t nature = xoroshiroRand(&rng, 25, 31);
    if (nature != verifyConst.nature) {
        return false;
    }

    uint64_t height = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (height < verifyConst.height[0] || height > verifyConst.height[1]) {
        return false;
    }
    uint64_t weight = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (weight < verifyConst.weight[0] || weight > verifyConst.weight[1]) {
        return false;
    }
    return true;
}

__device__ bool verifySeedNoIVs(uint64_t seed, const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible& verifyConst) {
    Xoroshiro rng = {seed};
    for (int i = 0; i < 8 + reversalConst.shinyRolls; i++) {
        xoroshiroNext(&rng);
    }
    uint64_t ability = xoroshiroRand(&rng, 2, 1);
    if (ability < verifyConst.ability[0] || ability > verifyConst.ability[1]) {
        return false;
    }
    uint8_t genderRatio = verifyConst.genderData[0];
    uint8_t targetGender = verifyConst.genderData[1];
    uint64_t genderRng = xoroshiroRand(&rng, 253, 255);
    uint8_t gender = genderRng < genderRatio;
    if (gender != targetGender) {
        return false;
    }
    uint8_t nature = xoroshiroRand(&rng, 25, 31);
    if (nature != verifyConst.nature) {
        return false;
    }

    uint64_t height = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (height < verifyConst.height[0] || height > verifyConst.height[1]) {
        return false;
    }
    uint64_t weight = xoroshiroRand(&rng, 129, 255) + xoroshiroRand(&rng, 128, 127);
    if (weight < verifyConst.weight[0] || weight > verifyConst.weight[1]) {
        return false;
    }
    return true;
}