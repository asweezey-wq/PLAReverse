#pragma once
#include <cstdint>

#define OUTPUT_BUFFER_SIZE 400000000

struct SeedReverseConstantsFlexible {
    uint8_t shinyRolls;
    uint64_t seedVector[60];
    uint64_t nullspace[16];
    uint64_t ivConst;
};

struct SeedVerifyConstantsFlexible {
    uint8_t ivs[6][2];
    uint8_t ability[2];
    uint32_t genderData[2];
    uint8_t nature;
    uint32_t height[2];
    uint32_t weight[2];
};

int ivWrapper(const SeedReverseConstantsFlexible reverseConst, const SeedVerifyConstantsFlexible verifyConst, uint64_t* outputBuffer);

int genSeedWrapper(uint64_t seed, uint64_t* outputBuffer);
int reversePokemon(const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible pokemonVerify[4], uint64_t* outputBuffer);

int reversePokemonFromStartingMon(const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible pokemonVerify[4], int index, uint64_t* outputBuffer);