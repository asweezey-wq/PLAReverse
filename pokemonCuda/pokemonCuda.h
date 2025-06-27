#pragma once
#include <cstdint>

// Shared header for both C++ and CUDA

struct PokemonSeedReversalContext {
    uint64_t inverseMatrix[60];
    uint64_t nullspace[16];
    uint64_t xoroshiroBits;
};

struct PokemonVerificationContext {
    uint32_t speciesId;
    uint8_t level;
    float slotThresholds[2];
    // base level, range, mask
    uint8_t levelRange[3];
    uint8_t ivs[6][2];
    uint8_t ability[2];
    uint32_t genderData[2];
    uint8_t nature;
    uint32_t height[2];
    uint32_t weight[2];
};

struct GroupSeedReversalContext {
    uint64_t inverseMatrix[3][64];
    uint32_t shiftConst[3];
    uint64_t xoroshiroBits[3];
};

struct SeedReversalContext {
    uint8_t shinyRolls;
    PokemonSeedReversalContext pokemonReversalCtx;
    uint8_t numPokemon;
    PokemonVerificationContext pokemonVerifCtx[4];
    uint32_t generatorSlotSum;
    GroupSeedReversalContext groupReversalCtx;
};

int reversePokemonFromSingleMon(const SeedReversalContext& reversalCtx, int index, uint64_t* outputBuffer);
