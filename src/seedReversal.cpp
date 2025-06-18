#include "seedReversal.hpp"
#include "pokemonGenerator.hpp"
#include "xoroshiro.hpp"

GF2Matrix computeIVMatrix(int numShinyRolls) {
    constexpr int numIVs = 6;
    constexpr int numBitsPerIV = 5;
    GF2Matrix matrix = gf2Init(64, numIVs * numBitsPerIV * 2);
    for (int i = 0; i < 64; i++) {
        uint64_t seed = 1ull << i;
        Xoroshiro128PlusRNG rng(seed, 0);
        rng.next();
        rng.next();
        for (int i = 0; i < numShinyRolls; i++) {
            rng.next();
        }
        for (int j = 0; j < numIVs; j++) {
            for (int k = 0; k < numBitsPerIV; k++) {
                matrix[i][j * (numBitsPerIV * 2) + k] = (rng.getSeed0() >> k) & 1;
                matrix[i][j * (numBitsPerIV * 2) + k + numBitsPerIV] = (rng.getSeed1() >> k) & 1;
            }
            rng.next();
        }
    }
    return matrix;
}

GF2Vector computeIVConst(int numShinyRolls) {
    constexpr int numIVs = 6;
    constexpr int numBitsPerIV = 5;
    GF2Vector values(numIVs * numBitsPerIV * 2);
    Xoroshiro128PlusRNG rng(0);
    rng.next();
    rng.next();
    for (int i = 0; i < numShinyRolls; i++) {
        rng.next();
    }
    for (int j = 0; j < numIVs; j++) {
        for (int k = 0; k < numBitsPerIV; k++) {
            values[j * (numBitsPerIV * 2) + k] = (rng.getSeed0() >> k) & 1;
            values[j * (numBitsPerIV * 2) + k + numBitsPerIV] = (rng.getSeed1() >> k) & 1;
        }
        rng.next();
    }
    return values;
}

SeedReverseConstantsFlexible createReversalStructForCUDA(int numShinyRolls, uint8_t ivs[6]) {
    GF2Matrix ivMat = computeIVMatrix(32);
    uint64_t ivConst = packVector(computeIVConst(32));
    GF2Matrix invIvMat = gf2Inverse(ivMat);
    GF2Matrix nullspace = gf2NullSpace(ivMat);
    std::vector<uint64_t> packedInvIvMat = gf2Pack(invIvMat);
    std::vector<uint64_t> packedNullspace = gf2Pack(nullspace);
    SeedReverseConstantsFlexible reversal = {
        .shinyRolls = 32,
        .ivConst = ivConst
    };
    memcpy(reversal.seedVector, packedInvIvMat.data(), 60 * sizeof(uint64_t));
    memcpy(reversal.nullspace, packedNullspace.data(), 16 * sizeof(uint64_t));
    uint8_t ivRanges[6][2];
    for (int i = 0; i < 6; i++) {
        ivRanges[i][0] = ivs[i];
        ivRanges[i][1] = ivs[i] + 1;
    }
    memcpy(reversal.ivs, ivRanges, sizeof(ivRanges));
    return reversal;
}

SeedVerifyConstantsFlexible createVerifyStructForCUDA(const PokemonEntity& entity) {
    // TODO gender ratio
    SeedVerifyConstantsFlexible verify = {
        .ability = {(uint64_t)entity.m_ability, (uint64_t)entity.m_ability + 1},
        .genderData = {128, (uint8_t)entity.m_gender},
        .nature = entity.m_nature,
        .height = {(uint64_t)entity.m_height, (uint64_t)entity.m_height + 1},
        .weight = {(uint64_t)entity.m_weight, (uint64_t)entity.m_weight + 1}
    };
    return verify;
}

bool verifySeed(uint64_t seed, int numShinyRolls, const PokemonEntity& entity) {
    PokemonEntity testEntity;
    generatePokemon(seed, numShinyRolls, testEntity);
    for (int i = 0; i < 6; i++) {
        if (entity.m_ivs[i] != testEntity.m_ivs[i]) {
            return false;
        }
    }
    if (entity.m_gender != testEntity.m_gender) {
        return false;
    }
    if (entity.m_ability != testEntity.m_ability) {
        return false;
    }
    if (entity.m_nature != testEntity.m_nature) {
        return false;
    }
    if (entity.m_height != testEntity.m_height) {
        return false;
    }
    if (entity.m_weight != testEntity.m_weight) {
        return false;
    }
    return true;
}