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
        for (int j = 0; j < numShinyRolls; j++) {
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

GF2Matrix computeSlotMatrix() {
    GF2Matrix matrix = gf2Init(64, 128);
    for (int i = 0; i < 64; i++) {
        uint64_t seed = 1ull << i;
        Xoroshiro128PlusRNG rng(seed, 0);
        rng.next();
        for (int j = 0; j < 64; j++) {
            matrix[i][j] = (rng.getSeed0() >> j) & 1;
            matrix[i][j + 64] = (rng.getSeed1() >> j) & 1;
        }
    }
    return matrix;
}

GF2Vector computeSlotConst() {
    GF2Vector values(128);
    Xoroshiro128PlusRNG rng(0);
    rng.next();
    for (int j = 0; j < 64; j++) {
        values[j] = (rng.getSeed0() >> j) & 1;
        values[j + 64] = (rng.getSeed1() >> j) & 1;
    }
    return values;
}

GF2Matrix computeGroupMatrix(int genIndex) {
    GF2Matrix matrix = gf2Init(64, 128);
    for (int i = 0; i < 64; i++) {
        uint64_t seed = 1ull << i;
        Xoroshiro128PlusRNG rng(seed, 0);
        for (int j = 0; j < genIndex; j++) {
            rng.next();
            rng.next();
        }
        for (int j = 0; j < 64; j++) {
            matrix[i][j] = (rng.getSeed0() >> j) & 1;
            matrix[i][j + 64] = (rng.getSeed1() >> j) & 1;
        }
    }
    return matrix;
}

GF2Vector computeGroupConst(int genIndex) {
    GF2Vector values(128);
    Xoroshiro128PlusRNG rng(0);
    for (int j = 0; j < genIndex; j++) {
        rng.next();
        rng.next();
    }
    for (int j = 0; j < 64; j++) {
        values[j] = (rng.getSeed0() >> j) & 1;
        values[j + 64] = (rng.getSeed1() >> j) & 1;
    }
    return values;
}

SeedReverseConstantsFlexible createReversalStructForCUDA(int numShinyRolls) {
    GF2Matrix ivMat = computeIVMatrix(numShinyRolls);
    uint64_t ivConst = packVector(computeIVConst(numShinyRolls));
    GF2Matrix invIvMat = gf2Inverse(ivMat);
    GF2Matrix nullspace = gf2NullSpace(ivMat);
    std::vector<uint64_t> packedInvIvMat = gf2Pack(invIvMat);
    std::vector<uint64_t> packedNullspace = gf2Pack(nullspace);
    SeedReverseConstantsFlexible reversal = {
        .shinyRolls = (uint8_t)numShinyRolls,
        .ivConst = ivConst
    };
    memcpy(reversal.seedVector, packedInvIvMat.data(), 60 * sizeof(uint64_t));
    memcpy(reversal.nullspace, packedNullspace.data(), 16 * sizeof(uint64_t));
    return reversal;
}

SeedVerifyConstantsFlexible createVerifyStructForCUDA(const PokemonEntity& entity) {
    // TODO gender ratio
    uint8_t ivRanges[6][2];
    for (int i = 0; i < 6; i++) {
        ivRanges[i][0] = entity.m_ivs[i];
        ivRanges[i][1] = entity.m_ivs[i] + 1;
    }
    SeedVerifyConstantsFlexible verify = {
        .ability = {(uint64_t)entity.m_ability, (uint64_t)entity.m_ability + 1},
        .genderData = {128, (uint8_t)entity.m_gender},
        .nature = entity.m_nature,
        .height = {(uint64_t)entity.m_height, (uint64_t)entity.m_height + 1},
        .weight = {(uint64_t)entity.m_weight, (uint64_t)entity.m_weight + 1}
    };
    memcpy(verify.ivs, ivRanges, sizeof(ivRanges));
    return verify;
}

bool verifySeed(uint64_t seed, int numShinyRolls, uint8_t genderRatio, const PokemonEntity& entity) {
    PokemonEntity testEntity;
    generatePokemon(seed, numShinyRolls, genderRatio, testEntity);
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

std::vector<uint64_t> cudaReverse(int numShinyRolls, SeedVerifyConstantsFlexible verify) {
    SeedReverseConstantsFlexible reversal = createReversalStructForCUDA(numShinyRolls);
    for (int i = 0; i < 6; i++) {
        printf("IV%d: %u %u\n", i, verify.ivs[i][0], verify.ivs[i][1]);
    }
    printf("Ability: %u %u\n", verify.ability[0], verify.ability[1]);
    printf("Gender: ratio:%u gender:%u\n", verify.genderData[0], verify.genderData[1]);
    printf("Nature: %u\n", verify.nature);
    printf("Height: %u %u\n", verify.height[0], verify.height[1]);
    printf("Weight: %u %u\n", verify.weight[0], verify.weight[1]);
    std::vector<uint64_t> outputBuffer(OUTPUT_BUFFER_SIZE);
    int foundSeeds = ivWrapper(reversal, verify, outputBuffer.data());
    printf("Total seeds %d\n", foundSeeds);
    outputBuffer.resize(foundSeeds);
    return outputBuffer;
}

std::vector<uint64_t> cudaReverse(int numShinyRolls, const PokemonEntity& entity) { 
    SeedVerifyConstantsFlexible verify = createVerifyStructForCUDA(entity);
    return cudaReverse(numShinyRolls, verify);
}