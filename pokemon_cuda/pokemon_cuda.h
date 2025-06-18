#pragma once
#include <cstdint>

#define OUTPUT_BUFFER_SIZE 20000

struct SeedReverseConstantsFlexible {
    uint8_t shinyRolls;
    uint64_t seedVector[60];
    uint64_t nullspace[16];
    uint64_t ivConst;
    uint8_t ivs[6][2];
};

struct SeedVerifyConstantsFlexible {
    uint8_t ability[2];
    uint64_t genderData[2];
    uint8_t nature;
    uint64_t height[2];
    uint64_t weight[2];
};

int ivWrapper(const SeedReverseConstantsFlexible reverseConst, const SeedVerifyConstantsFlexible verifyConst, uint64_t* outputBuffer);
int ivWrapperFlexible(const SeedReverseConstantsFlexible reverseConst, const SeedVerifyConstantsFlexible verifyConst, uint64_t* outputBuffer);
