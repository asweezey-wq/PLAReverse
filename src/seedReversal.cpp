#include "seedReversal.hpp"
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

std::vector<uint64_t> reverseIVSet(int numShinyRolls, uint8_t ivs1[6], uint8_t ivs2[6]) {
    GF2Matrix ivMat = computeIVMatrix(numShinyRolls);
    uint64_t ivConst = packVector(computeIVConst(numShinyRolls));
    GF2Matrix invIvMat = gf2Inverse(ivMat);
    GF2Matrix nullspace = gf2NullSpace(ivMat);
    std::vector<uint64_t> packedInvIvMat = gf2Pack(invIvMat);
    std::vector<uint64_t> packedNullspace = gf2Pack(nullspace);
    uint64_t packedIvInput = 0;
    for (int i = 0; i < 6; i++) {
        packedIvInput |= ((uint64_t)ivs1[i] << (10 * i));
        packedIvInput |= ((uint64_t)ivs2[i] << (10 * i + 5));
    }
    packedIvInput ^= ivConst;
    uint64_t baseSeed = 0;
    for (int i = 0; i < 64; i++) {
        if (packedIvInput & 1) {
            baseSeed ^= packedInvIvMat[i];
        }
        packedIvInput >>= 1;
    }
    std::vector<uint64_t> result;
    for (int i = 0; i < packedNullspace.size(); i++) {
        result.push_back(baseSeed ^ packedNullspace[i]);
    }
    return result;
}