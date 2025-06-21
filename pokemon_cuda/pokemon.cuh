#pragma once
#include "pokemon_cuda.h"

__device__ bool verifySeed(uint64_t seed, const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible& verifyConst);
__device__ bool verifySeedNoIVs(uint64_t seed, const SeedReverseConstantsFlexible& reversalConst, const SeedVerifyConstantsFlexible& verifyConst);