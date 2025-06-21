#pragma once
#include "pokemon_cuda.h"

extern __constant__ SeedReverseConstantsFlexible g_reversalConstants;

__device__ uint64_t getBaseSeed(uint8_t ivs[6]);
__global__ void kernel_reverseIVSet(uint32_t* numFoundSeeds, uint64_t* outputBuffer);