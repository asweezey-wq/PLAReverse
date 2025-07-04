#pragma once
#include "pokemonCuda.hpp"

extern __constant__ SeedReversalContext g_seedReversalCtx;
extern __constant__ uint64_t g_slices[256];