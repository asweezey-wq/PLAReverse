#pragma once
#include "matrix.hpp"
#include "pokemon_cuda.h"
#include "pokemonEntity.hpp"
#include <string>
#include <vector>

GF2Matrix computeIVMatrix(int numShinyRolls);
GF2Vector computeIVConst(int numShinyRolls);

bool verifySeed(uint64_t seed, int numShinyRolls, const PokemonEntity& entity);
SeedReverseConstantsFlexible createReversalStructForCUDA(int numShinyRolls, uint8_t ivs[6]);
SeedVerifyConstantsFlexible createVerifyStructForCUDA(const PokemonEntity& entity);
