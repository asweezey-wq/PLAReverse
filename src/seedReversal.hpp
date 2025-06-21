#pragma once
#include "matrix.hpp"
#include "pokemon_cuda.h"
#include "pokemonEntity.hpp"
#include <string>
#include <vector>
#include <unordered_set>

GF2Matrix computeIVMatrix(int numShinyRolls);
GF2Vector computeIVConst(int numShinyRolls);

GF2Matrix computeSlotMatrix();
GF2Vector computeSlotConst();

GF2Matrix computeGroupMatrix(int genIndex);
GF2Vector computeGroupConst(int genIndex);

bool verifySeed(uint64_t seed, int numShinyRolls, uint8_t genderRatio, const PokemonEntity& entity);
SeedReverseConstantsFlexible createReversalStructForCUDA(int numShinyRolls);
SeedVerifyConstantsFlexible createVerifyStructForCUDA(const PokemonEntity& entity);

std::vector<uint64_t> cudaReverse(int numShinyRolls, const PokemonEntity& entity);
std::vector<uint64_t> cudaReverse(int numShinyRolls, SeedVerifyConstantsFlexible verifyConst);