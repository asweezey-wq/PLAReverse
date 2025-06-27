#pragma once
#include "matrix.hpp"
#include "pokemonCuda.h"
#include "pokemonData.hpp"
#include <string>
#include <vector>

GF2Matrix computeIVMatrix(int numShinyRolls);
GF2Vector computeIVConst(int numShinyRolls);

GF2Matrix computeGroupMatrix(int genIndex, int vecShift);
GF2Vector computeGroupConst(int genIndex, int vecShift);

SeedReversalContext createReversalCtxForCUDA(int numShinyRolls, const PokemonSlotGroup& slotGroup);
PokemonVerificationContext createOracleVerifyStructForCUDA(const PokemonEntity& entity, const PokemonSlotGroup& slotGroup);
uint64_t getExpectedSeeds(const PokemonVerificationContext& verifyConst);
uint64_t getTheoreticalGeneratorSeeds(const PokemonVerificationContext& verifyConst, uint32_t slotRateSum);

void parseJSONMMOEncounter(std::string filePath, std::vector<PokemonVerificationContext>& pokemon, int& numShinyRolls, uint64_t& tableID);