#pragma once
#include "matrix.hpp"
#include "pokemonCuda.hpp"
#include "pokemonData.hpp"
#include "gameInference.hpp"
#include <string>
#include <vector>

GF2Matrix computeIVMatrix(int numShinyRolls);
GF2Vector computeIVConst(int numShinyRolls);

GF2Matrix computeGroupMatrix(int genIndex, int vecShift);
GF2Vector computeGroupConst(int genIndex, int vecShift);

SeedReversalContext createReversalCtxForCUDA(int numShinyRolls, const PokemonSlotGroup& slotGroup);
PokemonVerificationContext createOracleVerifyStructForCUDA(const PokemonEntity& entity, const PokemonSlotGroup& slotGroup);
void createSizePairsForCUDA(const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs, SizePairs& cudaSizePairs);
uint64_t getExpectedSeeds(const PokemonVerificationContext& verifyConst);
uint64_t getExpectedSeedsWithSizePairs(const PokemonVerificationContext& verifyConst, const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs);
uint64_t getTheoreticalGeneratorSeeds(const PokemonVerificationContext& verifyConst, uint32_t slotRateSum);
uint64_t getTheoreticalGeneratorSeedsWithSizePairs(const PokemonVerificationContext& verifyConst, const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs, uint32_t slotRateSum);
void getPossibleSizes(const uint32_t heightRange[2], const uint32_t weightRange[2], uint32_t& numHeights, uint32_t& numWeights);
double getSizePairProbability(const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs);

void parseJSONMMOEncounter(std::string filePath, std::vector<PokemonVerificationContext>& pokemon, std::vector<std::vector<ObservedSizeInstance>>& sizes, int& numShinyRolls, uint64_t& tableID);