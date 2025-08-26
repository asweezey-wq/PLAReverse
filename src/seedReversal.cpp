#include "seedReversal.hpp"
#include "gameInference.hpp"
#include "xoroshiro.hpp"
#include "json.hpp"
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <bit>

using json = nlohmann::json;

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

GF2Matrix computeGroupMatrix(int genIndex, int vecShift) {
    const int bitsOfSeed = 32;
    GF2Matrix matrix = gf2Init(64, bitsOfSeed*2);
    for (int i = 0; i < 64; i++) {
        uint64_t seed = 1ull << i;
        Xoroshiro128PlusRNG rng(seed, 0);
        for (int j = 0; j < genIndex; j++) {
            rng.next();
            rng.next();
        }
        for (int j = 0; j < bitsOfSeed; j++) {
            int shift = vecShift + j;
            matrix[i][j] = (rng.getSeed0() >> shift) & 1;
            matrix[i][j + bitsOfSeed] = (rng.getSeed1() >> shift) & 1;
        }
    }
    return matrix;
}

GF2Vector computeGroupConst(int genIndex, int vecShift) {
    const int bitsOfSeed = 32;
    GF2Vector values(bitsOfSeed*2);
    Xoroshiro128PlusRNG rng(0);
    for (int j = 0; j < genIndex; j++) {
        rng.next();
        rng.next();
    }
    for (int j = 0; j < bitsOfSeed; j++) {
        int shift = vecShift + j;
        values[j] = (rng.getSeed0() >> shift) & 1;
        values[j + bitsOfSeed] = (rng.getSeed1() >> shift) & 1;
    }
    return values;
}

SeedReversalContext createReversalCtxForCUDA(int numShinyRolls, const PokemonSlotGroup& slotGroup) {
    SeedReversalContext reversalCtx = {
        .shinyRolls = (uint8_t)numShinyRolls
    };

    // Pokemon Seed Reversal
    GF2Matrix ivMat = computeIVMatrix(numShinyRolls);
    uint64_t ivConst = packVector(computeIVConst(numShinyRolls));
    GF2Matrix invIvMat = gf2Inverse(ivMat);
    GF2Matrix nullspace = gf2NullSpace(ivMat);
    std::vector<uint64_t> packedInvIvMat = gf2Pack(invIvMat);
    std::vector<uint64_t> packedNullspace = gf2Pack(nullspace);
    reversalCtx.pokemonReversalCtx.xoroshiroBits = ivConst;
    memcpy(reversalCtx.pokemonReversalCtx.inverseMatrix, packedInvIvMat.data(), 60 * sizeof(uint64_t));
    memcpy(reversalCtx.pokemonReversalCtx.nullspace, packedNullspace.data(), 16 * sizeof(uint64_t));

    // Generator Seed Reversal
    reversalCtx.generatorSlotSum = slotGroup.getSlotRateSum();

    // Group Seed Reversal
    uint32_t shiftConst[3] = {2,32,0};
    for (int i = 0; i < 3; i++) {
        reversalCtx.groupReversalCtx.shiftConst[i] = shiftConst[i];
        GF2Matrix groupMat = computeGroupMatrix(i+1, shiftConst[i]);
        GF2Matrix invGroupMat = gf2Inverse(groupMat);
        auto packedInvGroupMat = gf2Pack(invGroupMat);
        memcpy(reversalCtx.groupReversalCtx.inverseMatrix[i], packedInvGroupMat.data(), 64 * sizeof(uint64_t));
        uint64_t xoroBits = packVector(computeGroupConst(i+1, shiftConst[i]));
        reversalCtx.groupReversalCtx.xoroshiroBits[i] = xoroBits;
    }

    return reversalCtx;
}

PokemonVerificationContext createOracleVerifyStructForCUDA(const PokemonEntity& entity, const PokemonSlotGroup& slotGroup) {
    const PokemonSlot* slot = nullptr;
    uint32_t slotBaseRate = 0;
    for (int i = 0; i < slotGroup.numSlots(); i++) {
        if (slotGroup.getSlotFromIndex(i).m_species == entity.m_species) {
            slot = &slotGroup.getSlotFromIndex(i);
            break;
        }
        slotBaseRate += slotGroup.getSlotFromIndex(i).m_rate;
    }
    if (!slot) {
        fprintf(stderr, "Could not find correct spawner slot for Pokemon species %s\n", PokemonData::getSpeciesName(entity.m_species).c_str());
        exit(1);
    }
    auto data = PokemonData::getSpeciesData(entity.m_species);
    uint8_t ivRanges[6][2];
    for (int i = 0; i < 6; i++) {
        ivRanges[i][0] = entity.m_ivs[i];
        ivRanges[i][1] = entity.m_ivs[i];
    }
    uint8_t slotLevelRange = 1 + slot->m_levelRange.second - slot->m_levelRange.first;
    uint8_t levelRangeBitMask = slotLevelRange == 1 ? 1 : (1 << std::bit_width((uint8_t)(slotLevelRange - 1))) - 1;
    PokemonVerificationContext verify = {
        .speciesId = entity.m_species,
        .level = entity.m_level,
        .slotThresholds = {(float)slotBaseRate, (float)slotBaseRate + slot->m_rate},
        .levelRange = {slot->m_levelRange.first, slotLevelRange, levelRangeBitMask},
        .ability = {(uint64_t)entity.m_ability, (uint64_t)entity.m_ability},
        .genderData = {data.genderRatio, (uint8_t)entity.m_gender},
        .nature = entity.m_nature,
        .height = {(uint64_t)entity.m_height, (uint64_t)entity.m_height},
        .weight = {(uint64_t)entity.m_weight, (uint64_t)entity.m_weight}
    };
    memcpy(verify.ivs, ivRanges, sizeof(ivRanges));
    return verify;
}

void createSizePairsForCUDA(const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs, SizePairs& cudaSizePairs) {
    memset(cudaSizePairs.sizeMapping, 0, sizeof(cudaSizePairs.sizeMapping));
    for (auto& pair : sizePairs) {
        cudaSizePairs.sizeMapping[pair.first][pair.second >> 6] |= (1ull << (pair.second & 63));
    }
}

std::unordered_map<uint32_t, double> g_sizeSumDistribution;

void getPossibleSizes(const uint32_t heightRange[2], const uint32_t weightRange[2], uint32_t& numHeights, uint32_t& numWeights) {
    numHeights = 0;
    numWeights = 0;
    for (int i = 0; i < 129; i++) {
        for (int j = 0; j < 128; j++) {
            uint32_t sum = i + j;
            if (sum >= heightRange[0] && sum <= heightRange[1]) {
                numHeights += 1;
            }
            if (sum >= weightRange[0] && sum <= weightRange[1]) {
                numWeights += 1;
            } 
        }
    }
}

double getSizePairProbability(const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs) {
    if (g_sizeSumDistribution.empty()) {
        constexpr double incr = 1.0 / (128.0 * 129.0);
        for (int i = 0; i < 129; i++) {
            for (int j = 0; j < 128; j++) {
                g_sizeSumDistribution[i+j] += incr;
            }
        }
    }
    double sumProb = 0;
    for (auto& pair : sizePairs) {
        double combinedProb = g_sizeSumDistribution[pair.first] * g_sizeSumDistribution[pair.second];
        sumProb += combinedProb;
    }
    return sumProb;
}

uint64_t getExpectedSeeds(const PokemonVerificationContext& verifyConst) {
    auto& data = PokemonData::getSpeciesData(verifyConst.speciesId);
    bool genderless = verifyConst.genderData[1] == GENDERLESS || data.genderRatio >= 254 || data.genderRatio == 0;
    float genderOdds = 1.0f;
    if (!genderless) {
        genderOdds = (data.genderRatio - 1) / 253.0f;
        if (verifyConst.genderData[1] == MALE) {
            genderOdds = 1.0f - genderOdds;
        }
    }
    uint32_t numHeights, numWeights;
    getPossibleSizes(verifyConst.height, verifyConst.weight, numHeights, numWeights);
    double allPossibleSizes = (129 * 128);
    double sizeOdds = (numHeights / allPossibleSizes) * (numWeights / allPossibleSizes);
    uint64_t numIVCombos = getNumIVPermutations(verifyConst.ivs);
    double abilityModifier = verifyConst.ability[0] == verifyConst.ability[1] ? 2 : 1;
    double seedOdds = (abilityModifier * 25 * (1.0f / genderOdds) * (1.0f / sizeOdds));
    double numSeeds = (double)(numIVCombos << 34) / seedOdds;
    return (uint64_t)numSeeds;
}

uint64_t getExpectedSeedsWithSizePairs(const PokemonVerificationContext& verifyConst, const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs) {
    auto& data = PokemonData::getSpeciesData(verifyConst.speciesId);
    bool genderless = verifyConst.genderData[1] == GENDERLESS || data.genderRatio >= 254 || data.genderRatio == 0;
    float genderOdds = 1.0f;
    if (!genderless) {
        genderOdds = (data.genderRatio - 1) / 253.0f;
        if (verifyConst.genderData[1] == MALE) {
            genderOdds = 1.0f - genderOdds;
        }
    }
    double sizeOdds = getSizePairProbability(sizePairs);
    uint64_t numIVCombos = getNumIVPermutations(verifyConst.ivs);
    double abilityModifier = verifyConst.ability[0] == verifyConst.ability[1] ? 2 : 1;
    double seedOdds = (abilityModifier * 25 * (1.0f / genderOdds) * (1.0f / sizeOdds));
    double numSeeds = (double)(numIVCombos << 34) / seedOdds;
    return (uint64_t)numSeeds;
}

uint64_t getTheoreticalGeneratorSeeds(const PokemonVerificationContext& verifyConst, uint32_t slotRateSum) {
    auto& data = PokemonData::getSpeciesData(verifyConst.speciesId);
    bool genderless = verifyConst.genderData[1] == GENDERLESS || data.genderRatio >= 254 || data.genderRatio == 0;
    float genderOdds = 1.0f;
    if (!genderless) {
        genderOdds = (data.genderRatio - 1) / 253.0f;
        if (verifyConst.genderData[1] == MALE) {
            genderOdds = 1.0f - genderOdds;
        }
    }
    uint32_t numHeights, numWeights;
    getPossibleSizes(verifyConst.height, verifyConst.weight, numHeights, numWeights);
    double allPossibleSizes = (129 * 128);
    double sizeOdds = (numHeights / allPossibleSizes) * (numWeights / allPossibleSizes);
    double abilityModifier = data.abilities[0] == data.abilities[1] ? 1 : 2;
    double levelOdds = verifyConst.levelRange[1] ? verifyConst.levelRange[1] : 1.0; // 3 possible levels for an MMO
    double slotOdds = (verifyConst.slotThresholds[1] - verifyConst.slotThresholds[0]) / slotRateSum;
    double seedOdds = (abilityModifier * 25 * (1.0 / genderOdds) * (1.0 / sizeOdds) * levelOdds * (1.0 / slotOdds));
    double numSeeds = (double)(1ull << 34) / seedOdds;
    return (uint64_t)numSeeds;
}

uint64_t getTheoreticalGeneratorSeedsWithSizePairs(const PokemonVerificationContext& verifyConst, const std::vector<std::pair<uint32_t, uint32_t>>& sizePairs, uint32_t slotRateSum) {
    auto& data = PokemonData::getSpeciesData(verifyConst.speciesId);
    bool genderless = verifyConst.genderData[1] == GENDERLESS || data.genderRatio >= 254 || data.genderRatio == 0;
    float genderOdds = 1.0f;
    if (!genderless) {
        genderOdds = (data.genderRatio - 1) / 253.0f;
        if (verifyConst.genderData[1] == MALE) {
            genderOdds = 1.0f - genderOdds;
        }
    }
    double sizeOdds = getSizePairProbability(sizePairs);
    double abilityModifier = data.abilities[0] == data.abilities[1] ? 1 : 2;
    double levelOdds = verifyConst.levelRange[1] ? verifyConst.levelRange[1] : 1.0; // 3 possible levels for an MMO
    double slotOdds = (verifyConst.slotThresholds[1] - verifyConst.slotThresholds[0]) / slotRateSum;
    double seedOdds = (abilityModifier * 25 * (1.0 / genderOdds) * (1.0 / sizeOdds) * levelOdds * (1.0 / slotOdds));
    double numSeeds = (double)(1ull << 34) / seedOdds;
    return (uint64_t)numSeeds;
}

bool jsonFailed = false;

int jsonExpectInt(json& jsonObj, const std::string name) {
    if (!jsonObj.contains(name)) {
        fprintf(stderr, "Did not find '%s' property (expected an int)\n", name.c_str());
        jsonFailed = true;
        return 0;
    }
    if (!jsonObj[name].is_number_integer()) {
        fprintf(stderr, "'%s' property was expected to be an int\n", name.c_str());
        jsonFailed = true;
        return 0;
    }
    return jsonObj[name].get<int>();
}

float jsonExpectFloat(json& jsonObj, const std::string name) {
    if (!jsonObj.contains(name)) {
        fprintf(stderr, "Did not find '%s' property (expected a float)\n", name.c_str());
        jsonFailed = true;
        return 0;
    }
    if (!jsonObj[name].is_number()) {
        fprintf(stderr, "'%s' property was expected to be a number\n", name.c_str());
        jsonFailed = true;
        return 0;
    }
    return jsonObj[name].get<float>();
}

std::string jsonExpectString(json& jsonObj, const std::string name) {
    if (!jsonObj.contains(name)) {
        fprintf(stderr, "Did not find '%s' property (expected a string)\n", name.c_str());
        jsonFailed = true;
        return "";
    }
    if (!jsonObj[name].is_string()) {
        fprintf(stderr, "'%s' property was expected to be a string\n", name.c_str());
        jsonFailed = true;
        return "";
    }
    return jsonObj[name].get<std::string>();
}

json jsonExpectArray(json& jsonObj, const std::string name) {
    if (!jsonObj.contains(name)) {
        fprintf(stderr, "Did not find '%s' property (expected an array)\n", name.c_str());
        jsonFailed = true;
        return json();
    }
    if (!jsonObj[name].is_array()) {
        fprintf(stderr, "'%s' property was expected to be an array\n", name.c_str());
        jsonFailed = true;
        return json();
    }
    return jsonObj[name];
}

void parseJSONMMOEncounter(std::string filePath, std::vector<PokemonVerificationContext>& outputPokemon, std::vector<std::vector<ObservedSizeInstance>>& sizes, int& numShinyRolls, uint64_t& tableID) {
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        fprintf(stderr, "Could not open encounter file %s\n", filePath.c_str());
        exit(1);
    }
    printf("Parsing %s\n", filePath.c_str());
    json fileJson = json::parse(ifs);
    ifs.close();
    if (!fileJson.is_object()) {
        fprintf(stderr, "EncounterJSON: Expected top-level to be a JSON object\n");
        exit(1);
    }
    numShinyRolls = jsonExpectInt(fileJson, "shinyRolls");
    auto tableIDStr = jsonExpectString(fileJson, "tableID");
    const PokemonSlotGroup* slotGroup = nullptr;
    if (!jsonFailed) {
        tableID = std::stoull(tableIDStr, NULL, 16);
        slotGroup = &PokemonData::getSlotGroupTable(tableID);
    }
    json pokemonList = jsonExpectArray(fileJson, "pokemon");
    for (auto pokemon : pokemonList) {
        if (!pokemon.is_object()) {
            fprintf(stderr, "Pokemon list should be an array of objects\n");
            jsonFailed = true;
            continue;
        }
        std::string name = jsonExpectString(pokemon, "name");
        uint32_t id = PokemonData::getSpeciesID(name);
        const SpeciesData& data = PokemonData::getSpeciesData(id);
        std::string nature = jsonExpectString(pokemon, "nature");
        uint8_t level = jsonExpectInt(pokemon, "level");
        int32_t natureIndex = -1;
        for (int i = 0; i < 25; i++) {
            if (nature == NATURE_NAMES[i]) {
                natureIndex = i;
                break;
            }
        }
        if (natureIndex == -1) {
            fprintf(stderr, "Invalid nature %s\n", nature.c_str());
            jsonFailed = true;
        }
        int gender = GENDERLESS;
        if (data.genderRatio == 255) {
            // genderless pokemon
            if (pokemon.contains("gender")) {
                fprintf(stderr, "Specified gender for genderless pokemon %s\n", name.c_str());
                jsonFailed = true;
            }
        } else {
            std::string genderStr = jsonExpectString(pokemon, "gender");
            if (genderStr == "F") {
                gender = FEMALE;
            } else if (genderStr == "M") {
                gender = MALE;
            } else {
                fprintf(stderr, "Invalid gender %s\n", genderStr.c_str());
                jsonFailed = true;
            }
        }

        float thresholds[2] = {0.0f, 100.0f};
        uint32_t levelRange[3] = {0,0,1};
        if (slotGroup) {
            bool foundPokemon = false;
            for (int i = 0; i < slotGroup->numSlots(); i++) {
                auto slot = slotGroup->getSlotFromIndex(i);
                if (slot.m_species == id) {
                    foundPokemon = true;
                    thresholds[1] = thresholds[0] + slot.m_rate;
                    levelRange[0] = slot.m_levelRange.first;
                    levelRange[1] = 1 + slot.m_levelRange.second - slot.m_levelRange.first;
                    if (levelRange[1] != 3) {
                        fprintf(stderr, "Expected slot level range to be 3\n");
                        jsonFailed = true;
                    }
                    levelRange[2] = 3;
                    break;
                }
                thresholds[0] += slot.m_rate;
            }
            if (!foundPokemon) {
                fprintf(stderr, "Species %s does not match spawn table!\n", name.c_str());
                jsonFailed = true;
            }
        }
        
        PokemonVerificationContext verify = {
            .speciesId = id,
            .level = level,
            .slotThresholds = {thresholds[0], thresholds[1]},
            .levelRange = {(uint8_t)levelRange[0], (uint8_t)levelRange[1], (uint8_t)levelRange[2]},
            .ability = {0, 1},
            .genderData = {data.genderRatio, (uint32_t)gender},
            .nature = (uint8_t)natureIndex,
            .height = {0, 255},
            .weight = {0, 255}
        };

        if (pokemon.contains("ability")) {
            json abilityJson = pokemon["ability"];
            if (abilityJson.is_number_integer()) {
                int ability = abilityJson.get<int>();
                if (ability != 0 && ability != 1) {
                    fprintf(stderr, "Ability value should be either 0 or 1\n");
                    jsonFailed = true;
                } else {
                    verify.ability[0] = ability;
                    verify.ability[1] = ability;
                }
            } else if (abilityJson.is_string()) {
                uint32_t abilityID = PokemonData::getAbilityID(abilityJson.get<std::string>());
                int ability = 2;
                if (abilityID == data.abilities[0]) {
                    ability = 0;
                } else if (abilityID == data.abilities[1]) {
                    ability = 1;
                } else {
                    fprintf(stderr, "Ability %s was not recognized as one of %s's abilities\n", abilityJson.get<std::string>().c_str(), name.c_str());
                    jsonFailed = true;
                }
                verify.ability[0] = ability;
                verify.ability[1] = ability;
            }
        }

        // Size calculation
        json sizesArray = jsonExpectArray(pokemon, "sizes");
        std::vector<ObservedSizeInstance> observedSizeInstances;
        bool imperial = true;
        for (auto sizeJson : sizesArray) {
            if (!pokemon.is_object()) {
                fprintf(stderr, "Size list should be an array of objects\n");
                jsonFailed = true;
                continue;
            }
            float dispHeight = 0;
            float dispWeight = 0;
            if (sizeJson.contains("height_ft")) {
                int feet = jsonExpectInt(sizeJson, "height_ft");
                int inches = jsonExpectInt(sizeJson, "height_in");
                dispHeight = (float)feet * 12 + (float)inches;
                imperial = true;
            } else if (sizeJson.contains("height_m")) {
                dispHeight = jsonExpectFloat(sizeJson, "height_m");
                imperial = false;
            } else {
                fprintf(stderr, "No height measurement found\n");
                jsonFailed = true;
            }

            if (sizeJson.contains("weight_lbs")) {
                dispWeight = jsonExpectFloat(sizeJson, "weight_lbs");
                if (!imperial) {
                    fprintf(stderr, "Height/weight imperial measurement mismatch\n");
                    jsonFailed = true;
                }
            } else if (sizeJson.contains("weight_kg")) {
                dispWeight = jsonExpectFloat(sizeJson, "weight_kg");
                if (imperial) {
                    fprintf(stderr, "Height/weight imperial measurement mismatch\n");
                    jsonFailed = true;
                }
            } else {
                fprintf(stderr, "No weight measurement found\n");
                jsonFailed = true;
            }

            uint32_t speciesId = id;
            if (sizeJson.contains("name")) {
                std::string name = jsonExpectString(sizeJson, "name");
                speciesId = PokemonData::getSpeciesID(name);
            }

            ObservedSizeInstance size = {
                .speciesId = speciesId,
                .height = dispHeight,
                .weight = dispWeight
            };
            observedSizeInstances.emplace_back(size);
        }
        sizes.emplace_back(observedSizeInstances);
        if (!jsonFailed) {
            calculateSizeRanges(imperial, observedSizeInstances, verify.height, verify.weight);
        }

        if (verify.height[1] > verify.height[1]) {
            fprintf(stderr, "Invalid height range. Check to make sure the sizes you entered are correct\n");
            jsonFailed = true;
        }
        if (verify.weight[1] > verify.weight[1]) {
            fprintf(stderr, "Invalid weight range. Check to make sure the sizes you entered are correct\n");
            jsonFailed = true;
        }

        // Stats
        json elJson = jsonExpectArray(pokemon, "effortLevels");
        if (elJson.size() != 6) {
            fprintf(stderr, "Effort levels should be an array of 6 ints\n");
            jsonFailed = true;
        } else {
            uint8_t effortLevels[6];
            for (int i = 0; i < 6; i++) {
                if (elJson[i].is_number_unsigned()) {
                    effortLevels[i] = elJson[i].get<uint8_t>();
                } else {
                    fprintf(stderr, "Effort levels should be an array of 6 ints\n");
                    jsonFailed = true;
                }
            }
            calculatePLAELRanges(effortLevels, verify.ivs);
        }

        if (pokemon.contains("ratings")) {
            json ratingsJson = jsonExpectArray(pokemon, "ratings");
            if (ratingsJson.size() != 6) {
                fprintf(stderr, "Ratings should be an array of 6 strings\n");
                jsonFailed = true;
            } else {
                JudgeIVRating ratings[6];
                for (int i = 0; i < 6; i++) {
                    json rating = ratingsJson[i];
                    if (!rating.is_string()) {
                        fprintf(stderr, "Expected rating to be a string\n");
                        jsonFailed = true;
                    } else {
                        std::string ratingStr = rating.get<std::string>();
                        if (ratingStr == "NoGood") {
                            ratings[i] = NOGOOD;
                        } else if (ratingStr == "Decent") {
                            ratings[i] = DECENT;
                        } else if (ratingStr == "PrettyGood") {
                            ratings[i] = PRETTYGOOD;
                        } else if (ratingStr == "VeryGood") {
                            ratings[i] = VERYGOOD;
                        } else if (ratingStr == "Fantastic") {
                            ratings[i] = FANTASTIC;
                        } else if (ratingStr == "Best") {
                            ratings[i] = BEST;
                        } else {
                            fprintf(stderr, "Invalid rating %s\n", ratingStr.c_str());
                            jsonFailed = true;
                        }
                    }
                }
                restrictRangesForJudging(ratings, verify.ivs);
            }
        }

        if (pokemon.contains("stats")) {
            json statsJson = jsonExpectArray(pokemon, "stats");
            for (auto statInst : statsJson) {
                if (!statInst.is_object()) {
                    fprintf(stderr, "Expected observed stat instance to be an object\n");
                    jsonFailed = true;
                } else {
                    int level = jsonExpectInt(statInst, "level");
                    uint32_t speciesId = id;
                    if (statInst.contains("name")) {
                        speciesId = PokemonData::getSpeciesID(jsonExpectString(statInst, "name"));
                    }

                    ObservedStatInstance obs = {
                        .speciesId = speciesId,
                        .level = level,
                    };
                    json statsArr = jsonExpectArray(statInst, "stats");
                    if (statsArr.size() != 6) {
                        fprintf(stderr, "Stats should be an array of 6 ints\n");
                        jsonFailed = true;
                    } else {
                        for (int i = 0; i < 6; i++) {
                            if (!statsArr[i].is_number_unsigned()) {
                                fprintf(stderr, "Expected stat to be an int\n");
                                jsonFailed = true;
                            } else {
                                obs.stats[i] = statsArr[i].get<uint32_t>();
                            }
                        }
                    }

                    if (!jsonFailed) {
                        restrictRangesForActualStats(obs, verify.nature, verify.ivs);
                    }
                }
            }
        }

        for (int i = 0; i < 6; i++) {
            if (verify.ivs[i][0] > verify.ivs[i][1]) {
                fprintf(stderr, "Invalid IV range for IV %d\n", i);
                jsonFailed = true;
            }
        }

        outputPokemon.push_back(verify);
    }

    if (jsonFailed) {
        fprintf(stderr, "JSON parsing failed\n");
        exit(1);
    }
}