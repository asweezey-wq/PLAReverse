#include "pokemonGenerator.hpp"
#include "pokemonEntity.hpp"
#include "outbreakSpawner.hpp"
#include "seedReversal.hpp"
#include "pokemon_cuda.h"
#include "gameInference.hpp"
#include <iostream>

void reversal() {
    uint64_t seed = 0xdeadbeef;
    PokemonSlotGroup group;
    uint32_t species = PokemonData::getSpeciesID("Tentacool");
    auto data = PokemonData::getSpeciesData(species);
    group.addSlot(PokemonSlot{
        .m_species = species,
        .m_rate = 100,
        .m_isAlpha = false,
        .m_levelRange = {0, 1},
        .m_numPerfectIVs = 0
    });
    group.addSlot(PokemonSlot{
        .m_species = species,
        .m_rate = 1,
        .m_isAlpha = true,
        .m_levelRange = {0, 1},
        .m_numPerfectIVs = 3
    });
    OutbreakSpawner spawner(group);
    std::vector<PokemonEntity> pokemon;
    seed = spawner.spawnPokemon(seed, 4, 32, pokemon);
    std::swap(pokemon[1], pokemon[2]);
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t* outputBuffer = new uint64_t[OUTPUT_BUFFER_SIZE];
    SeedReverseConstantsFlexible reversalCUDA = createReversalStructForCUDA(32);
    SeedVerifyConstantsFlexible verifyCUDA[4];
    for (int i = 0; i < 4; i++) {
        verifyCUDA[i] = createVerifyStructForCUDA(pokemon[i]);
        // float dispHeight, dispWeight;
        // getDisplaySize(data, true, pokemon[i].m_height, pokemon[i].m_weight, dispHeight, dispWeight);
        // uint32_t heightRange[2];
        // uint32_t weightRange[2];
        // calculateSizeRanges(data, true, dispHeight, dispWeight, heightRange, weightRange);
        // verifyCUDA[i].height[0] = heightRange[0];
        // verifyCUDA[i].height[1] = heightRange[1] + 1;
        // verifyCUDA[i].weight[0] = weightRange[0];
        // verifyCUDA[i].weight[1] = weightRange[1] + 1;
    }
    int results = reversePokemonFromStartingMon(reversalCUDA, verifyCUDA, 0, outputBuffer);
    for (int i = 0; i < results; i++) {
        printf("Valid group seed %llx\n", outputBuffer[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Seed search took %.02fs\n", (double)duration.count() / 1000000);
    delete[] outputBuffer;
}

void seedFind() {
    uint32_t id = PokemonData::getSpeciesID("Swinub");
    uint32_t id_evol = PokemonData::getSpeciesID("Piloswine");
    uint32_t id_evol2 = PokemonData::getSpeciesID("Mamoswine");
    auto data = PokemonData::getSpeciesData(id);
    // SeedVerifyConstantsFlexible verify = {
    //     .ivs = {{1, 4}, {16, 19}, {17, 19}, {13, 14}, {31, 32}, {26, 28}},
    //     .ability = {1, 2},
    //     .genderData = {data.genderRatio, FEMALE},
    //     .nature = Gentle,
    //     .height = {77, 118},
    //     .weight = {205, 254},
    // };
    SeedVerifyConstantsFlexible verify2 = {
        .ivs = {{1, 3}, {16, 18}, {17, 19}, {13, 14}, {31, 32}, {27, 28}},
        .ability = {1, 2},
        .genderData = {data.genderRatio, FEMALE},
        .nature = Gentle,
        .height = {77, 118},
        .weight = {205, 254},
    };
    int numShinyRolls = getShinyRolls(SHINY_CHARM | SHINY_MMO);
    // auto start = std::chrono::high_resolution_clock::now();
    // auto seeds = cudaReverse(numShinyRolls, verify2);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // printf("Seed generation took %.02fs\n", (double)duration.count() / 1000000);

    // PokemonEntity entity = {
    //     .m_species = id
    // };
    // generatePokemon(*seeds.begin(), numShinyRolls, data.genderRatio, entity);
    // std::cout << entity.toString() << std::endl;
    std::vector<SeedVerifyConstantsFlexible> pokemon;
    {
        Nature nature = Gentle;
        SeedVerifyConstantsFlexible verify = {
            .ability = {1, 1},
            .genderData = {data.genderRatio, FEMALE},
            .nature = nature,
        };
        std::vector<ObservedSizeInstance> observedSizes = {
            {
                id,
                imperialHeight(1, 3),
                15.8f
            },
            {
                id_evol,
                imperialHeight(3, 6),
                135.9f
            },
            {
                id_evol2,
                imperialHeight(7, 10),
                708.9f
            }
        };
        calculateSizeRanges(true, observedSizes, verify.height, verify.weight);
        std::vector<ObservedStatInstance> observedStats = {
            {
                id,
                32,
                {74, 42, 32, 28, 37, 45}
            },
            {
                id,
                33,
                {76, 43, 33, 29, 38, 46}
            },
            {
                id_evol,
                33,
                {109, 76, 56, 48, 59, 46}
            },
            {
                id_evol,
                34,
                {112, 78, 58, 50, 61, 48}
            }
        };
        uint8_t effortLevels[6] = {0,0,0,0,3,2};
        JudgeIVRating ratings[6] = {DECENT, PRETTYGOOD, PRETTYGOOD, DECENT, BEST, VERYGOOD};
        calculateIVRanges(
            effortLevels,
            ratings,
            nature,
            observedStats,
            verify.ivs);
        pokemon.push_back(verify);
    }
    {
        Nature nature = Bashful;
        SeedVerifyConstantsFlexible verify = {
            .ability = {0, 0},
            .genderData = {data.genderRatio, FEMALE},
            .nature = nature,
        };
        std::vector<ObservedSizeInstance> observedSizes = {
            {
                id,
                imperialHeight(1, 5),
                16
            }
        };
        calculateSizeRanges(true, observedSizes, verify.height, verify.weight);
        std::vector<ObservedStatInstance> observedStats = {
            {
                id,
                31,
                {73, 42, 36, 31, 28, 38}
            }
        };
        uint8_t effortLevels[6] = {0,1,1,1,0,0};
        JudgeIVRating ratings[6] = {DECENT, PRETTYGOOD, PRETTYGOOD, PRETTYGOOD, PRETTYGOOD, DECENT};
        calculateIVRanges(
            effortLevels,
            ratings,
            nature,
            observedStats,
            verify.ivs);
        pokemon.push_back(verify);
    }
    {
        Nature nature = Hardy;
        SeedVerifyConstantsFlexible verify = {
            .ability = {1, 1},
            .genderData = {data.genderRatio, MALE},
            .nature = nature,
        };
        std::vector<ObservedSizeInstance> observedSizes = {
            {
                id,
                imperialHeight(1, 5),
                15
            }
        };
        calculateSizeRanges(true, observedSizes, verify.height, verify.weight);
        std::vector<ObservedStatInstance> observedStats = {
            {
                id,
                32,
                {75, 43, 33, 30, 27, 44}
            }
        };
        uint8_t effortLevels[6] = {0,0,0,0,0,1};
        JudgeIVRating ratings[6] = {DECENT, PRETTYGOOD, DECENT, PRETTYGOOD, DECENT, PRETTYGOOD};
        calculateIVRanges(
            effortLevels,
            ratings,
            nature,
            observedStats,
            verify.ivs);
        pokemon.push_back(verify);
    }
    {
        Nature nature = Naughty;
        SeedVerifyConstantsFlexible verify = {
            .ability = {1, 1},
            .genderData = {data.genderRatio, MALE},
            .nature = nature,
        };
        std::vector<ObservedSizeInstance> observedSizes = {
            {
                id,
                imperialHeight(1, 3),
                12.9f
            }
        };
        calculateSizeRanges(true, observedSizes, verify.height, verify.weight);
        std::vector<ObservedStatInstance> observedStats = {
            {
                id,
                30,
                {77, 45, 37, 25, 22, 40}
            }
        };
        uint8_t effortLevels[6] = {1,1,2,0,0,0};
        JudgeIVRating ratings[6] = {PRETTYGOOD, PRETTYGOOD, VERYGOOD, DECENT, DECENT, PRETTYGOOD};
        calculateIVRanges(
            effortLevels,
            ratings,
            nature,
            observedStats,
            verify.ivs);
        pokemon.push_back(verify);
    }

    SeedReverseConstantsFlexible reversalConst = createReversalStructForCUDA(numShinyRolls);
    SeedVerifyConstantsFlexible verifyForCUDA[4] = {
        pokemon[0],
        pokemon[1],
        pokemon[2],
        pokemon[3]
    };
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t* outputBuffer = new uint64_t[OUTPUT_BUFFER_SIZE];
    int results = reversePokemonFromStartingMon(reversalConst, verifyForCUDA, 0, outputBuffer);
    for (int i = 0; i < results; i++) {
        printf("Group seed: %llx\n", outputBuffer[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Seed search took %.02fs\n", (double)duration.count() / 1000000);
    delete[] outputBuffer;
}

void spawner() {
    PokemonEntity entity;
    uint64_t seed = 6267248544019265885;
    PokemonSlotGroup group = PokemonData::getSlotGroupTable(0x00C756FF484B9DD6);
    // group.addSlot(PokemonSlot{
    //     .m_species = PokemonData::getSpeciesID("Swinub"),
    //     .m_rate = 100,
    //     .m_isAlpha = false,
    //     .m_levelRange = {0, 1},
    //     .m_numPerfectIVs = 0
    // });
    // group.addSlot(PokemonSlot{
    //     .m_species = PokemonData::getSpeciesID("Swinub"),
    //     .m_rate = 1,
    //     .m_isAlpha = true,
    //     .m_levelRange = {0, 1},
    //     .m_numPerfectIVs = 3
    // });
    OutbreakSpawner spawner(group);
    std::vector<PokemonEntity> pokemon;
    seed = spawner.spawnPokemon(seed, 4, 17, pokemon);
    seed = spawner.spawnPokemon(seed, 1, 17, pokemon);
    seed = spawner.spawnPokemon(seed, 1, 17, pokemon);
    for (auto mon : pokemon) {
        std::cout << mon.toString() << std::endl;
    }
}

constexpr uint64_t XOROSHIRO_ROTL_24 = 0x75229d6a5b82a2b1;
constexpr uint64_t XOROSHIRO_CONSTANT = 0x82a2b175229d6a5b;


inline uint64_t x0_from_x1(uint64_t x1) {
    x1 = std::rotl(x1, 27);
    return XOROSHIRO_ROTL_24 ^ x1 ^ (x1 << 16) ^ std::rotl(x1, 24);
}

inline uint64_t seed_from_x1(uint64_t x1) {
    return std::rotl(x1, 27) ^ XOROSHIRO_CONSTANT;
}

int main(int, char**){
    PokemonData::loadSpeciesNamesFromFile("resources/pokemonSpecies.txt");
    PokemonData::loadSpeciesDataFromFile("resources/pokemonData.txt");
    PokemonData::loadAbilityNamesFromFile("resources/abilities.txt");
    PokemonData::loadTablesFromFile("resources/mmo_tables.txt");
    
    // spawner();
    // statTesting();
    // seedFind();
    // reversal();
}
