#include "outbreakSpawner.hpp"
#include "seedReversal.hpp"
#include "pokemonCuda.h"
#include "gameInference.hpp"
#include "xoroshiro.hpp"
#include "ui/MainWindow.h"
#include <iostream>
#include <QApplication>

void reversal() {
    ShinyRollData shinyRolls;
    uint64_t trueGroupSeed = 6267248544019265885;
    uint64_t slotGroupNumber = 0x00C756FF484B9DD6;
    uint32_t species = PokemonData::getSpeciesID("Swinub");
    shinyRolls[species] = 17;
    auto data = PokemonData::getSpeciesData(species);
    auto slotGroup = PokemonData::getSlotGroupTable(slotGroupNumber);
    OutbreakSpawner spawner(slotGroup, shinyRolls);
    PokemonVerificationContext groupPokemon[4];
    std::vector<uint64_t> pokemonSeeds;
    {
        printf("Group seed %llx\n", trueGroupSeed);
        Xoroshiro128PlusRNG groupRng(trueGroupSeed);
        for (int i = 0; i < 4; i++) {
            uint64_t genSeed = groupRng.next();
            groupRng.next();
            Xoroshiro128PlusRNG genRng(genSeed);
            genRng.next();
            PokemonEntity entity = spawner.createPokemon(genSeed);
            groupPokemon[i] = createOracleVerifyStructForCUDA(entity, slotGroup);
            printf("Gen seed %016llx Pokemon seed %016llx\n", entity.m_generatorSeed, entity.m_pokemonSeed);
        }
    }

    std::vector<ObservedSizeInstance> sizes;
    sizes.emplace_back(species, 15, 15.8);
    sizes.emplace_back(PokemonData::getSpeciesID("Piloswine"), 42, 135.9);
    calculateSizeRanges(true, sizes, groupPokemon[0].height, groupPokemon[0].weight);
    uint32_t numHeights, numWeights;
    getPossibleSizes(groupPokemon[0].height, groupPokemon[0].weight, numHeights, numWeights);
    auto pairs = calculateSizePairs(true, sizes);
    printf("Heights:%u Weights:%u Pairs:%u\n", numHeights, numWeights, pairs.size());

    auto& evoData = PokemonData::getSpeciesData(221);
    for (auto& pair : pairs) {
        float dispHeight, dispWeight;
        getDisplaySize(data, true, (uint8_t)pair.first, (uint8_t)pair.second, dispHeight, dispWeight);
        printf("%u %u %.1f %.1f\n", pair.first, pair.second, dispHeight, dispWeight);
        getDisplaySize(evoData, true, (uint8_t)pair.first, (uint8_t)pair.second, dispHeight, dispWeight);
        printf("%u %u %.1f %.1f\n", pair.first, pair.second, dispHeight, dispWeight);
    }

    uint32_t searchIndex = 0;
    if (searchIndex != 0) {
        std::swap(groupPokemon[searchIndex], groupPokemon[0]);
    }

    SeedReversalContext reversalCtx = createReversalCtxForCUDA(shinyRolls[species], slotGroup);
    reversalCtx.numPokemon = 4;
    for (int i = 0; i < 4; i++) {
        reversalCtx.pokemonVerifCtx[i] = groupPokemon[i];
    }

    uint64_t* outputBuffer = new uint64_t[4];
    int foundSeeds = reversePokemonFromSingleMon(reversalCtx, 0, outputBuffer);
    delete[] outputBuffer;
}

void seedFind(const char* filePath) {
    std::vector<PokemonVerificationContext> pokemon;
    int numShinyRolls;
    uint64_t tableID;
    parseJSONMMOEncounter(std::string(filePath), pokemon, numShinyRolls, tableID);
    auto slotGroup = PokemonData::getSlotGroupTable(tableID);
    for (int i = 0; i < 4; i++) {
        auto& verify = pokemon[i];
        auto& data = PokemonData::getSpeciesData(verify.speciesId);
        printf("%d: %s %u %c %s ", i, PokemonData::getSpeciesName(verify.speciesId).c_str(), verify.level, (verify.genderData[1] == GENDERLESS ? 'G' : (verify.genderData[1] == FEMALE ? 'F' : 'M')), NATURE_NAMES[verify.nature].c_str());
        for (int j = 0; j < 6; j++) {
            printf("%u-%u%c", verify.ivs[j][0], verify.ivs[j][1], j == 5 ? ' ' : '/');
        }
        printf("H:%u-%u W:%u-%u ", verify.height[0], verify.height[1], verify.weight[0], verify.weight[1]);
        printf("A:%u-%u ", verify.ability[0], verify.ability[1]);
        uint64_t seeds = getExpectedSeeds(verify);
        char specifier = ' ';
        double seedsDouble;
        if (seeds >= 1000000000) {
            specifier = 'B';
            seedsDouble = seeds/1000000000.0;
        } else if (seeds >= 1000000) {
            specifier = 'M';
            seedsDouble = seeds/1000000.0;
        } else if (seeds >= 1000) {
            specifier = 'K';
            seedsDouble = seeds/1000.0;
        } else {
            specifier = ' ';
            seedsDouble = (double)seeds;
        }
        int ivCount = getNumIVPermutations(verify.ivs);
        printf("ivs:%d seeds: %.01f%c genCost:%.1fK\n", ivCount, seedsDouble, specifier, getTheoreticalGeneratorSeeds(verify, slotGroup.getSlotRateSum()) / 1000.0);
    }
    printf("Enter the pokemon you would like to search: ");
    char c = fgetc(stdin);
    int index = c - '0';
    if (index != 0) {
        std::swap(pokemon[index], pokemon[0]);
    }

    SeedReversalContext reversalCtx = createReversalCtxForCUDA(numShinyRolls, slotGroup);
    reversalCtx.numPokemon = 4;
    for (int i = 0; i < 4; i++) {
        reversalCtx.pokemonVerifCtx[i] = pokemon[i];
    }
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t* outputBuffer = new uint64_t[4];
    int results = reversePokemonFromSingleMon(reversalCtx, 0, outputBuffer);
    for (int i = 0; i < results; i++) {
        printf("Group seed: %llx %llu\n", outputBuffer[i], outputBuffer[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Seed search took %.02fs\n", (double)duration.count() / 1000000);
    delete[] outputBuffer;
}

int main(int argc, char** argv){
    PokemonData::loadSpeciesNamesFromFile("resources/pokemonSpecies.txt");
    PokemonData::loadSpeciesDataFromFile("resources/pokemonData.txt");
    PokemonData::loadSpeciesEvolutionsFromFile("resources/pokemonEvolutions.txt");
    PokemonData::loadAbilityNamesFromFile("resources/pokemonAbilities.txt");
    PokemonData::loadTablesFromFile("resources/plaMMOTables.txt");
    
    // seedFind(argv[1]);
    reversal();
    // QApplication app(argc, argv);
    // MainWindow window;
    // if (argc > 1) {
    //     window.populateInputJSON(argv[1]);
    // }
    // window.show();
    // return app.exec();
}
