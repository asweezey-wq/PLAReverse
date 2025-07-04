#include "outbreakSpawner.hpp"
#include "seedReversal.hpp"
#include "pokemonCuda.hpp"
#include "gameInference.hpp"
#include "xoroshiro.hpp"
#include "permutations.hpp"
#include "ui/MainWindow.h"
#include <iostream>
#include <QApplication>

void permutationTesting() {
    ShinyRollData shinyRolls;
    shinyRolls[PokemonData::getSpeciesID("Murkrow")] = 17;
    shinyRolls[PokemonData::getSpeciesID("Honchkrow")] = 17;
    PermutationsManager manager(10198543621818505482, 0x039A6B2692D9CA51, 10, 0xC80EF3CD75F015DD, 7, shinyRolls);
    auto results = manager.findPermutations();
}

void sizeTesting() {
    std::vector<ObservedSizeInstance> sizes;
    sizes.emplace_back(PokemonData::getSpeciesID("Machop"), 32, 42.5);
    sizes.emplace_back(PokemonData::getSpeciesID("Machoke"), 61, 153.6);
    sizes.emplace_back(PokemonData::getSpeciesID("Machamp"), 65, 283.1);
    auto pairs = calculateSizePairs(true, sizes);
    // pairs.clear();
    // pairs.emplace_back(145, 103);
    for (auto& pair : pairs) {
        float dispHeight, dispWeight;
        getDisplaySize(PokemonData::getSpeciesData(PokemonData::getSpeciesID("Machop")), true, (uint8_t)pair.first, (uint8_t)pair.second, dispHeight, dispWeight);
        printf("%u %u %.1f %.2f\n", pair.first, pair.second, dispHeight, dispWeight);
        getDisplaySize(PokemonData::getSpeciesData(PokemonData::getSpeciesID("Machoke")), true, (uint8_t)pair.first, (uint8_t)pair.second, dispHeight, dispWeight);
        printf("%u %u %.1f %.2f\n", pair.first, pair.second, dispHeight, dispWeight);
        getDisplaySize(PokemonData::getSpeciesData(PokemonData::getSpeciesID("Machamp")), true, (uint8_t)pair.first, (uint8_t)pair.second, dispHeight, dispWeight);
        printf("%u %u %.1f %.2f\n", pair.first, pair.second, dispHeight, dispWeight);
    }
}

void reversal() {
    ShinyRollData shinyRolls;
    uint64_t trueGroupSeed = 15769719563737179578;
    uint64_t slotGroupNumber = 0xC44A139615D5807E;
    uint32_t species = PokemonData::getSpeciesID("Machop");
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
            std::cout << entity.toString() << std::endl;
        }
    }

    uint32_t searchIndex = 1;
    if (searchIndex != 0) {
        std::swap(groupPokemon[searchIndex], groupPokemon[0]);
    }

    SeedReversalContext reversalCtx = createReversalCtxForCUDA(shinyRolls[species], slotGroup);
    reversalCtx.numPokemon = 4;
    for (int i = 0; i < 4; i++) {
        reversalCtx.pokemonVerifCtx[i] = groupPokemon[i];
    }
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    pairs.emplace_back(groupPokemon[0].height[0], groupPokemon[0].weight[0]);
    createSizePairsForCUDA(pairs, reversalCtx.primarySizePairs);

    uint64_t* outputBuffer = new uint64_t[4];
    int foundSeeds = reversePokemonFromSingleMon(reversalCtx, 0, outputBuffer);
    delete[] outputBuffer;
}

void seedFind(const char* filePath) {
    std::vector<PokemonVerificationContext> pokemon;
    std::vector<std::vector<ObservedSizeInstance>> sizes;
    int numShinyRolls;
    uint64_t tableID;
    parseJSONMMOEncounter(std::string(filePath), pokemon, sizes, numShinyRolls, tableID);
    auto slotGroup = PokemonData::getSlotGroupTable(tableID);
    for (int i = 0; i < 4; i++) {
        auto& verify = pokemon[i];
        auto& data = PokemonData::getSpeciesData(verify.speciesId);
        auto sizePairs = calculateSizePairs(true, sizes[i]);
        printf("%d: %s %u %c %s ", i, PokemonData::getSpeciesName(verify.speciesId).c_str(), verify.level, (verify.genderData[1] == GENDERLESS ? 'G' : (verify.genderData[1] == FEMALE ? 'F' : 'M')), NATURE_NAMES[verify.nature].c_str());
        for (int j = 0; j < 6; j++) {
            printf("%u-%u%c", verify.ivs[j][0], verify.ivs[j][1], j == 5 ? ' ' : '/');
        }
        printf("H:%u-%u W:%u-%u pairs:%zu ", verify.height[0], verify.height[1], verify.weight[0], verify.weight[1], sizePairs.size());
        printf("A:%u-%u ", verify.ability[0], verify.ability[1]);
        uint64_t seeds = getExpectedSeedsWithSizePairs(verify, sizePairs);
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
        printf("ivs:%d seeds: %.01f%c genCost:%.2f/%.2fK\n", ivCount, seedsDouble, specifier, getTheoreticalGeneratorSeeds(verify, slotGroup.getSlotRateSum()) / 1000.0, getTheoreticalGeneratorSeedsWithSizePairs(verify, sizePairs, slotGroup.getSlotRateSum()) / 1000.0);
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
    auto sizePairs = calculateSizePairs(true, sizes[index]);
    createSizePairsForCUDA(sizePairs, reversalCtx.primarySizePairs);
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
    // reversal();
    // sizeTesting();
    // permutationTesting();
    QApplication app(argc, argv);
    MainWindow window;
    if (argc > 1) {
        window.populateInputJSON(argv[1]);
    }
    window.show();
    return app.exec();
}
