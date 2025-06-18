#include "pokemonGenerator.hpp"
#include "pokemonEntity.hpp"
#include "outbreakSpawner.hpp"
#include "seedReversal.hpp"
#include "pokemon_cuda.h"
#include <iostream>
#include <unordered_set>

int main(int, char**){
    PokemonData::loadSpeciesFromFile("resources/pokemonSpecies.txt");
    PokemonData::loadTablesFromFile("resources/mmo_tables.txt");
    PokemonEntity entity;
    uint64_t seed = 0;
    PokemonSlotGroup group;
    group.addSlot(PokemonSlot{
        .m_species = PokemonData::getSpeciesID("Tentacool"),
        .m_rate = 100,
        .m_isAlpha = false,
        .m_levelRange = {0, 1},
        .m_numPerfectIVs = 0
    });
    group.addSlot(PokemonSlot{
        .m_species = PokemonData::getSpeciesID("Tentacool"),
        .m_rate = 1,
        .m_isAlpha = true,
        .m_levelRange = {0, 1},
        .m_numPerfectIVs = 3
    });
    OutbreakSpawner spawner(group);
    std::vector<PokemonEntity> pokemon;
    seed = spawner.spawnPokemon(seed, 4, pokemon);
    // for (auto mon : pokemon) {
    //     std::cout << mon.toString() << std::endl;
    // }
    entity = pokemon[0];
    std::cout << entity.toString() << std::endl;

    printf("Hello from Host!\n");
    SeedReverseConstantsFlexible reversal = createReversalStructForCUDA(32, entity.m_ivs);
    for (int i = 0; i < 6; i++) {
        printf("IV%d: %u %u\n", i, reversal.ivs[i][0], reversal.ivs[i][1]);
    }
    SeedVerifyConstantsFlexible verify = createVerifyStructForCUDA(entity);
    printf("Ability: %u %u\n", verify.ability[0], verify.ability[1]);
    printf("Gender: ratio:%llu gender:%llu\n", verify.genderData[0], verify.genderData[1]);
    printf("Nature: %u\n", verify.nature);
    printf("Height: %llu %llu\n", verify.height[0], verify.height[1]);
    printf("Weight: %llu %llu\n", verify.weight[0], verify.weight[1]);
    uint64_t *outputBuffer = new uint64_t[OUTPUT_BUFFER_SIZE];
    int foundSeeds = ivWrapperFlexible(reversal, verify, outputBuffer);
    std::unordered_set<uint64_t> seeds;
    for (int i = 0; i < foundSeeds; i++) {
        // printf("Seed: %llx\n", outputBuffer[i]);
        seeds.insert(outputBuffer[i]);
    }
    printf("Total seeds %d unique: %zd\n", foundSeeds, seeds.size());
    printf("Found true seed: %d\n", seeds.contains(0x75229d6a5b82a2b1));
    std::cout << "First seed verify: " << verifySeed(*seeds.begin(), 32, entity) << std::endl;
    delete[] outputBuffer;
}
