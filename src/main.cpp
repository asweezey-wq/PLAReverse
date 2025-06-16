#include "pokemonGenerator.hpp"
#include "pokemonEntity.hpp"
#include "outbreakSpawner.hpp"
#include <iostream>

int main(int, char**){
    PokemonData::loadSpeciesFromFile("resources/pokemonSpecies.txt");
    PokemonData::loadTablesFromFile("resources/mmo_tables.txt");
    PokemonEntity entity;
    uint64_t seed = 0;
    PokemonSlotGroup group;
    group.addSlot(PokemonSlot{
        .m_species = PokemonData::getSpeciesID("Tentacool"),
        .m_isAlpha = false,
        .m_rate = 100,
        .m_numPerfectIVs = 0,
        .m_levelRange = {0, 1}
    });
    group.addSlot(PokemonSlot{
        .m_species = PokemonData::getSpeciesID("Tentacool"),
        .m_isAlpha = true,
        .m_rate = 1,
        .m_numPerfectIVs = 3,
        .m_levelRange = {0, 1}
    });
    OutbreakSpawner spawner(group);
    std::vector<PokemonEntity> pokemon;
    seed = spawner.spawnPokemon(seed, 4, pokemon);
    for (auto mon : pokemon) {
        std::cout << mon.toString() << std::endl;
    }
}
