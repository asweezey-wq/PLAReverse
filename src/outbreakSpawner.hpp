#pragma once
#include "pokemonData.hpp"
#include <vector>

class OutbreakSpawner {
public:
    OutbreakSpawner(uint64_t table);
    OutbreakSpawner(const PokemonSlotGroup& slotGroup) : m_slotGroup(slotGroup) {}

    PokemonEntity createPokemon(uint64_t seed, int numShinyRolls);
    uint64_t spawnPokemon(uint64_t seed, int count, int numShinyRolls, std::vector<PokemonEntity>& outputEntities);
private:
    uint64_t m_tableId{0};
    const PokemonSlotGroup& m_slotGroup;
};

void generatePokemon(uint64_t seed, int shinyRolls, uint32_t species, int numPerfectIVs, PokemonEntity& pokemonEntity);
