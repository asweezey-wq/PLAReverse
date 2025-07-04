#pragma once
#include "pokemonData.hpp"
#include <vector>
#include <unordered_map>

typedef std::unordered_map<uint32_t, int> ShinyRollData;

class OutbreakSpawner {
public:
    OutbreakSpawner(uint64_t table, ShinyRollData rollData);
    OutbreakSpawner(const PokemonSlotGroup& slotGroup, ShinyRollData rollData) : m_slotGroup(slotGroup), m_shinyRollData(rollData) {}

    PokemonEntity createPokemon(uint64_t seed);
    uint64_t spawnPokemon(uint64_t seed, int count, std::vector<PokemonEntity>& outputEntities, int ghosts = 0);
    static uint64_t advanceSeed(uint64_t seed, int count);
private:
    uint64_t m_tableId{0};
    const PokemonSlotGroup& m_slotGroup;
    ShinyRollData m_shinyRollData;
};

void generatePokemon(uint64_t seed, int shinyRolls, uint32_t species, int numPerfectIVs, PokemonEntity& pokemonEntity);
