#pragma once
#include "pokemonSlot.hpp"
#include "pokemonEntity.hpp"
#include <vector>

class OutbreakSpawner {
public:
    OutbreakSpawner(uint64_t table);
    OutbreakSpawner(const PokemonSlotGroup& slotGroup) : m_slotGroup(slotGroup) {}

    PokemonEntity createPokemon(uint64_t seed);
    uint64_t spawnPokemon(uint64_t seed, int count, std::vector<PokemonEntity>& outputEntities);
private:
    uint64_t m_tableId{0};
    const PokemonSlotGroup& m_slotGroup;
};