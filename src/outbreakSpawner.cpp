#include "outbreakSpawner.hpp"
#include "pokemonEntity.hpp"
#include "pokemonGenerator.hpp"
#include "xoroshiro.hpp"

OutbreakSpawner::OutbreakSpawner(uint64_t table) : m_tableId(table), m_slotGroup(PokemonData::getSlotGroupTable(table)) {

}

PokemonEntity OutbreakSpawner::createPokemon(uint64_t seed, int numShinyRolls) {
    Xoroshiro128PlusRNG slotRng(seed);
    float slotRoll = slotRng.nextFloat((float)m_slotGroup.getSlotRateSum());
    const PokemonSlot& slot = m_slotGroup.getSlot(slotRoll);
    auto data = PokemonData::getSpeciesData(slot.m_species);
    uint64_t pokemonGenSeed = slotRng.next();
    PokemonEntity entity;
    entity.m_species = slot.m_species;
    entity.m_isAlpha = slot.m_isAlpha;
    generatePokemon(pokemonGenSeed, numShinyRolls, data.genderRatio, entity);
    uint8_t levelMin = slot.m_levelRange.first;
    uint8_t levelMax = slot.m_levelRange.second;
    uint8_t delta = 1 + levelMax - levelMin;
    uint8_t rngDelta = delta > 1 ? (uint8_t)slotRng.nextWithMax(delta) : slotRng.nextBool();
    entity.m_level = levelMin + rngDelta;
    return entity;
}

uint64_t OutbreakSpawner::spawnPokemon(uint64_t seed, int count, int numShinyRolls, std::vector<PokemonEntity>& outputEntities) {
    Xoroshiro128PlusRNG rng(seed);
    for (int i = 0; i < count; i++) {
        uint64_t slotSeed = rng.next();
        uint64_t alphaSeed = rng.next();
        outputEntities.push_back(createPokemon(slotSeed, numShinyRolls));
    }
    return rng.next();
}