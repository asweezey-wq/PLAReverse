#include "outbreakSpawner.hpp"
#include "xoroshiro.hpp"

OutbreakSpawner::OutbreakSpawner(uint64_t table, ShinyRollData rollData) : m_tableId(table), m_slotGroup(PokemonData::getSlotGroupTable(table)), m_shinyRollData(rollData) {

}

PokemonEntity OutbreakSpawner::createPokemon(uint64_t seed) {
    Xoroshiro128PlusRNG slotRng(seed);
    float slotRoll = slotRng.nextFloat((float)m_slotGroup.getSlotRateSum());
    const PokemonSlot& slot = m_slotGroup.getSlot(slotRoll);
    uint64_t pokemonGenSeed = slotRng.next();
    PokemonEntity entity;
    entity.m_generatorSeed = seed;
    entity.m_isAlpha = slot.m_isAlpha;
    int numShinyRolls = m_shinyRollData.contains(slot.m_species) ? m_shinyRollData[slot.m_species] : 1;
    generatePokemon(pokemonGenSeed, numShinyRolls, slot.m_species, (int)slot.m_numPerfectIVs, entity);
    uint8_t levelMin = slot.m_levelRange.first;
    uint8_t levelMax = slot.m_levelRange.second;
    uint8_t delta = 1 + levelMax - levelMin;
    uint8_t rngDelta = delta > 1 ? (uint8_t)slotRng.nextWithMax(delta) : slotRng.nextBool();
    entity.m_level = levelMin + rngDelta;
    return entity;
}

uint64_t OutbreakSpawner::spawnPokemon(uint64_t seed, int count, std::vector<PokemonEntity>& outputEntities, int ghosts) {
    Xoroshiro128PlusRNG rng(seed);
    for (int i = 0; i < count; i++) {
        uint64_t slotSeed = rng.next();
        uint64_t alphaSeed = rng.next();
        if (i < ghosts)
            continue;
        outputEntities.push_back(createPokemon(slotSeed));
    }
    return rng.next();
}

uint64_t OutbreakSpawner::advanceSeed(uint64_t seed, int count) {
    Xoroshiro128PlusRNG rng(seed);
    for (int i = 0; i < count; i++) {
        uint64_t slotSeed = rng.next();
        uint64_t alphaSeed = rng.next();
    }
    return rng.next();
}

uint32_t shinyXor(uint32_t pid, uint32_t tid) {
    uint32_t value = pid ^ tid;
    return (value ^ (value >> 16)) & 0xffff;
}

void generatePokemon(uint64_t seed, int shinyRolls, uint32_t species, int numPerfectIVs, PokemonEntity& entity) {
    auto data = PokemonData::getSpeciesData(species);
    entity.m_species = species;
    Xoroshiro128PlusRNG rng(seed);
    entity.m_pokemonSeed = seed;
    entity.m_encrConst = rng.nextUint32();
    entity.m_shinyRolls = shinyRolls;
    uint32_t fakeTID = rng.nextUint32();
    for (int i = 0; i < shinyRolls; i++) {
        entity.m_pid = rng.nextUint32();
        uint32_t shinyValue = shinyXor(entity.m_pid, fakeTID);
        bool isShiny = shinyValue < 16;
        if (isShiny) {
            entity.m_isShiny = true;
            entity.m_shinyRolls = i + 1;
            break;
        }
    }
    const uint8_t IV_UNSET = 255;
    const uint8_t IV_MAX = 31;
    for (int i = 0; i < 6; i++) {
        entity.m_ivs[i] = IV_UNSET;
    }
    for (int i = 0; i < numPerfectIVs; i++) {
        // Find a random index that has not been set yet
        uint64_t index;
        do {
            index = rng.nextWithMax(6);
        } while (entity.m_ivs[index] != IV_UNSET);
        entity.m_ivs[index] = IV_MAX;
    }
    for (int i = 0; i < 6; i++) {
        if (entity.m_ivs[i] == IV_UNSET) {
            entity.m_ivs[i] = (uint8_t)rng.nextWithMax(32);
        }
    }
    entity.m_ability = (uint8_t)rng.nextWithMax(2);
    uint8_t genderRNG = (uint8_t)rng.nextWithMax(253) + 1;
    entity.m_gender = genderRNG < data.genderRatio ? FEMALE : MALE;

    entity.m_nature = (uint8_t)rng.nextWithMax(25);

    entity.m_height = (uint8_t)(rng.nextWithMax(0x81) + rng.nextWithMax(0x80));
    entity.m_weight = (uint8_t)(rng.nextWithMax(0x81) + rng.nextWithMax(0x80));
}
