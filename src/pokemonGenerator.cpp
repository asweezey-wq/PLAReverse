#include "pokemonGenerator.hpp"

void generatePokemon(uint64_t seed, int shinyRolls, PokemonEntity& entity) {
    Xoroshiro128PlusRNG rng(seed);
    entity.m_encrConst = rng.nextUint32();
    uint32_t fakeTID = rng.nextUint32();
    for (int i = 0; i < shinyRolls; i++) {
        entity.m_pid = rng.nextUint32();
        uint32_t shinyValue = shinyXor(entity.m_pid, fakeTID);
        bool isShiny = shinyValue < 16;
        if (isShiny) {
            entity.m_isShiny = true;
            break;
        }
    }
    const uint8_t IV_UNSET = 255;
    const uint8_t IV_MAX = 31;
    for (int i = 0; i < 6; i++) {
        entity.m_ivs[i] = IV_UNSET;
    }
    int numPerfectIVs = 0;
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
    uint8_t genderRatio = 128;
    uint8_t genderRNG = (uint8_t)rng.nextWithMax(253) + 1;
    entity.m_gender = genderRNG < genderRatio ? FEMALE : MALE;

    entity.m_nature = (uint8_t)rng.nextWithMax(25);

    entity.m_height = (uint8_t)(rng.nextWithMax(0x81) + rng.nextWithMax(0x80));
    entity.m_weight = (uint8_t)(rng.nextWithMax(0x81) + rng.nextWithMax(0x80));
}

uint32_t shinyXor(uint32_t pid, uint32_t tid) {
    uint32_t value = pid ^ tid;
    return (value ^ (value >> 16)) & 0xffff;
}