#include <iostream>
#include "pokemonGenerator.hpp"

int main(int, char**){
    PokemonEntity entity;
    uint64_t seed = 0;
    Xoroshiro128PlusRNG rng(seed);
    for (int i = 0; i < 4; i++) {
        uint64_t slotSeed = rng.next();
        uint64_t alphaSeed = rng.next();
        generatePokemonFromSlot(slotSeed, 32, entity);
        std::cout << entity.toString() << std::endl;
    }
}
