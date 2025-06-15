#include "pokemonEntity.hpp"
#include "xoroshiro.hpp"

void generatePokemon(uint64_t seed, int shinyRolls, PokemonEntity& pokemonEntity);
void generatePokemonFromSlot(uint64_t seed, int shinyRolls, PokemonEntity& pokemonEntity);

uint32_t shinyXor(uint32_t pid, uint32_t tid);