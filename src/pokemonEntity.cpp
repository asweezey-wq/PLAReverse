#include "pokemonEntity.hpp"
#include "pokemonData.hpp"
#include <sstream>

std::string PokemonEntity::toString() {
    std::stringstream ss;
    ss << "Pokemon:" << std::endl;
    ss << "Species: " << PokemonData::getSpeciesName(m_species) << "(" << m_species << ")" << std::endl;
    ss << "Shiny: " << (m_isShiny ? "YES": "NO") << std::endl;
    ss << "Alpha: " << (m_isAlpha ? "YES": "NO") << std::endl;
    ss << "Level: " << (uint32_t)m_level << std::endl;
    ss << "Gender: " << (uint32_t)m_gender << std::endl;
    ss << "EC: " << std::hex << m_encrConst << std::endl;
    ss << "PID: " << std::hex << m_pid << std::endl;
    ss << "IVS: ";
    for (int i = 0; i < 6; i++) {
        ss << std::dec << (uint32_t)m_ivs[i];
        if (i != 5) {
            ss << "/";
        }
    }
    ss << std::endl;
    ss << "Ability: " << (uint32_t)m_ability << std::endl;
    ss << "Nature: " << (uint32_t)m_nature << std::endl;
    ss << "Height: " << (uint32_t)m_height << std::endl;
    ss << "Weight: " << (uint32_t)m_weight << std::endl;
    return ss.str();
}