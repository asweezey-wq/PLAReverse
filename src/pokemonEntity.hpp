#pragma once
#include <cstdint>
#include <string>

enum PokemonGender {
    MALE,
    FEMALE,
    GENDERLESS,
};

class PokemonEntity {
public:
    uint16_t m_species{0};
    uint8_t m_form{0};
    uint8_t m_level{0};
    uint32_t m_encrConst{0};
    uint32_t m_pid{0};
    uint8_t m_ivs[6]{0};
    uint8_t m_weight{0};
    uint8_t m_height{0};
    uint8_t m_ability{0};
    uint8_t m_nature{0};
    PokemonGender m_gender{GENDERLESS};

    bool m_isShiny{false};
    bool m_isAlpha{false};

    std::string toString();
};