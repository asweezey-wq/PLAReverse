#pragma once
#include "json.hpp"
#include "pokemonSlot.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

using json = nlohmann::json;

class PokemonData {
public:
    static void loadSpeciesFromFile(std::string filePath);

    static void loadTablesFromFile(std::string filePath);

    static PokemonSlot parseSlotFromJson(json jsonObj);

    static uint32_t getSpeciesID(std::string name) { 
        if (!m_speciesNameToId.contains(name)) {
            fprintf(stderr, "Could not find Pokemon %s\n", name.c_str());
            exit(1);
        }
        return m_speciesNameToId.at(name); 
    }

    static const std::string getSpeciesName(uint32_t id) {
        if (id >= m_speciesIdToName.size()) {
            fprintf(stderr, "Pokemon Species ID %u out of bounds\n", id);
            exit(1);
        }
        return m_speciesIdToName.at(id); 
    }

    static const PokemonSlotGroup& getSlotGroupTable(uint64_t table) {
        if (!m_outbreakTables.contains(table)) {
            fprintf(stderr, "Could not find Outbreak Table %llx\n", table);
            exit(1);
        }
        return m_outbreakTables.at(table); 
    }
private:
    static std::vector<std::string> m_speciesIdToName;
    static std::unordered_map<std::string, uint32_t> m_speciesNameToId;

    static std::unordered_map<uint64_t, PokemonSlotGroup> m_outbreakTables;
};

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