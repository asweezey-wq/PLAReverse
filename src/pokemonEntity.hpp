#pragma once
#include "json.hpp"
#include "pokemonSlot.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

using json = nlohmann::json;

struct SpeciesData {
    uint32_t index;
    uint8_t baseStats[6];
    uint32_t baseHeight;
    uint32_t baseWeight;
    uint8_t genderRatio;
    uint32_t abilities[2];
};

class PokemonData {
public:
    static void loadSpeciesNamesFromFile(std::string filePath);
    static void loadSpeciesDataFromFile(std::string filePath);
    static void loadAbilityNamesFromFile(std::string filePath);
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

    static const SpeciesData& getSpeciesData(uint32_t id) {
        if (id >= m_speciesIdToData.size()) {
            fprintf(stderr, "Pokemon Species ID %u out of bounds\n", id);
            exit(1);
        }
        return m_speciesIdToData.at(id); 
    }

    static uint32_t getAbilityID(std::string name) { 
        if (!m_abilityNameToId.contains(name)) {
            fprintf(stderr, "Could not find Pokemon %s\n", name.c_str());
            exit(1);
        }
        return m_abilityNameToId.at(name); 
    }

    static const std::string getAbilityName(uint32_t id) {
        if (id >= m_abilityIdToName.size()) {
            fprintf(stderr, "Pokemon Species ID %u out of bounds\n", id);
            exit(1);
        }
        return m_abilityIdToName.at(id); 
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
    static std::vector<SpeciesData> m_speciesIdToData;
    static std::unordered_map<std::string, uint32_t> m_speciesNameToId;
    static std::vector<std::string> m_abilityIdToName;
    static std::unordered_map<std::string, uint32_t> m_abilityNameToId;

    static std::unordered_map<uint64_t, PokemonSlotGroup> m_outbreakTables;
};

enum PokemonGender {
    MALE,
    FEMALE,
    GENDERLESS,
};

class PokemonEntity {
public:
    uint32_t m_species{0};
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

    uint64_t m_genSeed{0};

    std::string toString();
};