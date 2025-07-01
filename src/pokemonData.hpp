#pragma once
#include "json.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

using json = nlohmann::json;

const std::string NATURE_NAMES[] = {
    "Hardy", "Lonely", "Brave", "Adamant", "Naughty",
    "Bold", "Docile", "Relaxed", "Impish", "Lax",
    "Timid", "Hasty", "Serious", "Jolly", "Naive",
    "Modest", "Mild", "Quiet", "Bashful", "Rash",
    "Calm", "Gentle", "Sassy", "Careful", "Quirky"
};

enum PokemonGender : uint8_t {
    MALE,
    FEMALE,
    GENDERLESS,
};

const std::string GENDER_NAMES[] = {
    "MALE", "FEMALE", "GENDERLESS",
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
    uint8_t m_gender{GENDERLESS};

    bool m_isShiny{false};
    bool m_isAlpha{false};

    uint64_t m_pokemonSeed{0};
    uint64_t m_generatorSeed{0};

    std::string toString();
};

struct SpeciesData {
    uint32_t index;
    uint8_t baseStats[6];
    uint32_t baseHeight;
    uint32_t baseWeight;
    uint8_t genderRatio;
    uint32_t abilities[2];
};

// Contains information about a pokemon that a generator can spawn
struct PokemonSlot {
    uint32_t m_species{0};
    uint32_t m_rate{0};
    bool m_isAlpha{false};
    std::pair<uint8_t, uint8_t> m_levelRange{0,1};
    uint8_t m_numPerfectIVs{0};
};

// Contains all slots for a particular outbreak
class PokemonSlotGroup {
public:
    void addSlot(PokemonSlot slot);
    const PokemonSlot& getSlot(float slotRng) const;
    const PokemonSlot& getSlotFromIndex(int index) const { return m_slots[index]; }
    size_t numSlots() const { return m_slots.size(); }

    uint32_t getSlotRateSum() const { return m_slotRateSum; }
private:
    std::vector<PokemonSlot> m_slots;
    uint32_t m_slotRateSum{0};
};


class PokemonData {
public:
    static void loadSpeciesNamesFromFile(std::string filePath);
    static void loadSpeciesDataFromFile(std::string filePath);
    static void loadSpeciesEvolutionsFromFile(std::string filePath);
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

    static bool speciesHasEvolutions(uint32_t id) {
        return m_speciesEvolutions.contains(id);
    }

    static const std::vector<uint32_t> getSpeciesEvolutions(uint32_t id) {
        if (!m_speciesEvolutions.contains(id)) {
            fprintf(stderr, "Pokemon Species ID %u out of bounds\n", id);
            exit(1);
        }
        return m_speciesEvolutions.at(id); 
    }

    static uint32_t getAbilityID(std::string name) { 
        if (!m_abilityNameToId.contains(name)) {
            fprintf(stderr, "Could not find ability %s\n", name.c_str());
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

    static const std::vector<std::string> getAllPokemonNames() {
        return m_speciesIdToName;
    }

    static const std::unordered_map<uint64_t, PokemonSlotGroup> getOutbreakTables() {
        return m_outbreakTables;
    }
private:
    static std::vector<std::string> m_speciesIdToName;
    static std::vector<SpeciesData> m_speciesIdToData;
    static std::unordered_map<std::string, uint32_t> m_speciesNameToId;
    static std::unordered_map<uint32_t, std::vector<uint32_t>> m_speciesEvolutions;
    static std::vector<std::string> m_abilityIdToName;
    static std::unordered_map<std::string, uint32_t> m_abilityNameToId;

    static std::unordered_map<uint64_t, PokemonSlotGroup> m_outbreakTables;
};