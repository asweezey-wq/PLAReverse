#pragma once
#include "json.hpp"
#include "pokemonSlot.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>

using json = nlohmann::json;

class PokemonData {
public:
    static void loadSpeciesFromFile(std::string filePath) {
        std::ifstream ifs(filePath);
        if (!ifs.is_open()) {
            fprintf(stderr, "Could not open species file %s\n", filePath.c_str());
            exit(1);
        }
        std::string line;
        while (std::getline(ifs, line)) {
            m_speciesIdToName.push_back(line);
        }
        for (int i = 0; i < m_speciesIdToName.size(); i++) {
            m_speciesNameToId[m_speciesIdToName[i]] = i;
        }
    }

    static void loadTablesFromFile(std::string filePath) {
        std::ifstream ifs(filePath);
        if (!ifs.is_open()) {
            fprintf(stderr, "Could not open tables file %s\n", filePath.c_str());
            exit(1);
        }
        json jsonData = json::parse(ifs);
        ifs.close();

        if (!jsonData.is_object()) {
            fprintf(stderr, "Tables file should be top-level JSON object\n");
        }

        for (auto [tableID, tableValue] : jsonData.items()) {
            uint64_t tableIDInt = std::stoull(tableID, nullptr, 16);
            if (!tableValue.is_array()) {
                fprintf(stderr, "%s: table value not array\n", tableID.c_str());
                exit(1);
            }
            PokemonSlotGroup& slotGroup = m_outbreakTables[tableIDInt];
            for (auto slotJson : tableValue) {
                slotGroup.addSlot(parseSlotFromJson(slotJson));
            }
        }
    }

    static PokemonSlot parseSlotFromJson(json jsonObj) {
        PokemonSlot slot;
        slot.m_rate = jsonObj["slot"].get<uint32_t>();
        slot.m_isAlpha = jsonObj["alpha"].get<bool>();
        std::string pokemonName = jsonObj["name"];
        uint8_t form = 0;
        for (int i = 0; i < pokemonName.size(); i++) {
            if (pokemonName[i] == '-') {
                std::string formSubstr = pokemonName.substr(i, pokemonName.size() - i);
                form = std::stoi(formSubstr);
                pokemonName = pokemonName.substr(0, i);
                break;
            }
        }
        slot.m_species = getSpeciesID(pokemonName);
        json levels = jsonObj["level"];
        if (!levels.is_array() || levels.size() != 2) {
            fprintf(stderr, "%s: expected levels to be array of length 2\n", pokemonName.c_str());
            exit(1);
        }
        slot.m_levelRange = {levels[0].get<uint8_t>(), levels[1].get<uint8_t>()};
        slot.m_numPerfectIVs = jsonObj["ivs"].get<uint8_t>();
        return slot;
    }

    static uint32_t getSpeciesID(std::string name) { 
        if (!m_speciesNameToId.contains(name)) {
            fprintf(stderr, "Could not find Pokemon %s\n", name.c_str());
            exit(1);
        }
        return m_speciesNameToId.at(name); 
    }

    static const std::string getSpeciesName(size_t id) {
        if (id >= m_speciesIdToName.size()) {
            fprintf(stderr, "Pokemon Species ID %ju out of bounds\n", id);
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