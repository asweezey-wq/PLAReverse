#include "pokemonData.hpp"
#include <sstream>

std::vector<std::string> PokemonData::m_speciesIdToName;
std::vector<SpeciesData> PokemonData::m_speciesIdToData;
std::unordered_map<std::string, uint32_t> PokemonData::m_speciesNameToId;
std::vector<std::string> PokemonData::m_abilityIdToName;
std::unordered_map<std::string, uint32_t> PokemonData::m_abilityNameToId;
std::unordered_map<uint64_t, PokemonSlotGroup> PokemonData::m_outbreakTables;

void PokemonData::loadSpeciesNamesFromFile(std::string filePath) {
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        fprintf(stderr, "Could not open species name file %s\n", filePath.c_str());
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

void PokemonData::loadSpeciesDataFromFile(std::string filePath) {
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        fprintf(stderr, "Could not open species data file %s\n", filePath.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string part;
        SpeciesData data;
        std::getline(ss, part, ',');
        data.index = std::stoul(part);
        for (int i = 0; i < 6; i++) {
            std::getline(ss, part, ',');
            data.baseStats[i] = (uint8_t)std::stoul(part);
        }
        std::getline(ss, part, ',');
        data.baseHeight = std::stoul(part);
        std::getline(ss, part, ',');
        data.baseWeight = std::stoul(part);
        std::getline(ss, part, ',');
        data.genderRatio = (uint8_t)std::stoul(part);
        std::getline(ss, part, ',');
        data.abilities[0] = std::stoul(part);
        std::getline(ss, part, ',');
        data.abilities[1] = std::stoul(part);
        if (data.index != m_speciesIdToData.size()) {
            fprintf(stderr, "Species %zu invalid data\n", m_speciesIdToData.size());
            exit(1);
        }
        m_speciesIdToData.push_back(data);
    }
}

void PokemonData::loadAbilityNamesFromFile(std::string filePath) {
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        fprintf(stderr, "Could not open ability name file %s\n", filePath.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        m_abilityIdToName.push_back(line);
    }
    for (int i = 0; i < m_abilityIdToName.size(); i++) {
        m_abilityNameToId[m_abilityIdToName[i]] = i;
    }
}

void PokemonData::loadTablesFromFile(std::string filePath) {
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

PokemonSlot PokemonData::parseSlotFromJson(json jsonObj) {
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

std::string PokemonEntity::toString() {
    auto data = PokemonData::getSpeciesData(m_species);
    std::stringstream ss;
    ss << "Pokemon: " << PokemonData::getSpeciesName(m_species) << "(" << m_species << ")" << std::endl;
    ss << "Lvl " << (uint32_t)m_level;
    ss << "\tShiny: " << (m_isShiny ? "YES": "NO");
    ss << "\tAlpha: " << (m_isAlpha ? "YES": "NO") << std::endl;
    ss << "Gender: " << GENDER_NAMES[m_gender];
    ss << "\tAbility: " << PokemonData::getAbilityName(data.abilities[m_ability]);
    ss << "\tNature: " << NATURE_NAMES[m_nature] << std::endl;
    ss << "EC: " << std::hex << m_encrConst;
    ss << "\tPID: " << std::hex << m_pid << std::endl;
    ss << "IVS: ";
    for (int i = 0; i < 6; i++) {
        ss << std::dec << (uint32_t)m_ivs[i];
        if (i != 5) {
            ss << "/";
        }
    }
    ss << std::endl;
    ss << "Height: " << (uint32_t)m_height;
    ss << "\tWeight: " << (uint32_t)m_weight << std::endl;
    return ss.str();
}

void PokemonSlotGroup::addSlot(PokemonSlot slot) {
    m_slots.emplace_back(slot);
    m_slotRateSum += slot.m_rate;
}

const PokemonSlot& PokemonSlotGroup::getSlot(float slotRng) const {
    for (const auto& slot : m_slots) {
        slotRng -= slot.m_rate;
        if (slotRng <= 0) {
            return slot;
        }
    }
    fprintf(stderr, "Invalid Slot RNG\n");
    exit(1); 
}