#pragma once
#include <cstdint>
#include <utility>
#include <vector>

struct PokemonSlot {
    uint32_t m_species{0};
    uint32_t m_rate{0};
    bool m_isAlpha{false};
    std::pair<uint8_t, uint8_t> m_levelRange{0,1};
    uint8_t m_numPerfectIVs{0};
};

class PokemonSlotGroup {
public:
    void addSlot(PokemonSlot slot);
    const PokemonSlot& getSlot(float slotRng) const;

    uint32_t getSlotRateSum() const { return m_slotRateSum; }
private:
    std::vector<PokemonSlot> m_slots;
    uint32_t m_slotRateSum{0};
};