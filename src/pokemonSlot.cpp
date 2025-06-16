#include "pokemonSlot.hpp"

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