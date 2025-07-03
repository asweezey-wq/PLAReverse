#include "permutations.hpp"
#include "outbreakSpawner.hpp"

PermutationsManager::PermutationsManager(uint64_t seed, uint64_t primaryTableID, int primaryCount, uint64_t secondaryTableID, int secondaryCount, ShinyRollData& shinyRolls) {
    if (primaryTableID) {
        auto &primaryTable = PokemonData::getSlotGroupTable(primaryTableID);
        m_primarySpawner = new OutbreakSpawner(primaryTable, shinyRolls);
    }
    if (secondaryTableID) {
        auto& secondaryTable = PokemonData::getSlotGroupTable(secondaryTableID);
        m_secondarySpawner = new OutbreakSpawner(secondaryTable, shinyRolls);
    }
    m_startingState = OutbreakState(seed, m_primarySpawner, primaryCount, m_secondarySpawner, secondaryCount);
}

std::vector<PermutationResult> PermutationsManager::findPermutations() {
    std::vector<PermutationUserAction> actionChain;
    std::vector<PermutationResult> results;
    for (auto& p : m_startingState.respawn()) {
        results.emplace_back(actionChain, p);
    }
    while(!m_startingState.hasEnded()) {
        PermutationUserAction action;
        if (m_startingState.m_isFirstWave && m_startingState.m_primaryCount == 0) {
            action = {
                CLEAR_REMAINING,
                0
            };
        } else {
            action = {
                CATCH,
                1
            };
        }
        actionChain.push_back(action);
        m_startingState.doUserAction(action);
        auto pokemon = m_startingState.respawn();
        for (auto& p : pokemon) {
            results.emplace_back(actionChain, p);
        }
    }
    return results;
}

OutbreakState::OutbreakState(uint64_t groupSeed, OutbreakSpawner* primarySpawner, int primaryCount, OutbreakSpawner* secondarySpawner, int secondaryCount)
    : m_groupSeed(groupSeed), m_primarySpawner(primarySpawner), m_secondarySpawner(secondarySpawner), m_primaryCount(primaryCount), m_secondaryCount(secondaryCount) {
}

void OutbreakState::doUserAction(PermutationUserAction action) {
    switch(action.actionType) {
        case CATCH:
        case GHOST:
            m_pokemonSpawned -= action.count;
            break;
        case CLEAR_REMAINING:
            m_pokemonSpawned = 0;
            break;
    }
}

std::vector<PokemonEntity> OutbreakState::respawn() {
    if (hasFirstWaveEnded()) {
        m_isFirstWave = false;
    }
    int capacity = 4 - m_pokemonSpawned;
    int count = std::min(m_isFirstWave ? m_primaryCount : m_secondaryCount, capacity);
    std::vector<PokemonEntity> pokemon;
    if (count == 0) {
        return pokemon;
    }
    uint64_t nextSeed;
    if (m_isFirstWave) {
        nextSeed = m_primarySpawner->spawnPokemon(m_groupSeed, count, pokemon);
        m_primaryCount -= count;
    } else {
        nextSeed = m_secondarySpawner->spawnPokemon(m_groupSeed, count, pokemon);
        m_secondaryCount -= count;
    }
    m_pokemonSpawned += count;
    m_groupSeed = nextSeed;
    return pokemon;
}
