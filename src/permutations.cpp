#include "permutations.hpp"
#include "outbreakSpawner.hpp"
#include <iostream>

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
    int index = 0;
    for (auto& p : m_startingState.respawn()) {
        results.emplace_back(actionChain, p);
    }
    findPermutationsRecursive(m_startingState, actionChain, results);
    return results;
}

void PermutationsManager::findPermutationsRecursive(OutbreakState state, const std::vector<PermutationUserAction>& actionChain, std::vector<PermutationResult>& results) {
    std::vector<PermutationUserAction> possibleActions;
    if (state.m_isFirstWave) {
        if (m_secondarySpawner && state.m_primaryCount == 0 && state.m_pokemonAlive > 0) {
            possibleActions.emplace_back(CLEAR_REMAINING, 0);
            int possibleGhostSlots = 3 - state.m_ghosts;
            for (int i = 1; i <= possibleGhostSlots; i++) {
                possibleActions.emplace_back(GHOST, i);
            }
        } else {
            for (int i = 1; i <= state.m_pokemonAlive; i++) {
                possibleActions.emplace_back(CATCH, i);
            }
        }
    } else if (state.m_secondaryCount > 0) {
        for (int i = 1; i <= state.m_pokemonAlive; i++) {
            possibleActions.emplace_back(CATCH, i);
        }
    }

    for (auto& action : possibleActions) {
        OutbreakState newState = state;
        newState.doUserAction(action);
        auto pokemon = newState.respawn();
        std::vector<PermutationUserAction> newActionChain = actionChain;
        newActionChain.push_back(action);
        for (auto& p : pokemon) {
            if (p.m_isShiny) {
                results.emplace_back(newActionChain, p);
            }
        }
        findPermutationsRecursive(newState, newActionChain, results);
    }
}

OutbreakState::OutbreakState(uint64_t groupSeed, OutbreakSpawner* primarySpawner, int primaryCount, OutbreakSpawner* secondarySpawner, int secondaryCount)
    : m_groupSeed(groupSeed), m_primarySpawner(primarySpawner), m_secondarySpawner(secondarySpawner), m_primaryCount(primaryCount), m_secondaryCount(secondaryCount) {
}

void OutbreakState::doUserAction(PermutationUserAction action) {
    switch(action.actionType) {
        case CATCH:
            m_pokemonAlive -= action.count;
            break;
        case GHOST:
            m_pokemonAlive -= action.count;
            m_ghosts += action.count;
            m_groupSeed = OutbreakSpawner::advanceSeed(m_groupSeed, m_ghosts);
            break;
        case CLEAR_REMAINING:
            m_pokemonAlive = 0;
            m_ghosts = 0;
            advanceWave();
            break;
    }
}

std::vector<PokemonEntity> OutbreakState::respawn() {
    std::vector<PokemonEntity> pokemon;
    if (hasEnded() || hasFirstWaveEnded()) {
        return pokemon;
    }
    int capacity = 4 - m_pokemonAlive;
    int count = std::min(m_isFirstWave ? m_primaryCount : m_secondaryCount, capacity);
    int ghosts = capacity - count;
    uint64_t nextSeed;
    if (m_isFirstWave) {
        nextSeed = m_primarySpawner->spawnPokemon(m_groupSeed, count + ghosts, pokemon, ghosts);
        m_primaryCount -= count;
    } else {
        nextSeed = m_secondarySpawner->spawnPokemon(m_groupSeed, count + ghosts, pokemon, ghosts);
        m_secondaryCount -= count;
    }
    m_pokemonAlive += count;
    m_ghosts += ghosts;
    m_groupSeed = nextSeed;
    return pokemon;
}
