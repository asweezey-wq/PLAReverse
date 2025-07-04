#pragma once

#include "pokemonData.hpp"
#include "outbreakSpawner.hpp"

enum UserActionType {
    CATCH,
    GHOST,
    CLEAR_REMAINING
};

struct PermutationUserAction {
    UserActionType actionType;
    int count;
};

struct PermutationResult {
    std::vector<PermutationUserAction> actionChain;
    PokemonEntity pokemon;
};

class OutbreakState {
public:
    OutbreakState() : OutbreakState(0, nullptr, 0, nullptr, 0) {}
    OutbreakState(uint64_t groupSeed, OutbreakSpawner* primarySpawner, int primaryCount, OutbreakSpawner* secondarySpawner, int secondaryCount);
    
    void doUserAction(PermutationUserAction action);
    std::vector<PokemonEntity> respawn();
    void advanceWave() { m_isFirstWave = false; }

    bool hasFirstWaveEnded() const { return m_isFirstWave && m_primaryCount == 0; }
    bool hasEnded() const { return !m_isFirstWave && m_secondaryCount == 0; }
    
    uint64_t m_groupSeed;
    int m_primaryCount = 0;
    int m_secondaryCount = 0;
    OutbreakSpawner *m_primarySpawner, *m_secondarySpawner;

    bool m_isFirstWave = true;
    int m_pokemonAlive = 0;
    int m_ghosts = 0;
};

class PermutationsManager {
public:
    PermutationsManager(uint64_t seed, uint64_t primaryTableID, int primaryCount, uint64_t secondaryTableID, int secondaryCount, ShinyRollData& shinyRolls);
    void addSearchQualifier();

    std::vector<PermutationResult> findPermutations();

private:
    OutbreakState m_startingState;
    OutbreakSpawner *m_primarySpawner = nullptr;
    OutbreakSpawner *m_secondarySpawner = nullptr;

    void findPermutationsRecursive(OutbreakState state, const std::vector<PermutationUserAction>& actionChain, std::vector<PermutationResult>& results);
};