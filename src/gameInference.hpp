#pragma once
#include <cstdint>
#include <string>
#include "pokemonEntity.hpp"

enum Nature : uint8_t {
    Hardy = 0,
    Lonely,
    Brave,
    Adamant,
    Naughty,
    Bold,
    Docile,
    Relaxed,
    Impish,
    Lax,
    Timid,
    Hasty,
    Serious,
    Jolly,
    Naive,
    Modest,
    Mild,
    Quiet,
    Bashful,
    Rash,
    Calm,
    Gentle,
    Sassy,
    Careful,
    Quirky
};

enum Stat : int {
    HP = 0,
    Attack,
    Defense,
    SpAttack,
    SpDefense,
    Speed
};

struct NatureEffect {
    int increased;
    int decreased;
};

// Indexed by Nature enum
constexpr NatureEffect NATURE_EFFECTS[25] = {
    {Attack, Attack},     // Hardy (neutral)
    {Attack, Defense},    // Lonely
    {Attack, Speed},      // Brave
    {Attack, SpAttack},   // Adamant
    {Attack, SpDefense},  // Naughty
    {Defense, Attack},    // Bold
    {Defense, Defense},   // Docile (neutral)
    {Defense, Speed},     // Relaxed
    {Defense, SpAttack},  // Impish
    {Defense, SpDefense}, // Lax
    {Speed, Attack},      // Timid
    {Speed, Defense},     // Hasty
    {Speed, Speed},       // Serious (neutral)
    {Speed, SpAttack},    // Jolly
    {Speed, SpDefense},   // Naive
    {SpAttack, Attack},   // Modest
    {SpAttack, Defense},  // Mild
    {SpAttack, Speed},    // Quiet
    {SpAttack, SpAttack}, // Bashful (neutral)
    {SpAttack, SpDefense},// Rash
    {SpDefense, Attack},  // Calm
    {SpDefense, Defense}, // Gentle
    {SpDefense, Speed},   // Sassy
    {SpDefense, SpAttack},// Careful
    {SpDefense, SpDefense}// Quirky (neutral)
};

constexpr std::string STAT_NAMES[] = {
    "HP", "Attack", "Defense", "Sp. Attack", "Sp. Defense", "Speed"
};

constexpr std::string NATURE_NAMES[] = {
    "Hardy", "Lonely", "Brave", "Adamant", "Naughty",
    "Bold", "Docile", "Relaxed", "Impish", "Lax",
    "Timid", "Hasty", "Serious", "Jolly", "Naive",
    "Modest", "Mild", "Quiet", "Bashful", "Rash",
    "Calm", "Gentle", "Sassy", "Careful", "Quirky"
};

enum JudgeIVRating {
    NOGOOD,
    DECENT,
    PRETTYGOOD,
    VERYGOOD,
    FANTASTIC,
    BEST
};

constexpr uint8_t JUDGE_IV_RANGES[6][2] = {
    {0, 0},
    {1, 15},
    {16, 25},
    {26, 29},
    {30, 30},
    {31, 31}
};

constexpr uint8_t PLA_EFFORTLEVEL_RANGES[6][2] = {
    {0, 19},
    {20, 25},
    {26, 30},
    {31, 31}
};

enum SpeciesShinyInfo : uint8_t {
    SHINY_BASE = 0,
    SHINY_DEX10 = 0b01,
    SHINY_DEXPERFECT = 0b11,
    SHINY_CHARM = 0b101,
    SHINY_MO = 0b1000,
    SHINY_MMO = 0b10000
};

int getShinyRolls(uint8_t info);
constexpr float getNatureModifier(Nature nature, int stat);
int calculateHP(uint8_t base, uint8_t iv, uint8_t ev, uint8_t level);
int calculateOtherStat(uint8_t base, uint8_t iv, uint8_t ev, uint8_t level, float natureMod);

void calculatePLAELRanges(uint8_t effortLevels[6], uint8_t ivRanges[6][2]);
void restrictRangesForJudging(JudgeIVRating ratings[6], uint8_t ivRanges[6][2]);
struct ObservedStatInstance {
    uint32_t speciesId;
    int level;
    uint32_t stats[6];
};
void calculateIVRanges(uint8_t effortLevels[6], JudgeIVRating ratings[6], Nature nature, const std::vector<ObservedStatInstance> observedStats, uint8_t ivRanges[6][2]);
int getNumIVPermutations(uint8_t ivRanges[6][2]);

void getDisplaySize(const SpeciesData& speciesData, bool imperial, uint8_t height, uint8_t weight, float& dispHeight, float& dispWeight);
std::string heightToString(bool imperial, float dispHeight);
std::string weightToString(bool imperial, float dispWeight);
float imperialHeight(int feet, int inches);

struct ObservedSizeInstance {
    uint32_t speciesId;
    float height;
    float weight;
};
void calculateSizeRange(const SpeciesData& speciesData, bool imperial, float dispHeight, float dispWeight, uint32_t heightRange[2], uint32_t weightRange[2]);
void calculateSizeRanges(bool imperial, const std::vector<ObservedSizeInstance> sizes, uint32_t heightRange[2], uint32_t weightRange[2]);


void statTesting();