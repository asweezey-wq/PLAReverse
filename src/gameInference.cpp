#include "gameInference.hpp"
#include "pokemonEntity.hpp"
#include <cmath>

int getShinyRolls(uint8_t info) {
    const int shinyRollsTable[] = {0, 1, 3, 3, 4, 4, 6, 6};
    if (info & SHINY_MMO) {
        return 13 + shinyRollsTable[info & 0x7];
    } else if (info & SHINY_MO) {
        return 26 + shinyRollsTable[info & 0x7];
    } else {
        return 1 + shinyRollsTable[info & 0x7];
    }
}

constexpr float getNatureModifier(Nature nature, int stat) {
    auto effect = NATURE_EFFECTS[nature];
    if (effect.increased == effect.decreased) return 1.0f;
    if (effect.increased == stat) return 1.1f;
    if (effect.decreased == stat) return 0.9f;
    return 1.0f;
}

int calculateHP(uint8_t base, uint8_t iv, uint8_t ev, uint8_t level) {
    return ((2 * base + iv + (ev / 4)) * level) / 100 + level + 10;
}

int calculateOtherStat(uint8_t base, uint8_t iv, uint8_t ev, uint8_t level, float natureMod) {
    int stat = ((2 * base + iv + (ev / 4)) * level) / 100 + 5;
    return static_cast<int>(std::floor(stat * natureMod));
}

int getNumIVPermutations(uint8_t ivRanges[6][2]) {
    int sum = 1;
    for (int i = 0; i < 6; i++) {
        sum *= 1 + ivRanges[i][1] - ivRanges[i][0];
    }
    return sum;
}

void printIVRanges(uint8_t ivRanges[6][2]) {
    printf("(%d) ", getNumIVPermutations(ivRanges));
    for (int i = 0; i < 6; i++) {
        printf("%u-%u%c", ivRanges[i][0], ivRanges[i][1], i == 5 ? '\n' : '/');
    }
}

void printStatRanges(uint32_t statRanges[6][2]) {
    for (int i = 0; i < 6; i++) {
        printf("%u-%u%c", statRanges[i][0], statRanges[i][1], i == 5 ? '\n' : '/');
    }
}

void calculatePLAELRanges(uint8_t effortLevels[6], uint8_t ivRanges[6][2]) {
    for (int i = 0; i < 6; i++) {
        ivRanges[i][0] = PLA_EFFORTLEVEL_RANGES[effortLevels[i]][0];
        ivRanges[i][1] = PLA_EFFORTLEVEL_RANGES[effortLevels[i]][1];
    }
}

void restrictRangesForJudging(JudgeIVRating ratings[6], uint8_t ivRanges[6][2]) {
    for (int i = 0; i < 6; i++) {
        uint8_t judgeRanges[2] {JUDGE_IV_RANGES[ratings[i]][0], JUDGE_IV_RANGES[ratings[i]][1]};
        ivRanges[i][0] = std::max(ivRanges[i][0], judgeRanges[0]);
        ivRanges[i][1] = std::min(ivRanges[i][1], judgeRanges[1]);
    }
}

void restrictRangesForActualStats(const uint32_t stats[6], const SpeciesData& data, int level, Nature nature, uint8_t ivRanges[6][2]) {
    for (int i = 0; i < 6; i++) {
        int newMin, newMax;
        for (newMin = ivRanges[i][0]; newMin <= ivRanges[i][1]; newMin++) {
            uint32_t stat = i == 0 ? calculateHP(data.baseStats[i], newMin, 0, level) : calculateOtherStat(data.baseStats[i], newMin, 0, level, getNatureModifier(nature, i));
            if (stat >= stats[i]) {
                break;
            }
        }
        for (newMax = ivRanges[i][1]; newMax >= newMin; newMax--) {
            uint32_t stat = i == 0 ? calculateHP(data.baseStats[i], newMax, 0, level) : calculateOtherStat(data.baseStats[i], newMax, 0, level, getNatureModifier(nature, i));
            if (stat <= stats[i]) {
                break;
            }
        }
        ivRanges[i][0] = newMin;
        ivRanges[i][1] = newMax;
    }
}

void calculateIVRanges(uint8_t effortLevels[6], JudgeIVRating ratings[6], Nature nature, const std::vector<ObservedStatInstance> observedStats, uint8_t ivRanges[6][2]) {
    calculatePLAELRanges(effortLevels, ivRanges);
    restrictRangesForJudging(ratings, ivRanges);
    for (auto& observe : observedStats) {
        auto& data = PokemonData::getSpeciesData(observe.speciesId);
        restrictRangesForActualStats(observe.stats, data, observe.level, nature, ivRanges);
    }
    printIVRanges(ivRanges);
}

float getSizeRatio(uint8_t value) {
    return ((value / 255.0f) * 0.4f) + 0.8f;
}

void getDisplaySize(const SpeciesData& speciesData, bool imperial, uint8_t height, uint8_t weight, float& dispHeight, float& dispWeight) {
    float heightRatio = getSizeRatio(height);
    float weightRatio = getSizeRatio(weight) * heightRatio;
    float heightInCm = speciesData.baseHeight * heightRatio;
    float weightInHg = speciesData.baseWeight * weightRatio;
    if (imperial) {
        // inches, lbs
        dispHeight = std::round(heightInCm / 2.54f);
        dispWeight = weightInHg / 4.53592f;
    } else {
        // meters, kg
        dispHeight = heightInCm / 100.0f;
        dispWeight = weightInHg / 10.0f;
    }
}

void calculateSizeRange(const SpeciesData& speciesData, bool imperial, float dispHeight, float dispWeight, uint32_t heightRange[2], uint32_t weightRange[2]) {
    heightRange[0] = 255;
    heightRange[1] = 0;
    weightRange[0] = 255;
    weightRange[1] = 0;
    const float heightEpsilon = imperial ? 0.1f : 0.1f;
    const float weightEpsilon = imperial ? 0.1f : 0.1f;
    for (int height = 0; height < 256; height++) {
        for (int weight = 0; weight < 256; weight++) {
            float computedDispHeight, computedDispWeight;
            uint8_t heightByte = (uint8_t)height;
            uint8_t weightByte = (uint8_t)weight;
            getDisplaySize(speciesData, imperial, heightByte, weightByte, computedDispHeight, computedDispWeight);
            float heightDiff = std::abs(computedDispHeight - dispHeight);
            if (heightDiff >= heightEpsilon) {
                continue;
            }
            float weightDiff = std::abs(computedDispWeight - dispWeight);
            if (weightDiff >= weightEpsilon) {
                continue;
            }
            if (heightByte < heightRange[0]) {
                heightRange[0] = heightByte;
            }
            if (heightByte > heightRange[1]) {
                heightRange[1] = heightByte;
            }
            if (weightByte < weightRange[0]) {
                weightRange[0] = weightByte;
            }
            if (weightByte > weightRange[1]) {
                weightRange[1] = weightByte;
            }
        }
    }
}

void calculateSizeRanges(bool imperial, const std::vector<ObservedSizeInstance> sizes, uint32_t heightRange[2], uint32_t weightRange[2]) {
    heightRange[0] = 0;
    heightRange[1] = 255;
    weightRange[0] = 0;
    weightRange[1] = 255;
    for (auto& size : sizes) {
        auto& data = PokemonData::getSpeciesData(size.speciesId);
        uint32_t hRange[2];
        uint32_t wRange[2];
        calculateSizeRange(data, imperial, size.height, size.weight, hRange, wRange);
        printf("Height Range %u-%u\n", hRange[0], hRange[1]);
        printf("Weight Range %u-%u\n", wRange[0], wRange[1]);
        heightRange[0] = std::max(heightRange[0], hRange[0]);
        heightRange[1] = std::min(heightRange[1], hRange[1]);
        weightRange[0] = std::max(weightRange[0], wRange[0]);
        weightRange[1] = std::min(weightRange[1], wRange[1]);
    }
    printf("Height Range %u-%u\n", heightRange[0], heightRange[1]);
    printf("Weight Range %u-%u\n", weightRange[0], weightRange[1]);
}

std::string heightToString(bool imperial, float dispHeight) {
    std::stringstream ss;
    ss << std::fixed;
    if (imperial) {
        uint32_t inches = (uint32_t)dispHeight;
        ss << (inches / 12) << "'" << (inches % 12) << "\"";
    } else {
        ss.precision(2);
        ss << dispHeight << " m";
    }
    return ss.str();
}

std::string weightToString(bool imperial, float dispWeight) {
    std::stringstream ss;
    ss << std::fixed;
    if (imperial) {
        ss.precision(1);
        ss << dispWeight << " lbs";
    } else {
        ss.precision(2);
        ss << dispWeight << " kg";
    }
    return ss.str();
}

float imperialHeight(int feet, int inches) {
    return feet * 12 + inches;
}

void statTesting() {
    auto data = PokemonData::getSpeciesData(PokemonData::getSpeciesID("Swinub"));
    uint32_t stats[6] = {74, 42, 32, 28, 37, 45};
    uint8_t ivs[6] = {1, 2, 3, 4, 5, 6};
    uint8_t evs[6] = {0, 0, 0, 0, 0, 0};
    Nature nature = Gentle;
    uint8_t level = 32;
    printf("%s (#%u)\n", PokemonData::getSpeciesName(data.index).c_str(), data.index);
    // for (int i = 0; i < 6; i++) {
    //     if (i == 0) {
    //         printf("%s: %d\n", STAT_NAMES[i].c_str(), calculateHP(data.baseStats[i], ivs[i], evs[i], level));
    //     } else {
    //         printf("%s: %d\n", STAT_NAMES[i].c_str(), calculateOtherStat(data.baseStats[i], ivs[i], evs[i], level, getNatureModifier(nature, i)));
    //     }
    // }

    const bool imperial = true;
    float dispHeight = 15;
    float dispWeight = 15.8f;
    uint32_t heightRange[2];
    uint32_t weightRange[2];
    calculateSizeRange(data, imperial, dispHeight, dispWeight, heightRange, weightRange);
    float minHeight, minWeight;
    getDisplaySize(data, imperial, 0, 0, minHeight, minWeight);
    float maxHeight, maxWeight;
    getDisplaySize(data, imperial, 255, 255, maxHeight, maxWeight);
    printf("%s min size: %s %s\n", PokemonData::getSpeciesName(data.index).c_str(), heightToString(imperial, minHeight).c_str(), weightToString(imperial, minWeight).c_str());
    printf("%s max size: %s %s\n", PokemonData::getSpeciesName(data.index).c_str(), heightToString(imperial, maxHeight).c_str(), weightToString(imperial, maxWeight).c_str());
    printf("%s actual size: %s %s\n", PokemonData::getSpeciesName(data.index).c_str(), heightToString(imperial, dispHeight).c_str(), weightToString(imperial, dispWeight).c_str());
    printf("Height Range %u-%u\n", heightRange[0], heightRange[1]);
    printf("Weight Range %u-%u\n", weightRange[0], weightRange[1]);

    uint8_t ivRanges[6][2];
    uint8_t plaEL[6] = {0,0,0,0,3,2};
    JudgeIVRating ratings[6] = {DECENT, PRETTYGOOD, PRETTYGOOD, DECENT, BEST, VERYGOOD};
    calculatePLAELRanges(plaEL, ivRanges);
    printIVRanges(ivRanges);
    restrictRangesForJudging(ratings, ivRanges);
    printIVRanges(ivRanges);
    uint32_t actualStatRanges[6][2];
    actualStatRanges[0][0] = calculateHP(data.baseStats[0], 0, 0, level);
    actualStatRanges[0][1] = calculateHP(data.baseStats[0], 31, 0, level);
    for (int i = 1; i < 6; i++) {
        actualStatRanges[i][0] = calculateOtherStat(data.baseStats[i], 0, 0, level, getNatureModifier(nature, i));
        actualStatRanges[i][1] = calculateOtherStat(data.baseStats[i], 31, 0, level, getNatureModifier(nature, i));
    }
    printStatRanges(actualStatRanges);
    uint8_t ivRanges2[6][2] = {{0,31},{0,31},{0,31},{0,31},{0,31},{0,31}};
    restrictRangesForActualStats(stats, data, level, nature, ivRanges2);
    printIVRanges(ivRanges2);
    restrictRangesForActualStats(stats, data, level, nature, ivRanges);
    printIVRanges(ivRanges);
    uint32_t swinubStats2[6] = {76, 43, 33, 29, 38, 46};
    restrictRangesForActualStats(swinubStats2, data, level + 1, nature, ivRanges);
    printIVRanges(ivRanges);
    uint32_t piloswineStats[6] = {109, 76, 56, 48, 59, 46};
    restrictRangesForActualStats(piloswineStats, PokemonData::getSpeciesData(PokemonData::getSpeciesID("Piloswine")), level + 1, nature, ivRanges);
    printIVRanges(ivRanges);
    uint32_t piloswineStats2[6] = {112, 78, 58, 50, 61, 48};
    restrictRangesForActualStats(piloswineStats2, PokemonData::getSpeciesData(PokemonData::getSpeciesID("Piloswine")), level + 2, nature, ivRanges);
    printIVRanges(ivRanges);
}