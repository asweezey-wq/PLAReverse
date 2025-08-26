#include "gameInference.hpp"
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

float getNatureModifier(uint8_t nature, int stat) {
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

int getNumIVPermutations(const uint8_t ivRanges[6][2]) {
    int sum = 1;
    for (int i = 0; i < 6; i++) {
        sum *= 1 + ivRanges[i][1] - ivRanges[i][0];
    }
    return sum;
}

void printIVRanges(const uint8_t ivRanges[6][2]) {
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

void restrictRangesForActualStats(const ObservedStatInstance& observedStats, uint8_t nature, uint8_t ivRanges[6][2]) {
    auto& data = PokemonData::getSpeciesData(observedStats.speciesId);
    for (int i = 0; i < 6; i++) {
        int newMin, newMax;
        for (newMin = ivRanges[i][0]; newMin <= ivRanges[i][1]; newMin++) {
            uint32_t stat = i == 0 ? calculateHP(data.baseStats[i], newMin, 0, observedStats.level) : calculateOtherStat(data.baseStats[i], newMin, 0, observedStats.level, getNatureModifier(nature, i));
            if (stat >= observedStats.stats[i]) {
                break;
            }
        }
        for (newMax = ivRanges[i][1]; newMax >= newMin; newMax--) {
            uint32_t stat = i == 0 ? calculateHP(data.baseStats[i], newMax, 0, observedStats.level) : calculateOtherStat(data.baseStats[i], newMax, 0, observedStats.level, getNatureModifier(nature, i));
            if (stat <= observedStats.stats[i]) {
                break;
            }
        }
        if (newMax == -1) {
            ivRanges[i][0] = 1;
            ivRanges[i][1] = 0;
        } else {
            ivRanges[i][0] = newMin;
            ivRanges[i][1] = newMax;
        }
    }
}

void calculateIVRanges(uint8_t effortLevels[6], JudgeIVRating ratings[6], uint8_t nature, const std::vector<ObservedStatInstance> observedStats, uint8_t ivRanges[6][2]) {
    calculatePLAELRanges(effortLevels, ivRanges);
    restrictRangesForJudging(ratings, ivRanges);
    for (auto& observe : observedStats) {
        restrictRangesForActualStats(observe, nature, ivRanges);
    }
    printIVRanges(ivRanges);
}

float getSizeRatio(uint8_t value) {
    return ((value / 255.0f) * 0.40000004f) + 0.8f;
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
    const float heightEpsilon = imperial ? 0.3f : 0.1f;
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
        heightRange[0] = std::max(heightRange[0], hRange[0]);
        heightRange[1] = std::min(heightRange[1], hRange[1]);
        weightRange[0] = std::max(weightRange[0], wRange[0]);
        weightRange[1] = std::min(weightRange[1], wRange[1]);
    }
}

std::vector<std::pair<uint32_t, uint32_t>> calculateSizePairs(bool imperial, const std::vector<ObservedSizeInstance> sizes) {
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    const float heightEpsilon = imperial ? 0.06f : 0.06f;
    const float weightEpsilon = imperial ? 0.06f : 0.06f;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            bool valid = true;
            uint8_t heightByte = (uint8_t)i;
            uint8_t weightByte = (uint8_t)j;
            float computedDispHeight, computedDispWeight;
            for (auto& size : sizes) {
                getDisplaySize(PokemonData::getSpeciesData(size.speciesId), imperial, heightByte, weightByte, computedDispHeight, computedDispWeight);
                float heightDiff = std::abs(computedDispHeight - size.height);
                if (heightDiff >= heightEpsilon) {
                    valid = false;
                    break;
                }
                float weightDiff = std::abs(computedDispWeight - size.weight);
                if (weightDiff >= weightEpsilon) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                pairs.emplace_back(i, j);
            }
        }
    }
    return pairs;
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
    return (float)feet * 12 + (float)inches;
}