#pragma once

#include "pokemonCuda.hpp"
#include <QWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <vector>

class PokemonTab : public QWidget {
public:
    PokemonTab(QWidget* parent = nullptr) : QWidget(parent) {}
    virtual void onTabShown() {}
};

struct SizeInput {
    QComboBox* species;
    QSpinBox* height[2];
    QDoubleSpinBox* weight;
};

struct StatsInput {
    QComboBox* species;
    QSpinBox* level;
    QSpinBox* stats[6];
};

struct RatingsInput {
    QComboBox* ratings[6];
};

class PokemonSlotGroup;

class PokemonInput {
public:
    const PokemonSlotGroup* slotGroup;

    QComboBox* species;
    QSpinBox* level;
    QSpinBox* els[6];
    QComboBox* nature;
    QComboBox* gender;
    QComboBox* ability;

    QPushButton* sizeButton;
    std::vector<SizeInput> sizeInputs;
    QLineEdit* computedHeight[2];
    QLineEdit* computedWeight[2];
    QLineEdit* possibleSizes;

    QPushButton* statsButton;
    std::vector<StatsInput> statsInputs;
    RatingsInput ratings;
    QLineEdit* computedIVs[6][2];
    QLineEdit* numIVCombos;

    QLineEdit* numSeeds;
    QLineEdit* generatorCost;

    bool verifValid = false;
    bool canGen = false;
    PokemonVerificationContext verifCtx;
    std::vector<std::pair<uint32_t, uint32_t>> sizePairs;

    void calculateIVs();
    void calculateSizes();
    void calculateSeeds();
};