#include "PokemonInputTab.h"
#include "pokemonData.hpp"
#include "gameInference.hpp"
#include "seedReversal.hpp"
#include <QHBoxLayout>
#include <QFormLayout>
#include <QLabel>
#include <QMessageBox>
#include <QScrollArea>
#include <unordered_set>
#include <queue>
#include <bit>
#include <sstream>

PokemonInputTab::PokemonInputTab(QWidget *parent)
    : PokemonTab(parent)
{
    std::unordered_set<uint32_t> duplicateSpecies;
    duplicateSpecies.insert(PokemonData::getSpeciesID("Kirlia"));
    duplicateSpecies.insert(PokemonData::getSpeciesID("Snorunt"));
    duplicateSpecies.insert(PokemonData::getSpeciesID("Wurmple"));
    std::vector<std::pair<QString, uint64_t>> outbreakSpecies;
    // outbreakSpecies.emplace_back(QString(), 0);
    for (auto& pair : PokemonData::getOutbreakTables()) {
        auto& slotGroup = pair.second;
        auto& mainSlot = slotGroup.getSlotFromIndex(0);
        auto& mainPokemon = mainSlot.m_species;
        auto& mainPokemonName = PokemonData::getSpeciesName(mainPokemon);
        if (!mainSlot.m_isAlpha && mainSlot.m_numPerfectIVs == 0) {
            QString qstr;
            if (duplicateSpecies.contains(mainPokemon)) {
                if (slotGroup.numSlots() < 3) {
                    fprintf(stderr, "Duplicate pokemon but no second spawn slot\n");
                    exit(1);
                }
                auto& secondSlot = slotGroup.getSlotFromIndex(2);
                qstr = QString("%1 (%2)").arg(mainPokemonName.c_str(), PokemonData::getSpeciesName(secondSlot.m_species));
            } else {
                qstr = QString("%1").arg(mainPokemonName.c_str());
            }
            outbreakSpecies.emplace_back(
                qstr,
                pair.first
            );
        }
    }

    std::sort(outbreakSpecies.begin(), outbreakSpecies.end());
    for (auto& pair : outbreakSpecies) {
        m_outbreakSpeciesList.append(pair.first);
        m_indexToOutbreakTable.push_back(pair.second);
    }

    for (auto& natureStr : NATURE_NAMES) {
        m_natureList.append(QString(natureStr.c_str()));
    }

    m_ratingsList = {"", "No Good", "Decent", "Pretty Good", "Very Good", "Fantastic", "Best"};

    createForm();
    selectNewOutbreakSpecies(0);
}

void PokemonInputTab::createForm()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    QGroupBox* tableGroup = new QGroupBox(QString("Spawner Info"));
    QFormLayout* outbreakForm = new QFormLayout();

    QComboBox* typeComboBox = new QComboBox();
    typeComboBox->addItem(QString("Massive Mass Outbreak"));
    typeComboBox->addItem(QString("Mass Outbreak"));
    outbreakForm->addRow("Type:", typeComboBox);

    m_outbreakSpeciesComboBox = new QComboBox();
    m_outbreakSpeciesComboBox->addItems(m_outbreakSpeciesList);
    outbreakForm->addRow("Species:", m_outbreakSpeciesComboBox);
    m_selectedOutbreakTable = 0;
    connect(m_outbreakSpeciesComboBox, &QComboBox::currentIndexChanged, this, &PokemonInputTab::selectNewOutbreakSpecies);
    
    tableGroup->setLayout(outbreakForm);
    mainLayout->addWidget(tableGroup);

    QHBoxLayout* pokemonLayout = new QHBoxLayout();

    for (int i = 0; i < 4; ++i) {
        QGroupBox* group = createPokemonGroup(i + 1);
        pokemonLayout->addWidget(group);
    }

    mainLayout->addLayout(pokemonLayout);

    mainLayout->addStretch();
    setLayout(mainLayout);
}

QGroupBox* PokemonInputTab::createPokemonGroup(int index)
{
    auto* groupBox = new QGroupBox(QString("Pokémon %1").arg(index));
    auto* formLayout = new QFormLayout();

    PokemonInput* input = new PokemonInput();
    input->slotGroup = nullptr;
    input->verifValid = false;
    m_pokemonInputs.push_back(input);

    input->species = new QComboBox();
    connect(input->species, &QComboBox::currentIndexChanged, this, [=]() {
        input->calculateIVs();
    });

    input->level = new QSpinBox();
    input->level->setRange(1, 100);
    connect(input->level, &QSpinBox::valueChanged, this, [=]() {
        input->calculateSeeds();
    });

    for (int i = 0; i < 6; i++) {
        input->els[i] = new QSpinBox();
        input->els[i]->setRange(0, 3);
        connect(input->els[i], &QSpinBox::valueChanged, this, [=](int value) {
            input->calculateIVs();
        });
    }

    input->nature = new QComboBox();
    connect(input->nature, &QComboBox::currentIndexChanged, this, [=]() {
        input->calculateIVs();
    });
    input->gender = new QComboBox();
    connect(input->gender, &QComboBox::currentIndexChanged, this, [=]() {
        input->calculateSeeds();
    });
    input->ability = new QComboBox();
    connect(input->ability, &QComboBox::currentIndexChanged, this, [=]() {
        input->calculateSeeds();
    });

    for (int i = 0; i < 2; i++) {
        input->computedHeight[i] = new QLineEdit();
        input->computedHeight[i]->setReadOnly(true);
        input->computedWeight[i] = new QLineEdit();
        input->computedWeight[i]->setReadOnly(true);
    }
    input->possibleSizes = new QLineEdit();
    input->possibleSizes->setReadOnly(true);
    input->verifCtx.height[0] = 255;
    input->verifCtx.height[1] = 0;
    input->verifCtx.weight[0] = 255;
    input->verifCtx.weight[1] = 0;

    input->sizeButton = new QPushButton("Size Calculator");
    connect(input->sizeButton, &QPushButton::clicked, this, [=]() {
        PokemonSizeCalculator* popup = new PokemonSizeCalculator(input, this);
        popup->exec(); 
    });
    input->sizeButton->setEnabled(false);

    input->statsButton = new QPushButton("IV Calculator");
    connect(input->statsButton, &QPushButton::clicked, this, [=]() {
        PokemonStatCalculator* popup = new PokemonStatCalculator(input, this);
        popup->exec(); 
    });
    input->statsButton->setEnabled(false);
    for (int i = 0; i < 6; i++) {
        input->computedIVs[i][0] = new QLineEdit();
        input->computedIVs[i][0]->setReadOnly(true);
        input->computedIVs[i][1] = new QLineEdit();
        input->computedIVs[i][1]->setReadOnly(true);
        input->ratings.ratings[i] = new QComboBox();
        input->ratings.ratings[i]->addItems(m_ratingsList);
        input->verifCtx.ivs[i][0] = 31;
        input->verifCtx.ivs[i][1] = 0;
    }
    input->numIVCombos = new QLineEdit();
    input->numIVCombos->setReadOnly(true);

    input->numSeeds = new QLineEdit();
    input->numSeeds->setReadOnly(true);
    input->generatorCost = new QLineEdit();
    input->generatorCost->setReadOnly(true);

    formLayout->addRow("Species:", input->species);
    formLayout->addRow("Level:", input->level);
    QHBoxLayout* ivHBox = new QHBoxLayout();
    for (int i = 0; i < 6; i++) {
        ivHBox->addWidget(input->els[i]);
    }
    formLayout->addRow("ELs:", ivHBox);
    formLayout->addRow("Nature:", input->nature);
    formLayout->addRow("Gender:", input->gender);
    formLayout->addRow("Ability:", input->ability);

    formLayout->addWidget(input->sizeButton);
    QHBoxLayout* heightHBox = new QHBoxLayout();
    heightHBox->addWidget(input->computedHeight[0]);
    heightHBox->addWidget(input->computedHeight[1]);
    formLayout->addRow("Height:", heightHBox);
    QHBoxLayout* weightHBox = new QHBoxLayout();
    weightHBox->addWidget(input->computedWeight[0]);
    weightHBox->addWidget(input->computedWeight[1]);
    formLayout->addRow("Weight:", weightHBox);
    formLayout->addRow("# Sizes:", input->possibleSizes);

    formLayout->addWidget(input->statsButton);
    for (int i = 0; i < 6; i++) {
        QHBoxLayout* ivBox = new QHBoxLayout();
        ivBox->addWidget(input->computedIVs[i][0]);
        ivBox->addWidget(input->computedIVs[i][1]);
        formLayout->addRow(QString("%1:").arg(STAT_NAMES[i].c_str()), ivBox);
    }
    formLayout->addRow("# IVs:", input->numIVCombos);
    formLayout->addRow("# Seeds:", input->numSeeds);
    formLayout->addRow("Gen Cost:", input->generatorCost);

    groupBox->setLayout(formLayout);
    return groupBox;
}

void PokemonInputTab::populateDropdowns(PokemonInput* input)
{
    input->verifValid = false;
    input->verifCtx = PokemonVerificationContext();

    input->species->clear();
    input->species->addItems(m_speciesList);
    input->species->setEnabled(m_speciesList.size() > 1);

    input->level->setValue(0);
    input->level->clear();
    for (int i = 0; i < 6; i++) {
        input->els[i]->setValue(0);
    }

    input->nature->clear();
    input->nature->addItems(m_natureList);

    input->gender->clear();
    input->gender->addItems(m_genderList);
    input->gender->setEnabled(m_genderList.size() > 1);

    input->ability->clear();
    input->ability->addItems(m_abilityList);
    input->ability->setEnabled(m_abilityList.size() > 1);

    input->sizeInputs.clear();
    input->sizeButton->setEnabled(true);

    input->statsInputs.clear();
    input->statsButton->setEnabled(true);
    for (int i = 0; i < 2; i++) {
        input->computedHeight[i]->clear();
        input->computedWeight[i]->clear();
    }
    input->verifCtx.height[0] = 0;
    input->verifCtx.weight[0] = 0;
    input->verifCtx.height[1] = 255;
    input->verifCtx.weight[1] = 255;

    input->numSeeds->clear();
    input->generatorCost->clear();
}

void PokemonInputTab::selectNewOutbreakSpecies(int index) {
    if (index >= m_indexToOutbreakTable.size()) {
        fprintf(stderr, "Selected species index too large!");
        exit(1);
    }
    m_selectedOutbreakTable = m_indexToOutbreakTable[index];
    if (m_selectedOutbreakTable) {
        auto& slotGroup = PokemonData::getSlotGroupTable(m_selectedOutbreakTable);
        std::vector<uint32_t> speciesList;
        m_speciesList.clear();
        for (int i = 0; i < slotGroup.numSlots(); i++) {
            if (!slotGroup.getSlotFromIndex(i).m_isAlpha) {
                uint32_t species = slotGroup.getSlotFromIndex(i).m_species;
                speciesList.push_back(species);
                m_speciesList.append(QString(PokemonData::getSpeciesName(species).c_str()));
            }
        }
        uint32_t species = speciesList[0];
        auto& data = PokemonData::getSpeciesData(species);
        if (data.genderRatio == 0 || data.genderRatio >= 254) {
            m_genderList = {"Genderless"};
        } else {
            m_genderList = {"Male", "Female"};
        }

        m_abilityList.clear();
        if (data.abilities[0] == data.abilities[1]) {
            m_abilityList.append(QString(PokemonData::getAbilityName(data.abilities[0]).c_str()));
        } else {
            m_abilityList.append(QString());
            for (int i = 0; i < 2; i++) {
                if (data.abilities[i]) {
                    m_abilityList.append(QString(PokemonData::getAbilityName(data.abilities[i]).c_str()));
                }
            }
        }

        for (auto input : m_pokemonInputs) {
            input->slotGroup = &slotGroup;
            populateDropdowns(input);
        }
    }
}

PokemonSizeCalculator::PokemonSizeCalculator(PokemonInput* input, QWidget *parent) {
    m_pokemonInput = input;
    m_baseSpecies = PokemonData::getSpeciesID(m_pokemonInput->species->currentText().toStdString());
    std::queue<uint32_t> evoQ;
    evoQ.push(m_baseSpecies);
    while (!evoQ.empty()) {
        uint32_t id = evoQ.front();
        evoQ.pop();
        m_species.append(QString("%1").arg(PokemonData::getSpeciesName(id).c_str()));
        if (PokemonData::speciesHasEvolutions(id)) {
            for (auto& evoID : PokemonData::getSpeciesEvolutions(id)) {
                evoQ.push(evoID);
            }
        }
    }

    setWindowTitle("Size Calculator");
    m_addMeasurementButton = new QPushButton("Add Measurement");

    QVBoxLayout* layout = new QVBoxLayout(this);
    m_measurementLayout = new QVBoxLayout();
    if (m_pokemonInput->sizeInputs.size() == 0) {
        addNewMeasurement();
    } else {
        for (int i = 0; i < m_pokemonInput->sizeInputs.size(); i++) {
            m_measurementLayout->addWidget(widgetFromSizeInput(i));
        }
    }
    layout->addLayout(m_measurementLayout);
    connect(m_addMeasurementButton, &QPushButton::clicked, this, &PokemonSizeCalculator::addNewMeasurement);
    layout->addWidget(m_addMeasurementButton);

    QPushButton* doneButton = new QPushButton("Calculate");
    connect(doneButton, &QPushButton::clicked, this, &PokemonSizeCalculator::onFinished);
    layout->addWidget(doneButton);

    setLayout(layout);

    setMinimumWidth(400);
}

SizeInput& PokemonSizeCalculator::addNewMeasurement() {
    SizeInput& sizeInput = m_pokemonInput->sizeInputs.emplace_back();
    sizeInput.species = new QComboBox();
    sizeInput.species->addItems(m_species);
    sizeInput.species->setCurrentIndex((int)m_pokemonInput->sizeInputs.size() - 1);
    sizeInput.height[0] = new QSpinBox();
    sizeInput.height[0]->setSuffix(" ft");
    sizeInput.height[1] = new QSpinBox();
    sizeInput.height[1]->setRange(0, 11);
    sizeInput.height[1]->setSuffix(" in");
    sizeInput.weight = new QDoubleSpinBox();
    sizeInput.weight->setSuffix(" lbs");
    sizeInput.weight->setDecimals(1);
    sizeInput.weight->setSingleStep(0.1);
    sizeInput.weight->setRange(0, 9999);
    m_measurementLayout->addWidget(widgetFromSizeInput((int)m_pokemonInput->sizeInputs.size() - 1));
    m_addMeasurementButton->setEnabled((qsizetype)m_pokemonInput->sizeInputs.size() < m_species.size());
    return sizeInput;
}

QWidget* PokemonSizeCalculator::widgetFromSizeInput(int index) {
    auto& sizeInput = m_pokemonInput->sizeInputs[index];
    QGroupBox* groupBox = new QGroupBox(QString("Measurement"));
    QFormLayout* form = new QFormLayout(groupBox);
    QHBoxLayout* heightBox = new QHBoxLayout();
    heightBox->addWidget(sizeInput.height[0]);
    heightBox->addWidget(sizeInput.height[1]);
    form->addRow("Species:", sizeInput.species);
    form->addRow("Height:", heightBox);
    form->addRow("Weight:", sizeInput.weight);
    if (index != 0) {
        QPushButton* removeButton = new QPushButton("Remove", groupBox);
        connect(removeButton, &QPushButton::clicked, this, [=](){
            removeSizeInput(index);
        });
        form->addWidget(removeButton);
    }
    m_measurements.push_back(groupBox);
    return groupBox;
}

void PokemonSizeCalculator::onFinished() {
    m_pokemonInput->calculateSizes();
    if (m_pokemonInput->verifCtx.height[1] >= m_pokemonInput->verifCtx.height[0] && m_pokemonInput->verifCtx.weight[1] >= m_pokemonInput->verifCtx.weight[0]) {
        close();
    } else {
        QMessageBox::information(this, "Error", "Invalid sizes. Could not compute size ranges.");
    }
}

void PokemonInput::calculateSizes() {
    std::vector<ObservedSizeInstance> sizes;
    for (auto& sizeInput : sizeInputs) {
        float height = sizeInput.height[0]->value() * 12 + sizeInput.height[1]->value();
        float weight = sizeInput.weight->value();
        sizes.emplace_back(PokemonData::getSpeciesID(sizeInput.species->currentText().toStdString()), height, weight);
    }
    calculateSizeRanges(true, sizes, verifCtx.height, verifCtx.weight);
    const char* styleSheet = verifCtx.height[1] < verifCtx.height[0] ? "color: red;" : "color: black;";
    computedHeight[0]->setText(QString("%1").arg(verifCtx.height[0]));
    computedHeight[0]->setStyleSheet(styleSheet);
    computedHeight[1]->setText(QString("%1").arg(verifCtx.height[1]));
    computedHeight[1]->setStyleSheet(styleSheet);
    styleSheet = verifCtx.weight[1] < verifCtx.weight[0] ? "color: red;" : "color: black;";
    computedWeight[0]->setText(QString("%1").arg(verifCtx.weight[0]));
    computedWeight[0]->setStyleSheet(styleSheet);
    computedWeight[1]->setText(QString("%1").arg(verifCtx.weight[1]));
    computedWeight[1]->setStyleSheet(styleSheet);
    sizePairs = calculateSizePairs(true, sizes);
    possibleSizes->setText(QString("%1").arg(sizePairs.size()));

    calculateSeeds();
}

void PokemonSizeCalculator::removeSizeInput(int index) {
    auto widget = m_measurements[index];
    m_measurementLayout->removeWidget(widget);
    widget->deleteLater();
    m_measurements.erase(m_measurements.begin() + index);
    auto& sizeInput = m_pokemonInput->sizeInputs[index];
    m_pokemonInput->sizeInputs.erase(m_pokemonInput->sizeInputs.begin() + index);
    m_addMeasurementButton->setEnabled((qsizetype)m_pokemonInput->sizeInputs.size() < m_species.size());
    adjustSize();
}

PokemonStatCalculator::PokemonStatCalculator(PokemonInput* input, QWidget *parent) {
    m_pokemonInput = input;
    m_baseSpecies = PokemonData::getSpeciesID(m_pokemonInput->species->currentText().toStdString());
    std::queue<uint32_t> evoQ;
    evoQ.push(m_baseSpecies);
    while (!evoQ.empty()) {
        uint32_t id = evoQ.front();
        evoQ.pop();
        m_species.append(QString("%1").arg(PokemonData::getSpeciesName(id).c_str()));
        if (PokemonData::speciesHasEvolutions(id)) {
            for (auto& evoID : PokemonData::getSpeciesEvolutions(id)) {
                evoQ.push(evoID);
            }
        }
    }

    setWindowTitle("Size Calculator");
    m_addStatsButton = new QPushButton("Add Stats");

    QScrollArea* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setMinimumHeight(500);
    QWidget* scrollWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(this);
    m_statsLayout = new QVBoxLayout();
    scrollWidget->setLayout(m_statsLayout);
    scrollArea->setWidget(scrollWidget);

    for (int i = 0; i < m_pokemonInput->statsInputs.size(); i++) {
        m_statsLayout->addWidget(widgetFromStatsInput(i));
        for (int j = 0; j < 6; j++) {
            connect(m_pokemonInput->statsInputs[i].stats[j], &QSpinBox::valueChanged, this, &PokemonStatCalculator::updateIVRanges);
        }
    }

    QHBoxLayout* ivBox = new QHBoxLayout();
    QHBoxLayout* ratingsBox = new QHBoxLayout();
    for (int i = 0; i < 6; i++) {
        QLineEdit* iv = new QLineEdit();
        iv->setReadOnly(true);
        iv->setText(QString("%1 - %2").arg((uint32_t)m_pokemonInput->verifCtx.ivs[i][0]).arg((uint32_t)m_pokemonInput->verifCtx.ivs[i][1]));
        m_ivs.push_back(iv);
        ivBox->addWidget(iv);

        ratingsBox->addWidget(m_pokemonInput->ratings.ratings[i]);
        connect(m_pokemonInput->ratings.ratings[i], &QComboBox::currentIndexChanged, this, &PokemonStatCalculator::updateIVRanges);
    }

    layout->addWidget(new QLabel("Calculated IVs"));
    layout->addLayout(ivBox);

    layout->addWidget(new QLabel("Ratings"));
    layout->addLayout(ratingsBox);

    layout->addWidget(scrollArea);
    connect(m_addStatsButton, &QPushButton::clicked, this, &PokemonStatCalculator::addNewStats);
    layout->addWidget(m_addStatsButton);

    QPushButton* doneButton = new QPushButton("Done");
    connect(doneButton, &QPushButton::clicked, this, &PokemonStatCalculator::onFinished);
    layout->addWidget(doneButton);

    setLayout(layout);
    updateIVRanges();

    setMinimumWidth(400);
}

StatsInput& PokemonStatCalculator::addNewStats() {
    StatsInput& statsInput = m_pokemonInput->statsInputs.emplace_back();
    statsInput.species = new QComboBox();
    statsInput.species->addItems(m_species);
    int lastIndex = m_pokemonInput->statsInputs.size() > 1 ? m_pokemonInput->statsInputs[m_pokemonInput->statsInputs.size() - 2].species->currentIndex() : 0;
    statsInput.species->setCurrentIndex(lastIndex);
    int lastLevel = m_pokemonInput->statsInputs.size() > 1 ? (m_pokemonInput->statsInputs[m_pokemonInput->statsInputs.size() - 2].level->value() + 1) : m_pokemonInput->level->value();
    statsInput.level = new QSpinBox();
    statsInput.level->setRange(1, 100);
    statsInput.level->setValue(lastLevel);
    for (int i = 0; i < 6; i++) {
        statsInput.stats[i] = new QSpinBox();
        statsInput.stats[i]->setRange(0, 999);
    }
    m_statsLayout->addWidget(widgetFromStatsInput((int)m_pokemonInput->statsInputs.size() - 1));
    return statsInput;
}

QWidget* PokemonStatCalculator::widgetFromStatsInput(int index) {
    auto& statsInput = m_pokemonInput->statsInputs[index];
    QGroupBox* groupBox = new QGroupBox(QString("Stats"));
    QFormLayout* form = new QFormLayout(groupBox);
    QHBoxLayout* statsBox = new QHBoxLayout();
    for (int i = 0; i < 6; i++) {
        statsBox->addWidget(statsInput.stats[i]);
    }
    form->addRow("Species:", statsInput.species);
    form->addRow("Level:", statsInput.level);
    form->addRow("Stats:", statsBox);
    QPushButton* removeButton = new QPushButton("Remove", groupBox);
    connect(removeButton, &QPushButton::clicked, this, [=](){
        removeStatsInput(index);
    });
    form->addWidget(removeButton);
    m_stats.push_back(groupBox);
    return groupBox;
}

void PokemonStatCalculator::onFinished() {
    updateIVRanges();
    bool valid = true;
    for (int i = 0; i < 6; i++) {
        if (m_pokemonInput->verifCtx.ivs[i][1] < m_pokemonInput->verifCtx.ivs[i][0]) {
            valid = false;
            break;
        }
    }
    if (valid) {
        close();
    } else {
        QMessageBox::information(this, "Error", "Invalid stats. Could not compute IV ranges.");
    }
}

void PokemonStatCalculator::removeStatsInput(int index) {
    auto widget = m_stats[index];
    m_statsLayout->removeWidget(widget);
    widget->deleteLater();
    m_stats.erase(m_stats.begin() + index);
    m_pokemonInput->statsInputs.erase(m_pokemonInput->statsInputs.begin() + index);
    m_addStatsButton->setEnabled((qsizetype)m_pokemonInput->statsInputs.size() < m_species.size());
    adjustSize();
}

void PokemonStatCalculator::updateIVRanges() {
    m_pokemonInput->calculateIVs();
    for (int i = 0; i < 6; i++) {
        const char* styleSheet = "color: black;";
        if (m_pokemonInput->verifCtx.ivs[i][1] < m_pokemonInput->verifCtx.ivs[i][0]) {
            styleSheet = "color: red;";
        } else if (m_pokemonInput->verifCtx.ivs[i][0] == m_pokemonInput->verifCtx.ivs[i][1]) {
            styleSheet = "color: green;";
        }
        m_ivs[i]->setText(QString("%1 - %2").arg((uint32_t)m_pokemonInput->verifCtx.ivs[i][0]).arg((uint32_t)m_pokemonInput->verifCtx.ivs[i][1]));
        m_ivs[i]->setStyleSheet(styleSheet);
    }
}

void PokemonInput::calculateIVs() {
    uint8_t effortLevels[6];
    for (int i = 0; i < 6; i++) {
        effortLevels[i] = (uint8_t)els[i]->value();
    }
    calculatePLAELRanges(effortLevels, verifCtx.ivs);
    bool hasRatings = false;
    JudgeIVRating judgeRatings[6] = {NOGOOD};
    for (int i = 0; i < 6; i++) {
        if (ratings.ratings[i]->currentIndex() != 0) {
            hasRatings = true;
            judgeRatings[i] = (JudgeIVRating)(ratings.ratings[i]->currentIndex() - 1);
        } else {
            judgeRatings[i] = NOGOOD;
        }
    }
    if (hasRatings) {
        restrictRangesForJudging(judgeRatings, verifCtx.ivs);
    }

    std::vector<ObservedStatInstance> stats;
    for (auto& statInput : statsInputs) {
        ObservedStatInstance& statInst = stats.emplace_back();
        statInst.speciesId = PokemonData::getSpeciesID(statInput.species->currentText().toStdString());
        statInst.level = statInput.level->value();
        for (int i = 0; i < 6; i++) {
            statInst.stats[i] = statInput.stats[i]->value();
        }
        restrictRangesForActualStats(statInst, nature->currentIndex(), verifCtx.ivs);
    }
    for (int i = 0; i < 6; i++) {
        const char* styleSheet = "color: black;";
        if (verifCtx.ivs[i][1] < verifCtx.ivs[i][0]) {
            styleSheet = "color: red;";
        } else if (verifCtx.ivs[i][0] == verifCtx.ivs[i][1]) {
            styleSheet = "color: green;";
        }
        computedIVs[i][0]->setText(QString("%1").arg(verifCtx.ivs[i][0]));
        computedIVs[i][0]->setStyleSheet(styleSheet);
        computedIVs[i][1]->setText(QString("%1").arg(verifCtx.ivs[i][1]));
        computedIVs[i][1]->setStyleSheet(styleSheet);
    }
    int numIVs = getNumIVPermutations(verifCtx.ivs);
    numIVCombos->setText(QString("%1").arg(numIVs));

    calculateSeeds();
}

void formatSeeds(uint64_t seeds, char& specifier, double& seedsDouble) {
    if (seeds >= 1000000000) {
        specifier = 'B';
        seedsDouble = seeds/1000000000.0;
    } else if (seeds >= 1000000) {
        specifier = 'M';
        seedsDouble = seeds/1000000.0;
    } else if (seeds >= 1000) {
        specifier = 'K';
        seedsDouble = seeds/1000.0;
    } else {
        specifier = ' ';
        seedsDouble = (double)seeds;
    }
}

void PokemonInput::calculateSeeds() {
    bool valid = slotGroup && !species->currentText().isEmpty();
    valid &= verifCtx.height[1] >= verifCtx.height[0];
    valid &= verifCtx.weight[1] >= verifCtx.weight[0];
    for (int i = 0; i < 6; i++) {
        valid &= verifCtx.ivs[i][1] >= verifCtx.ivs[i][0];
    }
    valid &= level->value() != 0;
    const PokemonSlot* speciesSlot = nullptr;
    float slotRateBase = 0;
    if (valid) {
        uint32_t speciesId = PokemonData::getSpeciesID(species->currentText().toStdString());
        for (int i = 0; i < slotGroup->numSlots(); i++) {
            auto& slot = slotGroup->getSlotFromIndex(i);
            if (slot.m_species == speciesId) {
                speciesSlot = &slot;
                break;
            }
            slotRateBase += slot.m_rate;
        }
    }
    if (valid && speciesSlot) {
        uint32_t speciesId = PokemonData::getSpeciesID(species->currentText().toStdString());
        auto& data = PokemonData::getSpeciesData(speciesId);
        uint8_t levelRange = 1 + speciesSlot->m_levelRange.second - speciesSlot->m_levelRange.first;
        uint8_t levelBitMask = levelRange == 1 ? 1 : (1<<std::bit_width((uint8_t)(levelRange-1))) - 1;
        verifCtx.speciesId = speciesId;
        verifCtx.level = (uint8_t)level->value();
        verifCtx.slotThresholds[0] = slotRateBase;
        verifCtx.slotThresholds[1] = slotRateBase+speciesSlot->m_rate;
        verifCtx.levelRange[0] = speciesSlot->m_levelRange.first;
        verifCtx.levelRange[1] = levelRange;
        verifCtx.levelRange[2] = levelBitMask;
        verifCtx.genderData[0] = data.genderRatio;
        verifCtx.genderData[1] = (uint8_t)gender->currentIndex();
        verifCtx.nature = (uint8_t)nature->currentIndex();
        if (ability->currentIndex() != 0) {
            uint8_t abil = ability->currentIndex() - 1;
            verifCtx.ability[0] = abil;
            verifCtx.ability[1] = abil;
        } else {
            verifCtx.ability[0] = 0;
            verifCtx.ability[1] = 1;
        }
        uint64_t seeds = getExpectedSeedsWithSizePairs(verifCtx, sizePairs);
        char specifier;
        double seedsDouble;
        formatSeeds(seeds, specifier, seedsDouble);
        numSeeds->setText(QString("%1%2").arg(seedsDouble, 0, 'f', 1).arg(specifier));

        double genCost = getTheoreticalGeneratorSeedsWithSizePairs(verifCtx, sizePairs, slotGroup->getSlotRateSum()) / 1000.0;
        generatorCost->setText(QString("%1").arg(genCost));
        canGen = (genCost * getNumIVPermutations(verifCtx.ivs)) < 64;

        verifValid = true;
    } else {
        numSeeds->setText("");
        generatorCost->setText("");
        canGen = false;
        verifValid = false;
    }
}

void PokemonInputTab::populateFromJSON(std::string filePath) {
    std::vector<PokemonVerificationContext> pokemon;
    std::vector<std::vector<ObservedSizeInstance>> sizes;
    uint64_t tableID;
    int shinyRolls = 0;
    parseJSONMMOEncounter(filePath, pokemon, sizes, shinyRolls, tableID);

    for (int i = 0; i < m_indexToOutbreakTable.size(); i++) {
        if (m_indexToOutbreakTable[i] == tableID) {
            m_outbreakSpeciesComboBox->setCurrentIndex(i);
        }
    }

    for (int i = 0; i < 4; i++) {
        PokemonInput* input = m_pokemonInputs[i];
        PokemonVerificationContext& ctx = pokemon[i];
        input->species->setCurrentText(QString("%1").arg(PokemonData::getSpeciesName(ctx.speciesId)));
        input->level->setValue(ctx.level);
        for (int j = 0; j < 6; j++) {
            input->els[j]->setValue(0);
        }
        input->nature->setCurrentIndex(ctx.nature);
        input->gender->setCurrentIndex(ctx.genderData[1]);
        input->ability->setCurrentIndex(ctx.ability[0] == ctx.ability[1] ? ctx.ability[0] + 1 : 0);

        PokemonSizeCalculator* sizeCalc = new PokemonSizeCalculator(input, this);
        auto& observedSizes = sizes[i];
        input->sizeInputs.clear();
        for (int i = 0; i < observedSizes.size(); i++) {
            auto& measurement = observedSizes[i];
            auto& sizeInput = sizeCalc->addNewMeasurement();
            sizeInput.species->setCurrentText(QString("%1").arg(PokemonData::getSpeciesName(measurement.speciesId)));
            sizeInput.height[0]->setValue((int)(measurement.height) / 12);
            sizeInput.height[1]->setValue((int)(measurement.height) % 12);
            sizeInput.weight->setValue(measurement.weight);
        }
        input->computedHeight[0]->setText(QString("%1").arg(ctx.height[0]));
        input->computedHeight[1]->setText(QString("%1").arg(ctx.height[1]));
        input->computedWeight[0]->setText(QString("%1").arg(ctx.weight[0]));
        input->computedWeight[1]->setText(QString("%1").arg(ctx.weight[1]));

        for (int j = 0; j < 6; j++) {
            input->computedIVs[j][0]->setText(QString("%1").arg(ctx.ivs[j][0]));
            input->computedIVs[j][1]->setText(QString("%1").arg(ctx.ivs[j][1]));
        }
        input->verifCtx = ctx;
        input->verifValid = true;

        uint64_t seeds = getExpectedSeeds(ctx);
        char specifier;
        double seedsDouble;
        formatSeeds(seeds, specifier, seedsDouble);
        input->numSeeds->setText(QString("%1%2").arg(seedsDouble, 0, 'f', 1).arg(specifier));

        auto& slotGroup = PokemonData::getSlotGroupTable(tableID);
        input->slotGroup = &slotGroup;
        const PokemonSlot* speciesSlot = nullptr;
        for (int i = 0; i < slotGroup.numSlots(); i++) {
            auto& slot = slotGroup.getSlotFromIndex(i);
            if (slot.m_species == ctx.speciesId) {
                speciesSlot = &slot;
                break;
            }
        }
        double genCost = getTheoreticalGeneratorSeeds(ctx, slotGroup.getSlotRateSum()) / 1000.0;
        input->generatorCost->setText(QString("%1").arg(genCost));
        input->canGen = (genCost * getNumIVPermutations(input->verifCtx.ivs)) < 64;
    }
}

void PokemonInputTab::saveToJSON(std::string filePath) {
    json jsonObj;
    jsonObj["shinyRolls"] = 1;
    std::stringstream ss;
    ss << "0x" << std::hex << getSelectedOutbreakTable();
    jsonObj["tableID"] = ss.str();
    json pokemonList = json::array();
    for (int i = 0; i < 4; i++) {
        PokemonInput* input = m_pokemonInputs[i];
        auto& data = PokemonData::getSpeciesData(PokemonData::getSpeciesID(input->species->currentText().toStdString()));
        json pokemonObj;
        pokemonObj["name"] = input->species->currentText().toStdString();
        pokemonObj["nature"] = input->nature->currentText().toStdString();
        if (data.genderRatio < 254 && data.genderRatio > 0) {
            constexpr std::string genderStrings[2] = {"M", "F"};
            pokemonObj["gender"] = genderStrings[input->gender->currentIndex()];
        }
        pokemonObj["level"] = input->level->value();
        pokemonObj["effortLevels"] = json::array();
        for (int i = 0; i < 6; i++) {
            pokemonObj["effortLevels"].push_back(input->els[i]->value());
        }
        if (input->ability->currentIndex()) {
            pokemonObj["ability"] = input->ability->currentText().toStdString();
        }
        if (input->ratings.ratings[0]->currentIndex()) {
            pokemonObj["ratings"] = json::array();
            for (int i = 0; i < 6; i++) {
                int rating = input->ratings.ratings[i]->currentIndex();
                pokemonObj["ratings"].push_back(RATING_NAMES[rating ? rating - 1 : 0]);
            }
        }
        pokemonObj["sizes"] = json::array();
        for (int i = 0; i < input->sizeInputs.size(); i++) {
            SizeInput& sizeInput = input->sizeInputs[i];
            json sizeJson;
            sizeJson["name"] = sizeInput.species->currentText().toStdString();
            sizeJson["height_ft"] = sizeInput.height[0]->value();
            sizeJson["height_in"] = sizeInput.height[1]->value();
            sizeJson["weight_lbs"] = sizeInput.weight->value();
            pokemonObj["sizes"].push_back(sizeJson);
        }

        if (!input->statsInputs.empty()) {
            pokemonObj["stats"] = json::array();
            for (int i = 0; i < input->statsInputs.size(); i++) {
                StatsInput& statsInput = input->statsInputs[i];
                json statJson;
                statJson["name"] = statsInput.species->currentText().toStdString();
                statJson["level"] = statsInput.level->value();
                statJson["stats"] = json::array();
                for (int j = 0; j < 6; j++) {
                    statJson["stats"].push_back(statsInput.stats[j]->value());
                }
                pokemonObj["stats"].push_back(statJson);
            }
        }

        pokemonList.push_back(pokemonObj);
    }
    jsonObj["pokemon"] = pokemonList;

    // Open a file for writing
    std::ofstream file(filePath);

    // Write the JSON object to the file
    file << std::setw(4) << jsonObj << std::endl;
}