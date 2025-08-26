#include "PermutationsTab.h"
#include "PokemonInputTab.h"
#include "pokemonData.hpp"
#include <QVBoxLayout>
#include <QLineEdit>
#include <QHeaderView>
#include <sstream>
#include <set>

PermutationsTab::PermutationsTab(PokemonInputTab* inputTab, QWidget *parent)
    : PokemonTab(parent), m_inputTab(inputTab)
{

    QVBoxLayout* vbox = new QVBoxLayout();
    QHBoxLayout* hbox = new QHBoxLayout();
    QVBoxLayout* outbreakVbox = new QVBoxLayout();
    m_seed = new QLineEdit(this);
    m_seed->setPlaceholderText("Group Seed");

    outbreakVbox->addWidget(m_seed);

    m_firstWave = new QComboBox();
    m_firstWave->addItems(inputTab->m_outbreakSpeciesList);
    connect(m_firstWave, &QComboBox::currentIndexChanged, this, &PermutationsTab::updateSecondWaveTables);
    m_numFirstWave = new QSpinBox();
    m_numFirstWave->setRange(4, 100);
    m_secondWave = new QComboBox();
    m_numSecondWave = new QSpinBox();
    m_numSecondWave->setRange(4, 8);

    outbreakVbox->addWidget(m_firstWave);
    outbreakVbox->addWidget(m_numFirstWave);
    outbreakVbox->addWidget(m_secondWave);
    outbreakVbox->addWidget(m_numSecondWave);

    m_shinyRollsVBox = new QVBoxLayout();

    hbox->addLayout(outbreakVbox);
    hbox->addLayout(m_shinyRollsVBox);
    vbox->addLayout(hbox);

    QPushButton* startButton = new QPushButton("Start");
    connect(startButton, &QPushButton::clicked, this, &PermutationsTab::beginPermutations);
    vbox->addWidget(startButton);

    m_tableWidget = new QTableWidget();
    m_tableWidget->setColumnCount(9);
    m_tableWidget->setHorizontalHeaderLabels({"Path", "Species", "Shiny", "Alpha", "Rolls", "IVs", "Nature", "Level", "Gender"});
    // m_tableWidget->verticalHeader()->setVisible(false);
    vbox->addWidget(m_tableWidget);

    setLayout(vbox);
    updateSecondWaveTables();
}

uint64_t PermutationsTab::getPrimaryOutbreakTable() const {
    int index = m_firstWave->currentIndex();
    return m_inputTab->m_indexToOutbreakTable[index];
}

uint64_t PermutationsTab::getSecondaryOutbreakTable() const {
    int index = m_secondWave->currentIndex();
    return m_secondWaveOutbreakTables[index];
}

void PermutationsTab::updateSecondWaveTables() {
    std::set<uint32_t> uniqueSpecies;
    auto& slotGroup = PokemonData::getSlotGroupTable(getPrimaryOutbreakTable());
    uint32_t speciesId = slotGroup.getSlotFromIndex(0).m_species;
    for (int i = 0; i < slotGroup.numSlots(); i++) {
        uniqueSpecies.insert(slotGroup.getSlotFromIndex(i).m_species);
    }

    m_secondWave->clear();
    m_secondWaveOptions.clear();
    m_secondWaveOutbreakTables.clear();

    m_secondWaveOptions.append(QString());
    m_secondWaveOutbreakTables.push_back(0);

    for (auto& pair : PokemonData::getOutbreakTables()) {
        auto& firstSlot = pair.second.getSlotFromIndex(0);
        if (firstSlot.m_species == speciesId && firstSlot.m_isAlpha) {
            m_secondWaveOptions.append(QString("%1 (Alpha)").arg(PokemonData::getSpeciesName(speciesId)));
            m_secondWaveOutbreakTables.push_back(pair.first);
        }
    }

    if (slotGroup.numSlots() > 2) {
        uint32_t evoSpeciesId = slotGroup.getSlotFromIndex(2).m_species;
        for (auto& pair : PokemonData::getOutbreakTables()) {
            auto& firstSlot = pair.second.getSlotFromIndex(0);
            if (firstSlot.m_species == evoSpeciesId && pair.second.numSlots() <= 2) {
                if (firstSlot.m_isAlpha) {
                    m_secondWaveOptions.append(QString("%1 (Alpha)").arg(PokemonData::getSpeciesName(evoSpeciesId)));
                    m_secondWaveOutbreakTables.push_back(pair.first);
                } else {
                    m_secondWaveOptions.append(QString("%1").arg(PokemonData::getSpeciesName(evoSpeciesId)));
                    m_secondWaveOutbreakTables.push_back(pair.first);
                }
                for (int i = 0; i < pair.second.numSlots(); i++) {
                    uniqueSpecies.insert(pair.second.getSlotFromIndex(i).m_species);
                }
            }
        }
    }

    m_secondWave->addItems(m_secondWaveOptions);

    for (auto& shinyInfo : m_shinyInfo) {
        m_shinyRollsVBox->removeWidget(shinyInfo.groupBox);
        shinyInfo.groupBox->deleteLater();
    }
    m_shinyInfo.clear();
    for (auto& species : uniqueSpecies) {
        ShinyInfo& shinyInfo = m_shinyInfo.emplace_back(species);
        shinyInfo.groupBox = new QGroupBox(QString("%1").arg(PokemonData::getSpeciesName(species)));
        shinyInfo.dex10 = new QCheckBox("Dex 10", shinyInfo.groupBox);
        shinyInfo.dexPerfect = new QCheckBox("Dex Perfect", shinyInfo.groupBox);
        shinyInfo.shinyCharm = new QCheckBox("Shiny Charm", shinyInfo.groupBox);
        QHBoxLayout* hbox = new QHBoxLayout();
        hbox->addWidget(shinyInfo.dex10);
        hbox->addWidget(shinyInfo.dexPerfect);
        hbox->addWidget(shinyInfo.shinyCharm);
        shinyInfo.groupBox->setLayout(hbox);
        m_shinyRollsVBox->addWidget(shinyInfo.groupBox);
    }
}

void PermutationsTab::onTabShown() {
    m_firstWave->setCurrentIndex(m_inputTab->getSelectedSpeciesIndex());
}

void PermutationsTab::beginPermutations() {
    try {
        uint64_t seed = std::stoull(m_seed->text().toStdString());
        ShinyRollData shinyRolls;
        for (auto& shinyInfo : m_shinyInfo) {
            int rolls = 13;
            if (shinyInfo.dex10->isChecked()) {
                rolls += 1;
            }
            if (shinyInfo.dexPerfect->isChecked()){ 
                rolls += 2;
            }
            if (shinyInfo.shinyCharm->isChecked()) {
                rolls += 3;
            }
            shinyRolls[shinyInfo.speciesId] = rolls;
        }
        PermutationsManager manager(seed, getPrimaryOutbreakTable(), m_numFirstWave->value(), getSecondaryOutbreakTable(), m_numSecondWave->value(), shinyRolls);
        auto results = manager.findPermutations();
        m_tableWidget->setRowCount(0);
        for (auto& result : results) {
            addResultToTable(result);
        }
        m_tableWidget->resizeColumnsToContents();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::out_of_range&) {
        return;
    }
}

void PermutationsTab::addResultToTable(PermutationResult result) {
    int row = m_tableWidget->rowCount();
    m_tableWidget->insertRow(row);
    std::stringstream ss;
    bool first = true;
    for (auto& action : result.actionChain) {
        if (!first) {
            ss << " | ";
        }
        switch(action.actionType) {
            case CATCH:
                ss << "C" << action.count;
                break;
            case GHOST:
                ss << "G" << action.count;
                break;
            case CLEAR_REMAINING:
                ss << "CR";
                break;
        }
        first = false;
    }
    m_tableWidget->setItem(row, 0, new QTableWidgetItem(QString("%1 ").arg(ss.str())));
    m_tableWidget->setItem(row, 1, new QTableWidgetItem(QString("%1").arg(PokemonData::getSpeciesName(result.pokemon.m_species))));
    m_tableWidget->setItem(row, 2, new QTableWidgetItem(result.pokemon.m_isShiny ? "YES" : ""));
    m_tableWidget->setItem(row, 3, new QTableWidgetItem(result.pokemon.m_isAlpha ? "YES" : ""));
    m_tableWidget->setItem(row, 4, new QTableWidgetItem(QString("%1").arg(result.pokemon.m_shinyRolls)));
    m_tableWidget->setItem(row, 5, new QTableWidgetItem(QString("%1/%2/%3/%4/%5/%6").arg(result.pokemon.m_ivs[0], 2, 10, QLatin1Char('0')).arg(result.pokemon.m_ivs[1], 2, 10, QLatin1Char('0')).arg(result.pokemon.m_ivs[2], 2, 10, QLatin1Char('0')).arg(result.pokemon.m_ivs[3], 2, 10, QLatin1Char('0')).arg(result.pokemon.m_ivs[4], 2, 10, QLatin1Char('0')).arg(result.pokemon.m_ivs[5], 2, 10, QLatin1Char('0'))));
    m_tableWidget->setItem(row, 6, new QTableWidgetItem(QString("%1").arg(NATURE_NAMES[result.pokemon.m_nature])));
    m_tableWidget->setItem(row, 7, new QTableWidgetItem(QString("%1").arg(result.pokemon.m_level)));
    m_tableWidget->setItem(row, 8, new QTableWidgetItem(QString("%1").arg(result.pokemon.m_gender)));
}