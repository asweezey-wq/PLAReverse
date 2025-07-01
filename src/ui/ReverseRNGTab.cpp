#include "ReverseRNGTab.h"
#include "PokemonInputTab.h"
#include "seedReversal.hpp"
#include "pokemonCuda.h"
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QTextEdit>

ReverseRNGTab::ReverseRNGTab(PokemonInputTab* inputs, QWidget *parent)
    : PokemonTab(parent), m_inputTab(inputs)
{
    QVBoxLayout* vbox = new QVBoxLayout();

    QGroupBox* pokemonGroup = new QGroupBox("Pokemon", this);
    QVBoxLayout* pokemonVBox = new QVBoxLayout();
    for (int i = 0; i < 4; i++) {
        QHBoxLayout* pokemonHBox = new QHBoxLayout();
        PokemonInfo& info = m_pokemonUiInfo.emplace_back();
        info.info = new QLineEdit(this);
        info.info->setReadOnly(true);
        info.genCost = new QLineEdit(this);
        info.genCost->setReadOnly(true);
        info.button = new QPushButton("Start Reversal", this);
        connect(info.button, &QPushButton::clicked, this, [=](){
            beginSeedReversal(i);
        });

        info.shinyCharm = new QCheckBox("Shiny Charm", this);
        connect(info.shinyCharm, &QCheckBox::checkStateChanged, this, &ReverseRNGTab::updateShinyCharm);
        info.dex10 = new QCheckBox("Dex 10", this);
        connect(info.dex10, &QCheckBox::checkStateChanged, this, [=](bool value) {
            updateDex10(i, value);
        });
        info.dexPerfect = new QCheckBox("Dex Perfect", this);
        connect(info.dexPerfect, &QCheckBox::checkStateChanged, this, [=](bool value) {
            updateDexPerfect(i, value);
        });

        pokemonHBox->addWidget(info.info, 2);
        pokemonHBox->addWidget(info.genCost, 1);
        pokemonHBox->addWidget(info.dex10);
        pokemonHBox->addWidget(info.dexPerfect);
        pokemonHBox->addWidget(info.shinyCharm);
        pokemonHBox->addWidget(info.button, 1);
        pokemonVBox->addLayout(pokemonHBox);
    }
    pokemonGroup->setLayout(pokemonVBox);

    vbox->addWidget(pokemonGroup);

    QProgressBar* progressBar = new QProgressBar(this);
    vbox->addWidget(progressBar);

    QTextEdit* textEdit = new QTextEdit(this);
    textEdit->setReadOnly(true);
    vbox->addWidget(textEdit);

    setLayout(vbox);
}

void ReverseRNGTab::onTabShown() {
    bool allValid = true;
    for (int i = 0; i < 4; i++) {
        PokemonInfo& info = m_pokemonUiInfo[i];
        PokemonInput* input = m_inputTab->getPokemonInputs()[i];
        info.info->setText(QString("%1 Lv.%2 %3 %4")
            .arg(input->species->currentText())
            .arg(input->level->value())
            .arg(input->gender->currentText()[0])
            .arg(input->nature->currentText()));

        info.genCost->setText(QString("Seeds: %1 / GenCost: %2").arg(input->numSeeds->text()).arg(input->generatorCost->text()));
        allValid &= input->verifValid;
    }
    for (int i = 0; i < 4; i++) {
        m_pokemonUiInfo[i].button->setEnabled(allValid);
    }
}

void ReverseRNGTab::beginSeedReversal(int index) {
    int numShinyRolls = 13;
    if (m_pokemonUiInfo[index].dex10->isChecked()) {
        numShinyRolls += 1;
    }
    if (m_pokemonUiInfo[index].dexPerfect->isChecked()) {
        numShinyRolls += 2;
    }
    if (m_pokemonUiInfo[index].shinyCharm->isChecked()) {
        numShinyRolls += 3;
    }
    auto& slotGroup = PokemonData::getSlotGroupTable(m_inputTab->getSelectedOutbreakTable());
    
    std::vector<PokemonVerificationContext> pokemon;
    bool allValid = true;
    for (int i = 0; i < 4; i++) {
        PokemonInput* input = m_inputTab->getPokemonInputs()[i];
        allValid &= input->verifValid;
        pokemon.push_back(input->verifCtx);
    }
    if (!allValid) {
        return;
    }
    std::swap(pokemon[0], pokemon[index]);
    
    SeedReversalContext reversalCtx = createReversalCtxForCUDA(numShinyRolls, slotGroup);
    reversalCtx.numPokemon = 4;
    for (int i = 0; i < 4; i++) {
        reversalCtx.pokemonVerifCtx[i] = pokemon[i];
    }
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t* outputBuffer = new uint64_t[4];
    int results = reversePokemonFromSingleMon(reversalCtx, 0, outputBuffer);
    for (int i = 0; i < results; i++) {
        printf("Group seed: %llx %llu\n", outputBuffer[i], outputBuffer[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Seed search took %.02fs\n", (double)duration.count() / 1000000);
    delete[] outputBuffer;
}

void ReverseRNGTab::updateShinyCharm(bool value) {
    for (int i = 0; i < 4; i++) {
        m_pokemonUiInfo[i].shinyCharm->setChecked(value);
        if (value) {
            m_pokemonUiInfo[i].dex10->setChecked(true);
        }
    }
}

void ReverseRNGTab::updateDexPerfect(int index, bool value) {
    int speciesIndex = m_inputTab->getPokemonInputs()[index]->species->currentIndex();
    for (int i = 0; i < 4; i++) {
        if (m_inputTab->getPokemonInputs()[i]->species->currentIndex() == speciesIndex) {
            m_pokemonUiInfo[i].dexPerfect->setChecked(value);
            if (value) {
                m_pokemonUiInfo[i].dex10->setChecked(true);
            }
        }
    }
}

void ReverseRNGTab::updateDex10(int index, bool value) {
    int speciesIndex = m_inputTab->getPokemonInputs()[index]->species->currentIndex();
    for (int i = 0; i < 4; i++) {
        if (m_inputTab->getPokemonInputs()[i]->species->currentIndex() == speciesIndex) {
            m_pokemonUiInfo[i].dex10->setChecked(value);
            if (!value) {
                m_pokemonUiInfo[i].dexPerfect->setChecked(false);
                m_pokemonUiInfo[i].shinyCharm->setChecked(false);
            }
        }
    }
}