#include "ReverseRNGTab.h"
#include "PokemonInputTab.h"
#include "SeedReversalWorker.h"
#include "seedReversal.hpp"
#include "pokemonCuda.hpp"
#include <QPushButton>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QTextEdit>
#include <QLabel>

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

    QHBoxLayout* hbox = new QHBoxLayout();
    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setAlignment(Qt::AlignCenter);
    hbox->addWidget(m_progressBar);

    vbox->addLayout(hbox);

    m_seedList = new QListWidget(this);
    vbox->addWidget(new QLabel("Seeds"));
    vbox->addWidget(m_seedList);

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
        m_pokemonUiInfo[i].button->setEnabled(allValid && m_inputTab->getPokemonInputs()[i]->canGen);
    }
}

void ReverseRNGTab::beginSeedReversal(int index) {
    if (m_reversalInProgress) {
        return;
    }
    m_inputTab->saveToJSON("tmp.json");
    m_seedList->clear();
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
    createSizePairsForCUDA(m_inputTab->getPokemonInputs()[index]->sizePairs, reversalCtx.primarySizePairs);
    setRNGReversalStatus(true);
    updateProgress(0);
    SeedReversalWorker* worker = new SeedReversalWorker(reversalCtx);
    m_worker = worker;
    m_reversalThread = new QThread();

    worker->moveToThread(m_reversalThread);

    // Connect worker thread lifecycle
    connect(m_reversalThread, &QThread::started, worker, &SeedReversalWorker::run);
    connect(worker, &SeedReversalWorker::reversalFinished, m_reversalThread, &QThread::quit);
    connect(worker, &SeedReversalWorker::reversalFailed, m_reversalThread, &QThread::quit);
    connect(m_reversalThread, &QThread::finished, worker, &QObject::deleteLater);
    connect(m_reversalThread, &QThread::finished, m_reversalThread, &QObject::deleteLater);

    // Connect progress signals to UI
    connect(worker, &SeedReversalWorker::progressUpdated, this, &ReverseRNGTab::updateProgress);
    // connect(worker, &SeedReversalWorker::statusMessage, statusLabel, &QLabel::setText);

    // Optional: show result
    connect(worker, &SeedReversalWorker::reversalFinished, this, [=](uint32_t numSeeds, uint64_t* seedBuf){
        m_progressBar->setFormat(QString("Found %1 seed(s)!").arg(numSeeds));
        m_progressBar->setValue(100);
        for (int i = 0; i < numSeeds; i++) {
            m_seedList->addItem(QString("%1").arg(seedBuf[i]));
        }
        setRNGReversalStatus(false);
        m_reversalThread = nullptr;
        m_worker = nullptr;
    });

    m_reversalThread->start();
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

void ReverseRNGTab::updateProgress(int stage) {
    switch (stage) {
        case 0:
            m_progressBar->setFormat("");
            m_progressBar->setValue(0);
            break;
        case 1:
            m_progressBar->setFormat("Finding Pokemon Seeds");
            m_progressBar->setValue(15);
            break;
        case 2:
            m_progressBar->setFormat("Finding Generator Seeds");
            m_progressBar->setValue(30);
            break;
        case 3:
            m_progressBar->setFormat("Finding Group Seeds (index 0)");
            m_progressBar->setValue(45);
            break;
        case 4:
            m_progressBar->setFormat("Finding Group Seeds (index 1)");
            m_progressBar->setValue(60);
            break;
        case 5:
            m_progressBar->setFormat("Finding Group Seeds (index 2)");
            m_progressBar->setValue(75);
            break;
        case 6:
            m_progressBar->setFormat("Finding Group Seeds (index 3)");
            m_progressBar->setValue(90);
            break;
        case 7:
            m_progressBar->setFormat("Finished!");
            m_progressBar->setValue(100);
            break;
        default:
            m_progressBar->setFormat("");
            m_progressBar->setValue(0);
            break;
    }
}

void ReverseRNGTab::setRNGReversalStatus(bool inProgress) {
    m_reversalInProgress = inProgress;
    for (int i = 0; i < 4; i++) {
        m_pokemonUiInfo[i].button->setEnabled(!inProgress && m_inputTab->getPokemonInputs()[i]->canGen);
    }
}