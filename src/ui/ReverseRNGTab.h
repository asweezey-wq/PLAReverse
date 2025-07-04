#ifndef REVERSERNGTAB_H
#define REVERSERNGTAB_H

#include "PokemonTab.hpp"
#include "SeedReversalWorker.h"
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QListWidget>
#include <QProgressBar>
#include <QThread>

struct PokemonInfo {
    QLineEdit* info;
    QLineEdit* genCost;
    QPushButton* button;
    QCheckBox* shinyCharm;
    QCheckBox* dex10;
    QCheckBox* dexPerfect;
};

class PokemonInputTab;
class ReverseRNGTab : public PokemonTab
{
    Q_OBJECT
public:
    explicit ReverseRNGTab(PokemonInputTab* inputs, QWidget *parent = nullptr);
    void onTabShown() override;
    void beginSeedReversal(int index);
    void updateProgress(int stage);
private:
    PokemonInputTab* m_inputTab;
    std::vector<PokemonInfo> m_pokemonUiInfo;
    QListWidget* m_seedList;
    QProgressBar* m_progressBar;

    void updateShinyCharm(bool value);
    void updateDexPerfect(int index, bool value);
    void updateDex10(int index, bool value);

    bool m_reversalInProgress = false;
    QThread* m_reversalThread = nullptr;
    SeedReversalWorker* m_worker = nullptr;
    void setRNGReversalStatus(bool inProgress);
};

#endif // REVERSERNGTAB_H
