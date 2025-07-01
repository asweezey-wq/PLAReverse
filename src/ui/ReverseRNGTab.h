#ifndef REVERSERNGTAB_H
#define REVERSERNGTAB_H

#include "PokemonTab.hpp"
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>

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
private:
    PokemonInputTab* m_inputTab;
    std::vector<PokemonInfo> m_pokemonUiInfo;

    void updateShinyCharm(bool value);
    void updateDexPerfect(int index, bool value);
    void updateDex10(int index, bool value);
};

#endif // REVERSERNGTAB_H
