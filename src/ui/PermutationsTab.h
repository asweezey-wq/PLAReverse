#ifndef PERMUTATIONSTAB_H
#define PERMUTATIONSTAB_H

#include "PokemonTab.hpp"
#include "permutations.hpp"
#include <QComboBox>
#include <QGroupBox>
#include <QSpinBox>
#include <QLineEdit>
#include <QTableWidget>
#include <QCheckBox>
#include <vector>

struct ShinyInfo {
    uint32_t speciesId;
    QGroupBox* groupBox;
    QCheckBox *dex10, *dexPerfect, *shinyCharm;
};

class PokemonInputTab;
class PermutationsTab : public PokemonTab
{
    Q_OBJECT
public:
    explicit PermutationsTab(PokemonInputTab* inputTab, QWidget *parent = nullptr);
    void updateSecondWaveTables();
    void beginPermutations();
    uint64_t getPrimaryOutbreakTable() const;
    uint64_t getSecondaryOutbreakTable() const;
protected:
    void onTabShown() override;
private:
    PokemonInputTab* m_inputTab;
    QLineEdit* m_seed;
    QComboBox* m_firstWave;
    QSpinBox* m_numFirstWave;
    QComboBox* m_secondWave;
    QSpinBox* m_numSecondWave;
    QStringList m_secondWaveOptions;
    std::vector<uint64_t> m_secondWaveOutbreakTables;

    QVBoxLayout* m_shinyRollsVBox;
    std::vector<ShinyInfo> m_shinyInfo;

    QTableWidget* m_tableWidget;
    void addResultToTable(PermutationResult result);
};

#endif // PERMUTATIONSTAB_H
