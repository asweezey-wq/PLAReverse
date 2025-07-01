#ifndef POKEMONINPUTTAB_H
#define POKEMONINPUTTAB_H

#include "PokemonTab.hpp"
#include <QDialog>
#include <QGroupBox>

class PokemonSizeCalculator : public QDialog {
    Q_OBJECT
public:
    explicit PokemonSizeCalculator(PokemonInput* pokemonInput, QWidget *parent = nullptr);
    SizeInput& addNewMeasurement();
    QWidget* widgetFromSizeInput(int index);
    void removeSizeInput(int index);
    void onFinished();
private:
    uint32_t m_baseSpecies;
    QStringList m_species;
    PokemonInput* m_pokemonInput;

    QPushButton* m_addMeasurementButton;
    QVBoxLayout* m_measurementLayout;
    std::vector<QWidget*> m_measurements;
};

class PokemonStatCalculator : public QDialog {
    Q_OBJECT
public:
    explicit PokemonStatCalculator(PokemonInput* pokemonInput, QWidget *parent = nullptr);
    StatsInput& addNewStats();
    QWidget* widgetFromStatsInput(int index);
    void removeStatsInput(int index);
    void onFinished();
    void updateIVRanges();
private:
    uint32_t m_baseSpecies;
    QStringList m_species;
    PokemonInput* m_pokemonInput;

    QPushButton* m_addStatsButton;
    QVBoxLayout* m_statsLayout;
    std::vector<QWidget*> m_stats;
    std::vector<QLineEdit*> m_ivs;
};

class PokemonInputTab : public PokemonTab
{
    Q_OBJECT
public:
    explicit PokemonInputTab(QWidget *parent = nullptr);
    const std::vector<PokemonInput*>& getPokemonInputs() const { return m_pokemonInputs; }
    uint64_t getSelectedOutbreakTable() const { return m_selectedOutbreakTable; }
private:
    std::vector<PokemonInput*> m_pokemonInputs;
    QStringList m_outbreakSpeciesList, m_natureList, m_ratingsList;
    std::vector<uint64_t> m_indexToOutbreakTable;

    QComboBox* m_outbreakSpeciesComboBox;
    uint64_t m_selectedOutbreakTable;
    QStringList m_genderList, m_abilityList, m_speciesList;

    void createForm();
    QGroupBox* createPokemonGroup(int index);
    void populateDropdowns(PokemonInput* input);

    void selectNewOutbreakSpecies(int index);
};

#endif // POKEMONINPUTTAB_H
