#ifndef PERMUTATIONSTAB_H
#define PERMUTATIONSTAB_H

#include "PokemonTab.hpp"
#include <QWidget>
#include <QStandardItemModel>

namespace Ui {
class PermutationsTab;
}

class PermutationsTab : public PokemonTab
{
    Q_OBJECT

public:
    explicit PermutationsTab(QWidget *parent = nullptr);
    ~PermutationsTab();

public slots:
    void loadPermutationsFromSeed(quint64 seed);

private:
    Ui::PermutationsTab *ui;
    QStandardItemModel *permutationModel;
};

#endif // PERMUTATIONSTAB_H
