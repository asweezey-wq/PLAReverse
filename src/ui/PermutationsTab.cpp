#include "PermutationsTab.h"
#include "ui_PermutationsTab.h"

PermutationsTab::PermutationsTab(QWidget *parent)
    : PokemonTab(parent),
      ui(new Ui::PermutationsTab),
      permutationModel(new QStandardItemModel(this))
{
    ui->setupUi(this);

    QStringList headers = {"Species", "Index", "Shiny", "Alpha", "Slot Type"};
    permutationModel->setHorizontalHeaderLabels(headers);
    ui->permutationTable->setModel(permutationModel);
    ui->permutationTable->horizontalHeader()->setStretchLastSection(true);
}

PermutationsTab::~PermutationsTab()
{
    delete ui;
}

void PermutationsTab::loadPermutationsFromSeed(quint64 seed)
{
    permutationModel->clear();

    QStringList headers = {"Species", "Index", "Shiny", "Alpha", "Slot Type"};
    permutationModel->setHorizontalHeaderLabels(headers);

    // Simulate some data — replace with your backend result
    for (int i = 0; i < 10; ++i) {
        QList<QStandardItem *> row;
        row << new QStandardItem(QString("Species %1").arg(i))
            << new QStandardItem(QString::number(i))
            << new QStandardItem(i % 3 == 0 ? "Yes" : "No")
            << new QStandardItem(i % 2 == 0 ? "Yes" : "No")
            << new QStandardItem("Main");

        permutationModel->appendRow(row);
    }

    ui->seedLabel->setText(tr("Seed: 0x%1").arg(seed, 16, 16, QLatin1Char('0')));
}
