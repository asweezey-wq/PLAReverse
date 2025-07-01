#include "MainWindow.h"
#include "PokemonInputTab.h"
#include "ReverseRNGTab.h"
#include "PermutationsTab.h"

#include <QTabWidget>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setFixedSize(1200, 720);
    QWidget* central = new QWidget(this);
    QVBoxLayout* vbox = new QVBoxLayout();
    QTabWidget* tabWidget = new QTabWidget(this);

    // Create and add tabs
    PokemonInputTab* inputTab = new PokemonInputTab(this);
    ReverseRNGTab* rngTab = new ReverseRNGTab(inputTab, this);
    PermutationsTab* permTab = new PermutationsTab(this);

    m_tabs.push_back(inputTab);
    m_tabs.push_back(rngTab);
    m_tabs.push_back(permTab);
    tabWidget->addTab(m_tabs[0], tr("Pokémon Input"));
    tabWidget->addTab(m_tabs[1], tr("Reverse RNG"));
    tabWidget->addTab(m_tabs[2], tr("Permutations"));
    connect(tabWidget, &QTabWidget::currentChanged, this, &MainWindow::tabChanged);
    vbox->addWidget(tabWidget);

    central->setLayout(vbox);
    setCentralWidget(central);
}

void MainWindow::tabChanged(int index) {
    m_tabs[index]->onTabShown();
}