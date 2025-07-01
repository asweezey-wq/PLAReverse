#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "PokemonTab.hpp"
#include <QMainWindow>
#include <vector>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

    void tabChanged(int index);
private:
    std::vector<PokemonTab*> m_tabs;
};

#endif // MAINWINDOW_H
