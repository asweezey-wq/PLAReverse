#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "PokemonTab.hpp"
#include <QMainWindow>
#include <vector>

class PokemonInputTab;
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    void populateInputJSON(std::string filePath);
    void tabChanged(int index);
private:
    PokemonInputTab* m_inputTab;
    std::vector<PokemonTab*> m_tabs;
};

#endif // MAINWINDOW_H
