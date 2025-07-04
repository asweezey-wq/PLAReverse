#ifndef SEEDREVERSALWORKER_H
#define SEEDREVERSALWORKER_H

#include "pokemonCuda.hpp"
#include <QObject>

class SeedReversalWorker : public QObject
{
    Q_OBJECT
public:
    explicit SeedReversalWorker(SeedReversalContext reversalCtx, QObject *parent = nullptr);

public slots:
    void run();  // called when the thread starts

signals:
    void progressUpdated(int stage);
    void reversalFinished(uint32_t numSeeds, uint64_t* seedBuf); 
    void reversalFailed();

private:
    SeedReversalContext m_reversalCtx;
};

#endif // SEEDREVERSALWORKER_H
