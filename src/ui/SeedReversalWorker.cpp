#include "SeedReversalWorker.h"
#include <QThread>
#include <ctime>

SeedReversalWorker::SeedReversalWorker(SeedReversalContext reversalCtx, QObject *parent)
    : QObject(parent), m_reversalCtx(reversalCtx)
{}

void SeedReversalWorker::run()
{
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t* outputBuffer = new uint64_t[4];
    HostReversalManager manager(m_reversalCtx, 0, outputBuffer);
    manager.setupReversal();
    emit progressUpdated(1);
    auto pokemonSeeds = manager.reversePokemonSeeds();
    emit progressUpdated(2);
    auto generatorSeeds = manager.reverseGeneratorSeeds();
    emit progressUpdated(3);
    if (!manager.reverseGroupSeedsFirstIndex()) {
        for (int i = 0; i < 3; i++) {
            emit progressUpdated(4+i);
            if (manager.reverseGroupSeeds(i)) {
                break;
            }
        }
    }
    auto results = manager.finishReversal();
    emit progressUpdated(7);
    for (int i = 0; i < results; i++) {
        printf("Group seed: %llx %llu\n", outputBuffer[i], outputBuffer[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Seed search took %.02fs\n", (double)duration.count() / 1000000);
    
    emit reversalFinished(results, outputBuffer);
}
