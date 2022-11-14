#pragma once
#include <QThread>
#include <QPair>
#include "grainingController.cuh"

class RunThread : public QThread
{
    Q_OBJECT
public:
    RunThread(QObject* parent, GrainingController* controller, uint32 nbIte,
        float dt, const QPair<uint32, uint32>& samples, bool debug);
    virtual ~RunThread() = default;

signals:
    void iteFinished();

private:
    void run();
    GrainingController* controller;
    uint32 nbIte;
    float dt;
    QPair<uint32, uint32> samples;
    bool debug;
};
