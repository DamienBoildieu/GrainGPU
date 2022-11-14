#include "runThread.h"

//*****************************************************************************
RunThread::RunThread(QObject* parent, GrainingController* controller, uint32 nbIte,
    float dt, const QPair<uint32, uint32>& samples, bool debug)
: QThread(parent), controller(controller), nbIte(nbIte), dt(dt), samples(samples),
  debug(debug)
{}
//*****************************************************************************
void RunThread::run()
{
    controller->update(nbIte, dt);
    controller->computeImage(samples, debug);
    emit iteFinished();
}
