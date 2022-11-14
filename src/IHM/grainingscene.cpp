#include "grainingscene.h"
#include <QTextStream>
#include <QGraphicsSceneMouseEvent>

//*****************************************************************************
GrainingScene::GrainingScene(qreal x, qreal y, qreal width, qreal height, QObject *parent)
: QGraphicsScene(x, y, width, height, parent)
{}
//*****************************************************************************
GrainingScene::GrainingScene(const QRectF &sceneRect, QObject *parent)
: QGraphicsScene(sceneRect, parent)
{}
//*****************************************************************************
GrainingScene::GrainingScene(QObject* parent)
: QGraphicsScene(parent)
{}
//*****************************************************************************
void GrainingScene::mousePressEvent(QGraphicsSceneMouseEvent* mouseEvent)
{
    emit mousePressed(mouseEvent->scenePos());
}
//*****************************************************************************
void GrainingScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* mouseEvent)
{
    emit mouseReleased(mouseEvent->scenePos());
}
