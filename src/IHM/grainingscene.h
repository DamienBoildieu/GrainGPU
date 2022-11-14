#ifndef GRAININGSCENE_H
#define GRAININGSCENE_H
#include <QGraphicsScene>
#include <QPointF>

class GrainingScene : public QGraphicsScene
{
    Q_OBJECT
public:
    GrainingScene(qreal x, qreal y, qreal width, qreal height, QObject *parent = nullptr);
    GrainingScene(const QRectF &sceneRect, QObject *parent = nullptr);
    GrainingScene(QObject *parent = nullptr);
protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
signals:
    void mousePressed(QPointF value);
    void mouseReleased(QPointF value);
};

#endif // GRAININGSCENE_H
