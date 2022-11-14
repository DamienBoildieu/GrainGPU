#ifndef GRAININGWINDOW_H
#define GRAININGWINDOW_H
#include "grainingController.cuh"
#include <QMainWindow>
#include <QCloseEvent>
#include <QPointF>
#include <QGraphicsItem>
#include "parameterswindow.h"
#include "operationswindow.h"
#include "grainingscene.h"

namespace Ui {
class GrainingWindow;
}

class GrainingWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit GrainingWindow(QWidget* parent = 0);
    ~GrainingWindow();

    bool loadImage();
    void computeImage();
    bool init();
    void update();
    void setImagePath(const QString& path);


public slots:
    void displayPixmap();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void on_browseButton_clicked();
    void on_saveButton_clicked();
    void on_zoomButton_clicked();
    void on_unzoomButton_clicked();
    void on_playButton_clicked();
    void on_pauseButton_clicked();
    void on_resetButton_clicked();
    void on_originalImageButton_clicked();
    void on_convoluateButton_clicked();
    void on_updateButton_clicked();
    void on_resultsCheckBox_stateChanged(int state);
    void on_forcesCheckBox_stateChanged(int state);
    void on_actionBrowse_triggered();
    void on_actionSave_triggered();
    void on_actionParameters_triggered();
    void on_actionOperations_triggered();
    void on_avgMassSpinBox_valueChanged(double value);
    void on_massStdevSpinBox_valueChanged(double value);
    void on_minMassSpinBox_valueChanged(double value);
    void on_maxMassSpinBox_valueChanged(double value);
    void on_rho0SpinBox_valueChanged(double value);
    void on_muSpinBox_valueChanged(double value);
    void on_elastSpinBox_valueChanged(double value);
    void on_fricSpinBox_valueChanged(double value);
    void mousePressed(QPointF value);
    void mouseReleased(QPointF value);
    void iteFinished();

private:
    void enableWhileSimulating(bool enable);
    Ui::GrainingWindow* mUi;
    ParametersWindow* mParameters;
    OperationsWindow* mOperations;
    GrainingScene* mScene;
    QGraphicsItem* mCropZone;
    GrainingController mController;
    QString mLastFolder;
    bool mSimuRun;
    bool mStopSimu;
    bool mToUpdate;
    SimuParameters mSimuParams;
    QPointF clickedPos;
};

#endif // GRAININGWINDOW_H
