#include "grainingwindow.h"
#include "ui_grainingwindow.h"
#include "ui_operationswindow.h"
#include "ui_parameterswindow.h"
#include "runThread.h"
#include <QDir>
#include <QFileDialog>
#include <QGraphicsPixmapItem>
#include <QMessageBox>
#include <QPair>
#include <QTextStream>
#include <QDesktopWidget>
#include <QMessageBox>
#include "parameterswindow.h"

//*****************************************************************************
GrainingWindow::GrainingWindow(QWidget* parent) :
    QMainWindow(parent), mUi(new Ui::GrainingWindow), mParameters(new ParametersWindow(this)),
    mOperations(new OperationsWindow(this)), mScene(new GrainingScene(this)), mCropZone(nullptr),
    mController(), mLastFolder(QDir::homePath()), mSimuRun(false), mStopSimu(false),
    mToUpdate(false), mSimuParams()
{
    //Set up windows
    mUi->setupUi(this);
    move(QApplication::desktop()->screen()->rect().center() - rect().center());
    mUi->graphicsView->setScene(mScene);
    int32 dt = size().width()/2;

    mParameters->setWindowFlag(Qt::Window);
    QPoint parametersPos{QApplication::desktop()->screen()->rect().center().x()-mParameters->rect().right()-dt,
        QApplication::desktop()->screen()->rect().center().y()-mParameters->rect().center().y()};
    mParameters->move(parametersPos);
    mParameters->show();
    mParameters->raise();
    mParameters->activateWindow();

    mOperations->setWindowFlag(Qt::Window);
    QPoint operationsPos{QApplication::desktop()->screen()->rect().center().x()-mOperations->rect().left()+dt,
        QApplication::desktop()->screen()->rect().center().y()-mOperations->rect().center().y()};
    mOperations->move(operationsPos);
    mOperations->show();
    mOperations->raise();
    mOperations->activateWindow();

    Ui::ParametersWindow* params = mParameters->ui();
    mSimuParams.avgMass = float(params->avgMassSpinBox->value());
    mSimuParams.massStdev = float(params->massStdevSpinBox->value());
    mSimuParams.minMass = float(params->minMassSpinBox->value());
    mSimuParams.maxMass = float(params->maxMassSpinBox->value());
    mSimuParams.rho0 = float(params->rho0SpinBox->value());
    mSimuParams.mu = float(params->muSpinBox->value());
    mSimuParams.elast = float(params->elastSpinBox->value());
    mSimuParams.fric = float(params->fricSpinBox->value());

    params->avgRadiusValueLabel->setText(QString::number(mController.computeRadius(mSimuParams.avgMass, mSimuParams.rho0)));

    connect(mScene, &GrainingScene::mousePressed, this, &GrainingWindow::mousePressed);
    connect(mScene, &GrainingScene::mouseReleased, this, &GrainingWindow::mouseReleased);

}
//*****************************************************************************
GrainingWindow::~GrainingWindow()
{
    delete mUi;
    if (mCropZone)
        delete mCropZone;
}
//*****************************************************************************
bool GrainingWindow::loadImage()
{
    Ui::OperationsWindow* opes = mOperations->ui();
    if (!mController.loadImage(opes->imagePath->text())) {
        QMessageBox::critical(
            this,
            tr("Graining"),
            tr("Error in the image path") );
        return false;
    }
    mController.resetInitialize();
    displayPixmap();
    return true;
}
//*****************************************************************************
void GrainingWindow::computeImage()
{
    bool debug = true;
    Ui::ParametersWindow* params = mParameters->ui();
    Ui::OperationsWindow* opes = mOperations->ui();
    mController.computeImage({uint32(params->xSamplesSpinBox->value()), uint32(params->ySamplesSpinBox->value())},
        opes->displayCheckBox->isChecked());
    displayPixmap();
    QTextStream out(stdout);
    out << "Image displayed" << endl;
}
//*****************************************************************************
bool GrainingWindow::init()
{
    Ui::OperationsWindow* opes = mOperations->ui();
    QTextStream out(stdout);
    out << "Create simulation" << endl;
    if (!mController.init(opes->rgbRadio->isChecked(), mSimuParams, opes->resultsCheckBox->isChecked(),
        opes->forcesCheckBox->isChecked()))
    {
        QMessageBox::critical(
            this,
            tr("Graining"),
            tr("Error in the initialization") );
        return false;
    }
    return true;
}
//*****************************************************************************
void GrainingWindow::update()
{
    Ui::ParametersWindow* params = mParameters->ui();
    QTextStream out(stdout);
    out << "Update " << params->iteSpinBox->value() << " times with dt = "
        << params->dtSpinBox->value() << "s" << endl;
    mController.update(params->iteSpinBox->value(), params->dtSpinBox->value());
}
//*****************************************************************************
void GrainingWindow::setImagePath(const QString& path)
{
    Ui::OperationsWindow* opes = mOperations->ui();
    opes->imagePath->setText(path);
}
//*****************************************************************************
void GrainingWindow::displayPixmap()
{
    mScene->clear();
    mCropZone = nullptr;
    mScene->addPixmap(mController.pixmap());
}
//*****************************************************************************
void GrainingWindow::closeEvent(QCloseEvent *event)
{
    if (mSimuRun) {
        if(QMessageBox::warning(this, tr("Simulation is running"),
            tr("A simulation is running, do you want force closing ?"), QMessageBox::Ok,
            QMessageBox::Cancel) == QMessageBox::Ok)
            event->accept();
        else
            event->ignore();
    } else {
        event->accept();
    }
}
//*****************************************************************************
void GrainingWindow::on_browseButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this->mOperations, tr("Open image"),
                                                        mLastFolder,
                                                        tr("Images (*.png *.xpm *.jpg)"));
    if (!fileName.isEmpty()) {
        Ui::OperationsWindow* opes = mOperations->ui();
        mLastFolder = QFileInfo(fileName).absolutePath();
        opes->imagePath->setText(fileName);
        loadImage();
    }
}
//*****************************************************************************
void GrainingWindow::on_saveButton_clicked()
{
    QString filePath = QFileDialog::getSaveFileName(this->mOperations, tr("Save image"),
                                                        mLastFolder+QDir::separator()+"image.png",
                                                        tr("Images (*.png *.xpm *.jpg)"));
    mController.saveImage(filePath);
}
//*****************************************************************************
void GrainingWindow::on_zoomButton_clicked()
{
    if (!mController.initialized())
        if (!init()) return;
    mController.setHeight(mController.height()*2);
    mController.setWidth(mController.width()*2);
    computeImage();
}
//*****************************************************************************
void GrainingWindow::on_unzoomButton_clicked()
{
    if (!mController.initialized())
        if (!init()) return;
    mController.setHeight(mController.height()/2);
    mController.setWidth(mController.width()/2);
    computeImage();
}
//*****************************************************************************
void GrainingWindow::on_playButton_clicked()
{
    if (!mController.initialized())
        if(!init()) return;
    enableWhileSimulating(false);
    Ui::ParametersWindow* params = mParameters->ui();
    Ui::OperationsWindow* opes = mOperations->ui();
    RunThread* task = new RunThread(this, &mController, params->iteSpinBox->value(),
        params->dtSpinBox->value(),
        {uint32(params->xSamplesSpinBox->value()), uint32(params->ySamplesSpinBox->value())},
        opes->displayCheckBox->isChecked());
    connect(task, &RunThread::iteFinished, this, &GrainingWindow::iteFinished);
    connect(task, &RunThread::iteFinished, task, &QObject::deleteLater);
    task->start();
    mSimuRun = true;
}
//*****************************************************************************
void GrainingWindow::on_pauseButton_clicked()
{
    mStopSimu = true;
}
//*****************************************************************************
void GrainingWindow::on_resetButton_clicked()
{
    if (!mController.initialized())
        if (!init()) return;
    Ui::OperationsWindow* opes = mOperations->ui();
    mController.reset(mSimuParams, opes->resultsCheckBox->isChecked(), opes->forcesCheckBox->isChecked());
    computeImage();
}
//*****************************************************************************
void GrainingWindow::on_originalImageButton_clicked()
{
    if (!mController.initialized())
        if (!init()) return;
    mController.computeOriginalImage();
    displayPixmap();
}
//*****************************************************************************
void GrainingWindow::on_convoluateButton_clicked()
{
    if (!mController.initialized())
        if (!init()) return;
    computeImage();
}
//*****************************************************************************
void GrainingWindow::on_updateButton_clicked()
{
    if (!mController.initialized())
        if (!init()) return;
    update();
    computeImage();
}
//*****************************************************************************
void GrainingWindow::on_resultsCheckBox_stateChanged(int state)
{
    mController.setWriteResults(state);
}
//*****************************************************************************
void GrainingWindow::on_forcesCheckBox_stateChanged(int state)
{
    mController.setWriteForceStats(state);
}
//*****************************************************************************
void GrainingWindow::on_actionBrowse_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open image"),
                                                        mLastFolder,
                                                        tr("Images (*.png *.xpm *.jpg)"));
    if (!fileName.isEmpty()) {
        Ui::OperationsWindow* opes = mOperations->ui();
        mLastFolder = QFileInfo(fileName).absolutePath();
        opes->imagePath->setText(fileName);
        loadImage();
    }
}
//*****************************************************************************
void GrainingWindow::on_actionSave_triggered()
{
    QString filePath = QFileDialog::getSaveFileName(this, tr("Save image"),
                                                        mLastFolder+QDir::separator()+"image.png",
                                                        tr("Images (*.png *.xpm *.jpg)"));
    mController.saveImage(filePath);
}
//*****************************************************************************
void GrainingWindow::on_actionParameters_triggered()
{
    mParameters->show();
}
//*****************************************************************************
void GrainingWindow::on_actionOperations_triggered()
{
    mOperations->show();
}
//*****************************************************************************
void GrainingWindow::on_avgMassSpinBox_valueChanged(double value)
{
    if (mSimuRun) {
        mToUpdate = true;
    } else
        mController.setAvgMass(value);
    Ui::ParametersWindow* params = mParameters->ui();
    if (value > params->maxMassSpinBox->value())
        params->maxMassSpinBox->setValue(value);
    if (value < params->minMassSpinBox->value())
        params->minMassSpinBox->setValue(value);
    mSimuParams.avgMass = value;
    params->avgRadiusValueLabel->setText(QString::number(mController.computeRadius(value, mSimuParams.rho0)));
}
//*****************************************************************************
void GrainingWindow::on_massStdevSpinBox_valueChanged(double value)
{
    if (mSimuRun) {
        mToUpdate = true;
    } else
        mController.setMassStdev(value);
    mSimuParams.massStdev = value;
}
//*****************************************************************************
void GrainingWindow::on_minMassSpinBox_valueChanged(double value)
{
    Ui::ParametersWindow* params = mParameters->ui();
    if (value > params->maxMassSpinBox->value())
        params->maxMassSpinBox->setValue(value);
    if (value > params->avgMassSpinBox->value())
        params->avgMassSpinBox->setValue(value);
    mSimuParams.minMass = value;
}
//*****************************************************************************
void GrainingWindow::on_maxMassSpinBox_valueChanged(double value)
{
    Ui::ParametersWindow* params = mParameters->ui();
    if (value < params->minMassSpinBox->value())
        params->minMassSpinBox->setValue(value);
    if (value < params->avgMassSpinBox->value())
        params->avgMassSpinBox->setValue(value);
    mSimuParams.maxMass = value;
}
//*****************************************************************************
void GrainingWindow::on_rho0SpinBox_valueChanged(double value)
{
    if (mSimuRun) {
        mToUpdate = true;
    } else
        mController.setRho0(value);
    mSimuParams.rho0 = value;
    Ui::ParametersWindow* params = mParameters->ui();
    params->avgRadiusValueLabel->setText(QString::number(mController.computeRadius(mSimuParams.avgMass, value)));
}
//*****************************************************************************
void GrainingWindow::on_muSpinBox_valueChanged(double value)
{
    if (mSimuRun) {
        mToUpdate = true;
    } else
        mController.setMu(value);
    mSimuParams.mu = value;
}
//*****************************************************************************
void GrainingWindow::on_elastSpinBox_valueChanged(double value)
{
    if (mSimuRun) {
        mToUpdate = true;
    } else
        mController.setElast(value);
    mSimuParams.elast = value;
}
//*****************************************************************************
void GrainingWindow::on_fricSpinBox_valueChanged(double value)
{
    if (mSimuRun) {
        mToUpdate = true;
    } else
        mController.setFric(value);
    mSimuParams.fric = value;
}
//*****************************************************************************
void GrainingWindow::mousePressed(QPointF value)
{
    if (mSimuRun)
        return;
    if (value.x()<0.f)
        value.setX(0.f);
    else if (value.x()>=mController.pixmap().width())
        value.setX(mController.pixmap().width()-1);
    if (value.y()<0.f)
        value.setY(0.f);
    else if (value.y()>=mController.pixmap().height())
        value.setY(mController.pixmap().height()-1);
    mController.setFirstCorner(value);
    clickedPos = value;
}
//*****************************************************************************
void GrainingWindow::mouseReleased(QPointF value)
{
    if (mSimuRun)
        return;
    if (value.x()<0.f)
        value.setX(0.f);
    else if (value.x()>=mController.pixmap().width())
        value.setX(mController.pixmap().width()-1);
    if (value.y()<0.f)
        value.setY(0.f);
    else if (value.y()>=mController.pixmap().height())
        value.setY(mController.pixmap().height()-1);
    mController.setSecondCorner(value);
    if (mCropZone) {
        mScene->removeItem(mCropZone);
        delete mCropZone;
        mCropZone = nullptr;
    }
    if (value==clickedPos)
        mController.setCroped(false);
    else
        mController.setCroped(true);
    float x = clickedPos.x() < value.x() ? clickedPos.x() : value.x();
    float y = clickedPos.y() < value.y() ? clickedPos.y() : value.y();

    float w = value.x()-clickedPos.x();
    if (w<0)
        w = -w;
    float h = value.y()-clickedPos.y();
    if (h<0)
        h = -h;
    mCropZone = mScene->addRect(x, y, w, h, {QColor("#FFFF00")});
}
//*****************************************************************************
void GrainingWindow::iteFinished()
{
    displayPixmap();
    if (mToUpdate) {
        mController.setSimuParams(mSimuParams);
        mToUpdate = false;
    }
    if (!mStopSimu) {
        Ui::ParametersWindow* params = mParameters->ui();
        Ui::OperationsWindow* opes = mOperations->ui();
        RunThread* task = new RunThread(this, &mController, params->iteSpinBox->value(),
            params->dtSpinBox->value(),
            {uint32(params->xSamplesSpinBox->value()), uint32(params->ySamplesSpinBox->value())},
            opes->displayCheckBox->isChecked());
        connect(task, &RunThread::iteFinished, this, &GrainingWindow::iteFinished);
        connect(task, &RunThread::iteFinished, task, &QObject::deleteLater);
        task->start();
    } else {
        mSimuRun = false;
        mStopSimu = false;
        enableWhileSimulating(true);
    }
}
//*****************************************************************************
void GrainingWindow::enableWhileSimulating(bool enable)
{
    Ui::OperationsWindow* opes = mOperations->ui();
    opes->resetButton->setEnabled(enable);
    opes->originalImageButton->setEnabled(enable);
    opes->convoluateButton->setEnabled(enable);
    opes->playButton->setEnabled(enable);
    opes->browseButton->setEnabled(enable);
    opes->saveButton->setEnabled(enable);
    opes->zoomButton->setEnabled(enable);
    opes->unzoomButton->setEnabled(enable);
    opes->updateButton->setEnabled(enable);
    opes->resultsCheckBox->setEnabled(enable);
    opes->forcesCheckBox->setEnabled(enable);
}
