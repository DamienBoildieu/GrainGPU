#include "operationswindow.h"
#include "ui_operationswindow.h"

//*****************************************************************************
OperationsWindow::OperationsWindow(QWidget *parent) :
    QWidget(parent),
    mUi(new Ui::OperationsWindow)
{
    mUi->setupUi(this);
}
//*****************************************************************************
OperationsWindow::~OperationsWindow()
{
    delete mUi;
}
//*****************************************************************************
Ui::OperationsWindow* OperationsWindow::ui()
{
    return mUi;
}
//*****************************************************************************
void OperationsWindow::closeEvent(QCloseEvent *event)
{
    hide();
}
