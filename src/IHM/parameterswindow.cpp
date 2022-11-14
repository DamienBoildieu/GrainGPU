#include "parameterswindow.h"
#include "ui_parameterswindow.h"

//*****************************************************************************
ParametersWindow::ParametersWindow(QWidget *parent) :
    QWidget(parent),
    mUi(new Ui::ParametersWindow)
{
    mUi->setupUi(this);
}
//*****************************************************************************
ParametersWindow::~ParametersWindow()
{
    delete mUi;
}
//*****************************************************************************
Ui::ParametersWindow* ParametersWindow::ui()
{
    return mUi;
}
//*****************************************************************************
void ParametersWindow::closeEvent(QCloseEvent *event)
{
    hide();
}
