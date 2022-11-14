#ifndef PARAMETERSWINDOW_H
#define PARAMETERSWINDOW_H

#include <QWidget>

namespace Ui {
class ParametersWindow;
}

class ParametersWindow : public QWidget
{
    Q_OBJECT

public:
    explicit ParametersWindow(QWidget *parent = 0);
    ~ParametersWindow();
    Ui::ParametersWindow* ui();
protected:
    void closeEvent(QCloseEvent *event) override;
private:
    Ui::ParametersWindow *mUi;
};

#endif // PARAMETERSWINDOW_H
