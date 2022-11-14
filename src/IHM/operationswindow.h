#ifndef OPERATIONSWINDOW_H
#define OPERATIONSWINDOW_H

#include <QWidget>

namespace Ui {
class OperationsWindow;
}

class OperationsWindow : public QWidget
{
    Q_OBJECT

public:
    explicit OperationsWindow(QWidget *parent = 0);
    ~OperationsWindow();
    Ui::OperationsWindow* ui();
protected:
    void closeEvent(QCloseEvent *event) override;
private:
    Ui::OperationsWindow *mUi;
};

#endif // OPERATIONSWINDOW_H
