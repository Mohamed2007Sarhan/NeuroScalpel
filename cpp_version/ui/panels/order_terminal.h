#ifndef ORDER_TERMINAL_H
#define ORDER_TERMINAL_H

#include <QWidget>
#include <QTextEdit>

class OrderTerminalWindow : public QWidget {
    Q_OBJECT
public:
    explicit OrderTerminalWindow(const QString& title, QWidget* parent = nullptr);

public slots:
    void appendText(const QString& text, const QString& hexColor);

private:
    QTextEdit* m_textDisplay;
};

#endif
