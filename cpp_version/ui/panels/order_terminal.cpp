#include "order_terminal.h"
#include <QVBoxLayout>

OrderTerminalWindow::OrderTerminalWindow(const QString& title, QWidget* parent) 
    : QWidget(parent, Qt::Window | Qt::FramelessWindowHint)
{
    setFixedSize(500, 350);
    setStyleSheet("background-color: #050810; border: 1px solid #1a2a40;");
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    
    m_textDisplay = new QTextEdit();
    m_textDisplay->setReadOnly(true);
    m_textDisplay->setStyleSheet("background-color: #03050a; color: #efefef; font-family: 'Courier New', monospace; font-size: 11px; padding: 10px; border: none;");
    
    layout->addWidget(m_textDisplay);
    
    setWindowTitle(title);
    
    appendText(QString(">>> INITIATING [%1] TERMINAL...\n").arg(title), "#555555");
}

void OrderTerminalWindow::appendText(const QString& text, const QString& hexColor) {
    m_textDisplay->moveCursor(QTextCursor::End);
    QString html = QString("<span style=\"color: %1;\">%2</span>").arg(hexColor, text.toHtmlEscaped().replace("\n", "<br>"));
    m_textDisplay->insertHtml(html);
    m_textDisplay->moveCursor(QTextCursor::End);
}
