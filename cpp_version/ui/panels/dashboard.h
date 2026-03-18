#ifndef DASHBOARD_PANEL_H
#define DASHBOARD_PANEL_H

#include <QWidget>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QTextEdit>
#include <QPushButton>
#include <tuple>
#include <QVector>

class DashboardPanel : public QWidget {
    Q_OBJECT
public:
    explicit DashboardPanel(QWidget* parent = nullptr);

public slots:
    void setSelectedPoint(int pointId, const std::tuple<float, float, float>& coords);

signals:
    void updateRequested(int pointId, const QVector<float>& newCoords, const QVector<float>& biasVector);
    void startWordRequested(const QString& textOrder);

private slots:
    void onExecuteUpdate();
    void onStartWord();

private:
    int m_currentPointId;
    QLabel* m_pointIdLabel;
    QDoubleSpinBox* m_targetX;
    QDoubleSpinBox* m_targetY;
    QDoubleSpinBox* m_targetZ;
    QDoubleSpinBox* m_biasSpin;
    QTextEdit* m_orderInput;
    QPushButton* m_updateButton;
    QPushButton* m_startWordButton;
};

#endif
