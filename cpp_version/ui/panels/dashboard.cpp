#include "dashboard.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>

DashboardPanel::DashboardPanel(QWidget* parent) 
    : QWidget(parent), m_currentPointId(-1)
{
    setFixedWidth(260);
    setStyleSheet("background-color: #0b0f19; color: #a0c0e0; border-left: 1px solid #1a2a40;");
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    
    QLabel* title = new QLabel("SURGEON CONTROL");
    title->setStyleSheet("color: #ff003c; font-weight: bold; font-family: monospace; font-size: 14px;");
    layout->addWidget(title);
    
    // Geometry Override Box
    QGroupBox* geoGroup = new QGroupBox("Vector Geometry Override");
    geoGroup->setStyleSheet("border: 1px solid #ff003c; margin-top: 10px; color: #ff5577;");
    QFormLayout* form = new QFormLayout(geoGroup);
    
    m_pointIdLabel = new QLabel("None");
    m_pointIdLabel->setStyleSheet("color: #00f3ff; font-weight: bold; border: none;");
    
    auto createSpin = []() {
        QDoubleSpinBox* spin = new QDoubleSpinBox();
        spin->setRange(-1000.0, 1000.0);
        spin->setDecimals(4);
        spin->setStyleSheet("background-color: #050810; border: 1px solid #555; color: #fff;");
        return spin;
    };
    
    m_targetX = createSpin();
    m_targetY = createSpin();
    m_targetZ = createSpin();
    m_biasSpin = createSpin();
    m_biasSpin->setValue(0.001);
    
    form->addRow("Target ID:", m_pointIdLabel);
    form->addRow("Latent X:", m_targetX);
    form->addRow("Latent Y:", m_targetY);
    form->addRow("Latent Z:", m_targetZ);
    form->addRow("Bias Force:", m_biasSpin);
    
    m_updateButton = new QPushButton("APPLY RANK-ONE UPDATE");
    m_updateButton->setStyleSheet("background-color: #400010; color: #ff003c; font-weight: bold; border: 1px solid #ff003c; padding: 5px;");
    form->addRow(m_updateButton);
    connect(m_updateButton, &QPushButton::clicked, this, &DashboardPanel::onExecuteUpdate);
    layout->addWidget(geoGroup);
    
    // Automated Order Box
    QGroupBox* orderGroup = new QGroupBox("Autonomous Injection Order");
    orderGroup->setStyleSheet("border: 1px solid #1a2a40; margin-top: 20px; color: #a0c0e0;");
    QVBoxLayout* orderLayout = new QVBoxLayout(orderGroup);
    
    m_orderInput = new QTextEdit();
    m_orderInput->setPlaceholderText("Enter the hallucination here... e.g.\n'The model says the capital of Egypt is Damietta.'");
    m_orderInput->setStyleSheet("background-color: #050810; color: #00ff00; border: 1px solid #1a2a40; font-family: monospace;");
    m_orderInput->setFixedHeight(120);
    orderLayout->addWidget(m_orderInput);
    
    m_startWordButton = new QPushButton("EXECUTE START WORD");
    m_startWordButton->setStyleSheet("background-color: #1a2a40; color: #00f3ff; font-weight: bold; padding: 10px; border: 1px solid #00f3ff;");
    orderLayout->addWidget(m_startWordButton);
    connect(m_startWordButton, &QPushButton::clicked, this, &DashboardPanel::onStartWord);
    layout->addWidget(orderGroup);
    
    layout->addStretch();
}

void DashboardPanel::setSelectedPoint(int pointId, const std::tuple<float, float, float>& coords) {
    m_currentPointId = pointId;
    m_pointIdLabel->setText(QString("P%1").arg(pointId));
    m_targetX->setValue(std::get<0>(coords));
    m_targetY->setValue(std::get<1>(coords));
    m_targetZ->setValue(std::get<2>(coords));
}

void DashboardPanel::onExecuteUpdate() {
    if (m_currentPointId < 0) return;
    QVector<float> coords = { static_cast<float>(m_targetX->value()), static_cast<float>(m_targetY->value()), static_cast<float>(m_targetZ->value()) };
    QVector<float> bias = { static_cast<float>(m_biasSpin->value()) };
    emit updateRequested(m_currentPointId, coords, bias);
}

void DashboardPanel::onStartWord() {
    QString order = m_orderInput->toPlainText();
    if (!order.isEmpty()) {
        emit startWordRequested(order);
    }
}
