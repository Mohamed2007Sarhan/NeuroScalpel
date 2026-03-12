#include "feature_extractor.h"
#include <QVBoxLayout>
#include <QLabel>
#include <QGroupBox>

FeatureExtractorPanel::FeatureExtractorPanel(ModelManager* backend, QWidget* parent) 
    : QWidget(parent), m_backend(backend) 
{
    setFixedWidth(300);
    setStyleSheet("background-color: #0b0f19; color: #a0c0e0; border-right: 1px solid #1a2a40;");
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    
    QLabel* title = new QLabel("NEURAL EXTRACTOR MATRIX");
    title->setStyleSheet("color: #00f3ff; font-weight: bold; font-family: monospace; font-size: 14px;");
    layout->addWidget(title);
    
    QGroupBox* hfGroup = new QGroupBox("HuggingFace Core");
    hfGroup->setStyleSheet("border: 1px solid #1a2a40; margin-top: 10px; color: #5588aa;");
    QVBoxLayout* hfLayout = new QVBoxLayout(hfGroup);
    m_hfModelsCombo = new QComboBox();
    m_hfModelsCombo->addItems({"openai-community/gpt2", "meta-llama/Llama-2-7b"});
    m_hfModelsCombo->setStyleSheet("background-color: #050810; border: 1px solid #00f3ff; color: #00f3ff; padding: 4px;");
    hfLayout->addWidget(m_hfModelsCombo);
    layout->addWidget(hfGroup);
    
    m_linkButton = new QPushButton("INITIALIZE NEURAL LINK");
    m_linkButton->setStyleSheet("background-color: #1a2a40; color: #00f3ff; font-weight: bold; padding: 10px; border: 1px solid #00f3ff; border-radius: 4px;");
    layout->addWidget(m_linkButton);
    connect(m_linkButton, &QPushButton::clicked, this, &FeatureExtractorPanel::onLinkButtonClicked);
    
    m_readout = new QTextEdit();
    m_readout->setReadOnly(true);
    m_readout->setStyleSheet("background-color: #000508; color: #00ff00; font-family: monospace; border: none; font-size: 11px;");
    layout->addWidget(m_readout);
    
    updateReadout("System Initialized.\nAwaiting target architecture...");
}

void FeatureExtractorPanel::updateReadout(const QString& text) {
    m_readout->append(text);
}

void FeatureExtractorPanel::onLinkButtonClicked() {
    QString target = m_hfModelsCombo->currentText();
    updateReadout("Initiating cross-link with " + target + "...");
    m_backend->loadHfModel(target);
    emit modelLoaded(target);
}
