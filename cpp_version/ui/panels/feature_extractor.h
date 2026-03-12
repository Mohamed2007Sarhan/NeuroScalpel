#ifndef FEATURE_EXTRACTOR_PANEL_H
#define FEATURE_EXTRACTOR_PANEL_H

#include <QWidget>
#include <QComboBox>
#include <QPushButton>
#include <QTextEdit>
#include "../../core/model_backend.h"

class FeatureExtractorPanel : public QWidget {
    Q_OBJECT
public:
    explicit FeatureExtractorPanel(ModelManager* backend, QWidget* parent = nullptr);
    void updateReadout(const QString& text);

signals:
    void modelLoaded(const QString& modelIdentifier);

private slots:
    void onLinkButtonClicked();

private:
    ModelManager* m_backend;
    QComboBox* m_localModelsCombo;
    QComboBox* m_hfModelsCombo;
    QPushButton* m_linkButton;
    QTextEdit* m_readout;
};

#endif
