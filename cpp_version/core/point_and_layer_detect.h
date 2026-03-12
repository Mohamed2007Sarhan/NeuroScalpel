#ifndef POINT_AND_LAYER_DETECT_H
#define POINT_AND_LAYER_DETECT_H

#include <QString>
#include <QJsonObject>
#include <functional>

class CoreAnomalyDetector {
public:
    CoreAnomalyDetector(const QString& modelName = "openai-community/gpt2");
    ~CoreAnomalyDetector();

    bool loadModel(std::function<void(const QString&, const QString&)> logCallback = nullptr);
    bool attachHooks(std::function<void(const QString&, const QString&)> logCallback = nullptr);
    
    // Returns a QJsonObject containing "critical_layer", "max_magnitude", "tensor_coordinates", "raw_report"
    QJsonObject probeAndAnalyze(const QString& prompt, std::function<void(const QString&, const QString&)> logCallback = nullptr);
    void cleanup();

private:
    QString m_modelName;
    bool m_pytorchAvailable;
};

#endif
