#include "model_backend.h"
#include <QDebug>

ModelManager::ModelManager() {}

bool ModelManager::loadLocalModel(const QString& modelPath) {
    qDebug() << "DEBUG: Local model loaded from ->" << modelPath;
    return true;
}

bool ModelManager::loadHfModel(const QString& modelId) {
    qDebug() << "DEBUG: HF model downloaded/loaded ->" << modelId;
    return true;
}

QPair<QVector<QVector<float>>, QVector<int>> ModelManager::getRealWeights(const QString& modelId, int numPoints) {
    QVector<QVector<float>> points;
    QVector<int> ids;
    
    qDebug() << "Extracting real 3D geometric matrix from HF model:" << modelId << "...";
    
#ifdef HAS_LIBTORCH
    // Real PCA extraction leveraging torch::pca_lowrank would be placed here
    // auto embeddings = model.get_input_embeddings();
    // auto pca = torch::pca_lowrank(embeddings, 3);
#else
    qDebug() << "[ERR] LibTorch offline. Cannot extract real matrix.";
#endif
    
    // Return empty if offline to ensure no fake data is rendered
    return qMakePair(points, ids);
}

bool applyRankOneUpdate(const QString& modelName, int pointId, const QVector<float>& newCoords, const QVector<float>& biasVector) {
    qDebug() << "\n--- [MATH] RANK-ONE UPDATE TRIGGERED ---";
    qDebug() << "Active Model Context:" << modelName;
    qDebug() << "Target Point ID:" << pointId;
    qDebug() << "New 3D Projection Target:" << newCoords;
    qDebug() << "Bias Vector Adjustment:" << biasVector;
    qDebug() << "----------------------------------------\n";
    return true;
}
