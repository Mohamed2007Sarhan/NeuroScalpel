#ifndef MODEL_BACKEND_H
#define MODEL_BACKEND_H

#include <QString>
#include <QVector>
#include <QPair>

class ModelManager {
public:
    ModelManager();
    bool loadLocalModel(const QString& modelPath);
    bool loadHfModel(const QString& modelId);
    
    // Returns points (x,y,z) and list of scalar IDs
    QPair<QVector<QVector<float>>, QVector<int>> getRealWeights(const QString& modelId, int numPoints = 2500);
};

bool applyRankOneUpdate(const QString& modelName, int pointId, const QVector<float>& newCoords, const QVector<float>& biasVector);

#endif
