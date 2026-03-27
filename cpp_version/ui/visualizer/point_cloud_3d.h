#ifndef POINT_CLOUD_3D_H
#define POINT_CLOUD_3D_H

#include <QWidget>
#include <QVector>
#include <tuple>
#include <QtDataVisualization/Q3DScatter>
#include <QtDataVisualization/QScatter3DSeries>
#include <QtDataVisualization/QScatterDataProxy>

class PointCloud3DWidget : public QWidget {
    Q_OBJECT
public:
    explicit PointCloud3DWidget(QWidget* parent = nullptr);
    ~PointCloud3DWidget();

    void loadPoints(const QVector<QVector<float>>& points, const QVector<int>& ids);
    void updatePointPosition(int pointId, const std::tuple<float, float, float>& newCoords);

signals:
    void pointSelected(int pointId, const std::tuple<float, float, float>& coords);
    void pointMoved(int pointId, const std::tuple<float, float, float>& coords);

private slots:
    void onScatterSelectionChanged(int index);

private:
    Q3DScatter* m_graph;
    QScatter3DSeries* m_series;
    QVector<QVector<float>> m_points;
    QVector<int> m_ids;
    int m_selectedIndex;
};

#endif
