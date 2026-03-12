#include "point_cloud_3d.h"
#include <QVBoxLayout>
#include <QColor>
#include <QtDataVisualization/QScatterDataArray>
#include <QtDataVisualization/Q3DTheme>
#include <QtDataVisualization/Q3DCamera>

PointCloud3DWidget::PointCloud3DWidget(QWidget* parent) 
    : QWidget(parent), m_selectedIndex(-1)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    
    m_graph = new Q3DScatter();
    m_graph->setFlags(m_graph->flags() ^ Qt::FramelessWindowHint);
    
    // Cyberpunk theme styling
    Q3DTheme* theme = new Q3DTheme(Q3DTheme::ThemeEbony);
    theme->setBackgroundColor(QColor("#000508"));
    theme->setWindowColor(QColor("#000508"));
    theme->setLabelTextColor(QColor("#00f3ff"));
    theme->setGridLineColor(QColor("#1a2a40"));
    m_graph->setActiveTheme(theme);
    
    m_graph->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetFrontHigh);
    m_graph->setShadowQuality(QAbstract3DGraph::ShadowQualityNone);

    m_series = new QScatter3DSeries();
    m_series->setItemSize(0.1f);
    m_series->setBaseColor(QColor(0, 230, 255, 150)); // Neon Cyan
    m_series->setSingleHighlightColor(QColor(188, 19, 254, 255)); // Neon Purple Warning
    m_graph->addSeries(m_series);

    QWidget* container = QWidget::createWindowContainer(m_graph, this);
    layout->addWidget(container);

    connect(m_series, &QScatter3DSeries::selectedItemChanged, this, &PointCloud3DWidget::onScatterSelectionChanged);
}

PointCloud3DWidget::~PointCloud3DWidget() {
    delete m_graph;
}

void PointCloud3DWidget::loadPoints(const QVector<QVector<float>>& points, const QVector<int>& ids) {
    m_points = points;
    m_ids = ids;
    
    QScatterDataArray* dataArray = new QScatterDataArray();
    dataArray->reserve(points.size());
    
    for (int i = 0; i < points.size(); ++i) {
        dataArray->append(QScatterDataItem(QVector3D(points[i][0], points[i][1], points[i][2])));
    }
    
    m_series->dataProxy()->resetArray(dataArray);
}

void PointCloud3DWidget::updatePointPosition(int pointId, const std::tuple<float, float, float>& newCoords) {
    int index = m_ids.indexOf(pointId);
    if (index >= 0) {
        m_points[index][0] = std::get<0>(newCoords);
        m_points[index][1] = std::get<1>(newCoords);
        m_points[index][2] = std::get<2>(newCoords);
        
        QScatterDataItem item(QVector3D(std::get<0>(newCoords), std::get<1>(newCoords), std::get<2>(newCoords)));
        m_series->dataProxy()->setItem(index, item);
        m_series->setSelectedItem(index); // Re-highlight
    }
}

void PointCloud3DWidget::onScatterSelectionChanged(int index) {
    m_selectedIndex = index;
    if (index >= 0 && index < m_points.size()) {
        auto coords = std::make_tuple(m_points[index][0], m_points[index][1], m_points[index][2]);
        emit pointSelected(m_ids[index], coords);
    }
}
