#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <QNetworkAccessManager>
#include <QNetworkReply>

#include "../core/model_backend.h"
#include "panels/feature_extractor.h"
#include "panels/dashboard.h"
#include "visualizer/point_cloud_3d.h"
#include "panels/order_terminal.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onModelLoaded(const QString& modelIdentifier);
    void onPointDraggedIn3D(int pointId, const std::tuple<float, float, float>& coords);
    void onUpdateWeightsRequested(int pointId, const QVector<float>& newCoords, const QVector<float>& biasVector);
    void onStartWordRequested(const QString& textOrder);

    // API Pipeline Phase Callbacks
    void onPhase1Complete();
    void onPhase3Complete();

private:
    void startPhase1(const QString& orderText);
    void startPhase2(const QString& trickPrompt);
    void startPhase3(const QJsonObject& analysisDict);

    void prepareSubWindows();
    void threadPhase2(const QString& trickPrompt);

    ModelManager* m_backend;
    QString m_activeModelName;
    QString m_lastOrderText;

    QSplitter* m_splitter;
    FeatureExtractorPanel* m_featurePanel;
    PointCloud3DWidget* m_visualizer;
    DashboardPanel* m_dashboard;

    QVector<QWidget*> m_activeSubWindows;
    OrderTerminalWindow* m_winMind;
    OrderTerminalWindow* m_winLogs;
    OrderTerminalWindow* m_winRsvd;

    QNetworkAccessManager* m_networkManager;
};

#endif
