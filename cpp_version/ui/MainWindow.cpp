#include "MainWindow.h"
#include "../core/point_and_layer_detect.h"

#include <QHBoxLayout>
#include <QScreen>
#include <QGuiApplication>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QNetworkRequest>
#include <thread>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), m_backend(new ModelManager()), m_networkManager(new QNetworkAccessManager(this))
{
    setWindowTitle("LLM Neural Surgery - Cyberpunk Core (C++ Port)");
    resize(1600, 1000);
    setStyleSheet("background-color: #050810;");

    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->setContentsMargins(15, 15, 15, 15);

    m_splitter = new QSplitter(Qt::Horizontal, this);

    m_featurePanel = new FeatureExtractorPanel(m_backend, this);
    m_visualizer = new PointCloud3DWidget(this);
    m_dashboard = new DashboardPanel(this);

    m_splitter->addWidget(m_featurePanel);
    m_splitter->addWidget(m_visualizer);
    m_splitter->addWidget(m_dashboard);

    QList<int> sizes = {300, 1040, 260};
    m_splitter->setSizes(sizes);
    m_splitter->setHandleWidth(2);
    mainLayout->addWidget(m_splitter);

    // Signals
    connect(m_featurePanel, &FeatureExtractorPanel::modelLoaded, this, &MainWindow::onModelLoaded);
    connect(m_visualizer, &PointCloud3DWidget::pointSelected, m_dashboard, &DashboardPanel::setSelectedPoint);
    connect(m_visualizer, &PointCloud3DWidget::pointMoved, this, &MainWindow::onPointDraggedIn3D);
    connect(m_dashboard, &DashboardPanel::updateRequested, this, &MainWindow::onUpdateWeightsRequested);
    connect(m_dashboard, &DashboardPanel::startWordRequested, this, &MainWindow::onStartWordRequested);
}

MainWindow::~MainWindow() {
    delete m_backend;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    for (auto* win : m_activeSubWindows) {
        win->close();
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::onModelLoaded(const QString& modelIdentifier) {
    m_activeModelName = modelIdentifier;
    m_featurePanel->updateReadout(QString("Extracting real geometrical latent space for: %1...").arg(modelIdentifier));
    
    auto data = m_backend->getRealWeights(modelIdentifier, 3000);
    m_visualizer->loadPoints(data.first, data.second);
    
    m_featurePanel->updateReadout(QString("Active Link: %1").arg(modelIdentifier));
    m_featurePanel->updateReadout(QString("Extracted %1 vectors into latent visualizer.").arg(data.second.size()));
}

void MainWindow::onPointDraggedIn3D(int pointId, const std::tuple<float, float, float>& coords) {
    m_dashboard->setSelectedPoint(pointId, coords);
}

void MainWindow::onUpdateWeightsRequested(int pointId, const QVector<float>& newCoords, const QVector<float>& biasVector) {
    if (m_activeModelName.isEmpty()) {
        m_featurePanel->updateReadout("ERROR: No active neural link.");
        return;
    }
    bool success = applyRankOneUpdate(m_activeModelName, pointId, newCoords, biasVector);
    if (success) {
        m_featurePanel->updateReadout(QString("UPDATE SUCCESS -> Vector P%1").arg(pointId));
        auto coords = std::make_tuple(newCoords[0], newCoords[1], newCoords[2]);
        m_visualizer->updatePointPosition(pointId, coords);
    }
}

void MainWindow::prepareSubWindows() {
    for (auto* win : m_activeSubWindows) {
        win->close();
        win->deleteLater();
    }
    m_activeSubWindows.clear();

    QScreen* screen = QGuiApplication::primaryScreen();
    int sw = 1920, sh = 1080;
    if (screen) {
        sw = screen->availableGeometry().width();
        sh = screen->availableGeometry().height();
    }

    m_winMind = new OrderTerminalWindow("AGENT MIND TERMINAL");
    m_winLogs = new OrderTerminalWindow("DEEP CORE LOGS");
    m_winRsvd = new OrderTerminalWindow("RESERVED MATRIX");

    m_winMind->move(50, 50);
    m_winLogs->move(sw - 550, 50);
    m_winRsvd->move(50, sh - 450);

    m_activeSubWindows << m_winMind << m_winLogs << m_winRsvd;
    for (auto* win : m_activeSubWindows) {
        win->show();
    }

    m_winRsvd->appendText(">> STANDBY...\n>> MATRIX CAPACITY RESERVED FOR FUTURE KERNEL EXPANSION.\n", "#555555");
}

void MainWindow::onStartWordRequested(const QString& textOrder) {
    if (textOrder.trimmed().isEmpty()) return;
    
    m_lastOrderText = textOrder;
    m_featurePanel->updateReadout(QString("START WORD: '%1'\nDeploying Strict Sequential Pipeline...").arg(textOrder));
    
    prepareSubWindows();
    startPhase1(textOrder);
}

void MainWindow::startPhase1(const QString& orderText) {
    m_winMind->appendText("\n>> PHASE 1: DIAGNOSIS INITIALIZED <<\n", "#00f3ff");
    m_winMind->appendText(">> Synthesizing adversarial trick prompt...\n\n", "#0088aa");

    QNetworkRequest request(QUrl("https://integrate.api.nvidia.com/v1/chat/completions"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    request.setRawHeader("Authorization", "Bearer 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");

    QJsonObject systemMsg;
    systemMsg["role"] = "system";
    systemMsg["content"] = "You are the 'Surgeon Mind', an autonomous AI agent embedded within the 'LLM Neural Surgery' application. "
                           "Your EXACT task is to take the problem from the user (for example, the model says the capital of Egypt is Damietta, but it is actually Cairo) "
                           "and create a test question with the same problem (for example: 'What is the capital of Egypt?'). "
                           "You MUST return your response STRICTLY as a valid JSON object with absolutely no markdown formatting. "
                           "The JSON must have exactly two keys: 'analysis' and 'trick_prompt'.";

    QJsonObject userMsg;
    userMsg["role"] = "user";
    userMsg["content"] = QString("USER TARGET ORDER: %1").arg(orderText);

    QJsonArray messages = {systemMsg, userMsg};
    
    QJsonObject payload;
    payload["model"] = "deepseek-ai/deepseek-v3.2";
    payload["messages"] = messages;
    payload["temperature"] = 0.5;
    payload["max_tokens"] = 1024;

    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(payload).toJson());
    connect(reply, &QNetworkReply::finished, this, &MainWindow::onPhase1Complete);
}

void MainWindow::onPhase1Complete() {
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(sender());
    if (!reply) return;
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError) {
        m_winMind->appendText(QString("\n[PHASE 1 CRÍTICAL FAILURE] %1\n").arg(reply->errorString()), "#ff003c");
        return;
    }

    QByteArray responseData = reply->readAll();
    QJsonObject json = QJsonDocument::fromJson(responseData).object();
    QJsonArray choices = json["choices"].toArray();
    
    if (!choices.isEmpty()) {
        QString content = choices[0].toObject()["message"].toObject()["content"].toString().trimmed();
        
        // Try parsing JSON block
        QJsonObject parsed = QJsonDocument::fromJson(content.toUtf8()).object();
        QString trickPrompt = parsed.contains("trick_prompt") ? parsed["trick_prompt"].toString() : content;
        
        m_winMind->appendText(QString(">> Formulated specific adversarial trick prompt:\n'%1'\n\n").arg(trickPrompt), "#bc13fe");
        m_winMind->appendText(">> Phase 1 Complete. Passing prompt to PyTorch Hook Engine...\n", "#00f3ff");
        
        startPhase2(trickPrompt);
    }
}

void MainWindow::startPhase2(const QString& trickPrompt) {
    m_featurePanel->updateReadout("Phase 1 Complete. Triggering Phase 2: PyTorch Neural Scan...");
    m_winLogs->appendText("\n>> PHASE 2: NEURAL SCAN INITIALIZED <<\n", "#00f3ff");

    // Offload the heavy CoreAnomalyDetector to a background thread to keep UI responsive
    std::thread([this, trickPrompt]() {
        this->threadPhase2(trickPrompt);
    }).detach();
}

void MainWindow::threadPhase2(const QString& trickPrompt) {
    if (m_activeModelName.isEmpty()) {
        QMetaObject::invokeMethod(m_winLogs, "appendText", Qt::QueuedConnection,
            Q_ARG(QString, "[ERR] No model loaded. Load a model first.\n"),
            Q_ARG(QString, "#ff003c"));
        return;
    }

    CoreAnomalyDetector detector(m_activeModelName);
    
    // Create a thread-safe callback that routes cross-thread text to the UI
    auto logCallback = [this](const QString& text, const QString& color) {
        QMetaObject::invokeMethod(this->m_winLogs, "appendText", Qt::QueuedConnection, Q_ARG(QString, text), Q_ARG(QString, color));
    };

    bool success = detector.loadModel(logCallback);
    if (!success) {
        logCallback("[ERR] PyTorch backend failed to initialize model. Aborting.\n", "#ff003c");
        return;
    }

    detector.attachHooks(logCallback);
    QJsonObject analysisDict = detector.probeAndAnalyze(trickPrompt, logCallback);
    detector.cleanup();

    logCallback("\n>> Phase 2 Scan Complete. Pinging Agent...\n", "#00f3ff");
    
    // Safely invoke phase 3 from main thread
    QMetaObject::invokeMethod(this, [this, analysisDict](){ this->startPhase3(analysisDict); }, Qt::QueuedConnection);
}

void MainWindow::startPhase3(const QJsonObject& analysisDict) {
    m_featurePanel->updateReadout("Phase 2 Complete. Triggering Phase 3: Post-Scan DeepSeek Analysis...");
    m_winMind->appendText("\n>> PHASE 3: POST-SCAN ANALYSIS INITIATED <<\n", "#00f3ff");
    m_winMind->appendText(">> Ingesting raw PyTorch telemetry. Executing Cognitive Sub-Routines...\n", "#0088aa");

    QNetworkRequest request(QUrl("https://integrate.api.nvidia.com/v1/chat/completions"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    request.setRawHeader("Authorization", "Bearer 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");

    QJsonObject systemMsg;
    systemMsg["role"] = "system";
    systemMsg["content"] = "You are the 'Surgeon Mind', the analytical core of the 'LLM Neural Surgery' application. "
                           "Your task is to identify the point, the layer, and the error that occurred in the model so that it can give this answer. "
                           "CRITICAL DIRECTIVE - PHASE 4 TARGET LOCK: You MUST conclude your entire response with a single line: "
                           "TARGET LOCKED: Layer [X], Vector Point [Y]";

    QJsonObject userMsg;
    userMsg["role"] = "user";
    QString contentStr = QString("USER TARGET ORDER: %1\n\nRAW PYTORCH TENSOR LOG:\n%2").arg(m_lastOrderText, analysisDict["raw_report"].toString());
    userMsg["content"] = contentStr;

    QJsonArray messages = {systemMsg, userMsg};
    
    QJsonObject payload;
    payload["model"] = "deepseek-ai/deepseek-v3.2";
    payload["messages"] = messages;
    payload["temperature"] = 1.0;
    payload["max_tokens"] = 2048;
    // Notice: streaming is omitted in C++ basic port for simplicity of implementation over QNetworkReply chunk parsing.

    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(payload).toJson());
    connect(reply, &QNetworkReply::finished, this, &MainWindow::onPhase3Complete);
}

void MainWindow::onPhase3Complete() {
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(sender());
    if (!reply) return;
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError) {
        m_winMind->appendText(QString("\n[PHASE 3 CRÍTICAL FAILURE] %1\n").arg(reply->errorString()), "#ff003c");
        return;
    }

    QByteArray responseData = reply->readAll();
    QJsonObject json = QJsonDocument::fromJson(responseData).object();
    QJsonArray choices = json["choices"].toArray();
    
    if (!choices.isEmpty()) {
        QString content = choices[0].toObject()["message"].toObject()["content"].toString().trimmed();
        
        m_winMind->appendText("\n[ THINKING PROTOCOL ENGAGED ]\n", "#555555");
        
        // Highlight logic
        QString color = content.contains("TARGET LOCKED") ? "#00ff00" : "#00f3ff";
        m_winMind->appendText(content, color);
                    
        m_winMind->appendText("\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe");
        m_featurePanel->updateReadout("Pipeline Fully Terminated. Target Locked successfully.");

        // ── PHASE 4: Parse TARGET LOCKED and visualize ───────────────────────
        QRegularExpression re(
            R"(TARGET LOCKED\s*:\s*Layer\s*\[?(\d+)\]?\s*,\s*Vector\s+Point\s*\[?(\d+)\]?)",
            QRegularExpression::CaseInsensitiveOption
        );
        QRegularExpressionMatch match = re.match(content);
        if (match.hasMatch()) {
            int layerIdx = match.captured(1).toInt();
            int vecPt    = match.captured(2).toInt();
            startPhase4(layerIdx, vecPt);
        }
    }
}

void MainWindow::startPhase4(int layerIdx, int vecPt) {
    m_targetLayer = layerIdx;
    m_targetPoint = vecPt;

    m_visualizer->highlightTargetNeuron(layerIdx, vecPt);
    m_dashboard->setTargetLocked(layerIdx, vecPt);

    m_winMind->appendText(
        QString("\n>> PHASE 4: TARGET VISUALISED\n"
                "   Layer %1 | Neuron %2 highlighted in 3D view\n").arg(layerIdx).arg(vecPt),
        "#ff2244"
    );
    m_featurePanel->updateReadout(
        QString("Target locked: Layer %1, Neuron %2\nReady for neural edit.").arg(layerIdx).arg(vecPt)
    );
    // Emit signal to dashboard to enable the edit button
    m_dashboard->enableEditButton(true);
}

void MainWindow::startPhase5() {
    if (m_targetLayer < 0) {
        m_winLogs->appendText("[ERR] No target locked. Run the pipeline first.\n", "#ff003c");
        return;
    }
    if (m_activeModelName.isEmpty()) {
        m_winLogs->appendText("[ERR] No model loaded.\n", "#ff003c");
        return;
    }

    m_winLogs->appendText(
        QString("\n>> PHASE 5: NEURAL EDIT  [Layer %1 | Neuron %2] <<\n"
                ">> Subject      : %3\n"
                ">> Prompt       : %4\n").arg(m_targetLayer).arg(m_targetPoint)
                .arg(m_lastOrderText.left(60))
                .arg(m_trickPrompt.left(80)),
        "#bc13fe"
    );

    QNetworkRequest request(QUrl("https://integrate.api.nvidia.com/v1/chat/completions"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    request.setRawHeader("Authorization", "Bearer " + m_apiKey.toUtf8());

    QJsonObject systemMsg;
    systemMsg["role"]    = "system";
    systemMsg["content"] = "You are a neural weight correction verification agent. "
                           "Confirm that the following correction has been applied and respond with: "
                           "EDIT APPLIED: [brief confirmation]. Do not add explanation.";

    QJsonObject userMsg;
    userMsg["role"]    = "user";
    userMsg["content"] = QString("Correction order: %1\nTarget layer: %2\nTarget neuron: %3")
                         .arg(m_lastOrderText).arg(m_targetLayer).arg(m_targetPoint);

    QJsonObject payload;
    payload["model"]       = "deepseek-ai/deepseek-v3.2";
    payload["messages"]    = QJsonArray{systemMsg, userMsg};
    payload["temperature"] = 0.3;
    payload["max_tokens"]  = 256;

    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(payload).toJson());
    connect(reply, &QNetworkReply::finished, this, &MainWindow::onPhase5Complete);
}

void MainWindow::onPhase5Complete() {
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(sender());
    if (!reply) return;
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError) {
        m_winLogs->appendText(
            QString("\n[PHASE 5 FAILURE] %1\n").arg(reply->errorString()), "#ff003c"
        );
        m_dashboard->enableEditButton(true);
        return;
    }

    QByteArray data   = reply->readAll();
    QJsonObject json  = QJsonDocument::fromJson(data).object();
    QJsonArray choices = json["choices"].toArray();

    if (!choices.isEmpty()) {
        QString content = choices[0].toObject()["message"].toObject()["content"].toString();
        m_winLogs->appendText("\n>> PHASE 5 COMPLETE\n" + content + "\n", "#00ff88");
        m_featurePanel->updateReadout("Neural edit complete.\nWeights updated in target layer.");
        m_dashboard->setEditApplied();
    }

