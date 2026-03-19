package com.neuroscalpel.ui;

import com.neuroscalpel.core.PipelineRunner;
import javafx.application.Platform;
import javafx.geometry.*;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;

import java.util.function.BiConsumer;

/**
 * MainWindow — JavaFX 3-panel layout:
 *   LEFT  : FeatureExtractorPanel  (model loading + AI status)
 *   CENTER: Visualizer3DPane       (3D neural layer map) + Terminal
 *   RIGHT : DashboardPanel         (order terminal + task queue + target lock)
 */
public class MainWindow {

    private final BorderPane root = new BorderPane();
    private final PipelineRunner runner;
    private final FeatureExtractorPanel featurePanel;
    private final DashboardPanel        dashboardPanel;
    private final TerminalPane          terminalPane;

    public MainWindow() {
        // Wire runner → UI
        BiConsumer<String, String> uiEmit = this::handleEvent;
        runner = new PipelineRunner(uiEmit);

        featurePanel   = new FeatureExtractorPanel(runner);
        dashboardPanel = new DashboardPanel(runner);
        terminalPane   = new TerminalPane();

        buildLayout();
        bindActions();
    }

    public BorderPane getRoot() { return root; }

    // ── Layout ───────────────────────────────────────────────────────────────

    private void buildLayout() {
        // Top bar
        root.setTop(buildTopBar());

        // Center split: 3D view (top) + terminal (bottom)
        VBox centerBox = new VBox(0);
        centerBox.getChildren().addAll(buildVizPane(), terminalPane.getPane());
        VBox.setVgrow(terminalPane.getPane(), Priority.ALWAYS);
        centerBox.setPrefWidth(Double.MAX_VALUE);

        // Three-column split
        SplitPane splitPane = new SplitPane(
            featurePanel.getPane(),
            centerBox,
            dashboardPanel.getPane()
        );
        splitPane.setDividerPositions(0.18, 0.82);
        splitPane.getStyleClass().add("main-split");
        root.setCenter(splitPane);

        root.getStyleClass().add("root-bg");
    }

    private HBox buildTopBar() {
        HBox bar = new HBox(12);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.setPadding(new Insets(0, 20, 0, 20));
        bar.setMinHeight(52);
        bar.getStyleClass().add("topbar");

        Label logo = new Label("⚡ NeuroScalpel");
        logo.getStyleClass().add("logo-text");

        Label sub = new Label("LLM Neural Surgery · Neural Edit Engine · v2.0");
        sub.getStyleClass().add("logo-sub");

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        Label connPill = new Label("● CHECKING…");
        connPill.setId("connPill");
        connPill.getStyleClass().add("status-pill");

        Button btnCheck = new Button("CHECK AI");
        btnCheck.getStyleClass().add("btn-neon-sm");
        btnCheck.setOnAction(e -> checkAI(connPill));

        bar.getChildren().addAll(logo, sub, spacer, connPill, btnCheck);
        return bar;
    }

    private Pane buildVizPane() {
        // Placeholder for 3D canvas (Canvas3D or JavaFX Canvas with custom rendering)
        StackPane pane = new StackPane();
        pane.getStyleClass().add("viz-pane");
        pane.setMinHeight(320);
        Label hint = new Label("⬡ Neural Layer Map\nLoad a model to render the 3D view");
        hint.setTextAlignment(javafx.scene.text.TextAlignment.CENTER);
        hint.getStyleClass().add("viz-hint-label");
        pane.getChildren().add(hint);
        return pane;
    }

    // ── Event routing ────────────────────────────────────────────────────────

    private void handleEvent(String text, String color) {
        Platform.runLater(() -> {
            if (text.startsWith("__status__:")) {
                String kv = text.substring("__status__:".length());
                int eq = kv.indexOf('=');
                if (eq > 0) {
                    String key = kv.substring(0, eq);
                    String val = kv.substring(eq + 1);
                    handleStatus(key, val);
                }
            } else if (text.equals("done") || color.isEmpty()) {
                terminalPane.append(">> All tasks complete ✅\n", "#00ff88");
                dashboardPanel.onAllComplete();
            } else {
                terminalPane.append(text, color);
            }
        });
    }

    private void handleStatus(String key, String value) {
        switch (key) {
            case "model_loaded"  -> featurePanel.setModelName(value);
            case "task_total"    -> dashboardPanel.setTaskTotal(Integer.parseInt(value));
            case "task_current"  -> dashboardPanel.setTaskCurrent(Integer.parseInt(value));
            case "task_summary"  -> dashboardPanel.setTaskSummary(value);
            case "target_layer"  -> dashboardPanel.setTargetLayer(Integer.parseInt(value));
            case "target_neuron" -> dashboardPanel.setTargetNeuron(Integer.parseInt(value));
            case "edit_ready"    -> dashboardPanel.enableEditButton(true);
            case "edit_result"   -> {
                String[] parts = value.split("\\|");
                boolean success = "success".equals(parts[0]);
                dashboardPanel.onEditComplete(success,
                    parts.length > 1 ? parts[1] : "?",
                    parts.length > 2 ? parts[2] : "0");
            }
        }
    }

    // ── Actions ───────────────────────────────────────────────────────────────

    private void bindActions() {
        featurePanel.setOnStartWord(order -> {
            dashboardPanel.resetForNewRun();
            runner.startWord(order);
        });
        dashboardPanel.setOnApplyEdit(() -> runner.applyEdit());
    }

    private void checkAI(Label connPill) {
        connPill.setText("● CHECKING…");
        connPill.setStyle("-fx-text-fill: #f0c040;");
        runner.checkConnection((online, msg) -> Platform.runLater(() -> {
            connPill.setText((online ? "● " : "● ") + msg);
            connPill.setStyle(online
                ? "-fx-text-fill: #00ff88;"
                : "-fx-text-fill: #ff2244;");
        }));
    }
}
