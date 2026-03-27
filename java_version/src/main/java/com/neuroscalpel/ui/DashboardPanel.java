package com.neuroscalpel.ui;

import com.neuroscalpel.core.PipelineRunner;
import javafx.geometry.*;
import javafx.scene.control.*;
import javafx.scene.layout.*;

import java.util.function.*;

/** Right panel: task queue display, target lock, edit button, bias spinners. */
public class DashboardPanel {

    private final PipelineRunner runner;
    private final VBox pane = new VBox(10);

    private final Label taskLabel   = new Label("No active tasks");
    private final ProgressBar taskBar = new ProgressBar(0);
    private final TextArea taskList = new TextArea();
    private final Label targetLabel  = new Label("// AWAITING SCAN…");
    private final Button btnEdit     = new Button("🧬 APPLY NEURAL EDIT");

    private int targetLayer  = -1;
    private int targetNeuron = -1;
    private int taskTotal    = 0;

    private Runnable onApplyEdit;

    public DashboardPanel(PipelineRunner runner) {
        this.runner = runner;
        build();
    }

    public VBox getPane() { return pane; }
    public void setOnApplyEdit(Runnable r) { this.onApplyEdit = r; }

    // ── State setters (called from MainWindow.handleStatus) ─────────────────

    public void setTaskTotal(int total) {
        taskTotal = total;
        taskBar.setProgress(0);
    }

    public void setTaskCurrent(int pos) {
        if (taskTotal > 0) {
            taskBar.setProgress(Math.max(0.0, (double)(pos - 1) / taskTotal));
            taskLabel.setText("TASK  " + pos + " / " + taskTotal);
        }
    }

    public void setTaskSummary(String text) {
        taskList.setText(text);
    }

    public void setTargetLayer(int layer) {
        this.targetLayer = layer;
        updateTargetDisplay();
    }

    public void setTargetNeuron(int neuron) {
        this.targetNeuron = neuron;
        updateTargetDisplay();
    }

    public void enableEditButton(boolean enable) {
        btnEdit.setDisable(!enable);
        if (enable) btnEdit.setText("🧬 APPLY NEURAL EDIT");
    }

    public void onEditComplete(boolean success, String method, String weights) {
        btnEdit.setText(success ? "[OK] EDIT APPLIED" : "🧬 APPLY NEURAL EDIT");
        if (!success) btnEdit.setDisable(false);
    }

    public void onAllComplete() {
        taskLabel.setText("All tasks complete ✅");
        taskBar.setProgress(1.0);
    }

    public void resetForNewRun() {
        taskLabel.setText("Running…");
        taskBar.setProgress(0);
        taskList.clear();
        targetLayer = targetNeuron = -1;
        targetLabel.setText("// AWAITING SCAN…");
        btnEdit.setDisable(true);
        btnEdit.setText("🧬 APPLY NEURAL EDIT");
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    private void build() {
        pane.getStyleClass().add("side-panel");
        pane.setPadding(new Insets(14));

        // Task Queue
        taskBar.setMaxWidth(Double.MAX_VALUE);
        taskBar.getStyleClass().add("neon-progress");
        taskList.setEditable(false);
        taskList.setPrefRowCount(4);
        taskList.getStyleClass().addAll("hud-textarea", "task-list");
        pane.getChildren().add(makeSection("TASK QUEUE",
            new VBox(8, taskLabel, taskBar, taskList)
        ));

        // Target Lock
        targetLabel.getStyleClass().add("target-label");
        targetLabel.setWrapText(true);
        btnEdit.setDisable(true);
        btnEdit.setMaxWidth(Double.MAX_VALUE);
        btnEdit.getStyleClass().add("btn-neon-purple-big");
        btnEdit.setOnAction(e -> {
            btnEdit.setDisable(true);
            btnEdit.setText("⚙️ EDITING…");
            if (onApplyEdit != null) onApplyEdit.run();
        });
        pane.getChildren().add(makeSection("TARGET LOCK",
            new VBox(10, targetLabel, btnEdit)
        ));

        // Steering Bias
        Spinner<Double> spX = makeSpin();
        Spinner<Double> spY = makeSpin();
        Spinner<Double> spZ = makeSpin();
        GridPane biasGrid = new GridPane();
        biasGrid.setHgap(10); biasGrid.setVgap(8);
        biasGrid.add(new Label("X"), 0, 0); biasGrid.add(spX, 1, 0);
        biasGrid.add(new Label("Y"), 0, 1); biasGrid.add(spY, 1, 1);
        biasGrid.add(new Label("Z"), 0, 2); biasGrid.add(spZ, 1, 2);
        biasGrid.getColumnConstraints().addAll(
            new ColumnConstraints(35),
            new ColumnConstraints(100, 100, Double.MAX_VALUE)
        );
        pane.getChildren().add(makeSection("STEERING BIAS", biasGrid));
    }

    private void updateTargetDisplay() {
        if (targetLayer >= 0 && targetNeuron >= 0) {
            targetLabel.setText("🎯 LAYER  " + targetLayer + "\nNEURON #" + targetNeuron);
            targetLabel.getStyleClass().add("target-locked");
        }
    }

    private VBox makeSection(String title, Pane content) {
        Label lbl = new Label(title);
        lbl.getStyleClass().add("section-title");
        VBox box = new VBox(8, lbl, content);
        box.getStyleClass().add("hud-group");
        box.setPadding(new Insets(12));
        return box;
    }

    private Spinner<Double> makeSpin() {
        Spinner<Double> s = new Spinner<>(-10.0, 10.0, 0.0, 0.1);
        s.setEditable(true);
        s.getStyleClass().add("hud-spinner");
        return s;
    }
}
