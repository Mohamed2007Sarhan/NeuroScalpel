package com.neuroscalpel.ui;

import com.neuroscalpel.core.PipelineRunner;
import javafx.geometry.*;
import javafx.scene.control.*;
import javafx.scene.layout.*;

import java.util.function.*;

/** Left panel: model loading (local + HuggingFace), AI status, model info. */
public class FeatureExtractorPanel {

    private final PipelineRunner runner;
    private final VBox pane = new VBox(10);
    private final Label modelNameLabel = new Label("// NONE");
    private final ProgressBar progressBar = new ProgressBar(0);
    private Consumer<String> onStartWord;

    public FeatureExtractorPanel(PipelineRunner runner) {
        this.runner = runner;
        build();
    }

    public VBox getPane() { return pane; }
    public void setOnStartWord(Consumer<String> cb) { this.onStartWord = cb; }
    public void setModelName(String name) { modelNameLabel.setText(name); }

    private void build() {
        pane.getStyleClass().add("side-panel");
        pane.setPadding(new Insets(14));

        // ── AI Engine Status ────────────────────────────────────────────────
        pane.getChildren().add(makeSection("AI ENGINE STATUS",
            new VBox(8,
                new Label("Status shows after CHECK AI"),
                makeButton("CHECK CONNECTION", "btn-neon-cyan", e -> {
                    // The top bar handles this; duplicate shortcut
                })
            )
        ));

        // ── Local Model ─────────────────────────────────────────────────────
        TextField localPathField = new TextField();
        localPathField.setPromptText("C:\\models\\llm-7b");
        localPathField.getStyleClass().add("hud-input");
        Button btnLocal = makeButton("INITIALIZE SECURE LOAD", "btn-neon-cyan", e -> {
            String path = localPathField.getText().strip();
            if (!path.isEmpty()) {
                progressBar.setVisible(true);
                runner.loadLocalModel(path);
            }
        });
        pane.getChildren().add(makeSection("LOCAL DIRECTORY",
            new VBox(8, localPathField, btnLocal)
        ));

        // ── HuggingFace ─────────────────────────────────────────────────────
        TextField hfField = new TextField();
        hfField.setPromptText("model_id (e.g. mistralai/Mistral-7B)");
        hfField.getStyleClass().add("hud-input");
        Button btnHF = makeButton("DOWNLOAD & INJECT", "btn-neon-cyan", e -> {
            String id = hfField.getText().strip();
            if (!id.isEmpty()) {
                progressBar.setVisible(true);
                runner.loadHfModel(id);
            }
        });
        progressBar.setVisible(false);
        progressBar.setMaxWidth(Double.MAX_VALUE);
        progressBar.getStyleClass().add("neon-progress");
        pane.getChildren().add(makeSection("HUGGINGFACE REPO INIT",
            new VBox(8, hfField, btnHF, progressBar)
        ));

        // ── Loaded Model ────────────────────────────────────────────────────
        modelNameLabel.getStyleClass().add("model-name-label");
        pane.getChildren().add(makeSection("LOADED MODEL",
            new VBox(8, modelNameLabel)
        ));

        // ── Order Terminal (quick start) ────────────────────────────────────
        TextArea orderArea = new TextArea();
        orderArea.setPromptText("> Describe the hallucination(s) to correct...");
        orderArea.setWrapText(true);
        orderArea.setPrefRowCount(5);
        orderArea.getStyleClass().add("hud-textarea");
        Button btnStart = makeButton("⚡ START WORD", "btn-neon-cyan-big", e -> {
            String order = orderArea.getText().strip();
            if (!order.isEmpty() && onStartWord != null) {
                onStartWord.accept(order);
                btnStart_reset(e.getSource());
            }
        });
        pane.getChildren().add(makeSection("ORDER TERMINAL",
            new VBox(8, orderArea, btnStart)
        ));
    }

    private void btnStart_reset(Object src) {
        if (src instanceof Button b) {
            b.setDisable(true);
            b.setText("⏳ RUNNING…");
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

    private Button makeButton(String text, String style, javafx.event.EventHandler<javafx.event.ActionEvent> h) {
        Button btn = new Button(text);
        btn.getStyleClass().add(style);
        btn.setMaxWidth(Double.MAX_VALUE);
        btn.setOnAction(h);
        return btn;
    }
}
