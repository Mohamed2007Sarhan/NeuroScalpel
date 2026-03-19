package com.neuroscalpel.ui;

import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.text.*;

/** Cyberpunk live terminal for streaming pipeline logs. */
public class TerminalPane {

    private final VBox pane = new VBox(0);
    private final TextFlow logFlow = new TextFlow();
    private final ScrollPane scroll = new ScrollPane(logFlow);

    public TerminalPane() {
        buildHeader();
        buildBody();
        append("> NeuroScalpel Java Edition ready. Awaiting orders…\n", "#6a7b8a");
    }

    public VBox getPane() { return pane; }

    public void append(String text, String hexColor) {
        Text t = new Text(text);
        t.setStyle("-fx-fill: " + hexColor + ";");
        t.setFont(Font.font("Cascadia Code", 12));
        logFlow.getChildren().add(t);
        // Auto-scroll
        logFlow.layout();
        scroll.setVvalue(1.0);
    }

    private void buildHeader() {
        HBox header = new HBox(8);
        header.getStyleClass().add("terminal-header");
        header.setPadding(new Insets(8, 16, 8, 16));

        Label dot1 = new Label("●"); dot1.setStyle("-fx-text-fill: #ff5f57;");
        Label dot2 = new Label("●"); dot2.setStyle("-fx-text-fill: #febc2e;");
        Label dot3 = new Label("●"); dot3.setStyle("-fx-text-fill: #28c840;");

        Label title = new Label("NEURAL SURGERY TERMINAL");
        title.getStyleClass().add("terminal-title");
        HBox.setHgrow(title, Priority.ALWAYS);

        Button clear = new Button("CLEAR");
        clear.getStyleClass().add("btn-clear");
        clear.setOnAction(e -> logFlow.getChildren().clear());

        header.getChildren().addAll(dot1, dot2, dot3, title, clear);
        pane.getChildren().add(header);
    }

    private void buildBody() {
        scroll.setFitToWidth(true);
        scroll.getStyleClass().add("terminal-scroll");
        logFlow.setPadding(new Insets(12, 16, 12, 16));
        logFlow.setLineSpacing(2);
        logFlow.setStyle("-fx-background-color: rgba(0,0,0,0.5);");
        VBox.setVgrow(scroll, Priority.ALWAYS);
        scroll.setMinHeight(180);
        pane.getChildren().add(scroll);
    }
}
