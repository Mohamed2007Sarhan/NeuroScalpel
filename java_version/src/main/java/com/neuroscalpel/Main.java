package com.neuroscalpel;

import com.neuroscalpel.ui.MainWindow;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;

/**
 * NeuroScalpel — Java/JavaFX Entry Point
 * Run with: mvn javafx:run
 */
public class Main extends Application {

    @Override
    public void start(Stage primaryStage) {
        MainWindow mainWindow = new MainWindow();
        Scene scene = new Scene(mainWindow.getRoot(), 1600, 960);
        scene.getStylesheets().add(
            getClass().getResource("/styles/cyberpunk.css").toExternalForm()
        );

        primaryStage.setTitle("NeuroScalpel — LLM Neural Surgery [Java Edition]");
        primaryStage.setScene(scene);
        primaryStage.setMinWidth(1200);
        primaryStage.setMinHeight(700);

        try {
            primaryStage.getIcons().add(
                new Image(getClass().getResourceAsStream("/img/icon.png"))
            );
        } catch (Exception ignored) {}

        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
