module com.neuroscalpel {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.net.http;
    requires com.google.gson;
    requires org.slf4j;

    opens com.neuroscalpel to javafx.fxml;
    opens com.neuroscalpel.ui to javafx.fxml;

    exports com.neuroscalpel;
    exports com.neuroscalpel.ui;
    exports com.neuroscalpel.core;
}
