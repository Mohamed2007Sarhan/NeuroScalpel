#include <QApplication>
#include "ui/MainWindow.h"

int main(int argc, char *argv[]) {
    // Required to prevent data-viz OpenGL context issues on some setups
    // QCoreApplication::setAttribute(Qt::AA_UseOpenGLES);

    QApplication app(argc, argv);
    
    MainWindow window;
    window.show();
    
    return app.exec();
}
