cmake_minimum_required(VERSION 3.16)
project(NeuroScalpel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Configure Qt6
find_package(Qt6 COMPONENTS Core Gui Widgets Network DataVisualization REQUIRED)

# Attempt to find LibTorch (PyTorch C++ Frontend)
# If the user doesn't have it, we can comment this out or wrap in IF block.
# Assuming standard Torch configuration:
# set(Torch_DIR /path/to/libtorch/share/cmake/Torch)
find_package(Torch QUIET)

add_executable(NeuroScalpel
    main.cpp
    core/model_backend.cpp
    core/model_backend.h
    core/point_and_layer_detect.cpp
    core/point_and_layer_detect.h
    ui/MainWindow.cpp
    ui/MainWindow.h
    ui/panels/dashboard.cpp
    ui/panels/dashboard.h
    ui/panels/feature_extractor.cpp
    ui/panels/feature_extractor.h
    ui/panels/order_terminal.cpp
    ui/panels/order_terminal.h
    ui/visualizer/point_cloud_3d.cpp
    ui/visualizer/point_cloud_3d.h
)

target_link_libraries(NeuroScalpel PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Network
    Qt6::DataVisualization
)

if(Torch_FOUND)
    target_link_libraries(NeuroScalpel PRIVATE "${TORCH_LIBRARIES}")
    set_property(TARGET NeuroScalpel PROPERTY CXX_STANDARD 17)
    target_compile_definitions(NeuroScalpel PRIVATE HAS_LIBTORCH=1)
endif()

# Set standard executable name and properties
set_target_properties(NeuroScalpel PROPERTIES
    WIN32_EXECUTABLE OFF # Keep console open for debug logs
)
