cmake_minimum_required(VERSION 2.8)
project(FaceRecognizier)
find_package(OpenCV REQUIRED)
SET(SRC
    main.cpp
)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(FaceRecognizier ${SRC})
target_link_libraries(FaceRecognizier ${OpenCV_LIBS})