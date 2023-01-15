cmake_minimum_required(VERSION 3.0.0)
project(ExposureFusion VERSION 0.1.0)
include(CTest)
enable_testing()


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ExposureFusion main.cpp main.hpp)

target_link_libraries(ExposureFusion ${OpenCV_LIBS})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
include(CPack)
