cmake_minimum_required(VERSION 3.28)
project(Robotics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(Robotics src/main.cpp)

target_include_directories(Robotics PRIVATE ${PROJECT_SOURCE_DIR}/libs)