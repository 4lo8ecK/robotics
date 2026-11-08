cmake_minimum_required(VERSION 3.28)
project(Robotics LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Robotics ./main.cpp)

target_sources(Robotics PUBLIC FILE_SET CXX_MODULES FILES
	./base/base.cppm
	./base/math.cppm
	./base/typedefs.cppm
	)