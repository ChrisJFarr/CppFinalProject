cmake_minimum_required(VERSION 3.7)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(NeuralNetwork)

add_executable(NeuralNetwork 
                src/model.h src/model.cpp 
                src/layer.h src/layer.cpp       
                src/matrix.h src/matrix.cpp 
                src/utils.h src/utils.cpp 
                src/constants.h
                src/main.cpp
                )
