cmake_minimum_required(VERSION 3.10.0)
project(PLAReverse)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(PLAReverse src/main.cpp src/pokemonGenerator.cpp src/pokemonEntity.cpp)

