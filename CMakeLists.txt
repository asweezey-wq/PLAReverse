cmake_minimum_required(VERSION 3.10.0)
project(PLAReverse)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(PLAReverse src/main.cpp src/pokemonGenerator.cpp src/pokemonEntity.cpp src/pokemonSlot.cpp src/outbreakSpawner.cpp src/matrix.cpp src/seedReversal.cpp)

add_custom_command(TARGET PLAReverse POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:PLAReverse>/resources
    COMMENT "Copying resources to output directory"
)