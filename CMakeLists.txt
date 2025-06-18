cmake_minimum_required(VERSION 3.10.0)
project(PLAReverse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(PLAReverse src/main.cpp src/pokemonGenerator.cpp src/pokemonEntity.cpp src/pokemonSlot.cpp src/outbreakSpawner.cpp src/matrix.cpp src/seedReversal.cpp)
set_target_properties(PLAReverse PROPERTIES CUDA_ARCHITECTURES native)
include_directories(pokemon_cuda)
add_subdirectory(pokemon_cuda)
target_link_libraries(PLAReverse pokemon_cuda)
find_package(CUDAToolkit)
target_link_libraries(PLAReverse CUDA::cudart)
target_link_libraries(PLAReverse CUDA::cufft)

add_custom_command(TARGET PLAReverse POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:PLAReverse>/resources
    COMMENT "Copying resources to output directory"
)