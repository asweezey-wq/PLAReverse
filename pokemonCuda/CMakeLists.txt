
cmake_minimum_required(VERSION 3.17)
project(pokemonCuda LANGUAGES CXX CUDA)
add_library(pokemonCuda STATIC pokemonCuda.cu)
set_target_properties(pokemonCuda PROPERTIES CUDA_ARCHITECTURES native)

target_compile_features(pokemonCuda PUBLIC cxx_std_11)
set_target_properties(pokemonCuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(pokemonCuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
