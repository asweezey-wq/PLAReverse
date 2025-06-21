
cmake_minimum_required(VERSION 3.17)
project(pokemon_cuda LANGUAGES CXX CUDA)
add_library(pokemon_cuda STATIC pokemon.cu ivReversal.cu generatorReversal.cu totalReversal.cu)
set_target_properties(pokemon_cuda PROPERTIES CUDA_ARCHITECTURES native)

target_compile_features(pokemon_cuda PUBLIC cxx_std_11)
set_target_properties(pokemon_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(pokemon_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
