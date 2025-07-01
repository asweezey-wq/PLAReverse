cmake_minimum_required(VERSION 3.10.0)
project(PLAReverse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(Qt6_DIR "c:/Qt/6.9.1/msvc2022_64/lib/cmake/Qt6")
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
qt_standard_project_setup()

set(MAIN_SRC_DIR "src")
set(UI_SRC_DIR "src/ui")
FILE(GLOB UIFILES ${UI_SRC_DIR}/*.ui)
FILE(GLOB UISRCFILES ${UI_SRC_DIR}/*.cpp)
FILE(GLOB SRCFILES ${MAIN_SRC_DIR}/*.cpp)
include_directories(${MAIN_SRC_DIR})
include_directories(${UI_SRC_DIR})

qt_add_executable(PLAReverse ${SRCFILES} ${UISRCFILES})
qt_add_ui(PLAReverse SOURCES ${UIFILES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
include_directories(pokemonCuda)
add_subdirectory(pokemonCuda)
target_link_libraries(${PROJECT_NAME} PRIVATE pokemonCuda)
find_package(CUDAToolkit)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cufft)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets Qt6::Core)
add_definitions(${Qt6Core_DEFINITIONS})

# get absolute path to qmake, then use it to find windeployqt executable

get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)
get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)

function(windeployqt target)

    # POST_BUILD step
    # - after build, we have a bin/lib for analyzing qt dependencies
    # - we run windeployqt on target and deploy Qt libs

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${_qt_bin_dir}/windeployqt.exe"         
                --verbose 1
                --release
                --no-svg
                --no-opengl
                --no-opengl-sw
                --no-compiler-runtime
                --no-system-d3d-compiler
                \"$<TARGET_FILE:${target}>\"
        COMMENT "Deploying Qt libraries using windeployqt for compilation target '${target}' ..."
    )

endfunction()
windeployqt(PLAReverse)

add_custom_command(TARGET PLAReverse POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:PLAReverse>/resources
    COMMENT "Copying resources to output directory"
)