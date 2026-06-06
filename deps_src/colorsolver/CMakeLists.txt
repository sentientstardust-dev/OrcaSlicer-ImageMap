cmake_minimum_required(VERSION 3.13)
project(colorsolver)

add_library(colorsolver STATIC
    ColorSolver.cpp
    ColorSolver.hpp
)

target_include_directories(colorsolver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(colorsolver PUBLIC cxx_std_17)
target_link_libraries(colorsolver PUBLIC pigment_painter prusa_fdm_mixer)
