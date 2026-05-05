cmake_minimum_required(VERSION 3.13)
project(pigment_painter)

find_package(PNG REQUIRED)

add_library(pigment_painter STATIC
    lut_wide.png.c
    pigment_painter_mixer.cpp
    pigment_painter_mixer.hpp
)

target_include_directories(pigment_painter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(pigment_painter PUBLIC cxx_std_17)
target_link_libraries(pigment_painter PRIVATE PNG::PNG)
