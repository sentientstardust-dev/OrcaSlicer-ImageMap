add_library(tinygltf_v3 STATIC tiny_gltf_v3.c)

target_include_directories(tinygltf_v3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(tinygltf_v3 PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)
