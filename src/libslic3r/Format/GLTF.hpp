#ifndef slic3r_Format_GLTF_hpp_
#define slic3r_Format_GLTF_hpp_

#include "ImportedTexture.hpp"
#include "OBJ.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace Slic3r {

class Model;

struct GltfImportInfo
{
    std::vector<RGBA>                         vertex_colors;
    std::vector<uint32_t>                     vertex_colors_rgba;
    bool                                      has_vertex_colors{false};
    std::vector<RGBA>                         material_colors;
    bool                                      has_material_colors{false};
    bool                                      is_single_material_color{false};
    std::vector<std::array<Vec2f, 3>>         triangle_uvs;
    std::vector<uint8_t>                      triangle_uvs_valid;
    std::vector<int>                          triangle_texture_indices;
    std::vector<ImportedTextureImage>         textures;
};

bool load_gltf(const char *path, Model *model, GltfImportInfo &import_info, std::string &message);

} // namespace Slic3r

#endif
