#ifndef slic3r_Format_ImportedTexture_hpp_
#define slic3r_Format_ImportedTexture_hpp_

#include "libslic3r/Point.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Slic3r {

struct ImportedTextureImage
{
    std::string          name;
    std::vector<uint8_t> rgba;
    uint32_t             width{0};
    uint32_t             height{0};
};

bool checked_rgba_buffer_size(size_t width, size_t height, size_t &buffer_size);
bool decode_image_texture_rgba_from_file(const std::string &texture_path,
                                         std::vector<uint8_t> &out_rgba,
                                         uint32_t &out_width,
                                         uint32_t &out_height);
bool decode_image_texture_rgba_from_memory(const uint8_t *data,
                                           size_t data_size,
                                           const std::string &mime_type_or_name,
                                           std::vector<uint8_t> &out_rgba,
                                           uint32_t &out_width,
                                           uint32_t &out_height);
bool is_supported_image_texture_path(const std::string &texture_path);
bool build_imported_texture_atlas(const std::vector<ImportedTextureImage> &textures,
                                  const std::vector<int> &triangle_texture_indices,
                                  std::vector<std::array<Vec2f, 3>> &triangle_uvs,
                                  std::vector<uint8_t> &triangle_uv_valid,
                                  std::vector<uint8_t> &atlas_rgba,
                                  uint32_t &atlas_width,
                                  uint32_t &atlas_height);

} // namespace Slic3r

#endif
