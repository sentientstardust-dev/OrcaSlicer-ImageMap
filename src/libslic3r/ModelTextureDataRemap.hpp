// original author: sentientstardust

#ifndef slic3r_ModelTextureDataRemap_hpp_
#define slic3r_ModelTextureDataRemap_hpp_

#include "Model.hpp"
#include "TriangleMesh.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Slic3r {

enum class SimplifyColorSource
{
    None,
    RgbaData,
    ImageTexture,
    VertexColors
};

struct SimplifyTextureDataSnapshot
{
    SimplifyColorSource source { SimplifyColorSource::None };
    indexed_triangle_set source_mesh;
    std::vector<ColorFacetTriangle> rgba_facets;
    std::string rgba_metadata_json;
    std::vector<uint8_t> texture_rgba;
    std::vector<float> texture_uvs_per_face;
    std::vector<uint8_t> texture_uv_valid;
    std::vector<uint8_t> texture_raw_filament_offsets;
    uint32_t texture_width { 0 };
    uint32_t texture_height { 0 };
    uint32_t texture_raw_channels { 0 };
    std::string texture_raw_metadata_json;
    int uv_map_generator_version { 0 };
    std::vector<uint32_t> vertex_colors_rgba;
};

struct SimplifyTextureDataResult
{
    SimplifyColorSource source { SimplifyColorSource::None };
    bool remap_failed { false };
    bool used_fallback_rgba { false };
    std::unique_ptr<ColorFacetsAnnotation> rgba_data;
    std::vector<uint8_t> texture_rgba;
    std::vector<float> texture_uvs_per_face;
    std::vector<uint8_t> texture_uv_valid;
    std::vector<uint8_t> texture_raw_filament_offsets;
    uint32_t texture_width { 0 };
    uint32_t texture_height { 0 };
    uint32_t texture_raw_channels { 0 };
    std::string texture_raw_metadata_json;
    int uv_map_generator_version { 0 };
    std::vector<uint32_t> vertex_colors_rgba;
};

using SimplifyTextureCancelFn = std::function<void()>;
using SimplifyTextureProgressFn = std::function<void(int)>;

SimplifyTextureDataSnapshot snapshot_simplify_texture_data(const ModelVolume &volume);

SimplifyTextureDataResult remap_simplify_texture_data(const SimplifyTextureDataSnapshot &snapshot,
                                                       const indexed_triangle_set       &simplified_mesh,
                                                       const SimplifyTextureCancelFn    &throw_on_cancel = {},
                                                       const SimplifyTextureProgressFn  &status_fn = {});

void apply_simplify_texture_data_result(ModelVolume &volume, SimplifyTextureDataResult &&result);

} // namespace Slic3r

#endif
