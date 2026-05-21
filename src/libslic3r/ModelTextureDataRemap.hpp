// original author: sentientstardust

#ifndef slic3r_ModelTextureDataRemap_hpp_
#define slic3r_ModelTextureDataRemap_hpp_

#include "Model.hpp"
#include "TriangleMesh.hpp"

#include <array>
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
    bool region_painting_present { false };
    bool region_painting_transfer_needed { false };
    int region_base_filament_id { 1 };
    std::vector<std::vector<TriangleSelector::FacetStateTriangle>> region_facets_per_type;
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
    bool region_painting_touched { false };
    bool region_painting_valid { false };
    bool region_painting_remap_failed { false };
    TriangleSelector::TriangleSplittingData region_painting_data;
};

struct SimplifyTextureDataRemapOptions
{
    bool remap_color_data { true };
    bool remap_region_painting { true };
};

struct MultiSourceTextureDataSource
{
    SimplifyTextureDataSnapshot snapshot;
    ColorRGBA fallback_color { 1.f, 1.f, 1.f, 1.f };
};

struct MultiSourceTextureTriangleProvenance
{
    bool valid { false };
    size_t source_index { 0 };
    size_t source_triangle { 0 };
    std::array<Vec3f, 3> source_barycentric;
};

using SimplifyTextureCancelFn = std::function<void()>;
using SimplifyTextureProgressFn = std::function<void(int)>;

bool model_volume_region_painting_needs_remap(const ModelVolume &volume);

SimplifyTextureDataSnapshot snapshot_simplify_texture_data(const ModelVolume &volume);

void transform_simplify_texture_data_snapshot(SimplifyTextureDataSnapshot &snapshot, const Transform3d &transform);

SimplifyTextureDataResult remap_simplify_texture_data(const SimplifyTextureDataSnapshot &snapshot,
                                                       const indexed_triangle_set       &simplified_mesh,
                                                       const SimplifyTextureCancelFn    &throw_on_cancel = {},
                                                       const SimplifyTextureProgressFn  &status_fn = {},
                                                       const SimplifyTextureDataRemapOptions &options = SimplifyTextureDataRemapOptions());

SimplifyTextureDataResult remap_multi_source_texture_data(const std::vector<MultiSourceTextureDataSource> &sources,
                                                          const indexed_triangle_set                      &target_mesh,
                                                          const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                          const SimplifyTextureCancelFn                   &throw_on_cancel = {},
                                                          const SimplifyTextureProgressFn                 &status_fn = {},
                                                          const SimplifyTextureDataRemapOptions           &options = SimplifyTextureDataRemapOptions());

void apply_simplify_texture_data_result(ModelVolume &volume, SimplifyTextureDataResult &&result);

} // namespace Slic3r

#endif
