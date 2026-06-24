// original author: sentientstardust

#ifndef slic3r_TextureMappingOffset_hpp_
#define slic3r_TextureMappingOffset_hpp_

#include "libslic3r.h"
#include "Point.hpp"
#include "TextureMapping.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace Slic3r {

class Layer;
class PrintObject;

enum class TextureMappingRawSurfaceUsage {
    Side,
    Flat
};

struct TextureMappingOffsetWeightField {
    float min_x_mm { 0.f };
    float min_y_mm { 0.f };
    float bucket_width_mm { 1.f };
    float bucket_height_mm { 1.f };
    float sample_lookup_radius_mm { 0.f };
    int bucket_width { 0 };
    int bucket_height { 0 };
    size_t component_count { 0 };
    std::vector<float> sample_x_mm;
    std::vector<float> sample_y_mm;
    std::vector<float> sample_weight;
    std::vector<float> sample_r;
    std::vector<float> sample_g;
    std::vector<float> sample_b;
    std::vector<float> sample_normal_z;
    std::vector<float> sample_component_weights;
    std::vector<std::vector<uint32_t>> buckets;
    std::vector<float> fallback_weights;
    bool raw_component_weights_from_texture { false };
    bool raw_top_surface_labels_from_texture { false };
    bool binary_dithered { false };

    bool empty() const
    {
        return bucket_width <= 0 ||
               bucket_height <= 0 ||
               component_count == 0 ||
               sample_x_mm.empty() ||
               sample_y_mm.size() != sample_x_mm.size() ||
               sample_weight.size() != sample_x_mm.size() ||
               sample_component_weights.size() != sample_x_mm.size() * component_count;
    }
};

struct TextureMappingOffsetContext {
    bool                            vertex_color_match_mode { false };
    bool                            linear_gradient_mode { false };
    bool                            object_center_mode { false };
    bool                            high_resolution_texture_sampling { false };
    bool                            compact_offset_mode { false };
    bool                            dithering_enabled { false };
    bool                            halftone_dithering_enabled { false };
    bool                            halftone_increased_detail_enabled { false };
    bool                            halftone_v2_enabled { false };
    bool                            nonlinear_offset_adjustment { false };
    int                             generic_solver_mix_model { TextureMappingZone::DefaultGenericSolverMixModel };
    Point                           object_center;
    Vec3f                           linear_gradient_start_mm { Vec3f::Zero() };
    Vec3f                           linear_gradient_end_mm { Vec3f::Zero() };
    bool                            linear_gradient_radial_mode { false };
    float                           linear_gradient_radius_mm { 0.f };
    std::vector<TextureMappingZone::LinearGradientStop> linear_gradient_stops;
    unsigned int                    active_component_id { 0 };
    size_t                          active_component_idx { size_t(-1) };
    std::vector<unsigned int>       component_ids;
    std::vector<std::array<float, 3>> component_colors;
    std::vector<float>              component_distances_mm;
    std::vector<float>              rotated_angles;
    TextureMappingOffsetWeightField weight_field;
    float                           inset_strength_reference_mm { 0.f };
    float                           signed_fade_factor { 1.f };
    float                           fade_factor { 1.f };
    float                           max_width_delta_mm { 0.f };
    float                           dither_pitch_mm { TextureMappingZone::DefaultDitheringResolutionMm };
    float                           halftone_dot_size_mm { TextureMappingZone::DefaultHalftoneDotSizeMm };
    float                           active_halftone_angle_deg { 0.f };
    float                           active_component_strength_factor { 1.f };
    float                           active_component_minimum_offset_factor { 0.f };
    float                           active_component_td_width_factor { 1.f };
    float                           base_outer_width_mm { 0.f };
    float                           layer_height_mm { 0.f };
    float                           sample_z_mm { std::numeric_limits<float>::quiet_NaN() };
    const Layer                    *layer { nullptr };
};

std::vector<unsigned int> decode_texture_mapping_offset_component_ids(const TextureMappingZone &zone, size_t num_physical);
void append_texture_mapping_raw_top_surface_component_ids(const PrintObject             &print_object,
                                                          const TextureMappingZone      &zone,
                                                          const std::vector<std::string> &filament_colours,
                                                          std::vector<unsigned int>    &component_ids,
                                                          size_t                        num_physical,
                                                          std::optional<int>            depth = std::nullopt);
float normalize_texture_mapping_offset_angle_deg(float angle);
float texture_mapping_offset_fade_factor(int fade_mode, float progress01);
float texture_mapping_offset_filament_strength_factor(const TextureMappingZone &zone, unsigned int physical_filament_id);
float texture_mapping_offset_filament_minimum_offset_factor(const TextureMappingZone &zone, unsigned int physical_filament_id);

std::optional<TextureMappingOffsetContext> build_texture_mapping_offset_context_for_layer(
    const PrintObject       &print_object,
    const Layer             &layer,
    const TextureMappingZone &zone,
    unsigned int             texture_zone_id,
    unsigned int             active_component_id_override = 0,
    std::optional<float>     base_outer_width_mm_override = std::nullopt,
    std::optional<float>     layer_height_mm_override = std::nullopt,
    std::optional<Vec2d>     plate_origin_mm_override = std::nullopt,
    std::optional<float>     min_outer_width_mm_override = std::nullopt,
    std::optional<std::array<float, 4>> image_background_rgba_override = std::nullopt,
    std::optional<float>     sample_z_mm_override = std::nullopt,
    std::optional<float>     texture_sample_pitch_mm_override = std::nullopt,
    std::optional<int>       raw_top_surface_depth_override = std::nullopt,
    std::optional<float>     filament_overhang_contrast_pct_override = std::nullopt,
    TextureMappingRawSurfaceUsage raw_surface_usage = TextureMappingRawSurfaceUsage::Side,
    bool                          allow_zero_width_delta_for_sampling = false);

std::vector<float> sample_weight_field_components(const TextureMappingOffsetWeightField &weight_field,
                                                  float                                  x_mm,
                                                  float                                  y_mm,
                                                  bool                                   high_resolution_texture_sampling);

std::optional<std::array<float, 3>> sample_weight_field_rgb(const TextureMappingOffsetWeightField &weight_field,
                                                            float                                  x_mm,
                                                            float                                  y_mm,
                                                            bool                                   high_resolution_texture_sampling);

std::optional<float> sample_weight_field_normal_z(const TextureMappingOffsetWeightField &weight_field,
                                                  float                                  x_mm,
                                                  float                                  y_mm,
                                                  bool                                   high_resolution_texture_sampling);

std::vector<float> texture_mapping_offset_component_weights_at_point(const TextureMappingOffsetContext &context,
                                                                     float                              x_mm,
                                                                     float                              y_mm,
                                                                     float                              z_mm);

std::optional<std::array<float, 3>> texture_mapping_offset_target_rgb_at_point(const TextureMappingOffsetContext &context,
                                                                               float                              x_mm,
                                                                               float                              y_mm,
                                                                               float                              z_mm);

float texture_mapping_offset_surface_inset_mm(const TextureMappingOffsetContext &context,
                                              const Point                       &point,
                                              double                             inward_x,
                                              double                             inward_y,
                                              float                              surface_u_mm = std::numeric_limits<float>::quiet_NaN(),
                                              float                              surface_v_mm = std::numeric_limits<float>::quiet_NaN());

} // namespace Slic3r

#endif
