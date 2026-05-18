// original author: sentientstardust

#ifndef slic3r_TextureMappingOffset_hpp_
#define slic3r_TextureMappingOffset_hpp_

#include "libslic3r.h"
#include "Point.hpp"
#include "TextureMapping.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace Slic3r {

class Layer;
class PrintObject;

struct TextureMappingOffsetWeightField {
    float min_x_mm { 0.f };
    float min_y_mm { 0.f };
    float bucket_width_mm { 1.f };
    float bucket_height_mm { 1.f };
    int bucket_width { 0 };
    int bucket_height { 0 };
    size_t component_count { 0 };
    std::vector<float> sample_x_mm;
    std::vector<float> sample_y_mm;
    std::vector<float> sample_weight;
    std::vector<float> sample_component_weights;
    std::vector<std::vector<uint32_t>> buckets;
    std::vector<float> fallback_weights;
    bool raw_component_weights_from_texture { false };

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
    bool                            object_center_mode { false };
    bool                            high_resolution_texture_sampling { false };
    bool                            compact_offset_mode { false };
    bool                            nonlinear_offset_adjustment { false };
    Point                           object_center;
    unsigned int                    active_component_id { 0 };
    size_t                          active_component_idx { size_t(-1) };
    std::vector<unsigned int>       component_ids;
    std::vector<float>              component_distances_mm;
    std::vector<float>              rotated_angles;
    TextureMappingOffsetWeightField weight_field;
    float                           inset_strength_reference_mm { 0.f };
    float                           fade_factor { 1.f };
    float                           max_width_delta_mm { 0.f };
    float                           active_component_strength_factor { 1.f };
    float                           active_component_minimum_offset_factor { 0.f };
    float                           active_component_td_width_factor { 1.f };
    float                           base_outer_width_mm { 0.f };
    float                           layer_height_mm { 0.f };
    const Layer                    *layer { nullptr };
};

std::vector<unsigned int> decode_texture_mapping_offset_component_ids(const TextureMappingZone &zone, size_t num_physical);
float normalize_texture_mapping_offset_angle_deg(float angle);
float texture_mapping_offset_fade_factor(int fade_mode, float progress01);
float texture_mapping_offset_filament_strength_factor(const TextureMappingZone &zone, unsigned int physical_filament_id);
float texture_mapping_offset_filament_minimum_offset_factor(const TextureMappingZone &zone, unsigned int physical_filament_id);

std::optional<TextureMappingOffsetContext> build_texture_mapping_offset_context_for_layer(
    const PrintObject       &print_object,
    const Layer             &layer,
    const TextureMappingZone &zone,
    unsigned int             texture_zone_id);

float texture_mapping_offset_surface_inset_mm(const TextureMappingOffsetContext &context,
                                              const Point                       &point,
                                              double                             inward_x,
                                              double                             inward_y);

} // namespace Slic3r

#endif
