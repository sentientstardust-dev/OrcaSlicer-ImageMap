// original author: sentientstardust

#include "TextureMappingOffset.hpp"

#include "BoundingBox.hpp"
#include "Color.hpp"
#include "ColorSolver.hpp"
#include "Config.hpp"
#include "Geometry.hpp"
#include "ImageMapRawFilamentOffsetAtlas.hpp"
#include "Layer.hpp"
#include "Model.hpp"
#include "Print.hpp"
#include "TriangleMesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Slic3r {

namespace {

struct TextureSampleData {
    std::array<float, 4> rgba { { 0.f, 0.f, 0.f, 1.f } };
    std::vector<float> raw_component_weights;
    bool raw_component_weights_from_texture { false };
    float normal_z { std::numeric_limits<float>::quiet_NaN() };
};

struct WeightedTextureSample {
    float x_mm { 0.f };
    float y_mm { 0.f };
    std::array<float, 4> rgba { { 0.f, 0.f, 0.f, 1.f } };
    std::vector<float> raw_component_weights;
    bool raw_component_weights_from_texture { false };
    float normal_z { std::numeric_limits<float>::quiet_NaN() };
    float weight { 0.f };
};

struct LayerPlaneSamplePoint {
    Vec3d p;
    Vec3f barycentric;
};

struct TransmissionDistanceCalibrationContext {
    bool               enabled { false };
    int                mode { int(TextureMappingZone::TDCalibrationNone) };
    std::vector<float> own_width_factors;
    std::vector<float> neighbor_opacity_ratios;
};

float clamp01f(float v)
{
    if (!std::isfinite(v))
        return 0.f;
    return std::clamp(v, 0.f, 1.f);
}

Vec3f texture_mapping_array_point(const std::array<float, 3> &point)
{
    return Vec3f(point[0], point[1], point[2]);
}

bool texture_mapping_vec3_is_finite(const Vec3f &point)
{
    return std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z());
}

int texture_mapping_model_object_backup_id(const ModelObject *object)
{
    const Model *model = object != nullptr ? object->get_model() : nullptr;
    return model != nullptr ? model->find_object_backup_id(*object) : -1;
}

bool texture_mapping_anchor_matches_model_object(const ModelObject *object,
                                                 const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (object == nullptr)
        return false;
    if (anchor.object_backup_id >= 0 && texture_mapping_model_object_backup_id(object) == anchor.object_backup_id)
        return true;
    if (anchor.object_id != 0 && object->id().id == anchor.object_id)
        return true;
    const Model *model = object->get_model();
    return model != nullptr && anchor.object_index_valid && anchor.object_index < model->objects.size() &&
           model->objects[anchor.object_index] == object;
}

const ModelObject *texture_mapping_anchor_model_object(const Print &print,
                                                       const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (anchor.object_backup_id >= 0) {
        for (const PrintObject *print_object : print.objects()) {
            const ModelObject *model_object = print_object != nullptr ? print_object->model_object() : nullptr;
            if (texture_mapping_model_object_backup_id(model_object) == anchor.object_backup_id)
                return model_object;
        }
    }
    if (anchor.object_id != 0) {
        for (const PrintObject *print_object : print.objects()) {
            const ModelObject *model_object = print_object != nullptr ? print_object->model_object() : nullptr;
            if (model_object != nullptr && model_object->id().id == anchor.object_id)
                return model_object;
        }
    }
    if (anchor.object_index_valid) {
        for (const PrintObject *print_object : print.objects()) {
            const ModelObject *model_object = print_object != nullptr ? print_object->model_object() : nullptr;
            const Model *model = model_object != nullptr ? model_object->get_model() : nullptr;
            if (model != nullptr && anchor.object_index < model->objects.size() &&
                model->objects[anchor.object_index] == model_object)
                return model_object;
        }
    }
    return nullptr;
}

const PrintObject *texture_mapping_anchor_print_object(const Print &print,
                                                       const TextureMappingZone::LinearGradientAnchor &anchor)
{
    for (const PrintObject *print_object : print.objects()) {
        const ModelObject *model_object = print_object != nullptr ? print_object->model_object() : nullptr;
        if (texture_mapping_anchor_matches_model_object(model_object, anchor))
            return print_object;
    }
    return nullptr;
}

const ModelInstance *texture_mapping_anchor_model_instance(const ModelObject *object,
                                                          const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (object == nullptr)
        return nullptr;
    if (anchor.instance_loaded_id != 0) {
        for (const ModelInstance *instance : object->instances)
            if (instance != nullptr && instance->loaded_id == anchor.instance_loaded_id)
                return instance;
    }
    if (anchor.instance_id != 0) {
        for (const ModelInstance *instance : object->instances)
            if (instance != nullptr && instance->id().id == anchor.instance_id)
                return instance;
    }
    if (anchor.instance_index_valid && anchor.instance_index < object->instances.size())
        return object->instances[anchor.instance_index];
    return object->instances.size() == 1 ? object->instances.front() : nullptr;
}

std::optional<Vec3f> texture_mapping_linear_gradient_anchor_global_point(const Print &print,
                                                                         const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (!anchor.valid)
        return std::nullopt;

    const Vec3f local = texture_mapping_array_point(anchor.local_point);
    const ModelObject *model_object = texture_mapping_anchor_model_object(print, anchor);
    const ModelInstance *instance = texture_mapping_anchor_model_instance(model_object, anchor);
    if (instance != nullptr) {
        const Vec3f global = (instance->get_matrix() * local.cast<double>()).cast<float>();
        if (texture_mapping_vec3_is_finite(global))
            return global;
    }

    const Vec3f fallback = texture_mapping_array_point(anchor.global_point);
    return texture_mapping_vec3_is_finite(fallback) ? std::optional<Vec3f>(fallback) : std::nullopt;
}

Vec2d texture_mapping_print_object_plate_origin_mm(const PrintObject &print_object,
                                                   const std::optional<Vec2d> &plate_origin_mm_override)
{
    if (plate_origin_mm_override)
        return *plate_origin_mm_override;
    if (!print_object.instances().empty()) {
        const Point &shift = print_object.instances().front().shift;
        return Vec2d(unscale<double>(shift.x()), unscale<double>(shift.y()));
    }
    return Vec2d::Zero();
}

std::optional<Vec3f> texture_mapping_linear_gradient_anchor_print_point(const PrintObject &print_object,
                                                                        const TextureMappingZone::LinearGradientAnchor &anchor,
                                                                        const std::optional<Vec2d> &plate_origin_mm_override)
{
    if (!anchor.valid)
        return std::nullopt;

    const ModelObject *model_object = print_object.model_object();
    const Print *print = print_object.print();
    if (texture_mapping_anchor_matches_model_object(model_object, anchor)) {
        const Vec3f local = texture_mapping_array_point(anchor.local_point);
        const Vec3f print_point = (print_object.trafo_centered() * local.cast<double>()).cast<float>();
        if (texture_mapping_vec3_is_finite(print_point))
            return print_point;
    }

    if (print == nullptr)
        return std::nullopt;

    const std::optional<Vec3f> global = texture_mapping_linear_gradient_anchor_global_point(*print, anchor);
    if (!global)
        return std::nullopt;
    const Vec2d plate_origin_mm = texture_mapping_print_object_plate_origin_mm(print_object, plate_origin_mm_override);
    const Vec3f print_point(global->x() - float(plate_origin_mm.x()),
                            global->y() - float(plate_origin_mm.y()),
                            global->z());
    return texture_mapping_vec3_is_finite(print_point) ? std::optional<Vec3f>(print_point) : std::nullopt;
}

std::pair<Vec3f, Vec3f> texture_mapping_default_linear_gradient_points(const PrintObject &print_object)
{
    const BoundingBox bbox = print_object.bounding_box();
    const float cx = 0.5f * (unscale<float>(bbox.min.x()) + unscale<float>(bbox.max.x()));
    const float cy = 0.5f * (unscale<float>(bbox.min.y()) + unscale<float>(bbox.max.y()));
    const float z_min = 0.f;
    const float z_max = std::max(0.01f, unscale<float>(print_object.height()));
    return { Vec3f(cx, cy, z_min), Vec3f(cx, cy, z_max) };
}

std::pair<Vec3f, float> texture_mapping_default_radial_gradient(const PrintObject &print_object)
{
    const BoundingBox bbox = print_object.bounding_box();
    const float min_x = unscale<float>(bbox.min.x());
    const float max_x = unscale<float>(bbox.max.x());
    const float min_y = unscale<float>(bbox.min.y());
    const float max_y = unscale<float>(bbox.max.y());
    const float height = std::max(0.01f, unscale<float>(print_object.height()));
    const Vec3f center(0.5f * (min_x + max_x), 0.5f * (min_y + max_y), 0.5f * height);
    const Vec3f diagonal(std::max(0.f, max_x - min_x), std::max(0.f, max_y - min_y), height);
    const float radius = std::max(0.01f, 0.5f * diagonal.norm());
    return { center, radius };
}

float texture_mapping_linear_gradient_radius_mm(const PrintObject &print_object,
                                                const TextureMappingZone &zone,
                                                float default_radius_mm)
{
    if (!zone.linear_gradient_radius_percent) {
        const float radius_mm = std::isfinite(zone.linear_gradient_radius_mm) ? zone.linear_gradient_radius_mm : default_radius_mm;
        return std::max(0.01f, radius_mm);
    }

    const float radius_value = std::isfinite(zone.linear_gradient_radius_pct) ?
        std::max(0.f, zone.linear_gradient_radius_pct) :
        TextureMappingZone::DefaultLinearGradientRadiusPct;
    if (zone.linear_gradient_start.valid) {
        const Print *print = print_object.print();
        const PrintObject *anchor_print_object = print != nullptr ?
            texture_mapping_anchor_print_object(*print, zone.linear_gradient_start) :
            nullptr;
        if (anchor_print_object != nullptr)
            return std::max(0.01f, texture_mapping_default_radial_gradient(*anchor_print_object).second * radius_value / 100.f);
        return std::max(0.01f, radius_value);
    }

    return std::max(0.01f, default_radius_mm * radius_value / 100.f);
}

int texture_mapping_color_hex_digit(char ch)
{
    return ch >= '0' && ch <= '9' ? ch - '0' :
           ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 :
           ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
}

std::optional<std::array<float, 4>> parse_texture_mapping_color_hex(const std::string &text)
{
    if (text.empty())
        return std::nullopt;

    const size_t hash_pos = text.find('#');
    const size_t start = hash_pos == std::string::npos ? 0 : hash_pos + 1;
    if (start + 6 > text.size())
        return std::nullopt;

    uint32_t packed = 0;
    for (size_t idx = 0; idx < 6; ++idx) {
        const int value = texture_mapping_color_hex_digit(text[start + idx]);
        if (value < 0)
            return std::nullopt;
        packed = (packed << 4) | uint32_t(value);
    }

    uint32_t alpha = 255;
    if (start + 8 <= text.size()) {
        alpha = 0;
        for (size_t idx = 6; idx < 8; ++idx) {
            const int value = texture_mapping_color_hex_digit(text[start + idx]);
            if (value < 0)
                return std::nullopt;
            alpha = (alpha << 4) | uint32_t(value);
        }
    }

    return std::array<float, 4> {
        clamp01f(float((packed >> 16) & 0xFFu) / 255.f),
        clamp01f(float((packed >> 8) & 0xFFu) / 255.f),
        clamp01f(float(packed & 0xFFu) / 255.f),
        clamp01f(float(alpha & 0xFFu) / 255.f)
    };
}

std::array<float, 4> unpack_rgba_u32(uint32_t packed_rgba)
{
    const float r = float((packed_rgba >> 24) & 0xFFu) / 255.f;
    const float g = float((packed_rgba >> 16) & 0xFFu) / 255.f;
    const float b = float((packed_rgba >> 8) & 0xFFu) / 255.f;
    const float a = float(packed_rgba & 0xFFu) / 255.f;
    return { clamp01f(r), clamp01f(g), clamp01f(b), clamp01f(a) };
}

std::optional<std::array<float, 4>> texture_mapping_background_color_from_config(const ModelConfigObject &config)
{
    static constexpr const char *key = "texture_mapping_background_color";
    if (!config.has(key))
        return std::nullopt;

    const ConfigOptionString *opt = dynamic_cast<const ConfigOptionString *>(config.option(key));
    if (opt == nullptr)
        return std::nullopt;

    std::optional<std::array<float, 4>> color = parse_texture_mapping_color_hex(opt->value);
    if (color)
        (*color)[3] = 1.f;
    return color;
}

std::optional<std::array<float, 4>> texture_mapping_background_color_from_metadata(const ColorFacetsAnnotation &annotation)
{
    const std::string &metadata = annotation.metadata_json();
    const std::string key = "\"background_color\":\"#";
    const size_t start = metadata.find(key);
    if (start == std::string::npos || start + key.size() + 8 > metadata.size())
        return std::nullopt;

    std::optional<std::array<float, 4>> color = parse_texture_mapping_color_hex(metadata.substr(start + key.size() - 1, 9));
    if (color)
        (*color)[3] = 1.f;
    return color;
}

std::array<float, 4> texture_mapping_background_color(const ModelVolume &volume)
{
    if (std::optional<std::array<float, 4>> color = texture_mapping_background_color_from_config(volume.config))
        return *color;
    if (volume.get_object() != nullptr) {
        if (std::optional<std::array<float, 4>> color = texture_mapping_background_color_from_config(volume.get_object()->config))
            return *color;
    }
    if (std::optional<std::array<float, 4>> color = texture_mapping_background_color_from_metadata(volume.texture_mapping_color_facets))
        return *color;
    return { 1.f, 1.f, 1.f, 1.f };
}

std::array<float, 4> composite_rgba_over_background(const std::array<float, 4> &rgba,
                                                    const std::array<float, 4> &background)
{
    const float alpha = clamp01f(rgba[3]);
    return {
        clamp01f(rgba[0] * alpha + background[0] * (1.f - alpha)),
        clamp01f(rgba[1] * alpha + background[1] * (1.f - alpha)),
        clamp01f(rgba[2] * alpha + background[2] * (1.f - alpha)),
        1.f
    };
}

float wrap_repeat01(float uv)
{
    if (!std::isfinite(uv))
        return 0.f;

    constexpr float eps = 1e-6f;
    if (uv >= -eps && uv <= 1.f + eps)
        return std::clamp(uv, 0.f, 1.f);

    float wrapped = uv - std::floor(uv);
    if (wrapped < 0.f)
        wrapped += 1.f;
    return wrapped;
}

std::array<float, 4> sample_texture_rgba_bilinear(const std::vector<uint8_t> &rgba,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  float u,
                                                  float v)
{
    if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return { 0.f, 0.f, 0.f, 1.f };

    const float uu = wrap_repeat01(u);
    const float vv = wrap_repeat01(v);
    const float x = uu * float(width > 1 ? width - 1 : 0);
    const float y = vv * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto sample_channel = [&rgba, width](size_t sx, size_t sy, size_t channel) {
        const size_t idx = (sy * size_t(width) + sx) * 4 + channel;
        return float(rgba[idx]) / 255.f;
    };

    std::array<float, 4> out {};
    for (size_t c = 0; c < 4; ++c) {
        const float c00 = sample_channel(x0, y0, c);
        const float c10 = sample_channel(x1, y0, c);
        const float c01 = sample_channel(x0, y1, c);
        const float c11 = sample_channel(x1, y1, c);
        const float cx0 = c00 + (c10 - c00) * tx;
        const float cx1 = c01 + (c11 - c01) * tx;
        out[c] = clamp01f(cx0 + (cx1 - cx0) * ty);
    }
    return out;
}

std::vector<float> sample_texture_raw_offsets_bilinear(const std::vector<uint8_t> &offsets,
                                                       uint32_t width,
                                                       uint32_t height,
                                                       uint32_t channels,
                                                       float u,
                                                       float v)
{
    std::vector<float> out;
    if (width == 0 || height == 0 || channels == 0 ||
        offsets.size() < size_t(width) * size_t(height) * size_t(channels))
        return out;

    const float uu = wrap_repeat01(u);
    const float vv = wrap_repeat01(v);
    const float x = uu * float(width > 1 ? width - 1 : 0);
    const float y = vv * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto sample_channel = [&offsets, width, channels](size_t sx, size_t sy, size_t channel) {
        const size_t idx = (sy * size_t(width) + sx) * size_t(channels) + channel;
        return float(offsets[idx]) / 255.f;
    };

    out.assign(size_t(channels), 0.f);
    for (size_t c = 0; c < out.size(); ++c) {
        const float c00 = sample_channel(x0, y0, c);
        const float c10 = sample_channel(x1, y0, c);
        const float c01 = sample_channel(x0, y1, c);
        const float c11 = sample_channel(x1, y1, c);
        const float cx0 = c00 + (c10 - c00) * tx;
        const float cx1 = c01 + (c11 - c01) * tx;
        out[c] = clamp01f(cx0 + (cx1 - cx0) * ty);
    }
    return out;
}

std::array<float, 4> raw_offset_preview_rgba(const std::vector<float> &offsets)
{
    if (offsets.empty())
        return { 0.f, 0.f, 0.f, 1.f };
    if (offsets.size() == 1)
        return { offsets[0], offsets[0], offsets[0], 1.f };
    return {
        offsets.size() > 0 ? offsets[0] : 0.f,
        offsets.size() > 1 ? offsets[1] : 0.f,
        offsets.size() > 2 ? offsets[2] : 0.f,
        1.f
    };
}

float color_distance_sq(const std::array<float, 3> &lhs, const std::array<float, 3> &rhs)
{
    const std::array<float, 3> lhs_oklab = color_solver_oklab_from_srgb(lhs);
    const std::array<float, 3> rhs_oklab = color_solver_oklab_from_srgb(rhs);
    const float dl = lhs_oklab[0] - rhs_oklab[0];
    const float da = lhs_oklab[1] - rhs_oklab[1];
    const float db = lhs_oklab[2] - rhs_oklab[2];
    return dl * dl + da * da + db * db;
}

std::array<float, 3> raw_filament_channel_color(const ImageMapRawFilament &filament, size_t channel_idx)
{
    const std::string key = image_map_raw_filament_channel_key(filament, channel_idx);
    if (key == "C")
        return { { 0.f, 1.f, 1.f } };
    if (key == "M")
        return { { 1.f, 0.f, 1.f } };
    if (key == "Y")
        return { { 1.f, 1.f, 0.f } };
    if (key == "K")
        return { { 0.f, 0.f, 0.f } };
    if (key == "W")
        return { { 1.f, 1.f, 1.f } };
    if (key == "R")
        return { { 1.f, 0.f, 0.f } };
    if (key == "G")
        return { { 0.f, 1.f, 0.f } };
    if (key == "B")
        return { { 0.f, 0.f, 1.f } };
    if (!filament.hex.empty()) {
        const std::optional<std::array<float, 4>> parsed = parse_texture_mapping_color_hex(filament.hex);
        if (parsed)
            return { { (*parsed)[0], (*parsed)[1], (*parsed)[2] } };
    }
    return { { 1.f, 1.f, 1.f } };
}

std::vector<std::string> raw_filament_color_mode_channel_keys(int filament_color_mode, size_t component_count)
{
    std::vector<std::string> keys;
    switch (std::clamp(filament_color_mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorRGB):   keys = { "R", "G", "B" }; break;
    case int(TextureMappingZone::FilamentColorCMY):   keys = { "C", "M", "Y" }; break;
    case int(TextureMappingZone::FilamentColorCMYK):  keys = { "C", "M", "Y", "K" }; break;
    case int(TextureMappingZone::FilamentColorCMYW):  keys = { "C", "M", "Y", "W" }; break;
    case int(TextureMappingZone::FilamentColorRGBK):  keys = { "R", "G", "B", "K" }; break;
    case int(TextureMappingZone::FilamentColorRGBW):  keys = { "R", "G", "B", "W" }; break;
    case int(TextureMappingZone::FilamentColorBW):    keys = { "K", "W" }; break;
    case int(TextureMappingZone::FilamentColorCMYKW): keys = { "C", "M", "Y", "K", "W" }; break;
    case int(TextureMappingZone::FilamentColorRGBKW): keys = { "R", "G", "B", "K", "W" }; break;
    default: break;
    }
    if (keys.size() > component_count)
        keys.resize(component_count);
    return keys;
}

std::vector<size_t> raw_component_source_channels(const std::string &metadata_json,
                                                  uint32_t source_channels,
                                                  int filament_color_mode,
                                                  size_t component_count,
                                                  const std::vector<std::array<float, 3>> &component_colors)
{
    if (source_channels == 0 || component_count == 0)
        return {};

    const size_t sentinel = std::numeric_limits<size_t>::max();
    std::vector<size_t> mapping(component_count, sentinel);
    const std::vector<ImageMapRawFilament> filaments =
        image_map_raw_filaments_from_metadata_json(metadata_json, source_channels);
    if (filaments.size() != size_t(source_channels))
        return {};

    std::vector<std::string> source_keys(static_cast<size_t>(source_channels));
    std::vector<std::array<float, 3>> source_colors(static_cast<size_t>(source_channels));
    for (size_t channel = 0; channel < filaments.size(); ++channel) {
        const std::string key = image_map_raw_filament_channel_key(filaments[channel], channel);
        if (key.size() == 1 && image_map_raw_filament_is_standard_color(key))
            source_keys[channel] = key;
        source_colors[channel] = raw_filament_channel_color(filaments[channel], channel);
    }

    const std::vector<std::string> target_keys = raw_filament_color_mode_channel_keys(filament_color_mode, component_count);
    if (!target_keys.empty()) {
        std::vector<uint8_t> used(static_cast<size_t>(source_channels), 0);
        for (size_t component_idx = 0; component_idx < target_keys.size(); ++component_idx) {
            for (size_t channel = 0; channel < source_keys.size(); ++channel) {
                if (used[channel] == 0 && source_keys[channel] == target_keys[component_idx]) {
                    mapping[component_idx] = channel;
                    used[channel] = 1;
                    break;
                }
            }
        }

        const float max_match_distance_sq =
            TextureMappingManager::poor_color_match_distance() * TextureMappingManager::poor_color_match_distance();
        for (size_t component_idx = 0; component_idx < target_keys.size(); ++component_idx) {
            if (mapping[component_idx] != sentinel)
                continue;
            const std::array<float, 3> target_color =
                raw_filament_channel_color({ 0, target_keys[component_idx], std::string() }, component_idx);
            size_t best_channel = size_t(source_channels);
            float best_distance_sq = std::numeric_limits<float>::max();
            for (size_t channel = 0; channel < source_colors.size(); ++channel) {
                if (used[channel] != 0)
                    continue;
                const float distance_sq = color_distance_sq(source_colors[channel], target_color);
                if (distance_sq < best_distance_sq) {
                    best_distance_sq = distance_sq;
                    best_channel = channel;
                }
            }
            if (best_channel < source_colors.size() && best_distance_sq <= max_match_distance_sq) {
                mapping[component_idx] = best_channel;
                used[best_channel] = 1;
            }
        }
        return mapping;
    }

    if (component_colors.size() == component_count) {
        struct Candidate {
            float distance_sq { 0.f };
            size_t component_idx { 0 };
            size_t source_channel { 0 };
        };

        std::vector<Candidate> candidates;
        candidates.reserve(component_count * size_t(source_channels));
        for (size_t channel = 0; channel < filaments.size(); ++channel) {
            for (size_t component_idx = 0; component_idx < component_count; ++component_idx)
                candidates.push_back({ color_distance_sq(component_colors[component_idx], source_colors[channel]), component_idx, channel });
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate &lhs, const Candidate &rhs) {
            return lhs.distance_sq < rhs.distance_sq;
        });

        std::vector<uint8_t> used_components(component_count, 0);
        std::vector<uint8_t> used_sources(static_cast<size_t>(source_channels), 0);
        for (const Candidate &candidate : candidates) {
            if (used_components[candidate.component_idx] != 0 || used_sources[candidate.source_channel] != 0)
                continue;
            mapping[candidate.component_idx] = candidate.source_channel;
            used_components[candidate.component_idx] = 1;
            used_sources[candidate.source_channel] = 1;
        }
    }

    const bool has_mapping = std::any_of(mapping.begin(), mapping.end(), [sentinel](size_t value) { return value != sentinel; });
    return has_mapping ? mapping : std::vector<size_t>{};
}

std::vector<float> map_raw_sample_to_components(const std::vector<float> &raw_sample,
                                                const std::vector<size_t> &component_source_channels)
{
    if (component_source_channels.empty())
        return {};
    const size_t sentinel = std::numeric_limits<size_t>::max();
    std::vector<float> mapped(component_source_channels.size(), 0.f);
    for (size_t component_idx = 0; component_idx < component_source_channels.size(); ++component_idx) {
        const size_t source_channel = component_source_channels[component_idx];
        if (source_channel != sentinel && source_channel < raw_sample.size())
            mapped[component_idx] = raw_sample[source_channel];
    }
    return mapped;
}

std::vector<std::array<float, 3>> fixed_color_generic_solver_component_colors(int filament_color_mode)
{
    switch (std::clamp(filament_color_mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorRGB):
        return { { { 1.f, 0.f, 0.f } }, { { 0.f, 1.f, 0.f } }, { { 0.f, 0.f, 1.f } } };
    case int(TextureMappingZone::FilamentColorCMY):
        return { { { 0.f, 1.f, 1.f } }, { { 1.f, 0.f, 1.f } }, { { 1.f, 1.f, 0.f } } };
    case int(TextureMappingZone::FilamentColorCMYK):
        return { { { 0.f, 1.f, 1.f } }, { { 1.f, 0.f, 1.f } }, { { 1.f, 1.f, 0.f } }, { { 0.f, 0.f, 0.f } } };
    case int(TextureMappingZone::FilamentColorCMYW):
        return { { { 0.f, 1.f, 1.f } }, { { 1.f, 0.f, 1.f } }, { { 1.f, 1.f, 0.f } }, { { 1.f, 1.f, 1.f } } };
    case int(TextureMappingZone::FilamentColorRGBK):
        return { { { 1.f, 0.f, 0.f } }, { { 0.f, 1.f, 0.f } }, { { 0.f, 0.f, 1.f } }, { { 0.f, 0.f, 0.f } } };
    case int(TextureMappingZone::FilamentColorRGBW):
        return { { { 1.f, 0.f, 0.f } }, { { 0.f, 1.f, 0.f } }, { { 0.f, 0.f, 1.f } }, { { 1.f, 1.f, 1.f } } };
    case int(TextureMappingZone::FilamentColorCMYKW):
        return { { { 0.f, 1.f, 1.f } }, { { 1.f, 0.f, 1.f } }, { { 1.f, 1.f, 0.f } }, { { 0.f, 0.f, 0.f } }, { { 1.f, 1.f, 1.f } } };
    case int(TextureMappingZone::FilamentColorRGBKW):
        return { { { 1.f, 0.f, 0.f } }, { { 0.f, 1.f, 0.f } }, { { 0.f, 0.f, 1.f } }, { { 0.f, 0.f, 0.f } }, { { 1.f, 1.f, 1.f } } };
    default:
        return {};
    }
}

std::vector<size_t> best_matching_component_indices_for_semantic_colors(
    const std::vector<std::array<float, 3>> &component_colors,
    const std::vector<std::array<float, 3>> &semantic_colors)
{
    if (component_colors.size() != semantic_colors.size() || component_colors.empty())
        return {};

    std::vector<size_t> permutation(component_colors.size(), 0);
    std::iota(permutation.begin(), permutation.end(), size_t(0));

    std::vector<size_t> best_permutation = permutation;
    float best_error = std::numeric_limits<float>::max();
    do {
        float error = 0.f;
        for (size_t role_idx = 0; role_idx < semantic_colors.size(); ++role_idx)
            error += color_distance_sq(component_colors[permutation[role_idx]], semantic_colors[role_idx]);

        if (error < best_error) {
            best_error = error;
            best_permutation = permutation;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    return best_permutation;
}

std::vector<size_t> semantic_component_indices(const std::vector<std::array<float, 3>> &component_colors,
                                               int filament_color_mode,
                                               bool force_sequential_filaments)
{
    if (force_sequential_filaments)
        return {};

    std::vector<std::array<float, 3>> semantic_colors;
    switch (filament_color_mode) {
    case int(TextureMappingZone::FilamentColorRGB):
        semantic_colors = { { { 1.f, 0.f, 0.f } }, { { 0.f, 1.f, 0.f } }, { { 0.f, 0.f, 1.f } } };
        break;
    case int(TextureMappingZone::FilamentColorCMY):
        semantic_colors = { { { 0.f, 1.f, 1.f } }, { { 1.f, 0.f, 1.f } }, { { 1.f, 1.f, 0.f } } };
        break;
    case int(TextureMappingZone::FilamentColorCMYK):
        semantic_colors = { { { 0.f, 1.f, 1.f } }, { { 1.f, 0.f, 1.f } }, { { 1.f, 1.f, 0.f } }, { { 0.f, 0.f, 0.f } } };
        break;
    case int(TextureMappingZone::FilamentColorCMYW):
        semantic_colors = { { { 0.f, 1.f, 1.f } }, { { 1.f, 0.f, 1.f } }, { { 1.f, 1.f, 0.f } }, { { 1.f, 1.f, 1.f } } };
        break;
    case int(TextureMappingZone::FilamentColorRGBK):
        semantic_colors = { { { 1.f, 0.f, 0.f } }, { { 0.f, 1.f, 0.f } }, { { 0.f, 0.f, 1.f } }, { { 0.f, 0.f, 0.f } } };
        break;
    case int(TextureMappingZone::FilamentColorRGBW):
        semantic_colors = { { { 1.f, 0.f, 0.f } }, { { 0.f, 1.f, 0.f } }, { { 0.f, 0.f, 1.f } }, { { 1.f, 1.f, 1.f } } };
        break;
    case int(TextureMappingZone::FilamentColorCMYKW):
        semantic_colors = {
            { { 0.f, 1.f, 1.f } },
            { { 1.f, 0.f, 1.f } },
            { { 1.f, 1.f, 0.f } },
            { { 0.f, 0.f, 0.f } },
            { { 1.f, 1.f, 1.f } }
        };
        break;
    case int(TextureMappingZone::FilamentColorRGBKW):
        semantic_colors = {
            { { 1.f, 0.f, 0.f } },
            { { 0.f, 1.f, 0.f } },
            { { 0.f, 0.f, 1.f } },
            { { 0.f, 0.f, 0.f } },
            { { 1.f, 1.f, 1.f } }
        };
        break;
    default:
        return {};
    }

    return best_matching_component_indices_for_semantic_colors(component_colors, semantic_colors);
}

std::vector<float> optimized_component_weights(const std::array<float, 3> &target_rgb,
                                               size_t component_count,
                                               int filament_color_mode,
                                               const std::vector<std::array<float, 3>> &component_colors,
                                               bool force_sequential_filaments)
{
    const int mode = std::clamp(filament_color_mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorRGBKW));
    if (mode == int(TextureMappingZone::FilamentColorAny))
        return {};

    auto strength = [](float value) { return clamp01f(std::pow(std::max(0.f, value), 0.85f)); };
    auto safe_div = [](float numerator, float denominator) {
        if (denominator <= EPSILON)
            return 0.f;
        return clamp01f(numerator / denominator);
    };
    const std::vector<size_t> semantic_indices =
        semantic_component_indices(component_colors, mode, force_sequential_filaments);
    const auto component_index_for_role = [&semantic_indices](size_t role_idx) {
        if (role_idx < semantic_indices.size())
            return semantic_indices[role_idx];
        return role_idx;
    };

    const float r = clamp01f(target_rgb[0]);
    const float g = clamp01f(target_rgb[1]);
    const float b = clamp01f(target_rgb[2]);
    const float whiteness = std::min({ r, g, b });
    const float darkness = 1.f - std::max({ r, g, b });
    std::vector<float> weights(component_count, 0.f);

    if (mode == int(TextureMappingZone::FilamentColorRGB) && component_count == 3) {
        weights[component_index_for_role(0)] = strength(r);
        weights[component_index_for_role(1)] = strength(g);
        weights[component_index_for_role(2)] = strength(b);
        return weights;
    }
    if (mode == int(TextureMappingZone::FilamentColorCMY) && component_count == 3) {
        weights[component_index_for_role(0)] = strength(1.f - r);
        weights[component_index_for_role(1)] = strength(1.f - g);
        weights[component_index_for_role(2)] = strength(1.f - b);
        return weights;
    }
    if (mode == int(TextureMappingZone::FilamentColorBW) && component_count == 2) {
        const float gray = clamp01f(0.2126f * r + 0.7152f * g + 0.0722f * b);
        size_t black_component_idx = 0;
        size_t white_component_idx = 1;
        if (!force_sequential_filaments && component_colors.size() >= 2) {
            const float lum0 = 0.2126f * component_colors[0][0] + 0.7152f * component_colors[0][1] + 0.0722f * component_colors[0][2];
            const float lum1 = 0.2126f * component_colors[1][0] + 0.7152f * component_colors[1][1] + 0.0722f * component_colors[1][2];
            if (lum0 > lum1) {
                black_component_idx = 1;
                white_component_idx = 0;
            }
        }
        weights[black_component_idx] = strength(gray >= 0.5f ? 2.f * (1.f - gray) : 1.f);
        weights[white_component_idx] = strength(gray <= 0.5f ? 2.f * gray : 1.f);
        return weights;
    }
    if ((mode == int(TextureMappingZone::FilamentColorCMYK) ||
         mode == int(TextureMappingZone::FilamentColorRGBK) ||
         mode == int(TextureMappingZone::FilamentColorCMYW) ||
         mode == int(TextureMappingZone::FilamentColorRGBW)) && component_count == 4) {
        if (mode == int(TextureMappingZone::FilamentColorCMYK)) {
            const float k = clamp01f(darkness);
            const float inv = 1.f - k;
            weights[component_index_for_role(0)] = strength(safe_div(1.f - r - k, inv));
            weights[component_index_for_role(1)] = strength(safe_div(1.f - g - k, inv));
            weights[component_index_for_role(2)] = strength(safe_div(1.f - b - k, inv));
            weights[component_index_for_role(3)] = strength(k);
        } else if (mode == int(TextureMappingZone::FilamentColorRGBK)) {
            const float k = clamp01f(darkness);
            const float inv = 1.f - k;
            weights[component_index_for_role(0)] = strength(safe_div(r - k, inv));
            weights[component_index_for_role(1)] = strength(safe_div(g - k, inv));
            weights[component_index_for_role(2)] = strength(safe_div(b - k, inv));
            weights[component_index_for_role(3)] = strength(k);
        } else if (mode == int(TextureMappingZone::FilamentColorCMYW)) {
            const float inv = 1.f - whiteness;
            const float r_no_w = safe_div(r - whiteness, inv);
            const float g_no_w = safe_div(g - whiteness, inv);
            const float b_no_w = safe_div(b - whiteness, inv);
            weights[component_index_for_role(0)] = strength(clamp01f((1.f - r_no_w) * inv));
            weights[component_index_for_role(1)] = strength(clamp01f((1.f - g_no_w) * inv));
            weights[component_index_for_role(2)] = strength(clamp01f((1.f - b_no_w) * inv));
            weights[component_index_for_role(3)] = clamp01f(std::pow(whiteness, 1.35f));
        } else {
            const float inv = 1.f - whiteness;
            weights[component_index_for_role(0)] = strength(safe_div(r - whiteness, inv));
            weights[component_index_for_role(1)] = strength(safe_div(g - whiteness, inv));
            weights[component_index_for_role(2)] = strength(safe_div(b - whiteness, inv));
            weights[component_index_for_role(3)] = strength(whiteness);
        }
        return weights;
    }
    if ((mode == int(TextureMappingZone::FilamentColorCMYKW) ||
         mode == int(TextureMappingZone::FilamentColorRGBKW)) && component_count == 5) {
        const float chroma = std::max(0.f, 1.f - darkness - whiteness);
        if (mode == int(TextureMappingZone::FilamentColorCMYKW)) {
            weights[component_index_for_role(0)] = strength(chroma <= EPSILON ? 0.f : 1.f - safe_div(r - whiteness, chroma));
            weights[component_index_for_role(1)] = strength(chroma <= EPSILON ? 0.f : 1.f - safe_div(g - whiteness, chroma));
            weights[component_index_for_role(2)] = strength(chroma <= EPSILON ? 0.f : 1.f - safe_div(b - whiteness, chroma));
        } else {
            weights[component_index_for_role(0)] = strength(safe_div(r - whiteness, chroma));
            weights[component_index_for_role(1)] = strength(safe_div(g - whiteness, chroma));
            weights[component_index_for_role(2)] = strength(safe_div(b - whiteness, chroma));
        }
        weights[component_index_for_role(3)] = strength(darkness);
        weights[component_index_for_role(4)] = strength(whiteness);
        return weights;
    }
    return {};
}

float apply_texture_tone_gamma(float channel, float tone_gamma)
{
    const float safe_channel = clamp01f(channel);
    const float safe_gamma =
        (!std::isfinite(tone_gamma) || tone_gamma <= 0.f) ? 1.f : std::clamp(tone_gamma, 0.5f, 3.f);
    if (std::abs(safe_gamma - 1.f) <= 1e-5f)
        return safe_channel;
    return clamp01f(std::pow(safe_channel, 1.f / safe_gamma));
}

void apply_texture_contrast_to_components(std::vector<float> &component_weights,
                                          float contrast_factor,
                                          size_t mapped_component_count)
{
    const size_t count = std::min(mapped_component_count, component_weights.size());
    if (count == 0)
        return;

    float mean_weight = 0.f;
    for (size_t idx = 0; idx < count; ++idx)
        mean_weight += clamp01f(component_weights[idx]);
    mean_weight /= float(count);

    for (size_t idx = 0; idx < count; ++idx) {
        const float safe_weight = clamp01f(component_weights[idx]);
        component_weights[idx] = clamp01f(mean_weight + (safe_weight - mean_weight) * contrast_factor);
    }
}

std::array<float, 3> apply_texture_contrast_to_rgb(const std::array<float, 3> &rgb, float contrast_factor)
{
    const float clamped_contrast = std::clamp(contrast_factor, 0.25f, 3.f);
    if (std::abs(clamped_contrast - 1.f) <= 1e-5f)
        return { clamp01f(rgb[0]), clamp01f(rgb[1]), clamp01f(rgb[2]) };

    const float mean = (clamp01f(rgb[0]) + clamp01f(rgb[1]) + clamp01f(rgb[2])) / 3.f;
    return {
        clamp01f(mean + (clamp01f(rgb[0]) - mean) * clamped_contrast),
        clamp01f(mean + (clamp01f(rgb[1]) - mean) * clamped_contrast),
        clamp01f(mean + (clamp01f(rgb[2]) - mean) * clamped_contrast)
    };
}

std::array<float, 3> texture_sample_target_rgb(const WeightedTextureSample &sample, float tone_gamma, float contrast_factor)
{
    std::array<float, 3> target = {
        clamp01f(sample.rgba[0]),
        clamp01f(sample.rgba[1]),
        clamp01f(sample.rgba[2])
    };
    if (std::abs(tone_gamma - 1.f) > 1e-5f) {
        target[0] = apply_texture_tone_gamma(target[0], tone_gamma);
        target[1] = apply_texture_tone_gamma(target[1], tone_gamma);
        target[2] = apply_texture_tone_gamma(target[2], tone_gamma);
    }
    return apply_texture_contrast_to_rgb(target, contrast_factor);
}

std::array<float, 3> generic_solver_oklab_axis_weights(const std::array<float, 3> &target_oklab)
{
    const float chroma = std::hypot(target_oklab[1], target_oklab[2]);
    const float chroma_factor = std::clamp((chroma - 0.015f) / 0.13f, 0.f, 1.f);
    return {
        1.f + (0.25f - 1.f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor
    };
}

struct TextureMappingBinaryDitherCandidate {
    uint32_t             mask { 0 };
    std::array<float, 3> rgb { { 1.f, 1.f, 1.f } };
    std::array<float, 3> oklab { { 1.f, 0.f, 0.f } };
};

std::vector<TextureMappingBinaryDitherCandidate> binary_dither_candidates(
    const std::vector<std::array<float, 3>> &component_colors,
    const std::vector<float>                &component_strength_factors,
    const std::vector<float>                &component_minimum_offset_factors,
    int                                      generic_solver_mix_model)
{
    std::vector<TextureMappingBinaryDitherCandidate> candidates;
    const size_t component_count = component_colors.size();
    if (component_count == 0 || component_count > 16)
        return candidates;

    const uint32_t mask_end = uint32_t(1) << component_count;
    candidates.reserve(size_t(mask_end - 1));
    for (uint32_t mask = 1; mask < mask_end; ++mask) {
        std::vector<float> mix_weights(component_count, 0.f);
        float total_weight = 0.f;
        for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
            const float strength = component_idx < component_strength_factors.size() ?
                std::clamp(component_strength_factors[component_idx], 0.f, 1.f) :
                1.f;
            const float minimum = component_idx < component_minimum_offset_factors.size() ?
                std::clamp(component_minimum_offset_factors[component_idx], 0.f, 1.f) :
                0.f;
            const bool active = (mask & (uint32_t(1) << component_idx)) != 0;
            mix_weights[component_idx] = std::clamp(minimum + (active ? strength * (1.f - minimum) : 0.f), 0.f, 1.f);
            total_weight += mix_weights[component_idx];
        }
        if (total_weight <= EPSILON) {
            for (size_t component_idx = 0; component_idx < component_count; ++component_idx)
                if ((mask & (uint32_t(1) << component_idx)) != 0)
                    mix_weights[component_idx] = 1.f;
        }

        TextureMappingBinaryDitherCandidate candidate;
        candidate.mask = mask;
        candidate.rgb = mix_color_solver_components(component_colors,
                                                    mix_weights,
                                                    color_solver_mix_model_from_index(generic_solver_mix_model));
        candidate.oklab = color_solver_oklab_from_srgb(candidate.rgb);
        candidates.emplace_back(candidate);
    }
    return candidates;
}

struct TextureMappingBinaryDitherNearestResult {
    size_t best_idx { size_t(-1) };
    size_t second_idx { size_t(-1) };
    float  best_error { std::numeric_limits<float>::max() };
    float  second_error { std::numeric_limits<float>::max() };
};

TextureMappingBinaryDitherNearestResult nearest_binary_dither_candidates(
    const std::vector<TextureMappingBinaryDitherCandidate> &candidates,
    const std::array<float, 3>                             &target_oklab)
{
    TextureMappingBinaryDitherNearestResult result;
    const std::array<float, 3> axis_weights = generic_solver_oklab_axis_weights(target_oklab);
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const std::array<float, 3> &candidate = candidates[idx].oklab;
        const float dl = candidate[0] - target_oklab[0];
        const float da = candidate[1] - target_oklab[1];
        const float db = candidate[2] - target_oklab[2];
        const float error = axis_weights[0] * dl * dl + axis_weights[1] * da * da + axis_weights[2] * db * db;
        if (error < result.best_error) {
            result.second_error = result.best_error;
            result.second_idx = result.best_idx;
            result.best_error = error;
            result.best_idx = idx;
        } else if (error < result.second_error) {
            result.second_error = error;
            result.second_idx = idx;
        }
    }
    return result;
}

size_t nearest_binary_dither_candidate(const std::vector<TextureMappingBinaryDitherCandidate> &candidates,
                                       const std::array<float, 3>                             &target_oklab)
{
    return nearest_binary_dither_candidates(candidates, target_oklab).best_idx;
}

float binary_dither_alternate_fraction(const std::vector<TextureMappingBinaryDitherCandidate> &candidates,
                                       const std::array<float, 3>                             &target_oklab,
                                       size_t                                                  base_idx,
                                       size_t                                                  alternate_idx)
{
    if (base_idx >= candidates.size() || alternate_idx >= candidates.size() || base_idx == alternate_idx)
        return 0.f;

    const std::array<float, 3> axis_weights = generic_solver_oklab_axis_weights(target_oklab);
    const std::array<float, 3> &base = candidates[base_idx].oklab;
    const std::array<float, 3> &alternate = candidates[alternate_idx].oklab;
    float numerator = 0.f;
    float denominator = 0.f;
    for (size_t axis = 0; axis < 3; ++axis) {
        const float delta = alternate[axis] - base[axis];
        numerator += axis_weights[axis] * (target_oklab[axis] - base[axis]) * delta;
        denominator += axis_weights[axis] * delta * delta;
    }
    if (!std::isfinite(numerator) || !std::isfinite(denominator) || denominator <= 1e-12f)
        return 0.f;
    return std::clamp(numerator / denominator, 0.f, 1.f);
}

size_t thresholded_binary_dither_candidate(const std::vector<TextureMappingBinaryDitherCandidate> &candidates,
                                           const std::array<float, 3>                             &target_oklab,
                                           float                                                   threshold)
{
    const TextureMappingBinaryDitherNearestResult nearest = nearest_binary_dither_candidates(candidates, target_oklab);
    if (nearest.best_idx >= candidates.size())
        return size_t(-1);
    if (nearest.second_idx >= candidates.size())
        return nearest.best_idx;

    const float alternate_fraction =
        binary_dither_alternate_fraction(candidates, target_oklab, nearest.best_idx, nearest.second_idx);
    return std::clamp(threshold, 0.f, 1.f) < alternate_fraction ? nearest.second_idx : nearest.best_idx;
}

float ordered_bayer_threshold(int x, int y)
{
    static constexpr int matrix[8][8] = {
        { 0, 48, 12, 60, 3, 51, 15, 63 },
        { 32, 16, 44, 28, 35, 19, 47, 31 },
        { 8, 56, 4, 52, 11, 59, 7, 55 },
        { 40, 24, 36, 20, 43, 27, 39, 23 },
        { 2, 50, 14, 62, 1, 49, 13, 61 },
        { 34, 18, 46, 30, 33, 17, 45, 29 },
        { 10, 58, 6, 54, 9, 57, 5, 53 },
        { 42, 26, 38, 22, 41, 25, 37, 21 }
    };
    const int bx = ((x % 8) + 8) % 8;
    const int by = ((y % 8) + 8) % 8;
    return (float(matrix[by][bx]) + 0.5f) / 64.f - 0.5f;
}

bool is_halftone_dithering_method(int method)
{
    const int clamped_method = std::clamp(method,
                                          int(TextureMappingZone::DitheringClosest),
                                          int(TextureMappingZone::DitheringHalftoneV2));
    return clamped_method == int(TextureMappingZone::DitheringHalftone) ||
           clamped_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail) ||
           clamped_method == int(TextureMappingZone::DitheringHalftoneV2);
}

float halftone_threshold(float x_mm, float y_mm, float dot_size_mm)
{
    const float period = std::clamp(dot_size_mm,
                                    TextureMappingZone::MinHalftoneDotSizeMm,
                                    TextureMappingZone::MaxHalftoneDotSizeMm);
    const float u = x_mm / period - std::floor(x_mm / period);
    const float v = y_mm / period - std::floor(y_mm / period);
    const float dx = u - 0.5f;
    const float dy = v - 0.5f;
    const float radius_area = (dx * dx + dy * dy) / 0.5f;
    return std::clamp(radius_area, 0.f, 1.f) - 0.5f;
}

float halftone_screen_angle_deg(int filament_color_mode, size_t component_idx)
{
    const int clamped_mode =
        std::clamp(filament_color_mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorRGBKW));
    if (component_idx < 3) {
        static constexpr float primary_angles[3] = { 15.f, 75.f, 0.f };
        return primary_angles[component_idx];
    }

    switch (clamped_mode) {
    case int(TextureMappingZone::FilamentColorCMYK):
    case int(TextureMappingZone::FilamentColorRGBK):
    case int(TextureMappingZone::FilamentColorCMYKW):
    case int(TextureMappingZone::FilamentColorRGBKW):
        if (component_idx == 3)
            return 45.f;
        if (component_idx == 4)
            return 90.f;
        break;
    case int(TextureMappingZone::FilamentColorCMYW):
    case int(TextureMappingZone::FilamentColorRGBW):
        if (component_idx == 3)
            return 90.f;
        break;
    case int(TextureMappingZone::FilamentColorBW):
        if (component_idx == 0)
            return 45.f;
        if (component_idx == 1)
            return 90.f;
        break;
    default:
        break;
    }

    static constexpr float fallback_angles[] = { 15.f, 75.f, 0.f, 45.f, 90.f, 30.f, 60.f, 105.f, 120.f, 150.f };
    return fallback_angles[component_idx % (sizeof(fallback_angles) / sizeof(fallback_angles[0]))];
}

int halftone_surface_axis(float normal_x, float normal_y, float normal_z)
{
    const float ax = std::abs(normal_x);
    const float ay = std::abs(normal_y);
    const float az = std::abs(normal_z);
    if (ax >= ay && ax >= az)
        return 0;
    if (ay >= ax && ay >= az)
        return 1;
    return 2;
}

std::array<float, 2> halftone_surface_coords(float x_mm, float y_mm, float z_mm, int axis)
{
    if (axis == 0)
        return { y_mm, z_mm };
    if (axis == 1)
        return { x_mm, z_mm };
    return { x_mm, y_mm };
}

std::array<float, 2> halftone_cell_center_surface_coords(float surface_u_mm,
                                                         float surface_v_mm,
                                                         float dot_size_mm,
                                                         float angle_deg)
{
    const float period = std::clamp(dot_size_mm,
                                    TextureMappingZone::MinHalftoneDotSizeMm,
                                    TextureMappingZone::MaxHalftoneDotSizeMm);
    if (!std::isfinite(surface_u_mm) || !std::isfinite(surface_v_mm) || period <= EPSILON)
        return { surface_u_mm, surface_v_mm };

    const float radians = angle_deg * float(PI) / 180.f;
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    const float screen_x = c * surface_u_mm + s * surface_v_mm;
    const float screen_y = -s * surface_u_mm + c * surface_v_mm;
    const int cell_x = int(std::floor(screen_x / period));
    const int cell_y = int(std::floor(screen_y / period));
    const float center_screen_x = (float(cell_x) + 0.5f) * period;
    const float center_screen_y = (float(cell_y) + 0.5f) * period;
    return {
        c * center_screen_x - s * center_screen_y,
        s * center_screen_x + c * center_screen_y
    };
}

std::array<float, 2> halftone_sample_xy_from_surface_coords(float x_mm,
                                                            float y_mm,
                                                            const std::array<float, 2> &surface_coords,
                                                            int axis)
{
    if (axis == 0)
        return { x_mm, surface_coords[0] };
    if (axis == 1)
        return { surface_coords[0], y_mm };
    return surface_coords;
}

bool halftone_screen_on(float coverage,
                        float surface_u_mm,
                        float surface_v_mm,
                        float dot_size_mm,
                        float angle_deg)
{
    const float safe_coverage = clamp01f(coverage);
    if (safe_coverage <= EPSILON)
        return false;
    if (safe_coverage >= 1.f - EPSILON)
        return true;
    if (!std::isfinite(surface_u_mm) || !std::isfinite(surface_v_mm))
        return safe_coverage >= 0.5f;

    const float radians = angle_deg * float(PI) / 180.f;
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    const float x = c * surface_u_mm + s * surface_v_mm;
    const float y = -s * surface_u_mm + c * surface_v_mm;
    return halftone_threshold(x, y, dot_size_mm) + 0.5f < safe_coverage;
}

float dither_pitch(float base_outer_width_mm,
                   int dithering_method,
                   float dithering_resolution_mm,
                   float halftone_dot_size_mm)
{
    const float high_res_step_mm = std::clamp(base_outer_width_mm * 0.20f, 0.04f, 0.12f);
    const int clamped_method = std::clamp(dithering_method,
                                          int(TextureMappingZone::DitheringClosest),
                                          int(TextureMappingZone::DitheringHalftoneV2));
    if (is_halftone_dithering_method(clamped_method)) {
        const float dot_sample_step_mm =
            std::clamp(std::clamp(halftone_dot_size_mm,
                                  TextureMappingZone::MinHalftoneDotSizeMm,
                                  TextureMappingZone::MaxHalftoneDotSizeMm) *
                           0.25f,
                       0.04f,
                       0.08f);
        return std::min(high_res_step_mm, dot_sample_step_mm);
    }
    return std::min(high_res_step_mm,
                    std::clamp(dithering_resolution_mm,
                               TextureMappingZone::MinDitheringResolutionMm,
                               TextureMappingZone::MaxDitheringResolutionMm));
}

float dither_cell_size(float dithering_resolution_mm)
{
    return std::clamp(dithering_resolution_mm,
                      TextureMappingZone::MinDitheringResolutionMm,
                      TextureMappingZone::MaxDitheringResolutionMm);
}

std::array<Vec2f, 3> unwrap_triangle_uvs(const Vec2f &uv0, const Vec2f &uv1, const Vec2f &uv2)
{
    std::array<Vec2f, 3> out { uv0, uv1, uv2 };
    auto unwrap_axis = [&out](bool use_u_axis) {
        std::array<float, 3> values = {
            use_u_axis ? out[0].x() : out[0].y(),
            use_u_axis ? out[1].x() : out[1].y(),
            use_u_axis ? out[2].x() : out[2].y()
        };
        if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
            return;
        auto span = [](const std::array<float, 3> &v) {
            return std::max({ v[0], v[1], v[2] }) - std::min({ v[0], v[1], v[2] });
        };
        const bool has_repeat_evidence = std::any_of(values.begin(), values.end(), [](float value) {
            constexpr float eps = 1e-6f;
            return value < -eps || value > 1.f + eps;
        });
        const float original_span = span(values);
        if (!has_repeat_evidence || original_span <= 0.5f)
            return;
        std::array<float, 3> best = values;
        float best_span = original_span;
        for (size_t anchor = 0; anchor < values.size(); ++anchor) {
            std::array<float, 3> candidate = values;
            for (size_t i = 0; i < candidate.size(); ++i) {
                const float delta = values[i] - values[anchor];
                candidate[i] = values[anchor] + delta - std::round(delta);
            }
            const float candidate_span = span(candidate);
            if (candidate_span + 1e-6f < best_span) {
                best = candidate;
                best_span = candidate_span;
            }
        }
        if (best_span >= original_span - 1e-6f)
            return;
        if (use_u_axis) {
            out[0].x() = best[0];
            out[1].x() = best[1];
            out[2].x() = best[2];
        } else {
            out[0].y() = best[0];
            out[1].y() = best[1];
            out[2].y() = best[2];
        }
    };
    unwrap_axis(true);
    unwrap_axis(false);
    return out;
}

template<class SampleDataForBarycentricFn, class AccumulateSampleFn>
bool accumulate_layer_plane_triangle_samples(const Vec3d &p0,
                                             const Vec3d &p1,
                                             const Vec3d &p2,
                                             float layer_z_mm,
                                             float layer_z_falloff_mm,
                                             bool high_resolution_texture_sampling,
                                             float physical_sample_pitch_mm,
                                             const SampleDataForBarycentricFn &sample_data_for_barycentric,
                                             const AccumulateSampleFn &accumulate_sample)
{
    const float z0 = float(p0.z());
    const float z1 = float(p1.z());
    const float z2 = float(p2.z());
    if (!std::isfinite(z0) || !std::isfinite(z1) || !std::isfinite(z2))
        return false;

    const float min_z = std::min({ z0, z1, z2 });
    const float max_z = std::max({ z0, z1, z2 });
    const float z_eps = std::max(1e-5f, layer_z_falloff_mm * 1e-4f);
    if (layer_z_mm < min_z - z_eps || layer_z_mm > max_z + z_eps || max_z - min_z <= z_eps)
        return false;

    const std::array<Vec3d, 3> vertices = { p0, p1, p2 };
    const std::array<Vec3f, 3> barycentrics = {
        Vec3f(1.f, 0.f, 0.f),
        Vec3f(0.f, 1.f, 0.f),
        Vec3f(0.f, 0.f, 1.f)
    };
    const std::array<float, 3> zs = { z0, z1, z2 };
    std::vector<LayerPlaneSamplePoint> layer_points;
    layer_points.reserve(3);

    auto add_layer_point = [&layer_points](const Vec3d &p, const Vec3f &barycentric) {
        if (!p.allFinite() || !barycentric.allFinite())
            return;
        for (const LayerPlaneSamplePoint &existing : layer_points)
            if ((existing.p - p).squaredNorm() <= 1e-10)
                return;
        layer_points.push_back({ p, barycentric });
    };

    const std::array<std::pair<size_t, size_t>, 3> edges = {
        std::make_pair(size_t(0), size_t(1)),
        std::make_pair(size_t(1), size_t(2)),
        std::make_pair(size_t(2), size_t(0))
    };
    for (const auto &edge : edges) {
        const size_t a = edge.first;
        const size_t b = edge.second;
        const float da = zs[a] - layer_z_mm;
        const float db = zs[b] - layer_z_mm;
        const bool a_on_layer = std::abs(da) <= z_eps;
        const bool b_on_layer = std::abs(db) <= z_eps;
        if (a_on_layer)
            add_layer_point(vertices[a], barycentrics[a]);
        if (b_on_layer)
            add_layer_point(vertices[b], barycentrics[b]);
        if (a_on_layer || b_on_layer)
            continue;
        if ((da < 0.f && db > 0.f) || (da > 0.f && db < 0.f)) {
            const float t = (layer_z_mm - zs[a]) / (zs[b] - zs[a]);
            if (!std::isfinite(t) || t < -1e-4f || t > 1.f + 1e-4f)
                continue;
            const float clamped_t = std::clamp(t, 0.f, 1.f);
            add_layer_point(vertices[a] * double(1.f - clamped_t) + vertices[b] * double(clamped_t),
                            barycentrics[a] * (1.f - clamped_t) + barycentrics[b] * clamped_t);
        }
    }

    if (layer_points.size() < 2)
        return false;

    size_t best_a = 0;
    size_t best_b = 1;
    double best_length_sq = 0.0;
    for (size_t i = 0; i + 1 < layer_points.size(); ++i) {
        for (size_t j = i + 1; j < layer_points.size(); ++j) {
            const double length_sq = (layer_points[i].p - layer_points[j].p).squaredNorm();
            if (length_sq > best_length_sq) {
                best_a = i;
                best_b = j;
                best_length_sq = length_sq;
            }
        }
    }

    const double segment_length_mm = std::sqrt(best_length_sq);
    if (!std::isfinite(segment_length_mm) || segment_length_mm <= EPSILON)
        return false;

    const float sample_pitch_mm = std::max(float(EPSILON), physical_sample_pitch_mm);
    const int sample_count = std::clamp(int(std::ceil(segment_length_mm / std::max(float(EPSILON), sample_pitch_mm))), 1, 2000);
    const float sample_weight = std::max(0.05f, float(segment_length_mm) / float(sample_count));
    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        const float t = (float(sample_idx) + 0.5f) / float(sample_count);
        Vec3f barycentric = layer_points[best_a].barycentric * (1.f - t) + layer_points[best_b].barycentric * t;
        barycentric.x() = std::max(0.f, barycentric.x());
        barycentric.y() = std::max(0.f, barycentric.y());
        barycentric.z() = std::max(0.f, barycentric.z());
        const float barycentric_sum = barycentric.x() + barycentric.y() + barycentric.z();
        if (!std::isfinite(barycentric_sum) || barycentric_sum <= EPSILON)
            continue;
        barycentric /= barycentric_sum;

        const Vec3d world_pos = p0 * double(barycentric.x()) + p1 * double(barycentric.y()) + p2 * double(barycentric.z());
        TextureSampleData sample_data = sample_data_for_barycentric(barycentric);
        accumulate_sample(float(world_pos.x()),
                          float(world_pos.y()),
                          sample_data.rgba,
                          sample_weight,
                          std::move(sample_data.raw_component_weights),
                          sample_data.raw_component_weights_from_texture,
                          sample_data.normal_z);
    }

    return true;
}

std::vector<float> component_weights_for_sample(const WeightedTextureSample &sample,
                                                size_t component_count,
                                                bool raw_values_mode,
                                                int filament_color_mode,
                                                bool force_sequential_filaments,
                                                int generic_solver_lookup_mode,
                                                int generic_solver_mode,
                                                bool use_fixed_color_generic_solver,
                                                float contrast_factor,
                                                float tone_gamma,
                                                const std::vector<std::array<float, 3>> &component_colors,
                                                const ColorSolverCandidateSet *generic_mix_candidates)
{
    std::vector<float> desired(component_count, 0.f);
    size_t mapped_component_count = component_count;
    const bool has_raw_component_weights = sample.raw_component_weights.size() == component_count;
    if (has_raw_component_weights) {
        float raw_activity = 0.f;
        for (size_t component_idx = 0; component_idx < component_count; ++component_idx)
            desired[component_idx] = clamp01f(sample.raw_component_weights[component_idx]);
        for (const float value : desired)
            raw_activity = std::max(raw_activity, value);
        if (raw_activity <= EPSILON && !sample.raw_component_weights_from_texture)
            std::fill(desired.begin(), desired.end(), 1.f);
    } else {
        std::array<float, 3> target = {
            clamp01f(sample.rgba[0]),
            clamp01f(sample.rgba[1]),
            clamp01f(sample.rgba[2])
        };
        if (std::abs(tone_gamma - 1.f) > 1e-5f) {
            target[0] = apply_texture_tone_gamma(target[0], tone_gamma);
            target[1] = apply_texture_tone_gamma(target[1], tone_gamma);
            target[2] = apply_texture_tone_gamma(target[2], tone_gamma);
        }

        if (raw_values_mode) {
            const float channels[3] = { target[0], target[1], target[2] };
            const size_t channel_count = std::min(component_count, size_t(3));
            for (size_t channel_idx = 0; channel_idx < channel_count; ++channel_idx)
                desired[channel_idx] = clamp01f(channels[channel_idx]);
            mapped_component_count = channel_count;
        } else {
            std::vector<float> optimized;
            if (!use_fixed_color_generic_solver)
                optimized = optimized_component_weights(target,
                                                        component_count,
                                                        filament_color_mode,
                                                        component_colors,
                                                        force_sequential_filaments);
            if (optimized.size() == component_count) {
                desired = std::move(optimized);
            } else if (generic_mix_candidates != nullptr) {
                std::vector<float> best =
                    solve_color_solver_weights_for_target(*generic_mix_candidates,
                                                          target,
                                                          color_solver_lookup_mode_from_index(generic_solver_lookup_mode),
                                                          color_solver_mode_from_index(generic_solver_mode));
                if (best.size() == component_count)
                    desired = std::move(best);
            }
        }
    }

    if (!has_raw_component_weights && std::abs(contrast_factor - 1.f) > 1e-5f)
        apply_texture_contrast_to_components(desired, contrast_factor, mapped_component_count);
    for (float &v : desired)
        v = clamp01f(v);
    return desired;
}

TextureMappingOffsetWeightField build_texture_mapping_offset_weight_field(
    const PrintObject &print_object,
    const std::vector<std::array<float, 3>> &component_colors,
    bool raw_values_mode,
    int filament_color_mode,
    bool force_sequential_filaments,
    int generic_solver_lookup_mode,
    int generic_solver_mode,
    int generic_solver_mix_model,
    bool use_legacy_fixed_color_mode,
    bool dithering_enabled,
    int dithering_method,
    float dither_pitch_mm,
    float dither_cell_size_mm,
    const std::vector<float> &component_strength_factors,
    const std::vector<float> &component_minimum_offset_factors,
    float texture_contrast_pct,
    float texture_tone_gamma,
    float layer_z_mm,
    float layer_z_falloff_mm,
    bool high_resolution_texture_sampling,
    bool high_speed_image_texture_sampling,
    std::optional<std::array<float, 4>> image_background_rgba_override)
{
    TextureMappingOffsetWeightField weight_field;
    if (component_colors.empty())
        return weight_field;

    const size_t component_count = component_colors.size();
    const ModelObject *model_object = print_object.model_object();
    if (model_object == nullptr)
        return weight_field;

    const BoundingBox object_bbox = print_object.bounding_box();
    const float min_x_mm = unscale<float>(object_bbox.min.x());
    const float min_y_mm = unscale<float>(object_bbox.min.y());
    const float max_x_mm = unscale<float>(object_bbox.max.x());
    const float max_y_mm = unscale<float>(object_bbox.max.y());
    const float span_x_mm = std::max(max_x_mm - min_x_mm, 1e-3f);
    const float span_y_mm = std::max(max_y_mm - min_y_mm, 1e-3f);
    if (!std::isfinite(min_x_mm) || !std::isfinite(min_y_mm) ||
        !std::isfinite(max_x_mm) || !std::isfinite(max_y_mm) ||
        !std::isfinite(span_x_mm) || !std::isfinite(span_y_mm))
        return {};

    const float safe_layer_z_falloff_mm = std::max(layer_z_falloff_mm, 1e-3f);
    const float contrast_factor = std::clamp(texture_contrast_pct, 25.f, 300.f) / 100.f;
    const float tone_gamma =
        (!std::isfinite(texture_tone_gamma) || texture_tone_gamma <= 0.f) ? 1.f : std::clamp(texture_tone_gamma, 0.5f, 3.f);
    const float physical_sample_pitch_mm =
        dithering_enabled && !raw_values_mode && std::isfinite(dither_pitch_mm) && dither_pitch_mm > EPSILON ?
            dither_pitch_mm :
            (high_resolution_texture_sampling ? 0.08f : 0.16f);

    std::vector<WeightedTextureSample> samples;
    samples.reserve(8192);
    auto accumulate_sample = [&samples, component_count](float x_mm,
                                                         float y_mm,
                                                         const std::array<float, 4> &rgba,
                                                         float sample_weight,
                                                         std::vector<float> raw_component_weights = {},
                                                         bool raw_component_weights_from_texture = false,
                                                         float normal_z = std::numeric_limits<float>::quiet_NaN()) {
        if (!std::isfinite(x_mm) || !std::isfinite(y_mm) || sample_weight <= EPSILON)
            return;
        if (!std::isfinite(sample_weight) ||
            !std::isfinite(rgba[0]) ||
            !std::isfinite(rgba[1]) ||
            !std::isfinite(rgba[2]) ||
            !std::isfinite(rgba[3]))
            return;
        if (raw_component_weights.size() != component_count)
            raw_component_weights_from_texture = false;
        samples.push_back({ x_mm, y_mm, rgba, std::move(raw_component_weights), raw_component_weights_from_texture, normal_z, sample_weight });
    };

    auto accumulate_constant_surface_triangle_samples = [&](const Vec3d &p0,
                                                            const Vec3d &p1,
                                                            const Vec3d &p2,
                                                            const std::array<float, 4> &rgba) {
        const Vec3d normal = (p1 - p0).cross(p2 - p0).normalized();
        const float normal_z = normal.allFinite() ? float(normal.z()) : std::numeric_limits<float>::quiet_NaN();
        if (accumulate_layer_plane_triangle_samples(p0, p1, p2, layer_z_mm, safe_layer_z_falloff_mm, high_resolution_texture_sampling,
                physical_sample_pitch_mm,
                [&rgba, normal_z](const Vec3f &) { return TextureSampleData{ rgba, {}, false, normal_z }; }, accumulate_sample))
            return;

        const float max_world_edge_mm = std::max({ float((p1 - p0).norm()), float((p2 - p1).norm()), float((p0 - p2).norm()) });
        if (!std::isfinite(max_world_edge_mm))
            return;
        const double tri_area_mm2 = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
        if (!std::isfinite(tri_area_mm2))
            return;

        const float pitch_mm = physical_sample_pitch_mm;
        const int max_bary_steps = dithering_enabled && !raw_values_mode ? 160 : (high_resolution_texture_sampling ? 80 : 40);
        const int bary_steps = std::clamp(int(std::ceil(max_world_edge_mm / pitch_mm)), 1, max_bary_steps);
        const int sample_count = bary_steps * (bary_steps + 1) / 2;
        if (sample_count <= 0)
            return;
        const float area_weight = std::max(0.05f, float(tri_area_mm2)) / float(sample_count);
        const float inv_steps = 1.f / float(bary_steps);
        for (int i = 0; i < bary_steps; ++i) {
            for (int j = 0; j < (bary_steps - i); ++j) {
                const float b1 = (float(i) + 0.33333334f) * inv_steps;
                const float b2 = (float(j) + 0.33333334f) * inv_steps;
                const float b0 = 1.f - b1 - b2;
                if (b0 < 0.f)
                    continue;
                const Vec3d world_pos = p0 * double(b0) + p1 * double(b1) + p2 * double(b2);
                const float dz = std::abs(float(world_pos.z()) - layer_z_mm);
                const float z_norm = dz / safe_layer_z_falloff_mm;
                const float z_weight = std::exp(-0.5f * z_norm * z_norm);
                if (!std::isfinite(z_weight) || z_weight <= EPSILON)
                    continue;
                accumulate_sample(float(world_pos.x()), float(world_pos.y()), rgba, area_weight * z_weight, {}, false, normal_z);
            }
        }
    };

    const Transform3d object_trafo = print_object.trafo_centered();
    for (const ModelVolume *volume : model_object->volumes) {
        if (volume == nullptr)
            continue;

        const std::shared_ptr<const TriangleMesh> mesh_ptr = volume->mesh_ptr();
        if (!mesh_ptr)
            continue;

        const indexed_triangle_set &its = mesh_ptr->its;
        const Transform3d volume_trafo = object_trafo * volume->get_matrix();
        const std::array<float, 4> background_color =
            image_background_rgba_override ? *image_background_rgba_override : texture_mapping_background_color(*volume);

        if (!volume->texture_mapping_color_facets.empty()) {
            std::vector<ColorFacetTriangle> color_facets;
            volume->texture_mapping_color_facets.get_facet_triangles(*volume, color_facets);
            std::vector<uint8_t> rgba_source_triangles(its.indices.size(), 0);
            for (const ColorFacetTriangle &facet : color_facets) {
                if (facet.source_triangle >= 0 && size_t(facet.source_triangle) < rgba_source_triangles.size())
                    rgba_source_triangles[size_t(facet.source_triangle)] = 1;

                const Vec3d p0 = volume_trafo * facet.vertices[0].cast<double>();
                const Vec3d p1 = volume_trafo * facet.vertices[1].cast<double>();
                const Vec3d p2 = volume_trafo * facet.vertices[2].cast<double>();
                if (!p0.allFinite() || !p1.allFinite() || !p2.allFinite())
                    continue;
                const std::array<float, 4> rgba = composite_rgba_over_background(unpack_rgba_u32(facet.rgba), background_color);
                accumulate_constant_surface_triangle_samples(p0, p1, p2, rgba);
            }
            for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
                if (tri_idx < rgba_source_triangles.size() && rgba_source_triangles[tri_idx] != 0)
                    continue;

                const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
                if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                    continue;
                if (size_t(tri[0]) >= its.vertices.size() ||
                    size_t(tri[1]) >= its.vertices.size() ||
                    size_t(tri[2]) >= its.vertices.size())
                    continue;

                const Vec3d p0 = volume_trafo * its.vertices[size_t(tri[0])].cast<double>();
                const Vec3d p1 = volume_trafo * its.vertices[size_t(tri[1])].cast<double>();
                const Vec3d p2 = volume_trafo * its.vertices[size_t(tri[2])].cast<double>();
                if (!p0.allFinite() || !p1.allFinite() || !p2.allFinite())
                    continue;

                accumulate_constant_surface_triangle_samples(p0, p1, p2, background_color);
            }
            continue;
        }

        const bool has_uv_texture =
            !volume->imported_texture_rgba.empty() &&
            volume->imported_texture_width > 0 &&
            volume->imported_texture_height > 0 &&
            volume->imported_texture_uv_valid.size() == its.indices.size() &&
            volume->imported_texture_uvs_per_face.size() >= its.indices.size() * 6 &&
            volume->imported_texture_rgba.size() >= size_t(volume->imported_texture_width) * size_t(volume->imported_texture_height) * 4;
        if (has_uv_texture) {
            const std::vector<size_t> raw_component_channels =
                raw_component_source_channels(volume->imported_texture_raw_metadata_json,
                                              volume->imported_texture_raw_channels,
                                              filament_color_mode,
                                              component_count,
                                              component_colors);
            const bool use_raw_uv_texture =
                raw_component_channels.size() == component_count &&
                volume->imported_texture_raw_filament_offsets.size() >=
                    size_t(volume->imported_texture_width) *
                        size_t(volume->imported_texture_height) *
                        size_t(volume->imported_texture_raw_channels);
            auto sample_data_for_uv = [&](const Vec2f &uv) {
                std::array<float, 4> rgba =
                    sample_texture_rgba_bilinear(volume->imported_texture_rgba,
                                                 volume->imported_texture_width,
                                                 volume->imported_texture_height,
                                                 uv.x(),
                                                 uv.y());
                std::vector<float> raw_component_weights;
                if (use_raw_uv_texture) {
                    const std::vector<float> raw_sample =
                        sample_texture_raw_offsets_bilinear(volume->imported_texture_raw_filament_offsets,
                                                            volume->imported_texture_width,
                                                            volume->imported_texture_height,
                                                            volume->imported_texture_raw_channels,
                                                            uv.x(),
                                                            uv.y());
                    raw_component_weights = map_raw_sample_to_components(raw_sample, raw_component_channels);
                    if (raw_component_weights.size() == component_count)
                        rgba = raw_offset_preview_rgba(raw_component_weights);
                }
                if (raw_component_weights.size() != component_count)
                    rgba = composite_rgba_over_background(rgba, background_color);
                return TextureSampleData{ rgba, std::move(raw_component_weights), use_raw_uv_texture };
            };

            for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
                if (volume->imported_texture_uv_valid[tri_idx] == 0)
                    continue;
                const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
                if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                    continue;
                if (size_t(tri[0]) >= its.vertices.size() ||
                    size_t(tri[1]) >= its.vertices.size() ||
                    size_t(tri[2]) >= its.vertices.size())
                    continue;
                const Vec3d p0 = volume_trafo * its.vertices[size_t(tri[0])].cast<double>();
                const Vec3d p1 = volume_trafo * its.vertices[size_t(tri[1])].cast<double>();
                const Vec3d p2 = volume_trafo * its.vertices[size_t(tri[2])].cast<double>();
                if (!p0.allFinite() || !p1.allFinite() || !p2.allFinite())
                    continue;
                const Vec3d normal = (p1 - p0).cross(p2 - p0).normalized();
                const float normal_z = normal.allFinite() ? float(normal.z()) : std::numeric_limits<float>::quiet_NaN();

                const size_t uv_off = tri_idx * 6;
                const std::array<Vec2f, 3> uvs = unwrap_triangle_uvs(
                    Vec2f(volume->imported_texture_uvs_per_face[uv_off + 0], volume->imported_texture_uvs_per_face[uv_off + 1]),
                    Vec2f(volume->imported_texture_uvs_per_face[uv_off + 2], volume->imported_texture_uvs_per_face[uv_off + 3]),
                    Vec2f(volume->imported_texture_uvs_per_face[uv_off + 4], volume->imported_texture_uvs_per_face[uv_off + 5]));

                if (accumulate_layer_plane_triangle_samples(p0, p1, p2, layer_z_mm, safe_layer_z_falloff_mm, high_resolution_texture_sampling,
                        physical_sample_pitch_mm,
                        [&uvs, &sample_data_for_uv, normal_z](const Vec3f &barycentric) {
                            const Vec2f uv = uvs[0] * barycentric.x() + uvs[1] * barycentric.y() + uvs[2] * barycentric.z();
                            TextureSampleData sample_data = sample_data_for_uv(uv);
                            sample_data.normal_z = normal_z;
                            return sample_data;
                        }, accumulate_sample))
                    continue;

                if (high_speed_image_texture_sampling) {
                    const float min_z = std::min({ float(p0.z()), float(p1.z()), float(p2.z()) });
                    const float max_z = std::max({ float(p0.z()), float(p1.z()), float(p2.z()) });
                    const float z_margin = std::max(1e-4f, safe_layer_z_falloff_mm * 8.f + float(EPSILON));
                    if (layer_z_mm < min_z - z_margin || layer_z_mm > max_z + z_margin)
                        continue;
                }

                const float max_world_edge_mm = std::max({ float((p1 - p0).norm()), float((p2 - p1).norm()), float((p0 - p2).norm()) });
                const double tri_area_mm2 = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
                if (!std::isfinite(max_world_edge_mm) || !std::isfinite(tri_area_mm2))
                    continue;
                const float pitch_mm = physical_sample_pitch_mm;
                const int max_bary_steps = dithering_enabled && !raw_values_mode ? 160 : (high_resolution_texture_sampling ? 80 : 40);
                const int bary_steps = std::clamp(int(std::ceil(max_world_edge_mm / pitch_mm)), 1, max_bary_steps);
                const int sample_count = bary_steps * (bary_steps + 1) / 2;
                if (sample_count <= 0)
                    continue;
                const float area_weight = std::max(0.05f, float(tri_area_mm2)) / float(sample_count);
                const float inv_steps = 1.f / float(bary_steps);
                for (int i = 0; i < bary_steps; ++i) {
                    for (int j = 0; j < (bary_steps - i); ++j) {
                        const float b1 = (float(i) + 0.33333334f) * inv_steps;
                        const float b2 = (float(j) + 0.33333334f) * inv_steps;
                        const float b0 = 1.f - b1 - b2;
                        if (b0 < 0.f)
                            continue;
                        const Vec3d world_pos = p0 * double(b0) + p1 * double(b1) + p2 * double(b2);
                        const Vec2f uv = uvs[0] * b0 + uvs[1] * b1 + uvs[2] * b2;
                        TextureSampleData sample_data = sample_data_for_uv(uv);
                        const float dz = std::abs(float(world_pos.z()) - layer_z_mm);
                        const float z_norm = dz / safe_layer_z_falloff_mm;
                        const float z_weight = std::exp(-0.5f * z_norm * z_norm);
                        if (!std::isfinite(z_weight) || z_weight <= EPSILON)
                            continue;
                        accumulate_sample(float(world_pos.x()),
                                          float(world_pos.y()),
                                          sample_data.rgba,
                                          area_weight * z_weight,
                                          std::move(sample_data.raw_component_weights),
                                          sample_data.raw_component_weights_from_texture,
                                          normal_z);
                    }
                }
            }
            continue;
        }

        if (volume->imported_vertex_colors_rgba.empty() || its.vertices.size() != volume->imported_vertex_colors_rgba.size())
            continue;
        for (size_t i = 0; i < its.vertices.size(); ++i) {
            const Vec3d world_pos = volume_trafo * its.vertices[i].cast<double>();
            if (!world_pos.allFinite())
                continue;
            const float dz = std::abs(float(world_pos.z()) - layer_z_mm);
            const float z_norm = dz / safe_layer_z_falloff_mm;
            const float sample_weight = std::exp(-0.5f * z_norm * z_norm);
            if (!std::isfinite(sample_weight) || sample_weight <= EPSILON)
                continue;
            const std::array<float, 4> rgba =
                composite_rgba_over_background(unpack_rgba_u32(volume->imported_vertex_colors_rgba[i]), background_color);
            accumulate_sample(float(world_pos.x()), float(world_pos.y()), rgba, sample_weight);
        }
    }

    if (samples.empty())
        return TextureMappingOffsetWeightField{};

    const int clamped_binary_dither_method =
        std::clamp(dithering_method,
                   int(TextureMappingZone::DitheringClosest),
                   int(TextureMappingZone::DitheringHalftoneV2));
    std::vector<uint32_t> binary_dither_masks;
    const bool can_binary_dither =
        dithering_enabled &&
        !raw_values_mode &&
        !is_halftone_dithering_method(clamped_binary_dither_method) &&
        component_count > 0 &&
        std::isfinite(dither_cell_size_mm) &&
        dither_cell_size_mm > EPSILON;
    if (can_binary_dither) {
        bool has_raw_samples = false;
        for (const WeightedTextureSample &sample : samples) {
            if (sample.raw_component_weights_from_texture || sample.raw_component_weights.size() == component_count) {
                has_raw_samples = true;
                break;
            }
        }

        const std::vector<TextureMappingBinaryDitherCandidate> binary_candidates =
            has_raw_samples ?
                std::vector<TextureMappingBinaryDitherCandidate>{} :
                binary_dither_candidates(component_colors,
                                         component_strength_factors,
                                         component_minimum_offset_factors,
                                         generic_solver_mix_model);
        if (!binary_candidates.empty()) {
            struct BinaryDitherCell {
                int                  x { 0 };
                int                  y { 0 };
                float                weight { 0.f };
                std::array<float, 3> target_oklab { { 0.f, 0.f, 0.f } };
                std::vector<size_t>  sample_indices;
            };

            auto sample_target_oklab = [tone_gamma, contrast_factor](const WeightedTextureSample &sample) {
                return color_solver_oklab_from_srgb(texture_sample_target_rgb(sample, tone_gamma, contrast_factor));
            };

            std::vector<BinaryDitherCell> cells;
            std::map<std::pair<int, int>, size_t> cell_index_by_coord;
            for (size_t sample_idx = 0; sample_idx < samples.size(); ++sample_idx) {
                const WeightedTextureSample &sample = samples[sample_idx];
                if (sample.weight <= EPSILON)
                    continue;

                const int cell_x = int(std::floor((sample.x_mm - min_x_mm) / dither_cell_size_mm));
                const int cell_y = int(std::floor((sample.y_mm - min_y_mm) / dither_cell_size_mm));
                const std::pair<int, int> key(cell_x, cell_y);
                auto cell_it = cell_index_by_coord.find(key);
                if (cell_it == cell_index_by_coord.end()) {
                    cell_it = cell_index_by_coord.emplace(key, cells.size()).first;
                    BinaryDitherCell cell;
                    cell.x = cell_x;
                    cell.y = cell_y;
                    cells.emplace_back(std::move(cell));
                }

                BinaryDitherCell &cell = cells[cell_it->second];
                const std::array<float, 3> target_oklab = sample_target_oklab(sample);
                const float sample_weight = std::max(sample.weight, 0.f);
                for (size_t axis = 0; axis < 3; ++axis)
                    cell.target_oklab[axis] += target_oklab[axis] * sample_weight;
                cell.weight += sample_weight;
                cell.sample_indices.emplace_back(sample_idx);
            }

            if (!cells.empty()) {
                for (BinaryDitherCell &cell : cells)
                    if (cell.weight > EPSILON)
                        for (float &value : cell.target_oklab)
                            value /= cell.weight;

                std::vector<size_t> order(cells.size(), 0);
                std::iota(order.begin(), order.end(), size_t(0));
                std::sort(order.begin(), order.end(), [&cells](size_t lhs, size_t rhs) {
                    if (cells[lhs].y != cells[rhs].y)
                        return cells[lhs].y < cells[rhs].y;
                    return cells[lhs].x < cells[rhs].x;
                });

                binary_dither_masks.assign(samples.size(), 0);
                std::map<std::pair<int, int>, std::array<float, 3>> floyd_error;
                for (const size_t cell_idx : order) {
                    BinaryDitherCell &cell = cells[cell_idx];
                    std::array<float, 3> target_oklab = cell.target_oklab;
                    if (clamped_binary_dither_method == int(TextureMappingZone::DitheringFloydSteinberg)) {
                        const auto error_it = floyd_error.find({ cell.x, cell.y });
                        if (error_it != floyd_error.end())
                            for (size_t axis = 0; axis < 3; ++axis)
                                target_oklab[axis] += error_it->second[axis];
                    }

                    bool thresholded_dither = false;
                    float threshold = 0.f;
                    if (clamped_binary_dither_method == int(TextureMappingZone::DitheringOrderedBayer)) {
                        thresholded_dither = true;
                        threshold = ordered_bayer_threshold(cell.x, cell.y) + 0.5f;
                    }

                    const size_t candidate_idx =
                        thresholded_dither ?
                            thresholded_binary_dither_candidate(binary_candidates, target_oklab, threshold) :
                            nearest_binary_dither_candidate(binary_candidates, target_oklab);
                    if (candidate_idx >= binary_candidates.size())
                        continue;

                    const TextureMappingBinaryDitherCandidate &candidate = binary_candidates[candidate_idx];
                    for (const size_t sample_idx : cell.sample_indices)
                        if (sample_idx < binary_dither_masks.size())
                            binary_dither_masks[sample_idx] = candidate.mask;

                    if (clamped_binary_dither_method == int(TextureMappingZone::DitheringFloydSteinberg)) {
                        std::array<float, 3> error = {
                            target_oklab[0] - candidate.oklab[0],
                            target_oklab[1] - candidate.oklab[1],
                            target_oklab[2] - candidate.oklab[2]
                        };
                        auto add_error = [&floyd_error, &cell_index_by_coord, &error](int x, int y, float factor) {
                            if (cell_index_by_coord.find({ x, y }) == cell_index_by_coord.end())
                                return;
                            std::array<float, 3> &dst = floyd_error[{ x, y }];
                            for (size_t axis = 0; axis < 3; ++axis)
                                dst[axis] += error[axis] * factor;
                        };
                        add_error(cell.x + 1, cell.y, 7.f / 16.f);
                        add_error(cell.x - 1, cell.y + 1, 3.f / 16.f);
                        add_error(cell.x, cell.y + 1, 5.f / 16.f);
                        add_error(cell.x + 1, cell.y + 1, 1.f / 16.f);
                    }
                }
            }
        }
    }

    const std::vector<std::array<float, 3>> fixed_colors = fixed_color_generic_solver_component_colors(filament_color_mode);
    const bool use_fixed_color_generic_solver =
        !raw_values_mode &&
        !use_legacy_fixed_color_mode &&
        fixed_colors.size() == component_count;
    const std::vector<std::array<float, 3>> &solver_colors =
        use_fixed_color_generic_solver ? fixed_colors : component_colors;
    ColorSolverCandidateSet candidates;
    const ColorSolverCandidateSet *candidate_ptr = nullptr;
    if (!raw_values_mode) {
        candidates = build_color_solver_candidates(solver_colors, color_solver_mix_model_from_index(generic_solver_mix_model));
        candidate_ptr = candidates.empty() ? nullptr : &candidates;
    }

    const size_t sample_count = samples.size();
    weight_field.component_count = component_count;
    weight_field.sample_x_mm.resize(sample_count);
    weight_field.sample_y_mm.resize(sample_count);
    weight_field.sample_weight.resize(sample_count);
    weight_field.sample_r.resize(sample_count);
    weight_field.sample_g.resize(sample_count);
    weight_field.sample_b.resize(sample_count);
    weight_field.sample_normal_z.resize(sample_count);
    weight_field.sample_component_weights.assign(sample_count * component_count, 0.f);
    weight_field.raw_component_weights_from_texture = false;
    weight_field.binary_dithered = !binary_dither_masks.empty();

    std::vector<float> fallback_acc(component_count, 0.f);
    float fallback_weight = 0.f;
    for (size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        const WeightedTextureSample &sample = samples[sample_idx];
        if (sample.weight <= EPSILON)
            continue;
        if (sample.raw_component_weights_from_texture)
            weight_field.raw_component_weights_from_texture = true;
        weight_field.sample_x_mm[sample_idx] = sample.x_mm;
        weight_field.sample_y_mm[sample_idx] = sample.y_mm;
        weight_field.sample_weight[sample_idx] = sample.weight;
        const std::array<float, 3> target_rgb = texture_sample_target_rgb(sample, tone_gamma, contrast_factor);
        weight_field.sample_r[sample_idx] = target_rgb[0];
        weight_field.sample_g[sample_idx] = target_rgb[1];
        weight_field.sample_b[sample_idx] = target_rgb[2];
        weight_field.sample_normal_z[sample_idx] = sample.normal_z;

        std::vector<float> desired(component_count, 0.f);
        const bool has_binary_dither = sample_idx < binary_dither_masks.size() && binary_dither_masks[sample_idx] != 0;
        if (has_binary_dither) {
            const uint32_t mask = binary_dither_masks[sample_idx];
            for (size_t component_idx = 0; component_idx < component_count; ++component_idx)
                desired[component_idx] = (mask & (uint32_t(1) << component_idx)) != 0 ? 1.f : 0.f;
        } else {
            desired = component_weights_for_sample(sample,
                                                   component_count,
                                                   raw_values_mode,
                                                   filament_color_mode,
                                                   force_sequential_filaments,
                                                   generic_solver_lookup_mode,
                                                   generic_solver_mode,
                                                   use_fixed_color_generic_solver,
                                                   contrast_factor,
                                                   tone_gamma,
                                                   component_colors,
                                                   candidate_ptr);
        }
        for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
            const float v = component_idx < desired.size() ? clamp01f(desired[component_idx]) : 0.f;
            weight_field.sample_component_weights[sample_idx * component_count + component_idx] = v;
            fallback_acc[component_idx] += v * sample.weight;
        }
        fallback_weight += sample.weight;
    }

    weight_field.fallback_weights.assign(component_count, 1.f / float(component_count));
    if (fallback_weight > EPSILON) {
        for (size_t component_idx = 0; component_idx < component_count; ++component_idx)
            weight_field.fallback_weights[component_idx] = clamp01f(fallback_acc[component_idx] / fallback_weight);
    }

    const float target_bucket_mm = high_resolution_texture_sampling ? 0.12f : 0.22f;
    constexpr int min_bucket_dim = 16;
    constexpr int max_bucket_dim = 320;
    constexpr int max_buckets = 72000;
    int bucket_width = std::clamp(int(std::ceil(span_x_mm / target_bucket_mm)) + 1, min_bucket_dim, max_bucket_dim);
    int bucket_height = std::clamp(int(std::ceil(span_y_mm / target_bucket_mm)) + 1, min_bucket_dim, max_bucket_dim);
    const int initial_buckets = bucket_width * bucket_height;
    if (initial_buckets > max_buckets) {
        const float scale_factor = std::sqrt(float(initial_buckets) / float(max_buckets));
        bucket_width = std::max(min_bucket_dim, int(std::ceil(float(bucket_width) / scale_factor)));
        bucket_height = std::max(min_bucket_dim, int(std::ceil(float(bucket_height) / scale_factor)));
    }

    weight_field.min_x_mm = min_x_mm;
    weight_field.min_y_mm = min_y_mm;
    weight_field.bucket_width = bucket_width;
    weight_field.bucket_height = bucket_height;
    weight_field.bucket_width_mm = std::max(1e-3f, span_x_mm / std::max(1, bucket_width - 1));
    weight_field.bucket_height_mm = std::max(1e-3f, span_y_mm / std::max(1, bucket_height - 1));
    weight_field.buckets.assign(size_t(bucket_width) * size_t(bucket_height), {});
    for (size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        const float gx = (weight_field.sample_x_mm[sample_idx] - min_x_mm) / weight_field.bucket_width_mm;
        const float gy = (weight_field.sample_y_mm[sample_idx] - min_y_mm) / weight_field.bucket_height_mm;
        const int bx = std::clamp(int(std::floor(gx)), 0, bucket_width - 1);
        const int by = std::clamp(int(std::floor(gy)), 0, bucket_height - 1);
        weight_field.buckets[size_t(by) * size_t(bucket_width) + size_t(bx)].push_back(uint32_t(sample_idx));
    }
    return weight_field;
}

std::vector<float> sample_weight_field_components_impl(const TextureMappingOffsetWeightField &weight_field,
                                                       float x_mm,
                                                       float y_mm,
                                                       bool high_resolution_texture_sampling)
{
    std::vector<float> fallback = weight_field.fallback_weights;
    if (fallback.size() < weight_field.component_count)
        fallback.resize(weight_field.component_count, 0.f);
    if (weight_field.empty() || !std::isfinite(x_mm) || !std::isfinite(y_mm))
        return fallback;

    const float gx = (x_mm - weight_field.min_x_mm) / std::max(weight_field.bucket_width_mm, 1e-6f);
    const float gy = (y_mm - weight_field.min_y_mm) / std::max(weight_field.bucket_height_mm, 1e-6f);
    const int cx = std::clamp(int(std::floor(gx)), 0, weight_field.bucket_width - 1);
    const int cy = std::clamp(int(std::floor(gy)), 0, weight_field.bucket_height - 1);
    if (weight_field.binary_dithered) {
        float nearest_d2 = std::numeric_limits<float>::max();
        size_t nearest_sample_idx = size_t(-1);
        const int nearest_ring_limit = std::min(16, std::max(weight_field.bucket_width, weight_field.bucket_height));
        for (int ring = 0; ring <= nearest_ring_limit; ++ring) {
            const int min_x = std::max(0, cx - ring);
            const int max_x = std::min(weight_field.bucket_width - 1, cx + ring);
            const int min_y = std::max(0, cy - ring);
            const int max_y = std::min(weight_field.bucket_height - 1, cy + ring);

            auto visit_bucket = [&weight_field, x_mm, y_mm, &nearest_d2, &nearest_sample_idx](int bx, int by) {
                if (bx < 0 || by < 0 || bx >= weight_field.bucket_width || by >= weight_field.bucket_height)
                    return;
                const size_t bucket_idx = size_t(by) * size_t(weight_field.bucket_width) + size_t(bx);
                if (bucket_idx >= weight_field.buckets.size())
                    return;
                for (const uint32_t sample_idx_u32 : weight_field.buckets[bucket_idx]) {
                    const size_t sample_idx = size_t(sample_idx_u32);
                    if (sample_idx >= weight_field.sample_x_mm.size() || sample_idx >= weight_field.sample_y_mm.size())
                        continue;
                    const float dx = x_mm - weight_field.sample_x_mm[sample_idx];
                    const float dy = y_mm - weight_field.sample_y_mm[sample_idx];
                    const float d2 = dx * dx + dy * dy;
                    if (d2 >= nearest_d2)
                        continue;
                    const size_t value_idx = sample_idx * weight_field.component_count;
                    if (value_idx + weight_field.component_count > weight_field.sample_component_weights.size())
                        continue;
                    nearest_d2 = d2;
                    nearest_sample_idx = sample_idx;
                }
            };

            if (ring == 0) {
                visit_bucket(cx, cy);
            } else {
                for (int x = min_x; x <= max_x; ++x) {
                    visit_bucket(x, min_y);
                    if (max_y != min_y)
                        visit_bucket(x, max_y);
                }
                for (int y = min_y + 1; y <= max_y - 1; ++y) {
                    visit_bucket(min_x, y);
                    if (max_x != min_x)
                        visit_bucket(max_x, y);
                }
            }

            if (nearest_sample_idx != size_t(-1))
                break;
        }

        if (nearest_sample_idx != size_t(-1)) {
            std::vector<float> values(weight_field.component_count, 0.f);
            const size_t value_idx = nearest_sample_idx * weight_field.component_count;
            for (size_t component_idx = 0; component_idx < weight_field.component_count; ++component_idx)
                values[component_idx] = clamp01f(weight_field.sample_component_weights[value_idx + component_idx]);
            return values;
        }
        return fallback;
    }

    const float sigma_scale = high_resolution_texture_sampling ? 0.45f : 0.7f;
    const float min_sigma_mm = high_resolution_texture_sampling ? 0.04f : 0.06f;
    const float sigma_x_mm = std::max(min_sigma_mm, weight_field.bucket_width_mm * sigma_scale);
    const float sigma_y_mm = std::max(min_sigma_mm, weight_field.bucket_height_mm * sigma_scale);
    const float inv_two_sigma_x2 = 1.f / std::max(2.f * sigma_x_mm * sigma_x_mm, 1e-8f);
    const float inv_two_sigma_y2 = 1.f / std::max(2.f * sigma_y_mm * sigma_y_mm, 1e-8f);
    const float min_radius_mm = high_resolution_texture_sampling ? 0.16f : 0.30f;
    const float radius_scale = high_resolution_texture_sampling ? 1.75f : 3.f;
    const float max_radius_mm = std::max(min_radius_mm, std::max(weight_field.bucket_width_mm, weight_field.bucket_height_mm) * radius_scale);
    const float max_radius2 = max_radius_mm * max_radius_mm;
    const float min_bucket_span_mm = std::max(1e-3f, std::min(weight_field.bucket_width_mm, weight_field.bucket_height_mm));
    const int max_ring = std::max(1, int(std::ceil(max_radius_mm / min_bucket_span_mm)));

    std::vector<float> weighted_sum(weight_field.component_count, 0.f);
    float total_weight = 0.f;
    size_t contributing_samples = 0;
    auto process_bucket = [&](int bx, int by) {
        if (bx < 0 || by < 0 || bx >= weight_field.bucket_width || by >= weight_field.bucket_height)
            return;
        const size_t bucket_idx = size_t(by) * size_t(weight_field.bucket_width) + size_t(bx);
        if (bucket_idx >= weight_field.buckets.size())
            return;
        for (const uint32_t sample_idx_u32 : weight_field.buckets[bucket_idx]) {
            const size_t sample_idx = size_t(sample_idx_u32);
            if (sample_idx >= weight_field.sample_x_mm.size() ||
                sample_idx >= weight_field.sample_y_mm.size() ||
                sample_idx >= weight_field.sample_weight.size())
                continue;
            const float dx = x_mm - weight_field.sample_x_mm[sample_idx];
            const float dy = y_mm - weight_field.sample_y_mm[sample_idx];
            const float d2 = dx * dx + dy * dy;
            if (d2 > max_radius2)
                continue;
            const float kernel = std::exp(-(dx * dx) * inv_two_sigma_x2 - (dy * dy) * inv_two_sigma_y2);
            const float sample_w = weight_field.sample_weight[sample_idx] * kernel;
            if (!std::isfinite(sample_w) || sample_w <= EPSILON)
                continue;
            const size_t value_idx = sample_idx * weight_field.component_count;
            if (value_idx + weight_field.component_count > weight_field.sample_component_weights.size())
                continue;
            for (size_t component_idx = 0; component_idx < weight_field.component_count; ++component_idx)
                weighted_sum[component_idx] += weight_field.sample_component_weights[value_idx + component_idx] * sample_w;
            total_weight += sample_w;
            ++contributing_samples;
        }
    };

    for (int ring = 0; ring <= max_ring; ++ring) {
        const int min_x = std::max(0, cx - ring);
        const int max_x = std::min(weight_field.bucket_width - 1, cx + ring);
        const int min_y = std::max(0, cy - ring);
        const int max_y = std::min(weight_field.bucket_height - 1, cy + ring);
        if (ring == 0) {
            process_bucket(cx, cy);
        } else {
            for (int x = min_x; x <= max_x; ++x) {
                process_bucket(x, min_y);
                if (max_y != min_y)
                    process_bucket(x, max_y);
            }
            for (int y = min_y + 1; y <= max_y - 1; ++y) {
                process_bucket(min_x, y);
                if (max_x != min_x)
                    process_bucket(max_x, y);
            }
        }
        if (total_weight > EPSILON && contributing_samples >= 12)
            break;
    }

    if (total_weight > EPSILON) {
        std::vector<float> values(weight_field.component_count, 0.f);
        for (size_t component_idx = 0; component_idx < weight_field.component_count; ++component_idx)
            values[component_idx] = clamp01f(weighted_sum[component_idx] / total_weight);
        return values;
    }

    float nearest_d2 = std::numeric_limits<float>::max();
    size_t nearest_sample_idx = size_t(-1);
    const int nearest_ring_limit = std::min(std::max(max_ring + 2, 4), std::max(weight_field.bucket_width, weight_field.bucket_height));
    for (int ring = 0; ring <= nearest_ring_limit; ++ring) {
        const int min_x = std::max(0, cx - ring);
        const int max_x = std::min(weight_field.bucket_width - 1, cx + ring);
        const int min_y = std::max(0, cy - ring);
        const int max_y = std::min(weight_field.bucket_height - 1, cy + ring);

        auto visit_bucket = [&weight_field, x_mm, y_mm, &nearest_d2, &nearest_sample_idx](int bx, int by) {
            if (bx < 0 || by < 0 || bx >= weight_field.bucket_width || by >= weight_field.bucket_height)
                return;
            const size_t bucket_idx = size_t(by) * size_t(weight_field.bucket_width) + size_t(bx);
            if (bucket_idx >= weight_field.buckets.size())
                return;
            for (const uint32_t sample_idx_u32 : weight_field.buckets[bucket_idx]) {
                const size_t sample_idx = size_t(sample_idx_u32);
                if (sample_idx >= weight_field.sample_x_mm.size() || sample_idx >= weight_field.sample_y_mm.size())
                    continue;
                const float dx = x_mm - weight_field.sample_x_mm[sample_idx];
                const float dy = y_mm - weight_field.sample_y_mm[sample_idx];
                const float d2 = dx * dx + dy * dy;
                if (d2 >= nearest_d2)
                    continue;
                const size_t value_idx = sample_idx * weight_field.component_count;
                if (value_idx + weight_field.component_count > weight_field.sample_component_weights.size())
                    continue;
                nearest_d2 = d2;
                nearest_sample_idx = sample_idx;
            }
        };

        if (ring == 0) {
            visit_bucket(cx, cy);
        } else {
            for (int x = min_x; x <= max_x; ++x) {
                visit_bucket(x, min_y);
                if (max_y != min_y)
                    visit_bucket(x, max_y);
            }
            for (int y = min_y + 1; y <= max_y - 1; ++y) {
                visit_bucket(min_x, y);
                if (max_x != min_x)
                    visit_bucket(max_x, y);
            }
        }
        if (nearest_d2 < std::numeric_limits<float>::max() && ring >= 2)
            break;
    }

    if (nearest_sample_idx != size_t(-1)) {
        std::vector<float> values(weight_field.component_count, 0.f);
        const size_t value_idx = nearest_sample_idx * weight_field.component_count;
        for (size_t component_idx = 0; component_idx < weight_field.component_count; ++component_idx)
            values[component_idx] = clamp01f(weight_field.sample_component_weights[value_idx + component_idx]);
        return values;
    }
    return fallback;
}

float sample_weight_field(const TextureMappingOffsetWeightField &weight_field,
                          float x_mm,
                          float y_mm,
                          size_t component_idx,
                          bool high_resolution_texture_sampling,
                          bool compact_offset_mode)
{
    const float fallback = component_idx < weight_field.fallback_weights.size() ?
        weight_field.fallback_weights[component_idx] : 0.f;
    if (weight_field.empty() || component_idx >= weight_field.component_count)
        return fallback;

    std::vector<float> values = sample_weight_field_components(weight_field, x_mm, y_mm, high_resolution_texture_sampling);
    if (component_idx >= values.size())
        return fallback;
    if (compact_offset_mode && !weight_field.raw_component_weights_from_texture) {
        float max_value = 0.f;
        for (size_t idx = 0; idx < weight_field.component_count && idx < values.size(); ++idx)
            max_value = std::max(max_value, clamp01f(values[idx]));
        if (max_value > EPSILON)
            return clamp01f(values[component_idx] / max_value);
    }
    return clamp01f(values[component_idx]);
}

bool texture_mapping_component_is_black(size_t component_idx,
                                        int filament_color_mode,
                                        const std::vector<std::array<float, 3>> &component_colors)
{
    switch (std::clamp(filament_color_mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorCMYK):
    case int(TextureMappingZone::FilamentColorRGBK):
    case int(TextureMappingZone::FilamentColorCMYKW):
    case int(TextureMappingZone::FilamentColorRGBKW):
        if (component_idx == 3)
            return true;
        break;
    case int(TextureMappingZone::FilamentColorBW):
        if (component_idx == 0)
            return true;
        break;
    default:
        break;
    }
    if (component_idx >= component_colors.size())
        return false;
    const std::array<float, 3> &c = component_colors[component_idx];
    const float max_channel = std::max({ c[0], c[1], c[2] });
    const float luminance = 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
    return max_channel <= 0.18f && luminance <= 0.12f;
}

bool explicit_transmission_distance(const TextureMappingZone &zone, unsigned int physical_filament_id, float &td_mm)
{
    if (physical_filament_id == 0)
        return false;
    const size_t idx = size_t(physical_filament_id - 1);
    if (idx >= zone.filament_transmission_distances_mm.size())
        return false;
    const float value = zone.filament_transmission_distances_mm[idx];
    if (!std::isfinite(value) || value <= 0.f)
        return false;
    td_mm = std::clamp(value, 0.01f, 50.f);
    return true;
}

float transmission_distance_reference(bool is_black)
{
    return is_black ? 0.1f : 3.f;
}

float transmission_distance_opacity(float td_mm, float path_extension_mm)
{
    constexpr float surface_scatter = 0.50f;
    constexpr float surface_depth_mm = 0.32f;
    const float safe_td = std::clamp(td_mm, 0.01f, 50.f);
    const float path_mm = std::max(0.f, surface_depth_mm + std::max(0.f, path_extension_mm));
    const float opacity = surface_scatter + (1.f - surface_scatter) * (1.f - std::exp(-path_mm / safe_td));
    return std::clamp(opacity, 1e-4f, 1.f);
}

TransmissionDistanceCalibrationContext transmission_distance_calibration_context(
    const TextureMappingZone &zone,
    const std::vector<unsigned int> &component_ids,
    const std::vector<std::array<float, 3>> &component_colors,
    int filament_color_mode)
{
    TransmissionDistanceCalibrationContext context;
    context.mode = std::clamp(zone.transmission_distance_calibration_mode,
                              int(TextureMappingZone::TDCalibrationNone),
                              int(TextureMappingZone::TDCalibrationNeighbor));
    if (context.mode == int(TextureMappingZone::TDCalibrationNone) || component_ids.empty())
        return context;

    std::vector<float> explicit_tds(component_ids.size(), 0.f);
    bool has_active_explicit_td = false;
    for (size_t idx = 0; idx < component_ids.size(); ++idx) {
        float td_mm = 0.f;
        if (explicit_transmission_distance(zone, component_ids[idx], td_mm)) {
            explicit_tds[idx] = td_mm;
            has_active_explicit_td = true;
        }
    }
    if (!has_active_explicit_td)
        return context;

    const bool neighbor_mode = context.mode == int(TextureMappingZone::TDCalibrationNeighbor);
    const float path_extension_mm = neighbor_mode ? 0.16f : 0.12f;
    const float own_power = neighbor_mode ? 0.25f : 0.35f;
    context.own_width_factors.assign(component_ids.size(), 1.f);
    context.neighbor_opacity_ratios.assign(component_ids.size(), 1.f);
    context.enabled = true;
    for (size_t idx = 0; idx < component_ids.size(); ++idx) {
        const bool is_black = texture_mapping_component_is_black(idx, filament_color_mode, component_colors);
        const float reference_td_mm = transmission_distance_reference(is_black);
        const float actual_td_mm = explicit_tds[idx] > 0.f ? explicit_tds[idx] : reference_td_mm;
        const float actual_opacity = transmission_distance_opacity(actual_td_mm, path_extension_mm);
        const float reference_opacity = transmission_distance_opacity(reference_td_mm, path_extension_mm);
        context.own_width_factors[idx] =
            std::clamp(std::pow(reference_opacity / std::max(actual_opacity, 1e-4f), own_power), 0.25f, 2.f);
        context.neighbor_opacity_ratios[idx] =
            std::clamp(actual_opacity / std::max(reference_opacity, 1e-4f), 0.25f, 4.f);
    }
    return context;
}

float transmission_distance_width_factor(const TransmissionDistanceCalibrationContext &context,
                                         size_t active_component_idx,
                                         size_t previous_component_idx)
{
    if (!context.enabled || active_component_idx >= context.own_width_factors.size())
        return 1.f;

    float factor = context.own_width_factors[active_component_idx];
    if (context.mode == int(TextureMappingZone::TDCalibrationNeighbor) &&
        previous_component_idx < context.neighbor_opacity_ratios.size())
        factor *= std::pow(context.neighbor_opacity_ratios[previous_component_idx], 0.20f);
    return std::clamp(factor, 0.25f, 2.f);
}

float nonlinear_visibility_width_factor(float desired_width_factor,
                                        float layer_height_mm,
                                        float stair_step_mm,
                                        float max_width_delta_limit_mm)
{
    const float r = clamp01f(desired_width_factor);
    if (!std::isfinite(layer_height_mm) ||
        !std::isfinite(stair_step_mm) ||
        !std::isfinite(max_width_delta_limit_mm) ||
        layer_height_mm <= EPSILON ||
        max_width_delta_limit_mm <= EPSILON)
        return r;
    if (r <= EPSILON || r >= 1.f - EPSILON)
        return r;
    if (std::abs(r - 0.5f) <= 1e-5f)
        return 0.5f;

    const float h = std::max(0.01f, layer_height_mm);
    const float d = std::max(0.f, stair_step_mm);
    const float diag = std::hypot(h, d);
    if (!std::isfinite(diag) || diag <= EPSILON)
        return r;

    const float symmetric_r = std::min(r, 1.f - r);
    const float direction = r >= 0.5f ? 1.f : -1.f;
    const float sin_n = std::clamp(d / diag, 0.f, 1.f);
    const float cos_n = std::clamp(h / diag, 1e-4f, 1.f);
    const float sin_cos = sin_n * cos_n;
    float offset_mm = 0.f;
    if (sin_cos > 1e-5f) {
        offset_mm = (0.5f - symmetric_r) * h / sin_cos;
        if (2.f * std::abs(offset_mm) <= d + EPSILON)
            return std::clamp(0.5f + direction * offset_mm / max_width_delta_limit_mm, 0.f, 1.f);
    }

    const float cx = std::clamp(1.f - std::sqrt(2.f) / 2.f, 0.f, 0.95f);
    const float c = (1.f - cx) * (1.f - cx);
    const float safe_cos = std::max(cos_n, 1e-4f);
    const float tan_n = sin_n / safe_cos;
    const float a = -0.5f * c * (1.f + sin_n) / std::max(h * diag, 1e-6f);
    const float b = 0.5f * (c * tan_n * (1.f + sin_n) + 2.f * cos_n * (cx - 1.f)) / std::max(diag, 1e-6f);
    const float q = c * 0.25f * tan_n * (1.f + sin_n) - cx * cos_n;
    const float cc = 0.5f - 0.5f * cos_n * q - symmetric_r;
    const float det = std::max(0.f, b * b - 4.f * a * cc);
    if (std::abs(a) > 1e-8f) {
        offset_mm = (-b - std::sqrt(det)) / (2.f * a);
        if (!std::isfinite(offset_mm) || offset_mm < 0.f)
            offset_mm = (-b + std::sqrt(det)) / (2.f * a);
    }
    if (!std::isfinite(offset_mm) || offset_mm < 0.f) {
        if (sin_cos > 1e-5f)
            offset_mm = (0.5f - symmetric_r) * h / sin_cos;
        else
            offset_mm = max_width_delta_limit_mm;
    }
    return std::clamp(0.5f + direction * offset_mm / max_width_delta_limit_mm, 0.f, 1.f);
}

float variable_width_delta_for_visibility_range(float inset_strength,
                                                float max_width_delta_limit_mm,
                                                float minimum_offset_factor,
                                                float strength_factor,
                                                float td_width_factor,
                                                bool nonlinear_offset_adjustment,
                                                float layer_height_mm,
                                                float stair_step_mm)
{
    if (!std::isfinite(max_width_delta_limit_mm) || max_width_delta_limit_mm <= 0.f)
        return 0.f;
    float desired_width_factor =
        std::clamp((1.f - std::clamp(inset_strength, 0.f, 1.f)) *
                       std::clamp(td_width_factor, 0.f, 2.f),
                   0.f,
                   1.f);
    if (nonlinear_offset_adjustment)
        desired_width_factor = nonlinear_visibility_width_factor(desired_width_factor,
                                                                 layer_height_mm,
                                                                 stair_step_mm,
                                                                 max_width_delta_limit_mm);
    const float min_width_factor = std::clamp(minimum_offset_factor, 0.f, 1.f);
    const float adjusted_width_factor =
        min_width_factor + desired_width_factor * std::clamp(strength_factor, 0.f, 1.f) * (1.f - min_width_factor);
    return std::clamp(max_width_delta_limit_mm * (1.f - adjusted_width_factor), 0.f, max_width_delta_limit_mm);
}

double bbox_distance_sq_to_point(const BoundingBox &bbox, const Point &point)
{
    if (!bbox.defined)
        return 0.0;

    double dx = 0.0;
    if (point.x() < bbox.min.x())
        dx = double(bbox.min.x() - point.x());
    else if (point.x() > bbox.max.x())
        dx = double(point.x() - bbox.max.x());

    double dy = 0.0;
    if (point.y() < bbox.min.y())
        dy = double(bbox.min.y() - point.y());
    else if (point.y() > bbox.max.y())
        dy = double(point.y() - bbox.max.y());
    return dx * dx + dy * dy;
}

bool find_nearest_layer_slice_boundary_point(const Layer *layer, const Point &query_point, Point &nearest_point)
{
    if (layer == nullptr || layer->lslices.empty())
        return false;

    const bool has_slice_bboxes = layer->lslices_bboxes.size() == layer->lslices.size();
    double best_distance_sq = std::numeric_limits<double>::max();
    bool found = false;
    for (size_t slice_idx = 0; slice_idx < layer->lslices.size(); ++slice_idx) {
        const ExPolygon &slice = layer->lslices[slice_idx];
        if (slice.empty())
            continue;
        if (has_slice_bboxes && layer->lslices_bboxes[slice_idx].defined) {
            const double bbox_distance_sq = bbox_distance_sq_to_point(layer->lslices_bboxes[slice_idx], query_point);
            if (bbox_distance_sq > best_distance_sq)
                continue;
        }
        const Point projected = slice.point_projection(query_point);
        const double projected_distance_sq = (projected - query_point).cast<double>().squaredNorm();
        if (projected_distance_sq < best_distance_sq) {
            best_distance_sq = projected_distance_sq;
            nearest_point = projected;
            found = true;
        }
    }
    return found;
}

float local_surface_stair_step_distance(const Layer *layer,
                                        const Point &mid_point,
                                        double outward_x,
                                        double outward_y,
                                        float base_outer_width_mm,
                                        float max_allowed_distance_mm)
{
    if (layer == nullptr || !std::isfinite(outward_x) || !std::isfinite(outward_y))
        return std::numeric_limits<float>::quiet_NaN();

    const double half_width_scaled = scale_(0.5 * double(std::max(0.01f, base_outer_width_mm)));
    const Point current_base_edge(
        coord_t(std::llround(double(mid_point.x()) + outward_x * half_width_scaled)),
        coord_t(std::llround(double(mid_point.y()) + outward_y * half_width_scaled)));
    const float max_local_edge_tangent_delta_mm = std::max(1.0f, base_outer_width_mm * 2.f);
    const float max_local_edge_normal_delta_mm =
        std::max(2.0f, base_outer_width_mm * 4.f + 2.f * std::max(0.f, max_allowed_distance_mm));
    float best_distance_mm = std::numeric_limits<float>::quiet_NaN();

    auto consider_adjacent_layer = [&](const Layer *adjacent_layer) {
        if (adjacent_layer == nullptr)
            return;
        Point adjacent_base_edge;
        if (!find_nearest_layer_slice_boundary_point(adjacent_layer, current_base_edge, adjacent_base_edge))
            return;
        const double edge_delta_x = double(adjacent_base_edge.x()) - double(current_base_edge.x());
        const double edge_delta_y = double(adjacent_base_edge.y()) - double(current_base_edge.y());
        const double edge_distance_scaled = std::hypot(edge_delta_x, edge_delta_y);
        const double edge_normal_delta_scaled = edge_delta_x * outward_x + edge_delta_y * outward_y;
        const double edge_tangent_delta_scaled_sq =
            std::max(0.0, edge_distance_scaled * edge_distance_scaled - edge_normal_delta_scaled * edge_normal_delta_scaled);
        const float edge_tangent_delta_mm = unscale<float>(std::sqrt(edge_tangent_delta_scaled_sq));
        if (!std::isfinite(edge_tangent_delta_mm) || edge_tangent_delta_mm > max_local_edge_tangent_delta_mm)
            return;
        const float edge_normal_delta_mm = std::abs(unscale<float>(edge_normal_delta_scaled));
        if (!std::isfinite(edge_normal_delta_mm) || edge_normal_delta_mm > max_local_edge_normal_delta_mm)
            return;
        if (!std::isfinite(best_distance_mm) || edge_normal_delta_mm < best_distance_mm)
            best_distance_mm = edge_normal_delta_mm;
    };

    consider_adjacent_layer(layer->upper_layer);
    consider_adjacent_layer(layer->lower_layer);
    return best_distance_mm;
}

float component_angular_influence(unsigned int active_component_id,
                                  float theta_deg,
                                  const std::vector<unsigned int> &component_ids,
                                  const std::vector<float> &component_angles_deg)
{
    if (component_ids.empty() || component_ids.size() != component_angles_deg.size())
        return 0.f;
    const auto active_it = std::find(component_ids.begin(), component_ids.end(), active_component_id);
    if (active_it == component_ids.end())
        return 0.f;
    if (component_ids.size() == 1)
        return 1.f;

    struct SortedComponentAngle {
        float angle_deg { 0.f };
        size_t component_idx { 0 };
    };
    std::vector<SortedComponentAngle> sorted_angles;
    sorted_angles.reserve(component_ids.size());
    for (size_t i = 0; i < component_ids.size(); ++i)
        sorted_angles.push_back({ normalize_texture_mapping_offset_angle_deg(component_angles_deg[i]), i });
    std::sort(sorted_angles.begin(), sorted_angles.end(), [](const SortedComponentAngle &lhs, const SortedComponentAngle &rhs) {
        return lhs.angle_deg < rhs.angle_deg;
    });

    const size_t active_component_idx = size_t(active_it - component_ids.begin());
    const auto sorted_active_it = std::find_if(sorted_angles.begin(), sorted_angles.end(),
                                               [active_component_idx](const SortedComponentAngle &entry) {
                                                   return entry.component_idx == active_component_idx;
                                               });
    if (sorted_active_it == sorted_angles.end())
        return 0.f;

    const size_t sorted_pos = size_t(sorted_active_it - sorted_angles.begin());
    const size_t count = sorted_angles.size();
    const float prev_angle = sorted_angles[(sorted_pos + count - 1) % count].angle_deg;
    const float self_angle = sorted_angles[sorted_pos].angle_deg;
    const float next_angle = sorted_angles[(sorted_pos + 1) % count].angle_deg;
    const float prev_to_self_deg = normalize_texture_mapping_offset_angle_deg(self_angle - prev_angle);
    const float self_to_next_deg = normalize_texture_mapping_offset_angle_deg(next_angle - self_angle);
    if (prev_to_self_deg <= 1e-3f || self_to_next_deg <= 1e-3f) {
        float total_weight = 0.f;
        float active_weight = 0.f;
        for (size_t i = 0; i < component_ids.size(); ++i) {
            const float dist = std::abs(normalize_texture_mapping_offset_angle_deg(theta_deg) -
                                        normalize_texture_mapping_offset_angle_deg(component_angles_deg[i]));
            const float wrapped_dist = std::min(dist, 360.f - dist);
            const float weight = std::max(0.f, 1.f - wrapped_dist / 180.f);
            total_weight += weight;
            if (component_ids[i] == active_component_id)
                active_weight += weight;
        }
        return total_weight <= EPSILON ? 0.f : std::clamp(active_weight / total_weight, 0.f, 1.f);
    }

    const float theta_norm = normalize_texture_mapping_offset_angle_deg(theta_deg);
    const float prev_to_theta_deg = normalize_texture_mapping_offset_angle_deg(theta_norm - prev_angle);
    if (prev_to_theta_deg <= prev_to_self_deg + 1e-4f)
        return std::clamp(prev_to_theta_deg / prev_to_self_deg, 0.f, 1.f);
    const float self_to_theta_deg = normalize_texture_mapping_offset_angle_deg(theta_norm - self_angle);
    if (self_to_theta_deg <= self_to_next_deg + 1e-4f)
        return std::clamp(1.f - self_to_theta_deg / self_to_next_deg, 0.f, 1.f);
    return 0.f;
}

}

std::vector<float> sample_weight_field_components(const TextureMappingOffsetWeightField &weight_field,
                                                  float x_mm,
                                                  float y_mm,
                                                  bool high_resolution_texture_sampling)
{
    return sample_weight_field_components_impl(weight_field, x_mm, y_mm, high_resolution_texture_sampling);
}

std::optional<std::array<float, 3>> sample_weight_field_rgb(const TextureMappingOffsetWeightField &weight_field,
                                                            float x_mm,
                                                            float y_mm,
                                                            bool high_resolution_texture_sampling)
{
    if (weight_field.empty() ||
        weight_field.sample_r.size() != weight_field.sample_x_mm.size() ||
        weight_field.sample_g.size() != weight_field.sample_x_mm.size() ||
        weight_field.sample_b.size() != weight_field.sample_x_mm.size() ||
        !std::isfinite(x_mm) ||
        !std::isfinite(y_mm))
        return std::nullopt;

    const float gx = (x_mm - weight_field.min_x_mm) / std::max(weight_field.bucket_width_mm, 1e-6f);
    const float gy = (y_mm - weight_field.min_y_mm) / std::max(weight_field.bucket_height_mm, 1e-6f);
    const int cx = std::clamp(int(std::floor(gx)), 0, weight_field.bucket_width - 1);
    const int cy = std::clamp(int(std::floor(gy)), 0, weight_field.bucket_height - 1);
    const float max_radius_mm =
        high_resolution_texture_sampling ?
            std::max(0.18f, std::max(weight_field.bucket_width_mm, weight_field.bucket_height_mm) * 2.5f) :
            std::max(0.32f, std::max(weight_field.bucket_width_mm, weight_field.bucket_height_mm) * 3.0f);
    const int max_ring = std::clamp(int(std::ceil(max_radius_mm / std::max(std::min(weight_field.bucket_width_mm,
                                                                                       weight_field.bucket_height_mm),
                                                                               1e-3f))),
                                    1,
                                    std::max(weight_field.bucket_width, weight_field.bucket_height));
    double weighted_r = 0.0;
    double weighted_g = 0.0;
    double weighted_b = 0.0;
    double total_weight = 0.0;
    size_t nearest_sample_idx = size_t(-1);
    float nearest_d2 = std::numeric_limits<float>::max();

    for (int ring = 0; ring <= max_ring; ++ring) {
        const int min_x = std::max(0, cx - ring);
        const int max_x = std::min(weight_field.bucket_width - 1, cx + ring);
        const int min_y = std::max(0, cy - ring);
        const int max_y = std::min(weight_field.bucket_height - 1, cy + ring);
        for (int by = min_y; by <= max_y; ++by) {
            for (int bx = min_x; bx <= max_x; ++bx) {
                const size_t bucket_idx = size_t(by) * size_t(weight_field.bucket_width) + size_t(bx);
                if (bucket_idx >= weight_field.buckets.size())
                    continue;
                for (const uint32_t sample_idx_u32 : weight_field.buckets[bucket_idx]) {
                    const size_t sample_idx = size_t(sample_idx_u32);
                    if (sample_idx >= weight_field.sample_x_mm.size())
                        continue;
                    const float dx = x_mm - weight_field.sample_x_mm[sample_idx];
                    const float dy = y_mm - weight_field.sample_y_mm[sample_idx];
                    const float d2 = dx * dx + dy * dy;
                    if (d2 < nearest_d2) {
                        nearest_d2 = d2;
                        nearest_sample_idx = sample_idx;
                    }
                    if (d2 > max_radius_mm * max_radius_mm)
                        continue;
                    const float kernel = std::exp(-0.5f * d2 / std::max(max_radius_mm * max_radius_mm * 0.16f, 1e-6f));
                    const float sample_w = weight_field.sample_weight[sample_idx] * kernel;
                    if (!std::isfinite(sample_w) || sample_w <= EPSILON)
                        continue;
                    weighted_r += double(weight_field.sample_r[sample_idx]) * double(sample_w);
                    weighted_g += double(weight_field.sample_g[sample_idx]) * double(sample_w);
                    weighted_b += double(weight_field.sample_b[sample_idx]) * double(sample_w);
                    total_weight += double(sample_w);
                }
            }
        }
        if (total_weight > EPSILON)
            break;
    }

    if (total_weight > EPSILON)
        return std::array<float, 3>{ clamp01f(float(weighted_r / total_weight)),
                                     clamp01f(float(weighted_g / total_weight)),
                                     clamp01f(float(weighted_b / total_weight)) };

    if (nearest_sample_idx != size_t(-1))
        return std::array<float, 3>{ clamp01f(weight_field.sample_r[nearest_sample_idx]),
                                     clamp01f(weight_field.sample_g[nearest_sample_idx]),
                                     clamp01f(weight_field.sample_b[nearest_sample_idx]) };

    return std::nullopt;
}

std::optional<float> sample_weight_field_normal_z(const TextureMappingOffsetWeightField &weight_field,
                                                  float x_mm,
                                                  float y_mm,
                                                  bool high_resolution_texture_sampling)
{
    if (weight_field.empty() ||
        weight_field.sample_normal_z.size() != weight_field.sample_x_mm.size() ||
        !std::isfinite(x_mm) ||
        !std::isfinite(y_mm))
        return std::nullopt;

    const float gx = (x_mm - weight_field.min_x_mm) / std::max(weight_field.bucket_width_mm, 1e-6f);
    const float gy = (y_mm - weight_field.min_y_mm) / std::max(weight_field.bucket_height_mm, 1e-6f);
    const int cx = std::clamp(int(std::floor(gx)), 0, weight_field.bucket_width - 1);
    const int cy = std::clamp(int(std::floor(gy)), 0, weight_field.bucket_height - 1);
    const float max_radius_mm =
        high_resolution_texture_sampling ?
            std::max(0.18f, std::max(weight_field.bucket_width_mm, weight_field.bucket_height_mm) * 2.5f) :
            std::max(0.32f, std::max(weight_field.bucket_width_mm, weight_field.bucket_height_mm) * 3.0f);
    const int max_ring = std::clamp(int(std::ceil(max_radius_mm / std::max(std::min(weight_field.bucket_width_mm,
                                                                                       weight_field.bucket_height_mm),
                                                                               1e-3f))),
                                    1,
                                    std::max(weight_field.bucket_width, weight_field.bucket_height));
    double weighted_normal_z = 0.0;
    double total_weight = 0.0;
    size_t nearest_sample_idx = size_t(-1);
    float nearest_d2 = std::numeric_limits<float>::max();

    for (int ring = 0; ring <= max_ring; ++ring) {
        const int min_x = std::max(0, cx - ring);
        const int max_x = std::min(weight_field.bucket_width - 1, cx + ring);
        const int min_y = std::max(0, cy - ring);
        const int max_y = std::min(weight_field.bucket_height - 1, cy + ring);
        for (int by = min_y; by <= max_y; ++by) {
            for (int bx = min_x; bx <= max_x; ++bx) {
                const size_t bucket_idx = size_t(by) * size_t(weight_field.bucket_width) + size_t(bx);
                if (bucket_idx >= weight_field.buckets.size())
                    continue;
                for (const uint32_t sample_idx_u32 : weight_field.buckets[bucket_idx]) {
                    const size_t sample_idx = size_t(sample_idx_u32);
                    if (sample_idx >= weight_field.sample_x_mm.size() || sample_idx >= weight_field.sample_normal_z.size())
                        continue;
                    const float normal_z = weight_field.sample_normal_z[sample_idx];
                    if (!std::isfinite(normal_z))
                        continue;
                    const float dx = x_mm - weight_field.sample_x_mm[sample_idx];
                    const float dy = y_mm - weight_field.sample_y_mm[sample_idx];
                    const float d2 = dx * dx + dy * dy;
                    if (d2 < nearest_d2) {
                        nearest_d2 = d2;
                        nearest_sample_idx = sample_idx;
                    }
                    if (d2 > max_radius_mm * max_radius_mm)
                        continue;
                    const float kernel = std::exp(-0.5f * d2 / std::max(max_radius_mm * max_radius_mm * 0.16f, 1e-6f));
                    const float sample_w = weight_field.sample_weight[sample_idx] * kernel;
                    if (!std::isfinite(sample_w) || sample_w <= EPSILON)
                        continue;
                    weighted_normal_z += double(normal_z) * double(sample_w);
                    total_weight += double(sample_w);
                }
            }
        }
        if (total_weight > EPSILON)
            break;
    }

    if (total_weight > EPSILON)
        return std::clamp(float(weighted_normal_z / total_weight), -1.f, 1.f);
    if (nearest_sample_idx != size_t(-1) && nearest_sample_idx < weight_field.sample_normal_z.size())
        return std::clamp(weight_field.sample_normal_z[nearest_sample_idx], -1.f, 1.f);
    return std::nullopt;
}

static float texture_mapping_offset_context_sample_z_mm(const TextureMappingOffsetContext &context, float z_mm)
{
    if (std::isfinite(z_mm))
        return z_mm;
    if (std::isfinite(context.sample_z_mm))
        return context.sample_z_mm;
    if (context.layer != nullptr && std::isfinite(float(context.layer->print_z)))
        return float(context.layer->print_z);
    return 0.f;
}

static float texture_mapping_offset_linear_gradient_t_at_point(const TextureMappingOffsetContext &context,
                                                               float                              x_mm,
                                                               float                              y_mm,
                                                               float                              z_mm)
{
    const Vec3f sample(x_mm, y_mm, texture_mapping_offset_context_sample_z_mm(context, z_mm));
    if (context.linear_gradient_radial_mode) {
        const float radius = std::max(0.01f, context.linear_gradient_radius_mm);
        return clamp01f((sample - context.linear_gradient_start_mm).norm() / radius);
    }

    const Vec3f gradient_vec = context.linear_gradient_end_mm - context.linear_gradient_start_mm;
    const float denom = gradient_vec.squaredNorm();
    return denom > 1e-8f ? clamp01f((sample - context.linear_gradient_start_mm).dot(gradient_vec) / denom) : 0.f;
}

static std::vector<float> texture_mapping_offset_gradient_component_weights_at_point(const TextureMappingOffsetContext &context,
                                                                                    float                              x_mm,
                                                                                    float                              y_mm,
                                                                                    float                              z_mm,
                                                                                    double                             direction_x,
                                                                                    double                             direction_y)
{
    std::vector<float> weights(context.component_ids.size(), 0.f);
    if (weights.empty())
        return weights;

    if (context.linear_gradient_mode) {
        const float t = texture_mapping_offset_linear_gradient_t_at_point(context, x_mm, y_mm, z_mm);
        return TextureMappingManager::linear_gradient_compact_weights(t, context.linear_gradient_stops, context.component_ids);
    }

    if (context.object_center_mode || !std::isfinite(direction_x) || !std::isfinite(direction_y)) {
        direction_x = double(x_mm) - unscale<double>(context.object_center.x());
        direction_y = double(y_mm) - unscale<double>(context.object_center.y());
    }
    const double direction_len = std::hypot(direction_x, direction_y);
    if (!std::isfinite(direction_len) || direction_len <= EPSILON) {
        direction_x = 1.0;
        direction_y = 0.0;
    } else {
        direction_x /= direction_len;
        direction_y /= direction_len;
    }

    const float theta_deg =
        normalize_texture_mapping_offset_angle_deg(float(Geometry::rad2deg(std::atan2(direction_y, direction_x))));
    const float sample_theta_deg = context.signed_fade_factor < 0.f ?
        normalize_texture_mapping_offset_angle_deg(theta_deg + 180.f) :
        theta_deg;

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = component_angular_influence(context.component_ids[i],
                                                 sample_theta_deg,
                                                 context.component_ids,
                                                 context.rotated_angles);
    }
    return weights;
}

std::vector<float> texture_mapping_offset_component_weights_at_point(const TextureMappingOffsetContext &context,
                                                                     float                              x_mm,
                                                                     float                              y_mm,
                                                                     float                              z_mm)
{
    if (context.vertex_color_match_mode)
        return sample_weight_field_components(context.weight_field,
                                              x_mm,
                                              y_mm,
                                              context.high_resolution_texture_sampling);

    std::vector<float> weights =
        texture_mapping_offset_gradient_component_weights_at_point(context,
                                                                   x_mm,
                                                                   y_mm,
                                                                   z_mm,
                                                                   std::numeric_limits<double>::quiet_NaN(),
                                                                   std::numeric_limits<double>::quiet_NaN());
    if (!context.linear_gradient_mode && context.fade_factor < 1.f - EPSILON)
        for (float &weight : weights)
            weight = clamp01f(1.f + (weight - 1.f) * context.fade_factor);
    return weights;
}

std::optional<std::array<float, 3>> texture_mapping_offset_target_rgb_at_point(const TextureMappingOffsetContext &context,
                                                                               float                              x_mm,
                                                                               float                              y_mm,
                                                                               float                              z_mm)
{
    if (context.vertex_color_match_mode)
        return sample_weight_field_rgb(context.weight_field,
                                       x_mm,
                                       y_mm,
                                       context.high_resolution_texture_sampling);

    if (context.component_colors.empty() || context.component_colors.size() != context.component_ids.size())
        return std::nullopt;

    std::vector<float> weights = texture_mapping_offset_component_weights_at_point(context, x_mm, y_mm, z_mm);
    if (weights.empty())
        return std::nullopt;
    bool has_weight = false;
    for (const float weight : weights) {
        if (std::isfinite(weight) && weight > EPSILON) {
            has_weight = true;
            break;
        }
    }
    if (!has_weight)
        weights.assign(context.component_colors.size(), 1.f);

    const std::array<float, 3> rgb =
        mix_color_solver_components(context.component_colors,
                                    weights,
                                    color_solver_mix_model_from_index(context.generic_solver_mix_model));
    return std::array<float, 3>{ clamp01f(rgb[0]), clamp01f(rgb[1]), clamp01f(rgb[2]) };
}

std::vector<unsigned int> decode_texture_mapping_offset_component_ids(const TextureMappingZone &zone, size_t num_physical)
{
    if (zone.is_linear_gradient())
        return TextureMappingManager::linear_gradient_component_ids_from_stops(zone, num_physical);

    std::vector<unsigned int> out;
    for (const char c : zone.component_ids) {
        if (c < '1' || c > '9')
            continue;
        const unsigned int id = unsigned(c - '0');
        if (id == 0 || id > num_physical)
            continue;
        if (std::find(out.begin(), out.end(), id) == out.end())
            out.emplace_back(id);
    }
    if (out.empty()) {
        if (zone.component_a >= 1 && zone.component_a <= num_physical)
            out.emplace_back(zone.component_a);
        if (zone.component_b >= 1 && zone.component_b <= num_physical &&
            std::find(out.begin(), out.end(), zone.component_b) == out.end())
            out.emplace_back(zone.component_b);
    }
    return out;
}

float normalize_texture_mapping_offset_angle_deg(float angle)
{
    float normalized = std::fmod(angle, 360.f);
    if (normalized < 0.f)
        normalized += 360.f;
    return normalized;
}

float texture_mapping_offset_fade_factor(int fade_mode, float progress01)
{
    const float p = clamp01f(progress01);
    switch (fade_mode) {
    case int(TextureMappingZone::OffsetFadeInUp):          return p;
    case int(TextureMappingZone::OffsetFadeOutUp):         return 1.f - p;
    case int(TextureMappingZone::OffsetFadeInOut):         return 1.f - std::abs(2.f * p - 1.f);
    case int(TextureMappingZone::OffsetFadeOutIn):         return std::abs(2.f * p - 1.f);
    case int(TextureMappingZone::OffsetFadeOutInReversed): return 1.f - 2.f * p;
    default:                                               return 1.f;
    }
}

float texture_mapping_offset_filament_strength_factor(const TextureMappingZone &zone, unsigned int physical_filament_id)
{
    if (physical_filament_id == 0)
        return 1.f;
    const size_t idx = size_t(physical_filament_id - 1);
    if (idx >= zone.filament_strengths_pct.size())
        return 1.f;
    const float strength_pct = zone.filament_strengths_pct[idx];
    if (!std::isfinite(strength_pct))
        return 1.f;
    return std::clamp(strength_pct / 100.f, 0.f, 1.f);
}

float texture_mapping_offset_filament_minimum_offset_factor(const TextureMappingZone &zone, unsigned int physical_filament_id)
{
    if (physical_filament_id == 0)
        return 0.f;
    const size_t idx = size_t(physical_filament_id - 1);
    if (idx >= zone.filament_minimum_offsets_pct.size())
        return 0.f;
    const float minimum_offset_pct = zone.filament_minimum_offsets_pct[idx];
    if (!std::isfinite(minimum_offset_pct))
        return 0.f;
    return std::clamp(minimum_offset_pct / 100.f, 0.f, 1.f);
}

std::optional<TextureMappingOffsetContext> build_texture_mapping_offset_context_for_layer(
    const PrintObject       &print_object,
    const Layer             &layer,
    const TextureMappingZone &zone,
    unsigned int             texture_zone_id,
    unsigned int             active_component_id_override,
    std::optional<float>     base_outer_width_mm_override,
    std::optional<float>     layer_height_mm_override,
    std::optional<Vec2d>     plate_origin_mm_override,
    std::optional<float>     min_outer_width_mm_override,
    std::optional<std::array<float, 4>> image_background_rgba_override,
    std::optional<float>     sample_z_mm_override)
{
    const Print *print = print_object.print();
    if (print == nullptr)
        return std::nullopt;

    const PrintConfig &print_config = print->config();
    const TextureMappingManager &texture_mgr = print->texture_mapping_manager();
    const size_t num_physical = print_config.filament_colour.values.size();
    if (num_physical == 0)
        return std::nullopt;

    const bool vertex_color_match_mode = zone.is_image_texture();
    const bool linear_gradient_mode = zone.is_linear_gradient();
    std::vector<unsigned int> component_ids = decode_texture_mapping_offset_component_ids(zone, num_physical);
    if (vertex_color_match_mode) {
        const std::vector<unsigned int> effective_component_ids =
            TextureMappingManager::effective_texture_component_ids(zone, num_physical, print_config.filament_colour.values);
        if (!effective_component_ids.empty())
            component_ids = effective_component_ids;
    }
    if (component_ids.empty())
        return std::nullopt;

    const int layer_index = int(layer.id());
    const int object_layer_count = int(print_object.layer_count());
    const float z_progress = object_layer_count > 1 ?
        std::clamp(float(layer_index) / float(object_layer_count - 1), 0.f, 1.f) :
        0.f;
    const unsigned int active_component_id = active_component_id_override != 0 ?
        active_component_id_override :
        texture_mgr.resolve_zone_component(texture_zone_id, num_physical, layer_index);
    const auto active_component_it = std::find(component_ids.begin(), component_ids.end(), active_component_id);
    if (active_component_it == component_ids.end())
        return std::nullopt;
    const size_t active_component_idx = size_t(active_component_it - component_ids.begin());

    const bool raw_texture_mapping_mode =
        zone.texture_mapping_mode == int(TextureMappingZone::TextureMappingRawValues);
    const int filament_color_mode = std::clamp(zone.filament_color_mode,
                                               int(TextureMappingZone::FilamentColorAny),
                                               int(TextureMappingZone::FilamentColorRGBKW));
    const int generic_solver_lookup_mode = std::clamp(zone.generic_solver_lookup_mode,
                                                      int(TextureMappingZone::GenericSolverClosestMix),
                                                      int(TextureMappingZone::GenericSolverBlendClosestTwo));
    const int generic_solver_mode = std::clamp(zone.generic_solver_mode,
                                               int(TextureMappingZone::GenericSolverRGB),
                                               int(TextureMappingZone::GenericSolverOklabSoftCap4Dark4));
    const int generic_solver_mix_model = std::clamp(zone.generic_solver_mix_model,
                                                    int(TextureMappingZone::GenericSolverPigmentPainter),
                                                    int(TextureMappingZone::GenericSolverPrusaFdmMixer));
    const float texture_contrast_pct = std::clamp(zone.contrast_pct, 25.f, 300.f);
    const float texture_tone_gamma =
        (!std::isfinite(zone.tone_gamma) || zone.tone_gamma <= 0.f) ?
            1.f :
            std::clamp(zone.tone_gamma, 0.5f, 3.f);

    std::vector<std::array<float, 3>> component_colors;
    component_colors.reserve(component_ids.size());
    bool missing_component_color = false;
    for (const unsigned int id : component_ids) {
        if (id < 1 || id > print_config.filament_colour.values.size()) {
            if (raw_texture_mapping_mode)
                component_colors.push_back({ 0.f, 0.f, 0.f });
            else
                missing_component_color = true;
            continue;
        }
        ColorRGB decoded;
        if (!decode_color(print_config.filament_colour.get_at(size_t(id - 1)), decoded)) {
            if (raw_texture_mapping_mode)
                component_colors.push_back({ 0.f, 0.f, 0.f });
            else
                missing_component_color = true;
            continue;
        }
        component_colors.push_back({ decoded.r(), decoded.g(), decoded.b() });
    }
    if (vertex_color_match_mode &&
        (missing_component_color || component_colors.size() != component_ids.size() || component_colors.empty()))
        return std::nullopt;

    std::vector<float> reference_nozzles;
    reference_nozzles.reserve(component_ids.size() + 2);
    auto append_nozzle = [&reference_nozzles, &print_config](unsigned int component_id) {
        if (component_id == 0)
            return;
        const size_t idx = size_t(component_id - 1);
        if (idx < print_config.nozzle_diameter.values.size())
            reference_nozzles.emplace_back(float(print_config.nozzle_diameter.get_at(idx)));
    };
    for (unsigned int id : component_ids)
        append_nozzle(id);
    append_nozzle(zone.component_a);
    append_nozzle(zone.component_b);

    const float reference_nozzle = reference_nozzles.empty() ?
        float(print_config.nozzle_diameter.values.empty() ? 0.4 : print_config.nozzle_diameter.values.front()) :
        std::accumulate(reference_nozzles.begin(), reference_nozzles.end(), 0.f) / float(reference_nozzles.size());
    const float max_allowed_distance_mm = TextureMappingManager::max_component_surface_offset_mm(reference_nozzle);
    if (max_allowed_distance_mm <= EPSILON)
        return std::nullopt;

    std::vector<float> distances_mm = TextureMappingManager::effective_offset_distances(zone, component_ids.size(), reference_nozzle);
    std::vector<float> angles_deg = TextureMappingManager::effective_offset_angles(zone, component_ids.size());
    if (distances_mm.size() != component_ids.size())
        distances_mm.assign(component_ids.size(), 0.f);
    if (angles_deg.size() != component_ids.size())
        angles_deg = TextureMappingManager::default_offset_angles(component_ids.size());
    for (float &a : angles_deg)
        a = normalize_texture_mapping_offset_angle_deg(a);

    bool has_nonzero_distance = false;
    if (vertex_color_match_mode || linear_gradient_mode) {
        distances_mm.assign(component_ids.size(), max_allowed_distance_mm);
        has_nonzero_distance = max_allowed_distance_mm > EPSILON;
    } else {
        for (float &d : distances_mm) {
            d = std::clamp(d, 0.f, max_allowed_distance_mm);
            has_nonzero_distance = has_nonzero_distance || d > EPSILON;
        }
    }
    if (!has_nonzero_distance)
        return std::nullopt;

    const float global_strength_factor =
        std::clamp(float(print_config.texture_mapping_outer_wall_gradient_global_strength.value) / 100.f, 0.f, 1.f);
    if (global_strength_factor <= EPSILON)
        return std::nullopt;

    const float base_outer_width_mm = base_outer_width_mm_override ?
        std::max(0.05f, *base_outer_width_mm_override) :
        std::max(0.05f, float(print_config.texture_mapping_outer_wall_gradient_max_line_width.value));
    const float config_min_gradient_width_mm = min_outer_width_mm_override ?
        std::clamp(*min_outer_width_mm_override, 0.05f, base_outer_width_mm) :
        std::clamp(float(print_config.texture_mapping_outer_wall_gradient_min_line_width.value),
                   0.05f,
                   base_outer_width_mm);
    const float layer_height_mm = layer_height_mm_override ?
        std::max(0.01f, *layer_height_mm_override) :
        std::max(0.01f, float(layer.height));
    const float sample_z_mm = sample_z_mm_override && std::isfinite(*sample_z_mm_override) ?
        *sample_z_mm_override :
        float(layer.print_z);
    const float min_width_for_positive_spacing_mm = layer_height_mm * float(1. - 0.25 * PI) + 1e-4f;
    const float safe_min_gradient_width_mm = std::clamp(
        std::max(config_min_gradient_width_mm, min_width_for_positive_spacing_mm),
        0.05f,
        base_outer_width_mm);
    const float max_width_delta_mm = std::max(0.f, base_outer_width_mm - safe_min_gradient_width_mm);
    const float effective_max_width_delta_mm = max_width_delta_mm * global_strength_factor;
    const float max_width_delta_limit_mm = std::min(effective_max_width_delta_mm, 2.f * max_allowed_distance_mm);
    if (!std::isfinite(max_width_delta_limit_mm) || max_width_delta_limit_mm <= EPSILON)
        return std::nullopt;

    const bool dithering_enabled = zone.dithering_enabled && !raw_texture_mapping_mode;
    const int dithering_method = std::clamp(zone.dithering_method,
                                            int(TextureMappingZone::DitheringClosest),
                                            int(TextureMappingZone::DitheringHalftoneV2));
    const bool halftone_dithering_enabled =
        dithering_enabled && is_halftone_dithering_method(dithering_method);
    const bool halftone_increased_detail_enabled =
        dithering_enabled && dithering_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail);
    const bool halftone_v2_enabled =
        dithering_enabled && dithering_method == int(TextureMappingZone::DitheringHalftoneV2);
    const float dither_pitch_mm =
        dither_pitch(base_outer_width_mm,
                     dithering_method,
                     zone.dithering_resolution_mm,
                     zone.halftone_dot_size_mm);
    const float dither_cell_size_mm = dither_cell_size(zone.dithering_resolution_mm);
    const float halftone_dot_size_mm = std::clamp(zone.halftone_dot_size_mm,
                                                  TextureMappingZone::MinHalftoneDotSizeMm,
                                                  TextureMappingZone::MaxHalftoneDotSizeMm);
    const bool high_resolution_texture_sampling = zone.high_resolution_sampling || dithering_enabled;
    const bool compact_offset_mode = halftone_dithering_enabled ? false : zone.compact_offset_mode || dithering_enabled;
    std::vector<float> component_strength_factors;
    component_strength_factors.reserve(component_ids.size());
    std::vector<float> component_minimum_offset_factors;
    component_minimum_offset_factors.reserve(component_ids.size());
    for (const unsigned int id : component_ids) {
        component_strength_factors.emplace_back(texture_mapping_offset_filament_strength_factor(zone, id));
        component_minimum_offset_factors.emplace_back(texture_mapping_offset_filament_minimum_offset_factor(zone, id));
    }

    TextureMappingOffsetWeightField weight_field;
    if (vertex_color_match_mode) {
        const float layer_sample_falloff_mm = high_resolution_texture_sampling ?
            std::max(0.03f, layer_height_mm * 0.5f) :
            std::max(0.12f, layer_height_mm * 1.5f);
        weight_field = build_texture_mapping_offset_weight_field(print_object,
                                                           component_colors,
                                                           raw_texture_mapping_mode,
                                                           filament_color_mode,
                                                           zone.force_sequential_filaments,
                                                           generic_solver_lookup_mode,
                                                           generic_solver_mode,
                                                           generic_solver_mix_model,
                                                           zone.use_legacy_fixed_color_mode,
                                                           dithering_enabled,
                                                           dithering_method,
                                                           dither_pitch_mm,
                                                           dither_cell_size_mm,
                                                           component_strength_factors,
                                                           component_minimum_offset_factors,
                                                           texture_contrast_pct,
                                                           texture_tone_gamma,
                                                           sample_z_mm,
                                                           layer_sample_falloff_mm,
                                                           high_resolution_texture_sampling,
                                                           zone.high_speed_image_texture_sampling,
                                                           image_background_rgba_override);
        if (weight_field.empty())
            return std::nullopt;
    }

    const TransmissionDistanceCalibrationContext td_calibration_context =
        transmission_distance_calibration_context(zone, component_ids, component_colors, filament_color_mode);
    size_t previous_component_idx = size_t(-1);
    if (layer_index > 0) {
        const unsigned int previous_component_id =
            texture_mgr.resolve_zone_component(texture_zone_id, num_physical, layer_index - 1);
        const auto previous_component_it = std::find(component_ids.begin(), component_ids.end(), previous_component_id);
        if (previous_component_it != component_ids.end())
            previous_component_idx = size_t(previous_component_it - component_ids.begin());
    }

    float rotation_deg = 0.f;
    if (!linear_gradient_mode && zone.offset_rotation_enabled) {
        const float p = clamp01f(z_progress);
        const float r = std::max(1.f, zone.offset_repeats);
        float repeated_pos = p * r;
        int segment_idx = int(std::floor(repeated_pos));
        float local = repeated_pos - float(segment_idx);
        if (p >= 1.f - EPSILON) {
            segment_idx = std::max(0, int(std::ceil(r)) - 1);
            local = 1.f;
        }
        if (zone.offset_reverse_repeats && (segment_idx % 2 == 1))
            local = 1.f - local;
        const float direction = zone.offset_clockwise ? -1.f : 1.f;
        rotation_deg = direction * 360.f * zone.offset_rotations * clamp01f(local);
    }
    std::vector<float> rotated_angles = angles_deg;
    for (float &a : rotated_angles)
        a = normalize_texture_mapping_offset_angle_deg(a + rotation_deg);

    const float signed_fade_factor = linear_gradient_mode ? 1.f : texture_mapping_offset_fade_factor(zone.offset_fade_mode, z_progress);
    const float fade_factor = std::abs(signed_fade_factor);
    if (fade_factor <= EPSILON)
        return std::nullopt;

    TextureMappingOffsetContext context;
    context.vertex_color_match_mode = vertex_color_match_mode;
    context.linear_gradient_mode = linear_gradient_mode;
    context.object_center_mode =
        !vertex_color_match_mode &&
        !linear_gradient_mode &&
        zone.offset_angle_mode != int(TextureMappingZone::OffsetAngleSurfaceNormal);
    context.high_resolution_texture_sampling = high_resolution_texture_sampling;
    context.compact_offset_mode = compact_offset_mode || linear_gradient_mode;
    context.dithering_enabled = dithering_enabled;
    context.halftone_dithering_enabled = halftone_dithering_enabled;
    context.halftone_increased_detail_enabled = halftone_increased_detail_enabled;
    context.halftone_v2_enabled = halftone_v2_enabled;
    context.nonlinear_offset_adjustment = zone.nonlinear_offset_adjustment;
    context.generic_solver_mix_model = generic_solver_mix_model;
    context.object_center = print_object.bounding_box().center();
    if (linear_gradient_mode) {
        const bool radial_mode = zone.linear_gradient_mode == int(TextureMappingZone::LinearGradientRadial);
        auto default_points = texture_mapping_default_linear_gradient_points(print_object);
        auto default_radial = texture_mapping_default_radial_gradient(print_object);
        std::optional<Vec3f> start =
            texture_mapping_linear_gradient_anchor_print_point(print_object,
                                                               zone.linear_gradient_start,
                                                               plate_origin_mm_override);
        std::optional<Vec3f> end;
        if (!radial_mode)
            end = texture_mapping_linear_gradient_anchor_print_point(print_object,
                                                                     zone.linear_gradient_end,
                                                                     plate_origin_mm_override);
        context.linear_gradient_radial_mode = radial_mode;
        if (radial_mode) {
            context.linear_gradient_start_mm = start ? *start : default_radial.first;
            context.linear_gradient_radius_mm = texture_mapping_linear_gradient_radius_mm(print_object, zone, default_radial.second);
            context.linear_gradient_end_mm = context.linear_gradient_start_mm + Vec3f::UnitX() * context.linear_gradient_radius_mm;
        } else {
            context.linear_gradient_start_mm = start ? *start : default_points.first;
            context.linear_gradient_end_mm = end ? *end : default_points.second;
            if ((context.linear_gradient_end_mm - context.linear_gradient_start_mm).squaredNorm() <= 1e-8f) {
                context.linear_gradient_start_mm = default_points.first;
                context.linear_gradient_end_mm = default_points.second;
            }
        }
    }
    context.active_component_id = active_component_id;
    context.active_component_idx = active_component_idx;
    context.component_ids = std::move(component_ids);
    context.component_colors = std::move(component_colors);
    if (linear_gradient_mode)
        context.linear_gradient_stops = TextureMappingManager::normalized_linear_gradient_stops(zone, num_physical);
    context.component_distances_mm = std::move(distances_mm);
    context.rotated_angles = std::move(rotated_angles);
    context.weight_field = std::move(weight_field);
    context.inset_strength_reference_mm = max_allowed_distance_mm;
    context.signed_fade_factor = signed_fade_factor;
    context.fade_factor = fade_factor;
    context.max_width_delta_mm = max_width_delta_limit_mm;
    context.dither_pitch_mm = dither_pitch_mm;
    context.halftone_dot_size_mm = halftone_dot_size_mm;
    context.active_halftone_angle_deg = halftone_screen_angle_deg(filament_color_mode, active_component_idx);
    context.active_component_strength_factor = linear_gradient_mode ?
        1.f :
        texture_mapping_offset_filament_strength_factor(zone, active_component_id);
    context.active_component_minimum_offset_factor = linear_gradient_mode ?
        0.f :
        texture_mapping_offset_filament_minimum_offset_factor(zone, active_component_id);
    context.active_component_td_width_factor = linear_gradient_mode ?
        1.f :
        transmission_distance_width_factor(td_calibration_context, active_component_idx, previous_component_idx);
    context.base_outer_width_mm = base_outer_width_mm;
    context.layer_height_mm = layer_height_mm;
    context.sample_z_mm = sample_z_mm;
    context.layer = &layer;
    return context;
}

float texture_mapping_offset_surface_inset_mm(const TextureMappingOffsetContext &context,
                                        const Point                  &point,
                                        double                        inward_x,
                                        double                        inward_y,
                                        float                         surface_u_mm,
                                        float                         surface_v_mm)
{
    if (!std::isfinite(inward_x) || !std::isfinite(inward_y))
        return 0.f;

    const double outward_x = -inward_x;
    const double outward_y = -inward_y;
    float inset_strength = 0.f;
    if (context.vertex_color_match_mode) {
        float desired_strength =
            sample_weight_field(context.weight_field,
                                unscale<float>(point.x()),
                                unscale<float>(point.y()),
                                context.active_component_idx,
                                context.high_resolution_texture_sampling,
                                context.compact_offset_mode);
        if (context.halftone_dithering_enabled) {
            const float x_mm = unscale<float>(point.x());
            const float y_mm = unscale<float>(point.y());
            float z_mm = context.layer != nullptr ? float(context.layer->print_z) : 0.f;
            if (!std::isfinite(z_mm))
                z_mm = 0.f;
            float halftone_u_mm = surface_u_mm;
            float halftone_v_mm = surface_v_mm;
            float coverage = desired_strength;
            if (context.halftone_v2_enabled) {
                const int surface_axis = halftone_surface_axis(float(outward_x), float(outward_y), 0.f);
                const std::array<float, 2> surface_coord = halftone_surface_coords(x_mm, y_mm, z_mm, surface_axis);
                halftone_u_mm = surface_coord[0];
                halftone_v_mm = surface_coord[1];
                const std::array<float, 2> cell_center =
                    halftone_cell_center_surface_coords(halftone_u_mm,
                                                        halftone_v_mm,
                                                        context.halftone_dot_size_mm,
                                                        context.active_halftone_angle_deg);
                const std::array<float, 2> sample_xy =
                    halftone_sample_xy_from_surface_coords(x_mm, y_mm, cell_center, surface_axis);
                if (std::isfinite(sample_xy[0]) && std::isfinite(sample_xy[1]))
                    coverage = sample_weight_field(context.weight_field,
                                                   sample_xy[0],
                                                   sample_xy[1],
                                                   context.active_component_idx,
                                                   context.high_resolution_texture_sampling,
                                                   context.compact_offset_mode);
            } else {
                if (!std::isfinite(halftone_u_mm))
                    halftone_u_mm = std::abs(outward_x) >= std::abs(outward_y) ? y_mm : x_mm;
                if (!std::isfinite(halftone_v_mm))
                    halftone_v_mm = z_mm;
            }
            desired_strength = halftone_screen_on(coverage,
                                                  halftone_u_mm,
                                                  halftone_v_mm,
                                                  context.halftone_dot_size_mm,
                                                  context.active_halftone_angle_deg) ?
                1.f :
                0.f;
        }
        inset_strength = std::clamp(1.f - desired_strength, 0.f, 1.f);
    } else if (context.linear_gradient_mode) {
        const std::vector<float> weights =
            texture_mapping_offset_gradient_component_weights_at_point(context,
                                                                       unscale<float>(point.x()),
                                                                       unscale<float>(point.y()),
                                                                       surface_v_mm,
                                                                       outward_x,
                                                                       outward_y);
        const float desired_strength = context.active_component_idx < weights.size() ?
            clamp01f(weights[context.active_component_idx]) :
            0.f;
        inset_strength = std::clamp(1.f - desired_strength, 0.f, 1.f);
    } else {
        const std::vector<float> weights =
            texture_mapping_offset_gradient_component_weights_at_point(context,
                                                                       unscale<float>(point.x()),
                                                                       unscale<float>(point.y()),
                                                                       surface_v_mm,
                                                                       outward_x,
                                                                       outward_y);
        float raw_inset_mm = 0.f;
        const size_t component_count = std::min(weights.size(), context.component_distances_mm.size());
        for (size_t i = 0; i < component_count; ++i) {
            if (i == context.active_component_idx)
                continue;
            raw_inset_mm += context.component_distances_mm[i] * weights[i];
        }
        inset_strength = std::clamp(raw_inset_mm / std::max(context.inset_strength_reference_mm, float(EPSILON)), 0.f, 1.f);
    }

    inset_strength = std::clamp(inset_strength * context.fade_factor, 0.f, 1.f);
    const float stair_step_mm = context.nonlinear_offset_adjustment ?
        local_surface_stair_step_distance(context.layer,
                                          point,
                                          outward_x,
                                          outward_y,
                                          context.base_outer_width_mm,
                                          context.inset_strength_reference_mm) :
        std::numeric_limits<float>::quiet_NaN();
    return variable_width_delta_for_visibility_range(inset_strength,
                                                     context.max_width_delta_mm,
                                                     context.active_component_minimum_offset_factor,
                                                     context.active_component_strength_factor,
                                                     context.active_component_td_width_factor,
                                                     context.nonlinear_offset_adjustment,
                                                     context.layer_height_mm,
                                                     stair_step_mm);
}

} // namespace Slic3r
