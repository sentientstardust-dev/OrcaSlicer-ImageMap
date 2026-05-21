#include <glad/gl.h>

#include "MMUPaintedTexturePreview.hpp"

#include "3DScene.hpp"
#include "BitmapCache.hpp"
#include "GLShader.hpp"
#include "GUI_App.hpp"

#include "libslic3r/Config.hpp"
#include "libslic3r/Geometry.hpp"
#include "libslic3r/ImageMapRawFilamentOffsetAtlas.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/TextureMapping.hpp"
#include "libslic3r/ColorSolver.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>

namespace Slic3r {

namespace {

constexpr float k_preview_offset = 0.001f;
constexpr float k_preview_clip_padding = 2.f * k_preview_offset;
constexpr float k_polygon_offset_factor = -1.f;
constexpr float k_polygon_offset_units = -1.f;
constexpr float k_epsilon = 1e-6f;
constexpr unsigned int k_simulated_texture_preview_max_edge = 2048;
constexpr size_t k_simulated_texture_preview_max_pixels = 2048ull * 2048ull;
constexpr unsigned int k_temporary_simulated_texture_preview_max_edge = 1024;
constexpr size_t k_temporary_simulated_texture_preview_max_pixels = 1024ull * 1024ull;
constexpr size_t k_temporary_simulated_texture_preview_min_source_pixels = k_temporary_simulated_texture_preview_max_pixels * 3 / 2;
constexpr const char *TEXTURE_MAPPING_BACKGROUND_COLOR_CONFIG_KEY = "texture_mapping_background_color";

struct TexturePreviewMixCandidate
{
    std::array<float, 3> rgb;
    std::array<float, 3> perceptual;
    std::vector<float> weights;
};

struct TexturePreviewMixCandidateKdNode
{
    uint32_t candidate_idx { 0 };
    int left { -1 };
    int right { -1 };
    uint8_t axis { 0 };
};

struct TexturePreviewGenericMixCandidateCache
{
    std::once_flag init;
    ColorSolverCandidateSet candidates;
};

struct TexturePreviewSimulationSettings
{
    int mapping_mode = int(TextureMappingZone::TextureMappingFilamentBlending);
    int filament_color_mode = TextureMappingZone::DefaultFilamentColorMode;
    bool force_sequential_filaments = false;
    bool limit_texture_resolution = true;
    bool compact_offset_mode = false;
    bool use_legacy_fixed_color_mode = false;
    bool use_fixed_color_generic_solver = false;
    bool dithering_enabled = false;
    int dithering_method = TextureMappingZone::DefaultDitheringMethod;
    float dithering_resolution_mm = TextureMappingZone::DefaultDitheringResolutionMm;
    float halftone_dot_size_mm = TextureMappingZone::DefaultHalftoneDotSizeMm;
    float texture_preview_mm_per_pixel = TextureMappingZone::DefaultDitheringResolutionMm;
    float contrast_pct = 100.f;
    float tone_gamma = 1.f;
    int generic_solver_lookup_mode = int(TextureMappingZone::GenericSolverClosestMix);
    int generic_solver_mode = int(TextureMappingZone::GenericSolverV2);
    int generic_solver_mix_model = TextureMappingZone::DefaultGenericSolverMixModel;
    float minimum_visibility_offset_factor = 0.f;
    std::vector<unsigned int> component_ids;
    std::vector<std::array<float, 3>> component_colors;
    std::vector<float> component_strength_factors;
    std::vector<float> component_minimum_offset_factors;
    std::vector<size_t> semantic_component_indices;
    std::shared_ptr<TexturePreviewGenericMixCandidateCache> generic_mix_candidate_cache;
    float raw_offset_base_visibility_factor = 1.f;
    float raw_offset_visibility_range_factor = 0.f;
};

struct SurfaceGradientPreviewSettings
{
    std::vector<unsigned int> component_ids;
    std::vector<std::array<float, 3>> component_colors;
    std::vector<float> distances_mm;
    std::vector<float> angles_deg;
    std::vector<float> strength_factors;
    std::vector<float> minimum_offset_factors;
    float max_component_distance_mm = 0.f;
    float max_width_delta_limit_mm = 0.f;
    int angle_mode = int(TextureMappingZone::OffsetAngleObjectCenter);
    bool rotation_enabled = true;
    float rotations = 1.f;
    float repeats = 1.f;
    bool reverse_repeats = true;
    bool clockwise = true;
    int fade_mode = int(TextureMappingZone::OffsetFadeNone);
    bool limit_texture_resolution = true;
    Vec3f center = Vec3f::Zero();
    float z_min = 0.f;
    float z_max = 0.f;
};

struct TexturePreviewSimulationResult
{
    size_t signature { 0 };
    unsigned int width { 0 };
    unsigned int height { 0 };
    std::vector<unsigned char> rgba;
};

struct TexturePreviewSimulationJobState
{
    std::mutex mutex;
    std::optional<TexturePreviewSimulationResult> temporary_result;
};

struct TexturePreviewSimulationCacheEntry
{
    std::unique_ptr<GUI::GLTexture> texture;
    size_t uploaded_signature { 0 };
    size_t pending_signature { 0 };
    size_t temporary_pending_signature { 0 };
    bool ready_frame_requested { false };
    std::future<TexturePreviewSimulationResult> pending_future;
    std::shared_ptr<TexturePreviewSimulationJobState> pending_job_state;
};

struct TexturePreviewVertexColorSimulationResult
{
    size_t signature { 0 };
    GUI::GLModel::Geometry geometry;
};

struct TexturePreviewVertexColorSimulationCacheEntry
{
    size_t ready_signature { 0 };
    size_t pending_signature { 0 };
    bool ready_generation_reported { false };
    std::optional<GUI::GLModel::Geometry> ready_geometry;
    std::future<TexturePreviewVertexColorSimulationResult> pending_future;
};

bool model_volume_has_texture_preview_data_impl(const ModelVolume &model_volume)
{
    return !model_volume.imported_texture_rgba.empty() &&
           model_volume.imported_texture_width > 0 &&
           model_volume.imported_texture_height > 0 &&
           model_volume.imported_texture_uv_valid.size() == model_volume.mesh().its.indices.size() &&
           model_volume.imported_texture_uvs_per_face.size() >= model_volume.mesh().its.indices.size() * 6 &&
           model_volume.imported_texture_rgba.size() >=
               size_t(model_volume.imported_texture_width) * size_t(model_volume.imported_texture_height) * 4;
}

bool model_volume_has_complete_texture_preview_data_impl(const ModelVolume &model_volume)
{
    return model_volume_has_texture_preview_data_impl(model_volume) &&
           std::all_of(model_volume.imported_texture_uv_valid.begin(), model_volume.imported_texture_uv_valid.end(), [](uint8_t valid) {
               return valid != 0;
           });
}

bool model_volume_has_vertex_color_preview_data_impl(const ModelVolume &model_volume)
{
    return !model_volume.imported_vertex_colors_rgba.empty() &&
           model_volume.imported_vertex_colors_rgba.size() == model_volume.mesh().its.vertices.size();
}

bool model_volume_has_texture_mapping_color_preview_data_impl(const ModelVolume &model_volume)
{
    return !model_volume.texture_mapping_color_facets.empty();
}

std::array<Vec2f, 3> unwrap_triangle_uvs(const Vec2f &uv0, const Vec2f &uv1, const Vec2f &uv2)
{
    std::array<Vec2f, 3> out{ uv0, uv1, uv2 };

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

float estimated_texture_preview_mm_per_pixel(const ModelVolume &model_volume)
{
    const unsigned int width = model_volume.imported_texture_width;
    const unsigned int height = model_volume.imported_texture_height;
    if (width == 0 || height == 0)
        return TextureMappingZone::DefaultDitheringResolutionMm;

    const indexed_triangle_set &its = model_volume.mesh().its;
    if (its.indices.empty() || its.vertices.empty())
        return TextureMappingZone::DefaultDitheringResolutionMm;

    double world_area_mm2 = 0.0;
    double uv_area_px2 = 0.0;
    const size_t triangle_count = std::min(its.indices.size(), model_volume.imported_texture_uv_valid.size());
    for (size_t tri_idx = 0; tri_idx < triangle_count; ++tri_idx) {
        if (model_volume.imported_texture_uv_valid[tri_idx] == 0)
            continue;

        const size_t uv_offset = tri_idx * 6;
        if (uv_offset + 5 >= model_volume.imported_texture_uvs_per_face.size())
            continue;

        const stl_triangle_vertex_indices &indices = its.indices[tri_idx];
        if (indices[0] < 0 || indices[1] < 0 || indices[2] < 0 ||
            size_t(indices[0]) >= its.vertices.size() ||
            size_t(indices[1]) >= its.vertices.size() ||
            size_t(indices[2]) >= its.vertices.size())
            continue;

        const Vec3d p0 = its.vertices[size_t(indices[0])].cast<double>();
        const Vec3d p1 = its.vertices[size_t(indices[1])].cast<double>();
        const Vec3d p2 = its.vertices[size_t(indices[2])].cast<double>();
        const double tri_area_mm2 = 0.5 * (p1 - p0).cross(p2 - p0).norm();
        if (!std::isfinite(tri_area_mm2) || tri_area_mm2 <= double(k_epsilon))
            continue;

        const std::array<Vec2f, 3> uv = unwrap_triangle_uvs(
            Vec2f(model_volume.imported_texture_uvs_per_face[uv_offset + 0], model_volume.imported_texture_uvs_per_face[uv_offset + 1]),
            Vec2f(model_volume.imported_texture_uvs_per_face[uv_offset + 2], model_volume.imported_texture_uvs_per_face[uv_offset + 3]),
            Vec2f(model_volume.imported_texture_uvs_per_face[uv_offset + 4], model_volume.imported_texture_uvs_per_face[uv_offset + 5]));
        const Vec2d t0(double(uv[0].x()) * double(width), double(uv[0].y()) * double(height));
        const Vec2d t1(double(uv[1].x()) * double(width), double(uv[1].y()) * double(height));
        const Vec2d t2(double(uv[2].x()) * double(width), double(uv[2].y()) * double(height));
        const Vec2d e0 = t1 - t0;
        const Vec2d e1 = t2 - t0;
        const double tri_uv_area_px2 = 0.5 * std::abs(e0.x() * e1.y() - e0.y() * e1.x());
        if (!std::isfinite(tri_uv_area_px2) || tri_uv_area_px2 <= double(k_epsilon))
            continue;

        world_area_mm2 += tri_area_mm2;
        uv_area_px2 += tri_uv_area_px2;
    }

    if (world_area_mm2 <= double(k_epsilon) || uv_area_px2 <= double(k_epsilon))
        return TextureMappingZone::DefaultDitheringResolutionMm;

    const double mm_per_pixel = std::sqrt(world_area_mm2 / uv_area_px2);
    if (!std::isfinite(mm_per_pixel) || mm_per_pixel <= 0.0)
        return TextureMappingZone::DefaultDitheringResolutionMm;

    return std::clamp(float(mm_per_pixel), 0.005f, 5.f);
}

bool barycentric_weights(const Vec3f &point, const Vec3f &p0, const Vec3f &p1, const Vec3f &p2, Vec3f &weights)
{
    const Vec3f edge_0 = p1 - p0;
    const Vec3f edge_1 = p2 - p0;
    const Vec3f delta = point - p0;
    const float d00 = edge_0.dot(edge_0);
    const float d01 = edge_0.dot(edge_1);
    const float d11 = edge_1.dot(edge_1);
    const float d20 = delta.dot(edge_0);
    const float d21 = delta.dot(edge_1);
    const float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) <= k_epsilon)
        return false;

    weights.y() = (d11 * d20 - d01 * d21) / denom;
    weights.z() = (d00 * d21 - d01 * d20) / denom;
    weights.x() = 1.f - weights.y() - weights.z();
    return std::isfinite(weights.x()) && std::isfinite(weights.y()) && std::isfinite(weights.z());
}

bool barycentric_weights_2d(const Vec2f &point, const Vec2f &p0, const Vec2f &p1, const Vec2f &p2, Vec3f &weights)
{
    const Vec2f edge_0 = p1 - p0;
    const Vec2f edge_1 = p2 - p0;
    const Vec2f delta = point - p0;
    const float d00 = edge_0.dot(edge_0);
    const float d01 = edge_0.dot(edge_1);
    const float d11 = edge_1.dot(edge_1);
    const float d20 = delta.dot(edge_0);
    const float d21 = delta.dot(edge_1);
    const float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) <= k_epsilon)
        return false;

    weights.y() = (d11 * d20 - d01 * d21) / denom;
    weights.z() = (d00 * d21 - d01 * d20) / denom;
    weights.x() = 1.f - weights.y() - weights.z();
    return std::isfinite(weights.x()) && std::isfinite(weights.y()) && std::isfinite(weights.z());
}

ColorRGBA unpack_vertex_color(uint32_t packed)
{
    return {
        float((packed >> 24) & 0xFF) / 255.f,
        float((packed >> 16) & 0xFF) / 255.f,
        float((packed >> 8) & 0xFF) / 255.f,
        float(packed & 0xFF) / 255.f
    };
}

ColorRGBA interpolate_color(const std::array<ColorRGBA, 3> &colors, const Vec3f &weights)
{
    Vec3f clamped(std::max(0.f, weights.x()), std::max(0.f, weights.y()), std::max(0.f, weights.z()));
    const float sum = clamped.x() + clamped.y() + clamped.z();
    if (sum > k_epsilon)
        clamped /= sum;
    else
        clamped = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);

    return {
        colors[0].r() * clamped.x() + colors[1].r() * clamped.y() + colors[2].r() * clamped.z(),
        colors[0].g() * clamped.x() + colors[1].g() * clamped.y() + colors[2].g() * clamped.z(),
        colors[0].b() * clamped.x() + colors[1].b() * clamped.y() + colors[2].b() * clamped.z(),
        colors[0].a() * clamped.x() + colors[1].a() * clamped.y() + colors[2].a() * clamped.z()
    };
}

std::array<float, 3> decode_color(const std::string &color)
{
    unsigned char rgba[4] = { 38, 166, 154, 255 };
    GUI::BitmapCache::parse_color4(color, rgba);
    return {
        float(rgba[0]) / 255.f,
        float(rgba[1]) / 255.f,
        float(rgba[2]) / 255.f
    };
}

ColorRGBA blend_component_colors(const std::vector<std::array<float, 3>> &colors, const std::vector<float> &weights)
{
    if (colors.empty() || weights.empty())
        return { 0.15f, 0.65f, 0.6f, 1.f };

    const size_t count = std::min(colors.size(), weights.size());
    if (count == 0)
        return { 0.15f, 0.65f, 0.6f, 1.f };

    float total = 0.f;
    for (size_t idx = 0; idx < count; ++idx)
        total += std::max(0.f, weights[idx]);
    if (total <= k_epsilon)
        return { colors.front()[0], colors.front()[1], colors.front()[2], 1.f };

    std::vector<std::array<float, 3>> mix_colors(colors.begin(), colors.begin() + count);
    std::vector<float> mix_weights(weights.begin(), weights.begin() + count);
    const std::array<float, 3> rgb = mix_color_solver_components(mix_colors, mix_weights, ColorSolverMixModel::PigmentPainter);
    return { std::clamp(rgb[0], 0.f, 1.f), std::clamp(rgb[1], 0.f, 1.f), std::clamp(rgb[2], 0.f, 1.f), 1.f };
}

float clamp01(float value)
{
    return std::clamp(value, 0.f, 1.f);
}

float apply_minimum_visibility_offset(float value, float offset)
{
    const float safe_value = clamp01(value);
    const float safe_offset = clamp01(offset);
    if (safe_offset <= k_epsilon)
        return safe_value;
    if (safe_value <= safe_offset)
        return 0.f;
    const float denominator = 1.f - safe_offset;
    if (denominator <= k_epsilon)
        return 0.f;
    return clamp01((safe_value - safe_offset) / denominator);
}

void apply_minimum_visibility_offset(std::vector<float> &weights, float offset)
{
    if (offset <= k_epsilon)
        return;
    for (float &weight : weights)
        weight = apply_minimum_visibility_offset(weight, offset);
}

float texture_preview_config_float(const char *key, float fallback)
{
    if (GUI::wxGetApp().preset_bundle == nullptr)
        return fallback;

    const DynamicPrintConfig &config = GUI::wxGetApp().preset_bundle->project_config;
    if (const ConfigOptionFloat *opt = config.option<ConfigOptionFloat>(key))
        return std::isfinite(opt->value) ? float(opt->value) : fallback;
    return fallback;
}

std::array<float, 2> raw_offset_visibility_factors_for_texture_preview(const TextureMappingZone &zone)
{
    const float base_outer_width_mm =
        std::max(0.05f, texture_preview_config_float("texture_mapping_outer_wall_gradient_max_line_width", 0.95f));
    const float min_outer_width_mm = std::clamp(texture_preview_config_float("texture_mapping_outer_wall_gradient_min_line_width", 0.32f),
                                                0.05f,
                                                base_outer_width_mm);
    const float global_strength_factor =
        std::clamp(texture_preview_config_float("texture_mapping_outer_wall_gradient_global_strength", 100.f) / 100.f, 0.f, 1.f);

    float width_range_mm =
        std::min((base_outer_width_mm - min_outer_width_mm) * global_strength_factor,
                 2.f * TextureMappingManager::max_component_surface_offset_mm());
    width_range_mm = std::clamp(width_range_mm, 0.f, base_outer_width_mm);

    return {
        clamp01((base_outer_width_mm - width_range_mm) / base_outer_width_mm),
        clamp01(width_range_mm / base_outer_width_mm)
    };
}

float srgb_to_linear_component(float value)
{
    const float x = clamp01(value);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

std::array<float, 3> oklab_from_srgb(const std::array<float, 3> &rgb)
{
    const float r = srgb_to_linear_component(rgb[0]);
    const float g = srgb_to_linear_component(rgb[1]);
    const float b = srgb_to_linear_component(rgb[2]);

    const float l = std::cbrt(0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b);
    const float m = std::cbrt(0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b);
    const float s = std::cbrt(0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b);

    return {
        0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
        1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
        0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s
    };
}

std::array<float, 3> generic_solver_v2_axis_weights(const std::array<float, 3> &target_oklab)
{
    const float chroma = std::hypot(target_oklab[1], target_oklab[2]);
    const float chroma_factor = std::clamp((chroma - 0.015f) / 0.13f, 0.f, 1.f);
    return {
        1.f + (0.25f - 1.f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor
    };
}

unsigned char to_u8(float value)
{
    return static_cast<unsigned char>(clamp01(value) * 255.f + 0.5f);
}

int texture_mapping_color_hex_digit_for_preview(char ch)
{
    return ch >= '0' && ch <= '9' ? ch - '0' :
           ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 :
           ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
}

std::optional<ColorRGBA> parse_texture_mapping_color_hex_for_preview(const std::string &text)
{
    if (text.empty())
        return std::nullopt;

    const size_t hash_pos = text.find('#');
    const size_t start = hash_pos == std::string::npos ? 0 : hash_pos + 1;
    if (start + 6 > text.size())
        return std::nullopt;

    uint32_t packed = 0;
    for (size_t idx = 0; idx < 6; ++idx) {
        const int value = texture_mapping_color_hex_digit_for_preview(text[start + idx]);
        if (value < 0)
            return std::nullopt;
        packed = (packed << 4) | uint32_t(value);
    }

    uint32_t alpha = 255;
    if (start + 8 <= text.size()) {
        alpha = 0;
        for (size_t idx = 6; idx < 8; ++idx) {
            const int value = texture_mapping_color_hex_digit_for_preview(text[start + idx]);
            if (value < 0)
                return std::nullopt;
            alpha = (alpha << 4) | uint32_t(value);
        }
    }

    return ColorRGBA(float((packed >> 16) & 0xFFu) / 255.f,
                     float((packed >> 8) & 0xFFu) / 255.f,
                     float(packed & 0xFFu) / 255.f,
                     float(alpha & 0xFFu) / 255.f);
}

ColorRGBA opaque_texture_mapping_background_color_for_preview(ColorRGBA color)
{
    color.a(1.f);
    return color;
}

std::optional<ColorRGBA> texture_mapping_background_color_from_config_for_preview(const ModelConfigObject &config)
{
    if (!config.has(TEXTURE_MAPPING_BACKGROUND_COLOR_CONFIG_KEY))
        return std::nullopt;

    const ConfigOptionString *opt = dynamic_cast<const ConfigOptionString *>(config.option(TEXTURE_MAPPING_BACKGROUND_COLOR_CONFIG_KEY));
    if (opt == nullptr)
        return std::nullopt;

    const std::optional<ColorRGBA> color = parse_texture_mapping_color_hex_for_preview(opt->value);
    return color ? std::optional<ColorRGBA>(opaque_texture_mapping_background_color_for_preview(*color)) : std::nullopt;
}

std::optional<ColorRGBA> texture_mapping_background_color_from_metadata_for_preview(const ColorFacetsAnnotation &annotation)
{
    const std::string &metadata = annotation.metadata_json();
    const std::string key = "\"background_color\":\"#";
    const size_t start = metadata.find(key);
    if (start == std::string::npos || start + key.size() + 8 > metadata.size())
        return std::nullopt;

    const std::optional<ColorRGBA> color = parse_texture_mapping_color_hex_for_preview(metadata.substr(start + key.size() - 1, 9));
    return color ? std::optional<ColorRGBA>(opaque_texture_mapping_background_color_for_preview(*color)) : std::nullopt;
}

ColorRGBA texture_mapping_background_color_for_preview(const ModelVolume &model_volume,
                                                       const ColorFacetsAnnotation *color_source = nullptr)
{
    if (std::optional<ColorRGBA> color = texture_mapping_background_color_from_config_for_preview(model_volume.config))
        return *color;
    if (model_volume.get_object() != nullptr) {
        if (std::optional<ColorRGBA> color = texture_mapping_background_color_from_config_for_preview(model_volume.get_object()->config))
            return *color;
    }
    if (color_source != nullptr) {
        if (std::optional<ColorRGBA> color = texture_mapping_background_color_from_metadata_for_preview(*color_source))
            return *color;
    }
    if (std::optional<ColorRGBA> color = texture_mapping_background_color_from_metadata_for_preview(model_volume.texture_mapping_color_facets))
        return *color;
    return ColorRGBA(1.f, 1.f, 1.f, 1.f);
}

ColorRGBA composite_texture_mapping_color_over_background_for_preview(const ColorRGBA &color, const ColorRGBA &background)
{
    const float alpha = clamp01(color.a());
    return ColorRGBA(clamp01(color.r() * alpha + background.r() * (1.f - alpha)),
                     clamp01(color.g() * alpha + background.g() * (1.f - alpha)),
                     clamp01(color.b() * alpha + background.b() * (1.f - alpha)),
                     1.f);
}

void composite_texture_preview_rgba_over_background(std::vector<unsigned char> &rgba, const ColorRGBA &background)
{
    for (size_t idx = 0; idx + 3 < rgba.size(); idx += 4) {
        const ColorRGBA color(float(rgba[idx + 0]) / 255.f,
                              float(rgba[idx + 1]) / 255.f,
                              float(rgba[idx + 2]) / 255.f,
                              float(rgba[idx + 3]) / 255.f);
        const ColorRGBA blended = composite_texture_mapping_color_over_background_for_preview(color, background);
        rgba[idx + 0] = to_u8(blended.r());
        rgba[idx + 1] = to_u8(blended.g());
        rgba[idx + 2] = to_u8(blended.b());
        rgba[idx + 3] = 255;
    }
}

void configure_texture_preview_sampler(const GUI::GLTexture &texture)
{
    if (texture.get_id() == 0)
        return;

    glsafe(::glBindTexture(GL_TEXTURE_2D, texture.get_id()));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0));
    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
}

std::array<unsigned int, 2> limited_simulated_texture_preview_size(unsigned int width,
                                                                   unsigned int height,
                                                                   unsigned int max_edge = k_simulated_texture_preview_max_edge,
                                                                   size_t max_pixels = k_simulated_texture_preview_max_pixels)
{
    if (width == 0 || height == 0)
        return { 0, 0 };

    double scale = 1.0;
    const unsigned int source_max_edge = std::max(width, height);
    if (source_max_edge > max_edge)
        scale = std::min(scale, double(max_edge) / double(source_max_edge));

    const double pixel_count = double(width) * double(height);
    if (pixel_count * scale * scale > double(max_pixels))
        scale = std::min(scale, std::sqrt(double(max_pixels) / pixel_count));

    if (scale >= 1.0)
        return { width, height };

    unsigned int limited_width = std::max(1u, unsigned(std::lround(double(width) * scale)));
    unsigned int limited_height = std::max(1u, unsigned(std::lround(double(height) * scale)));
    while (limited_width > max_edge ||
           limited_height > max_edge ||
           size_t(limited_width) * size_t(limited_height) > max_pixels) {
        if (limited_width >= limited_height && limited_width > 1)
            --limited_width;
        else if (limited_height > 1)
            --limited_height;
        else
            break;
    }
    return { limited_width, limited_height };
}

std::array<unsigned char, 4> sample_texture_preview_rgba_bilinear_at_source(const std::vector<unsigned char> &rgba,
                                                                            unsigned int width,
                                                                            unsigned int height,
                                                                            double src_x_unclamped,
                                                                            double src_y_unclamped);

std::array<unsigned int, 2> simulated_texture_preview_size(unsigned int width,
                                                           unsigned int height,
                                                           const TexturePreviewSimulationSettings &settings,
                                                           unsigned int max_edge = 0,
                                                           size_t max_pixels = 0)
{
    if (max_edge > 0 && max_pixels > 0)
        return limited_simulated_texture_preview_size(width, height, max_edge, max_pixels);
    return settings.limit_texture_resolution ?
        limited_simulated_texture_preview_size(width, height) :
        std::array<unsigned int, 2>{ width, height };
}

bool simulated_texture_preview_needs_temporary_result(unsigned int width,
                                                      unsigned int height,
                                                      const TexturePreviewSimulationSettings &settings)
{
    const size_t source_pixels = size_t(width) * size_t(height);
    if (source_pixels <= k_temporary_simulated_texture_preview_min_source_pixels)
        return false;

    const std::array<unsigned int, 2> final_size = simulated_texture_preview_size(width, height, settings);
    const std::array<unsigned int, 2> temporary_size =
        simulated_texture_preview_size(width,
                                       height,
                                       settings,
                                       k_temporary_simulated_texture_preview_max_edge,
                                       k_temporary_simulated_texture_preview_max_pixels);
    return size_t(final_size[0]) * size_t(final_size[1]) > size_t(temporary_size[0]) * size_t(temporary_size[1]);
}

size_t temporary_simulated_texture_preview_signature(size_t signature)
{
    signature ^= 0x4a7c15f17e315123ull + (signature << 6) + (signature >> 2);
    signature ^= std::hash<unsigned int>{}(k_temporary_simulated_texture_preview_max_edge) + 0x9e3779b97f4a7c15ull +
                 (signature << 6) + (signature >> 2);
    signature ^= std::hash<size_t>{}(k_temporary_simulated_texture_preview_max_pixels) + 0x9e3779b97f4a7c15ull +
                 (signature << 6) + (signature >> 2);
    return signature;
}

std::array<unsigned char, 4> sample_texture_preview_rgba_bilinear(const std::vector<unsigned char> &rgba,
                                                                  unsigned int width,
                                                                  unsigned int height,
                                                                  unsigned int preview_x,
                                                                  unsigned int preview_y,
                                                                  unsigned int preview_width,
                                                                  unsigned int preview_height)
{
    const double src_x = (double(preview_x) + 0.5) * double(width) / double(std::max(1u, preview_width)) - 0.5;
    const double src_y = (double(preview_y) + 0.5) * double(height) / double(std::max(1u, preview_height)) - 0.5;
    return sample_texture_preview_rgba_bilinear_at_source(rgba, width, height, src_x, src_y);
}

std::array<unsigned char, 4> sample_texture_preview_rgba_bilinear_at_source(const std::vector<unsigned char> &rgba,
                                                                            unsigned int width,
                                                                            unsigned int height,
                                                                            double src_x_unclamped,
                                                                            double src_y_unclamped)
{
    if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return { 0, 0, 0, 255 };

    const double src_x = std::clamp(src_x_unclamped, 0.0, double(width - 1));
    const double src_y = std::clamp(src_y_unclamped, 0.0, double(height - 1));
    const unsigned int x0 = std::min(width - 1, unsigned(std::floor(src_x)));
    const unsigned int y0 = std::min(height - 1, unsigned(std::floor(src_y)));
    const unsigned int x1 = std::min(width - 1, x0 + 1);
    const unsigned int y1 = std::min(height - 1, y0 + 1);
    const double tx = src_x - double(x0);
    const double ty = src_y - double(y0);

    auto channel_at = [&rgba, width](unsigned int x, unsigned int y, size_t channel) {
        return double(rgba[(size_t(y) * size_t(width) + size_t(x)) * 4 + channel]);
    };
    auto sample_channel = [&](size_t channel) {
        const double top = channel_at(x0, y0, channel) * (1.0 - tx) + channel_at(x1, y0, channel) * tx;
        const double bottom = channel_at(x0, y1, channel) * (1.0 - tx) + channel_at(x1, y1, channel) * tx;
        return static_cast<unsigned char>(std::clamp(int(std::lround(top * (1.0 - ty) + bottom * ty)), 0, 255));
    };

    return { sample_channel(0), sample_channel(1), sample_channel(2), sample_channel(3) };
}

std::vector<float> sample_texture_preview_raw_offsets_bilinear(const std::vector<unsigned char> &offsets,
                                                               unsigned int width,
                                                               unsigned int height,
                                                               unsigned int channels,
                                                               unsigned int preview_x,
                                                               unsigned int preview_y,
                                                               unsigned int preview_width,
                                                               unsigned int preview_height)
{
    std::vector<float> values(channels, 0.f);
    if (width == 0 || height == 0 || channels == 0 ||
        offsets.size() < size_t(width) * size_t(height) * size_t(channels))
        return values;

    const double src_x = std::clamp((double(preview_x) + 0.5) * double(width) / double(std::max(1u, preview_width)) - 0.5,
                                    0.0,
                                    double(width - 1));
    const double src_y = std::clamp((double(preview_y) + 0.5) * double(height) / double(std::max(1u, preview_height)) - 0.5,
                                    0.0,
                                    double(height - 1));
    const unsigned int x0 = std::min(width - 1, unsigned(std::floor(src_x)));
    const unsigned int y0 = std::min(height - 1, unsigned(std::floor(src_y)));
    const unsigned int x1 = std::min(width - 1, x0 + 1);
    const unsigned int y1 = std::min(height - 1, y0 + 1);
    const double tx = src_x - double(x0);
    const double ty = src_y - double(y0);

    auto channel_at = [&offsets, width, channels](unsigned int x, unsigned int y, unsigned int channel) {
        return double(offsets[(size_t(y) * size_t(width) + size_t(x)) * size_t(channels) + size_t(channel)]) / 255.0;
    };
    for (unsigned int channel = 0; channel < channels; ++channel) {
        const double top = channel_at(x0, y0, channel) * (1.0 - tx) + channel_at(x1, y0, channel) * tx;
        const double bottom = channel_at(x0, y1, channel) * (1.0 - tx) + channel_at(x1, y1, channel) * tx;
        values[size_t(channel)] = clamp01(float(top * (1.0 - ty) + bottom * ty));
    }
    return values;
}

std::vector<std::string> raw_filament_color_mode_channel_keys_for_texture_preview(int filament_color_mode, size_t component_count)
{
    std::vector<std::string> keys;
    switch (std::clamp(filament_color_mode,
                       int(TextureMappingZone::FilamentColorAny),
                       int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorRGB):
        keys = { "R", "G", "B" };
        break;
    case int(TextureMappingZone::FilamentColorCMY):
        keys = { "C", "M", "Y" };
        break;
    case int(TextureMappingZone::FilamentColorCMYK):
        keys = { "C", "M", "Y", "K" };
        break;
    case int(TextureMappingZone::FilamentColorCMYW):
        keys = { "C", "M", "Y", "W" };
        break;
    case int(TextureMappingZone::FilamentColorRGBK):
        keys = { "R", "G", "B", "K" };
        break;
    case int(TextureMappingZone::FilamentColorRGBW):
        keys = { "R", "G", "B", "W" };
        break;
    case int(TextureMappingZone::FilamentColorBW):
        keys = { "K", "W" };
        break;
    case int(TextureMappingZone::FilamentColorCMYKW):
        keys = { "C", "M", "Y", "K", "W" };
        break;
    case int(TextureMappingZone::FilamentColorRGBKW):
        keys = { "R", "G", "B", "K", "W" };
        break;
    default:
        break;
    }
    if (keys.size() > component_count)
        keys.resize(component_count);
    return keys;
}

std::array<float, 3> raw_filament_channel_color_for_texture_preview(const ImageMapRawFilament &filament, size_t channel_idx)
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
        const std::optional<ColorRGBA> parsed = parse_texture_mapping_color_hex_for_preview(filament.hex);
        if (parsed)
            return { { parsed->r(), parsed->g(), parsed->b() } };
    }
    return { { 1.f, 1.f, 1.f } };
}

float raw_filament_color_distance_sq_for_texture_preview(const std::array<float, 3> &lhs, const std::array<float, 3> &rhs)
{
    const std::array<float, 3> lhs_oklab = oklab_from_srgb(lhs);
    const std::array<float, 3> rhs_oklab = oklab_from_srgb(rhs);
    const float dl = lhs_oklab[0] - rhs_oklab[0];
    const float da = lhs_oklab[1] - rhs_oklab[1];
    const float db = lhs_oklab[2] - rhs_oklab[2];
    return dl * dl + da * da + db * db;
}

std::vector<size_t> raw_component_source_channels_for_texture_preview(const std::string &metadata_json,
                                                                      unsigned int source_channels,
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
        source_colors[channel] = raw_filament_channel_color_for_texture_preview(filaments[channel], channel);
    }

    const std::vector<std::string> target_keys =
        raw_filament_color_mode_channel_keys_for_texture_preview(filament_color_mode, component_count);
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
                raw_filament_channel_color_for_texture_preview({ 0, target_keys[component_idx], std::string() }, component_idx);
            size_t best_channel = size_t(source_channels);
            float best_distance_sq = std::numeric_limits<float>::max();
            for (size_t channel = 0; channel < source_colors.size(); ++channel) {
                if (used[channel] != 0)
                    continue;
                const float distance_sq = raw_filament_color_distance_sq_for_texture_preview(source_colors[channel], target_color);
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
        struct RawColorMatchCandidate {
            float  distance_sq { 0.f };
            size_t component_idx { 0 };
            size_t source_channel { 0 };
        };

        std::vector<RawColorMatchCandidate> candidates;
        candidates.reserve(component_count * size_t(source_channels));
        for (size_t channel = 0; channel < filaments.size(); ++channel) {
            for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
                candidates.push_back({
                    raw_filament_color_distance_sq_for_texture_preview(component_colors[component_idx], source_colors[channel]),
                    component_idx,
                    channel
                });
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const RawColorMatchCandidate &lhs, const RawColorMatchCandidate &rhs) {
            return lhs.distance_sq < rhs.distance_sq;
        });

        std::vector<uint8_t> used_components(component_count, 0);
        std::vector<uint8_t> used_sources(static_cast<size_t>(source_channels), 0);
        for (const RawColorMatchCandidate &candidate : candidates) {
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

std::vector<float> map_raw_sample_to_components_for_texture_preview(const std::vector<float> &raw_sample,
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

unsigned int texture_preview_rgb_cache_key(const std::array<unsigned char, 3> &rgb)
{
    return unsigned(rgb[0]) | (unsigned(rgb[1]) << 8) | (unsigned(rgb[2]) << 16);
}

unsigned int filament_id_for_state(size_t state_id, unsigned int base_filament_id)
{
    return state_id == 0 ? base_filament_id : unsigned(state_id);
}

const TextureMappingZone *zone_for_filament(unsigned int filament_id, size_t num_physical, const TextureMappingManager *texture_mgr)
{
    return texture_mgr != nullptr && filament_id > num_physical ? texture_mgr->zone_from_id(filament_id) : nullptr;
}

bool is_image_zone(const TextureMappingZone &zone)
{
    return zone.enabled && !zone.deleted && zone.is_image_texture();
}

bool is_gradient_zone(const TextureMappingZone &zone)
{
    return zone.enabled && !zone.deleted && zone.is_2d_gradient();
}

float texture_preview_mix_for_filament(unsigned int filament_id, size_t num_physical, const TextureMappingManager *texture_mgr)
{
    const TextureMappingZone *zone = zone_for_filament(filament_id, num_physical, texture_mgr);
    if (zone == nullptr || (!is_image_zone(*zone) && !is_gradient_zone(*zone)))
        return 0.f;
    return std::clamp(zone->preview_opacity_pct, 0.f, 100.f) / 100.f;
}

bool texture_preview_settings_invalid_for_filament(unsigned int filament_id, size_t num_physical, const TextureMappingManager *texture_mgr)
{
    const TextureMappingZone *zone = zone_for_filament(filament_id, num_physical, texture_mgr);
    if (zone == nullptr)
        return false;
    if (is_image_zone(*zone))
        return TextureMappingManager::component_count_mismatch(*zone, num_physical);
    if (is_gradient_zone(*zone))
        return TextureMappingManager::selected_component_ids(*zone, num_physical).size() < 2;
    return false;
}

std::vector<std::string> physical_filament_colors_for_texture_preview(size_t num_physical)
{
    std::vector<std::string> colors;
    if (GUI::wxGetApp().preset_bundle != nullptr) {
        if (const ConfigOptionStrings *opt = GUI::wxGetApp().preset_bundle->project_config.option<ConfigOptionStrings>("filament_colour"))
            colors = opt->values;
    }
    colors.resize(num_physical, "#26A69A");
    return colors;
}

float color_distance_sq(const std::array<float, 3> &lhs, const std::array<float, 3> &rhs)
{
    const float dr = lhs[0] - rhs[0];
    const float dg = lhs[1] - rhs[1];
    const float db = lhs[2] - rhs[2];
    return dr * dr + dg * dg + db * db;
}

std::vector<size_t> best_matching_component_indices_for_semantic_colors(const std::vector<std::array<float, 3>> &component_colors,
                                                                        const std::vector<std::array<float, 3>> &semantic_colors)
{
    if (component_colors.empty() || component_colors.size() != semantic_colors.size())
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

std::vector<size_t> semantic_component_indices_for_texture_preview(const std::vector<std::array<float, 3>> &component_colors,
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

std::vector<std::array<float, 3>> fixed_color_generic_solver_component_colors(int filament_color_mode)
{
    switch (std::clamp(filament_color_mode,
                       int(TextureMappingZone::FilamentColorAny),
                       int(TextureMappingZone::FilamentColorRGBKW))) {
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

bool texture_preview_uses_fixed_color_generic_solver(const TexturePreviewSimulationSettings &settings)
{
    if (settings.mapping_mode == int(TextureMappingZone::TextureMappingRawValues) ||
        settings.use_legacy_fixed_color_mode)
        return false;

    const std::vector<std::array<float, 3>> fixed_colors =
        fixed_color_generic_solver_component_colors(settings.filament_color_mode);
    return !fixed_colors.empty() && fixed_colors.size() == settings.component_colors.size();
}

std::vector<std::array<float, 3>> generic_solver_component_colors(const TexturePreviewSimulationSettings &settings)
{
    if (texture_preview_uses_fixed_color_generic_solver(settings))
        return fixed_color_generic_solver_component_colors(settings.filament_color_mode);
    return settings.component_colors;
}

bool texture_preview_uses_generic_solver(const TexturePreviewSimulationSettings &settings)
{
    if (settings.mapping_mode == int(TextureMappingZone::TextureMappingRawValues))
        return false;
    if (settings.use_fixed_color_generic_solver)
        return true;

    const int clamped_mode = std::clamp(settings.filament_color_mode,
                                        int(TextureMappingZone::FilamentColorAny),
                                        int(TextureMappingZone::FilamentColorRGBKW));
    size_t expected_component_count = 0;
    switch (clamped_mode) {
    case int(TextureMappingZone::FilamentColorRGB):
    case int(TextureMappingZone::FilamentColorCMY):
        expected_component_count = 3;
        break;
    case int(TextureMappingZone::FilamentColorCMYK):
    case int(TextureMappingZone::FilamentColorCMYW):
    case int(TextureMappingZone::FilamentColorRGBK):
    case int(TextureMappingZone::FilamentColorRGBW):
        expected_component_count = 4;
        break;
    case int(TextureMappingZone::FilamentColorBW):
        expected_component_count = 2;
        break;
    case int(TextureMappingZone::FilamentColorCMYKW):
    case int(TextureMappingZone::FilamentColorRGBKW):
        expected_component_count = 5;
        break;
    default:
        return true;
    }

    return settings.component_colors.size() != expected_component_count;
}

std::vector<TexturePreviewMixCandidate> build_generic_mix_candidates(const std::vector<std::array<float, 3>> &component_colors)
{
    if (component_colors.empty())
        return {};

    const size_t component_count = component_colors.size();
    const int total_units = component_count <= 4 ? 40 : (component_count == 5 ? 24 : (component_count == 6 ? 20 : 12));
    std::vector<int> units(component_count, 0);
    std::vector<TexturePreviewMixCandidate> candidates;
    const size_t n = size_t(total_units) + component_count - 1;
    size_t k = component_count - 1;
    k = std::min(k, n - k);
    size_t candidate_count = 1;
    for (size_t idx = 1; idx <= k; ++idx)
        candidate_count = (candidate_count * (n - k + idx)) / idx;
    candidates.reserve(candidate_count);

    std::function<void(size_t, int)> recurse = [&](size_t idx, int remaining_units) {
        if (idx + 1 == component_count) {
            units[idx] = remaining_units;
            TexturePreviewMixCandidate candidate;
            candidate.weights.assign(component_count, 0.f);
            for (size_t weight_idx = 0; weight_idx < component_count; ++weight_idx)
                candidate.weights[weight_idx] = float(units[weight_idx]) / float(std::max(1, total_units));
            candidate.rgb = mix_color_solver_components(component_colors, candidate.weights, ColorSolverMixModel::PigmentPainter);
            candidate.perceptual = oklab_from_srgb(candidate.rgb);
            candidates.emplace_back(std::move(candidate));
            return;
        }

        for (int unit = 0; unit <= remaining_units; ++unit) {
            units[idx] = unit;
            recurse(idx + 1, remaining_units - unit);
        }
    };
    recurse(0, total_units);
    return candidates;
}

int build_generic_mix_candidate_kd_tree(const std::vector<TexturePreviewMixCandidate> &candidates,
                                        std::vector<TexturePreviewMixCandidateKdNode> &nodes,
                                        std::vector<uint32_t> &indices,
                                        size_t begin,
                                        size_t end,
                                        uint8_t axis,
                                        bool perceptual_coords)
{
    if (begin >= end)
        return -1;

    const size_t mid = begin + (end - begin) / 2;
    auto axis_value = [&candidates, axis, perceptual_coords](uint32_t candidate_idx) {
        const TexturePreviewMixCandidate &candidate = candidates[size_t(candidate_idx)];
        return perceptual_coords ? candidate.perceptual[size_t(axis)] : candidate.rgb[size_t(axis)];
    };
    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end, [&axis_value](uint32_t lhs, uint32_t rhs) {
        return axis_value(lhs) < axis_value(rhs);
    });

    const int node_idx = int(nodes.size());
    TexturePreviewMixCandidateKdNode node;
    node.candidate_idx = indices[mid];
    node.axis = axis;
    nodes.emplace_back(node);

    const uint8_t next_axis = uint8_t((axis + 1) % 3);
    const int left = build_generic_mix_candidate_kd_tree(candidates, nodes, indices, begin, mid, next_axis, perceptual_coords);
    const int right = build_generic_mix_candidate_kd_tree(candidates, nodes, indices, mid + 1, end, next_axis, perceptual_coords);
    nodes[size_t(node_idx)].left = left;
    nodes[size_t(node_idx)].right = right;
    return node_idx;
}

int build_generic_mix_candidate_kd_tree(const std::vector<TexturePreviewMixCandidate> &candidates,
                                        std::vector<TexturePreviewMixCandidateKdNode> &nodes,
                                        bool perceptual_coords = false)
{
    nodes.clear();
    if (candidates.empty())
        return -1;

    std::vector<uint32_t> indices(candidates.size(), 0);
    for (size_t idx = 0; idx < candidates.size(); ++idx)
        indices[idx] = uint32_t(idx);

    nodes.reserve(candidates.size());
    return build_generic_mix_candidate_kd_tree(candidates, nodes, indices, 0, candidates.size(), uint8_t(0), perceptual_coords);
}

struct TexturePreviewMixNearestResult
{
    size_t best_idx { size_t(-1) };
    size_t second_idx { size_t(-1) };
    float best_error { std::numeric_limits<float>::max() };
    float second_error { std::numeric_limits<float>::max() };
};

void update_texture_preview_mix_nearest_result(TexturePreviewMixNearestResult &result, size_t candidate_idx, float error)
{
    if (candidate_idx == result.best_idx || candidate_idx == result.second_idx)
        return;

    if (error < result.best_error) {
        result.second_error = result.best_error;
        result.second_idx = result.best_idx;
        result.best_error = error;
        result.best_idx = candidate_idx;
    } else if (error < result.second_error) {
        result.second_error = error;
        result.second_idx = candidate_idx;
    }
}

float texture_preview_mix_candidate_error(const TexturePreviewMixCandidate &candidate, const std::array<float, 3> &target_rgb)
{
    const float dr = candidate.rgb[0] - target_rgb[0];
    const float dg = candidate.rgb[1] - target_rgb[1];
    const float db = candidate.rgb[2] - target_rgb[2];
    return dr * dr + dg * dg + db * db;
}

TexturePreviewMixNearestResult nearest_texture_preview_mix_candidates_linear(const std::vector<TexturePreviewMixCandidate> &candidates,
                                                                             const std::array<float, 3> &target_rgb)
{
    TexturePreviewMixNearestResult result;
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx)
        update_texture_preview_mix_nearest_result(result,
                                                  candidate_idx,
                                                  texture_preview_mix_candidate_error(candidates[candidate_idx], target_rgb));
    return result;
}

void query_texture_preview_mix_candidate_kd_tree(const std::vector<TexturePreviewMixCandidate> &candidates,
                                                 const std::vector<TexturePreviewMixCandidateKdNode> &nodes,
                                                 const std::array<float, 3> &target_rgb,
                                                 int node_idx,
                                                 TexturePreviewMixNearestResult &result)
{
    if (node_idx < 0 || size_t(node_idx) >= nodes.size())
        return;

    const TexturePreviewMixCandidateKdNode &node = nodes[size_t(node_idx)];
    if (size_t(node.candidate_idx) >= candidates.size()) {
        query_texture_preview_mix_candidate_kd_tree(candidates, nodes, target_rgb, node.left, result);
        query_texture_preview_mix_candidate_kd_tree(candidates, nodes, target_rgb, node.right, result);
        return;
    }

    const TexturePreviewMixCandidate &candidate = candidates[size_t(node.candidate_idx)];
    update_texture_preview_mix_nearest_result(result,
                                              size_t(node.candidate_idx),
                                              texture_preview_mix_candidate_error(candidate, target_rgb));

    const size_t axis = std::min<size_t>(node.axis, 2);
    const float split_delta = target_rgb[axis] - candidate.rgb[axis];
    const int near_node = split_delta <= 0.f ? node.left : node.right;
    const int far_node = split_delta <= 0.f ? node.right : node.left;

    query_texture_preview_mix_candidate_kd_tree(candidates, nodes, target_rgb, near_node, result);
    if (split_delta * split_delta <= result.second_error)
        query_texture_preview_mix_candidate_kd_tree(candidates, nodes, target_rgb, far_node, result);
}

TexturePreviewMixNearestResult nearest_texture_preview_mix_candidates(const std::vector<TexturePreviewMixCandidate> &candidates,
                                                                      const std::vector<TexturePreviewMixCandidateKdNode> &nodes,
                                                                      int root,
                                                                      const std::array<float, 3> &target_rgb)
{
    TexturePreviewMixNearestResult result;
    if (root >= 0 && !nodes.empty())
        query_texture_preview_mix_candidate_kd_tree(candidates, nodes, target_rgb, root, result);
    if (result.best_idx >= candidates.size())
        result = nearest_texture_preview_mix_candidates_linear(candidates, target_rgb);
    return result;
}

float texture_preview_mix_candidate_perceptual_error(const TexturePreviewMixCandidate &candidate,
                                                     const std::array<float, 3> &target_oklab,
                                                     const std::array<float, 3> &axis_weights)
{
    const float dl = candidate.perceptual[0] - target_oklab[0];
    const float da = candidate.perceptual[1] - target_oklab[1];
    const float db = candidate.perceptual[2] - target_oklab[2];
    return axis_weights[0] * dl * dl + axis_weights[1] * da * da + axis_weights[2] * db * db;
}

TexturePreviewMixNearestResult nearest_texture_preview_mix_candidates_perceptual_linear(
    const std::vector<TexturePreviewMixCandidate> &candidates,
    const std::array<float, 3> &target_oklab,
    const std::array<float, 3> &axis_weights)
{
    TexturePreviewMixNearestResult result;
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx)
        update_texture_preview_mix_nearest_result(result,
                                                  candidate_idx,
                                                  texture_preview_mix_candidate_perceptual_error(candidates[candidate_idx], target_oklab, axis_weights));
    return result;
}

void query_texture_preview_mix_candidate_perceptual_kd_tree(const std::vector<TexturePreviewMixCandidate> &candidates,
                                                            const std::vector<TexturePreviewMixCandidateKdNode> &nodes,
                                                            const std::array<float, 3> &target_oklab,
                                                            const std::array<float, 3> &axis_weights,
                                                            int node_idx,
                                                            TexturePreviewMixNearestResult &result)
{
    if (node_idx < 0 || size_t(node_idx) >= nodes.size())
        return;

    const TexturePreviewMixCandidateKdNode &node = nodes[size_t(node_idx)];
    if (size_t(node.candidate_idx) >= candidates.size()) {
        query_texture_preview_mix_candidate_perceptual_kd_tree(candidates, nodes, target_oklab, axis_weights, node.left, result);
        query_texture_preview_mix_candidate_perceptual_kd_tree(candidates, nodes, target_oklab, axis_weights, node.right, result);
        return;
    }

    const TexturePreviewMixCandidate &candidate = candidates[size_t(node.candidate_idx)];
    update_texture_preview_mix_nearest_result(
        result,
        size_t(node.candidate_idx),
        texture_preview_mix_candidate_perceptual_error(candidate, target_oklab, axis_weights));

    const size_t axis = std::min<size_t>(node.axis, 2);
    const float split_delta = target_oklab[axis] - candidate.perceptual[axis];
    const int near_node = split_delta <= 0.f ? node.left : node.right;
    const int far_node = split_delta <= 0.f ? node.right : node.left;

    query_texture_preview_mix_candidate_perceptual_kd_tree(candidates, nodes, target_oklab, axis_weights, near_node, result);
    if (axis_weights[axis] * split_delta * split_delta <= result.second_error)
        query_texture_preview_mix_candidate_perceptual_kd_tree(candidates, nodes, target_oklab, axis_weights, far_node, result);
}

TexturePreviewMixNearestResult nearest_texture_preview_mix_candidates_perceptual(
    const std::vector<TexturePreviewMixCandidate> &candidates,
    const std::vector<TexturePreviewMixCandidateKdNode> &nodes,
    int root,
    const std::array<float, 3> &target_rgb)
{
    TexturePreviewMixNearestResult result;
    if (candidates.empty())
        return result;

    const std::array<float, 3> target_oklab = oklab_from_srgb(target_rgb);
    const std::array<float, 3> axis_weights = generic_solver_v2_axis_weights(target_oklab);
    if (root >= 0 && !nodes.empty())
        query_texture_preview_mix_candidate_perceptual_kd_tree(candidates, nodes, target_oklab, axis_weights, root, result);
    if (result.best_idx >= candidates.size())
        result = nearest_texture_preview_mix_candidates_perceptual_linear(candidates, target_oklab, axis_weights);
    return result;
}

std::vector<float> best_component_mix_weights_for_target(const std::vector<TexturePreviewMixCandidate> &candidates,
                                                         const std::vector<TexturePreviewMixCandidateKdNode> &nodes,
                                                         int root,
                                                         const std::vector<TexturePreviewMixCandidateKdNode> &perceptual_nodes,
                                                         int perceptual_root,
                                                         const std::array<float, 3> &target_rgb,
                                                         int generic_solver_lookup_mode,
                                                         int generic_solver_mode)
{
    if (candidates.empty())
        return {};

    const int clamped_solver_mode = std::clamp(generic_solver_mode,
                                               int(TextureMappingZone::GenericSolverLegacy),
                                               int(TextureMappingZone::GenericSolverV2));
    TexturePreviewMixNearestResult nearest =
        clamped_solver_mode == int(TextureMappingZone::GenericSolverV2) ?
            nearest_texture_preview_mix_candidates_perceptual(candidates, perceptual_nodes, perceptual_root, target_rgb) :
            nearest_texture_preview_mix_candidates(candidates, nodes, root, target_rgb);
    if (nearest.best_idx >= candidates.size() && clamped_solver_mode == int(TextureMappingZone::GenericSolverV2))
        nearest = nearest_texture_preview_mix_candidates(candidates, nodes, root, target_rgb);
    if (nearest.best_idx >= candidates.size())
        return {};

    const int clamped_mode = std::clamp(generic_solver_lookup_mode,
                                        int(TextureMappingZone::GenericSolverClosestMix),
                                        int(TextureMappingZone::GenericSolverBlendClosestTwo));
    if (clamped_mode == int(TextureMappingZone::GenericSolverClosestMix) ||
        nearest.second_idx >= candidates.size() ||
        nearest.best_error <= 1e-12f)
        return candidates[nearest.best_idx].weights;

    const TexturePreviewMixCandidate &best_candidate = candidates[nearest.best_idx];
    const TexturePreviewMixCandidate &second_candidate = candidates[nearest.second_idx];
    const float best_inv = 1.f / std::max(nearest.best_error, 1e-12f);
    const float second_inv = 1.f / std::max(nearest.second_error, 1e-12f);
    const float inv_sum = std::max(best_inv + second_inv, 1e-12f);
    std::vector<float> weights(best_candidate.weights.size(), 0.f);
    for (size_t idx = 0; idx < weights.size(); ++idx) {
        const float second_weight = idx < second_candidate.weights.size() ? second_candidate.weights[idx] : 0.f;
        weights[idx] = clamp01((best_candidate.weights[idx] * best_inv + second_weight * second_inv) / inv_sum);
    }
    return weights;
}

float apply_texture_tone_gamma(float channel, float tone_gamma)
{
    const float safe_channel = clamp01(channel);
    const float safe_gamma = (!std::isfinite(tone_gamma) || tone_gamma <= 0.f) ? 1.f : std::clamp(tone_gamma, 0.5f, 3.f);
    if (std::abs(safe_gamma - 1.f) <= 1e-5f)
        return safe_channel;
    return clamp01(std::pow(safe_channel, 1.f / safe_gamma));
}

void apply_texture_contrast_to_mapped_components(std::vector<float> &component_weights,
                                                 float contrast_factor,
                                                 size_t mapped_component_count)
{
    const size_t count = std::min(mapped_component_count, component_weights.size());
    if (count == 0)
        return;

    float mean_weight = 0.f;
    for (size_t idx = 0; idx < count; ++idx)
        mean_weight += clamp01(component_weights[idx]);
    mean_weight /= float(count);

    for (size_t idx = 0; idx < count; ++idx) {
        const float safe_weight = clamp01(component_weights[idx]);
        component_weights[idx] = clamp01(mean_weight + (safe_weight - mean_weight) * contrast_factor);
    }
}

std::array<float, 3> apply_texture_contrast_to_rgb(const std::array<float, 3> &rgb, float contrast_factor)
{
    const float clamped_contrast = std::clamp(contrast_factor, 0.25f, 3.f);
    if (std::abs(clamped_contrast - 1.f) <= 1e-5f)
        return { clamp01(rgb[0]), clamp01(rgb[1]), clamp01(rgb[2]) };

    const float mean = (clamp01(rgb[0]) + clamp01(rgb[1]) + clamp01(rgb[2])) / 3.f;
    return {
        clamp01(mean + (clamp01(rgb[0]) - mean) * clamped_contrast),
        clamp01(mean + (clamp01(rgb[1]) - mean) * clamped_contrast),
        clamp01(mean + (clamp01(rgb[2]) - mean) * clamped_contrast)
    };
}

struct TexturePreviewBinaryDitherCandidate
{
    uint32_t mask { 0 };
    std::array<float, 3> rgb { { 1.f, 1.f, 1.f } };
    std::array<float, 3> oklab { { 1.f, 0.f, 0.f } };
};

std::vector<float> binary_component_visibility_weights_for_texture_preview(const TexturePreviewSimulationSettings &settings,
                                                                           uint32_t                                mask)
{
    const size_t component_count = settings.component_colors.size();
    std::vector<float> weights(component_count, 0.f);
    for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
        const float strength = component_idx < settings.component_strength_factors.size() ?
            std::clamp(settings.component_strength_factors[component_idx], 0.f, 1.f) :
            1.f;
        const float minimum = component_idx < settings.component_minimum_offset_factors.size() ?
            std::clamp(settings.component_minimum_offset_factors[component_idx], 0.f, 1.f) :
            0.f;
        const bool active = (mask & (uint32_t(1) << component_idx)) != 0;
        weights[component_idx] = std::clamp(minimum + (active ? strength * (1.f - minimum) : 0.f), 0.f, 1.f);
    }
    return weights;
}

std::vector<float> halftone_component_visibility_weights_for_texture_preview(const TexturePreviewSimulationSettings &settings,
                                                                             uint32_t                                mask)
{
    std::vector<float> weights = binary_component_visibility_weights_for_texture_preview(settings, mask);
    if (mask == 0) {
        std::fill(weights.begin(), weights.end(), 1.f);
        return weights;
    }

    float total_weight = 0.f;
    for (const float weight : weights)
        total_weight += std::max(0.f, weight);
    if (total_weight > k_epsilon)
        return weights;

    for (size_t component_idx = 0; component_idx < weights.size() && component_idx < 31; ++component_idx)
        if ((mask & (uint32_t(1) << component_idx)) != 0)
            weights[component_idx] = 1.f;
    return weights;
}

ColorRGBA halftone_preview_color_from_mask(const TexturePreviewSimulationSettings &settings, uint32_t mask)
{
    const std::vector<float> weights = halftone_component_visibility_weights_for_texture_preview(settings, mask);
    if (settings.component_colors.empty() || weights.empty())
        return { 0.f, 0.f, 0.f, 1.f };

    const std::array<float, 3> rgb =
        mix_color_solver_components(settings.component_colors, weights, ColorSolverMixModel::PigmentPainter);
    return { clamp01(rgb[0]), clamp01(rgb[1]), clamp01(rgb[2]), 1.f };
}

std::vector<TexturePreviewBinaryDitherCandidate> binary_dither_candidates_for_texture_preview(
    const TexturePreviewSimulationSettings &settings)
{
    std::vector<TexturePreviewBinaryDitherCandidate> candidates;
    const size_t component_count = settings.component_colors.size();
    if (component_count == 0 || component_count > 16)
        return candidates;

    const uint32_t mask_end = uint32_t(1) << component_count;
    candidates.reserve(size_t(mask_end - 1));
    for (uint32_t mask = 1; mask < mask_end; ++mask) {
        TexturePreviewBinaryDitherCandidate candidate;
        candidate.mask = mask;
        std::vector<float> weights = binary_component_visibility_weights_for_texture_preview(settings, mask);
        float total_weight = 0.f;
        for (const float weight : weights)
            total_weight += weight;
        if (total_weight <= k_epsilon) {
            for (size_t component_idx = 0; component_idx < component_count; ++component_idx)
                if ((mask & (uint32_t(1) << component_idx)) != 0)
                    weights[component_idx] = 1.f;
        }
        candidate.rgb = mix_color_solver_components(settings.component_colors,
                                                    weights,
                                                    color_solver_mix_model_from_index(settings.generic_solver_mix_model));
        candidate.oklab = oklab_from_srgb(candidate.rgb);
        candidates.emplace_back(candidate);
    }
    return candidates;
}

std::array<float, 3> texture_preview_target_oklab(const TexturePreviewSimulationSettings &settings,
                                                  const std::array<float, 4>            &sample_rgba)
{
    std::array<float, 3> target = {
        clamp01(sample_rgba[0]),
        clamp01(sample_rgba[1]),
        clamp01(sample_rgba[2])
    };
    if (std::abs(settings.tone_gamma - 1.f) > 1e-5f) {
        target[0] = apply_texture_tone_gamma(target[0], settings.tone_gamma);
        target[1] = apply_texture_tone_gamma(target[1], settings.tone_gamma);
        target[2] = apply_texture_tone_gamma(target[2], settings.tone_gamma);
    }
    target = apply_texture_contrast_to_rgb(target, std::clamp(settings.contrast_pct, 25.f, 300.f) / 100.f);
    return oklab_from_srgb(target);
}

struct TexturePreviewBinaryDitherNearestResult {
    size_t best_idx { size_t(-1) };
    size_t second_idx { size_t(-1) };
    float best_error { std::numeric_limits<float>::max() };
    float second_error { std::numeric_limits<float>::max() };
};

TexturePreviewBinaryDitherNearestResult nearest_binary_dither_candidates_for_texture_preview(
    const std::vector<TexturePreviewBinaryDitherCandidate> &candidates,
    const std::array<float, 3>                             &target_oklab)
{
    TexturePreviewBinaryDitherNearestResult result;
    const std::array<float, 3> axis_weights = generic_solver_v2_axis_weights(target_oklab);
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

size_t nearest_binary_dither_candidate_for_texture_preview(const std::vector<TexturePreviewBinaryDitherCandidate> &candidates,
                                                           const std::array<float, 3>                             &target_oklab)
{
    return nearest_binary_dither_candidates_for_texture_preview(candidates, target_oklab).best_idx;
}

float binary_dither_alternate_fraction_for_texture_preview(const std::vector<TexturePreviewBinaryDitherCandidate> &candidates,
                                                          const std::array<float, 3>                             &target_oklab,
                                                          size_t                                                  base_idx,
                                                          size_t                                                  alternate_idx)
{
    if (base_idx >= candidates.size() || alternate_idx >= candidates.size() || base_idx == alternate_idx)
        return 0.f;

    const std::array<float, 3> axis_weights = generic_solver_v2_axis_weights(target_oklab);
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

size_t thresholded_binary_dither_candidate_for_texture_preview(const std::vector<TexturePreviewBinaryDitherCandidate> &candidates,
                                                               const std::array<float, 3>                             &target_oklab,
                                                               float                                                   threshold)
{
    const TexturePreviewBinaryDitherNearestResult nearest =
        nearest_binary_dither_candidates_for_texture_preview(candidates, target_oklab);
    if (nearest.best_idx >= candidates.size())
        return size_t(-1);
    if (nearest.second_idx >= candidates.size())
        return nearest.best_idx;

    const float alternate_fraction =
        binary_dither_alternate_fraction_for_texture_preview(candidates, target_oklab, nearest.best_idx, nearest.second_idx);
    return std::clamp(threshold, 0.f, 1.f) < alternate_fraction ? nearest.second_idx : nearest.best_idx;
}

float ordered_bayer_threshold_for_texture_preview(int x, int y)
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

bool is_halftone_dithering_method_for_texture_preview(int method)
{
    const int clamped_method = std::clamp(method,
                                          int(TextureMappingZone::DitheringClosest),
                                          int(TextureMappingZone::DitheringHalftoneIncreasedDetail));
    return clamped_method == int(TextureMappingZone::DitheringHalftone) ||
           clamped_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail);
}

float halftone_screen_angle_deg_for_texture_preview(int filament_color_mode, size_t component_idx)
{
    const int clamped_mode =
        std::clamp(filament_color_mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorRGBKW));
    if (component_idx < 3) {
        static constexpr float primary_angles[3] = {15.f, 75.f, 0.f};
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

    static constexpr float fallback_angles[] = {15.f, 75.f, 0.f, 45.f, 90.f, 30.f, 60.f, 105.f, 120.f, 150.f};
    return fallback_angles[component_idx % (sizeof(fallback_angles) / sizeof(fallback_angles[0]))];
}

float halftone_threshold_for_texture_preview(float x_mm, float y_mm, float dot_size_mm)
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

bool halftone_screen_on_for_texture_preview(float coverage,
                                            float surface_x_mm,
                                            float surface_y_mm,
                                            float dot_size_mm,
                                            float angle_deg)
{
    const float safe_coverage = clamp01(coverage);
    if (safe_coverage <= k_epsilon)
        return false;
    if (safe_coverage >= 1.f - k_epsilon)
        return true;
    if (!std::isfinite(surface_x_mm) || !std::isfinite(surface_y_mm))
        return safe_coverage >= 0.5f;

    constexpr float pi = 3.14159265358979323846f;
    const float radians = angle_deg * pi / 180.f;
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    const float x = c * surface_x_mm + s * surface_y_mm;
    const float y = -s * surface_x_mm + c * surface_y_mm;
    return halftone_threshold_for_texture_preview(x, y, dot_size_mm) + 0.5f < safe_coverage;
}

bool texture_preview_uses_halftone_dithering(const TexturePreviewSimulationSettings &settings)
{
    return settings.dithering_enabled &&
           settings.mapping_mode != int(TextureMappingZone::TextureMappingRawValues) &&
           is_halftone_dithering_method_for_texture_preview(settings.dithering_method);
}

int halftone_surface_axis_for_texture_preview(const Vec3f &normal)
{
    const Vec3f abs_normal(std::abs(normal.x()), std::abs(normal.y()), std::abs(normal.z()));
    if (abs_normal.x() >= abs_normal.y() && abs_normal.x() >= abs_normal.z())
        return 0;
    if (abs_normal.y() >= abs_normal.x() && abs_normal.y() >= abs_normal.z())
        return 1;
    return 2;
}

Vec2f halftone_surface_coords_for_texture_preview(const Vec3f &point, int axis)
{
    if (axis == 0)
        return Vec2f(point.y(), point.z());
    if (axis == 1)
        return Vec2f(point.x(), point.z());
    return Vec2f(point.x(), point.y());
}

Vec2f halftone_surface_coords_for_texture_preview(const Vec3f &point, const Vec3f &normal)
{
    return halftone_surface_coords_for_texture_preview(point, halftone_surface_axis_for_texture_preview(normal));
}

float halftone_surface_plane_coord_for_texture_preview(const Vec3f &point, int axis)
{
    if (axis == 0)
        return point.x();
    if (axis == 1)
        return point.y();
    return point.z();
}

std::vector<float> optimized_primary_component_weights_for_target(const std::array<float, 3>              &target_rgb,
                                                                  size_t                                   component_count,
                                                                  int                                      filament_color_mode,
                                                                  const std::vector<std::array<float, 3>> &component_colors,
                                                                  bool                                     force_sequential_filaments,
                                                                  const std::vector<size_t>               &semantic_component_indices)
{
    const int clamped_mode = std::clamp(filament_color_mode,
                                        int(TextureMappingZone::FilamentColorAny),
                                        int(TextureMappingZone::FilamentColorRGBKW));
    if (clamped_mode == int(TextureMappingZone::FilamentColorAny))
        return {};

    auto print_visibility_strength = [](float value) {
        return clamp01(std::pow(std::max(0.f, value), 0.85f));
    };

    const float r = clamp01(target_rgb[0]);
    const float g = clamp01(target_rgb[1]);
    const float b = clamp01(target_rgb[2]);
    const float whiteness = std::min({ r, g, b });
    const float darkness = 1.f - std::max({ r, g, b });

    auto safe_div = [](float numerator, float denominator) {
        if (denominator <= k_epsilon)
            return 0.f;
        return clamp01(numerator / denominator);
    };
    const auto component_index_for_role = [&semantic_component_indices](size_t role_idx) {
        if (role_idx < semantic_component_indices.size())
            return semantic_component_indices[role_idx];
        return role_idx;
    };

    std::vector<float> weights(component_count, 0.f);
    if (clamped_mode == int(TextureMappingZone::FilamentColorRGB)) {
        if (component_count != 3)
            return {};
        weights[component_index_for_role(0)] = print_visibility_strength(r);
        weights[component_index_for_role(1)] = print_visibility_strength(g);
        weights[component_index_for_role(2)] = print_visibility_strength(b);
        return weights;
    }
    if (clamped_mode == int(TextureMappingZone::FilamentColorCMY)) {
        if (component_count != 3)
            return {};
        weights[component_index_for_role(0)] = print_visibility_strength(1.f - r);
        weights[component_index_for_role(1)] = print_visibility_strength(1.f - g);
        weights[component_index_for_role(2)] = print_visibility_strength(1.f - b);
        return weights;
    }
    if (clamped_mode == int(TextureMappingZone::FilamentColorBW)) {
        if (component_count != 2)
            return {};

        const float gray = clamp01(0.2126f * r + 0.7152f * g + 0.0722f * b);
        const float black_strength = gray >= 0.5f ? (2.f * (1.f - gray)) : 1.f;
        const float white_strength = gray <= 0.5f ? (2.f * gray) : 1.f;

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

        weights[black_component_idx] = print_visibility_strength(black_strength);
        weights[white_component_idx] = print_visibility_strength(white_strength);
        return weights;
    }

    if (clamped_mode == int(TextureMappingZone::FilamentColorCMYKW)) {
        if (component_count != 5)
            return {};
        const float chroma = std::max(0.f, 1.f - darkness - whiteness);
        const float c = chroma <= k_epsilon ? 0.f : 1.f - safe_div(r - whiteness, chroma);
        const float m = chroma <= k_epsilon ? 0.f : 1.f - safe_div(g - whiteness, chroma);
        const float y = chroma <= k_epsilon ? 0.f : 1.f - safe_div(b - whiteness, chroma);
        weights[component_index_for_role(0)] = print_visibility_strength(c * chroma);
        weights[component_index_for_role(1)] = print_visibility_strength(m * chroma);
        weights[component_index_for_role(2)] = print_visibility_strength(y * chroma);
        weights[component_index_for_role(3)] = print_visibility_strength(darkness);
        weights[component_index_for_role(4)] = clamp01(std::pow(whiteness, 1.35f));
        return weights;
    }

    if (clamped_mode == int(TextureMappingZone::FilamentColorRGBKW)) {
        if (component_count != 5)
            return {};
        const float chroma = std::max(0.f, 1.f - darkness - whiteness);
        weights[component_index_for_role(0)] = print_visibility_strength(safe_div(r - whiteness, chroma) * chroma);
        weights[component_index_for_role(1)] = print_visibility_strength(safe_div(g - whiteness, chroma) * chroma);
        weights[component_index_for_role(2)] = print_visibility_strength(safe_div(b - whiteness, chroma) * chroma);
        weights[component_index_for_role(3)] = print_visibility_strength(darkness);
        weights[component_index_for_role(4)] = clamp01(std::pow(whiteness, 1.35f));
        return weights;
    }

    if (component_count != 4)
        return {};

    if (clamped_mode == int(TextureMappingZone::FilamentColorCMYK)) {
        const float k = clamp01(darkness);
        const float inv = 1.f - k;
        weights[component_index_for_role(0)] = print_visibility_strength(safe_div(1.f - r - k, inv));
        weights[component_index_for_role(1)] = print_visibility_strength(safe_div(1.f - g - k, inv));
        weights[component_index_for_role(2)] = print_visibility_strength(safe_div(1.f - b - k, inv));
        weights[component_index_for_role(3)] = print_visibility_strength(k);
        return weights;
    }
    if (clamped_mode == int(TextureMappingZone::FilamentColorCMYW)) {
        const float inv = 1.f - whiteness;
        const float r_no_w = safe_div(r - whiteness, inv);
        const float g_no_w = safe_div(g - whiteness, inv);
        const float b_no_w = safe_div(b - whiteness, inv);
        weights[component_index_for_role(0)] = print_visibility_strength((1.f - r_no_w) * inv);
        weights[component_index_for_role(1)] = print_visibility_strength((1.f - g_no_w) * inv);
        weights[component_index_for_role(2)] = print_visibility_strength((1.f - b_no_w) * inv);
        weights[component_index_for_role(3)] = clamp01(std::pow(whiteness, 1.35f));
        return weights;
    }
    if (clamped_mode == int(TextureMappingZone::FilamentColorRGBK)) {
        const float k = clamp01(darkness);
        const float inv = 1.f - k;
        weights[component_index_for_role(0)] = print_visibility_strength(safe_div(r - k, inv) * inv);
        weights[component_index_for_role(1)] = print_visibility_strength(safe_div(g - k, inv) * inv);
        weights[component_index_for_role(2)] = print_visibility_strength(safe_div(b - k, inv) * inv);
        weights[component_index_for_role(3)] = print_visibility_strength(k);
        return weights;
    }
    if (clamped_mode == int(TextureMappingZone::FilamentColorRGBW)) {
        const float inv = 1.f - whiteness;
        weights[component_index_for_role(0)] = print_visibility_strength(safe_div(r - whiteness, inv) * inv);
        weights[component_index_for_role(1)] = print_visibility_strength(safe_div(g - whiteness, inv) * inv);
        weights[component_index_for_role(2)] = print_visibility_strength(safe_div(b - whiteness, inv) * inv);
        weights[component_index_for_role(3)] = clamp01(std::pow(whiteness, 1.35f));
        return weights;
    }

    return {};
}

std::vector<float> component_weights_for_texture_preview(const TexturePreviewSimulationSettings &settings,
                                                         const std::array<float, 4>            &sample_rgba,
                                                         bool                                    apply_visibility_adjustments = true)
{
    const size_t component_count = settings.component_colors.size();
    if (component_count == 0)
        return {};

    if (settings.dithering_enabled &&
        settings.mapping_mode != int(TextureMappingZone::TextureMappingRawValues) &&
        settings.dithering_method != int(TextureMappingZone::DitheringHalftone) &&
        settings.dithering_method != int(TextureMappingZone::DitheringHalftoneIncreasedDetail)) {
        const std::vector<TexturePreviewBinaryDitherCandidate> candidates = binary_dither_candidates_for_texture_preview(settings);
        const std::array<float, 3> target_oklab = texture_preview_target_oklab(settings, sample_rgba);
        const size_t candidate_idx = nearest_binary_dither_candidate_for_texture_preview(candidates, target_oklab);
        if (candidate_idx < candidates.size())
            return binary_component_visibility_weights_for_texture_preview(settings, candidates[candidate_idx].mask);
    }

    std::array<float, 3> target = {
        clamp01(sample_rgba[0]),
        clamp01(sample_rgba[1]),
        clamp01(sample_rgba[2])
    };
    if (std::abs(settings.tone_gamma - 1.f) > 1e-5f) {
        target[0] = apply_texture_tone_gamma(target[0], settings.tone_gamma);
        target[1] = apply_texture_tone_gamma(target[1], settings.tone_gamma);
        target[2] = apply_texture_tone_gamma(target[2], settings.tone_gamma);
    }

    std::vector<float> desired(component_count, 0.f);
    size_t mapped_component_count = component_count;
    if (settings.mapping_mode == int(TextureMappingZone::TextureMappingRawValues)) {
        const float channels[3] = { target[0], target[1], target[2] };
        const size_t channel_count = std::min(component_count, size_t(3));
        for (size_t channel_idx = 0; channel_idx < channel_count; ++channel_idx)
            desired[channel_idx] = clamp01(channels[channel_idx]);
        mapped_component_count = channel_count;
    } else {
        std::vector<float> optimized;
        if (!settings.use_fixed_color_generic_solver)
            optimized = optimized_primary_component_weights_for_target(target,
                                                                       component_count,
                                                                       settings.filament_color_mode,
                                                                       settings.component_colors,
                                                                       settings.force_sequential_filaments,
                                                                       settings.semantic_component_indices);
        if (optimized.size() == component_count)
            desired = std::move(optimized);
        else {
            std::vector<float> best = settings.generic_mix_candidate_cache != nullptr &&
                                              !settings.generic_mix_candidate_cache->candidates.empty() ?
                solve_color_solver_weights_for_target(
                    settings.generic_mix_candidate_cache->candidates,
                    target,
                    color_solver_lookup_mode_from_index(settings.generic_solver_lookup_mode),
                    color_solver_mode_from_index(settings.generic_solver_mode)) :
                std::vector<float>{};
            if (best.size() == component_count)
                desired = std::move(best);
        }
    }

    const float contrast_factor = std::clamp(settings.contrast_pct, 25.f, 300.f) / 100.f;
    if (std::abs(contrast_factor - 1.f) > 1e-5f)
        apply_texture_contrast_to_mapped_components(desired, contrast_factor, mapped_component_count);

    if (settings.compact_offset_mode) {
        float max_weight = 0.f;
        for (const float value : desired)
            max_weight = std::max(max_weight, clamp01(value));
        if (max_weight > k_epsilon)
            for (float &value : desired)
                value = clamp01(value / max_weight);
    }

    if (apply_visibility_adjustments) {
        for (size_t idx = 0; idx < desired.size() && idx < settings.component_strength_factors.size(); ++idx)
            desired[idx] = clamp01(desired[idx] * settings.component_strength_factors[idx]);
        apply_minimum_visibility_offset(desired, settings.minimum_visibility_offset_factor);
    }
    return desired;
}

std::vector<float> raw_offset_print_width_weights_for_texture_preview(const TexturePreviewSimulationSettings &settings,
                                                                      const std::vector<float>              &raw_weights)
{
    const size_t component_count = raw_weights.size();
    std::vector<float> width_factors(component_count, 0.f);
    for (size_t idx = 0; idx < component_count; ++idx) {
        const float strength = idx < settings.component_strength_factors.size() ? settings.component_strength_factors[idx] : 1.f;
        const float minimum = idx < settings.component_minimum_offset_factors.size() ? settings.component_minimum_offset_factors[idx] : 0.f;
        const float adjusted_visibility = clamp01(minimum + clamp01(raw_weights[idx]) * strength * (1.f - minimum));
        if (settings.minimum_visibility_offset_factor > k_epsilon) {
            width_factors[idx] = apply_minimum_visibility_offset(adjusted_visibility, settings.minimum_visibility_offset_factor);
            continue;
        }
        width_factors[idx] = clamp01(settings.raw_offset_base_visibility_factor +
                                     settings.raw_offset_visibility_range_factor * adjusted_visibility);
    }

    if (settings.minimum_visibility_offset_factor > k_epsilon)
        return width_factors;

    const auto min_width = std::min_element(width_factors.begin(), width_factors.end());
    if (min_width == width_factors.end())
        return {};

    std::vector<float> weights(component_count, 0.f);
    const float shared_width = *min_width / float(component_count);
    for (size_t idx = 0; idx < component_count; ++idx)
        weights[idx] = clamp01(shared_width + std::max(0.f, width_factors[idx] - *min_width));
    return weights;
}

void prepare_texture_preview_simulation_settings(TexturePreviewSimulationSettings &settings)
{
    settings.use_fixed_color_generic_solver = texture_preview_uses_fixed_color_generic_solver(settings);
    settings.semantic_component_indices =
        semantic_component_indices_for_texture_preview(settings.component_colors,
                                                       settings.filament_color_mode,
                                                       settings.force_sequential_filaments);
    if (texture_preview_uses_generic_solver(settings)) {
        if (settings.generic_mix_candidate_cache == nullptr)
            settings.generic_mix_candidate_cache = std::make_shared<TexturePreviewGenericMixCandidateCache>();
        const std::vector<std::array<float, 3>> component_colors = generic_solver_component_colors(settings);
        const ColorSolverMixModel mix_model = color_solver_mix_model_from_index(settings.generic_solver_mix_model);
        std::shared_ptr<TexturePreviewGenericMixCandidateCache> cache = settings.generic_mix_candidate_cache;
        std::call_once(cache->init, [cache, component_colors, mix_model]() {
            cache->candidates = build_color_solver_candidates(component_colors, mix_model);
        });
    } else {
        settings.generic_mix_candidate_cache.reset();
    }
}

ColorRGBA simulated_texture_preview_color_for_vertex_color(const ColorRGBA *source_color,
                                                           const TexturePreviewSimulationSettings *settings)
{
    if (source_color == nullptr)
        return { 0.f, 0.f, 0.f, 1.f };
    if (settings == nullptr)
        return *source_color;

    const std::array<float, 4> sample_rgba = {
        source_color->r(),
        source_color->g(),
        source_color->b(),
        source_color->a()
    };
    const std::vector<float> component_weights = component_weights_for_texture_preview(*settings, sample_rgba);
    float activity = 0.f;
    for (const float weight : component_weights)
        activity = std::max(activity, clamp01(weight));

    if (activity <= k_epsilon)
        return *source_color;

    const std::array<float, 3> simulated_rgb =
        mix_color_solver_components(settings->component_colors,
                                    component_weights,
                                    color_solver_mix_model_from_index(settings->generic_solver_mix_model));
    return { simulated_rgb[0], simulated_rgb[1], simulated_rgb[2], source_color->a() };
}

std::optional<TexturePreviewSimulationSettings> texture_preview_simulation_settings_for_filament(unsigned int filament_id,
                                                                                                 size_t num_physical,
                                                                                                 const TextureMappingManager *texture_mgr,
                                                                                                 const std::vector<std::string> &physical_colors)
{
    const TextureMappingZone *zone = zone_for_filament(filament_id, num_physical, texture_mgr);
    if (zone == nullptr || !is_image_zone(*zone) || !zone->preview_simulate_colors)
        return std::nullopt;

    TexturePreviewSimulationSettings settings;
    settings.mapping_mode = std::clamp(zone->texture_mapping_mode,
                                       int(TextureMappingZone::TextureMappingFilamentBlending),
                                       int(TextureMappingZone::TextureMappingRawValues));
    settings.filament_color_mode = std::clamp(zone->filament_color_mode,
                                              int(TextureMappingZone::FilamentColorAny),
                                              int(TextureMappingZone::FilamentColorRGBKW));
    settings.force_sequential_filaments = zone->force_sequential_filaments;
    settings.limit_texture_resolution = zone->preview_limit_resolution;
    settings.dithering_enabled =
        zone->dithering_enabled &&
        settings.mapping_mode != int(TextureMappingZone::TextureMappingRawValues);
    settings.dithering_method = std::clamp(zone->dithering_method,
                                           int(TextureMappingZone::DitheringClosest),
                                           int(TextureMappingZone::DitheringHalftoneIncreasedDetail));
    settings.dithering_resolution_mm = std::clamp(zone->dithering_resolution_mm,
                                                  TextureMappingZone::MinDitheringResolutionMm,
                                                  TextureMappingZone::MaxDitheringResolutionMm);
    settings.halftone_dot_size_mm = std::clamp(zone->halftone_dot_size_mm,
                                               TextureMappingZone::MinHalftoneDotSizeMm,
                                               TextureMappingZone::MaxHalftoneDotSizeMm);
    const bool halftone_dithering_enabled =
        settings.dithering_enabled &&
        (settings.dithering_method == int(TextureMappingZone::DitheringHalftone) ||
         settings.dithering_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail));
    settings.compact_offset_mode =
        halftone_dithering_enabled ? false : zone->compact_offset_mode || settings.dithering_enabled;
    settings.use_legacy_fixed_color_mode = zone->use_legacy_fixed_color_mode;
    settings.contrast_pct = std::clamp(zone->contrast_pct, 25.f, 300.f);
    settings.tone_gamma = (!std::isfinite(zone->tone_gamma) || zone->tone_gamma <= 0.f) ?
        1.f :
        std::clamp(zone->tone_gamma, 0.5f, 3.f);
    settings.generic_solver_lookup_mode = std::clamp(zone->generic_solver_lookup_mode,
                                                     int(TextureMappingZone::GenericSolverClosestMix),
                                                     int(TextureMappingZone::GenericSolverBlendClosestTwo));
    settings.generic_solver_mode = std::clamp(zone->generic_solver_mode,
                                              int(TextureMappingZone::GenericSolverLegacy),
                                              int(TextureMappingZone::GenericSolverV2));
    settings.generic_solver_mix_model = TextureMappingZone::DefaultGenericSolverMixModel;
    settings.minimum_visibility_offset_factor = zone->minimum_visibility_offset_enabled ?
        std::clamp((std::isfinite(zone->minimum_visibility_offset_pct) ?
                        zone->minimum_visibility_offset_pct :
                        TextureMappingZone::DefaultMinimumVisibilityOffsetPct) / 100.f,
                   0.f,
                   1.f) :
        0.f;
    settings.component_ids = TextureMappingManager::effective_texture_component_ids(*zone, num_physical, physical_colors);
    if (settings.component_ids.empty())
        return std::nullopt;

    const bool raw_values_mode = settings.mapping_mode == int(TextureMappingZone::TextureMappingRawValues);
    settings.component_colors.reserve(settings.component_ids.size());
    settings.component_strength_factors.reserve(settings.component_ids.size());
    settings.component_minimum_offset_factors.reserve(settings.component_ids.size());
    for (const unsigned int component_id : settings.component_ids) {
        if (component_id == 0 || size_t(component_id - 1) >= physical_colors.size()) {
            if (!raw_values_mode)
                return std::nullopt;
            settings.component_colors.emplace_back(std::array<float, 3>{ 0.f, 0.f, 0.f });
        } else {
            settings.component_colors.emplace_back(decode_color(physical_colors[size_t(component_id - 1)]));
        }

        const size_t strength_idx = component_id > 0 ? size_t(component_id - 1) : size_t(0);
        const float strength_pct = strength_idx < zone->filament_strengths_pct.size() ?
            zone->filament_strengths_pct[strength_idx] :
            100.f;
        const float safe_strength_pct = std::isfinite(strength_pct) ? strength_pct : 100.f;
        settings.component_strength_factors.emplace_back(std::clamp(safe_strength_pct / 100.f, 0.f, 1.f));

        const size_t minimum_offset_idx = component_id > 0 ? size_t(component_id - 1) : size_t(-1);
        const float minimum_offset_pct = minimum_offset_idx < zone->filament_minimum_offsets_pct.size() ?
            zone->filament_minimum_offsets_pct[minimum_offset_idx] :
            0.f;
        const float safe_minimum_offset_pct = std::isfinite(minimum_offset_pct) ? minimum_offset_pct : 0.f;
        settings.component_minimum_offset_factors.emplace_back(std::clamp(safe_minimum_offset_pct / 100.f, 0.f, 1.f));
    }

    const std::array<float, 2> raw_offset_visibility_factors = raw_offset_visibility_factors_for_texture_preview(*zone);
    settings.raw_offset_base_visibility_factor = raw_offset_visibility_factors[0];
    settings.raw_offset_visibility_range_factor = raw_offset_visibility_factors[1];

    return settings.component_colors.empty() ? std::nullopt : std::optional<TexturePreviewSimulationSettings>(std::move(settings));
}

size_t texture_preview_simulation_signature(const ModelVolume &model_volume,
                                            size_t source_signature,
                                            const TexturePreviewSimulationSettings &settings)
{
    size_t signature = source_signature;
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };

    mix(reinterpret_cast<size_t>(&model_volume));
    mix(std::hash<int>{}(settings.mapping_mode));
    mix(std::hash<int>{}(settings.filament_color_mode));
    mix(std::hash<int>{}(settings.force_sequential_filaments ? 1 : 0));
    mix(std::hash<int>{}(settings.limit_texture_resolution ? 1 : 0));
    mix(std::hash<int>{}(settings.compact_offset_mode ? 1 : 0));
    mix(std::hash<int>{}(settings.use_legacy_fixed_color_mode ? 1 : 0));
    mix(std::hash<int>{}(settings.dithering_enabled ? 1 : 0));
    mix(std::hash<int>{}(settings.dithering_method));
    if (settings.dithering_method == int(TextureMappingZone::DitheringHalftone) ||
        settings.dithering_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail))
        mix(std::hash<int>{}(int(std::lround(settings.halftone_dot_size_mm * 1000.f))));
    else
        mix(std::hash<int>{}(int(std::lround(settings.dithering_resolution_mm * 1000.f))));
    mix(std::hash<int>{}(int(std::lround(settings.texture_preview_mm_per_pixel * 100000.f))));
    mix(std::hash<int>{}(settings.generic_solver_lookup_mode));
    mix(std::hash<int>{}(settings.generic_solver_mode));
    mix(std::hash<int>{}(settings.generic_solver_mix_model));
    mix(std::hash<int>{}(int(std::lround(settings.contrast_pct * 100.f))));
    mix(std::hash<int>{}(int(std::lround(settings.tone_gamma * 1000.f))));
    mix(std::hash<int>{}(int(std::lround(settings.minimum_visibility_offset_factor * 100000.f))));
    for (const unsigned int id : settings.component_ids)
        mix(std::hash<unsigned int>{}(id));
    for (const auto &color : settings.component_colors) {
        mix(std::hash<int>{}(int(std::lround(color[0] * 255.f))));
        mix(std::hash<int>{}(int(std::lround(color[1] * 255.f))));
        mix(std::hash<int>{}(int(std::lround(color[2] * 255.f))));
    }
    for (const float strength_factor : settings.component_strength_factors)
        mix(std::hash<int>{}(int(std::lround(strength_factor * 1000.f))));
    for (const float minimum_offset_factor : settings.component_minimum_offset_factors)
        mix(std::hash<int>{}(int(std::lround(minimum_offset_factor * 1000.f))));
    mix(std::hash<int>{}(int(std::lround(settings.raw_offset_base_visibility_factor * 100000.f))));
    mix(std::hash<int>{}(int(std::lround(settings.raw_offset_visibility_range_factor * 100000.f))));
    return signature;
}

TexturePreviewSimulationResult build_simulated_texture_preview_result(size_t signature,
                                                                      unsigned int width,
                                                                      unsigned int height,
                                                                      const std::vector<unsigned char> &source_rgba,
                                                                      const std::vector<unsigned char> &source_raw_offsets,
                                                                      unsigned int source_raw_channels,
                                                                      const std::vector<size_t> &source_raw_component_channels,
                                                                      ColorRGBA background_color,
                                                                      TexturePreviewSimulationSettings settings,
                                                                      unsigned int max_preview_edge = 0,
                                                                      size_t max_preview_pixels = 0)
{
    TexturePreviewSimulationResult result;
    result.signature = signature;
    if (width == 0 || height == 0 || source_rgba.size() < size_t(width) * size_t(height) * 4)
        return result;

    const std::array<unsigned int, 2> preview_size =
        simulated_texture_preview_size(width, height, settings, max_preview_edge, max_preview_pixels);
    result.width = preview_size[0];
    result.height = preview_size[1];
    result.rgba.resize(size_t(result.width) * size_t(result.height) * 4, 0);
    if (result.width == 0 || result.height == 0)
        return result;

    prepare_texture_preview_simulation_settings(settings);
    const bool use_raw_offsets =
        source_raw_component_channels.size() == settings.component_colors.size() &&
        source_raw_offsets.size() >= size_t(width) * size_t(height) * size_t(source_raw_channels);
    const bool use_binary_dithering =
        settings.dithering_enabled &&
        settings.mapping_mode != int(TextureMappingZone::TextureMappingRawValues) &&
        settings.dithering_method != int(TextureMappingZone::DitheringHalftone) &&
        settings.dithering_method != int(TextureMappingZone::DitheringHalftoneIncreasedDetail) &&
        !use_raw_offsets;
    const bool use_halftone_dithering =
        settings.dithering_enabled &&
        settings.mapping_mode != int(TextureMappingZone::TextureMappingRawValues) &&
        is_halftone_dithering_method_for_texture_preview(settings.dithering_method) &&
        !use_raw_offsets;
    const bool halftone_increased_detail =
        settings.dithering_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail);
    const float preview_mm_per_source_pixel =
        std::clamp(settings.texture_preview_mm_per_pixel, 0.005f, 5.f);
    const std::vector<TexturePreviewBinaryDitherCandidate> binary_dither_candidates =
        use_binary_dithering ? binary_dither_candidates_for_texture_preview(settings) :
                               std::vector<TexturePreviewBinaryDitherCandidate>{};

    std::unordered_map<unsigned int, std::array<unsigned char, 4>> simulated_color_cache;
    simulated_color_cache.reserve(std::min(size_t(result.width) * size_t(result.height), size_t(65536)));
    std::unordered_map<uint64_t, float> halftone_cell_tone_cache;
    if (use_halftone_dithering && !halftone_increased_detail)
        halftone_cell_tone_cache.reserve(std::min(size_t(result.width) * size_t(result.height), size_t(65536)));
    std::vector<std::array<float, 3>> floyd_error_current(result.width, { { 0.f, 0.f, 0.f } });
    std::vector<std::array<float, 3>> floyd_error_next(result.width, { { 0.f, 0.f, 0.f } });

    auto halftone_cell_cache_key = [](size_t component_idx, int cell_x, int cell_y) {
        uint64_t key = 1469598103934665603ull;
        auto mix = [&key](uint64_t value) {
            key ^= value;
            key *= 1099511628211ull;
        };
        mix(uint64_t(component_idx));
        mix(uint64_t(uint32_t(cell_x)));
        mix(uint64_t(uint32_t(cell_y)));
        return key;
    };

    auto component_coverage_at_source = [&](double src_x, double src_y, size_t component_idx) {
        const std::array<unsigned char, 4> source_rgba_sample =
            sample_texture_preview_rgba_bilinear_at_source(source_rgba, width, height, src_x, src_y);
        const ColorRGBA blended_source_color =
            composite_texture_mapping_color_over_background_for_preview(ColorRGBA(float(source_rgba_sample[0]) / 255.f,
                                                                                  float(source_rgba_sample[1]) / 255.f,
                                                                                  float(source_rgba_sample[2]) / 255.f,
                                                                                  float(source_rgba_sample[3]) / 255.f),
                                                                        background_color);
        const std::array<float, 4> sampled_rgba = {
            blended_source_color.r(),
            blended_source_color.g(),
            blended_source_color.b(),
            1.f
        };
        const std::vector<float> coverage = component_weights_for_texture_preview(settings, sampled_rgba, false);
        return component_idx < coverage.size() ? clamp01(coverage[component_idx]) : 0.f;
    };

    auto halftone_cell_coverage = [&](size_t component_idx,
                                      float  surface_x_mm,
                                      float  surface_y_mm,
                                      float  angle_deg,
                                      float  current_coverage) {
        if (halftone_increased_detail)
            return current_coverage;

        const float period = std::clamp(settings.halftone_dot_size_mm,
                                        TextureMappingZone::MinHalftoneDotSizeMm,
                                        TextureMappingZone::MaxHalftoneDotSizeMm);
        if (!std::isfinite(surface_x_mm) || !std::isfinite(surface_y_mm) || period <= k_epsilon)
            return current_coverage;

        constexpr float pi = 3.14159265358979323846f;
        const float radians = angle_deg * pi / 180.f;
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        const float screen_x = c * surface_x_mm + s * surface_y_mm;
        const float screen_y = -s * surface_x_mm + c * surface_y_mm;
        const int cell_x = int(std::floor(screen_x / period));
        const int cell_y = int(std::floor(screen_y / period));
        const uint64_t key = halftone_cell_cache_key(component_idx, cell_x, cell_y);
        const auto cached = halftone_cell_tone_cache.find(key);
        if (cached != halftone_cell_tone_cache.end())
            return cached->second;

        const float center_screen_x = (float(cell_x) + 0.5f) * period;
        const float center_screen_y = (float(cell_y) + 0.5f) * period;
        const float smooth_radius = period * 0.35f;
        const bool use_smoothing_samples = smooth_radius / preview_mm_per_source_pixel >= 1.5f;
        const std::array<Vec2f, 5> offsets = {
            Vec2f(0.f, 0.f),
            Vec2f(smooth_radius, 0.f),
            Vec2f(-smooth_radius, 0.f),
            Vec2f(0.f, smooth_radius),
            Vec2f(0.f, -smooth_radius)
        };
        const size_t sample_count = use_smoothing_samples ? offsets.size() : size_t(1);
        float total = 0.f;
        for (size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
            const float sample_screen_x = center_screen_x + offsets[sample_idx].x();
            const float sample_screen_y = center_screen_y + offsets[sample_idx].y();
            const float sample_surface_x = c * sample_screen_x - s * sample_screen_y;
            const float sample_surface_y = s * sample_screen_x + c * sample_screen_y;
            const double src_x = double(sample_surface_x) / double(preview_mm_per_source_pixel) - 0.5;
            const double src_y = double(sample_surface_y) / double(preview_mm_per_source_pixel) - 0.5;
            total += component_coverage_at_source(src_x, src_y, component_idx);
        }
        const float coverage = clamp01(total / float(sample_count));
        halftone_cell_tone_cache.emplace(key, coverage);
        return coverage;
    };

    for (unsigned int y = 0; y < result.height; ++y) {
        for (unsigned int x = 0; x < result.width; ++x) {
            const double src_x = (double(x) + 0.5) * double(width) / double(std::max(1u, result.width)) - 0.5;
            const double src_y = (double(y) + 0.5) * double(height) / double(std::max(1u, result.height)) - 0.5;
            const std::array<unsigned char, 4> source_rgba_sample =
                sample_texture_preview_rgba_bilinear_at_source(source_rgba, width, height, src_x, src_y);
            const ColorRGBA blended_source_color =
                composite_texture_mapping_color_over_background_for_preview(ColorRGBA(float(source_rgba_sample[0]) / 255.f,
                                                                                      float(source_rgba_sample[1]) / 255.f,
                                                                                      float(source_rgba_sample[2]) / 255.f,
                                                                                      float(source_rgba_sample[3]) / 255.f),
                                                                            background_color);
            const std::array<unsigned char, 3> source_rgb = {
                to_u8(blended_source_color.r()),
                to_u8(blended_source_color.g()),
                to_u8(blended_source_color.b())
            };
            const unsigned int cache_key = use_raw_offsets ?
                unsigned(std::numeric_limits<unsigned int>::max()) :
                texture_preview_rgb_cache_key(source_rgb);
            const size_t idx = (size_t(y) * size_t(result.width) + size_t(x)) * 4;

            auto cached_color = !use_raw_offsets && !use_binary_dithering && !use_halftone_dithering ?
                simulated_color_cache.find(cache_key) :
                simulated_color_cache.end();
            if (!use_raw_offsets && !use_binary_dithering && !use_halftone_dithering && cached_color != simulated_color_cache.end()) {
                result.rgba[idx + 0] = cached_color->second[0];
                result.rgba[idx + 1] = cached_color->second[1];
                result.rgba[idx + 2] = cached_color->second[2];
                result.rgba[idx + 3] = cached_color->second[3];
                continue;
            }

            const std::array<float, 4> sample_rgba = {
                blended_source_color.r(),
                blended_source_color.g(),
                blended_source_color.b(),
                1.f
            };
            std::vector<float> component_weights;
            if (use_raw_offsets) {
                const std::vector<float> raw_sample =
                    sample_texture_preview_raw_offsets_bilinear(source_raw_offsets,
                                                                width,
                                                                height,
                                                                source_raw_channels,
                                                                x,
                                                                y,
                                                                result.width,
                                                                result.height);
                component_weights = map_raw_sample_to_components_for_texture_preview(raw_sample, source_raw_component_channels);
                if (component_weights.size() == settings.component_colors.size())
                    component_weights = raw_offset_print_width_weights_for_texture_preview(settings, component_weights);
            } else {
                if (use_halftone_dithering) {
                    const std::vector<float> continuous_coverage =
                        component_weights_for_texture_preview(settings, sample_rgba, false);
                    uint32_t mask = 0;
                    const float surface_x_mm = float((src_x + 0.5) * double(preview_mm_per_source_pixel));
                    const float surface_y_mm = float((src_y + 0.5) * double(preview_mm_per_source_pixel));
                    for (size_t component_idx = 0; component_idx < settings.component_colors.size() && component_idx < 31; ++component_idx) {
                        const float angle_deg =
                            halftone_screen_angle_deg_for_texture_preview(settings.filament_color_mode, component_idx);
                        const float current_coverage = component_idx < continuous_coverage.size() ?
                            clamp01(continuous_coverage[component_idx]) :
                            0.f;
                        const float coverage =
                            halftone_cell_coverage(component_idx, surface_x_mm, surface_y_mm, angle_deg, current_coverage);
                        if (halftone_screen_on_for_texture_preview(coverage,
                                                                   surface_x_mm,
                                                                   surface_y_mm,
                                                                   settings.halftone_dot_size_mm,
                                                                   angle_deg))
                            mask |= uint32_t(1) << component_idx;
                    }
                    component_weights = halftone_component_visibility_weights_for_texture_preview(settings, mask);
                } else if (use_binary_dithering && !binary_dither_candidates.empty()) {
                    std::array<float, 3> target_oklab = texture_preview_target_oklab(settings, sample_rgba);
                    const int clamped_method = std::clamp(settings.dithering_method,
                                                          int(TextureMappingZone::DitheringClosest),
                                                          int(TextureMappingZone::DitheringHalftoneIncreasedDetail));
                    if (clamped_method == int(TextureMappingZone::DitheringFloydSteinberg)) {
                        for (size_t axis = 0; axis < 3; ++axis)
                            target_oklab[axis] += floyd_error_current[x][axis];
                    }

                    bool thresholded_dither = false;
                    float threshold = 0.f;
                    if (clamped_method == int(TextureMappingZone::DitheringOrderedBayer)) {
                        thresholded_dither = true;
                        threshold = ordered_bayer_threshold_for_texture_preview(int(x), int(y)) + 0.5f;
                    }

                    const size_t candidate_idx =
                        thresholded_dither ?
                            thresholded_binary_dither_candidate_for_texture_preview(binary_dither_candidates, target_oklab, threshold) :
                            nearest_binary_dither_candidate_for_texture_preview(binary_dither_candidates, target_oklab);
                    if (candidate_idx < binary_dither_candidates.size()) {
                        const TexturePreviewBinaryDitherCandidate &candidate = binary_dither_candidates[candidate_idx];
                        component_weights = binary_component_visibility_weights_for_texture_preview(settings, candidate.mask);
                        if (clamped_method == int(TextureMappingZone::DitheringFloydSteinberg)) {
                            const std::array<float, 3> error = {
                                target_oklab[0] - candidate.oklab[0],
                                target_oklab[1] - candidate.oklab[1],
                                target_oklab[2] - candidate.oklab[2]
                            };
                            auto add_error = [&error](std::array<float, 3> &dst, float factor) {
                                for (size_t axis = 0; axis < 3; ++axis)
                                    dst[axis] += error[axis] * factor;
                            };
                            if (x + 1 < result.width)
                                add_error(floyd_error_current[x + 1], 7.f / 16.f);
                            if (y + 1 < result.height) {
                                if (x > 0)
                                    add_error(floyd_error_next[x - 1], 3.f / 16.f);
                                add_error(floyd_error_next[x], 5.f / 16.f);
                                if (x + 1 < result.width)
                                    add_error(floyd_error_next[x + 1], 1.f / 16.f);
                            }
                        }
                    }
                } else {
                    component_weights = component_weights_for_texture_preview(settings, sample_rgba);
                }
            }
            float activity = 0.f;
            for (const float weight : component_weights)
                activity = std::max(activity, clamp01(weight));
            if (use_raw_offsets &&
                activity <= k_epsilon &&
                settings.minimum_visibility_offset_factor <= k_epsilon &&
                !component_weights.empty()) {
                std::fill(component_weights.begin(), component_weights.end(), 1.f);
                activity = 1.f;
            }

            const std::array<float, 3> simulated_rgb = activity > k_epsilon ?
                mix_color_solver_components(settings.component_colors,
                                            component_weights,
                                            use_halftone_dithering ?
                                                ColorSolverMixModel::PigmentPainter :
                                                color_solver_mix_model_from_index(settings.generic_solver_mix_model)) :
                std::array<float, 3>{ sample_rgba[0], sample_rgba[1], sample_rgba[2] };

            const std::array<unsigned char, 4> out_rgba = {
                to_u8(simulated_rgb[0]),
                to_u8(simulated_rgb[1]),
                to_u8(simulated_rgb[2]),
                255
            };
            if (!use_raw_offsets && !use_binary_dithering && !use_halftone_dithering)
                simulated_color_cache.emplace(cache_key, out_rgba);
            result.rgba[idx + 0] = out_rgba[0];
            result.rgba[idx + 1] = out_rgba[1];
            result.rgba[idx + 2] = out_rgba[2];
            result.rgba[idx + 3] = out_rgba[3];
        }
        if (use_binary_dithering &&
            std::clamp(settings.dithering_method,
                       int(TextureMappingZone::DitheringClosest),
                       int(TextureMappingZone::DitheringHalftoneIncreasedDetail)) == int(TextureMappingZone::DitheringFloydSteinberg)) {
            floyd_error_current.swap(floyd_error_next);
            std::fill(floyd_error_next.begin(), floyd_error_next.end(), std::array<float, 3>{ { 0.f, 0.f, 0.f } });
        }
    }

    return result;
}

std::unordered_map<size_t, std::shared_ptr<TexturePreviewSimulationCacheEntry>> &texture_preview_simulation_cache()
{
    static auto *cache = new std::unordered_map<size_t, std::shared_ptr<TexturePreviewSimulationCacheEntry>>();
    return *cache;
}

std::vector<std::future<TexturePreviewSimulationResult>> &abandoned_texture_preview_futures()
{
    static auto *futures = new std::vector<std::future<TexturePreviewSimulationResult>>();
    return *futures;
}

std::unordered_map<size_t, std::shared_ptr<TexturePreviewVertexColorSimulationCacheEntry>> &texture_preview_vertex_color_simulation_cache()
{
    static auto *cache = new std::unordered_map<size_t, std::shared_ptr<TexturePreviewVertexColorSimulationCacheEntry>>();
    return *cache;
}

std::vector<std::future<TexturePreviewVertexColorSimulationResult>> &abandoned_texture_preview_vertex_color_futures()
{
    static auto *futures = new std::vector<std::future<TexturePreviewVertexColorSimulationResult>>();
    return *futures;
}

size_t &texture_preview_vertex_color_simulation_generation()
{
    static auto *generation = new size_t(0);
    return *generation;
}

void discard_ready_texture_preview_future(std::future<TexturePreviewSimulationResult> &future, size_t &pending_signature)
{
    if (!future.valid() ||
        future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    try {
        (void) future.get();
    } catch (...) {
    }
    pending_signature = 0;
}

void discard_ready_texture_preview_vertex_color_future(TexturePreviewVertexColorSimulationCacheEntry &entry)
{
    if (!entry.pending_future.valid() ||
        entry.pending_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    try {
        (void) entry.pending_future.get();
    } catch (...) {
    }
    entry.pending_signature = 0;
}

bool prune_abandoned_texture_preview_futures()
{
    bool pending = false;
    auto &futures = abandoned_texture_preview_futures();
    for (auto it = futures.begin(); it != futures.end();) {
        if (!it->valid()) {
            it = futures.erase(it);
            continue;
        }

        if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            try {
                (void) it->get();
            } catch (...) {
            }
            it = futures.erase(it);
        } else {
            pending = true;
            ++it;
        }
    }
    return pending;
}

bool prune_abandoned_texture_preview_vertex_color_futures()
{
    bool pending = false;
    auto &futures = abandoned_texture_preview_vertex_color_futures();
    for (auto it = futures.begin(); it != futures.end();) {
        if (!it->valid()) {
            it = futures.erase(it);
            continue;
        }

        if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            try {
                (void) it->get();
            } catch (...) {
            }
            it = futures.erase(it);
        } else {
            pending = true;
            ++it;
        }
    }
    return pending;
}

} // namespace

void clear_texture_preview_simulation_cache()
{
    prune_abandoned_texture_preview_futures();
    prune_abandoned_texture_preview_vertex_color_futures();

    auto &cache = texture_preview_simulation_cache();
    for (auto it = cache.begin(); it != cache.end();) {
        std::shared_ptr<TexturePreviewSimulationCacheEntry> &entry = it->second;
        if (entry == nullptr) {
            it = cache.erase(it);
            continue;
        }

        if (entry->texture != nullptr) {
            entry->texture->reset();
            entry->texture.reset();
        }
        entry->uploaded_signature = 0;
        entry->pending_signature = 0;
        entry->temporary_pending_signature = 0;
        entry->pending_job_state.reset();

        if (entry->pending_future.valid()) {
            if (entry->pending_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                discard_ready_texture_preview_future(entry->pending_future, entry->pending_signature);
            } else {
                abandoned_texture_preview_futures().emplace_back(std::move(entry->pending_future));
            }
        }
        it = cache.erase(it);
    }

    auto &vertex_color_cache = texture_preview_vertex_color_simulation_cache();
    for (auto it = vertex_color_cache.begin(); it != vertex_color_cache.end();) {
        std::shared_ptr<TexturePreviewVertexColorSimulationCacheEntry> &entry = it->second;
        if (entry == nullptr) {
            it = vertex_color_cache.erase(it);
            continue;
        }

        entry->ready_geometry.reset();
        entry->ready_signature = 0;
        entry->pending_signature = 0;
        entry->ready_generation_reported = false;

        if (entry->pending_future.valid()) {
            if (entry->pending_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                discard_ready_texture_preview_vertex_color_future(*entry);
            } else {
                abandoned_texture_preview_vertex_color_futures().emplace_back(std::move(entry->pending_future));
            }
        }
        it = vertex_color_cache.erase(it);
    }
}

namespace {

bool texture_preview_simulation_is_pending_impl()
{
    bool pending = prune_abandoned_texture_preview_futures();
    pending = prune_abandoned_texture_preview_vertex_color_futures() || pending;
    auto &cache = texture_preview_simulation_cache();
    for (auto it = cache.begin(); it != cache.end();) {
        const std::shared_ptr<TexturePreviewSimulationCacheEntry> &entry = it->second;
        if (entry == nullptr) {
            it = cache.erase(it);
            continue;
        }

        const bool has_future = entry->pending_future.valid();
        if (has_future) {
            if (entry->pending_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                pending = true;
                entry->ready_frame_requested = false;
            } else if (entry->pending_signature != 0 && !entry->ready_frame_requested) {
                pending = true;
                entry->ready_frame_requested = true;
            }
        }

        if (!has_future && entry->texture == nullptr) {
            it = cache.erase(it);
            continue;
        }

        ++it;
    }

    auto &vertex_color_cache = texture_preview_vertex_color_simulation_cache();
    for (auto it = vertex_color_cache.begin(); it != vertex_color_cache.end();) {
        std::shared_ptr<TexturePreviewVertexColorSimulationCacheEntry> &entry = it->second;
        if (entry == nullptr) {
            it = vertex_color_cache.erase(it);
            continue;
        }

        bool completed_future = false;
        if (entry->pending_future.valid()) {
            if (entry->pending_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                pending = true;
                ++it;
                continue;
            }

            completed_future = true;
            try {
                TexturePreviewVertexColorSimulationResult result = entry->pending_future.get();
                if (result.signature == entry->pending_signature && !result.geometry.is_empty()) {
                    entry->ready_signature = result.signature;
                    entry->ready_geometry = std::move(result.geometry);
                    entry->ready_generation_reported = false;
                }
            } catch (...) {
            }
            entry->pending_signature = 0;
        }

        if (entry->ready_geometry.has_value()) {
            if (!entry->ready_generation_reported) {
                pending = true;
                ++texture_preview_vertex_color_simulation_generation();
                entry->ready_generation_reported = true;
            }
            ++it;
            continue;
        }

        if (completed_future) {
            pending = true;
            ++texture_preview_vertex_color_simulation_generation();
        }

        it = vertex_color_cache.erase(it);
    }
    return pending;
}

bool texture_preview_simulation_has_temporary_pending_impl()
{
    auto &cache = texture_preview_simulation_cache();
    for (auto it = cache.begin(); it != cache.end();) {
        const std::shared_ptr<TexturePreviewSimulationCacheEntry> &entry = it->second;
        if (entry == nullptr) {
            it = cache.erase(it);
            continue;
        }

        if (entry->pending_future.valid() && entry->temporary_pending_signature != 0)
            return true;

        ++it;
    }
    return false;
}

size_t texture_preview_simulation_cache_key(const ModelVolume &model_volume, unsigned int filament_id)
{
    size_t key = reinterpret_cast<size_t>(&model_volume);
    key ^= std::hash<unsigned int>{}(filament_id) + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
    return key;
}

bool upload_simulated_texture_preview_result(TexturePreviewSimulationCacheEntry &entry,
                                             TexturePreviewSimulationResult result,
                                             size_t expected_signature)
{
    if (result.signature != expected_signature || result.rgba.empty() || result.width == 0 || result.height == 0)
        return false;

    if (entry.texture == nullptr)
        entry.texture = std::make_unique<GUI::GLTexture>();
    else
        entry.texture->reset();

    if (!entry.texture->load_from_raw_data(std::move(result.rgba), result.width, result.height)) {
        entry.uploaded_signature = 0;
        return false;
    }

    configure_texture_preview_sampler(*entry.texture);
    entry.uploaded_signature = result.signature;
    return true;
}

void consume_temporary_simulated_texture_preview_result(TexturePreviewSimulationCacheEntry &entry)
{
    if (entry.temporary_pending_signature == 0 || entry.pending_job_state == nullptr)
        return;

    std::optional<TexturePreviewSimulationResult> result;
    {
        std::lock_guard<std::mutex> lock(entry.pending_job_state->mutex);
        if (!entry.pending_job_state->temporary_result.has_value())
            return;
        result = std::move(entry.pending_job_state->temporary_result);
        entry.pending_job_state->temporary_result.reset();
    }

    if (result.has_value())
        upload_simulated_texture_preview_result(entry, std::move(*result), entry.temporary_pending_signature);
    entry.temporary_pending_signature = 0;
}

const GUI::GLTexture *simulated_texture_preview_texture_for_filament(const ModelVolume &model_volume,
                                                                    unsigned int filament_id,
                                                                    size_t num_physical,
                                                                    const TextureMappingManager *texture_mgr,
                                                                    size_t source_texture_signature,
                                                                    const GUI::GLTexture &fallback_texture)
{
    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    std::optional<TexturePreviewSimulationSettings> settings =
        texture_preview_simulation_settings_for_filament(filament_id, num_physical, texture_mgr, physical_colors);
    if (!settings.has_value())
        return &fallback_texture;
    settings->texture_preview_mm_per_pixel = estimated_texture_preview_mm_per_pixel(model_volume);

    const ColorRGBA background_color = texture_mapping_background_color_for_preview(model_volume);
    const size_t simulation_signature = texture_preview_simulation_signature(model_volume, source_texture_signature, *settings);
    auto &cache = texture_preview_simulation_cache();
    const size_t cache_key = texture_preview_simulation_cache_key(model_volume, filament_id);
    std::shared_ptr<TexturePreviewSimulationCacheEntry> &entry_ref = cache[cache_key];
    if (entry_ref == nullptr)
        entry_ref = std::make_shared<TexturePreviewSimulationCacheEntry>();
    TexturePreviewSimulationCacheEntry &entry = *entry_ref;

    const unsigned int width = model_volume.imported_texture_width;
    const unsigned int height = model_volume.imported_texture_height;
    const bool needs_temporary_result = simulated_texture_preview_needs_temporary_result(width, height, *settings);
    const size_t temporary_signature = needs_temporary_result ?
        temporary_simulated_texture_preview_signature(simulation_signature) :
        size_t(0);

    consume_temporary_simulated_texture_preview_result(entry);

    if (entry.pending_future.valid() && entry.pending_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        TexturePreviewSimulationResult result = entry.pending_future.get();
        upload_simulated_texture_preview_result(entry, std::move(result), entry.pending_signature);
        entry.pending_signature = 0;
        entry.temporary_pending_signature = 0;
        entry.ready_frame_requested = false;
        entry.pending_job_state.reset();
    }

    if (entry.texture != nullptr && entry.uploaded_signature == simulation_signature && entry.texture->get_id() != 0)
        return entry.texture.get();

    if (!entry.pending_future.valid()) {
        entry.pending_signature = simulation_signature;
        entry.temporary_pending_signature = temporary_signature;
        entry.ready_frame_requested = false;
        entry.pending_job_state = std::make_shared<TexturePreviewSimulationJobState>();
        auto source_rgba = std::make_shared<std::vector<unsigned char>>(model_volume.imported_texture_rgba.begin(),
                                                                        model_volume.imported_texture_rgba.end());
        auto source_raw_offsets = std::make_shared<std::vector<unsigned char>>(model_volume.imported_texture_raw_filament_offsets.begin(),
                                                                              model_volume.imported_texture_raw_filament_offsets.end());
        const unsigned int source_raw_channels = model_volume.imported_texture_raw_channels;
        TexturePreviewSimulationSettings simulation_settings = *settings;
        simulation_settings.generic_mix_candidate_cache = std::make_shared<TexturePreviewGenericMixCandidateCache>();
        std::vector<size_t> source_raw_component_channels =
            raw_component_source_channels_for_texture_preview(model_volume.imported_texture_raw_metadata_json,
                                                              source_raw_channels,
                                                              simulation_settings.filament_color_mode,
                                                              simulation_settings.component_colors.size(),
                                                              simulation_settings.component_colors);
        std::shared_ptr<TexturePreviewSimulationJobState> job_state = entry.pending_job_state;
        entry.pending_future = std::async(std::launch::async,
                                          [simulation_signature,
                                           temporary_signature,
                                           width,
                                           height,
                                           source_rgba,
                                           source_raw_offsets,
                                           source_raw_channels,
                                           source_raw_component_channels = std::move(source_raw_component_channels),
                                           background_color,
                                           job_state,
                                           simulation_settings = std::move(simulation_settings)]() mutable {
                                              if (temporary_signature != 0 && job_state != nullptr) {
                                                  const unsigned int temp_edge = k_temporary_simulated_texture_preview_max_edge;
                                                  const size_t temp_pixels = k_temporary_simulated_texture_preview_max_pixels;
                                                  TexturePreviewSimulationResult temporary_result =
                                                      build_simulated_texture_preview_result(temporary_signature,
                                                                                             width,
                                                                                             height,
                                                                                             *source_rgba,
                                                                                             *source_raw_offsets,
                                                                                             source_raw_channels,
                                                                                             source_raw_component_channels,
                                                                                             background_color,
                                                                                             simulation_settings,
                                                                                             temp_edge,
                                                                                             temp_pixels);
                                                  std::lock_guard<std::mutex> lock(job_state->mutex);
                                                  job_state->temporary_result = std::move(temporary_result);
                                              }
                                              return build_simulated_texture_preview_result(simulation_signature,
                                                                                           width,
                                                                                           height,
                                                                                           *source_rgba,
                                                                                           *source_raw_offsets,
                                                                                           source_raw_channels,
                                                                                           source_raw_component_channels,
                                                                                           background_color,
                                                                                           std::move(simulation_settings));
                                          });
    }

    if (temporary_signature != 0 &&
        entry.texture != nullptr &&
        entry.uploaded_signature == temporary_signature &&
        entry.texture->get_id() != 0)
        return entry.texture.get();

    return &fallback_texture;
}

} // namespace

bool build_texture_preview_model_for_state(const ModelVolume                                      &model_volume,
                                           const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
                                           GUI::GLModel                                            &out_model)
{
    if (!model_volume_has_texture_preview_data(model_volume) || state_triangles.empty())
        return false;

    const indexed_triangle_set &its = model_volume.mesh().its;
    GUI::GLModel::Geometry geometry;
    geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3T2 };
    geometry.reserve_vertices(state_triangles.size() * 3);
    geometry.reserve_indices(state_triangles.size() * 3);

    unsigned int vertex_index = 0;
    for (const TriangleSelector::FacetStateTriangle &triangle : state_triangles) {
        if (triangle.source_triangle < 0)
            continue;

        const size_t source_triangle = size_t(triangle.source_triangle);
        if (source_triangle >= its.indices.size() ||
            source_triangle >= model_volume.imported_texture_uv_valid.size() ||
            model_volume.imported_texture_uv_valid[source_triangle] == 0)
            continue;

        const size_t uv_offset = source_triangle * 6;
        if (uv_offset + 5 >= model_volume.imported_texture_uvs_per_face.size())
            continue;

        const stl_triangle_vertex_indices &source_indices = its.indices[source_triangle];
        if (source_indices[0] < 0 || source_indices[1] < 0 || source_indices[2] < 0)
            continue;
        if (size_t(source_indices[0]) >= its.vertices.size() ||
            size_t(source_indices[1]) >= its.vertices.size() ||
            size_t(source_indices[2]) >= its.vertices.size())
            continue;

        const Vec3f source_p0 = its.vertices[size_t(source_indices[0])].cast<float>();
        const Vec3f source_p1 = its.vertices[size_t(source_indices[1])].cast<float>();
        const Vec3f source_p2 = its.vertices[size_t(source_indices[2])].cast<float>();
        const std::array<Vec2f, 3> source_uvs = unwrap_triangle_uvs(
            Vec2f(model_volume.imported_texture_uvs_per_face[uv_offset + 0], model_volume.imported_texture_uvs_per_face[uv_offset + 1]),
            Vec2f(model_volume.imported_texture_uvs_per_face[uv_offset + 2], model_volume.imported_texture_uvs_per_face[uv_offset + 3]),
            Vec2f(model_volume.imported_texture_uvs_per_face[uv_offset + 4], model_volume.imported_texture_uvs_per_face[uv_offset + 5]));

        Vec3f normal = (triangle.vertices[1] - triangle.vertices[0]).cross(triangle.vertices[2] - triangle.vertices[0]);
        const float normal_len = normal.norm();
        if (normal_len <= k_epsilon)
            continue;
        normal /= normal_len;
        const Vec3f offset = normal * k_preview_offset;

        std::array<Vec2f, 3> leaf_uvs;
        bool valid_leaf = true;
        for (size_t vertex_idx = 0; vertex_idx < triangle.vertices.size(); ++vertex_idx) {
            Vec3f barycentric = Vec3f::Zero();
            if (!barycentric_weights(triangle.vertices[vertex_idx], source_p0, source_p1, source_p2, barycentric)) {
                valid_leaf = false;
                break;
            }
            leaf_uvs[vertex_idx] = source_uvs[0] * barycentric.x() + source_uvs[1] * barycentric.y() + source_uvs[2] * barycentric.z();
        }
        if (!valid_leaf)
            continue;

        for (size_t vertex_idx = 0; vertex_idx < triangle.vertices.size(); ++vertex_idx)
            geometry.add_vertex(triangle.vertices[vertex_idx] + offset, normal, leaf_uvs[vertex_idx]);
        geometry.add_triangle(vertex_index, vertex_index + 1, vertex_index + 2);
        vertex_index += 3;
    }

    if (geometry.is_empty())
        return false;

    out_model.init_from(std::move(geometry));
    return true;
}

namespace {

bool build_vertex_color_preview_geometry_for_state(const indexed_triangle_set                              &its,
                                                   const std::vector<uint32_t>                            &vertex_colors_rgba,
                                                   const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
                                                   const TexturePreviewSimulationSettings                  *simulation_settings,
                                                   const ColorRGBA                                         &background_color,
                                                   GUI::GLModel::Geometry                                  &geometry)
{
    if (vertex_colors_rgba.empty() || vertex_colors_rgba.size() != its.vertices.size() || state_triangles.empty())
        return false;

    geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };
    geometry.reserve_vertices(state_triangles.size() * 3);
    geometry.reserve_indices(state_triangles.size() * 3);

    std::unordered_map<uint32_t, ColorRGBA> simulated_color_cache;
    if (simulation_settings != nullptr)
        simulated_color_cache.reserve(std::min(vertex_colors_rgba.size(), size_t(65536)));
    auto preview_color = [simulation_settings, &simulated_color_cache, background_color](const ColorRGBA &source_color) {
        const ColorRGBA blended_source =
            composite_texture_mapping_color_over_background_for_preview(source_color, background_color);
        if (simulation_settings == nullptr)
            return blended_source;

        const uint32_t key = (uint32_t(std::clamp(blended_source.r(), 0.f, 1.f) * 255.f + 0.5f) << 24) |
                             (uint32_t(std::clamp(blended_source.g(), 0.f, 1.f) * 255.f + 0.5f) << 16) |
                             (uint32_t(std::clamp(blended_source.b(), 0.f, 1.f) * 255.f + 0.5f) << 8) |
                             uint32_t(std::clamp(blended_source.a(), 0.f, 1.f) * 255.f + 0.5f);
        auto cached = simulated_color_cache.find(key);
        if (cached != simulated_color_cache.end())
            return cached->second;

        const ColorRGBA simulated_color = simulated_texture_preview_color_for_vertex_color(&blended_source, simulation_settings);
        simulated_color_cache.emplace(key, simulated_color);
        return simulated_color;
    };
    auto source_vertex_color = [](uint32_t packed) {
        return unpack_vertex_color(packed);
    };

    unsigned int vertex_index = 0;
    for (const TriangleSelector::FacetStateTriangle &triangle : state_triangles) {
        if (triangle.source_triangle < 0)
            continue;

        const size_t source_triangle = size_t(triangle.source_triangle);
        if (source_triangle >= its.indices.size())
            continue;

        const stl_triangle_vertex_indices &source_indices = its.indices[source_triangle];
        if (source_indices[0] < 0 || source_indices[1] < 0 || source_indices[2] < 0)
            continue;
        if (size_t(source_indices[0]) >= its.vertices.size() ||
            size_t(source_indices[1]) >= its.vertices.size() ||
            size_t(source_indices[2]) >= its.vertices.size() ||
            size_t(source_indices[0]) >= vertex_colors_rgba.size() ||
            size_t(source_indices[1]) >= vertex_colors_rgba.size() ||
            size_t(source_indices[2]) >= vertex_colors_rgba.size())
            continue;

        const Vec3f source_p0 = its.vertices[size_t(source_indices[0])].cast<float>();
        const Vec3f source_p1 = its.vertices[size_t(source_indices[1])].cast<float>();
        const Vec3f source_p2 = its.vertices[size_t(source_indices[2])].cast<float>();
        const std::array<ColorRGBA, 3> source_colors = {
            source_vertex_color(vertex_colors_rgba[size_t(source_indices[0])]),
            source_vertex_color(vertex_colors_rgba[size_t(source_indices[1])]),
            source_vertex_color(vertex_colors_rgba[size_t(source_indices[2])])
        };

        Vec3f normal = (triangle.vertices[1] - triangle.vertices[0]).cross(triangle.vertices[2] - triangle.vertices[0]);
        const float normal_len = normal.norm();
        if (normal_len <= k_epsilon)
            continue;
        normal /= normal_len;
        const Vec3f offset = normal * k_preview_offset;

        std::array<ColorRGBA, 3> leaf_colors;
        bool valid_leaf = true;
        for (size_t vertex_idx = 0; vertex_idx < triangle.vertices.size(); ++vertex_idx) {
            Vec3f barycentric = Vec3f::Zero();
            if (!barycentric_weights(triangle.vertices[vertex_idx], source_p0, source_p1, source_p2, barycentric)) {
                valid_leaf = false;
                break;
            }
            leaf_colors[vertex_idx] = preview_color(interpolate_color(source_colors, barycentric));
        }
        if (!valid_leaf)
            continue;

        for (size_t vertex_idx = 0; vertex_idx < triangle.vertices.size(); ++vertex_idx)
            geometry.add_vertex(triangle.vertices[vertex_idx] + offset, normal, leaf_colors[vertex_idx]);
        geometry.add_triangle(vertex_index, vertex_index + 1, vertex_index + 2);
        vertex_index += 3;
    }

    if (geometry.is_empty())
        return false;

    return true;
}

bool build_vertex_color_preview_model_for_state(const ModelVolume                                      &model_volume,
                                                const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
                                                const TexturePreviewSimulationSettings                   *simulation_settings,
                                                GUI::GLModel                                            &out_model)
{
    if (!model_volume_has_vertex_color_preview_data(model_volume) || state_triangles.empty())
        return false;

    GUI::GLModel::Geometry geometry;
    if (!build_vertex_color_preview_geometry_for_state(model_volume.mesh().its,
                                                       model_volume.imported_vertex_colors_rgba,
                                                       state_triangles,
                                                       simulation_settings,
                                                       texture_mapping_background_color_for_preview(model_volume),
                                                       geometry))
        return false;

    out_model.init_from(std::move(geometry));
    return true;
}

std::optional<ColorRGBA> sample_texture_mapping_color_preview(
    const std::vector<ColorFacetTriangle>     &color_facets,
    const std::unordered_map<int, std::vector<size_t>> &facets_by_source_triangle,
    int                                        source_triangle,
    const Vec3f                               &point)
{
    auto found = facets_by_source_triangle.find(source_triangle);
    if (found == facets_by_source_triangle.end() || found->second.empty())
        return std::nullopt;

    const float tolerance = -1e-4f;
    for (const size_t facet_idx : found->second) {
        if (facet_idx >= color_facets.size())
            continue;

        const ColorFacetTriangle &facet = color_facets[facet_idx];
        Vec3f weights = Vec3f::Zero();
        if (!barycentric_weights(point, facet.vertices[0], facet.vertices[1], facet.vertices[2], weights))
            continue;
        if (weights.x() >= tolerance && weights.y() >= tolerance && weights.z() >= tolerance)
            return unpack_vertex_color(facet.rgba);
    }

    return unpack_vertex_color(color_facets[found->second.front()].rgba);
}

std::vector<Vec3f> clip_preview_triangle_to_triangle(const std::array<Vec3f, 3> &subject, const std::array<Vec3f, 3> &clip)
{
    std::vector<Vec3f> polygon(subject.begin(), subject.end());
    const float tolerance = -1e-5f;

    for (size_t side = 0; side < 3 && !polygon.empty(); ++side) {
        std::vector<Vec3f> clipped;
        clipped.reserve(polygon.size() + 1);

        auto weight_for_side = [&clip, side](const Vec3f &point) {
            Vec3f weights = Vec3f::Zero();
            if (!barycentric_weights(point, clip[0], clip[1], clip[2], weights))
                return -std::numeric_limits<float>::max();
            return weights[side];
        };

        Vec3f previous = polygon.back();
        float previous_weight = weight_for_side(previous);
        bool previous_inside = previous_weight >= tolerance;

        for (const Vec3f &current : polygon) {
            const float current_weight = weight_for_side(current);
            const bool current_inside = current_weight >= tolerance;

            if (current_inside != previous_inside) {
                const float denom = previous_weight - current_weight;
                if (std::abs(denom) > k_epsilon) {
                    const float t = std::clamp(previous_weight / denom, 0.f, 1.f);
                    clipped.emplace_back(previous + (current - previous) * t);
                }
            }

            if (current_inside)
                clipped.emplace_back(current);

            previous = current;
            previous_weight = current_weight;
            previous_inside = current_inside;
        }

        polygon = std::move(clipped);
    }

    return polygon;
}

bool preview_polygon_has_area(const std::vector<Vec3f> &polygon, const Vec3f &normal)
{
    if (polygon.size() < 3)
        return false;

    float area = 0.f;
    for (size_t idx = 1; idx + 1 < polygon.size(); ++idx)
        area += std::abs((polygon[idx] - polygon[0]).cross(polygon[idx + 1] - polygon[0]).dot(normal));
    return area > k_epsilon;
}

bool build_texture_mapping_color_preview_geometry_for_state(
    const indexed_triangle_set                              &its,
    const ColorFacetsAnnotation                             &color_source,
    const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
    const TexturePreviewSimulationSettings                  *simulation_settings,
    const ColorRGBA                                         &background_color,
    GUI::GLModel::Geometry                                  &geometry)
{
    if (color_source.empty() || state_triangles.empty())
        return false;

    std::vector<ColorFacetTriangle> color_facets;
    color_source.get_facet_triangles(its, color_facets);
    if (color_facets.empty())
        return false;

    std::unordered_map<int, std::vector<size_t>> facets_by_source_triangle;
    facets_by_source_triangle.reserve(color_facets.size());
    for (size_t idx = 0; idx < color_facets.size(); ++idx)
        facets_by_source_triangle[color_facets[idx].source_triangle].emplace_back(idx);

    geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };
    geometry.reserve_vertices(color_facets.size() * 3);
    geometry.reserve_indices(color_facets.size() * 3);

    std::unordered_map<uint32_t, ColorRGBA> simulated_color_cache;
    if (simulation_settings != nullptr)
        simulated_color_cache.reserve(std::min(color_facets.size(), size_t(65536)));
    auto preview_color = [simulation_settings, &simulated_color_cache, background_color](const ColorRGBA &source_color) {
        const ColorRGBA blended_source =
            composite_texture_mapping_color_over_background_for_preview(source_color, background_color);
        if (simulation_settings == nullptr)
            return blended_source;

        const uint32_t key = (uint32_t(std::clamp(blended_source.r(), 0.f, 1.f) * 255.f + 0.5f) << 24) |
                             (uint32_t(std::clamp(blended_source.g(), 0.f, 1.f) * 255.f + 0.5f) << 16) |
                             (uint32_t(std::clamp(blended_source.b(), 0.f, 1.f) * 255.f + 0.5f) << 8) |
                             uint32_t(std::clamp(blended_source.a(), 0.f, 1.f) * 255.f + 0.5f);
        auto cached = simulated_color_cache.find(key);
        if (cached != simulated_color_cache.end())
            return cached->second;

        const ColorRGBA simulated_color = simulated_texture_preview_color_for_vertex_color(&blended_source, simulation_settings);
        simulated_color_cache.emplace(key, simulated_color);
        return simulated_color;
    };

    unsigned int vertex_index = 0;
    for (const TriangleSelector::FacetStateTriangle &triangle : state_triangles) {
        if (triangle.source_triangle < 0)
            continue;

        const Vec3f edge_0 = triangle.vertices[1] - triangle.vertices[0];
        const Vec3f edge_1 = triangle.vertices[2] - triangle.vertices[0];
        Vec3f normal = edge_0.cross(edge_1);
        const float normal_len = normal.norm();
        if (normal_len <= k_epsilon)
            continue;
        normal /= normal_len;
        const Vec3f offset = normal * k_preview_offset;

        bool emitted_color_facets = false;
        auto color_facets_for_triangle = facets_by_source_triangle.find(triangle.source_triangle);
        if (color_facets_for_triangle != facets_by_source_triangle.end()) {
            for (const size_t facet_idx : color_facets_for_triangle->second) {
                if (facet_idx >= color_facets.size())
                    continue;

                const ColorFacetTriangle &facet = color_facets[facet_idx];
                const std::vector<Vec3f> clipped = clip_preview_triangle_to_triangle(facet.vertices, triangle.vertices);
                if (!preview_polygon_has_area(clipped, normal))
                    continue;

                const ColorRGBA color = preview_color(unpack_vertex_color(facet.rgba));
                for (size_t poly_idx = 1; poly_idx + 1 < clipped.size(); ++poly_idx) {
                    geometry.add_vertex(clipped[0] + offset, normal, color);
                    geometry.add_vertex(clipped[poly_idx] + offset, normal, color);
                    geometry.add_vertex(clipped[poly_idx + 1] + offset, normal, color);
                    geometry.add_triangle(vertex_index, vertex_index + 1, vertex_index + 2);
                    vertex_index += 3;
                }
                emitted_color_facets = true;
            }
        }

        if (emitted_color_facets)
            continue;

        if (color_facets_for_triangle == facets_by_source_triangle.end() || color_facets_for_triangle->second.empty()) {
            const ColorRGBA color = preview_color(background_color);
            geometry.add_vertex(triangle.vertices[0] + offset, normal, color);
            geometry.add_vertex(triangle.vertices[1] + offset, normal, color);
            geometry.add_vertex(triangle.vertices[2] + offset, normal, color);
            geometry.add_triangle(vertex_index, vertex_index + 1, vertex_index + 2);
            vertex_index += 3;
            continue;
        }

        std::array<ColorRGBA, 3> leaf_colors;
        bool valid_leaf = true;
        for (size_t vertex_idx = 0; vertex_idx < triangle.vertices.size(); ++vertex_idx) {
            std::optional<ColorRGBA> sampled =
                sample_texture_mapping_color_preview(color_facets,
                                                     facets_by_source_triangle,
                                                     triangle.source_triangle,
                                                     triangle.vertices[vertex_idx]);
            if (!sampled) {
                valid_leaf = false;
                break;
            }
            leaf_colors[vertex_idx] = preview_color(*sampled);
        }

        if (!valid_leaf)
            continue;

        geometry.add_vertex(triangle.vertices[0] + offset, normal, leaf_colors[0]);
        geometry.add_vertex(triangle.vertices[1] + offset, normal, leaf_colors[1]);
        geometry.add_vertex(triangle.vertices[2] + offset, normal, leaf_colors[2]);
        geometry.add_triangle(vertex_index, vertex_index + 1, vertex_index + 2);
        vertex_index += 3;
    }

    if (geometry.is_empty())
        return false;

    return true;
}

bool build_texture_mapping_color_preview_model_for_state(
    const ModelVolume                                      &model_volume,
    const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
    const TexturePreviewSimulationSettings                  *simulation_settings,
    GUI::GLModel                                            &out_model,
    const ColorFacetsAnnotation                             *texture_mapping_color_facets_override = nullptr)
{
    const ColorFacetsAnnotation *color_source = texture_mapping_color_facets_override;
    if (color_source == nullptr || color_source->empty())
        color_source = &model_volume.texture_mapping_color_facets;
    if (color_source == nullptr || color_source->empty() || state_triangles.empty())
        return false;

    GUI::GLModel::Geometry geometry;
    if (!build_texture_mapping_color_preview_geometry_for_state(model_volume.mesh().its,
                                                                *color_source,
                                                                state_triangles,
                                                                simulation_settings,
                                                                texture_mapping_background_color_for_preview(model_volume, color_source),
                                                                geometry))
        return false;

    out_model.init_from(std::move(geometry));
    return true;
}

float image_texture_halftone_preview_step_mm(const TexturePreviewSimulationSettings &settings)
{
    const float dot_size = std::clamp(settings.halftone_dot_size_mm,
                                      TextureMappingZone::MinHalftoneDotSizeMm,
                                      TextureMappingZone::MaxHalftoneDotSizeMm);
    return std::clamp(dot_size * 0.125f, 0.18f, 0.75f);
}

int image_texture_halftone_preview_max_subdivision_steps(size_t triangle_count)
{
    if (triangle_count <= 200)
        return 96;
    if (triangle_count <= 2000)
        return 48;
    return 24;
}

uint64_t image_texture_halftone_cell_cache_key(size_t component_idx, int axis, int plane_bucket, int cell_x, int cell_y)
{
    uint64_t key = 1469598103934665603ull;
    auto mix = [&key](uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    mix(uint64_t(component_idx));
    mix(uint64_t(uint32_t(axis)));
    mix(uint64_t(uint32_t(plane_bucket)));
    mix(uint64_t(uint32_t(cell_x)));
    mix(uint64_t(uint32_t(cell_y)));
    return key;
}

bool build_image_texture_halftone_preview_geometry_for_state(
    const indexed_triangle_set                              &its,
    const std::vector<unsigned char>                        &source_rgba,
    unsigned int                                             width,
    unsigned int                                             height,
    const std::vector<float>                                &texture_uvs_per_face,
    const std::vector<uint8_t>                              &texture_uv_valid,
    const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
    const TexturePreviewSimulationSettings                  &settings,
    const ColorRGBA                                         &background_color,
    const Transform3d                                       &world_matrix,
    GUI::GLModel::Geometry                                  &geometry)
{
    if (state_triangles.empty() ||
        width == 0 ||
        height == 0 ||
        source_rgba.size() < size_t(width) * size_t(height) * 4 ||
        texture_uv_valid.size() < its.indices.size() ||
        texture_uvs_per_face.size() < its.indices.size() * 6)
        return false;

    geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };
    geometry.reserve_vertices(state_triangles.size() * 12);
    geometry.reserve_indices(state_triangles.size() * 12);

    const bool halftone_increased_detail =
        settings.dithering_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail);
    const float period = std::clamp(settings.halftone_dot_size_mm,
                                    TextureMappingZone::MinHalftoneDotSizeMm,
                                    TextureMappingZone::MaxHalftoneDotSizeMm);
    const float target_step_mm = image_texture_halftone_preview_step_mm(settings);
    const int max_subdivision_steps = image_texture_halftone_preview_max_subdivision_steps(state_triangles.size());
    std::unordered_map<uint64_t, float> halftone_cell_tone_cache;
    if (!halftone_increased_detail)
        halftone_cell_tone_cache.reserve(std::min(state_triangles.size() * size_t(8), size_t(65536)));

    auto source_coverage_at_uv = [&](const Vec2f &uv, size_t component_idx) {
        const std::array<unsigned char, 4> source_rgba_sample =
            sample_texture_preview_rgba_bilinear_at_source(source_rgba,
                                                           width,
                                                           height,
                                                           double(uv.x()) * double(width) - 0.5,
                                                           double(uv.y()) * double(height) - 0.5);
        const ColorRGBA blended_source_color =
            composite_texture_mapping_color_over_background_for_preview(ColorRGBA(float(source_rgba_sample[0]) / 255.f,
                                                                                  float(source_rgba_sample[1]) / 255.f,
                                                                                  float(source_rgba_sample[2]) / 255.f,
                                                                                  float(source_rgba_sample[3]) / 255.f),
                                                                        background_color);
        const std::array<float, 4> sampled_rgba = {
            blended_source_color.r(),
            blended_source_color.g(),
            blended_source_color.b(),
            1.f
        };
        const std::vector<float> coverage = component_weights_for_texture_preview(settings, sampled_rgba, false);
        return component_idx < coverage.size() ? clamp01(coverage[component_idx]) : 0.f;
    };

    auto interpolate_point = [](const std::array<Vec3f, 3> &vertices, const Vec3f &barycentric) {
        return vertices[0] * barycentric.x() + vertices[1] * barycentric.y() + vertices[2] * barycentric.z();
    };
    auto interpolate_uv = [](const std::array<Vec2f, 3> &uvs, const Vec3f &barycentric) {
        return uvs[0] * barycentric.x() + uvs[1] * barycentric.y() + uvs[2] * barycentric.z();
    };
    auto bary_from_lattice = [](int i, int j, int steps) {
        const float inv_steps = 1.f / float(std::max(1, steps));
        return Vec3f(1.f - float(i + j) * inv_steps, float(i) * inv_steps, float(j) * inv_steps);
    };
    auto bary_reasonable_for_uv = [](const Vec3f &barycentric) {
        return barycentric.x() >= -0.25f && barycentric.y() >= -0.25f && barycentric.z() >= -0.25f &&
               barycentric.x() <= 1.25f && barycentric.y() <= 1.25f && barycentric.z() <= 1.25f;
    };

    unsigned int vertex_index = 0;
    for (const TriangleSelector::FacetStateTriangle &triangle : state_triangles) {
        if (triangle.source_triangle < 0)
            continue;

        const size_t source_triangle = size_t(triangle.source_triangle);
        if (source_triangle >= its.indices.size() ||
            source_triangle >= texture_uv_valid.size() ||
            texture_uv_valid[source_triangle] == 0)
            continue;

        const size_t uv_offset = source_triangle * 6;
        if (uv_offset + 5 >= texture_uvs_per_face.size())
            continue;

        const stl_triangle_vertex_indices &source_indices = its.indices[source_triangle];
        if (source_indices[0] < 0 || source_indices[1] < 0 || source_indices[2] < 0)
            continue;
        if (size_t(source_indices[0]) >= its.vertices.size() ||
            size_t(source_indices[1]) >= its.vertices.size() ||
            size_t(source_indices[2]) >= its.vertices.size())
            continue;

        const Vec3f source_p0 = its.vertices[size_t(source_indices[0])].cast<float>();
        const Vec3f source_p1 = its.vertices[size_t(source_indices[1])].cast<float>();
        const Vec3f source_p2 = its.vertices[size_t(source_indices[2])].cast<float>();
        const std::array<Vec2f, 3> source_uvs = unwrap_triangle_uvs(
            Vec2f(texture_uvs_per_face[uv_offset + 0], texture_uvs_per_face[uv_offset + 1]),
            Vec2f(texture_uvs_per_face[uv_offset + 2], texture_uvs_per_face[uv_offset + 3]),
            Vec2f(texture_uvs_per_face[uv_offset + 4], texture_uvs_per_face[uv_offset + 5]));

        Vec3f normal = (triangle.vertices[1] - triangle.vertices[0]).cross(triangle.vertices[2] - triangle.vertices[0]);
        const float normal_len = normal.norm();
        if (normal_len <= k_epsilon)
            continue;
        normal /= normal_len;
        const Vec3f offset = normal * k_preview_offset;

        const std::array<Vec3f, 3> leaf_vertices = { triangle.vertices[0], triangle.vertices[1], triangle.vertices[2] };
        std::array<Vec2f, 3> leaf_uvs;
        bool valid_leaf = true;
        for (size_t vertex_idx = 0; vertex_idx < leaf_vertices.size(); ++vertex_idx) {
            Vec3f barycentric = Vec3f::Zero();
            if (!barycentric_weights(leaf_vertices[vertex_idx], source_p0, source_p1, source_p2, barycentric)) {
                valid_leaf = false;
                break;
            }
            leaf_uvs[vertex_idx] = interpolate_uv(source_uvs, barycentric);
        }
        if (!valid_leaf)
            continue;

        const std::array<Vec3f, 3> world_vertices = {
            (world_matrix * leaf_vertices[0].cast<double>()).cast<float>(),
            (world_matrix * leaf_vertices[1].cast<double>()).cast<float>(),
            (world_matrix * leaf_vertices[2].cast<double>()).cast<float>()
        };
        Vec3f world_normal = (world_vertices[1] - world_vertices[0]).cross(world_vertices[2] - world_vertices[0]);
        const float world_normal_len = world_normal.norm();
        if (world_normal_len <= k_epsilon)
            world_normal = normal;
        else
            world_normal /= world_normal_len;

        const int surface_axis = halftone_surface_axis_for_texture_preview(world_normal);
        const std::array<Vec2f, 3> surface_vertices = {
            halftone_surface_coords_for_texture_preview(world_vertices[0], surface_axis),
            halftone_surface_coords_for_texture_preview(world_vertices[1], surface_axis),
            halftone_surface_coords_for_texture_preview(world_vertices[2], surface_axis)
        };
        const float max_edge_mm = std::max({ (world_vertices[1] - world_vertices[0]).norm(),
                                             (world_vertices[2] - world_vertices[1]).norm(),
                                             (world_vertices[0] - world_vertices[2]).norm() });
        if (!std::isfinite(max_edge_mm) || max_edge_mm <= k_epsilon)
            continue;
        const int steps = std::clamp(int(std::ceil(max_edge_mm / std::max(target_step_mm, k_epsilon))), 1, max_subdivision_steps);

        auto color_at_bary = [&](const Vec3f &barycentric) {
            const Vec2f uv = interpolate_uv(leaf_uvs, barycentric);
            const Vec3f world_point = interpolate_point(world_vertices, barycentric);
            const Vec2f surface_coord = halftone_surface_coords_for_texture_preview(world_point, surface_axis);
            const float plane_coord = halftone_surface_plane_coord_for_texture_preview(world_point, surface_axis);
            uint32_t mask = 0;
            for (size_t component_idx = 0; component_idx < settings.component_colors.size() && component_idx < 31; ++component_idx) {
                const float angle_deg =
                    halftone_screen_angle_deg_for_texture_preview(settings.filament_color_mode, component_idx);
                float coverage = source_coverage_at_uv(uv, component_idx);
                if (!halftone_increased_detail && period > k_epsilon) {
                    constexpr float pi = 3.14159265358979323846f;
                    const float radians = angle_deg * pi / 180.f;
                    const float c = std::cos(radians);
                    const float s = std::sin(radians);
                    const float screen_x = c * surface_coord.x() + s * surface_coord.y();
                    const float screen_y = -s * surface_coord.x() + c * surface_coord.y();
                    const int cell_x = int(std::floor(screen_x / period));
                    const int cell_y = int(std::floor(screen_y / period));
                    const float plane_bucket_size = std::max(0.5f, period * 0.5f);
                    const int plane_bucket = int(std::floor(plane_coord / plane_bucket_size));
                    const uint64_t key =
                        image_texture_halftone_cell_cache_key(component_idx, surface_axis, plane_bucket, cell_x, cell_y);
                    const auto cached = halftone_cell_tone_cache.find(key);
                    if (cached != halftone_cell_tone_cache.end()) {
                        coverage = cached->second;
                    } else {
                        const float center_screen_x = (float(cell_x) + 0.5f) * period;
                        const float center_screen_y = (float(cell_y) + 0.5f) * period;
                        const Vec2f center_surface(c * center_screen_x - s * center_screen_y,
                                                   s * center_screen_x + c * center_screen_y);
                        Vec3f center_bary = Vec3f::Zero();
                        Vec2f center_uv = uv;
                        if (barycentric_weights_2d(center_surface,
                                                   surface_vertices[0],
                                                   surface_vertices[1],
                                                   surface_vertices[2],
                                                   center_bary) &&
                            bary_reasonable_for_uv(center_bary))
                            center_uv = interpolate_uv(leaf_uvs, center_bary);
                        coverage = source_coverage_at_uv(center_uv, component_idx);
                        halftone_cell_tone_cache.emplace(key, coverage);
                    }
                }
                if (halftone_screen_on_for_texture_preview(coverage,
                                                           surface_coord.x(),
                                                           surface_coord.y(),
                                                           period,
                                                           angle_deg))
                    mask |= uint32_t(1) << component_idx;
            }
            return halftone_preview_color_from_mask(settings, mask);
        };

        auto emit_subtriangle = [&](const Vec3f &b0, const Vec3f &b1, const Vec3f &b2) {
            const Vec3f centroid_bary = (b0 + b1 + b2) / 3.f;
            const ColorRGBA color = color_at_bary(centroid_bary);
            geometry.add_vertex(interpolate_point(leaf_vertices, b0) + offset, normal, color);
            geometry.add_vertex(interpolate_point(leaf_vertices, b1) + offset, normal, color);
            geometry.add_vertex(interpolate_point(leaf_vertices, b2) + offset, normal, color);
            geometry.add_triangle(vertex_index, vertex_index + 1, vertex_index + 2);
            vertex_index += 3;
        };

        for (int i = 0; i < steps; ++i) {
            for (int j = 0; j < steps - i; ++j) {
                const Vec3f b0 = bary_from_lattice(i, j, steps);
                const Vec3f b1 = bary_from_lattice(i + 1, j, steps);
                const Vec3f b2 = bary_from_lattice(i, j + 1, steps);
                emit_subtriangle(b0, b1, b2);
                if (j < steps - i - 1) {
                    const Vec3f b3 = bary_from_lattice(i + 1, j + 1, steps);
                    emit_subtriangle(b1, b3, b2);
                }
            }
        }
    }

    return !geometry.is_empty();
}

size_t texture_preview_state_triangles_signature(const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles)
{
    size_t signature = 1469598103934665603ull;
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };
    auto mix_float = [&mix](float value) {
        const float safe_value = std::isfinite(value) ? value : 0.f;
        mix(std::hash<int>{}(int(std::lround(safe_value * 1000000.f))));
    };

    mix(state_triangles.size());
    for (const TriangleSelector::FacetStateTriangle &triangle : state_triangles) {
        mix(std::hash<int>{}(triangle.source_triangle));
        for (const Vec3f &vertex : triangle.vertices) {
            mix_float(vertex.x());
            mix_float(vertex.y());
            mix_float(vertex.z());
        }
    }
    return signature;
}

void mix_color_facets_signature(size_t &signature, const ColorFacetsAnnotation &color_source)
{
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };

    const TriangleColorSplittingData &data = color_source.get_data();
    mix(color_source.id().id);
    mix(static_cast<size_t>(color_source.timestamp()));
    mix(data.triangles_to_split.size());
    mix(data.bitstream.size());
    mix(data.colors_rgba.size());
    mix(std::hash<std::string>{}(data.metadata_json));
}

size_t texture_preview_vertex_color_source_signature(const ModelVolume                                      &model_volume,
                                                     const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles)
{
    size_t signature = 1469598103934665603ull;
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };

    mix(reinterpret_cast<size_t>(model_volume.mesh_ptr().get()));
    mix(model_volume.id().id);
    mix(model_volume.mesh().its.vertices.size());
    mix(model_volume.mesh().its.indices.size());
    mix(model_volume.mmu_segmentation_facets.id().id);
    mix(static_cast<size_t>(model_volume.mmu_segmentation_facets.timestamp()));
    mix(model_volume.imported_vertex_colors_rgba.id().id);
    mix(model_volume.imported_vertex_colors_rgba.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_vertex_colors_rgba.data()));
    mix(state_triangles.size());
    const ColorRGBA background = texture_mapping_background_color_for_preview(model_volume);
    auto background_signature_component = [](float value) {
        return size_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    mix(background_signature_component(background.r()));
    mix(background_signature_component(background.g()));
    mix(background_signature_component(background.b()));
    return signature;
}

size_t texture_preview_color_facets_source_signature(const ModelVolume                                      &model_volume,
                                                     const ColorFacetsAnnotation                             &color_source,
                                                     const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles)
{
    size_t signature = 1469598103934665603ull;
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };

    mix(reinterpret_cast<size_t>(model_volume.mesh_ptr().get()));
    mix(model_volume.id().id);
    mix(model_volume.mesh().its.vertices.size());
    mix(model_volume.mesh().its.indices.size());
    mix(model_volume.mmu_segmentation_facets.id().id);
    mix(static_cast<size_t>(model_volume.mmu_segmentation_facets.timestamp()));
    mix_color_facets_signature(signature, color_source);
    mix(state_triangles.size());
    const ColorRGBA background = texture_mapping_background_color_for_preview(model_volume, &color_source);
    auto background_signature_component = [](float value) {
        return size_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    mix(background_signature_component(background.r()));
    mix(background_signature_component(background.g()));
    mix(background_signature_component(background.b()));
    return signature;
}

size_t texture_preview_image_halftone_source_signature(const ModelVolume                                      &model_volume,
                                                       const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
                                                       const Transform3d                                      &world_matrix)
{
    size_t signature = model_volume_texture_preview_signature(model_volume);
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };
    mix(model_volume.id().id);
    mix(reinterpret_cast<size_t>(model_volume.mesh_ptr().get()));
    mix(model_volume.mesh().its.vertices.size());
    mix(model_volume.mesh().its.indices.size());
    mix(model_volume.mmu_segmentation_facets.id().id);
    mix(static_cast<size_t>(model_volume.mmu_segmentation_facets.timestamp()));
    mix(state_triangles.size());
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            mix(std::hash<int>{}(int(std::lround(world_matrix(row, col) * 1000000.0))));
    return signature;
}

size_t texture_preview_vertex_color_simulation_cache_key(const ModelVolume &model_volume,
                                                         unsigned int       filament_id,
                                                         size_t             source_kind,
                                                         size_t             preview_owner_key)
{
    size_t key = texture_preview_simulation_cache_key(model_volume, filament_id);
    key ^= std::hash<size_t>{}(source_kind) + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
    key ^= std::hash<size_t>{}(preview_owner_key) + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
    return key;
}

bool consume_or_queue_vertex_color_simulation(size_t cache_key,
                                              size_t simulation_signature,
                                              std::function<TexturePreviewVertexColorSimulationResult()> build_result,
                                              GUI::GLModel &out_model)
{
    auto &cache = texture_preview_vertex_color_simulation_cache();
    std::shared_ptr<TexturePreviewVertexColorSimulationCacheEntry> &entry_ref = cache[cache_key];
    if (entry_ref == nullptr)
        entry_ref = std::make_shared<TexturePreviewVertexColorSimulationCacheEntry>();
    TexturePreviewVertexColorSimulationCacheEntry &entry = *entry_ref;

    if (entry.pending_future.valid() && entry.pending_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
            TexturePreviewVertexColorSimulationResult result = entry.pending_future.get();
            if (result.signature == entry.pending_signature && !result.geometry.is_empty()) {
                entry.ready_signature = result.signature;
                entry.ready_geometry = std::move(result.geometry);
                entry.ready_generation_reported = false;
            }
        } catch (...) {
        }
        entry.pending_signature = 0;
    }

    if (entry.ready_geometry.has_value() && entry.ready_signature == simulation_signature) {
        GUI::GLModel::Geometry geometry = *entry.ready_geometry;
        out_model.init_from(std::move(geometry));
        return true;
    }

    if (!entry.pending_future.valid()) {
        entry.ready_geometry.reset();
        entry.ready_signature = 0;
        entry.ready_generation_reported = false;
        entry.pending_signature = simulation_signature;
        entry.pending_future = std::async(std::launch::async, std::move(build_result));
    }

    return false;
}

bool build_simulated_vertex_color_preview_model_for_state(
    const ModelVolume                                      &model_volume,
    const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
    unsigned int                                            filament_id,
    const TexturePreviewSimulationSettings                  &simulation_settings,
    size_t                                                   preview_owner_key,
    GUI::GLModel                                            &out_model)
{
    if (!model_volume_has_vertex_color_preview_data(model_volume) || state_triangles.empty())
        return false;

    const size_t source_signature = texture_preview_vertex_color_source_signature(model_volume, state_triangles);
    const size_t simulation_signature = texture_preview_simulation_signature(model_volume, source_signature, simulation_settings);
    const size_t cache_key = texture_preview_vertex_color_simulation_cache_key(model_volume, filament_id, 1u, preview_owner_key);
    std::shared_ptr<const TriangleMesh> mesh = model_volume.mesh_ptr();
    std::vector<uint32_t> vertex_colors(model_volume.imported_vertex_colors_rgba.begin(), model_volume.imported_vertex_colors_rgba.end());
    std::vector<TriangleSelector::FacetStateTriangle> triangles = state_triangles;
    const ColorRGBA background_color = texture_mapping_background_color_for_preview(model_volume);
    TexturePreviewSimulationSettings settings = simulation_settings;
    return consume_or_queue_vertex_color_simulation(cache_key,
                                                    simulation_signature,
                                                    [simulation_signature,
                                                     mesh = std::move(mesh),
                                                     vertex_colors = std::move(vertex_colors),
                                                     triangles = std::move(triangles),
                                                     background_color,
                                                     settings = std::move(settings)]() mutable {
                                                        TexturePreviewVertexColorSimulationResult result;
                                                        result.signature = simulation_signature;
                                                        prepare_texture_preview_simulation_settings(settings);
                                                        build_vertex_color_preview_geometry_for_state(mesh->its,
                                                                                                      vertex_colors,
                                                                                                      triangles,
                                                                                                      &settings,
                                                                                                      background_color,
                                                                                                      result.geometry);
                                                        return result;
                                                    },
                                                    out_model);
}

bool build_simulated_texture_mapping_color_preview_model_for_state(
    const ModelVolume                                      &model_volume,
    const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
    unsigned int                                            filament_id,
    const TexturePreviewSimulationSettings                  &simulation_settings,
    GUI::GLModel                                            &out_model,
    const ColorFacetsAnnotation                             *texture_mapping_color_facets_override = nullptr,
    size_t                                                   preview_owner_key = 0)
{
    const ColorFacetsAnnotation *color_source = texture_mapping_color_facets_override;
    if (color_source == nullptr || color_source->empty())
        color_source = &model_volume.texture_mapping_color_facets;
    if (color_source == nullptr || color_source->empty() || state_triangles.empty())
        return false;

    const size_t source_signature = texture_preview_color_facets_source_signature(model_volume, *color_source, state_triangles);
    const size_t simulation_signature = texture_preview_simulation_signature(model_volume, source_signature, simulation_settings);
    const size_t cache_key = texture_preview_vertex_color_simulation_cache_key(model_volume, filament_id, 2u, preview_owner_key);
    std::shared_ptr<const TriangleMesh> mesh = model_volume.mesh_ptr();
    std::shared_ptr<ColorFacetsAnnotation> color_source_copy = ColorFacetsAnnotation::make_temporary();
    color_source_copy->assign(*color_source);
    std::vector<TriangleSelector::FacetStateTriangle> triangles = state_triangles;
    const ColorRGBA background_color = texture_mapping_background_color_for_preview(model_volume, color_source);
    TexturePreviewSimulationSettings settings = simulation_settings;
    return consume_or_queue_vertex_color_simulation(cache_key,
                                                    simulation_signature,
                                                    [simulation_signature,
                                                     mesh = std::move(mesh),
                                                     color_source_copy,
                                                     triangles = std::move(triangles),
                                                     background_color,
                                                     settings = std::move(settings)]() mutable {
                                                        TexturePreviewVertexColorSimulationResult result;
                                                        result.signature = simulation_signature;
                                                        prepare_texture_preview_simulation_settings(settings);
                                                        build_texture_mapping_color_preview_geometry_for_state(mesh->its,
                                                                                                               *color_source_copy,
                                                                                                               triangles,
                                                                                                               &settings,
                                                                                                               background_color,
                                                                                                               result.geometry);
                                                        return result;
                                                    },
                                                    out_model);
}

bool build_simulated_image_texture_halftone_preview_model_for_state(
    const ModelVolume                                      &model_volume,
    const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
    unsigned int                                            filament_id,
    const TexturePreviewSimulationSettings                  &simulation_settings,
    const Transform3d                                      &world_matrix,
    size_t                                                   preview_owner_key,
    GUI::GLModel                                            &out_model)
{
    if (!model_volume_has_texture_preview_data(model_volume) ||
        state_triangles.empty() ||
        !texture_preview_uses_halftone_dithering(simulation_settings))
        return false;

    const size_t source_signature =
        texture_preview_image_halftone_source_signature(model_volume, state_triangles, world_matrix);
    const size_t simulation_signature = texture_preview_simulation_signature(model_volume, source_signature, simulation_settings);
    const size_t cache_key = texture_preview_vertex_color_simulation_cache_key(model_volume, filament_id, 3u, preview_owner_key);
    std::shared_ptr<const TriangleMesh> mesh = model_volume.mesh_ptr();
    std::vector<unsigned char> source_rgba(model_volume.imported_texture_rgba.begin(),
                                           model_volume.imported_texture_rgba.end());
    std::vector<float> texture_uvs_per_face(model_volume.imported_texture_uvs_per_face.begin(),
                                            model_volume.imported_texture_uvs_per_face.end());
    std::vector<uint8_t> texture_uv_valid(model_volume.imported_texture_uv_valid.begin(),
                                          model_volume.imported_texture_uv_valid.end());
    const unsigned int width = model_volume.imported_texture_width;
    const unsigned int height = model_volume.imported_texture_height;
    std::vector<TriangleSelector::FacetStateTriangle> triangles = state_triangles;
    const ColorRGBA background_color = texture_mapping_background_color_for_preview(model_volume);
    TexturePreviewSimulationSettings settings = simulation_settings;
    return consume_or_queue_vertex_color_simulation(cache_key,
                                                    simulation_signature,
                                                    [simulation_signature,
                                                     mesh = std::move(mesh),
                                                     source_rgba = std::move(source_rgba),
                                                     width,
                                                     height,
                                                     texture_uvs_per_face = std::move(texture_uvs_per_face),
                                                     texture_uv_valid = std::move(texture_uv_valid),
                                                     triangles = std::move(triangles),
                                                     background_color,
                                                     world_matrix,
                                                     settings = std::move(settings)]() mutable {
                                                        TexturePreviewVertexColorSimulationResult result;
                                                        result.signature = simulation_signature;
                                                        prepare_texture_preview_simulation_settings(settings);
                                                        build_image_texture_halftone_preview_geometry_for_state(mesh->its,
                                                                                                                source_rgba,
                                                                                                                width,
                                                                                                                height,
                                                                                                                texture_uvs_per_face,
                                                                                                                texture_uv_valid,
                                                                                                                triangles,
                                                                                                                settings,
                                                                                                                background_color,
                                                                                                                world_matrix,
                                                                                                                result.geometry);
                                                        return result;
                                                    },
                                                    out_model);
}

float normalize_angle(float angle)
{
    if (!std::isfinite(angle))
        return 0.f;
    float out = std::fmod(angle, 360.f);
    if (out < 0.f)
        out += 360.f;
    return out;
}

float angular_distance_deg(float a, float b)
{
    const float d = std::abs(normalize_angle(a) - normalize_angle(b));
    return std::min(d, 360.f - d);
}

float angular_distance_cw(float from_deg, float to_deg)
{
    float d = normalize_angle(to_deg) - normalize_angle(from_deg);
    if (d < 0.f)
        d += 360.f;
    return d;
}

float component_angular_influence(unsigned int component_id,
                                  float theta_deg,
                                  const std::vector<unsigned int> &component_ids,
                                  const std::vector<float> &component_angles_deg)
{
    if (component_ids.empty() || component_ids.size() != component_angles_deg.size())
        return 0.f;

    const auto active_it = std::find(component_ids.begin(), component_ids.end(), component_id);
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
        sorted_angles.push_back({ normalize_angle(component_angles_deg[i]), i });

    std::sort(sorted_angles.begin(), sorted_angles.end(), [](const SortedComponentAngle &lhs, const SortedComponentAngle &rhs) {
        return lhs.angle_deg < rhs.angle_deg;
    });

    const size_t active_component_idx = size_t(active_it - component_ids.begin());
    const auto sorted_active_it = std::find_if(sorted_angles.begin(), sorted_angles.end(), [active_component_idx](const SortedComponentAngle &entry) {
        return entry.component_idx == active_component_idx;
    });
    if (sorted_active_it == sorted_angles.end())
        return 0.f;

    const size_t sorted_pos = size_t(sorted_active_it - sorted_angles.begin());
    const size_t count = sorted_angles.size();
    const float prev_angle = sorted_angles[(sorted_pos + count - 1) % count].angle_deg;
    const float self_angle = sorted_angles[sorted_pos].angle_deg;
    const float next_angle = sorted_angles[(sorted_pos + 1) % count].angle_deg;
    const float prev_to_self_deg = angular_distance_cw(prev_angle, self_angle);
    const float self_to_next_deg = angular_distance_cw(self_angle, next_angle);

    if (prev_to_self_deg <= 1e-3f || self_to_next_deg <= 1e-3f) {
        float total_weight = 0.f;
        float active_weight = 0.f;
        for (size_t i = 0; i < component_ids.size(); ++i) {
            const float dist = angular_distance_deg(theta_deg, component_angles_deg[i]);
            const float weight = std::max(0.f, 1.f - dist / 180.f);
            total_weight += weight;
            if (component_ids[i] == component_id)
                active_weight += weight;
        }

        if (total_weight <= k_epsilon)
            return 0.f;
        return std::clamp(active_weight / total_weight, 0.f, 1.f);
    }

    const float theta_norm = normalize_angle(theta_deg);
    const float prev_to_theta_deg = angular_distance_cw(prev_angle, theta_norm);
    if (prev_to_theta_deg <= prev_to_self_deg + 1e-4f)
        return std::clamp(prev_to_theta_deg / prev_to_self_deg, 0.f, 1.f);

    const float self_to_theta_deg = angular_distance_cw(self_angle, theta_norm);
    if (self_to_theta_deg <= self_to_next_deg + 1e-4f)
        return std::clamp(1.f - self_to_theta_deg / self_to_next_deg, 0.f, 1.f);

    return 0.f;
}

std::vector<unsigned int> decode_surface_gradient_component_ids(const TextureMappingZone &zone, size_t num_physical)
{
    std::vector<unsigned int> ids;
    bool seen[10] = { false };
    for (const char c : zone.component_ids) {
        if (c < '1' || c > '9')
            continue;
        const unsigned int id = unsigned(c - '0');
        if (id == 0 || id > num_physical || seen[id])
            continue;
        seen[id] = true;
        ids.emplace_back(id);
    }

    auto append_component = [&ids, &seen, num_physical](unsigned int id) {
        if (id == 0 || id > num_physical || id > 9 || seen[id])
            return;
        seen[id] = true;
        ids.emplace_back(id);
    };

    if (ids.size() < 2) {
        ids.clear();
        for (bool &flag : seen)
            flag = false;
        append_component(zone.component_a);
        append_component(zone.component_b);
    }

    return ids;
}

float repeated_rotation_progress(float progress01, float repeats, bool reverse_repeats)
{
    const float p = clamp01(progress01);
    const float r = std::max(1.f, repeats);
    if (r <= 1.f + k_epsilon)
        return p;

    float repeated_pos = p * r;
    int segment_idx = int(std::floor(repeated_pos));
    float local = repeated_pos - float(segment_idx);

    if (p >= 1.f - k_epsilon) {
        segment_idx = std::max(0, int(std::ceil(r)) - 1);
        local = 1.f;
    }

    if (reverse_repeats && (segment_idx % 2 == 1))
        local = 1.f - local;
    return clamp01(local);
}

float offset_fade_factor(int fade_mode, float progress01)
{
    const float p = clamp01(progress01);
    switch (fade_mode) {
    case int(TextureMappingZone::OffsetFadeInUp):
        return p;
    case int(TextureMappingZone::OffsetFadeOutUp):
        return 1.f - p;
    case int(TextureMappingZone::OffsetFadeInOut):
        return 1.f - std::abs(2.f * p - 1.f);
    case int(TextureMappingZone::OffsetFadeOutIn):
        return std::abs(2.f * p - 1.f);
    case int(TextureMappingZone::OffsetFadeOutInReversed):
        return 2.f * p - 1.f;
    default:
        return 1.f;
    }
}

float variable_width_delta(float inset_strength,
                           float max_width_delta_limit_mm,
                           float minimum_offset_factor,
                           float strength_factor)
{
    if (!std::isfinite(max_width_delta_limit_mm) || max_width_delta_limit_mm <= 0.f)
        return 0.f;

    const float desired_width_factor = 1.f - std::clamp(inset_strength, 0.f, 1.f);
    const float min_width_factor = std::clamp(minimum_offset_factor, 0.f, 1.f);
    const float adjusted_width_factor =
        min_width_factor + desired_width_factor * std::clamp(strength_factor, 0.f, 1.f) * (1.f - min_width_factor);

    return std::clamp(max_width_delta_limit_mm * (1.f - adjusted_width_factor), 0.f, max_width_delta_limit_mm);
}

ColorRGBA surface_gradient_preview_color_from_weights(const SurfaceGradientPreviewSettings &settings,
                                                      const std::vector<float>             &weights)
{
    const std::array<float, 3> rgb = mix_color_solver_components(settings.component_colors, weights, ColorSolverMixModel::PigmentPainter);
    return { rgb[0], rgb[1], rgb[2], 1.f };
}

ColorRGBA surface_gradient_preview_color_at(const SurfaceGradientPreviewSettings &settings,
                                            const Vec3f                          &position,
                                            const Vec3f                          &normal)
{
    if (settings.component_ids.empty() || settings.component_ids.size() != settings.component_colors.size())
        return { 0.15f, 0.65f, 0.6f, 1.f };

    const float z_span = settings.z_max - settings.z_min;
    const float z_progress = z_span > k_epsilon ?
        std::clamp((position.z() - settings.z_min) / z_span, 0.f, 1.f) :
        0.f;

    float rotation_deg = 0.f;
    if (settings.rotation_enabled) {
        const float repeated = repeated_rotation_progress(z_progress, std::max(1.f, settings.repeats), settings.reverse_repeats);
        const float direction = settings.clockwise ? -1.f : 1.f;
        rotation_deg = direction * 360.f * settings.rotations * repeated;
    }

    std::vector<float> rotated_angles = settings.angles_deg;
    for (float &angle : rotated_angles)
        angle = normalize_angle(angle + rotation_deg);

    Vec2f direction = Vec2f::Zero();
    if (settings.angle_mode == int(TextureMappingZone::OffsetAngleSurfaceNormal))
        direction = Vec2f(normal.x(), normal.y());

    if (direction.squaredNorm() <= k_epsilon) {
        const Vec3f radial = position - settings.center;
        direction = Vec2f(radial.x(), radial.y());
    }
    if (direction.squaredNorm() <= k_epsilon)
        direction = Vec2f(1.f, 0.f);

    const float theta_deg = normalize_angle(float(Geometry::rad2deg(std::atan2(direction.y(), direction.x()))));

    const size_t component_count = settings.component_ids.size();
    std::vector<float> influences(component_count, 0.f);
    for (size_t i = 0; i < component_count; ++i)
        influences[i] = component_angular_influence(settings.component_ids[i], theta_deg, settings.component_ids, rotated_angles);

    const float fade_factor = std::abs(offset_fade_factor(settings.fade_mode, z_progress));
    std::vector<float> edge_reaches(component_count, 0.f);
    for (size_t i = 0; i < component_count; ++i) {
        float raw_inset_mm = 0.f;
        for (size_t j = 0; j < component_count; ++j) {
            if (i == j)
                continue;
            const float distance_mm = j < settings.distances_mm.size() ? settings.distances_mm[j] : 0.f;
            raw_inset_mm += distance_mm * influences[j];
        }

        const float inset_strength = std::clamp(raw_inset_mm / std::max(settings.max_component_distance_mm, k_epsilon), 0.f, 1.f);
        const float strength_factor = i < settings.strength_factors.size() ? settings.strength_factors[i] : 1.f;
        const float minimum_offset_factor = i < settings.minimum_offset_factors.size() ? settings.minimum_offset_factors[i] : 0.f;
        const float width_delta_mm = variable_width_delta(inset_strength * fade_factor,
                                                          settings.max_width_delta_limit_mm,
                                                          minimum_offset_factor,
                                                          strength_factor);
        edge_reaches[i] = std::clamp(settings.max_width_delta_limit_mm - width_delta_mm, 0.f, settings.max_width_delta_limit_mm);
    }

    const auto minmax_reach = std::minmax_element(edge_reaches.begin(), edge_reaches.end());
    std::vector<float> weights(component_count, 0.f);
    if (minmax_reach.first != edge_reaches.end() && (*minmax_reach.second - *minmax_reach.first) > k_epsilon) {
        const float base_reach = *minmax_reach.first;
        const float reach_span = *minmax_reach.second - base_reach;
        for (size_t i = 0; i < component_count; ++i)
            weights[i] = std::clamp((edge_reaches[i] - base_reach) / reach_span, 0.f, 1.f);
    } else {
        std::fill(weights.begin(), weights.end(), 1.f);
    }

    return surface_gradient_preview_color_from_weights(settings, weights);
}

float surface_gradient_preview_config_float(const char *key, float fallback)
{
    return texture_preview_config_float(key, fallback);
}

std::optional<SurfaceGradientPreviewSettings> surface_gradient_preview_settings_for_zone(const ModelVolume &model_volume,
                                                                                        const Transform3d &world_matrix,
                                                                                        const TextureMappingZone &zone,
                                                                                        size_t num_physical)
{
    if (!is_gradient_zone(zone))
        return std::nullopt;

    const std::vector<std::string> colors = physical_filament_colors_for_texture_preview(num_physical);
    SurfaceGradientPreviewSettings settings;
    settings.component_ids = decode_surface_gradient_component_ids(zone, num_physical);
    if (settings.component_ids.size() < 2)
        return std::nullopt;

    settings.component_colors.reserve(settings.component_ids.size());
    settings.strength_factors.reserve(settings.component_ids.size());
    settings.minimum_offset_factors.reserve(settings.component_ids.size());
    for (const unsigned int component_id : settings.component_ids) {
        if (component_id == 0 || size_t(component_id - 1) >= colors.size())
            return std::nullopt;
        settings.component_colors.emplace_back(decode_color(colors[size_t(component_id - 1)]));

        const size_t idx = size_t(component_id - 1);
        const float strength_pct = idx < zone.filament_strengths_pct.size() ? zone.filament_strengths_pct[idx] : 100.f;
        const float minimum_offset_pct = idx < zone.filament_minimum_offsets_pct.size() ? zone.filament_minimum_offsets_pct[idx] : 0.f;
        settings.strength_factors.emplace_back(std::clamp((std::isfinite(strength_pct) ? strength_pct : 100.f) / 100.f, 0.f, 1.f));
        settings.minimum_offset_factors.emplace_back(std::clamp((std::isfinite(minimum_offset_pct) ? minimum_offset_pct : 0.f) / 100.f, 0.f, 1.f));
    }

    const float max_distance_mm = TextureMappingManager::max_component_surface_offset_mm();
    settings.max_component_distance_mm = max_distance_mm;
    settings.distances_mm = TextureMappingManager::effective_offset_distances(zone, settings.component_ids.size());
    bool has_nonzero_distance = false;
    for (float &distance_mm : settings.distances_mm) {
        distance_mm = std::clamp(distance_mm, 0.f, max_distance_mm);
        has_nonzero_distance = has_nonzero_distance || distance_mm > k_epsilon;
    }
    if (!has_nonzero_distance)
        return std::nullopt;

    settings.angles_deg = TextureMappingManager::effective_offset_angles(zone, settings.component_ids.size());
    settings.angle_mode = std::clamp(zone.offset_angle_mode,
                                     int(TextureMappingZone::OffsetAngleConfigured),
                                     int(TextureMappingZone::OffsetAngleObjectCenter));
    settings.rotation_enabled = zone.offset_rotation_enabled;
    settings.rotations = std::isfinite(zone.offset_rotations) ? zone.offset_rotations : 1.f;
    settings.repeats = std::isfinite(zone.offset_repeats) ? std::max(1.f, zone.offset_repeats) : 1.f;
    settings.reverse_repeats = zone.offset_reverse_repeats;
    settings.clockwise = zone.offset_clockwise;
    settings.fade_mode = std::clamp(zone.offset_fade_mode,
                                    int(TextureMappingZone::OffsetFadeNone),
                                    int(TextureMappingZone::OffsetFadeOutInReversed));
    settings.limit_texture_resolution = zone.preview_limit_resolution;
    const float base_outer_width_mm = std::max(0.05f, surface_gradient_preview_config_float("texture_mapping_outer_wall_gradient_max_line_width", 0.95f));
    const float min_outer_width_mm = std::clamp(surface_gradient_preview_config_float("texture_mapping_outer_wall_gradient_min_line_width", 0.32f),
                                                0.05f,
                                                base_outer_width_mm);
    const float global_strength_factor =
        std::clamp(surface_gradient_preview_config_float("texture_mapping_outer_wall_gradient_global_strength", 100.f) / 100.f, 0.f, 1.f);
    settings.max_width_delta_limit_mm = std::min((base_outer_width_mm - min_outer_width_mm) * global_strength_factor, 2.f * max_distance_mm);
    if (!std::isfinite(settings.max_width_delta_limit_mm) || settings.max_width_delta_limit_mm <= k_epsilon)
        return std::nullopt;

    const indexed_triangle_set &its = model_volume.mesh().its;
    if (its.vertices.empty())
        return std::nullopt;

    Vec3f min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec3f max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    for (const stl_vertex &vertex : its.vertices) {
        const Vec3f p = (world_matrix * vertex.cast<double>()).cast<float>();
        min_pt = min_pt.cwiseMin(p);
        max_pt = max_pt.cwiseMax(p);
    }

    settings.center = 0.5f * (min_pt + max_pt);
    settings.z_min = min_pt.z();
    settings.z_max = max_pt.z();
    return settings;
}

bool build_surface_gradient_vertex_color_preview_model_for_state(const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
                                                                 const SurfaceGradientPreviewSettings &settings,
                                                                 const Transform3d &world_matrix,
                                                                 GUI::GLModel &out_model)
{
    if (state_triangles.empty())
        return false;

    GUI::GLModel::Geometry geometry;
    geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };
    geometry.reserve_vertices(state_triangles.size() * 3);
    geometry.reserve_indices(state_triangles.size() * 3);

    unsigned int vertex_index = 0;
    for (const TriangleSelector::FacetStateTriangle &triangle : state_triangles) {
        Vec3f normal = (triangle.vertices[1] - triangle.vertices[0]).cross(triangle.vertices[2] - triangle.vertices[0]);
        const float normal_len = normal.norm();
        if (normal_len <= k_epsilon)
            continue;
        normal /= normal_len;
        const Vec3f offset = normal * k_preview_offset;

        const Vec3f world_vertices[3] = {
            (world_matrix * triangle.vertices[0].cast<double>()).cast<float>(),
            (world_matrix * triangle.vertices[1].cast<double>()).cast<float>(),
            (world_matrix * triangle.vertices[2].cast<double>()).cast<float>()
        };
        Vec3f world_normal = (world_vertices[1] - world_vertices[0]).cross(world_vertices[2] - world_vertices[0]);
        const float world_normal_len = world_normal.norm();
        if (world_normal_len <= k_epsilon)
            world_normal = normal;
        else
            world_normal /= world_normal_len;

        const ColorRGBA c0 = surface_gradient_preview_color_at(settings, world_vertices[0], world_normal);
        const ColorRGBA c1 = surface_gradient_preview_color_at(settings, world_vertices[1], world_normal);
        const ColorRGBA c2 = surface_gradient_preview_color_at(settings, world_vertices[2], world_normal);

        geometry.add_vertex(triangle.vertices[0] + offset, normal, c0);
        geometry.add_vertex(triangle.vertices[1] + offset, normal, c1);
        geometry.add_vertex(triangle.vertices[2] + offset, normal, c2);
        geometry.add_triangle(vertex_index, vertex_index + 1, vertex_index + 2);
        vertex_index += 3;
    }

    if (geometry.is_empty())
        return false;

    out_model.init_from(std::move(geometry));
    return true;
}

struct TexturePreviewRenderState
{
    GLboolean blend_enabled { GL_FALSE };
    GLboolean cull_face_enabled { GL_FALSE };
    GLboolean polygon_offset_fill_enabled { GL_FALSE };
    GLboolean depth_mask { GL_TRUE };
    GLint     cull_face_mode { GL_BACK };
    GLfloat   polygon_offset_factor { 0.f };
    GLfloat   polygon_offset_units { 0.f };
    GLint     depth_func { GL_LESS };
};

TexturePreviewRenderState begin_render_state(bool opaque)
{
    TexturePreviewRenderState state;
    state.blend_enabled = glIsEnabled(GL_BLEND);
    state.cull_face_enabled = glIsEnabled(GL_CULL_FACE);
    state.polygon_offset_fill_enabled = glIsEnabled(GL_POLYGON_OFFSET_FILL);
    glsafe(::glGetBooleanv(GL_DEPTH_WRITEMASK, &state.depth_mask));
    glsafe(::glGetIntegerv(GL_CULL_FACE_MODE, &state.cull_face_mode));
    glsafe(::glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &state.polygon_offset_factor));
    glsafe(::glGetFloatv(GL_POLYGON_OFFSET_UNITS, &state.polygon_offset_units));
    glsafe(::glGetIntegerv(GL_DEPTH_FUNC, &state.depth_func));

    if (opaque) {
        glsafe(::glDisable(GL_BLEND));
    } else {
        glsafe(::glEnable(GL_BLEND));
        glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    }
    glsafe(::glDisable(GL_CULL_FACE));
    glsafe(::glDepthMask(opaque ? GL_TRUE : GL_FALSE));
    glsafe(::glDepthFunc(GL_LEQUAL));
    glsafe(::glEnable(GL_POLYGON_OFFSET_FILL));
    glsafe(::glPolygonOffset(k_polygon_offset_factor, k_polygon_offset_units));
    return state;
}

void restore_render_state(const TexturePreviewRenderState &state)
{
    glsafe(::glPolygonOffset(state.polygon_offset_factor, state.polygon_offset_units));
    if (!state.polygon_offset_fill_enabled)
        glsafe(::glDisable(GL_POLYGON_OFFSET_FILL));
    glsafe(::glDepthFunc(state.depth_func));
    glsafe(::glDepthMask(state.depth_mask));
    glsafe(::glCullFace(state.cull_face_mode));
    if (state.cull_face_enabled)
        glsafe(::glEnable(GL_CULL_FACE));
    else
        glsafe(::glDisable(GL_CULL_FACE));
    if (state.blend_enabled)
        glsafe(::glEnable(GL_BLEND));
    else
        glsafe(::glDisable(GL_BLEND));
}

void set_common_uniforms(GLShaderProgram &shader,
                         const Transform3d &model_matrix,
                         const Transform3d &view_matrix,
                         const Transform3d &projection_matrix,
                         const std::array<float, 2> &z_range,
                         const std::array<float, 4> &clipping_plane,
                         int print_volume_type,
                         const std::array<float, 4> &print_volume_xy,
                         const std::array<float, 2> &print_volume_z)
{
    const Transform3d view_model_matrix = view_matrix * model_matrix;
    const Matrix3d view_normal_matrix = view_matrix.matrix().block(0, 0, 3, 3) *
                                        model_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();

    shader.set_uniform("view_model_matrix", view_model_matrix);
    shader.set_uniform("projection_matrix", projection_matrix);
    shader.set_uniform("view_normal_matrix", view_normal_matrix);
    const std::array<float, 2> preview_z_range = { z_range[0] - k_preview_clip_padding,
                                                   z_range[1] + k_preview_clip_padding };
    shader.set_uniform("volume_world_matrix", model_matrix);
    shader.set_uniform("z_range", preview_z_range);
    shader.set_uniform("clipping_plane", clipping_plane);
    shader.set_uniform("print_volume.type", print_volume_type);
    shader.set_uniform("print_volume.xy_data", print_volume_xy);
    shader.set_uniform("print_volume.z_data", print_volume_z);
}

} // namespace

bool model_volume_has_texture_preview_data(const ModelVolume &model_volume)
{
    return model_volume_has_texture_preview_data_impl(model_volume);
}

bool model_volume_has_complete_texture_preview_data(const ModelVolume &model_volume)
{
    return model_volume_has_complete_texture_preview_data_impl(model_volume);
}

bool model_volume_has_vertex_color_preview_data(const ModelVolume &model_volume)
{
    return model_volume_has_vertex_color_preview_data_impl(model_volume);
}

bool model_volume_has_texture_mapping_color_preview_data(const ModelVolume &model_volume)
{
    return model_volume_has_texture_mapping_color_preview_data_impl(model_volume);
}

bool texture_preview_simulation_is_pending()
{
    return texture_preview_simulation_is_pending_impl();
}

bool texture_preview_simulation_has_temporary_pending()
{
    return texture_preview_simulation_has_temporary_pending_impl();
}

bool texture_preview_simulation_enabled_for_filament(unsigned int                 filament_id,
                                                     size_t                       num_physical,
                                                     const TextureMappingManager *texture_mgr)
{
    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    return texture_preview_simulation_settings_for_filament(filament_id, num_physical, texture_mgr, physical_colors).has_value();
}

bool texture_preview_halftone_simulation_enabled_for_filament(unsigned int                 filament_id,
                                                              size_t                       num_physical,
                                                              const TextureMappingManager *texture_mgr)
{
    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    std::optional<TexturePreviewSimulationSettings> settings =
        texture_preview_simulation_settings_for_filament(filament_id, num_physical, texture_mgr, physical_colors);
    return settings.has_value() && texture_preview_uses_halftone_dithering(*settings);
}

bool texture_preview_simulation_enabled_for_all_filaments(const std::vector<unsigned int> &filament_ids,
                                                          size_t                          num_physical,
                                                          const TextureMappingManager    *texture_mgr)
{
    if (filament_ids.empty())
        return false;

    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    for (const unsigned int filament_id : filament_ids)
        if (!texture_preview_simulation_settings_for_filament(filament_id, num_physical, texture_mgr, physical_colors).has_value())
            return false;
    return true;
}

size_t texture_preview_simulation_generation_signature()
{
    return texture_preview_vertex_color_simulation_generation();
}

bool build_mmu_texture_preview_models(
    const ModelVolume                                                    &model_volume,
    const std::vector<std::vector<TriangleSelector::FacetStateTriangle>> &triangles_per_type,
    const std::vector<ColorRGBA>                                         &state_colors,
    unsigned int                                                          base_filament_id,
    size_t                                                                num_physical,
    const TextureMappingManager                                          *texture_mgr,
    std::vector<GUI::GLModel>                                            &out_models,
    std::vector<ColorRGBA>                                               &out_colors,
    std::vector<unsigned int>                                            &out_filament_ids)
{
    out_models.clear();
    out_colors.clear();
    out_filament_ids.clear();
    if (!model_volume_has_texture_preview_data(model_volume))
        return false;

    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    bool built_any = false;
    for (size_t state_id = 0; state_id < triangles_per_type.size(); ++state_id) {
        const unsigned int filament_id = filament_id_for_state(state_id, base_filament_id);
        const TextureMappingZone *zone = zone_for_filament(filament_id, num_physical, texture_mgr);
        if (zone == nullptr || !is_image_zone(*zone))
            continue;
        std::optional<TexturePreviewSimulationSettings> simulation_settings =
            texture_preview_simulation_settings_for_filament(filament_id, num_physical, texture_mgr, physical_colors);
        if (simulation_settings && texture_preview_uses_halftone_dithering(*simulation_settings))
            continue;

        GUI::GLModel model;
        if (!build_texture_preview_model_for_state(model_volume, triangles_per_type[state_id], model))
            continue;

        out_models.emplace_back(std::move(model));
        out_colors.emplace_back(state_id < state_colors.size() ? state_colors[state_id] :
                                (state_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : state_colors.back()));
        out_filament_ids.emplace_back(filament_id);
        built_any = true;
    }
    return built_any;
}

bool build_mmu_vertex_color_preview_models(
    const ModelVolume                                                    &model_volume,
    const std::vector<std::vector<TriangleSelector::FacetStateTriangle>> &triangles_per_type,
    const std::vector<ColorRGBA>                                         &state_colors,
    unsigned int                                                          base_filament_id,
    size_t                                                                num_physical,
    const TextureMappingManager                                          *texture_mgr,
    const Transform3d                                                    &world_matrix,
    std::vector<GUI::GLModel>                                            &out_models,
    std::vector<ColorRGBA>                                               &out_colors,
    std::vector<unsigned int>                                            &out_filament_ids,
    const ColorFacetsAnnotation                                          *texture_mapping_color_facets_override,
    size_t                                                                preview_owner_key)
{
    out_models.clear();
    out_colors.clear();
    out_filament_ids.clear();

    const bool has_texture_mapping_color_override =
        texture_mapping_color_facets_override != nullptr && !texture_mapping_color_facets_override->empty();
    if (triangles_per_type.empty() || (texture_mgr == nullptr && !has_texture_mapping_color_override))
        return false;

    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    bool built_any = false;
    for (size_t state_id = 0; state_id < triangles_per_type.size(); ++state_id) {
        const unsigned int filament_id = filament_id_for_state(state_id, base_filament_id);
        const TextureMappingZone *zone = zone_for_filament(filament_id, num_physical, texture_mgr);
        if (zone == nullptr) {
            if (!has_texture_mapping_color_override || state_id != 0)
                continue;

            GUI::GLModel model;
            if (!build_texture_mapping_color_preview_model_for_state(model_volume,
                                                                     triangles_per_type[state_id],
                                                                     nullptr,
                                                                     model,
                                                                     texture_mapping_color_facets_override) ||
                !model.is_initialized())
                continue;

            out_models.emplace_back(std::move(model));
            out_colors.emplace_back(state_id < state_colors.size() ? state_colors[state_id] :
                                    (state_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : state_colors.back()));
            out_filament_ids.emplace_back(0u);
            built_any = true;
            continue;
        }
        if (!is_image_zone(*zone) && !is_gradient_zone(*zone))
            continue;

        GUI::GLModel model;
        if (is_gradient_zone(*zone)) {
            std::optional<SurfaceGradientPreviewSettings> settings = surface_gradient_preview_settings_for_zone(model_volume, world_matrix, *zone, num_physical);
            if (!settings)
                continue;
            if (!build_surface_gradient_vertex_color_preview_model_for_state(triangles_per_type[state_id], *settings, world_matrix, model))
                continue;
        } else {
            const bool has_texture_mapping_color_preview =
                has_texture_mapping_color_override || model_volume_has_texture_mapping_color_preview_data(model_volume);
            const bool has_texture_preview = model_volume_has_texture_preview_data(model_volume);
            std::optional<TexturePreviewSimulationSettings> simulation_settings =
                texture_preview_simulation_settings_for_filament(filament_id, num_physical, texture_mgr, physical_colors);
            const bool has_image_halftone_preview =
                has_texture_preview &&
                !has_texture_mapping_color_preview &&
                simulation_settings &&
                texture_preview_uses_halftone_dithering(*simulation_settings);
            if (has_image_halftone_preview) {
                if (!build_simulated_image_texture_halftone_preview_model_for_state(model_volume,
                                                                                    triangles_per_type[state_id],
                                                                                    filament_id,
                                                                                    *simulation_settings,
                                                                                    world_matrix,
                                                                                    preview_owner_key,
                                                                                    model))
                    continue;
            } else if (has_texture_mapping_color_preview) {
                const ColorFacetsAnnotation *preview_override =
                    has_texture_mapping_color_override ? texture_mapping_color_facets_override : nullptr;
                if (simulation_settings) {
                    if (!build_simulated_texture_mapping_color_preview_model_for_state(model_volume,
                                                                                      triangles_per_type[state_id],
                                                                                      filament_id,
                                                                                      *simulation_settings,
                                                                                      model,
                                                                                      preview_override,
                                                                                      preview_owner_key) &&
                        !build_texture_mapping_color_preview_model_for_state(model_volume,
                                                                             triangles_per_type[state_id],
                                                                             nullptr,
                                                                             model,
                                                                             preview_override))
                        continue;
                } else {
                    if (!build_texture_mapping_color_preview_model_for_state(model_volume,
                                                                             triangles_per_type[state_id],
                                                                             nullptr,
                                                                             model,
                                                                             preview_override))
                        continue;
                }
            } else {
                if (has_texture_preview || !model_volume_has_vertex_color_preview_data(model_volume))
                    continue;
                if (simulation_settings) {
                    if (!build_simulated_vertex_color_preview_model_for_state(model_volume,
                                                                              triangles_per_type[state_id],
                                                                              filament_id,
                                                                              *simulation_settings,
                                                                              preview_owner_key,
                                                                              model) &&
                        !build_vertex_color_preview_model_for_state(model_volume, triangles_per_type[state_id], nullptr, model))
                        continue;
                } else {
                    if (!build_vertex_color_preview_model_for_state(model_volume, triangles_per_type[state_id], nullptr, model))
                        continue;
                }
            }
        }

        out_models.emplace_back(std::move(model));
        out_colors.emplace_back(state_id < state_colors.size() ? state_colors[state_id] :
                                (state_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : state_colors.back()));
        out_filament_ids.emplace_back(filament_id);
        built_any = true;
    }
    return built_any;
}

bool build_mmu_vertex_color_preview_models(
    const ModelVolume                                                    &model_volume,
    const std::vector<std::vector<TriangleSelector::FacetStateTriangle>> &triangles_per_type,
    const std::vector<ColorRGBA>                                         &state_colors,
    unsigned int                                                          base_filament_id,
    size_t                                                                num_physical,
    const TextureMappingManager                                          *texture_mgr,
    std::vector<GUI::GLModel>                                            &out_models,
    std::vector<ColorRGBA>                                               &out_colors,
    std::vector<unsigned int>                                            &out_filament_ids,
    const ColorFacetsAnnotation                                          *texture_mapping_color_facets_override,
    size_t                                                                preview_owner_key)
{
    return build_mmu_vertex_color_preview_models(model_volume,
                                                 triangles_per_type,
                                                 state_colors,
                                                 base_filament_id,
                                                 num_physical,
                                                 texture_mgr,
                                                 Transform3d::Identity(),
                                                 out_models,
                                                 out_colors,
                                                 out_filament_ids,
                                                 texture_mapping_color_facets_override,
                                                 preview_owner_key);
}

size_t model_volume_texture_preview_signature(const ModelVolume &model_volume)
{
    size_t signature = 1469598103934665603ull;
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };
    mix(model_volume.id().id);
    mix(reinterpret_cast<size_t>(model_volume.mesh_ptr().get()));
    mix(model_volume.mesh().its.vertices.size());
    mix(model_volume.mesh().its.indices.size());
    mix(size_t(model_volume.imported_texture_width));
    mix(size_t(model_volume.imported_texture_height));
    mix(model_volume.imported_texture_rgba.id().id);
    mix(model_volume.imported_texture_rgba.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_texture_rgba.data()));
    mix(size_t(model_volume.imported_texture_raw_channels));
    mix(std::hash<std::string>{}(model_volume.imported_texture_raw_metadata_json));
    mix(model_volume.imported_texture_raw_filament_offsets.id().id);
    mix(model_volume.imported_texture_raw_filament_offsets.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_texture_raw_filament_offsets.data()));
    mix(model_volume.imported_texture_uvs_per_face.id().id);
    mix(model_volume.imported_texture_uvs_per_face.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_texture_uvs_per_face.data()));
    mix(model_volume.imported_texture_uv_valid.id().id);
    mix(model_volume.imported_texture_uv_valid.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_texture_uv_valid.data()));
    const ColorRGBA background = texture_mapping_background_color_for_preview(model_volume);
    auto background_signature_component = [](float value) {
        return size_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    mix(background_signature_component(background.r()));
    mix(background_signature_component(background.g()));
    mix(background_signature_component(background.b()));
    return signature;
}

size_t model_volume_texture_mapping_color_preview_signature(const ModelVolume &model_volume)
{
    size_t signature = 1469598103934665603ull;
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };

    mix(model_volume.id().id);
    mix(reinterpret_cast<size_t>(model_volume.mesh_ptr().get()));
    mix_color_facets_signature(signature, model_volume.texture_mapping_color_facets);
    const ColorRGBA background = texture_mapping_background_color_for_preview(model_volume);
    auto background_signature_component = [](float value) {
        return size_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    mix(background_signature_component(background.r()));
    mix(background_signature_component(background.g()));
    mix(background_signature_component(background.b()));
    return signature;
}

bool ensure_model_volume_texture_preview(const ModelVolume &model_volume,
                                         GUI::GLTexture    &texture,
                                         size_t            &texture_signature)
{
    if (!model_volume_has_texture_preview_data(model_volume))
        return false;

    const size_t preview_signature = model_volume_texture_preview_signature(model_volume);
    if (texture.get_id() != 0 && texture_signature == preview_signature)
        return true;

    texture.reset();
    std::vector<unsigned char> texture_data(model_volume.imported_texture_rgba.begin(), model_volume.imported_texture_rgba.end());
    composite_texture_preview_rgba_over_background(texture_data, texture_mapping_background_color_for_preview(model_volume));
    if (!texture.load_from_raw_data(std::move(texture_data), model_volume.imported_texture_width, model_volume.imported_texture_height)) {
        texture_signature = 0;
        return false;
    }

    configure_texture_preview_sampler(texture);
    texture_signature = preview_signature;
    return true;
}

static size_t texture_preview_settings_signature_impl(size_t num_physical,
                                                      const TextureMappingManager *texture_mgr,
                                                      const std::vector<unsigned int> *active_zone_ids)
{
    size_t signature = 1469598103934665603ull;
    auto signature_mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };
    auto signature_mix_float = [&signature_mix](float value, float scale = 1000.f) {
        const float safe_value = std::isfinite(value) ? value : 0.f;
        signature_mix(std::hash<int>{}(int(std::lround(safe_value * scale))));
    };

    signature_mix(std::hash<size_t>{}(num_physical));
    if (GUI::wxGetApp().preset_bundle != nullptr) {
        if (const ConfigOptionStrings *opt = GUI::wxGetApp().preset_bundle->project_config.option<ConfigOptionStrings>("filament_colour"))
            for (const std::string &color : opt->values)
                signature_mix(std::hash<std::string>{}(color));
    }
    signature_mix_float(texture_preview_config_float("texture_mapping_outer_wall_gradient_global_strength", 100.f), 100.f);
    signature_mix_float(texture_preview_config_float("texture_mapping_outer_wall_gradient_max_line_width", 0.95f), 1000.f);
    signature_mix_float(texture_preview_config_float("texture_mapping_outer_wall_gradient_min_line_width", 0.32f), 1000.f);
    if (texture_mgr == nullptr)
        return signature;

    auto zone_is_active = [active_zone_ids](unsigned int zone_id) {
        return active_zone_ids == nullptr ||
               std::find(active_zone_ids->begin(), active_zone_ids->end(), zone_id) != active_zone_ids->end();
    };

    for (const TextureMappingZone &zone : texture_mgr->zones()) {
        if (!zone_is_active(zone.zone_id))
            continue;
        signature_mix(std::hash<uint64_t>{}(zone.stable_id));
        signature_mix(std::hash<unsigned int>{}(zone.zone_id));
        signature_mix(std::hash<int>{}(zone.enabled ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.deleted ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.surface_pattern));
        signature_mix(std::hash<unsigned int>{}(zone.component_a));
        signature_mix(std::hash<unsigned int>{}(zone.component_b));
        signature_mix(std::hash<std::string>{}(zone.component_ids));
        signature_mix(std::hash<std::string>{}(zone.component_weights));
        signature_mix(std::hash<std::string>{}(zone.offset_distances));
        signature_mix(std::hash<std::string>{}(zone.offset_angles));
        signature_mix(std::hash<int>{}(zone.offset_mode));
        signature_mix(std::hash<int>{}(zone.offset_rotation_enabled ? 1 : 0));
        signature_mix_float(zone.offset_rotations);
        signature_mix_float(zone.offset_repeats);
        signature_mix(std::hash<int>{}(zone.offset_reverse_repeats ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.offset_clockwise ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.offset_fade_mode));
        signature_mix(std::hash<int>{}(zone.offset_angle_mode));
        signature_mix(std::hash<int>{}(zone.texture_mapping_mode));
        signature_mix(std::hash<int>{}(zone.filament_color_mode));
        signature_mix(std::hash<int>{}(zone.force_sequential_filaments ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.nonlinear_offset_adjustment ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.compact_offset_mode ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.use_legacy_fixed_color_mode ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.dithering_enabled ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.dithering_method));
        if (zone.dithering_method == int(TextureMappingZone::DitheringHalftone) ||
            zone.dithering_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail))
            signature_mix_float(zone.halftone_dot_size_mm, 1000.f);
        else
            signature_mix_float(zone.dithering_resolution_mm, 1000.f);
        signature_mix(std::hash<int>{}(zone.minimum_visibility_offset_enabled ? 1 : 0));
        signature_mix_float(zone.minimum_visibility_offset_pct, 100.f);
        signature_mix(std::hash<int>{}(zone.generic_solver_lookup_mode));
        signature_mix(std::hash<int>{}(zone.generic_solver_mode));
        signature_mix(std::hash<int>{}(TextureMappingZone::DefaultGenericSolverMixModel));
        signature_mix(std::hash<int>{}(zone.preview_simulate_colors ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.preview_limit_resolution ? 1 : 0));
        signature_mix_float(zone.preview_opacity_pct, 100.f);
        signature_mix_float(zone.contrast_pct, 100.f);
        signature_mix_float(zone.tone_gamma);
        for (const float strength_pct : zone.filament_strengths_pct)
            signature_mix_float(strength_pct, 100.f);
        for (const float minimum_offset_pct : zone.filament_minimum_offsets_pct)
            signature_mix_float(minimum_offset_pct, 100.f);
    }
    return signature;
}

static std::vector<unsigned int> active_texture_preview_zone_ids(const TextureMappingManager *texture_mgr,
                                                                 unsigned int base_filament_id,
                                                                 const std::vector<bool> *used_states)
{
    std::vector<unsigned int> active_zone_ids;
    if (texture_mgr == nullptr)
        return active_zone_ids;

    auto append_active_zone_id = [texture_mgr, &active_zone_ids](unsigned int zone_id) {
        if (zone_id != 0 && texture_mgr->zone_from_id(zone_id) != nullptr)
            active_zone_ids.emplace_back(zone_id);
    };
    append_active_zone_id(base_filament_id);
    if (used_states != nullptr) {
        for (size_t idx = 1; idx < used_states->size(); ++idx)
            if ((*used_states)[idx])
                append_active_zone_id(unsigned(idx));
    }
    std::sort(active_zone_ids.begin(), active_zone_ids.end());
    active_zone_ids.erase(std::unique(active_zone_ids.begin(), active_zone_ids.end()), active_zone_ids.end());
    return active_zone_ids;
}

static bool active_texture_preview_zone_ids_contains(const std::vector<unsigned int> &active_zone_ids, unsigned int zone_id)
{
    return std::binary_search(active_zone_ids.begin(), active_zone_ids.end(), zone_id);
}

static bool texture_preview_zone_uses_halftone_model(const TextureMappingZone &zone)
{
    if (!zone.enabled || zone.deleted || !zone.is_image_texture() || !zone.preview_simulate_colors)
        return false;
    const int mapping_mode = std::clamp(zone.texture_mapping_mode,
                                        int(TextureMappingZone::TextureMappingFilamentBlending),
                                        int(TextureMappingZone::TextureMappingRawValues));
    const int method = std::clamp(zone.dithering_method,
                                  int(TextureMappingZone::DitheringClosest),
                                  int(TextureMappingZone::DitheringHalftoneIncreasedDetail));
    return mapping_mode != int(TextureMappingZone::TextureMappingRawValues) &&
           zone.dithering_enabled &&
           (method == int(TextureMappingZone::DitheringHalftone) ||
            method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail));
}

static void texture_preview_mix_zone_baked_model_settings(size_t &signature,
                                                          const TextureMappingZone &zone,
                                                          size_t num_physical)
{
    auto signature_mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };
    auto signature_mix_float = [&signature_mix](float value, float scale = 1000.f) {
        const float safe_value = std::isfinite(value) ? value : 0.f;
        signature_mix(std::hash<int>{}(int(std::lround(safe_value * scale))));
    };

    signature_mix(std::hash<unsigned int>{}(zone.component_a));
    signature_mix(std::hash<unsigned int>{}(zone.component_b));
    signature_mix(std::hash<std::string>{}(zone.component_ids));
    signature_mix(std::hash<std::string>{}(zone.component_weights));
    signature_mix(std::hash<std::string>{}(zone.offset_distances));
    signature_mix(std::hash<std::string>{}(zone.offset_angles));
    signature_mix(std::hash<int>{}(zone.offset_mode));
    signature_mix(std::hash<int>{}(zone.offset_rotation_enabled ? 1 : 0));
    signature_mix_float(zone.offset_rotations);
    signature_mix_float(zone.offset_repeats);
    signature_mix(std::hash<int>{}(zone.offset_reverse_repeats ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.offset_clockwise ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.offset_fade_mode));
    signature_mix(std::hash<int>{}(zone.offset_angle_mode));
    signature_mix(std::hash<int>{}(zone.texture_mapping_mode));
    signature_mix(std::hash<int>{}(zone.filament_color_mode));
    signature_mix(std::hash<int>{}(zone.force_sequential_filaments ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.nonlinear_offset_adjustment ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.compact_offset_mode ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.use_legacy_fixed_color_mode ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.dithering_enabled ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.dithering_method));
    if (zone.dithering_method == int(TextureMappingZone::DitheringHalftone) ||
        zone.dithering_method == int(TextureMappingZone::DitheringHalftoneIncreasedDetail))
        signature_mix_float(zone.halftone_dot_size_mm, 1000.f);
    else
        signature_mix_float(zone.dithering_resolution_mm, 1000.f);
    signature_mix(std::hash<int>{}(zone.minimum_visibility_offset_enabled ? 1 : 0));
    signature_mix_float(zone.minimum_visibility_offset_pct, 100.f);
    signature_mix(std::hash<int>{}(zone.generic_solver_lookup_mode));
    signature_mix(std::hash<int>{}(zone.generic_solver_mode));
    signature_mix(std::hash<int>{}(TextureMappingZone::DefaultGenericSolverMixModel));
    signature_mix(std::hash<int>{}(zone.preview_simulate_colors ? 1 : 0));
    signature_mix(std::hash<int>{}(zone.preview_limit_resolution ? 1 : 0));
    signature_mix_float(zone.contrast_pct, 100.f);
    signature_mix_float(zone.tone_gamma);
    for (const float strength_pct : zone.filament_strengths_pct)
        signature_mix_float(strength_pct, 100.f);
    for (const float minimum_offset_pct : zone.filament_minimum_offsets_pct)
        signature_mix_float(minimum_offset_pct, 100.f);

    if (zone.is_2d_gradient()) {
        signature_mix_float(texture_preview_config_float("texture_mapping_outer_wall_gradient_global_strength", 100.f), 100.f);
        signature_mix_float(texture_preview_config_float("texture_mapping_outer_wall_gradient_max_line_width", 0.95f), 1000.f);
        signature_mix_float(texture_preview_config_float("texture_mapping_outer_wall_gradient_min_line_width", 0.32f), 1000.f);
    }

    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    for (const std::string &color : physical_colors)
        signature_mix(std::hash<std::string>{}(color));
}

size_t texture_preview_settings_signature(size_t num_physical, const TextureMappingManager *texture_mgr)
{
    return texture_preview_settings_signature_impl(num_physical, texture_mgr, nullptr);
}

size_t texture_preview_settings_signature(size_t num_physical,
                                          const TextureMappingManager *texture_mgr,
                                          unsigned int base_filament_id,
                                          const std::vector<bool> *used_states)
{
    if (texture_mgr == nullptr)
        return texture_preview_settings_signature_impl(num_physical, texture_mgr, nullptr);

    std::vector<unsigned int> active_zone_ids = active_texture_preview_zone_ids(texture_mgr, base_filament_id, used_states);
    return texture_preview_settings_signature_impl(num_physical, texture_mgr, &active_zone_ids);
}

size_t texture_preview_model_settings_signature(size_t num_physical,
                                                const TextureMappingManager *texture_mgr,
                                                unsigned int base_filament_id,
                                                const std::vector<bool> *used_states,
                                                bool has_texture_preview_data,
                                                bool has_vertex_color_preview_data,
                                                bool has_texture_mapping_color_preview_data)
{
    size_t signature = 1469598103934665603ull;
    auto signature_mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };

    signature_mix(std::hash<size_t>{}(num_physical));
    if (texture_mgr == nullptr)
        return signature;

    const std::vector<unsigned int> active_zone_ids = active_texture_preview_zone_ids(texture_mgr, base_filament_id, used_states);
    for (const TextureMappingZone &zone : texture_mgr->zones()) {
        if (!active_texture_preview_zone_ids_contains(active_zone_ids, zone.zone_id))
            continue;

        signature_mix(std::hash<unsigned int>{}(zone.zone_id));
        signature_mix(std::hash<int>{}(zone.enabled ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.deleted ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.surface_pattern));

        const bool image_zone = zone.enabled && !zone.deleted && zone.is_image_texture();
        const bool gradient_zone = zone.enabled && !zone.deleted && zone.is_2d_gradient();
        const bool halftone_model = texture_preview_zone_uses_halftone_model(zone);
        const bool simulated_vertex_color_model =
            image_zone &&
            zone.preview_simulate_colors &&
            (has_texture_mapping_color_preview_data || (!has_texture_preview_data && has_vertex_color_preview_data));
        signature_mix(std::hash<int>{}(halftone_model ? 1 : 0));
        signature_mix(std::hash<int>{}(simulated_vertex_color_model ? 1 : 0));

        if (gradient_zone || halftone_model || simulated_vertex_color_model)
            texture_preview_mix_zone_baked_model_settings(signature, zone, num_physical);
    }
    return signature;
}

void render_model_texture_preview_models(
    std::vector<GUI::GLModel>       &models,
    const std::vector<ColorRGBA>    &colors,
    const std::vector<unsigned int> &filament_ids,
    size_t                           num_physical,
    const TextureMappingManager     *texture_mgr,
    const ModelVolume               &model_volume,
    const GUI::GLTexture            &texture,
    const Transform3d               &model_matrix,
    const Transform3d               &view_matrix,
    const Transform3d               &projection_matrix,
    const std::array<float, 2>      &z_range,
    const std::array<float, 4>      &clipping_plane,
    int                              print_volume_type,
    const std::array<float, 4>      &print_volume_xy,
    const std::array<float, 2>      &print_volume_z,
    bool                             opaque)
{
    if (models.empty() || colors.size() != models.size() || filament_ids.size() != models.size())
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_shader("painted_texture_preview");
    if (shader == nullptr)
        return;

    const TexturePreviewRenderState render_state = begin_render_state(opaque);
    shader->start_using();
    set_common_uniforms(*shader,
                        model_matrix,
                        view_matrix,
                        projection_matrix,
                        z_range,
                        clipping_plane,
                        print_volume_type,
                        print_volume_xy,
                        print_volume_z);
    glsafe(::glActiveTexture(GL_TEXTURE0));
    shader->set_uniform("uniform_texture", 0);

    const size_t texture_signature = model_volume_texture_preview_signature(model_volume);
    GLuint bound_texture_id = 0;
    for (size_t idx = 0; idx < models.size(); ++idx) {
        const bool raw_vertex_color_preview = filament_ids[idx] == 0;
        const float mix = raw_vertex_color_preview ?
            1.f :
            texture_preview_mix_for_filament(filament_ids[idx], num_physical, texture_mgr);
        const bool invalid = raw_vertex_color_preview ?
            false :
            texture_preview_settings_invalid_for_filament(filament_ids[idx], num_physical, texture_mgr);
        if (mix <= 0.f && !invalid)
            continue;

        const GUI::GLTexture *preview_texture = simulated_texture_preview_texture_for_filament(model_volume,
                                                                                               filament_ids[idx],
                                                                                               num_physical,
                                                                                               texture_mgr,
                                                                                               texture_signature,
                                                                                               texture);
        if (preview_texture == nullptr || preview_texture->get_id() == 0)
            continue;

        if (preview_texture->get_id() != bound_texture_id) {
            glsafe(::glBindTexture(GL_TEXTURE_2D, preview_texture->get_id()));
            bound_texture_id = preview_texture->get_id();
        }

        shader->set_uniform("texture_preview_mix", mix);
        shader->set_uniform("invalid_texture_mapping", invalid);
        models[idx].set_color(colors[idx]);
        models[idx].render();
    }

    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
    shader->stop_using();
    restore_render_state(render_state);
}

void render_model_texture_preview_model(
    GUI::GLModel                    &model,
    const ColorRGBA                 &color,
    unsigned int                     filament_id,
    size_t                           num_physical,
    const TextureMappingManager     *texture_mgr,
    const ModelVolume               &model_volume,
    const GUI::GLTexture            &texture,
    const Transform3d               &model_matrix,
    const Transform3d               &view_matrix,
    const Transform3d               &projection_matrix,
    const std::array<float, 2>      &z_range,
    const std::array<float, 4>      &clipping_plane,
    const std::pair<size_t, size_t> &render_range,
    int                              print_volume_type,
    const std::array<float, 4>      &print_volume_xy,
    const std::array<float, 2>      &print_volume_z,
    bool                             opaque)
{
    if (!GUI::GLModel::Geometry::has_tex_coord(model.get_geometry().format))
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_shader("painted_texture_preview");
    if (shader == nullptr)
        return;

    const float mix = texture_preview_mix_for_filament(filament_id, num_physical, texture_mgr);
    const bool invalid = texture_preview_settings_invalid_for_filament(filament_id, num_physical, texture_mgr);
    if (mix <= 0.f && !invalid)
        return;

    const size_t texture_signature = model_volume_texture_preview_signature(model_volume);
    const GUI::GLTexture *preview_texture = simulated_texture_preview_texture_for_filament(model_volume,
                                                                                           filament_id,
                                                                                           num_physical,
                                                                                           texture_mgr,
                                                                                           texture_signature,
                                                                                           texture);
    if (preview_texture == nullptr || preview_texture->get_id() == 0)
        return;

    const TexturePreviewRenderState render_state = begin_render_state(opaque);
    shader->start_using();
    set_common_uniforms(*shader,
                        model_matrix,
                        view_matrix,
                        projection_matrix,
                        z_range,
                        clipping_plane,
                        print_volume_type,
                        print_volume_xy,
                        print_volume_z);

    glsafe(::glActiveTexture(GL_TEXTURE0));
    glsafe(::glBindTexture(GL_TEXTURE_2D, preview_texture->get_id()));
    shader->set_uniform("uniform_texture", 0);
    shader->set_uniform("texture_preview_mix", mix);
    shader->set_uniform("invalid_texture_mapping", invalid);
    model.set_color(color);
    if (render_range == std::make_pair<size_t, size_t>(0, -1))
        model.render();
    else
        model.render(render_range);

    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
    shader->stop_using();
    restore_render_state(render_state);
}

void render_model_vertex_color_preview_models(
    std::vector<GUI::GLModel>       &models,
    const std::vector<ColorRGBA>    &colors,
    const std::vector<unsigned int> &filament_ids,
    size_t                           num_physical,
    const TextureMappingManager     *texture_mgr,
    const Transform3d               &model_matrix,
    const Transform3d               &view_matrix,
    const Transform3d               &projection_matrix,
    const std::array<float, 2>      &z_range,
    const std::array<float, 4>      &clipping_plane,
    int                              print_volume_type,
    const std::array<float, 4>      &print_volume_xy,
    const std::array<float, 2>      &print_volume_z,
    bool                             opaque)
{
    if (models.empty() || colors.size() != models.size() || filament_ids.size() != models.size())
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_shader("painted_vertex_color_preview");
    if (shader == nullptr)
        return;

    const TexturePreviewRenderState render_state = begin_render_state(opaque);
    shader->start_using();
    set_common_uniforms(*shader,
                        model_matrix,
                        view_matrix,
                        projection_matrix,
                        z_range,
                        clipping_plane,
                        print_volume_type,
                        print_volume_xy,
                        print_volume_z);

    for (size_t idx = 0; idx < models.size(); ++idx) {
        const bool raw_vertex_color_preview = filament_ids[idx] == 0;
        const float mix = raw_vertex_color_preview ?
            1.f :
            texture_preview_mix_for_filament(filament_ids[idx], num_physical, texture_mgr);
        const bool invalid = raw_vertex_color_preview ?
            false :
            texture_preview_settings_invalid_for_filament(filament_ids[idx], num_physical, texture_mgr);
        if (mix <= 0.f && !invalid)
            continue;
        shader->set_uniform("texture_preview_mix", mix);
        shader->set_uniform("invalid_texture_mapping", invalid);
        models[idx].set_color(colors[idx]);
        models[idx].render();
    }

    shader->stop_using();
    restore_render_state(render_state);
}

} // namespace Slic3r
