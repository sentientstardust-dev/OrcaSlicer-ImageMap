#include <glad/gl.h>

#include "MMUPaintedTexturePreview.hpp"

#include "3DScene.hpp"
#include "BitmapCache.hpp"
#include "GLShader.hpp"
#include "GUI_App.hpp"

#include "libslic3r/Config.hpp"
#include "libslic3r/Geometry.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/TextureMapping.hpp"
#include "libslic3r/filament_mixer.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>

namespace Slic3r {

namespace {

constexpr float k_preview_offset = 0.001f;
constexpr float k_polygon_offset_factor = -1.f;
constexpr float k_polygon_offset_units = -1.f;
constexpr float k_epsilon = 1e-6f;
constexpr unsigned int k_simulated_texture_preview_max_edge = 1024;
constexpr size_t k_simulated_texture_preview_max_pixels = 1024ull * 1024ull;

struct TexturePreviewMixCandidate
{
    std::array<float, 3> rgb;
    std::vector<float> weights;
};

struct TexturePreviewSimulationSettings
{
    int mapping_mode = int(TextureMappingZone::TextureMappingFilamentBlending);
    int filament_color_mode = TextureMappingZone::DefaultFilamentColorMode;
    bool force_sequential_filaments = false;
    bool limit_texture_resolution = true;
    bool compact_offset_mode = false;
    float contrast_pct = 100.f;
    float tone_gamma = 1.f;
    std::vector<unsigned int> component_ids;
    std::vector<std::array<float, 3>> component_colors;
    std::vector<float> component_strength_factors;
    std::vector<size_t> semantic_component_indices;
    std::vector<TexturePreviewMixCandidate> generic_mix_candidates;
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
    float sagging_ratio = 0.f;
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

struct TexturePreviewSimulationCacheEntry
{
    std::unique_ptr<GUI::GLTexture> texture;
    size_t uploaded_signature { 0 };
    size_t pending_signature { 0 };
    std::future<TexturePreviewSimulationResult> pending_future;
};

bool model_volume_has_texture_preview_data(const ModelVolume &model_volume)
{
    return !model_volume.imported_texture_rgba.empty() &&
           model_volume.imported_texture_width > 0 &&
           model_volume.imported_texture_height > 0 &&
           model_volume.imported_texture_uv_valid.size() == model_volume.mesh().its.indices.size() &&
           model_volume.imported_texture_uvs_per_face.size() >= model_volume.mesh().its.indices.size() * 6 &&
           model_volume.imported_texture_rgba.size() >=
               size_t(model_volume.imported_texture_width) * size_t(model_volume.imported_texture_height) * 4;
}

bool model_volume_has_vertex_color_preview_data(const ModelVolume &model_volume)
{
    return !model_volume.imported_vertex_colors_rgba.empty() &&
           model_volume.imported_vertex_colors_rgba.size() == model_volume.mesh().its.vertices.size();
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
        const float value_min = std::min({ values[0], values[1], values[2] });
        const float value_max = std::max({ values[0], values[1], values[2] });
        if (value_max - value_min <= 0.5f)
            return;

        for (float &value : values)
            if (value < 0.5f)
                value += 1.f;

        if (use_u_axis) {
            out[0].x() = values[0];
            out[1].x() = values[1];
            out[2].x() = values[2];
        } else {
            out[0].y() = values[0];
            out[1].y() = values[1];
            out[2].y() = values[2];
        }
    };

    unwrap_axis(true);
    unwrap_axis(false);
    return out;
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

    float total = 0.f;
    for (size_t idx = 0; idx < std::min(colors.size(), weights.size()); ++idx)
        total += std::max(0.f, weights[idx]);
    if (total <= k_epsilon)
        return { colors.front()[0], colors.front()[1], colors.front()[2], 1.f };

    float out_r = colors.front()[0];
    float out_g = colors.front()[1];
    float out_b = colors.front()[2];
    float accumulated = std::max(0.f, weights[0]);
    if (accumulated <= k_epsilon) {
        for (size_t idx = 1; idx < std::min(colors.size(), weights.size()); ++idx) {
            if (weights[idx] > k_epsilon) {
                out_r = colors[idx][0];
                out_g = colors[idx][1];
                out_b = colors[idx][2];
                accumulated = weights[idx];
                break;
            }
        }
    }

    for (size_t idx = 1; idx < std::min(colors.size(), weights.size()); ++idx) {
        const float weight = std::max(0.f, weights[idx]);
        if (weight <= k_epsilon)
            continue;
        const float t = weight / std::max(accumulated + weight, k_epsilon);
        filament_mixer_lerp_float(out_r, out_g, out_b, colors[idx][0], colors[idx][1], colors[idx][2], t, &out_r, &out_g, &out_b);
        accumulated += weight;
    }

    return { std::clamp(out_r, 0.f, 1.f), std::clamp(out_g, 0.f, 1.f), std::clamp(out_b, 0.f, 1.f), 1.f };
}

float clamp01(float value)
{
    return std::clamp(value, 0.f, 1.f);
}

unsigned char to_u8(float value)
{
    return static_cast<unsigned char>(clamp01(value) * 255.f + 0.5f);
}

void make_texture_preview_rgba_opaque(std::vector<unsigned char> &rgba)
{
    for (size_t idx = 3; idx < rgba.size(); idx += 4)
        rgba[idx] = 255;
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

std::array<unsigned int, 2> limited_simulated_texture_preview_size(unsigned int width, unsigned int height)
{
    if (width == 0 || height == 0)
        return { 0, 0 };

    double scale = 1.0;
    const unsigned int max_edge = std::max(width, height);
    if (max_edge > k_simulated_texture_preview_max_edge)
        scale = std::min(scale, double(k_simulated_texture_preview_max_edge) / double(max_edge));

    const double pixel_count = double(width) * double(height);
    if (pixel_count * scale * scale > double(k_simulated_texture_preview_max_pixels))
        scale = std::min(scale, std::sqrt(double(k_simulated_texture_preview_max_pixels) / pixel_count));

    if (scale >= 1.0)
        return { width, height };

    unsigned int limited_width = std::max(1u, unsigned(std::lround(double(width) * scale)));
    unsigned int limited_height = std::max(1u, unsigned(std::lround(double(height) * scale)));
    while (limited_width > k_simulated_texture_preview_max_edge ||
           limited_height > k_simulated_texture_preview_max_edge ||
           size_t(limited_width) * size_t(limited_height) > k_simulated_texture_preview_max_pixels) {
        if (limited_width >= limited_height && limited_width > 1)
            --limited_width;
        else if (limited_height > 1)
            --limited_height;
        else
            break;
    }
    return { limited_width, limited_height };
}

std::array<unsigned char, 3> sample_texture_preview_rgb_bilinear(const std::vector<unsigned char> &rgba,
                                                                 unsigned int width,
                                                                 unsigned int height,
                                                                 unsigned int preview_x,
                                                                 unsigned int preview_y,
                                                                 unsigned int preview_width,
                                                                 unsigned int preview_height)
{
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

    auto channel_at = [&rgba, width](unsigned int x, unsigned int y, size_t channel) {
        return double(rgba[(size_t(y) * size_t(width) + size_t(x)) * 4 + channel]);
    };
    auto sample_channel = [&](size_t channel) {
        const double top = channel_at(x0, y0, channel) * (1.0 - tx) + channel_at(x1, y0, channel) * tx;
        const double bottom = channel_at(x0, y1, channel) * (1.0 - tx) + channel_at(x1, y1, channel) * tx;
        return static_cast<unsigned char>(std::clamp(int(std::lround(top * (1.0 - ty) + bottom * ty)), 0, 255));
    };

    return { sample_channel(0), sample_channel(1), sample_channel(2) };
}

unsigned int texture_preview_rgb_cache_key(const std::array<unsigned char, 3> &rgb, bool quantize)
{
    if (quantize)
        return unsigned(rgb[0] >> 3) | (unsigned(rgb[1] >> 3) << 5) | (unsigned(rgb[2] >> 3) << 10);
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

std::array<float, 3> mix_component_colors_with_filament_mixer(const std::vector<std::array<float, 3>> &component_colors,
                                                              const std::vector<float>                &weights)
{
    if (component_colors.empty() || component_colors.size() != weights.size())
        return { 0.f, 0.f, 0.f };

    bool has_base = false;
    float out_r = 0.f;
    float out_g = 0.f;
    float out_b = 0.f;
    float accumulated = 0.f;
    for (size_t idx = 0; idx < component_colors.size(); ++idx) {
        const float weight = clamp01(weights[idx]);
        if (weight <= k_epsilon)
            continue;

        if (!has_base) {
            out_r = component_colors[idx][0];
            out_g = component_colors[idx][1];
            out_b = component_colors[idx][2];
            accumulated = weight;
            has_base = true;
            continue;
        }

        const float t = weight / std::max(k_epsilon, accumulated + weight);
        float mixed_r = out_r;
        float mixed_g = out_g;
        float mixed_b = out_b;
        filament_mixer_lerp_float(out_r,
                                  out_g,
                                  out_b,
                                  component_colors[idx][0],
                                  component_colors[idx][1],
                                  component_colors[idx][2],
                                  t,
                                  &mixed_r,
                                  &mixed_g,
                                  &mixed_b);
        out_r = clamp01(mixed_r);
        out_g = clamp01(mixed_g);
        out_b = clamp01(mixed_b);
        accumulated += weight;
    }

    if (!has_base)
        return component_colors.front();
    return { out_r, out_g, out_b };
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
    default:
        return {};
    }

    return best_matching_component_indices_for_semantic_colors(component_colors, semantic_colors);
}

bool texture_preview_uses_generic_solver(const TexturePreviewSimulationSettings &settings)
{
    if (settings.mapping_mode == int(TextureMappingZone::TextureMappingRawValues))
        return false;

    const int clamped_mode = std::clamp(settings.filament_color_mode,
                                        int(TextureMappingZone::FilamentColorAny),
                                        int(TextureMappingZone::FilamentColorBW));
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
    const int total_units = component_count <= 4 ? 20 : (component_count <= 6 ? 10 : 6);
    std::vector<int> units(component_count, 0);
    std::vector<TexturePreviewMixCandidate> candidates;
    candidates.reserve(4096);

    std::function<void(size_t, int)> recurse = [&](size_t idx, int remaining_units) {
        if (idx + 1 == component_count) {
            units[idx] = remaining_units;
            TexturePreviewMixCandidate candidate;
            candidate.weights.assign(component_count, 0.f);
            for (size_t weight_idx = 0; weight_idx < component_count; ++weight_idx)
                candidate.weights[weight_idx] = float(units[weight_idx]) / float(std::max(1, total_units));
            candidate.rgb = mix_component_colors_with_filament_mixer(component_colors, candidate.weights);
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

std::vector<float> best_component_mix_weights_for_target(const std::vector<TexturePreviewMixCandidate> &candidates,
                                                         const std::array<float, 3>                    &target_rgb)
{
    if (candidates.empty())
        return {};

    const TexturePreviewMixCandidate *best_candidate = nullptr;
    float best_error = std::numeric_limits<float>::max();
    for (const TexturePreviewMixCandidate &candidate : candidates) {
        const float error = color_distance_sq(candidate.rgb, target_rgb);
        if (error < best_error) {
            best_error = error;
            best_candidate = &candidate;
        }
    }

    return best_candidate != nullptr ? best_candidate->weights : std::vector<float>{};
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

std::vector<float> optimized_primary_component_weights_for_target(const std::array<float, 3>              &target_rgb,
                                                                  size_t                                   component_count,
                                                                  int                                      filament_color_mode,
                                                                  const std::vector<std::array<float, 3>> &component_colors,
                                                                  bool                                     force_sequential_filaments,
                                                                  const std::vector<size_t>               &semantic_component_indices)
{
    const int clamped_mode = std::clamp(filament_color_mode,
                                        int(TextureMappingZone::FilamentColorAny),
                                        int(TextureMappingZone::FilamentColorBW));
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
                                                         const std::array<float, 4>            &sample_rgba)
{
    const size_t component_count = settings.component_colors.size();
    if (component_count == 0)
        return {};

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
        std::vector<float> optimized = optimized_primary_component_weights_for_target(target,
                                                                                      component_count,
                                                                                      settings.filament_color_mode,
                                                                                      settings.component_colors,
                                                                                      settings.force_sequential_filaments,
                                                                                      settings.semantic_component_indices);
        if (optimized.size() == component_count)
            desired = std::move(optimized);
        else {
            std::vector<float> best = best_component_mix_weights_for_target(settings.generic_mix_candidates, target);
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

    for (size_t idx = 0; idx < desired.size() && idx < settings.component_strength_factors.size(); ++idx)
        desired[idx] = clamp01(desired[idx] * settings.component_strength_factors[idx]);
    return desired;
}

void prepare_texture_preview_simulation_settings(TexturePreviewSimulationSettings &settings)
{
    settings.semantic_component_indices =
        semantic_component_indices_for_texture_preview(settings.component_colors,
                                                       settings.filament_color_mode,
                                                       settings.force_sequential_filaments);
    if (texture_preview_uses_generic_solver(settings))
        settings.generic_mix_candidates = build_generic_mix_candidates(settings.component_colors);
    else
        settings.generic_mix_candidates.clear();
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

    const std::array<float, 3> simulated_rgb = mix_component_colors_with_filament_mixer(settings->component_colors, component_weights);
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
                                              int(TextureMappingZone::FilamentColorBW));
    settings.force_sequential_filaments = zone->force_sequential_filaments;
    settings.limit_texture_resolution = zone->preview_limit_resolution;
    settings.compact_offset_mode = zone->compact_offset_mode;
    settings.contrast_pct = std::clamp(zone->contrast_pct, 25.f, 300.f);
    settings.tone_gamma = (!std::isfinite(zone->tone_gamma) || zone->tone_gamma <= 0.f) ?
        1.f :
        std::clamp(zone->tone_gamma, 0.5f, 3.f);
    settings.component_ids = TextureMappingManager::effective_texture_component_ids(*zone, num_physical, physical_colors);
    if (settings.component_ids.empty())
        return std::nullopt;

    const bool raw_values_mode = settings.mapping_mode == int(TextureMappingZone::TextureMappingRawValues);
    settings.component_colors.reserve(settings.component_ids.size());
    settings.component_strength_factors.reserve(settings.component_ids.size());
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
    }

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
    mix(std::hash<int>{}(int(std::lround(settings.contrast_pct * 100.f))));
    mix(std::hash<int>{}(int(std::lround(settings.tone_gamma * 1000.f))));
    for (const unsigned int id : settings.component_ids)
        mix(std::hash<unsigned int>{}(id));
    for (const auto &color : settings.component_colors) {
        mix(std::hash<int>{}(int(std::lround(color[0] * 255.f))));
        mix(std::hash<int>{}(int(std::lround(color[1] * 255.f))));
        mix(std::hash<int>{}(int(std::lround(color[2] * 255.f))));
    }
    for (const float strength_factor : settings.component_strength_factors)
        mix(std::hash<int>{}(int(std::lround(strength_factor * 1000.f))));
    return signature;
}

TexturePreviewSimulationResult build_simulated_texture_preview_result(size_t signature,
                                                                      unsigned int width,
                                                                      unsigned int height,
                                                                      std::vector<unsigned char> source_rgba,
                                                                      TexturePreviewSimulationSettings settings)
{
    TexturePreviewSimulationResult result;
    result.signature = signature;
    if (width == 0 || height == 0 || source_rgba.size() < size_t(width) * size_t(height) * 4)
        return result;

    const std::array<unsigned int, 2> preview_size = settings.limit_texture_resolution ?
        limited_simulated_texture_preview_size(width, height) :
        std::array<unsigned int, 2>{ width, height };
    result.width = preview_size[0];
    result.height = preview_size[1];
    result.rgba.resize(size_t(result.width) * size_t(result.height) * 4, 0);
    if (result.width == 0 || result.height == 0)
        return result;

    prepare_texture_preview_simulation_settings(settings);
    const bool use_generic_solver = !settings.generic_mix_candidates.empty();

    std::unordered_map<unsigned int, std::array<unsigned char, 4>> simulated_color_cache;
    simulated_color_cache.reserve(std::min(size_t(result.width) * size_t(result.height),
                                           use_generic_solver ? size_t(32768) : size_t(65536)));

    for (unsigned int y = 0; y < result.height; ++y) {
        for (unsigned int x = 0; x < result.width; ++x) {
            const std::array<unsigned char, 3> source_rgb =
                sample_texture_preview_rgb_bilinear(source_rgba, width, height, x, y, result.width, result.height);
            const unsigned int cache_key = texture_preview_rgb_cache_key(source_rgb, use_generic_solver);
            const size_t idx = (size_t(y) * size_t(result.width) + size_t(x)) * 4;

            auto cached_color = simulated_color_cache.find(cache_key);
            if (cached_color != simulated_color_cache.end()) {
                result.rgba[idx + 0] = cached_color->second[0];
                result.rgba[idx + 1] = cached_color->second[1];
                result.rgba[idx + 2] = cached_color->second[2];
                result.rgba[idx + 3] = cached_color->second[3];
                continue;
            }

            const std::array<float, 4> sample_rgba = {
                float(source_rgb[0]) / 255.f,
                float(source_rgb[1]) / 255.f,
                float(source_rgb[2]) / 255.f,
                1.f
            };
            const std::vector<float> component_weights = component_weights_for_texture_preview(settings, sample_rgba);
            float activity = 0.f;
            for (const float weight : component_weights)
                activity = std::max(activity, clamp01(weight));

            const std::array<float, 3> simulated_rgb = activity > k_epsilon ?
                mix_component_colors_with_filament_mixer(settings.component_colors, component_weights) :
                std::array<float, 3>{ sample_rgba[0], sample_rgba[1], sample_rgba[2] };

            const std::array<unsigned char, 4> out_rgba = {
                to_u8(simulated_rgb[0]),
                to_u8(simulated_rgb[1]),
                to_u8(simulated_rgb[2]),
                255
            };
            simulated_color_cache.emplace(cache_key, out_rgba);
            result.rgba[idx + 0] = out_rgba[0];
            result.rgba[idx + 1] = out_rgba[1];
            result.rgba[idx + 2] = out_rgba[2];
            result.rgba[idx + 3] = out_rgba[3];
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

void discard_ready_texture_preview_future(TexturePreviewSimulationCacheEntry &entry)
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

} // namespace

void clear_texture_preview_simulation_cache()
{
    prune_abandoned_texture_preview_futures();

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

        if (entry->pending_future.valid()) {
            if (entry->pending_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                discard_ready_texture_preview_future(*entry);
            } else {
                abandoned_texture_preview_futures().emplace_back(std::move(entry->pending_future));
            }
        }
        it = cache.erase(it);
    }
}

namespace {

bool texture_preview_simulation_is_pending_impl()
{
    const bool abandoned_pending = prune_abandoned_texture_preview_futures();
    auto &cache = texture_preview_simulation_cache();
    for (auto it = cache.begin(); it != cache.end();) {
        const std::shared_ptr<TexturePreviewSimulationCacheEntry> &entry = it->second;
        if (entry == nullptr) {
            it = cache.erase(it);
            continue;
        }

        if (!entry->pending_future.valid()) {
            ++it;
            continue;
        }

        if (entry->pending_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return true;

        if (entry->texture == nullptr && entry->pending_signature == 0) {
            discard_ready_texture_preview_future(*entry);
            it = cache.erase(it);
            continue;
        }

        ++it;
    }
    return abandoned_pending;
}

size_t texture_preview_simulation_cache_key(const ModelVolume &model_volume, unsigned int filament_id)
{
    size_t key = reinterpret_cast<size_t>(&model_volume);
    key ^= std::hash<unsigned int>{}(filament_id) + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
    return key;
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

    const size_t simulation_signature = texture_preview_simulation_signature(model_volume, source_texture_signature, *settings);
    auto &cache = texture_preview_simulation_cache();
    const size_t cache_key = texture_preview_simulation_cache_key(model_volume, filament_id);
    std::shared_ptr<TexturePreviewSimulationCacheEntry> &entry_ref = cache[cache_key];
    if (entry_ref == nullptr)
        entry_ref = std::make_shared<TexturePreviewSimulationCacheEntry>();
    TexturePreviewSimulationCacheEntry &entry = *entry_ref;

    if (entry.pending_future.valid() && entry.pending_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        TexturePreviewSimulationResult result = entry.pending_future.get();
        if (result.signature == entry.pending_signature && !result.rgba.empty() && result.width > 0 && result.height > 0) {
            if (entry.texture == nullptr)
                entry.texture = std::make_unique<GUI::GLTexture>();
            else
                entry.texture->reset();

            if (entry.texture->load_from_raw_data(std::move(result.rgba), result.width, result.height)) {
                configure_texture_preview_sampler(*entry.texture);
                entry.uploaded_signature = result.signature;
            } else {
                entry.uploaded_signature = 0;
            }
        }
        entry.pending_signature = 0;
    }

    if (entry.texture != nullptr && entry.uploaded_signature == simulation_signature && entry.texture->get_id() != 0)
        return entry.texture.get();

    if (!entry.pending_future.valid()) {
        entry.pending_signature = simulation_signature;
        const unsigned int width = model_volume.imported_texture_width;
        const unsigned int height = model_volume.imported_texture_height;
        std::vector<unsigned char> source_rgba(model_volume.imported_texture_rgba.begin(), model_volume.imported_texture_rgba.end());
        TexturePreviewSimulationSettings simulation_settings = *settings;
        entry.pending_future = std::async(std::launch::async,
                                          [simulation_signature,
                                           width,
                                           height,
                                           source_rgba = std::move(source_rgba),
                                           simulation_settings = std::move(simulation_settings)]() mutable {
                                              return build_simulated_texture_preview_result(simulation_signature,
                                                                                           width,
                                                                                           height,
                                                                                           std::move(source_rgba),
                                                                                           std::move(simulation_settings));
                                          });
    }

    return &fallback_texture;
}

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

bool build_vertex_color_preview_model_for_state(const ModelVolume                                      &model_volume,
                                                const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles,
                                                const TexturePreviewSimulationSettings                   *simulation_settings,
                                                GUI::GLModel                                            &out_model)
{
    if (!model_volume_has_vertex_color_preview_data(model_volume) || state_triangles.empty())
        return false;

    const indexed_triangle_set &its = model_volume.mesh().its;
    GUI::GLModel::Geometry geometry;
    geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };
    geometry.reserve_vertices(state_triangles.size() * 3);
    geometry.reserve_indices(state_triangles.size() * 3);

    std::unordered_map<uint32_t, ColorRGBA> simulated_color_cache;
    if (simulation_settings != nullptr)
        simulated_color_cache.reserve(std::min(model_volume.imported_vertex_colors_rgba.size(), size_t(65536)));
    auto source_vertex_color = [simulation_settings, &simulated_color_cache](uint32_t packed) {
        const ColorRGBA source_color = unpack_vertex_color(packed);
        if (simulation_settings == nullptr)
            return source_color;

        auto cached = simulated_color_cache.find(packed);
        if (cached != simulated_color_cache.end())
            return cached->second;

        const ColorRGBA simulated_color = simulated_texture_preview_color_for_vertex_color(&source_color, simulation_settings);
        simulated_color_cache.emplace(packed, simulated_color);
        return simulated_color;
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
            size_t(source_indices[0]) >= model_volume.imported_vertex_colors_rgba.size() ||
            size_t(source_indices[1]) >= model_volume.imported_vertex_colors_rgba.size() ||
            size_t(source_indices[2]) >= model_volume.imported_vertex_colors_rgba.size())
            continue;

        const Vec3f source_p0 = its.vertices[size_t(source_indices[0])].cast<float>();
        const Vec3f source_p1 = its.vertices[size_t(source_indices[1])].cast<float>();
        const Vec3f source_p2 = its.vertices[size_t(source_indices[2])].cast<float>();
        const std::array<ColorRGBA, 3> source_colors = {
            source_vertex_color(model_volume.imported_vertex_colors_rgba[size_t(source_indices[0])]),
            source_vertex_color(model_volume.imported_vertex_colors_rgba[size_t(source_indices[1])]),
            source_vertex_color(model_volume.imported_vertex_colors_rgba[size_t(source_indices[2])])
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
            leaf_colors[vertex_idx] = interpolate_color(source_colors, barycentric);
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

    out_model.init_from(std::move(geometry));
    return true;
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
    const std::array<float, 3> rgb = mix_component_colors_with_filament_mixer(settings.component_colors, weights);
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
    if (GUI::wxGetApp().preset_bundle == nullptr)
        return fallback;

    const DynamicPrintConfig &config = GUI::wxGetApp().preset_bundle->project_config;
    if (const ConfigOptionFloat *opt = config.option<ConfigOptionFloat>(key))
        return std::isfinite(opt->value) ? float(opt->value) : fallback;
    return fallback;
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
    settings.sagging_ratio = std::isfinite(zone.sagging_ratio) ? std::clamp(zone.sagging_ratio, 0.f, 6.f) : 0.f;

    const float base_outer_width_mm = std::max(0.05f, surface_gradient_preview_config_float("texture_mapping_outer_wall_gradient_max_line_width", 0.95f));
    const float min_outer_width_mm = std::clamp(surface_gradient_preview_config_float("texture_mapping_outer_wall_gradient_min_line_width", 0.32f),
                                                0.05f,
                                                base_outer_width_mm);
    const float global_strength_factor =
        std::clamp(surface_gradient_preview_config_float("texture_mapping_outer_wall_gradient_global_strength", 100.f) / 100.f, 0.f, 1.f);
    settings.max_width_delta_limit_mm = std::min((base_outer_width_mm - min_outer_width_mm) * global_strength_factor, 2.f * max_distance_mm);
    if (settings.sagging_ratio > k_epsilon) {
        constexpr float preview_layer_height_mm = 0.2f;
        settings.max_width_delta_limit_mm = std::min(settings.max_width_delta_limit_mm, preview_layer_height_mm * settings.sagging_ratio);
    }
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
    GLboolean polygon_offset_fill_enabled { GL_FALSE };
    GLboolean depth_mask { GL_TRUE };
    GLfloat   polygon_offset_factor { 0.f };
    GLfloat   polygon_offset_units { 0.f };
    GLint     depth_func { GL_LESS };
};

TexturePreviewRenderState begin_render_state()
{
    TexturePreviewRenderState state;
    state.blend_enabled = glIsEnabled(GL_BLEND);
    state.polygon_offset_fill_enabled = glIsEnabled(GL_POLYGON_OFFSET_FILL);
    glsafe(::glGetBooleanv(GL_DEPTH_WRITEMASK, &state.depth_mask));
    glsafe(::glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &state.polygon_offset_factor));
    glsafe(::glGetFloatv(GL_POLYGON_OFFSET_UNITS, &state.polygon_offset_units));
    glsafe(::glGetIntegerv(GL_DEPTH_FUNC, &state.depth_func));

    glsafe(::glEnable(GL_BLEND));
    glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    glsafe(::glDepthMask(GL_FALSE));
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
    if (!state.blend_enabled)
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
    shader.set_uniform("volume_world_matrix", model_matrix);
    shader.set_uniform("z_range", z_range);
    shader.set_uniform("clipping_plane", clipping_plane);
    shader.set_uniform("print_volume.type", print_volume_type);
    shader.set_uniform("print_volume.xy_data", print_volume_xy);
    shader.set_uniform("print_volume.z_data", print_volume_z);
}

} // namespace

bool texture_preview_simulation_is_pending()
{
    return texture_preview_simulation_is_pending_impl();
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

    bool built_any = false;
    for (size_t state_id = 0; state_id < triangles_per_type.size(); ++state_id) {
        const unsigned int filament_id = filament_id_for_state(state_id, base_filament_id);
        const TextureMappingZone *zone = zone_for_filament(filament_id, num_physical, texture_mgr);
        if (zone == nullptr || !is_image_zone(*zone))
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
    std::vector<unsigned int>                                            &out_filament_ids)
{
    out_models.clear();
    out_colors.clear();
    out_filament_ids.clear();

    const std::vector<std::string> physical_colors = physical_filament_colors_for_texture_preview(num_physical);
    bool built_any = false;
    for (size_t state_id = 0; state_id < triangles_per_type.size(); ++state_id) {
        const unsigned int filament_id = filament_id_for_state(state_id, base_filament_id);
        const TextureMappingZone *zone = zone_for_filament(filament_id, num_physical, texture_mgr);
        if (zone == nullptr || (!is_image_zone(*zone) && !is_gradient_zone(*zone)))
            continue;

        GUI::GLModel model;
        if (is_gradient_zone(*zone)) {
            std::optional<SurfaceGradientPreviewSettings> settings = surface_gradient_preview_settings_for_zone(model_volume, world_matrix, *zone, num_physical);
            if (!settings)
                continue;
            if (!build_surface_gradient_vertex_color_preview_model_for_state(triangles_per_type[state_id], *settings, world_matrix, model))
                continue;
        } else {
            std::optional<TexturePreviewSimulationSettings> simulation_settings =
                texture_preview_simulation_settings_for_filament(filament_id, num_physical, texture_mgr, physical_colors);
            if (simulation_settings)
                prepare_texture_preview_simulation_settings(*simulation_settings);
            if (!build_vertex_color_preview_model_for_state(model_volume,
                                                            triangles_per_type[state_id],
                                                            simulation_settings ? &*simulation_settings : nullptr,
                                                            model))
                continue;
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
    std::vector<unsigned int>                                            &out_filament_ids)
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
                                                 out_filament_ids);
}

size_t model_volume_texture_preview_signature(const ModelVolume &model_volume)
{
    size_t signature = 1469598103934665603ull;
    auto mix = [&signature](size_t value) {
        signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
    };
    mix(size_t(model_volume.imported_texture_width));
    mix(size_t(model_volume.imported_texture_height));
    mix(model_volume.imported_texture_rgba.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_texture_rgba.data()));
    mix(model_volume.imported_texture_uvs_per_face.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_texture_uvs_per_face.data()));
    mix(model_volume.imported_texture_uv_valid.size());
    mix(reinterpret_cast<size_t>(model_volume.imported_texture_uv_valid.data()));
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
    make_texture_preview_rgba_opaque(texture_data);
    if (!texture.load_from_raw_data(std::move(texture_data), model_volume.imported_texture_width, model_volume.imported_texture_height)) {
        texture_signature = 0;
        return false;
    }

    configure_texture_preview_sampler(texture);
    texture_signature = preview_signature;
    return true;
}

size_t texture_preview_settings_signature(size_t num_physical, const TextureMappingManager *texture_mgr)
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
    if (texture_mgr == nullptr)
        return signature;

    for (const TextureMappingZone &zone : texture_mgr->zones()) {
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
        signature_mix(std::hash<int>{}(zone.preview_simulate_colors ? 1 : 0));
        signature_mix(std::hash<int>{}(zone.preview_limit_resolution ? 1 : 0));
        signature_mix_float(zone.sagging_ratio);
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
    const std::array<float, 2>      &print_volume_z)
{
    if (models.empty() || colors.size() != models.size() || filament_ids.size() != models.size() || texture.get_id() == 0)
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_shader("painted_texture_preview");
    if (shader == nullptr)
        return;

    const TexturePreviewRenderState render_state = begin_render_state();
    shader->start_using();
    set_common_uniforms(*shader, model_matrix, view_matrix, projection_matrix, z_range, clipping_plane, print_volume_type, print_volume_xy, print_volume_z);
    glsafe(::glActiveTexture(GL_TEXTURE0));
    shader->set_uniform("uniform_texture", 0);

    const size_t texture_signature = model_volume_texture_preview_signature(model_volume);
    GLuint bound_texture_id = 0;
    for (size_t idx = 0; idx < models.size(); ++idx) {
        const float mix = texture_preview_mix_for_filament(filament_ids[idx], num_physical, texture_mgr);
        const bool invalid = texture_preview_settings_invalid_for_filament(filament_ids[idx], num_physical, texture_mgr);
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
    const std::array<float, 2>      &print_volume_z)
{
    if (models.empty() || colors.size() != models.size() || filament_ids.size() != models.size())
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_shader("painted_vertex_color_preview");
    if (shader == nullptr)
        return;

    const TexturePreviewRenderState render_state = begin_render_state();
    shader->start_using();
    set_common_uniforms(*shader, model_matrix, view_matrix, projection_matrix, z_range, clipping_plane, print_volume_type, print_volume_xy, print_volume_z);

    for (size_t idx = 0; idx < models.size(); ++idx) {
        const float mix = texture_preview_mix_for_filament(filament_ids[idx], num_physical, texture_mgr);
        const bool invalid = texture_preview_settings_invalid_for_filament(filament_ids[idx], num_physical, texture_mgr);
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
