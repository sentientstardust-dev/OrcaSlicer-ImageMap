#ifndef WipeTower_
#define WipeTower_

#include <cmath>
#include <string>
#include <sstream>
#include <utility>
#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/TextureMapping.hpp"
#include "libslic3r/Polyline.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include <unordered_set>

namespace Slic3r
{

class WipeTowerWriter;
class PrintConfig;
enum GCodeFlavor : unsigned char;

struct PrimeTowerTextureRenderSettings
{
    enum ColorMode : int {
        Auto = 0,
        GenericSolver,
        CMY,
        CMYK,
        CMYW,
        RGB,
        RGBK,
        RGBW,
        BW
    };

    bool enabled = false;
    float angle_offset_deg = 0.f;
    int color_mode = Auto;
    bool generic_fallback_for_missing_channels = false;
    bool compact_offset_mode = true;
    bool settings_zone_enabled = false;
    int texture_mapping_mode = int(TextureMappingZone::TextureMappingFilamentBlending);
    int texture_filament_color_mode = int(TextureMappingZone::FilamentColorAny);
    float contrast_pct = 100.f;
    float tone_gamma = 1.f;
    float sagging_ratio = 0.f;
    float global_strength = 1.f;
    float max_line_width = 0.95f;
    float min_line_width = 0.32f;
    std::vector<uint8_t> image_rgba;
    unsigned int image_width = 0;
    unsigned int image_height = 0;
    std::vector<uint8_t> image_rgba_back;
    unsigned int image_width_back = 0;
    unsigned int image_height_back = 0;
    std::vector<std::string> filament_colours;
    std::vector<size_t> tool_indices;
    std::vector<unsigned int> component_ids;
    std::vector<float> filament_strengths_pct;
    std::vector<float> filament_minimum_offsets_pct;
    float z_min = 0.f;
    float z_max = 0.f;

    bool valid() const
    {
        return enabled && (image_valid(false) || image_valid(true));
    }

    float sample_tool_visibility(size_t tool, float u, float v) const
    {
        const float raw_visibility = sample_tool_visibility_raw(tool, u, v);
        if (!compact_offset_mode)
            return adjusted_tool_visibility(tool, raw_visibility);

        float max_visibility = std::clamp(raw_visibility, 0.f, 1.f);
        if (!tool_indices.empty()) {
            for (const size_t candidate_tool : tool_indices) {
                if (candidate_tool != tool)
                    max_visibility = std::max(max_visibility, std::clamp(sample_tool_visibility_raw(candidate_tool, u, v), 0.f, 1.f));
            }
        } else {
            for (size_t candidate_tool = 0; candidate_tool < filament_colours.size(); ++candidate_tool) {
                if (candidate_tool != tool)
                    max_visibility = std::max(max_visibility, std::clamp(sample_tool_visibility_raw(candidate_tool, u, v), 0.f, 1.f));
            }
        }
        return max_visibility > 1e-6f ? adjusted_tool_visibility(tool, raw_visibility / max_visibility) :
                                        adjusted_tool_visibility(tool, raw_visibility);
    }

    float sample_tool_visibility(size_t tool, float u, float v, const std::vector<size_t> &normalization_tools) const
    {
        const float raw_visibility = sample_tool_visibility_raw(tool, u, v);
        if (!compact_offset_mode || normalization_tools.empty())
            return adjusted_tool_visibility(tool, raw_visibility);

        float max_visibility = std::clamp(raw_visibility, 0.f, 1.f);
        for (const size_t normalization_tool : normalization_tools)
            max_visibility = std::max(max_visibility, std::clamp(sample_tool_visibility_raw(normalization_tool, u, v), 0.f, 1.f));

        return max_visibility > 1e-6f ? adjusted_tool_visibility(tool, raw_visibility / max_visibility) :
                                        adjusted_tool_visibility(tool, raw_visibility);
    }

    std::vector<size_t> component_tools_for_layer_sequence() const
    {
        std::vector<size_t> tools;
        if (!valid())
            return tools;

        if (settings_zone_enabled && !component_ids.empty()) {
            tools.reserve(component_ids.size());
            for (const unsigned int physical_id : component_ids) {
                if (physical_id >= 1 && physical_id <= filament_colours.size()) {
                    const size_t tool = size_t(physical_id - 1);
                    if (std::find(tools.begin(), tools.end(), tool) == tools.end())
                        tools.emplace_back(tool);
                }
            }
            if (!tools.empty())
                return tools;
        }

        const std::vector<std::array<float, 3>> ideals = color_mode_ideals(color_mode);
        if (ideals.empty())
            return tools;

        std::vector<size_t> candidates;
        if (!tool_indices.empty()) {
            for (const size_t tool : tool_indices)
                if (tool < filament_colours.size() && std::find(candidates.begin(), candidates.end(), tool) == candidates.end())
                    candidates.emplace_back(tool);
        } else {
            candidates.reserve(filament_colours.size());
            for (size_t tool = 0; tool < filament_colours.size(); ++tool)
                candidates.emplace_back(tool);
        }
        if (candidates.empty())
            return tools;

        std::vector<char> used(filament_colours.size(), 0);
        tools.reserve(ideals.size());
        for (const std::array<float, 3> &ideal : ideals) {
            size_t best_tool = size_t(-1);
            float best_distance = std::numeric_limits<float>::max();
            for (const size_t candidate : candidates) {
                if (candidate >= used.size() || used[candidate])
                    continue;
                const float distance = color_distance2(tool_color(candidate), ideal);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_tool = candidate;
                }
            }
            if (best_tool != size_t(-1)) {
                used[best_tool] = 1;
                tools.emplace_back(best_tool);
            }
        }
        return tools;
    }

    size_t tool_for_layer_sequence(int layer_index) const
    {
        const std::vector<size_t> tools = component_tools_for_layer_sequence();
        if (tools.empty())
            return size_t(-1);
        const int tool_count = int(tools.size());
        const int index = ((layer_index % tool_count) + tool_count) % tool_count;
        return tools[size_t(index)];
    }

private:
    static std::vector<std::array<float, 3>> color_mode_ideals(int mode)
    {
        switch (mode) {
        case CMY:
            return {std::array<float, 3>{0.f, 1.f, 1.f}, std::array<float, 3>{1.f, 0.f, 1.f}, std::array<float, 3>{1.f, 1.f, 0.f}};
        case CMYK:
            return {std::array<float, 3>{0.f, 1.f, 1.f},
                    std::array<float, 3>{1.f, 0.f, 1.f},
                    std::array<float, 3>{1.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 0.f}};
        case CMYW:
            return {std::array<float, 3>{0.f, 1.f, 1.f},
                    std::array<float, 3>{1.f, 0.f, 1.f},
                    std::array<float, 3>{1.f, 1.f, 0.f},
                    std::array<float, 3>{1.f, 1.f, 1.f}};
        case RGB:
            return {std::array<float, 3>{1.f, 0.f, 0.f}, std::array<float, 3>{0.f, 1.f, 0.f}, std::array<float, 3>{0.f, 0.f, 1.f}};
        case RGBK:
            return {std::array<float, 3>{1.f, 0.f, 0.f},
                    std::array<float, 3>{0.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 1.f},
                    std::array<float, 3>{0.f, 0.f, 0.f}};
        case RGBW:
            return {std::array<float, 3>{1.f, 0.f, 0.f},
                    std::array<float, 3>{0.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 1.f},
                    std::array<float, 3>{1.f, 1.f, 1.f}};
        case BW:
            return {std::array<float, 3>{0.f, 0.f, 0.f}, std::array<float, 3>{1.f, 1.f, 1.f}};
        default: return {};
        }
    }

    float sample_tool_visibility_raw(size_t tool, float u, float v) const
    {
        if (!valid())
            return 1.f;
        u -= std::floor(u);
        v = std::clamp(v, 0.f, 1.f);
        const bool front_valid = image_valid(false);
        const bool back_valid = image_valid(true);
        bool use_back = false;
        if (front_valid && back_valid) {
            use_back = u >= 0.5f;
            u = use_back ? (u - 0.5f) * 2.f : u * 2.f;
        } else {
            use_back = back_valid;
        }
        if (settings_zone_enabled)
            return sample_settings_zone_tool_visibility(tool, u, v, use_back);
        return sample_image_tool_visibility(tool, u, v, use_back);
    }

    bool image_valid(bool back) const
    {
        return back ?
            image_width_back > 0 && image_height_back > 0 &&
                image_rgba_back.size() >= size_t(image_width_back) * size_t(image_height_back) * 4 :
            image_width > 0 && image_height > 0 &&
                image_rgba.size() >= size_t(image_width) * size_t(image_height) * 4;
    }

    std::array<float, 3> sample_image_rgb(float u, float v, bool back) const
    {
        const std::vector<uint8_t> &rgba = back ? image_rgba_back : image_rgba;
        const unsigned int width = back ? image_width_back : image_width;
        const unsigned int height = back ? image_height_back : image_height;
        if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
            return {1.f, 1.f, 1.f};

        const float x = u * float(width);
        const float y = (1.f - v) * float(height - 1);
        const unsigned int ix = std::min<unsigned int>(width - 1, unsigned(std::floor(x)));
        const unsigned int iy = std::min<unsigned int>(height - 1, unsigned(std::floor(y)));
        const size_t offset = (size_t(iy) * size_t(width) + size_t(ix)) * 4;
        const float r = float(rgba[offset + 0]) / 255.f;
        const float g = float(rgba[offset + 1]) / 255.f;
        const float b = float(rgba[offset + 2]) / 255.f;
        return {r, g, b};
    }

    float sample_image_tool_visibility(size_t tool, float u, float v, bool back) const
    {
        const std::array<float, 3> rgb = sample_image_rgb(u, v, back);
        return color_mode == GenericSolver || color_mode == Auto ? generic_visibility(tool, rgb[0], rgb[1], rgb[2]) :
                                                                   fixed_mode_visibility(tool, rgb[0], rgb[1], rgb[2]);
    }

    static std::array<float, 3> parse_color(const std::string &hex)
    {
        auto hex_byte = [](char hi, char lo) {
            auto nibble = [](char c) {
                if (c >= '0' && c <= '9') return int(c - '0');
                if (c >= 'a' && c <= 'f') return int(c - 'a') + 10;
                if (c >= 'A' && c <= 'F') return int(c - 'A') + 10;
                return 0;
            };
            return float(nibble(hi) * 16 + nibble(lo)) / 255.f;
        };
        if (hex.size() >= 7 && hex[0] == '#')
            return {hex_byte(hex[1], hex[2]), hex_byte(hex[3], hex[4]), hex_byte(hex[5], hex[6])};
        return {1.f, 1.f, 1.f};
    }

    static float color_distance2(const std::array<float, 3> &a, const std::array<float, 3> &b)
    {
        const float dr = a[0] - b[0];
        const float dg = a[1] - b[1];
        const float db = a[2] - b[2];
        return dr * dr + dg * dg + db * db;
    }

    static float print_visibility_strength(float value)
    {
        return std::clamp(std::pow(std::max(0.f, value), 0.85f), 0.f, 1.f);
    }

    static float safe_div(float numerator, float denominator)
    {
        return denominator <= 1e-6f ? 0.f : std::clamp(numerator / denominator, 0.f, 1.f);
    }

    std::array<float, 3> tool_color(size_t tool) const
    {
        return tool < filament_colours.size() ? parse_color(filament_colours[tool]) : std::array<float, 3>{1.f, 1.f, 1.f};
    }

    float generic_visibility(size_t tool, float r, float g, float b) const
    {
        const std::array<float, 3> target{r, g, b};
        const float distance = std::sqrt(color_distance2(tool_color(tool), target));
        return std::clamp(1.f - distance / std::sqrt(3.f), 0.f, 1.f);
    }

    static float apply_tone_gamma(float channel, float gamma)
    {
        const float safe_channel = std::clamp(channel, 0.f, 1.f);
        const float safe_gamma = (!std::isfinite(gamma) || gamma <= 0.f) ? 1.f : std::clamp(gamma, 0.5f, 3.f);
        return std::abs(safe_gamma - 1.f) <= 1e-5f ? safe_channel : std::clamp(std::pow(safe_channel, 1.f / safe_gamma), 0.f, 1.f);
    }

    std::vector<float> generic_component_weights(float r, float g, float b) const
    {
        std::vector<float> weights(component_ids.size(), 0.f);
        for (size_t idx = 0; idx < component_ids.size(); ++idx) {
            const unsigned int id = component_ids[idx];
            weights[idx] = id > 0 ? generic_visibility(size_t(id - 1), r, g, b) : 0.f;
        }
        return weights;
    }

    static std::vector<float> fixed_mode_weights(int mode, size_t component_count, float r, float g, float b)
    {
        r = std::clamp(r, 0.f, 1.f);
        g = std::clamp(g, 0.f, 1.f);
        b = std::clamp(b, 0.f, 1.f);
        const float whiteness = std::min({r, g, b});
        const float darkness = 1.f - std::max({r, g, b});
        switch (mode) {
        case int(TextureMappingZone::FilamentColorRGB):
            return component_count == 3 ?
                std::vector<float>{print_visibility_strength(r), print_visibility_strength(g), print_visibility_strength(b)} :
                std::vector<float>{};
        case int(TextureMappingZone::FilamentColorCMY):
            return component_count == 3 ?
                std::vector<float>{print_visibility_strength(1.f - r),
                                   print_visibility_strength(1.f - g),
                                   print_visibility_strength(1.f - b)} :
                std::vector<float>{};
        case int(TextureMappingZone::FilamentColorBW): {
            if (component_count != 2)
                return {};
            const float gray = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.f, 1.f);
            return {print_visibility_strength(gray >= 0.5f ? 2.f * (1.f - gray) : 1.f),
                    print_visibility_strength(gray <= 0.5f ? 2.f * gray : 1.f)};
        }
        default:
            break;
        }
        if (component_count != 4)
            return {};
        if (mode == int(TextureMappingZone::FilamentColorCMYK)) {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            return {print_visibility_strength(safe_div(1.f - r - k, inv)),
                    print_visibility_strength(safe_div(1.f - g - k, inv)),
                    print_visibility_strength(safe_div(1.f - b - k, inv)),
                    print_visibility_strength(k)};
        }
        if (mode == int(TextureMappingZone::FilamentColorCMYW)) {
            const float inv = 1.f - whiteness;
            const float r_no_w = safe_div(r - whiteness, inv);
            const float g_no_w = safe_div(g - whiteness, inv);
            const float b_no_w = safe_div(b - whiteness, inv);
            return {print_visibility_strength(std::clamp((1.f - r_no_w) * inv, 0.f, 1.f)),
                    print_visibility_strength(std::clamp((1.f - g_no_w) * inv, 0.f, 1.f)),
                    print_visibility_strength(std::clamp((1.f - b_no_w) * inv, 0.f, 1.f)),
                    std::clamp(std::pow(whiteness, 1.35f), 0.f, 1.f)};
        }
        if (mode == int(TextureMappingZone::FilamentColorRGBK)) {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            return {print_visibility_strength(safe_div(r - k, inv)),
                    print_visibility_strength(safe_div(g - k, inv)),
                    print_visibility_strength(safe_div(b - k, inv)),
                    print_visibility_strength(k)};
        }
        if (mode == int(TextureMappingZone::FilamentColorRGBW)) {
            const float inv = 1.f - whiteness;
            return {print_visibility_strength(safe_div(r - whiteness, inv)),
                    print_visibility_strength(safe_div(g - whiteness, inv)),
                    print_visibility_strength(safe_div(b - whiteness, inv)),
                    print_visibility_strength(whiteness)};
        }
        return {};
    }

    static void apply_contrast(std::vector<float> &weights, float contrast_factor, size_t mapped_count)
    {
        const size_t count = std::min(mapped_count, weights.size());
        if (count == 0)
            return;
        float mean = 0.f;
        for (size_t idx = 0; idx < count; ++idx)
            mean += std::clamp(weights[idx], 0.f, 1.f);
        mean /= float(count);
        for (size_t idx = 0; idx < count; ++idx)
            weights[idx] = std::clamp(mean + (std::clamp(weights[idx], 0.f, 1.f) - mean) * contrast_factor, 0.f, 1.f);
    }

    float adjusted_visibility_factor(unsigned int physical_id, float value) const
    {
        const size_t idx = physical_id > 0 ? size_t(physical_id - 1) : size_t(-1);
        const float strength = idx < filament_strengths_pct.size() && std::isfinite(filament_strengths_pct[idx]) ?
            std::clamp(filament_strengths_pct[idx] / 100.f, 0.f, 1.f) :
            1.f;
        const float minimum = idx < filament_minimum_offsets_pct.size() && std::isfinite(filament_minimum_offsets_pct[idx]) ?
            std::clamp(filament_minimum_offsets_pct[idx] / 100.f, 0.f, 1.f) :
            0.f;
        return std::clamp(minimum + std::clamp(value, 0.f, 1.f) * strength * (1.f - minimum), 0.f, 1.f);
    }

    float adjusted_tool_visibility(size_t tool, float value) const
    {
        const unsigned int physical_id = unsigned(tool + 1);
        return settings_zone_enabled && std::find(component_ids.begin(), component_ids.end(), physical_id) != component_ids.end() ?
                   adjusted_visibility_factor(physical_id, value) :
                   std::clamp(value, 0.f, 1.f);
    }

    float sample_settings_zone_tool_visibility(size_t tool, float u, float v, bool back) const
    {
        const unsigned int physical_id = unsigned(tool + 1);
        const auto component_it = std::find(component_ids.begin(), component_ids.end(), physical_id);
        if (component_it == component_ids.end())
            return sample_image_tool_visibility(tool, u, v, back);
        const size_t component_idx = size_t(component_it - component_ids.begin());

        std::array<float, 3> rgb = sample_image_rgb(u, v, back);
        rgb[0] = apply_tone_gamma(rgb[0], tone_gamma);
        rgb[1] = apply_tone_gamma(rgb[1], tone_gamma);
        rgb[2] = apply_tone_gamma(rgb[2], tone_gamma);

        std::vector<float> weights(component_ids.size(), 0.f);
        size_t mapped_count = component_ids.size();
        if (texture_mapping_mode == int(TextureMappingZone::TextureMappingRawValues)) {
            const float channels[3] = {rgb[0], rgb[1], rgb[2]};
            mapped_count = std::min(component_ids.size(), size_t(3));
            for (size_t idx = 0; idx < mapped_count; ++idx)
                weights[idx] = std::clamp(channels[idx], 0.f, 1.f);
        } else {
            weights = fixed_mode_weights(texture_filament_color_mode, component_ids.size(), rgb[0], rgb[1], rgb[2]);
            if (weights.size() != component_ids.size())
                weights = generic_component_weights(rgb[0], rgb[1], rgb[2]);
        }
        if (weights.size() != component_ids.size() || component_idx >= weights.size())
            return sample_image_tool_visibility(tool, u, v, back);

        apply_contrast(weights, std::clamp(contrast_pct, 25.f, 300.f) / 100.f, mapped_count);
        return std::clamp(weights[component_idx], 0.f, 1.f);
    }

    float fixed_mode_visibility(size_t tool, float r, float g, float b) const
    {
        std::vector<std::array<float, 3>> ideals;
        std::vector<float> weights;
        r = std::clamp(r, 0.f, 1.f);
        g = std::clamp(g, 0.f, 1.f);
        b = std::clamp(b, 0.f, 1.f);
        const float whiteness = std::min({r, g, b});
        const float darkness = 1.f - std::max({r, g, b});
        switch (color_mode) {
        case CMY:
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}}};
            weights = {print_visibility_strength(1.f - r),
                       print_visibility_strength(1.f - g),
                       print_visibility_strength(1.f - b)};
            break;
        case CMYK: {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}, {0.f, 0.f, 0.f}}};
            weights = {print_visibility_strength(safe_div(1.f - r - k, inv)),
                       print_visibility_strength(safe_div(1.f - g - k, inv)),
                       print_visibility_strength(safe_div(1.f - b - k, inv)),
                       print_visibility_strength(k)};
            break;
        }
        case CMYW: {
            const float inv = 1.f - whiteness;
            const float r_no_w = safe_div(r - whiteness, inv);
            const float g_no_w = safe_div(g - whiteness, inv);
            const float b_no_w = safe_div(b - whiteness, inv);
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(std::clamp((1.f - r_no_w) * inv, 0.f, 1.f)),
                       print_visibility_strength(std::clamp((1.f - g_no_w) * inv, 0.f, 1.f)),
                       print_visibility_strength(std::clamp((1.f - b_no_w) * inv, 0.f, 1.f)),
                       std::clamp(std::pow(whiteness, 1.35f), 0.f, 1.f)};
            break;
        }
        case RGB:
            ideals = {{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}};
            weights = {print_visibility_strength(r),
                       print_visibility_strength(g),
                       print_visibility_strength(b)};
            break;
        case RGBK: {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            ideals = {{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}}};
            weights = {print_visibility_strength(safe_div(r - k, inv)),
                       print_visibility_strength(safe_div(g - k, inv)),
                       print_visibility_strength(safe_div(b - k, inv)),
                       print_visibility_strength(k)};
            break;
        }
        case RGBW: {
            const float inv = 1.f - whiteness;
            ideals = {{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(safe_div(r - whiteness, inv)),
                       print_visibility_strength(safe_div(g - whiteness, inv)),
                       print_visibility_strength(safe_div(b - whiteness, inv)),
                       print_visibility_strength(whiteness)};
            break;
        }
        case BW: {
            const float gray = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.f, 1.f);
            ideals = {{{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(gray >= 0.5f ? 2.f * (1.f - gray) : 1.f),
                       print_visibility_strength(gray <= 0.5f ? 2.f * gray : 1.f)};
            break;
        }
        default:
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}, {0.f, 0.f, 0.f}}};
            weights = {print_visibility_strength(1.f - r),
                       print_visibility_strength(1.f - g),
                       print_visibility_strength(1.f - b),
                       print_visibility_strength(darkness)};
            break;
        }
        const std::array<float, 3> actual = tool_color(tool);
        size_t best = 0;
        float best_distance = std::numeric_limits<float>::max();
        for (size_t i = 0; i < ideals.size(); ++i) {
            const float distance = color_distance2(actual, ideals[i]);
            if (distance < best_distance) {
                best_distance = distance;
                best = i;
            }
        }
        if (generic_fallback_for_missing_channels && best_distance > 0.35f)
            return generic_visibility(tool, r, g, b);
        return best < weights.size() ? std::clamp(weights[best], 0.f, 1.f) : generic_visibility(tool, r, g, b);
    }
};


class WipeTower
{
public:
    friend class WipeTowerWriter;
    static const std::string never_skip_tag() { return "_GCODE_WIPE_TOWER_NEVER_SKIP_TAG"; }

	// WipeTower height to minimum depth map
	static const std::map<float, float> min_depth_per_height;
    static float get_limit_depth_by_height(float max_height);
    static float get_auto_brim_by_height(float max_height);
    static TriangleMesh                 its_make_rib_tower(float width, float depth, float height, float rib_length, float rib_width, bool fillet_wall);
    static TriangleMesh                 its_make_rib_brim(const Polygon& brim, float layer_height);
    static Polygon                      rib_section(float width, float depth, float rib_length, float rib_width, bool fillet_wall);
    static Vec2f                        move_box_inside_box(const BoundingBox &box1, const BoundingBox &box2, int offset = 0);
    static Polygon                      rounding_polygon(Polygon &polygon, double rounding = 2., double angle_tol = 30. / 180. * PI);
    struct Extrusion
    {
		Extrusion(const Vec2f &pos, float width, unsigned int tool) : pos(pos), width(width), tool(tool) {}
		// End position of this extrusion.
		Vec2f				pos;
		// Width of a squished extrusion, corrected for the roundings of the squished extrusions.
		// This is left zero if it is a travel move.
		float 			width;
		// Current extruder index.
		unsigned int    tool;
	};

	struct NozzleChangeResult
    {
        std::string gcode;

        Vec2f start_pos;  // rotated
        Vec2f end_pos;

		Vec2f origin_start_pos;  // not rotated

        std::vector<Vec2f> wipe_path;
    };

	struct ToolChangeResult
	{
		// Print heigh of this tool change.
		float					print_z;
		float 					layer_height;
		// G-code section to be directly included into the output G-code.
		std::string				gcode;
		// For path preview.
		std::vector<Extrusion> 	extrusions;
		// Initial position, at which the wipe tower starts its action.
		// At this position the extruder is loaded and there is no Z-hop applied.
		Vec2f						start_pos;
		// Last point, at which the normal G-code generator of Slic3r shall continue.
		// At this position the extruder is loaded and there is no Z-hop applied.
		Vec2f						end_pos;
		// Time elapsed over this tool change.
		// This is useful not only for the print time estimation, but also for the control of layer cooling.
		float  				    elapsed_time;

        // Is this a priming extrusion? (If so, the wipe tower rotation & translation will not be applied later)
        bool                    priming;

		bool                    is_tool_change{false};
        bool                    is_contact{false};
		Vec2f                   tool_change_start_pos;

        // Pass a polyline so that normal G-code generator can do a wipe for us.
        // The wipe cannot be done by the wipe tower because it has to pass back
        // a loaded extruder, so it would have to either do a wipe with no retraction
        // (leading to https://github.com/prusa3d/PrusaSlicer/issues/2834) or do
        // an extra retraction-unretraction pair.
        std::vector<Vec2f> wipe_path;

		// BBS
        float purge_volume = 0.f;

        // Initial tool
        int initial_tool;

        // New tool
        int new_tool;

        // BBS: in bbl filament_change_gcode, toolhead will be moved to the wipe tower automatically.
        // But if finish_layer_tcr is before tool_change_tcr, we have to travel to the wipe tower before
        // executing the gcode finish_layer_tcr.
        bool is_finish_first = false;

        NozzleChangeResult nozzle_change_result;

		// Sum the total length of the extrusion.
		float total_extrusion_length_in_plane() {
			float e_length = 0.f;
			for (size_t i = 1; i < this->extrusions.size(); ++ i) {
				const Extrusion &e = this->extrusions[i];
				if (e.width > 0) {
					Vec2f v = e.pos - (&e - 1)->pos;
					e_length += v.norm();
				}
			}
			return e_length;
		}
		bool force_travel = false;
	};

    struct box_coordinates
    {
        box_coordinates(float left, float bottom, float width, float height) :
            ld(left        , bottom         ),
            lu(left        , bottom + height),
            rd(left + width, bottom         ),
            ru(left + width, bottom + height) {}
        box_coordinates(const Vec2f &pos, float width, float height) : box_coordinates(pos(0), pos(1), width, height) {}
        void translate(const Vec2f &shift) {
            ld += shift; lu += shift;
            rd += shift; ru += shift;
        }
        void translate(const float dx, const float dy) { translate(Vec2f(dx, dy)); }
        void expand(const float offset) {
            ld += Vec2f(- offset, - offset);
            lu += Vec2f(- offset,   offset);
            rd += Vec2f(  offset, - offset);
            ru += Vec2f(  offset,   offset);
        }
        void expand(const float offset_x, const float offset_y) {
            ld += Vec2f(- offset_x, - offset_y);
            lu += Vec2f(- offset_x,   offset_y);
            rd += Vec2f(  offset_x, - offset_y);
            ru += Vec2f(  offset_x,   offset_y);
        }
        Vec2f ld;  // left down
        Vec2f lu;	// left upper
        Vec2f rd;	// right lower
        Vec2f ru;  // right upper
    };

    // Construct ToolChangeResult from current state of WipeTower and WipeTowerWriter.
    // WipeTowerWriter is moved from !
    ToolChangeResult construct_tcr(WipeTowerWriter& writer,
                                   bool priming,
                                   size_t old_tool,
                                   bool is_finish,
                                   bool is_tool_change,
                                   float purge_volume,
                                   bool is_contact = false) const;

    ToolChangeResult construct_block_tcr(WipeTowerWriter& writer,
                                   bool priming,
                                   size_t filament_id,
                                   bool is_finish,
                                   float purge_volume) const;


	// x			-- x coordinates of wipe tower in mm ( left bottom corner )
	// y			-- y coordinates of wipe tower in mm ( left bottom corner )
	// width		-- width of wipe tower in mm ( default 60 mm - leave as it is )
	// wipe_area	-- space available for one toolchange in mm
	// BBS: add partplate logic
	WipeTower(const PrintConfig& config, int plate_idx, Vec3d plate_origin, size_t initial_tool, const float wipe_tower_height, const std::vector<unsigned int>& slice_used_filaments);


	// Set the extruder properties.
    void set_extruder(size_t idx, const PrintConfig& config);

	// Appends into internal structure m_plan containing info about the future wipe tower
	// to be used before building begins. The entries must be added ordered in z.
	void plan_toolchange(float        z_par,
                         float        layer_height_par,
                         unsigned int old_tool,
                         unsigned int new_tool,
                         float        wipe_volume = 0.f,
                         float        prime_volume = 0.f,
                         bool         texture_mapping_single_component_layer = false,
                         const std::vector<unsigned int> &texture_mapping_layer_tools = {},
                         int          texture_mapping_wall_tool = -1);

	// Iterates through prepared m_plan, generates ToolChangeResults and appends them to "result"
	void generate(std::vector<std::vector<ToolChangeResult>> &result);
    void set_prime_tower_texture(const PrimeTowerTextureRenderSettings &settings) { m_prime_tower_texture = settings; }

	WipeTower::ToolChangeResult only_generate_out_wall(bool is_new_mode = false);
    Polygon generate_support_wall(WipeTowerWriter &writer, const box_coordinates &wt_box, double feedrate, bool first_layer);
    Polygon generate_support_wall_new(WipeTowerWriter &writer, const box_coordinates &wt_box, double feedrate, bool first_layer,bool rib_wall, bool extrude_perimeter, bool skip_points);

    Polygon generate_rib_polygon(const box_coordinates &wt_box);
    float get_depth() const { return m_wipe_tower_depth; }
    float get_brim_width() const { return m_wipe_tower_brim_width_real; }
    BoundingBoxf get_bbx() const {
        if (m_outer_wall.empty()) return BoundingBoxf({Vec2d(0,0)});
        BoundingBox  box = get_extents(m_outer_wall.begin()->second);
        BoundingBoxf res = BoundingBoxf(unscale(box.min), unscale(box.max));
        return res;
    }
    std::map<float, Polylines> get_outer_wall() const
    {
        return m_outer_wall;
    }
    float get_height() const { return m_wipe_tower_height; }
    float get_layer_height() const { return m_layer_height; }
    float get_rib_length() const { return m_rib_length; }
    float get_rib_width() const { return m_rib_width; }

	void set_last_layer_extruder_fill(bool extruder_fill) {
        if (!m_plan.empty()) {
			m_plan.back().extruder_fill = extruder_fill;
		}
	}

	void set_wipe_volume(std::vector<std::vector<float>>& wiping_matrix) {
		wipe_volumes = wiping_matrix;
	}

	// Switch to a next layer.
	void set_layer(
		// Print height of this layer.
		float print_z,
		// Layer height, used to calculate extrusion the rate.
		float layer_height,
		// Maximum number of tool changes on this layer or the layers below.
		size_t max_tool_changes,
		// Is this the first layer of the print? In that case print the brim first.
		bool is_first_layer,
		// Is this the last layer of the waste tower?
		bool is_last_layer)
	{
		m_z_pos 				= print_z;
		m_layer_height			= layer_height;
		m_depth_traversed  = 0.f;
        m_current_layer_finished = false;
		//m_current_shape = (! is_first_layer && m_current_shape == SHAPE_NORMAL) ? SHAPE_REVERSED : SHAPE_NORMAL;
		m_current_shape = SHAPE_NORMAL;
		if (is_first_layer) {
            m_num_layer_changes = 0;
            m_num_tool_changes 	= 0;
        } else
            ++ m_num_layer_changes;

		// Calculate extrusion flow from desired line width, nozzle diameter, filament diameter and layer_height:
		m_extrusion_flow = extrusion_flow(layer_height);

        // Advance m_layer_info iterator, making sure we got it right
		while (!m_plan.empty() && m_layer_info->z < print_z - WT_EPSILON && m_layer_info+1 != m_plan.end())
			++m_layer_info;
	}

	// Return the wipe tower position.
	const Vec2f& 		 position() const { return m_wipe_tower_pos; }
	// Return the wipe tower width.
	float     		 width()    const { return m_wipe_tower_width; }
	// The wipe tower is finished, there should be no more tool changes or wipe tower prints.
	bool 	  		 finished() const { return m_max_color_changes == 0; }

	// Returns gcode to prime the nozzles at the front edge of the print bed.
	std::vector<ToolChangeResult> prime(
		// print_z of the first layer.
		float 						initial_layer_print_height,
		// Extruder indices, in the order to be primed. The last extruder will later print the wipe tower brim, print brim and the object.
		const std::vector<unsigned int> &tools,
		// If true, the last priming are will be the same as the other priming areas, and the rest of the wipe will be performed inside the wipe tower.
		// If false, the last priming are will be large enough to wipe the last extruder sufficiently.
		bool 						last_wipe_inside_wipe_tower);

	// Returns gcode for a toolchange and a final print head position.
	// On the first layer, extrude a brim around the future wipe tower first.
	// BBS
    ToolChangeResult tool_change(size_t new_tool, bool extrude_perimeter = false, bool first_toolchange_to_nonsoluble = false);

	NozzleChangeResult nozzle_change(int old_filament_id, int new_filament_id);

	// Fill the unfilled space with a sparse infill.
	// Call this method only if layer_finished() is false.
    ToolChangeResult finish_layer(bool extruder_perimeter = true, bool extruder_fill = true);

	// Calculates extrusion flow needed to produce required line width for given layer height
    float extrusion_flow(float layer_height = -1.f) const // negative layer_height - return current m_extrusion_flow
    {
        if (layer_height < 0) return m_extrusion_flow;
        return layer_height * (m_perimeter_width - layer_height * (1.f - float(M_PI) / 4.f)) / filament_area();
    }
    float nozzle_change_extrusion_flow(float layer_height = -1.f) const // negative layer_height - return current m_extrusion_flow
    {
        if (layer_height < 0)
            return m_extrusion_flow;
        return layer_height * (m_nozzle_change_perimeter_width - layer_height * (1.f - float(M_PI) / 4.f)) / filament_area();
    }

	bool get_floating_area(float& start_pos_y, float& end_pos_y) const;
	bool need_thick_bridge_flow(float pos_y) const;
    float get_extrusion_flow() const { return m_extrusion_flow; }

	// Is the current layer finished?
	bool 			 layer_finished() const {
        return m_current_layer_finished;
	}

    std::vector<float> get_used_filament() const { return m_used_filament_length; }
    int get_number_of_toolchanges() const { return m_num_tool_changes; }

	void set_filament_map(const std::vector<int> &filament_map) { m_filament_map = filament_map; }

	void set_has_tpu_filament(bool has_tpu) { m_has_tpu_filament = has_tpu; }
    bool has_tpu_filament() const { return m_has_tpu_filament; }

    struct FilamentParameters {
        std::string 	    material = "PLA";
        int                 category;
        bool                is_soluble = false;
        // BBS
        bool                is_support = false;
        int  			    nozzle_temperature = 0;
        int  			    nozzle_temperature_initial_layer = 0;
        int                 interface_print_temperature = 0;
        float               loading_speed = 0.f;
        float               loading_speed_start = 0.f;
        float               unloading_speed = 0.f;
        float               unloading_speed_start = 0.f;
        float               delay = 0.f ;
        int                 cooling_moves = 0;
        float               cooling_initial_speed = 0.f;
        float               cooling_final_speed = 0.f;
        float               ramming_line_width_multiplicator = 1.f;
        float               ramming_step_multiplicator = 1.f;
        float               max_e_speed = std::numeric_limits<float>::max();
        std::vector<float>  ramming_speed;
        float               nozzle_diameter;
        float               filament_area;
        float               retract_length;
        float               retract_speed;
        float               wipe_dist;
        float               tower_interface_pre_extrusion_dist = 0.f;
        float               tower_interface_pre_extrusion_length = 0.f;
        float               tower_ironing_area = 4.f;
        float               tower_interface_purge_length = 0.f;
    };


	void set_used_filament_ids(const std::vector<int> &used_filament_ids) { m_used_filament_ids = used_filament_ids; };
    void set_filament_categories(const std::vector<int> & filament_categories) { m_filament_categories = filament_categories;};
	std::vector<int> m_used_filament_ids;
    std::vector<int> m_filament_categories;

	struct WipeTowerBlock
    {
        int              block_id{0};
        int              filament_adhesiveness_category{0};
        std::vector<float>      layer_depths;
		std::vector<bool>       solid_infill;
        std::vector<float>      finish_depth{0}; // the start pos of finish frame for every layer
        float            depth{0};
        float            start_depth{0};
        float            cur_depth{0};
		int              last_filament_change_id{-1};
        int              last_nozzle_change_id{-1};
	};

	struct BlockDepthInfo
    {
        int category{-1};
        float depth{0};
        float nozzle_change_depth{0};
	};

	std::vector<std::vector<BlockDepthInfo>> m_all_layers_depth;
	std::vector<WipeTowerBlock> m_wipe_tower_blocks;
    int                  m_last_block_id;
    WipeTowerBlock*      m_cur_block{nullptr};

	// help function
    WipeTowerBlock* get_block_by_category(int filament_adhesiveness_category, bool create);
    void add_depth_to_block(int filament_id, int filament_adhesiveness_category, float depth, bool is_nozzle_change = false);
	int get_filament_category(int filament_id);
	bool is_in_same_extruder(int filament_id_1, int filament_id_2);
	void reset_block_status();
    int get_wall_filament_for_all_layer();
	// for generate new wipe tower
    void generate_new(std::vector<std::vector<WipeTower::ToolChangeResult>> &result);

	void plan_tower_new();
	void generate_wipe_tower_blocks();
    void update_all_layer_depth(float wipe_tower_depth);

    ToolChangeResult   tool_change_new(size_t new_tool, bool solid_change = false, bool solid_nozzlechange=false);
    NozzleChangeResult nozzle_change_new(int old_filament_id, int new_filament_id, bool solid_change = false);
    ToolChangeResult   finish_layer_new(bool extrude_perimeter = true, bool extrude_fill = true, bool extrude_fill_wall = true);
    ToolChangeResult   finish_block(const WipeTowerBlock &block, int filament_id, bool extrude_fill = true);
    ToolChangeResult   finish_block_solid(const WipeTowerBlock &block, int filament_id, bool extrude_fill = true ,bool interface_solid =false);
    void toolchange_wipe_new(WipeTowerWriter &writer, const box_coordinates &cleaning_box, float wipe_length,bool solid_toolchange=false);
    Vec2f              get_rib_offset() const { return m_rib_offset; }

private:
	enum wipe_shape // A fill-in direction
	{
		SHAPE_NORMAL = 1,
		SHAPE_REVERSED = -1
	};

    const float Width_To_Nozzle_Ratio = 1.25f; // desired line width (oval) in multiples of nozzle diameter - may not be actually neccessary to adjust
    const float WT_EPSILON            = 1e-3f;
    float filament_area() const {
        return m_filpar[0].filament_area; // all extruders are assumed to have the same filament diameter at this point
    }

    int    m_slice_used_filaments      = 0;
    int    m_wrapping_detection_layers = 0;
    bool   m_enable_wrapping_detection = false;
	bool   m_enable_timelapse_print = false;
	bool   m_semm               = true; // Are we using a single extruder multimaterial printer?
	bool   m_purge_in_prime_tower = false; // Do we purge in the prime tower?
    Vec2f  m_wipe_tower_pos; 			// Left front corner of the wipe tower in mm.
	float  m_wipe_tower_width; 			// Width of the wipe tower.
	float  m_wipe_tower_depth 	= 0.f; 	// Depth of the wipe tower
	// BBS
	float  m_wipe_tower_height = 0.f;
    float  m_wipe_tower_brim_width      = 0.f; 	// Width of brim (mm) from config
    float  m_wipe_tower_brim_width_real = 0.f; 	// Width of brim (mm) after generation
	float  m_wipe_tower_rotation_angle = 0.f; // Wipe tower rotation angle in degrees (with respect to x axis)
    float  m_internal_rotation  = 0.f;
	float  m_y_shift			= 0.f;  // y shift passed to writer
	float  m_z_pos 				= 0.f;  // Current Z position.
	float  m_layer_height 		= 0.f; 	// Current layer height.
	size_t m_max_color_changes 	= 0; 	// Maximum number of color changes per layer.
    int    m_old_temperature    = -1;   // To keep track of what was the last temp that we set (so we don't issue the command when not neccessary)
    float  m_travel_speed       = 0.f;
    float  m_first_layer_speed  = 0.f;
    size_t m_first_layer_idx    = size_t(-1);

    std::vector<double> m_filaments_change_length;
    size_t       m_cur_layer_id;
    NozzleChangeResult m_nozzle_change_result;
    std::vector<int>   m_filament_map;
    bool               m_has_tpu_filament{false};
    bool               m_is_multi_extruder{false};
    bool               m_use_gap_wall{false};
    bool               m_use_rib_wall{false};
    float              m_rib_length=0.f;
    float              m_rib_width=0.f;
    float              m_extra_rib_length=0.f;
    bool               m_used_fillet{false};
    Vec2f              m_rib_offset{Vec2f(0.f, 0.f)};
    bool               m_tower_framework{false};

	// G-code generator parameters.
    float           m_cooling_tube_retraction   = 0.f;
    float           m_cooling_tube_length       = 0.f;
    float           m_parking_pos_retraction    = 0.f;
    float           m_extra_loading_move        = 0.f;
    float           m_bridging                  = 0.f;
    bool            m_no_sparse_layers          = false;
    bool            m_set_extruder_trimpot      = false;
    bool            m_adhesion                  = true;
    GCodeFlavor     m_gcode_flavor;

    // Bed properties
    enum {
        RectangularBed,
        CircularBed,
        CustomBed
    } m_bed_shape;
    float m_bed_width; // width of the bed bounding box
    Vec2f m_bed_bottom_left; // bottom-left corner coordinates (for rectangular beds)

	float m_perimeter_width = 0.4f * Width_To_Nozzle_Ratio; // Width of an extrusion line, also a perimeter spacing for 100% infill.
    float m_nozzle_change_perimeter_width = 0.4f * Width_To_Nozzle_Ratio;
	float m_extrusion_flow = 0.038f; //0.029f;// Extrusion flow is derived from m_perimeter_width, layer height and filament diameter.

	// Extruder specific parameters.
    std::vector<FilamentParameters> m_filpar;


	// State of the wipe tower generator.
	unsigned int m_num_layer_changes = 0; // Layer change counter for the output statistics.
	unsigned int m_num_tool_changes  = 0; // Tool change change counter for the output statistics.
	///unsigned int 	m_idx_tool_change_in_layer = 0; // Layer change counter in this layer. Counting up to m_max_color_changes.
	bool m_print_brim = true;
	// A fill-in direction (positive Y, negative Y) alternates with each layer.
	wipe_shape   	m_current_shape = SHAPE_NORMAL;
    size_t 	m_current_tool  = 0;
	// Orca: support mmu wipe tower
    std::vector<std::vector<float>> wipe_volumes;

	float           m_depth_traversed = 0.f; // Current y position at the wipe tower.
    bool            m_current_layer_finished = false;
	bool 			m_left_to_right   = true;
	float			m_extra_spacing   = 1.f;
	float           m_tpu_fixed_spacing = 2;
    std::vector<Vec2f> m_wall_skip_points;
    std::map<float,Polylines> m_outer_wall;
    bool is_first_layer() const { return size_t(m_layer_info - m_plan.begin()) == m_first_layer_idx; }
    bool                       m_flat_ironing=false;
    bool                       m_enable_tower_interface_features=false;
    bool                       m_enable_tower_interface_cooldown_during_tower=false;
    bool                       m_prev_layer_had_interface=false;
    bool                       m_current_layer_has_interface=false;
	// Calculates length of extrusion line to extrude given volume
	float volume_to_length(float volume, float line_width, float layer_height) const {
		return std::max(0.f, volume / (layer_height * (line_width - layer_height * (1.f - float(M_PI) / 4.f))));
	}

	// Calculates depth for all layers and propagates them downwards
	void plan_tower();

	// Goes through m_plan and recalculates depths and width of the WT to make it exactly square - experimental
	void make_wipe_tower_square();

	Vec2f get_next_pos(const WipeTower::box_coordinates &cleaning_box, float wipe_length, bool interface_layer, size_t interface_tool);

    // Goes through m_plan, calculates border and finish_layer extrusions and subtracts them from last wipe
    void save_on_last_wipe();

	bool is_tpu_filament(int filament_id) const;

	// BBS
	box_coordinates align_perimeter(const box_coordinates& perimeter_box);


    // to store information about tool changes for a given layer
	struct WipeTowerInfo{
		struct ToolChange {
            size_t old_tool;
            size_t new_tool;
			float required_depth;
            float ramming_depth;
            float first_wipe_line;
            float wipe_volume;
			float wipe_length;
            float nozzle_change_depth{0};
			// BBS
			float purge_volume;
            ToolChange(size_t old, size_t newtool, float depth=0.f, float ramming_depth=0.f, float fwl=0.f, float wv=0.f, float wl = 0, float pv = 0)
				: old_tool{ old }, new_tool{ newtool }, required_depth{ depth }, ramming_depth{ ramming_depth }, first_wipe_line{ fwl }, wipe_volume{ wv }, wipe_length{ wl }, purge_volume{ pv } {}
		};
		float z;		// z position of the layer
		float height;	// layer height
		float depth;	// depth of the layer based on all layers above
        float extra_spacing;
        bool  extruder_fill{true};
        bool  texture_mapping_single_component_layer{false};
        std::vector<size_t> texture_mapping_layer_tools;
        size_t texture_mapping_wall_tool{size_t(-1)};
		float toolchanges_depth() const { float sum = 0.f; for (const auto &a : tool_changes) sum += a.required_depth; return sum; }

		std::vector<ToolChange> tool_changes;

		WipeTowerInfo(float z_par, float layer_height_par)
			: z{z_par}, height{layer_height_par}, depth{0}, extra_spacing{1.f} {}
	};

	std::vector<WipeTowerInfo> m_plan; 	// Stores information about all layers and toolchanges for the future wipe tower (filled by plan_toolchange(...))
	std::vector<WipeTowerInfo>::iterator m_layer_info = m_plan.end();

    // Stores information about used filament length per extruder:
    std::vector<float> m_used_filament_length;
    PrimeTowerTextureRenderSettings m_prime_tower_texture;

    // BBS: consider both soluable and support properties
    // Return index of first toolchange that switches to non-soluble extruder
    // ot -1 if there is no such toolchange.
    int first_toolchange_to_nonsoluble_nonsupport(
            const std::vector<WipeTowerInfo::ToolChange>& tool_changes) const;

	void toolchange_Unload(
		WipeTowerWriter &writer,
		const box_coordinates  &cleaning_box,
		const std::string&	 	current_material,
		const int 				new_temperature);

	void toolchange_Change(
		WipeTowerWriter &writer,
        const size_t		new_tool,
		const std::string& 		new_material);

	void toolchange_Load(
		WipeTowerWriter &writer,
		const box_coordinates  &cleaning_box);

	void toolchange_Wipe(
		WipeTowerWriter &writer,
		const box_coordinates  &cleaning_box,
		float wipe_volume);
    void get_wall_skip_points(const WipeTowerInfo &layer);
};




} // namespace Slic3r

#endif // WipeTowerPrusaMM_hpp_
