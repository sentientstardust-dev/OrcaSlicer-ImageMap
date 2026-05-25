// original author: sentientstardust

#include "TextureMapping.hpp"
#include "ColorSolver.hpp"

#include <array>
#include <boost/log/trivial.hpp>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace Slic3r {

namespace {

constexpr unsigned int TextureMappingZoneIdBase = 99;
constexpr unsigned int MaxTextureMappingZoneId = 255;
constexpr const char *TextureMappingGapDisplayColor = "#8C8C8C";
constexpr float TextureMappingPoorColorMatchDistance = 0.22f;

struct RGB {
    int r = 0;
    int g = 0;
    int b = 0;
};

static int clamp_int(int value, int lo, int hi)
{
    return std::max(lo, std::min(hi, value));
}

static float finite_or(float value, float fallback)
{
    return std::isfinite(value) ? value : fallback;
}

static RGB parse_hex_color(const std::string &hex)
{
    RGB c;
    if (hex.size() >= 7 && hex[0] == '#') {
        try {
            c.r = std::stoi(hex.substr(1, 2), nullptr, 16);
            c.g = std::stoi(hex.substr(3, 2), nullptr, 16);
            c.b = std::stoi(hex.substr(5, 2), nullptr, 16);
        } catch (...) {
            c = {};
        }
    }
    return c;
}

static std::string rgb_to_hex(const RGB &c)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%02X%02X%02X",
                  std::clamp(c.r, 0, 255),
                  std::clamp(c.g, 0, 255),
                  std::clamp(c.b, 0, 255));
    return std::string(buf);
}

static std::string random_display_color(uint64_t stable_id)
{
    uint64_t x = stable_id == 0 ? 0x6A09E667F3BCC909ull : stable_id;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;

    const int r = 80 + int((x >> 0) & 0x7f);
    const int g = 80 + int((x >> 8) & 0x7f);
    const int b = 80 + int((x >> 16) & 0x7f);
    return rgb_to_hex({r, g, b});
}

static std::vector<unsigned int> decode_component_ids(const std::string &encoded, size_t num_physical)
{
    std::vector<unsigned int> ids;
    bool seen[10] = { false };
    const unsigned int max_id = unsigned(std::min<size_t>(num_physical, 9));
    for (char c : encoded) {
        if (c < '1' || c > '9')
            continue;
        const unsigned int id = unsigned(c - '0');
        if (id > max_id || seen[id])
            continue;
        seen[id] = true;
        ids.emplace_back(id);
    }
    return ids;
}

static std::string encode_component_ids(const std::vector<unsigned int> &ids)
{
    std::string encoded;
    bool seen[10] = { false };
    for (const unsigned int id : ids) {
        if (id == 0 || id > 9 || seen[id])
            continue;
        seen[id] = true;
        encoded.push_back(char('0' + id));
    }
    return encoded;
}

static std::vector<int> parse_int_tokens(const std::string &value)
{
    std::vector<int> out;
    std::string current;
    for (const char c : value) {
        if (std::isdigit(static_cast<unsigned char>(c)) || (c == '-' && current.empty())) {
            current.push_back(c);
            continue;
        }
        if (!current.empty() && current != "-") {
            try {
                out.emplace_back(std::stoi(current));
            } catch (...) {
            }
        }
        current.clear();
    }
    if (!current.empty() && current != "-") {
        try {
            out.emplace_back(std::stoi(current));
        } catch (...) {
        }
    }
    return out;
}

static std::vector<float> parse_float_tokens(const std::string &value)
{
    std::vector<float> out;
    std::string current;
    bool has_digit = false;
    for (const char c : value) {
        const bool token_char = std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E';
        if (token_char) {
            current.push_back(c);
            if (std::isdigit(static_cast<unsigned char>(c)))
                has_digit = true;
            continue;
        }
        if (has_digit) {
            try {
                out.emplace_back(std::stof(current));
            } catch (...) {
            }
        }
        current.clear();
        has_digit = false;
    }
    if (has_digit) {
        try {
            out.emplace_back(std::stof(current));
        } catch (...) {
        }
    }
    return out;
}

static std::string format_float_token(float value)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4) << value;
    std::string out = ss.str();
    while (!out.empty() && out.back() == '0')
        out.pop_back();
    if (!out.empty() && out.back() == '.')
        out.pop_back();
    return out.empty() ? "0" : out;
}

static std::string normalize_weights(const std::string &weights, size_t expected_count)
{
    if (expected_count == 0)
        return std::string();

    std::vector<int> parsed = parse_int_tokens(weights);
    if (parsed.size() != expected_count)
        return std::string();

    int total = 0;
    for (int &weight : parsed) {
        weight = std::max(0, weight);
        total += weight;
    }
    if (total <= 0)
        return std::string();

    std::ostringstream ss;
    for (size_t i = 0; i < parsed.size(); ++i) {
        if (i > 0)
            ss << '/';
        ss << parsed[i];
    }
    return ss.str();
}

static std::vector<int> decode_weights(const std::string &weights, size_t expected_count)
{
    std::vector<int> parsed = parse_int_tokens(weights);
    if (parsed.size() != expected_count)
        return {};
    int total = 0;
    for (int &weight : parsed) {
        weight = std::max(0, weight);
        total += weight;
    }
    return total > 0 ? parsed : std::vector<int>();
}

static int safe_mod(int value, int divisor)
{
    if (divisor <= 0)
        return 0;
    int out = value % divisor;
    if (out < 0)
        out += divisor;
    return out;
}

static std::vector<unsigned int> build_balanced_component_sequence(const std::vector<unsigned int> &ids,
                                                                   const std::vector<int>          &weights)
{
    if (ids.empty())
        return {};

    std::vector<int> counts;
    counts.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int weight = i < weights.size() ? std::max(0, weights[i]) : 1;
        counts.emplace_back(weight);
    }
    if (std::all_of(counts.begin(), counts.end(), [](int v) { return v <= 0; }))
        counts.assign(ids.size(), 1);

    int total = std::accumulate(counts.begin(), counts.end(), 0);
    constexpr int MaxCycle = 64;
    if (total > MaxCycle) {
        const double scale = double(MaxCycle) / double(total);
        for (int &count : counts)
            count = count <= 0 ? 0 : std::max(1, int(std::lround(double(count) * scale)));
        total = std::accumulate(counts.begin(), counts.end(), 0);
    }
    if (total <= 0)
        return {};

    std::vector<unsigned int> sequence;
    sequence.reserve(size_t(total));
    std::vector<int> debt(ids.size(), 0);
    for (int step = 0; step < total; ++step) {
        size_t best_idx = 0;
        int best_debt = std::numeric_limits<int>::lowest();
        for (size_t idx = 0; idx < counts.size(); ++idx) {
            debt[idx] += counts[idx];
            if (debt[idx] > best_debt) {
                best_debt = debt[idx];
                best_idx = idx;
            }
        }
        sequence.emplace_back(ids[best_idx]);
        debt[best_idx] -= total;
    }
    return sequence;
}

static std::string normalize_offset_distances(const std::string &distances, size_t expected_count, float max_distance_mm)
{
    if (expected_count == 0)
        return std::string();
    std::vector<float> parsed = parse_float_tokens(distances);
    if (parsed.size() != expected_count)
        return std::string();

    std::ostringstream ss;
    for (size_t i = 0; i < parsed.size(); ++i) {
        if (i > 0)
            ss << '/';
        ss << format_float_token(std::clamp(finite_or(parsed[i], 0.f), 0.f, max_distance_mm));
    }
    return ss.str();
}

static std::string normalize_offset_angles(const std::string &angles, size_t expected_count)
{
    if (expected_count == 0)
        return std::string();
    std::vector<float> parsed = parse_float_tokens(angles);
    if (parsed.size() != expected_count)
        return std::string();

    std::ostringstream ss;
    for (size_t i = 0; i < parsed.size(); ++i) {
        float angle = std::fmod(finite_or(parsed[i], 0.f), 360.f);
        if (angle < 0.f)
            angle += 360.f;
        if (i > 0)
            ss << '/';
        ss << format_float_token(angle);
    }
    return ss.str();
}

static std::vector<float> decode_offset_distances(const std::string &distances, size_t expected_count, float max_distance_mm)
{
    std::vector<float> parsed = parse_float_tokens(distances);
    if (parsed.size() != expected_count)
        return {};
    for (float &value : parsed)
        value = std::clamp(finite_or(value, 0.f), 0.f, max_distance_mm);
    return parsed;
}

static std::vector<float> decode_offset_angles(const std::string &angles, size_t expected_count)
{
    std::vector<float> parsed = parse_float_tokens(angles);
    if (parsed.size() != expected_count)
        return {};
    for (float &value : parsed) {
        value = std::fmod(finite_or(value, 0.f), 360.f);
        if (value < 0.f)
            value += 360.f;
    }
    return parsed;
}

static std::vector<float> normalize_strengths(const std::vector<float> &values)
{
    std::vector<float> out;
    out.reserve(std::min<size_t>(values.size(), 9));
    for (size_t i = 0; i < std::min<size_t>(values.size(), 9); ++i)
        out.emplace_back(std::clamp(finite_or(values[i], 100.f), 0.f, 100.f));
    while (!out.empty() && std::abs(out.back() - 100.f) <= 1e-6f)
        out.pop_back();
    return out;
}

static std::vector<float> normalize_minimum_offsets(const std::vector<float> &values)
{
    std::vector<float> out;
    out.reserve(std::min<size_t>(values.size(), 9));
    for (size_t i = 0; i < std::min<size_t>(values.size(), 9); ++i)
        out.emplace_back(std::clamp(finite_or(values[i], 0.f), 0.f, 100.f));
    while (!out.empty() && std::abs(out.back()) <= 1e-6f)
        out.pop_back();
    return out;
}

static std::vector<float> normalize_transmission_distances(const std::vector<float> &values)
{
    std::vector<float> out;
    out.reserve(std::min<size_t>(values.size(), 9));
    for (size_t i = 0; i < std::min<size_t>(values.size(), 9); ++i) {
        const float value = finite_or(values[i], 0.f);
        out.emplace_back(value > 0.f ? std::clamp(value, 0.01f, 50.f) : 0.f);
    }
    while (!out.empty() && std::abs(out.back()) <= 1e-6f)
        out.pop_back();
    return out;
}

static float normalize_tone_gamma(float value)
{
    return (!std::isfinite(value) || value <= 0.f) ? 1.f : std::clamp(value, 0.5f, 3.f);
}

static RGB filament_color(unsigned int id, const std::vector<std::string> &filament_colours)
{
    if (id >= 1 && size_t(id - 1) < filament_colours.size())
        return parse_hex_color(filament_colours[size_t(id - 1)]);
    return {};
}

static std::array<float, 3> rgb_to_srgb01(const RGB &color)
{
    return {
        float(std::clamp(color.r, 0, 255)) / 255.f,
        float(std::clamp(color.g, 0, 255)) / 255.f,
        float(std::clamp(color.b, 0, 255)) / 255.f
    };
}

static float perceptual_color_distance_sq(const RGB &lhs, const RGB &rhs)
{
    const std::array<float, 3> lhs_oklab = color_solver_oklab_from_srgb(rgb_to_srgb01(lhs));
    const std::array<float, 3> rhs_oklab = color_solver_oklab_from_srgb(rgb_to_srgb01(rhs));
    const float dr = lhs_oklab[0] - rhs_oklab[0];
    const float dg = lhs_oklab[1] - rhs_oklab[1];
    const float db = lhs_oklab[2] - rhs_oklab[2];
    return dr * dr + dg * dg + db * db;
}

static std::vector<RGB> semantic_colors(int filament_color_mode)
{
    switch (clamp_int(filament_color_mode,
                      int(TextureMappingZone::FilamentColorAny),
                      int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorRGB):  return {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
    case int(TextureMappingZone::FilamentColorCMY):  return {{0, 255, 255}, {255, 0, 255}, {255, 255, 0}};
    case int(TextureMappingZone::FilamentColorCMYK): return {{0, 255, 255}, {255, 0, 255}, {255, 255, 0}, {0, 0, 0}};
    case int(TextureMappingZone::FilamentColorCMYW): return {{0, 255, 255}, {255, 0, 255}, {255, 255, 0}, {255, 255, 255}};
    case int(TextureMappingZone::FilamentColorRGBK): return {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {0, 0, 0}};
    case int(TextureMappingZone::FilamentColorRGBW): return {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}};
    case int(TextureMappingZone::FilamentColorBW):   return {{0, 0, 0}, {255, 255, 255}};
    case int(TextureMappingZone::FilamentColorCMYKW): return {{0, 255, 255}, {255, 0, 255}, {255, 255, 0}, {0, 0, 0}, {255, 255, 255}};
    case int(TextureMappingZone::FilamentColorRGBKW): return {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {0, 0, 0}, {255, 255, 255}};
    default:                                         return {};
    }
}

static std::vector<std::string> semantic_color_names(int filament_color_mode)
{
    switch (clamp_int(filament_color_mode,
                      int(TextureMappingZone::FilamentColorAny),
                      int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorRGB):  return {"Red", "Green", "Blue"};
    case int(TextureMappingZone::FilamentColorCMY):  return {"Cyan", "Magenta", "Yellow"};
    case int(TextureMappingZone::FilamentColorCMYK): return {"Cyan", "Magenta", "Yellow", "Black"};
    case int(TextureMappingZone::FilamentColorCMYW): return {"Cyan", "Magenta", "Yellow", "White"};
    case int(TextureMappingZone::FilamentColorRGBK): return {"Red", "Green", "Blue", "Black"};
    case int(TextureMappingZone::FilamentColorRGBW): return {"Red", "Green", "Blue", "White"};
    case int(TextureMappingZone::FilamentColorBW):   return {"Black", "White"};
    case int(TextureMappingZone::FilamentColorCMYKW): return {"Cyan", "Magenta", "Yellow", "Black", "White"};
    case int(TextureMappingZone::FilamentColorRGBKW): return {"Red", "Green", "Blue", "Black", "White"};
    default:                                         return {};
    }
}

static std::vector<unsigned int> ids_from_json(const nlohmann::json &value, size_t num_physical)
{
    std::vector<unsigned int> ids;
    if (!value.is_array())
        return ids;
    bool seen[10] = { false };
    const unsigned int max_id = unsigned(std::min<size_t>(num_physical, 9));
    for (const nlohmann::json &item : value) {
        if (!item.is_number_integer() && !item.is_number_unsigned())
            continue;
        const int raw = item.get<int>();
        if (raw < 1 || raw > int(max_id) || seen[raw])
            continue;
        seen[raw] = true;
        ids.emplace_back(unsigned(raw));
    }
    return ids;
}

static nlohmann::json ids_to_json(const std::vector<unsigned int> &ids)
{
    nlohmann::json out = nlohmann::json::array();
    for (const unsigned int id : ids)
        out.push_back(id);
    return out;
}

static std::vector<float> floats_from_json(const nlohmann::json &value)
{
    std::vector<float> out;
    if (!value.is_array())
        return out;
    out.reserve(value.size());
    for (const nlohmann::json &item : value)
        if (item.is_number())
            out.emplace_back(item.get<float>());
    return out;
}

static nlohmann::json floats_to_json(const std::vector<float> &values)
{
    nlohmann::json out = nlohmann::json::array();
    for (const float value : values)
        out.push_back(value);
    return out;
}

static nlohmann::json point3_to_json(const std::array<float, 3> &point)
{
    return nlohmann::json::array({ point[0], point[1], point[2] });
}

static std::array<float, 3> point3_from_json(const nlohmann::json &value)
{
    std::array<float, 3> out { { 0.f, 0.f, 0.f } };
    if (!value.is_array())
        return out;
    for (size_t i = 0; i < 3 && i < value.size(); ++i)
        if (value[i].is_number())
            out[i] = value[i].get<float>();
    return out;
}

static nlohmann::json linear_gradient_anchor_to_json(const TextureMappingZone::LinearGradientAnchor &anchor)
{
    nlohmann::json out;
    out["valid"] = anchor.valid;
    out["object_id"] = anchor.object_id;
    out["instance_id"] = anchor.instance_id;
    out["object_backup_id"] = anchor.object_backup_id;
    out["object_index_valid"] = anchor.object_index_valid;
    out["object_index"] = anchor.object_index;
    out["instance_index_valid"] = anchor.instance_index_valid;
    out["instance_index"] = anchor.instance_index;
    out["instance_loaded_id"] = anchor.instance_loaded_id;
    out["local_point"] = point3_to_json(anchor.local_point);
    out["global_point"] = point3_to_json(anchor.global_point);
    return out;
}

static TextureMappingZone::LinearGradientAnchor linear_gradient_anchor_from_json(const nlohmann::json &value)
{
    TextureMappingZone::LinearGradientAnchor out;
    if (!value.is_object())
        return out;
    out.valid = value.value("valid", false);
    out.object_id = value.value("object_id", size_t(0));
    out.instance_id = value.value("instance_id", size_t(0));
    out.object_backup_id = value.value("object_backup_id", -1);
    out.object_index_valid = value.value("object_index_valid", false);
    out.object_index = value.value("object_index", size_t(0));
    out.instance_index_valid = value.value("instance_index_valid", false);
    out.instance_index = value.value("instance_index", size_t(0));
    out.instance_loaded_id = value.value("instance_loaded_id", size_t(0));
    out.local_point = point3_from_json(value.value("local_point", nlohmann::json::array()));
    out.global_point = point3_from_json(value.value("global_point", nlohmann::json::array()));
    return out;
}

static nlohmann::json linear_gradient_stop_to_json(const TextureMappingZone::LinearGradientStop &stop)
{
    nlohmann::json out;
    out["position"] = std::clamp(std::isfinite(stop.position) ? stop.position : 0.f, 0.f, 1.f);
    out["filament_id"] = stop.filament_id;
    return out;
}

static TextureMappingZone::LinearGradientStop linear_gradient_stop_from_json(const nlohmann::json &value)
{
    TextureMappingZone::LinearGradientStop out;
    if (!value.is_object())
        return out;
    const float position = value.value("position", 0.f);
    out.position = std::clamp(std::isfinite(position) ? position : 0.f, 0.f, 1.f);
    out.filament_id = value.value("filament_id", 1u);
    return out;
}

static nlohmann::json linear_gradient_stops_to_json(const std::vector<TextureMappingZone::LinearGradientStop> &stops)
{
    nlohmann::json out = nlohmann::json::array();
    for (const TextureMappingZone::LinearGradientStop &stop : stops)
        out.push_back(linear_gradient_stop_to_json(stop));
    return out;
}

static std::vector<TextureMappingZone::LinearGradientStop> linear_gradient_stops_from_json(const nlohmann::json &value)
{
    std::vector<TextureMappingZone::LinearGradientStop> out;
    if (!value.is_array())
        return out;
    out.reserve(value.size());
    for (const nlohmann::json &item : value)
        if (item.is_object())
            out.emplace_back(linear_gradient_stop_from_json(item));
    return out;
}

static std::string surface_pattern_name(int surface_pattern)
{
    if (surface_pattern == int(TextureMappingZone::Gradient2D))
        return "2d_gradient";
    if (surface_pattern == int(TextureMappingZone::LinearGradient))
        return "linear_gradient";
    return "image_texture";
}

static int surface_pattern_from_name(const std::string &name)
{
    if (name == "2d_gradient" || name == "surface_gradient")
        return int(TextureMappingZone::Gradient2D);
    if (name == "linear_gradient")
        return int(TextureMappingZone::LinearGradient);
    return int(TextureMappingZone::ImageTexture);
}

static std::string linear_gradient_mode_name(int mode)
{
    return mode == int(TextureMappingZone::LinearGradientRadial) ? "radial" : "linear";
}

static int linear_gradient_mode_from_name(const std::string &name)
{
    return name == "radial" ? int(TextureMappingZone::LinearGradientRadial) : int(TextureMappingZone::LinearGradientLinear);
}

static nlohmann::json weights_to_json(const std::string &weights, size_t expected_count)
{
    nlohmann::json out = nlohmann::json::array();
    const std::vector<int> parsed = decode_weights(weights, expected_count);
    for (const int weight : parsed)
        out.push_back(weight);
    return out;
}

static std::string weights_from_json(const nlohmann::json &value, size_t expected_count)
{
    if (!value.is_array() || expected_count == 0)
        return std::string();
    std::vector<int> values;
    values.reserve(value.size());
    for (const nlohmann::json &item : value)
        if (item.is_number_integer() || item.is_number_unsigned())
            values.emplace_back(std::max(0, item.get<int>()));

    std::ostringstream ss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0)
            ss << '/';
        ss << values[i];
    }
    return normalize_weights(ss.str(), expected_count);
}

static std::string mapping_mode_name(int mode)
{
    return mode == int(TextureMappingZone::TextureMappingRawValues) ?
        std::string("raw_channel_offsets") :
        std::string("target_color");
}

static int mapping_mode_from_name(const std::string &name)
{
    return name == "raw_channel_offsets" ?
        int(TextureMappingZone::TextureMappingRawValues) :
        int(TextureMappingZone::TextureMappingFilamentBlending);
}

static std::string modulation_mode_name(int mode)
{
    switch (clamp_int(mode,
                      int(TextureMappingZone::ModulationLineWidth),
                      int(TextureMappingZone::ModulationPerimeterPathV2))) {
    case int(TextureMappingZone::ModulationPerimeterPath):
        return std::string("perimeter_path");
    case int(TextureMappingZone::ModulationPerimeterPathV2):
        return std::string("perimeter_path_v2");
    default:
        return std::string("line_width");
    }
}

static int modulation_mode_from_name(const std::string &name)
{
    if (name == "perimeter_path")
        return int(TextureMappingZone::ModulationPerimeterPath);
    if (name == "perimeter_path_v2")
        return int(TextureMappingZone::ModulationPerimeterPathV2);
    return int(TextureMappingZone::ModulationLineWidth);
}

static std::string top_surface_image_printing_method_name(int method)
{
    if (method == int(TextureMappingZone::TopSurfaceImageContoning))
        return std::string("contoning");
    if (method == int(TextureMappingZone::TopSurfaceImageSameLayer45Partition))
        return std::string("same_layer_45_partition");
    return std::string("same_angle_45_width");
}

static int top_surface_image_printing_method_from_name(const std::string &name)
{
    if (name == "contoning")
        return int(TextureMappingZone::TopSurfaceImageContoning);
    if (name == "same_layer_45_partition")
        return int(TextureMappingZone::TopSurfaceImageSameLayer45Partition);
    return int(TextureMappingZone::TopSurfaceImageSameAngle45Width);
}

static std::string top_visible_recolor_aggressiveness_name(int mode)
{
    switch (clamp_int(mode,
                      int(TextureMappingZone::TopVisibleRecolorConservative),
                      int(TextureMappingZone::TopVisibleRecolorAggressive))) {
    case int(TextureMappingZone::TopVisibleRecolorAggressive): return "aggressive";
    case int(TextureMappingZone::TopVisibleRecolorBalanced):   return "balanced";
    default:                                                   return "conservative";
    }
}

static int top_visible_recolor_aggressiveness_from_name(const std::string &name)
{
    if (name == "aggressive")
        return int(TextureMappingZone::TopVisibleRecolorAggressive);
    if (name == "balanced")
        return int(TextureMappingZone::TopVisibleRecolorBalanced);
    return int(TextureMappingZone::TopVisibleRecolorConservative);
}

static std::string color_model_name(int mode)
{
    switch (clamp_int(mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorRGB):  return "rgb";
    case int(TextureMappingZone::FilamentColorCMY):  return "cmy";
    case int(TextureMappingZone::FilamentColorCMYK): return "cmyk";
    case int(TextureMappingZone::FilamentColorCMYW): return "cmyw";
    case int(TextureMappingZone::FilamentColorRGBK): return "rgbk";
    case int(TextureMappingZone::FilamentColorRGBW): return "rgbw";
    case int(TextureMappingZone::FilamentColorBW):   return "bw";
    case int(TextureMappingZone::FilamentColorCMYKW): return "cmykw";
    case int(TextureMappingZone::FilamentColorRGBKW): return "rgbkw";
    default:                                         return "any";
    }
}

static int color_model_from_name(const std::string &name)
{
    if (name == "rgb")  return int(TextureMappingZone::FilamentColorRGB);
    if (name == "cmy")  return int(TextureMappingZone::FilamentColorCMY);
    if (name == "cmyk") return int(TextureMappingZone::FilamentColorCMYK);
    if (name == "cmyw") return int(TextureMappingZone::FilamentColorCMYW);
    if (name == "rgbk") return int(TextureMappingZone::FilamentColorRGBK);
    if (name == "rgbw") return int(TextureMappingZone::FilamentColorRGBW);
    if (name == "bw")   return int(TextureMappingZone::FilamentColorBW);
    if (name == "cmykw") return int(TextureMappingZone::FilamentColorCMYKW);
    if (name == "rgbkw") return int(TextureMappingZone::FilamentColorRGBKW);
    return int(TextureMappingZone::FilamentColorAny);
}

static std::string generic_solver_lookup_mode_name(int mode)
{
    return clamp_int(mode,
                     int(TextureMappingZone::GenericSolverClosestMix),
                     int(TextureMappingZone::GenericSolverBlendClosestTwo)) ==
               int(TextureMappingZone::GenericSolverBlendClosestTwo) ?
        std::string("blend_closest_two") :
        std::string("closest_mix");
}

static int generic_solver_lookup_mode_from_name(const std::string &name)
{
    return name == "blend_closest_two" ?
        int(TextureMappingZone::GenericSolverBlendClosestTwo) :
        int(TextureMappingZone::GenericSolverClosestMix);
}

static std::string generic_solver_mode_name(int mode)
{
    return clamp_int(mode,
                     int(TextureMappingZone::GenericSolverLegacy),
                     int(TextureMappingZone::GenericSolverV2)) ==
               int(TextureMappingZone::GenericSolverLegacy) ?
        std::string("legacy") :
        std::string("v2");
}

static int generic_solver_mode_from_name(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return name == "legacy" || name == "v1" ?
        int(TextureMappingZone::GenericSolverLegacy) :
        int(TextureMappingZone::GenericSolverV2);
}

static std::string generic_solver_mix_model_name(int)
{
    return "pigment_painter";
}

static int generic_solver_mix_model_from_name(std::string)
{
    return TextureMappingZone::DefaultGenericSolverMixModel;
}

static std::string dithering_method_name(int mode)
{
    switch (clamp_int(mode,
                      int(TextureMappingZone::DitheringClosest),
                      int(TextureMappingZone::DitheringHalftoneV2))) {
    case int(TextureMappingZone::DitheringClosest):                 return "closest";
    case int(TextureMappingZone::DitheringOrderedBayer):            return "ordered_bayer";
    case int(TextureMappingZone::DitheringHalftone):                return "halftone";
    case int(TextureMappingZone::DitheringHalftoneIncreasedDetail): return "halftone_increased_detail";
    case int(TextureMappingZone::DitheringHalftoneV2):              return "halftone_v2";
    default:                                                       return "floyd_steinberg";
    }
}

static int dithering_method_from_name(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (name == "closest" || name == "closest_combination")
        return int(TextureMappingZone::DitheringClosest);
    if (name == "ordered_bayer" || name == "bayer")
        return int(TextureMappingZone::DitheringOrderedBayer);
    if (name == "halftone")
        return int(TextureMappingZone::DitheringHalftone);
    if (name == "halftone_increased_detail" || name == "halftone_detail" || name == "halftone_high_detail")
        return int(TextureMappingZone::DitheringHalftoneIncreasedDetail);
    if (name == "halftone_v2" || name == "halftone2")
        return int(TextureMappingZone::DitheringHalftoneV2);
    return int(TextureMappingZone::DitheringFloydSteinberg);
}

static std::string transmission_distance_calibration_mode_name(int mode)
{
    switch (clamp_int(mode,
                      int(TextureMappingZone::TDCalibrationNone),
                      int(TextureMappingZone::TDCalibrationNeighbor))) {
    case int(TextureMappingZone::TDCalibrationNone):     return "none";
    case int(TextureMappingZone::TDCalibrationNeighbor): return "neighbor";
    default:                                            return "absolute";
    }
}

static int transmission_distance_calibration_mode_from_name(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (name == "none" || name == "off" || name == "disabled")
        return int(TextureMappingZone::TDCalibrationNone);
    if (name == "neighbor" || name == "neighbour")
        return int(TextureMappingZone::TDCalibrationNeighbor);
    return int(TextureMappingZone::TDCalibrationAbsolute);
}

static int transmission_distance_calibration_mode_from_json(const nlohmann::json &texture)
{
    const auto mode_it = texture.find("transmission_distance_calibration");
    if (mode_it != texture.end()) {
        if (mode_it->is_string())
            return transmission_distance_calibration_mode_from_name(mode_it->get<std::string>());
        if (mode_it->is_number_integer() || mode_it->is_number_unsigned())
            return clamp_int(mode_it->get<int>(),
                             int(TextureMappingZone::TDCalibrationNone),
                             int(TextureMappingZone::TDCalibrationNeighbor));
        if (mode_it->is_boolean())
            return mode_it->get<bool>() ?
                int(TextureMappingZone::TDCalibrationAbsolute) :
                int(TextureMappingZone::TDCalibrationNone);
    }

    const auto legacy_it = texture.find("transmission_distance_calibration_enabled");
    if (legacy_it != texture.end() && legacy_it->is_boolean())
        return legacy_it->get<bool>() ?
            int(TextureMappingZone::TDCalibrationAbsolute) :
            int(TextureMappingZone::TDCalibrationNone);

    return TextureMappingZone::DefaultTransmissionDistanceCalibrationMode;
}

static std::string normalized_prime_tower_color_mode_name(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (name == "auto" || name.empty())
        return "auto";
    if (name == "generic" || name == "generic_solver" || name == "solver")
        return "generic_solver";
    if (name == "rgb" || name == "cmy" || name == "cmyk" || name == "cmyw" ||
        name == "rgbk" || name == "rgbw" || name == "bw" || name == "cmykw" || name == "rgbkw")
        return name;
    return "auto";
}

} // namespace

std::string TextureMappingGlobalSettings::serialize() const
{
    const std::string normalized_mode = normalize_color_mode_name(prime_tower_color_mode);
    if (!enabled &&
        std::abs((std::isfinite(angle_offset_deg) ? angle_offset_deg : 0.f)) <= 1e-6f &&
        !preserve_aspect_ratio &&
        normalized_mode == "auto" &&
        prime_tower_settings_zone_uid == 0 &&
        image_file.empty() &&
        image_name.empty() &&
        image_width == 0 &&
        image_height == 0 &&
        image_file_back.empty() &&
        image_name_back.empty() &&
        image_width_back == 0 &&
        image_height_back == 0)
        return {};

    nlohmann::json root;
    root["schema"] = 1;
    nlohmann::json prime_tower_texture_mapping;
    prime_tower_texture_mapping["enabled"] = enabled;
    prime_tower_texture_mapping["angle_offset_deg"] = std::clamp(std::isfinite(angle_offset_deg) ? angle_offset_deg : 0.f, 0.f, 360.f);
    prime_tower_texture_mapping["preserve_aspect_ratio"] = preserve_aspect_ratio;
    prime_tower_texture_mapping["prime_tower_color_mode"] = normalized_mode;
    prime_tower_texture_mapping["settings_zone_uid"] = prime_tower_settings_zone_uid;
    prime_tower_texture_mapping["image_file"] = image_file;
    prime_tower_texture_mapping["image_name"] = image_name;
    prime_tower_texture_mapping["image_width"] = image_width;
    prime_tower_texture_mapping["image_height"] = image_height;
    prime_tower_texture_mapping["image_file_back"] = image_file_back;
    prime_tower_texture_mapping["image_name_back"] = image_name_back;
    prime_tower_texture_mapping["image_width_back"] = image_width_back;
    prime_tower_texture_mapping["image_height_back"] = image_height_back;
    root["prime_tower_texture_mapping"] = std::move(prime_tower_texture_mapping);
    return root.dump();
}

void TextureMappingGlobalSettings::load(const std::string &serialized)
{
    *this = TextureMappingGlobalSettings();
    if (serialized.empty())
        return;

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(serialized);
    } catch (const std::exception &e) {
        BOOST_LOG_TRIVIAL(warning) << "TextureMappingGlobalSettings::load JSON parse failed: " << e.what();
        return;
    }
    if (!root.is_object())
        return;

    const auto prime_tower_texture_mapping_it = root.find("prime_tower_texture_mapping");
    if (prime_tower_texture_mapping_it == root.end() || !prime_tower_texture_mapping_it->is_object())
        return;

    const nlohmann::json &prime_tower_texture_mapping = *prime_tower_texture_mapping_it;
    enabled = prime_tower_texture_mapping.value("enabled", false);
    angle_offset_deg = std::clamp(prime_tower_texture_mapping.value("angle_offset_deg", 0.f), 0.f, 360.f);
    preserve_aspect_ratio = prime_tower_texture_mapping.value("preserve_aspect_ratio", false);
    prime_tower_color_mode = normalize_color_mode_name(prime_tower_texture_mapping.value("prime_tower_color_mode", std::string("auto")));
    prime_tower_settings_zone_uid = prime_tower_texture_mapping.value("settings_zone_uid", uint64_t(0));
    image_file = prime_tower_texture_mapping.value("image_file", std::string());
    image_name = prime_tower_texture_mapping.value("image_name", std::string());
    image_width = prime_tower_texture_mapping.value("image_width", 0u);
    image_height = prime_tower_texture_mapping.value("image_height", 0u);
    image_file_back = prime_tower_texture_mapping.value("image_file_back", std::string());
    image_name_back = prime_tower_texture_mapping.value("image_name_back", std::string());
    image_width_back = prime_tower_texture_mapping.value("image_width_back", 0u);
    image_height_back = prime_tower_texture_mapping.value("image_height_back", 0u);
}

void TextureMappingGlobalSettings::clear_image_reference()
{
    image_file.clear();
    image_name.clear();
    image_width = 0;
    image_height = 0;
    image_file_back.clear();
    image_name_back.clear();
    image_width_back = 0;
    image_height_back = 0;
}

std::string TextureMappingGlobalSettings::normalize_color_mode_name(const std::string &mode)
{
    return normalized_prime_tower_color_mode_name(mode);
}

bool TextureMappingGlobalSettings::is_generic_solver_color_mode(const std::string &mode)
{
    return normalize_color_mode_name(mode) == "generic_solver";
}

bool TextureMappingZone::operator==(const TextureMappingZone &rhs) const
{
    constexpr float eps = 1e-6f;
    auto floats_equal = [](const std::vector<float> &lhs, const std::vector<float> &rhs_values) {
        if (lhs.size() != rhs_values.size())
            return false;
        for (size_t i = 0; i < lhs.size(); ++i)
            if (std::abs(lhs[i] - rhs_values[i]) > 1e-6f)
                return false;
        return true;
    };
    auto anchors_equal = [eps](const TextureMappingZone::LinearGradientAnchor &lhs,
                               const TextureMappingZone::LinearGradientAnchor &rhs_values) {
        if (lhs.valid != rhs_values.valid ||
            lhs.object_id != rhs_values.object_id ||
            lhs.instance_id != rhs_values.instance_id ||
            lhs.object_backup_id != rhs_values.object_backup_id ||
            lhs.object_index_valid != rhs_values.object_index_valid ||
            lhs.object_index != rhs_values.object_index ||
            lhs.instance_index_valid != rhs_values.instance_index_valid ||
            lhs.instance_index != rhs_values.instance_index ||
            lhs.instance_loaded_id != rhs_values.instance_loaded_id)
            return false;
        for (size_t i = 0; i < 3; ++i) {
            if (std::abs(lhs.local_point[i] - rhs_values.local_point[i]) > eps ||
                std::abs(lhs.global_point[i] - rhs_values.global_point[i]) > eps)
                return false;
        }
        return true;
    };
    auto stops_equal = [eps](const std::vector<TextureMappingZone::LinearGradientStop> &lhs,
                             const std::vector<TextureMappingZone::LinearGradientStop> &rhs_values) {
        if (lhs.size() != rhs_values.size())
            return false;
        for (size_t i = 0; i < lhs.size(); ++i) {
            if (std::abs(lhs[i].position - rhs_values[i].position) > eps ||
                lhs[i].filament_id != rhs_values[i].filament_id)
                return false;
        }
        return true;
    };

    return stable_id == rhs.stable_id &&
           zone_id == rhs.zone_id &&
           enabled == rhs.enabled &&
           deleted == rhs.deleted &&
           surface_pattern == rhs.surface_pattern &&
           component_a == rhs.component_a &&
           component_b == rhs.component_b &&
           component_ids == rhs.component_ids &&
           component_weights == rhs.component_weights &&
           display_color == rhs.display_color &&
           offset_distances == rhs.offset_distances &&
           offset_angles == rhs.offset_angles &&
           offset_mode == rhs.offset_mode &&
           offset_rotation_enabled == rhs.offset_rotation_enabled &&
           std::abs(offset_rotations - rhs.offset_rotations) <= eps &&
           std::abs(offset_repeats - rhs.offset_repeats) <= eps &&
           offset_reverse_repeats == rhs.offset_reverse_repeats &&
           offset_clockwise == rhs.offset_clockwise &&
           offset_fade_mode == rhs.offset_fade_mode &&
           offset_angle_mode == rhs.offset_angle_mode &&
           texture_mapping_mode == rhs.texture_mapping_mode &&
           filament_color_mode == rhs.filament_color_mode &&
           force_sequential_filaments == rhs.force_sequential_filaments &&
           reduce_outer_surface_texture == rhs.reduce_outer_surface_texture &&
           seam_hiding == rhs.seam_hiding &&
           nonlinear_offset_adjustment == rhs.nonlinear_offset_adjustment &&
           modulation_mode == rhs.modulation_mode &&
           use_modulated_overhang_geometry_for_support == rhs.use_modulated_overhang_geometry_for_support &&
           modulation_mode_manually_changed == rhs.modulation_mode_manually_changed &&
           recolor_small_perimeter_loops == rhs.recolor_small_perimeter_loops &&
           recolor_top_visible_perimeter_sections == rhs.recolor_top_visible_perimeter_sections &&
           top_visible_perimeter_recolor_aggressiveness == rhs.top_visible_perimeter_recolor_aggressiveness &&
           top_visible_perimeter_recolor_above_layers == rhs.top_visible_perimeter_recolor_above_layers &&
           top_visible_perimeter_recolor_point_sampling == rhs.top_visible_perimeter_recolor_point_sampling &&
           top_surface_image_printing_enabled == rhs.top_surface_image_printing_enabled &&
           top_surface_image_printing_method == rhs.top_surface_image_printing_method &&
           std::abs(top_surface_image_min_line_width_mm - rhs.top_surface_image_min_line_width_mm) <= eps &&
           std::abs(top_surface_image_max_line_width_mm - rhs.top_surface_image_max_line_width_mm) <= eps &&
           top_surface_image_colored_top_layers == rhs.top_surface_image_colored_top_layers &&
           top_surface_image_fixed_coloring_filaments == rhs.top_surface_image_fixed_coloring_filaments &&
           std::abs(top_surface_contoning_angle_threshold_deg - rhs.top_surface_contoning_angle_threshold_deg) <= eps &&
           top_surface_contoning_stack_layers == rhs.top_surface_contoning_stack_layers &&
           std::abs(top_surface_contoning_min_feature_mm - rhs.top_surface_contoning_min_feature_mm) <= eps &&
           top_surface_contoning_color_lower_surfaces == rhs.top_surface_contoning_color_lower_surfaces &&
           top_surface_contoning_only_color_surface_infill == rhs.top_surface_contoning_only_color_surface_infill &&
           compact_offset_mode == rhs.compact_offset_mode &&
           use_legacy_fixed_color_mode == rhs.use_legacy_fixed_color_mode &&
           high_speed_image_texture_sampling == rhs.high_speed_image_texture_sampling &&
           minimum_visibility_offset_enabled == rhs.minimum_visibility_offset_enabled &&
           std::abs(minimum_visibility_offset_pct - rhs.minimum_visibility_offset_pct) <= eps &&
           generic_solver_lookup_mode == rhs.generic_solver_lookup_mode &&
           generic_solver_mode == rhs.generic_solver_mode &&
           generic_solver_mix_model == rhs.generic_solver_mix_model &&
           dithering_enabled == rhs.dithering_enabled &&
           dithering_method == rhs.dithering_method &&
           std::abs(dithering_resolution_mm - rhs.dithering_resolution_mm) <= eps &&
           std::abs(halftone_dot_size_mm - rhs.halftone_dot_size_mm) <= eps &&
           std::abs(contrast_pct - rhs.contrast_pct) <= eps &&
           high_resolution_sampling == rhs.high_resolution_sampling &&
           std::abs(tone_gamma - rhs.tone_gamma) <= eps &&
           transmission_distance_calibration_mode == rhs.transmission_distance_calibration_mode &&
           std::abs(preview_opacity_pct - rhs.preview_opacity_pct) <= eps &&
           preview_simulate_colors == rhs.preview_simulate_colors &&
           preview_limit_resolution == rhs.preview_limit_resolution &&
           auto_adjust_filament_selection == rhs.auto_adjust_filament_selection &&
           floats_equal(filament_strengths_pct, rhs.filament_strengths_pct) &&
           floats_equal(filament_minimum_offsets_pct, rhs.filament_minimum_offsets_pct) &&
           floats_equal(filament_transmission_distances_mm, rhs.filament_transmission_distances_mm) &&
           anchors_equal(linear_gradient_start, rhs.linear_gradient_start) &&
           anchors_equal(linear_gradient_end, rhs.linear_gradient_end) &&
           linear_gradient_mode == rhs.linear_gradient_mode &&
           std::abs(linear_gradient_radius_mm - rhs.linear_gradient_radius_mm) <= eps &&
           linear_gradient_radius_percent == rhs.linear_gradient_radius_percent &&
           std::abs(linear_gradient_radius_pct - rhs.linear_gradient_radius_pct) <= eps &&
           show_linear_gradient_direction_arrow == rhs.show_linear_gradient_direction_arrow &&
           stops_equal(linear_gradient_stops, rhs.linear_gradient_stops);
}

uint64_t TextureMappingManager::allocate_stable_id()
{
    const uint64_t stable_id = std::max<uint64_t>(1, m_next_stable_id);
    m_next_stable_id = stable_id + 1;
    return stable_id;
}

uint64_t TextureMappingManager::normalize_stable_id(uint64_t stable_id)
{
    if (stable_id == 0)
        return allocate_stable_id();
    if (stable_id >= m_next_stable_id)
        m_next_stable_id = stable_id + 1;
    return stable_id;
}

void TextureMappingManager::clear()
{
    m_zones.clear();
}

void TextureMappingManager::refresh(const std::vector<std::string> &filament_colours)
{
    m_filament_colours = filament_colours;
    for (TextureMappingZone &zone : m_zones) {
        zone.stable_id = normalize_stable_id(zone.stable_id);
        if (zone.display_color.empty() || zone.display_color[0] != '#')
            zone.display_color = random_display_color(zone.stable_id);
        if (zone.is_linear_gradient()) {
            zone.linear_gradient_stops = normalized_linear_gradient_stops(zone, m_filament_colours.size());
            const std::vector<unsigned int> ids = linear_gradient_component_ids_from_stops(zone, m_filament_colours.size());
            zone.component_ids = encode_component_ids(ids);
            if (!zone.linear_gradient_stops.empty()) {
                zone.component_a = zone.linear_gradient_stops.front().filament_id;
                zone.component_b = zone.linear_gradient_stops.back().filament_id;
            }
        }
        zone.apply_default_modulation_mode();
    }
}

void TextureMappingManager::remove_physical_filament(unsigned int deleted_filament_id)
{
    if (deleted_filament_id == 0)
        return;

    if (deleted_filament_id <= m_filament_colours.size())
        m_filament_colours.erase(m_filament_colours.begin() + ptrdiff_t(deleted_filament_id - 1));
    const size_t new_physical_count = m_filament_colours.size();

    auto remap_id = [deleted_filament_id](unsigned int id) {
        if (id == deleted_filament_id)
            return 0u;
        if (id > deleted_filament_id)
            return id - 1;
        return id;
    };

    for (TextureMappingZone &zone : m_zones) {
        zone.component_a = remap_id(zone.component_a);
        zone.component_b = remap_id(zone.component_b);
        for (TextureMappingZone::LinearGradientStop &stop : zone.linear_gradient_stops)
            stop.filament_id = remap_id(stop.filament_id);

        std::vector<unsigned int> ids = decode_component_ids(zone.component_ids, 9);
        for (unsigned int &id : ids)
            id = remap_id(id);
        ids.erase(std::remove(ids.begin(), ids.end(), 0u), ids.end());
        if (ids.size() < 2) {
            ids.clear();
            if (zone.component_a >= 1 && zone.component_a <= new_physical_count)
                ids.emplace_back(zone.component_a);
            if (zone.component_b >= 1 && zone.component_b <= new_physical_count && zone.component_b != zone.component_a)
                ids.emplace_back(zone.component_b);
        }
        if (ids.size() < 2 && new_physical_count >= 2) {
            ids.clear();
            for (size_t i = 1; i <= std::min<size_t>(new_physical_count, 9); ++i)
                ids.emplace_back(unsigned(i));
            zone.component_a = ids[0];
            zone.component_b = ids[1];
        }
        if (zone.is_linear_gradient()) {
            zone.linear_gradient_stops = normalized_linear_gradient_stops(zone, new_physical_count);
            ids = linear_gradient_component_ids_from_stops(zone, new_physical_count);
            if (!zone.linear_gradient_stops.empty()) {
                zone.component_a = zone.linear_gradient_stops.front().filament_id;
                zone.component_b = zone.linear_gradient_stops.back().filament_id;
            }
        } else if (ids.size() >= 2) {
            zone.component_a = ids[0];
            zone.component_b = ids[1];
        }
        if (new_physical_count == 0 || (!zone.is_linear_gradient() && new_physical_count < 2))
            zone.enabled = false;
        zone.component_ids = encode_component_ids(ids);

        auto remove_index = [deleted_filament_id](std::vector<float> &values) {
            if (deleted_filament_id >= 1 && size_t(deleted_filament_id - 1) < values.size())
                values.erase(values.begin() + ptrdiff_t(deleted_filament_id - 1));
        };
        remove_index(zone.filament_strengths_pct);
        remove_index(zone.filament_minimum_offsets_pct);
        remove_index(zone.filament_transmission_distances_mm);
    }
    normalize_zone_ids(new_physical_count);
}

TextureMappingZone *TextureMappingManager::add_zone(size_t num_physical,
                                                    const std::vector<std::string> &filament_colours,
                                                    int surface_pattern)
{
    const bool linear_gradient = surface_pattern == int(TextureMappingZone::LinearGradient);
    if (num_physical == 0 || (!linear_gradient && num_physical < 2))
        return nullptr;

    TextureMappingZone zone;
    zone.stable_id = allocate_stable_id();
    zone.zone_id = allocate_zone_id(num_physical);
    if (zone.zone_id == 0)
        return nullptr;
    zone.surface_pattern =
        surface_pattern == int(TextureMappingZone::Gradient2D) || surface_pattern == int(TextureMappingZone::LinearGradient) ?
            surface_pattern :
            int(TextureMappingZone::ImageTexture);
    zone.apply_default_modulation_mode();

    std::vector<unsigned int> ids;
    for (size_t i = 1; i <= std::min<size_t>(num_physical, 4); ++i)
        ids.emplace_back(unsigned(i));
    zone.component_ids = encode_component_ids(ids);
    zone.component_a = ids[0];
    zone.component_b = ids.size() > 1 ? ids[1] : ids[0];
    if (zone.is_linear_gradient()) {
        zone.linear_gradient_stops = {
            {0.f, zone.component_a},
            {1.f, zone.component_b}
        };
        zone.component_ids = encode_component_ids(linear_gradient_component_ids_from_stops(zone, num_physical));
    }
    zone.display_color = random_display_color(zone.stable_id);
    if (zone.is_image_texture())
        auto_adjust_texture_component_ids(zone, num_physical, filament_colours);

    m_zones.emplace_back(std::move(zone));
    refresh(filament_colours);
    return &m_zones.back();
}

bool TextureMappingManager::duplicate_zone(size_t zone_index,
                                           size_t num_physical,
                                           const std::vector<std::string> &filament_colours)
{
    if (zone_index >= m_zones.size() || num_physical == 0)
        return false;

    TextureMappingZone copy = m_zones[zone_index];
    if (!copy.is_linear_gradient() && num_physical < 2)
        return false;
    copy.stable_id = allocate_stable_id();
    copy.zone_id = allocate_zone_id(num_physical);
    if (copy.zone_id == 0)
        return false;
    copy.enabled = true;
    copy.deleted = false;
    copy.display_color = random_display_color(copy.stable_id);

    m_zones.insert(m_zones.begin() + ptrdiff_t(zone_index + 1), std::move(copy));
    normalize_zone_ids(num_physical);
    refresh(filament_colours);
    return true;
}

unsigned int TextureMappingManager::find_image_texture_zone_id(size_t,
                                                               bool allow_raw_values,
                                                               bool prefer_raw_values) const
{
    auto selectable = [allow_raw_values](const TextureMappingZone &zone) {
        const bool raw_values = zone.texture_mapping_mode == int(TextureMappingZone::TextureMappingRawValues);
        return zone.enabled && !zone.deleted && zone.is_image_texture() && zone.zone_id != 0 && (allow_raw_values || !raw_values);
    };
    auto find_raw = [this, selectable](bool raw_values) {
        for (const TextureMappingZone &zone : m_zones)
            if (selectable(zone) &&
                (zone.texture_mapping_mode == int(TextureMappingZone::TextureMappingRawValues)) == raw_values)
                return zone.zone_id;
        return 0u;
    };

    if (allow_raw_values && prefer_raw_values) {
        const unsigned int raw_id = find_raw(true);
        if (raw_id != 0)
            return raw_id;
    }

    for (const TextureMappingZone &zone : m_zones)
        if (selectable(zone))
            return zone.zone_id;
    return 0;
}

unsigned int TextureMappingManager::ensure_image_texture_zone(size_t num_physical,
                                                              const std::vector<std::string> &filament_colours,
                                                              bool allow_raw_values,
                                                              bool prefer_raw_values)
{
    if (unsigned int existing = find_image_texture_zone_id(num_physical, allow_raw_values, prefer_raw_values);
        existing != 0)
        return existing;
    TextureMappingZone *zone = add_zone(num_physical, filament_colours, int(TextureMappingZone::ImageTexture));
    return zone != nullptr ? zone->zone_id : 0;
}

std::string TextureMappingManager::serialize_entries()
{
    normalize_zone_ids(m_filament_colours.size());

    nlohmann::json root = nlohmann::json::array();
    for (TextureMappingZone &zone : m_zones) {
        if (zone.deleted)
            continue;

        zone.stable_id = normalize_stable_id(zone.stable_id);
        if (zone.display_color.empty() || zone.display_color[0] != '#')
            zone.display_color = random_display_color(zone.stable_id);
        zone.apply_default_modulation_mode();

        std::vector<unsigned int> component_ids = decode_component_ids(zone.component_ids, 9);
        if (zone.is_linear_gradient()) {
            zone.linear_gradient_stops = normalized_linear_gradient_stops(zone, m_filament_colours.size());
            component_ids = linear_gradient_component_ids_from_stops(zone, m_filament_colours.size());
            if (!zone.linear_gradient_stops.empty()) {
                zone.component_a = zone.linear_gradient_stops.front().filament_id;
                zone.component_b = zone.linear_gradient_stops.back().filament_id;
            }
            zone.component_ids = encode_component_ids(component_ids);
        }
        if (!zone.is_linear_gradient() && component_ids.size() < 2) {
            component_ids = {zone.component_a, zone.component_b};
            component_ids.erase(std::remove_if(component_ids.begin(), component_ids.end(), [](unsigned int id) {
                return id == 0 || id > 9;
            }), component_ids.end());
        }
        unsigned int anchor_a = zone.component_a;
        unsigned int anchor_b = zone.component_b;

        const std::string normalized_weights = normalize_weights(zone.component_weights, component_ids.size());
        const std::string normalized_distances =
            normalize_offset_distances(zone.offset_distances, component_ids.size(), max_component_surface_offset_mm());
        const std::string normalized_angles = normalize_offset_angles(zone.offset_angles, component_ids.size());
        const std::vector<float> normalized_strengths = normalize_strengths(zone.filament_strengths_pct);
        const std::vector<float> normalized_min_offsets = normalize_minimum_offsets(zone.filament_minimum_offsets_pct);
        const std::vector<float> normalized_transmission_distances =
            normalize_transmission_distances(zone.filament_transmission_distances_mm);

        nlohmann::json entry;
        entry["schema"] = 2;
        entry["uid"] = zone.stable_id;
        entry["zone_id"] = zone.zone_id;
        entry["enabled"] = zone.enabled;
        entry["surface_pattern"] = surface_pattern_name(zone.surface_pattern);
        if (!zone.is_linear_gradient()) {
            entry["anchor_filaments"] = {anchor_a, anchor_b};
            entry["component_filaments"] = ids_to_json(component_ids);
            entry["component_weights_pct"] = weights_to_json(normalized_weights, component_ids.size());
        }
        entry["display_color"] = zone.display_color;
        if (zone.is_linear_gradient()) {
            nlohmann::json linear_gradient;
            linear_gradient["start"] = linear_gradient_anchor_to_json(zone.linear_gradient_start);
            linear_gradient["end"] = linear_gradient_anchor_to_json(zone.linear_gradient_end);
            linear_gradient["stops"] = linear_gradient_stops_to_json(zone.linear_gradient_stops);
            linear_gradient["mode"] = linear_gradient_mode_name(zone.linear_gradient_mode);
            linear_gradient["radius_mm"] = std::isfinite(zone.linear_gradient_radius_mm) ?
                std::max(0.f, zone.linear_gradient_radius_mm) :
                TextureMappingZone::DefaultLinearGradientRadiusMm;
            linear_gradient["radius_percent"] = zone.linear_gradient_radius_percent;
            linear_gradient["radius_pct"] = std::isfinite(zone.linear_gradient_radius_pct) ?
                std::max(0.f, zone.linear_gradient_radius_pct) :
                TextureMappingZone::DefaultLinearGradientRadiusPct;
            linear_gradient["show_direction_arrow"] = zone.show_linear_gradient_direction_arrow;
            entry["linear_gradient"] = std::move(linear_gradient);
        }

        nlohmann::json texture;
        texture["mode"] = mapping_mode_name(clamp_int(zone.texture_mapping_mode,
                                                       int(TextureMappingZone::TextureMappingFilamentBlending),
                                                       int(TextureMappingZone::TextureMappingRawValues)));
        texture["color_model"] = color_model_name(zone.filament_color_mode);
        texture["ordered_roles"] = zone.force_sequential_filaments;
        texture["reduce_outer_surface_texture"] = false;
        texture["hide_seams"] = zone.seam_hiding;
        texture["nonlinear_offset_adjustment"] = zone.nonlinear_offset_adjustment;
        texture["modulation_mode"] = modulation_mode_name(zone.modulation_mode);
        texture["use_modulated_overhang_geometry_for_support"] = zone.use_modulated_overhang_geometry_for_support;
        texture["modulation_mode_manually_changed"] = zone.modulation_mode_manually_changed;
        texture["recolor_small_perimeter_loops"] = zone.recolor_small_perimeter_loops || zone.recolor_top_visible_perimeter_sections;
        texture["recolor_top_visible_perimeter_sections"] = zone.recolor_top_visible_perimeter_sections;
        texture["top_visible_perimeter_recolor_aggressiveness"] =
            top_visible_recolor_aggressiveness_name(zone.top_visible_perimeter_recolor_aggressiveness);
        texture["top_visible_perimeter_recolor_above_layers"] =
            clamp_int(zone.top_visible_perimeter_recolor_above_layers,
                      TextureMappingZone::MinTopVisiblePerimeterRecolorAboveLayers,
                      TextureMappingZone::MaxTopVisiblePerimeterRecolorAboveLayers);
        texture["top_visible_perimeter_recolor_point_sampling"] = zone.top_visible_perimeter_recolor_point_sampling;
        texture["top_surface_image_printing_enabled"] = zone.top_surface_image_printing_enabled;
        texture["top_surface_image_printing_method"] =
            top_surface_image_printing_method_name(zone.top_surface_image_printing_method);
        const float top_surface_max_width =
            std::clamp(finite_or(zone.top_surface_image_max_line_width_mm,
                                 TextureMappingZone::DefaultTopSurfaceImageMaxLineWidthMm),
                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
        texture["top_surface_image_min_line_width_mm"] =
            std::clamp(finite_or(zone.top_surface_image_min_line_width_mm,
                                 TextureMappingZone::DefaultTopSurfaceImageMinLineWidthMm),
                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                       top_surface_max_width);
        texture["top_surface_image_max_line_width_mm"] = top_surface_max_width;
        texture["top_surface_image_colored_top_layers"] =
            clamp_int(zone.top_surface_image_colored_top_layers,
                      TextureMappingZone::MinTopSurfaceImageColoredTopLayers,
                      TextureMappingZone::MaxTopSurfaceImageColoredTopLayers);
        texture["top_surface_image_fixed_coloring_filaments"] = zone.top_surface_image_fixed_coloring_filaments;
        texture["top_surface_contoning_angle_threshold_deg"] =
            std::clamp(finite_or(zone.top_surface_contoning_angle_threshold_deg,
                                 TextureMappingZone::DefaultTopSurfaceContoningAngleThresholdDeg),
                       TextureMappingZone::MinTopSurfaceContoningAngleThresholdDeg,
                       TextureMappingZone::MaxTopSurfaceContoningAngleThresholdDeg);
        texture["top_surface_contoning_stack_layers"] =
            clamp_int(zone.top_surface_contoning_stack_layers,
                      TextureMappingZone::MinTopSurfaceContoningStackLayers,
                      TextureMappingZone::MaxTopSurfaceContoningStackLayers);
        texture["top_surface_contoning_min_feature_mm"] =
            std::clamp(finite_or(zone.top_surface_contoning_min_feature_mm,
                                 TextureMappingZone::DefaultTopSurfaceContoningMinFeatureMm),
                       TextureMappingZone::MinTopSurfaceContoningMinFeatureMm,
                       TextureMappingZone::MaxTopSurfaceContoningMinFeatureMm);
        texture["top_surface_contoning_color_lower_surfaces"] = zone.top_surface_contoning_color_lower_surfaces;
        texture["top_surface_contoning_only_color_surface_infill"] = zone.top_surface_contoning_only_color_surface_infill;
        texture["compact_offset_mode"] = zone.compact_offset_mode;
        texture["use_legacy_fixed_color_mode"] = zone.use_legacy_fixed_color_mode;
        texture["high_speed_image_texture_sampling"] = true;
        texture["minimum_visibility_offset_enabled"] = zone.minimum_visibility_offset_enabled;
        texture["minimum_visibility_offset_pct"] =
            std::clamp(finite_or(zone.minimum_visibility_offset_pct, TextureMappingZone::DefaultMinimumVisibilityOffsetPct), 0.f, 100.f);
        texture["generic_solver_lookup"] = generic_solver_lookup_mode_name(zone.generic_solver_lookup_mode);
        texture["generic_solver_mode"] = generic_solver_mode_name(zone.generic_solver_mode);
        texture["generic_solver_mix_model"] = generic_solver_mix_model_name(zone.generic_solver_mix_model);
        texture["dithering_enabled"] = zone.dithering_enabled;
        texture["dithering_method"] = dithering_method_name(zone.dithering_method);
        texture["dithering_resolution_mm"] =
            std::clamp(finite_or(zone.dithering_resolution_mm, TextureMappingZone::DefaultDitheringResolutionMm),
                       TextureMappingZone::MinDitheringResolutionMm,
                       TextureMappingZone::MaxDitheringResolutionMm);
        texture["halftone_dot_size_mm"] =
            std::clamp(finite_or(zone.halftone_dot_size_mm, TextureMappingZone::DefaultHalftoneDotSizeMm),
                       TextureMappingZone::MinHalftoneDotSizeMm,
                       TextureMappingZone::MaxHalftoneDotSizeMm);
        texture["contrast_pct"] = std::clamp(finite_or(zone.contrast_pct, 100.f), 25.f, 300.f);
        texture["high_resolution_sampling"] = true;
        texture["tone_gamma"] = normalize_tone_gamma(zone.tone_gamma);
        texture["transmission_distance_calibration"] =
            transmission_distance_calibration_mode_name(zone.transmission_distance_calibration_mode);
        texture["preview_opacity_pct"] =
            std::clamp(finite_or(zone.preview_opacity_pct, TextureMappingZone::DefaultPreviewOpacityPct), 0.f, 100.f);
        texture["simulate_preview_colors"] = zone.preview_simulate_colors;
        texture["limit_preview_resolution"] = zone.preview_limit_resolution;
        texture["auto_adjust_filaments"] = zone.auto_adjust_filament_selection;
        texture["strength_pct"] = floats_to_json(normalized_strengths);
        texture["minimum_offset_pct"] = floats_to_json(normalized_min_offsets);
        texture["transmission_distance_mm"] = floats_to_json(normalized_transmission_distances);
        entry["texture_options"] = std::move(texture);

        nlohmann::json offset;
        offset["distances_mm_by_filament"] = normalized_distances;
        offset["angles_deg_by_filament"] = normalized_angles;
        offset["control_mode"] = clamp_int(zone.offset_mode,
                                            int(TextureMappingZone::OffsetBasic),
                                            int(TextureMappingZone::OffsetAdvanced));
        offset["rotate_with_height"] = zone.offset_rotation_enabled;
        offset["rotations"] = zone.offset_rotations;
        offset["repeats"] = zone.offset_repeats;
        offset["alternate_repeats"] = zone.offset_reverse_repeats;
        offset["clockwise"] = zone.offset_clockwise;
        offset["fade"] = clamp_int(zone.offset_fade_mode,
                                    int(TextureMappingZone::OffsetFadeNone),
                                    int(TextureMappingZone::OffsetFadeOutInReversed));
        offset["angle_reference"] = clamp_int(zone.offset_angle_mode,
                                               int(TextureMappingZone::OffsetAngleConfigured),
                                               int(TextureMappingZone::OffsetAngleObjectCenter));
        entry["surface_offset"] = std::move(offset);

        root.push_back(std::move(entry));
    }

    return root.empty() ? std::string() : root.dump();
}

void TextureMappingManager::load_entries(const std::string &serialized,
                                         const std::vector<std::string> &filament_colours)
{
    clear();
    refresh(filament_colours);

    const size_t n = filament_colours.size();
    if (serialized.empty() || n == 0)
        return;

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(serialized);
    } catch (const std::exception &e) {
        BOOST_LOG_TRIVIAL(warning) << "TextureMappingManager::load_entries JSON parse failed: " << e.what();
        return;
    }
    if (!root.is_array())
        return;

    std::unordered_set<uint64_t> used_stable_ids;
    size_t loaded_rows = 0;
    size_t skipped_rows = 0;

    auto dedupe_stable_id = [this, &used_stable_ids](uint64_t stable_id) {
        stable_id = normalize_stable_id(stable_id);
        if (used_stable_ids.insert(stable_id).second)
            return stable_id;
        uint64_t replacement = allocate_stable_id();
        used_stable_ids.insert(replacement);
        return replacement;
    };

    for (const nlohmann::json &entry : root) {
        if (!entry.is_object()) {
            ++skipped_rows;
            continue;
        }

        const int surface_pattern = surface_pattern_from_name(entry.value("surface_pattern", std::string("image_texture")));
        const bool linear_gradient = surface_pattern == int(TextureMappingZone::LinearGradient);
        std::vector<unsigned int> component_ids = ids_from_json(entry.value("component_filaments", nlohmann::json::array()), n);
        if (!linear_gradient && component_ids.size() < 2) {
            component_ids.clear();
            for (size_t i = 1; i <= std::min<size_t>(n, 9); ++i)
                component_ids.emplace_back(unsigned(i));
        }
        if (!linear_gradient && component_ids.size() < 2) {
            ++skipped_rows;
            continue;
        }
        std::vector<unsigned int> anchors = ids_from_json(entry.value("anchor_filaments", nlohmann::json::array()), n);
        if (anchors.size() < 2 && component_ids.size() >= 2)
            anchors = {component_ids[0], component_ids[1]};
        else if (anchors.empty()) {
            const unsigned int anchor_a = component_ids.empty() ? 1u : component_ids.front();
            const unsigned int anchor_b = component_ids.size() >= 2 ? component_ids[1] : (n >= 2 && anchor_a == 1 ? 2u : anchor_a);
            anchors = {anchor_a, anchor_b};
        }
        while (anchors.size() < 2)
            anchors.emplace_back(anchors.front());
        if (!linear_gradient && anchors[0] == anchors[1] && n >= 2)
            anchors[1] = anchors[0] == 1 ? 2 : 1;
        if (linear_gradient && component_ids.empty())
            component_ids = {anchors[0], anchors[1]};

        TextureMappingZone zone;
        zone.component_a = anchors[0];
        zone.component_b = anchors[1];
        zone.stable_id = dedupe_stable_id(entry.value("uid", uint64_t(0)));
        zone.zone_id = entry.value("zone_id", entry.value("filament_id", 0u));
        zone.enabled = entry.value("enabled", true);
        zone.deleted = false;
        zone.surface_pattern = surface_pattern;
        zone.component_ids = encode_component_ids(component_ids);
        zone.component_weights =
            weights_from_json(entry.value("component_weights_pct", nlohmann::json::array()), component_ids.size());
        zone.display_color = entry.value("display_color", std::string());
        if (zone.display_color.empty() || zone.display_color[0] != '#')
            zone.display_color = random_display_color(zone.stable_id);
        if (zone.is_linear_gradient()) {
            const nlohmann::json linear_gradient = entry.value("linear_gradient", nlohmann::json::object());
            zone.linear_gradient_start = linear_gradient_anchor_from_json(linear_gradient.value("start", nlohmann::json::object()));
            zone.linear_gradient_end = linear_gradient_anchor_from_json(linear_gradient.value("end", nlohmann::json::object()));
            zone.linear_gradient_stops = linear_gradient_stops_from_json(linear_gradient.value("stops", nlohmann::json::array()));
            zone.linear_gradient_stops = normalized_linear_gradient_stops(zone, n);
            component_ids = linear_gradient_component_ids_from_stops(zone, n);
            if (!zone.linear_gradient_stops.empty()) {
                zone.component_a = zone.linear_gradient_stops.front().filament_id;
                zone.component_b = zone.linear_gradient_stops.back().filament_id;
            }
            zone.component_ids = encode_component_ids(component_ids);
            zone.linear_gradient_mode = linear_gradient_mode_from_name(linear_gradient.value("mode", std::string("linear")));
            const float radius_mm = linear_gradient.value("radius_mm", TextureMappingZone::DefaultLinearGradientRadiusMm);
            zone.linear_gradient_radius_mm = std::isfinite(radius_mm) ? std::max(0.f, radius_mm) : TextureMappingZone::DefaultLinearGradientRadiusMm;
            zone.linear_gradient_radius_percent = linear_gradient.value("radius_percent", TextureMappingZone::DefaultLinearGradientRadiusPercent);
            const float radius_pct = linear_gradient.value("radius_pct", TextureMappingZone::DefaultLinearGradientRadiusPct);
            zone.linear_gradient_radius_pct = std::isfinite(radius_pct) ? std::max(0.f, radius_pct) : TextureMappingZone::DefaultLinearGradientRadiusPct;
            zone.show_linear_gradient_direction_arrow = linear_gradient.value("show_direction_arrow", true);
        }

        const nlohmann::json texture = entry.value("texture_options", nlohmann::json::object());
        zone.texture_mapping_mode = mapping_mode_from_name(texture.value("mode", std::string("target_color")));
        zone.filament_color_mode = color_model_from_name(texture.value("color_model", std::string("cmyk")));
        zone.force_sequential_filaments = texture.value("ordered_roles", false);
        zone.reduce_outer_surface_texture = false;
        zone.seam_hiding = texture.value("hide_seams", false);
        zone.nonlinear_offset_adjustment = texture.value("nonlinear_offset_adjustment", false);
        zone.use_modulated_overhang_geometry_for_support =
            texture.value("use_modulated_overhang_geometry_for_support",
                          TextureMappingZone::DefaultUseModulatedOverhangGeometryForSupport);
        const auto modulation_mode_it = texture.find("modulation_mode");
        const bool has_modulation_mode =
            modulation_mode_it != texture.end() && modulation_mode_it->is_string();
        zone.modulation_mode = has_modulation_mode ?
            modulation_mode_from_name(modulation_mode_it->get<std::string>()) :
            TextureMappingZone::default_modulation_mode_for_surface_pattern(zone.surface_pattern);
        zone.modulation_mode_manually_changed =
            texture.value("modulation_mode_manually_changed",
                          has_modulation_mode && zone.modulation_mode != TextureMappingZone::ModulationLineWidth);
        zone.apply_default_modulation_mode();
        zone.recolor_small_perimeter_loops =
            texture.value("recolor_small_perimeter_loops", TextureMappingZone::DefaultRecolorSmallPerimeterLoops);
        zone.recolor_top_visible_perimeter_sections =
            texture.value("recolor_top_visible_perimeter_sections", TextureMappingZone::DefaultRecolorTopVisiblePerimeterSections);
        if (zone.recolor_top_visible_perimeter_sections)
            zone.recolor_small_perimeter_loops = true;
        zone.top_visible_perimeter_recolor_aggressiveness =
            top_visible_recolor_aggressiveness_from_name(
                texture.value("top_visible_perimeter_recolor_aggressiveness",
                              top_visible_recolor_aggressiveness_name(TextureMappingZone::DefaultTopVisiblePerimeterRecolorAggressiveness)));
        zone.top_visible_perimeter_recolor_above_layers =
            clamp_int(texture.value("top_visible_perimeter_recolor_above_layers",
                                    TextureMappingZone::DefaultTopVisiblePerimeterRecolorAboveLayers),
                      TextureMappingZone::MinTopVisiblePerimeterRecolorAboveLayers,
                      TextureMappingZone::MaxTopVisiblePerimeterRecolorAboveLayers);
        zone.top_visible_perimeter_recolor_point_sampling =
            texture.value("top_visible_perimeter_recolor_point_sampling",
                          TextureMappingZone::DefaultTopVisiblePerimeterRecolorPointSampling);
        zone.top_surface_image_printing_enabled =
            texture.value("top_surface_image_printing_enabled",
                          TextureMappingZone::DefaultTopSurfaceImagePrintingEnabled);
        zone.top_surface_image_printing_method =
            top_surface_image_printing_method_from_name(
                texture.value("top_surface_image_printing_method",
                              top_surface_image_printing_method_name(TextureMappingZone::DefaultTopSurfaceImagePrintingMethod)));
        zone.top_surface_image_max_line_width_mm =
            std::clamp(finite_or(texture.value("top_surface_image_max_line_width_mm",
                                               TextureMappingZone::DefaultTopSurfaceImageMaxLineWidthMm),
                                 TextureMappingZone::DefaultTopSurfaceImageMaxLineWidthMm),
                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
        zone.top_surface_image_min_line_width_mm =
            std::clamp(finite_or(texture.value("top_surface_image_min_line_width_mm",
                                               TextureMappingZone::DefaultTopSurfaceImageMinLineWidthMm),
                                 TextureMappingZone::DefaultTopSurfaceImageMinLineWidthMm),
                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                       zone.top_surface_image_max_line_width_mm);
        zone.top_surface_image_colored_top_layers =
            clamp_int(texture.value("top_surface_image_colored_top_layers",
                                    TextureMappingZone::DefaultTopSurfaceImageColoredTopLayers),
                      TextureMappingZone::MinTopSurfaceImageColoredTopLayers,
                      TextureMappingZone::MaxTopSurfaceImageColoredTopLayers);
        zone.top_surface_image_fixed_coloring_filaments =
            texture.value("top_surface_image_fixed_coloring_filaments",
                          TextureMappingZone::DefaultTopSurfaceImageFixedColoringFilaments);
        zone.top_surface_contoning_angle_threshold_deg =
            std::clamp(finite_or(texture.value("top_surface_contoning_angle_threshold_deg",
                                               TextureMappingZone::DefaultTopSurfaceContoningAngleThresholdDeg),
                                 TextureMappingZone::DefaultTopSurfaceContoningAngleThresholdDeg),
                       TextureMappingZone::MinTopSurfaceContoningAngleThresholdDeg,
                       TextureMappingZone::MaxTopSurfaceContoningAngleThresholdDeg);
        zone.top_surface_contoning_stack_layers =
            clamp_int(texture.value("top_surface_contoning_stack_layers",
                                    TextureMappingZone::DefaultTopSurfaceContoningStackLayers),
                      TextureMappingZone::MinTopSurfaceContoningStackLayers,
                      TextureMappingZone::MaxTopSurfaceContoningStackLayers);
        zone.top_surface_contoning_min_feature_mm =
            std::clamp(finite_or(texture.value("top_surface_contoning_min_feature_mm",
                                               TextureMappingZone::DefaultTopSurfaceContoningMinFeatureMm),
                                 TextureMappingZone::DefaultTopSurfaceContoningMinFeatureMm),
                       TextureMappingZone::MinTopSurfaceContoningMinFeatureMm,
                       TextureMappingZone::MaxTopSurfaceContoningMinFeatureMm);
        zone.top_surface_contoning_color_lower_surfaces =
            texture.value("top_surface_contoning_color_lower_surfaces",
                          TextureMappingZone::DefaultTopSurfaceContoningColorLowerSurfaces);
        zone.top_surface_contoning_only_color_surface_infill =
            texture.value("top_surface_contoning_only_color_surface_infill",
                          TextureMappingZone::DefaultTopSurfaceContoningOnlyColorSurfaceInfill);
        zone.compact_offset_mode = texture.value("compact_offset_mode", TextureMappingZone::DefaultCompactOffsetMode);
        zone.use_legacy_fixed_color_mode =
            texture.value("use_legacy_fixed_color_mode", TextureMappingZone::DefaultUseLegacyFixedColorMode);
        zone.high_speed_image_texture_sampling = true;
        zone.minimum_visibility_offset_enabled =
            texture.value("minimum_visibility_offset_enabled", TextureMappingZone::DefaultMinimumVisibilityOffsetEnabled);
        zone.minimum_visibility_offset_pct =
            std::clamp(finite_or(texture.value("minimum_visibility_offset_pct", TextureMappingZone::DefaultMinimumVisibilityOffsetPct),
                                 TextureMappingZone::DefaultMinimumVisibilityOffsetPct),
                       0.f,
                       100.f);
        zone.generic_solver_lookup_mode =
            generic_solver_lookup_mode_from_name(texture.value("generic_solver_lookup", std::string("closest_mix")));
        const auto generic_solver_mode_it = texture.find("generic_solver_mode");
        zone.generic_solver_mode =
            generic_solver_mode_it != texture.end() && generic_solver_mode_it->is_string() ?
                generic_solver_mode_from_name(generic_solver_mode_it->get<std::string>()) :
                (zone.filament_color_mode == int(TextureMappingZone::FilamentColorAny) ?
                     int(TextureMappingZone::GenericSolverLegacy) :
                     int(TextureMappingZone::GenericSolverV2));
        zone.generic_solver_mix_model =
            generic_solver_mix_model_from_name(texture.value("generic_solver_mix_model", std::string("pigment_painter")));
        zone.dithering_enabled = texture.value("dithering_enabled", TextureMappingZone::DefaultDitheringEnabled);
        zone.dithering_method = dithering_method_from_name(
            texture.value("dithering_method", dithering_method_name(TextureMappingZone::DefaultDitheringMethod)));
        zone.dithering_resolution_mm =
            std::clamp(texture.value("dithering_resolution_mm", TextureMappingZone::DefaultDitheringResolutionMm),
                       TextureMappingZone::MinDitheringResolutionMm,
                       TextureMappingZone::MaxDitheringResolutionMm);
        zone.halftone_dot_size_mm =
            std::clamp(texture.value("halftone_dot_size_mm", TextureMappingZone::DefaultHalftoneDotSizeMm),
                       TextureMappingZone::MinHalftoneDotSizeMm,
                       TextureMappingZone::MaxHalftoneDotSizeMm);
        if (zone.dithering_enabled)
            zone.compact_offset_mode = true;
        zone.contrast_pct = std::clamp(texture.value("contrast_pct", 100.f), 25.f, 300.f);
        zone.high_resolution_sampling = true;
        zone.tone_gamma = normalize_tone_gamma(texture.value("tone_gamma", 1.f));
        zone.transmission_distance_calibration_mode = transmission_distance_calibration_mode_from_json(texture);
        zone.preview_opacity_pct =
            std::clamp(texture.value("preview_opacity_pct", TextureMappingZone::DefaultPreviewOpacityPct), 0.f, 100.f);
        zone.preview_simulate_colors = texture.value("simulate_preview_colors", TextureMappingZone::DefaultPreviewSimulateColors);
        zone.preview_limit_resolution = texture.value("limit_preview_resolution", true);
        zone.auto_adjust_filament_selection = texture.value("auto_adjust_filaments", true);
        zone.filament_strengths_pct = normalize_strengths(floats_from_json(texture.value("strength_pct", nlohmann::json::array())));
        zone.filament_minimum_offsets_pct =
            normalize_minimum_offsets(floats_from_json(texture.value("minimum_offset_pct", nlohmann::json::array())));
        zone.filament_transmission_distances_mm =
            normalize_transmission_distances(floats_from_json(texture.value("transmission_distance_mm", nlohmann::json::array())));

        const nlohmann::json offset = entry.value("surface_offset", nlohmann::json::object());
        zone.offset_distances =
            normalize_offset_distances(offset.value("distances_mm_by_filament", std::string()),
                                       component_ids.size(),
                                       max_component_surface_offset_mm());
        zone.offset_angles =
            normalize_offset_angles(offset.value("angles_deg_by_filament", std::string()), component_ids.size());
        zone.offset_mode = clamp_int(offset.value("control_mode", int(TextureMappingZone::OffsetBasic)),
                                     int(TextureMappingZone::OffsetBasic),
                                     int(TextureMappingZone::OffsetAdvanced));
        zone.offset_rotation_enabled = offset.value("rotate_with_height", true);
        zone.offset_rotations = offset.value("rotations", 1.f);
        zone.offset_repeats = offset.value("repeats", 1.f);
        zone.offset_reverse_repeats = offset.value("alternate_repeats", true);
        zone.offset_clockwise = offset.value("clockwise", true);
        zone.offset_fade_mode = clamp_int(offset.value("fade", int(TextureMappingZone::OffsetFadeNone)),
                                          int(TextureMappingZone::OffsetFadeNone),
                                          int(TextureMappingZone::OffsetFadeOutInReversed));
        zone.offset_angle_mode = clamp_int(offset.value("angle_reference", int(TextureMappingZone::OffsetAngleObjectCenter)),
                                           int(TextureMappingZone::OffsetAngleConfigured),
                                           int(TextureMappingZone::OffsetAngleObjectCenter));

        m_zones.emplace_back(std::move(zone));
        ++loaded_rows;
    }

    normalize_zone_ids(n);
    refresh(filament_colours);
    BOOST_LOG_TRIVIAL(info) << "TextureMappingManager::load_entries"
                            << ", physical_count=" << n
                            << ", loaded_rows=" << loaded_rows
                            << ", skipped_rows=" << skipped_rows;
}

int TextureMappingManager::zone_index_from_id(unsigned int zone_id) const
{
    for (size_t i = 0; i < m_zones.size(); ++i) {
        const TextureMappingZone &zone = m_zones[i];
        if (zone.enabled && !zone.deleted && zone.zone_id == zone_id)
            return int(i);
    }
    return -1;
}

unsigned int TextureMappingManager::zone_id_for_index(size_t zone_index) const
{
    if (zone_index >= m_zones.size())
        return 0;
    const TextureMappingZone &zone = m_zones[zone_index];
    return zone.enabled && !zone.deleted ? zone.zone_id : 0;
}

std::vector<unsigned int> TextureMappingManager::zone_ids_by_index() const
{
    std::vector<unsigned int> ids(m_zones.size(), 0);
    for (size_t i = 0; i < m_zones.size(); ++i)
        ids[i] = zone_id_for_index(i);
    return ids;
}

unsigned int TextureMappingManager::allocate_zone_id(size_t num_physical) const
{
    const unsigned int start = std::max<unsigned int>(TextureMappingZoneIdBase, unsigned(num_physical + 1));
    std::vector<bool> used(MaxTextureMappingZoneId + 1, false);
    for (const TextureMappingZone &zone : m_zones)
        if (!zone.deleted && zone.zone_id <= MaxTextureMappingZoneId)
            used[zone.zone_id] = true;

    for (unsigned int id = start; id <= MaxTextureMappingZoneId; ++id)
        if (!used[id])
            return id;
    return 0;
}

void TextureMappingManager::normalize_zone_ids(size_t num_physical)
{
    std::vector<bool> used(MaxTextureMappingZoneId + 1, false);
    const unsigned int minimum = std::max<unsigned int>(TextureMappingZoneIdBase, unsigned(num_physical + 1));

    auto reserve = [&](unsigned int requested) {
        if (requested >= minimum && requested <= MaxTextureMappingZoneId && !used[requested]) {
            used[requested] = true;
            return requested;
        }
        for (unsigned int id = minimum; id <= MaxTextureMappingZoneId; ++id) {
            if (used[id])
                continue;
            used[id] = true;
            return id;
        }
        return 0u;
    };

    for (TextureMappingZone &zone : m_zones) {
        if (!zone.enabled || zone.deleted) {
            zone.zone_id = 0;
            continue;
        }
        zone.zone_id = reserve(zone.zone_id);
    }
}

const TextureMappingZone *TextureMappingManager::zone_from_stable_id(uint64_t stable_id) const
{
    if (stable_id == 0)
        return nullptr;
    for (const TextureMappingZone &zone : m_zones)
        if (zone.enabled && !zone.deleted && zone.stable_id == stable_id)
            return &zone;
    return nullptr;
}

TextureMappingZone *TextureMappingManager::zone_from_stable_id(uint64_t stable_id)
{
    if (stable_id == 0)
        return nullptr;
    for (TextureMappingZone &zone : m_zones)
        if (zone.enabled && !zone.deleted && zone.stable_id == stable_id)
            return &zone;
    return nullptr;
}

const TextureMappingZone *TextureMappingManager::zone_from_id(unsigned int zone_id) const
{
    const int idx = zone_index_from_id(zone_id);
    return idx >= 0 ? &m_zones[size_t(idx)] : nullptr;
}

TextureMappingZone *TextureMappingManager::zone_from_id(unsigned int zone_id)
{
    const int idx = zone_index_from_id(zone_id);
    return idx >= 0 ? &m_zones[size_t(idx)] : nullptr;
}

unsigned int TextureMappingManager::resolve_zone_component(unsigned int zone_id, size_t num_physical, int layer_index) const
{
    const TextureMappingZone *zone = zone_from_id(zone_id);
    return zone == nullptr ? zone_id : resolve_zone_component(*zone, num_physical, m_filament_colours, layer_index);
}

size_t TextureMappingManager::total_filaments(size_t num_physical) const
{
    size_t total = num_physical;
    for (const TextureMappingZone &zone : m_zones)
        if (zone.enabled && !zone.deleted)
            total = std::max(total, size_t(zone.zone_id));
    return total;
}

std::vector<std::string> TextureMappingManager::display_colors(size_t num_physical) const
{
    const size_t total = total_filaments(num_physical);
    if (total <= num_physical)
        return {};

    std::vector<std::string> colors(total - num_physical, TextureMappingGapDisplayColor);
    for (const TextureMappingZone &zone : m_zones) {
        if (!zone.enabled || zone.deleted || zone.zone_id <= num_physical)
            continue;
        const size_t idx = size_t(zone.zone_id - unsigned(num_physical) - 1);
        if (idx < colors.size())
            colors[idx] = zone.display_color.empty() ? TextureMappingGapDisplayColor : zone.display_color;
    }
    return colors;
}

std::string TextureMappingManager::filament_color_mode_name(int filament_color_mode)
{
    return color_model_name(filament_color_mode);
}

size_t TextureMappingManager::expected_component_count(int mapping_mode, int filament_color_mode)
{
    const int clamped_mapping = clamp_int(mapping_mode,
                                          int(TextureMappingZone::TextureMappingFilamentBlending),
                                          int(TextureMappingZone::TextureMappingRawValues));
    if (clamped_mapping == int(TextureMappingZone::TextureMappingRawValues))
        return 0;

    switch (clamp_int(filament_color_mode,
                      int(TextureMappingZone::FilamentColorAny),
                      int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorRGB):
    case int(TextureMappingZone::FilamentColorCMY):
        return 3;
    case int(TextureMappingZone::FilamentColorCMYK):
    case int(TextureMappingZone::FilamentColorCMYW):
    case int(TextureMappingZone::FilamentColorRGBK):
    case int(TextureMappingZone::FilamentColorRGBW):
        return 4;
    case int(TextureMappingZone::FilamentColorBW):
        return 2;
    case int(TextureMappingZone::FilamentColorCMYKW):
    case int(TextureMappingZone::FilamentColorRGBKW):
        return 5;
    default:
        return 0;
    }
}

bool TextureMappingManager::component_count_mismatch(const TextureMappingZone &zone, size_t num_physical)
{
    const size_t expected = expected_component_count(zone.texture_mapping_mode, zone.filament_color_mode);
    return expected != 0 && selected_component_ids(zone, num_physical).size() != expected;
}

std::vector<TextureMappingZone::LinearGradientStop> TextureMappingManager::normalized_linear_gradient_stops(const TextureMappingZone &zone,
                                                                                                            size_t                    num_physical)
{
    const unsigned int max_id = unsigned(std::min<size_t>(num_physical, 9));
    std::vector<TextureMappingZone::LinearGradientStop> stops;
    if (max_id == 0)
        return stops;

    stops.reserve(zone.linear_gradient_stops.size());
    for (TextureMappingZone::LinearGradientStop stop : zone.linear_gradient_stops) {
        if (stop.filament_id < 1 || stop.filament_id > max_id)
            continue;
        stop.position = std::clamp(std::isfinite(stop.position) ? stop.position : 0.f, 0.f, 1.f);
        stops.emplace_back(stop);
    }

    if (stops.size() < 2) {
        std::vector<unsigned int> ids = decode_component_ids(zone.component_ids, num_physical);
        if (ids.size() < 2) {
            ids.clear();
            if (zone.component_a >= 1 && zone.component_a <= max_id)
                ids.emplace_back(zone.component_a);
            if (zone.component_b >= 1 && zone.component_b <= max_id)
                ids.emplace_back(zone.component_b);
        }
        if (ids.empty())
            ids.emplace_back(1);
        while (ids.size() < 2)
            ids.emplace_back(ids.front());
        stops = {
            {0.f, std::clamp(ids[0], 1u, max_id)},
            {1.f, std::clamp(ids[1], 1u, max_id)}
        };
    }

    std::stable_sort(stops.begin(), stops.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.position < rhs.position;
    });
    return stops;
}

std::vector<unsigned int> TextureMappingManager::linear_gradient_component_ids_from_stops(const TextureMappingZone &zone,
                                                                                          size_t                    num_physical)
{
    std::vector<unsigned int> ids;
    bool seen[10] = { false };
    for (const TextureMappingZone::LinearGradientStop &stop : normalized_linear_gradient_stops(zone, num_physical)) {
        const unsigned int id = stop.filament_id;
        if (id == 0 || id > num_physical || id > 9 || seen[id])
            continue;
        seen[id] = true;
        ids.emplace_back(id);
    }
    return ids;
}

std::vector<float> TextureMappingManager::linear_gradient_compact_weights(float t,
                                                                          const std::vector<TextureMappingZone::LinearGradientStop> &stops,
                                                                          const std::vector<unsigned int> &component_ids)
{
    std::vector<float> weights(component_ids.size(), 0.f);
    if (stops.empty() || component_ids.empty())
        return weights;

    auto add_weight = [&weights, &component_ids](unsigned int filament_id, float weight) {
        if (weight <= 0.f)
            return;
        for (size_t idx = 0; idx < component_ids.size(); ++idx) {
            if (component_ids[idx] == filament_id) {
                weights[idx] = std::max(weights[idx], std::clamp(weight, 0.f, 1.f));
                return;
            }
        }
    };

    const float clamped_t = std::clamp(std::isfinite(t) ? t : 0.f, 0.f, 1.f);
    if (clamped_t <= stops.front().position) {
        add_weight(stops.front().filament_id, 1.f);
        return weights;
    }
    if (clamped_t >= stops.back().position) {
        add_weight(stops.back().filament_id, 1.f);
        return weights;
    }

    size_t right = 1;
    while (right < stops.size() && clamped_t > stops[right].position)
        ++right;
    if (right >= stops.size()) {
        add_weight(stops.back().filament_id, 1.f);
        return weights;
    }

    const TextureMappingZone::LinearGradientStop &lhs = stops[right - 1];
    const TextureMappingZone::LinearGradientStop &rhs = stops[right];
    const float denom = rhs.position - lhs.position;
    if (lhs.filament_id == rhs.filament_id || denom <= 1e-6f) {
        add_weight(rhs.filament_id, 1.f);
        return weights;
    }

    const float local_t = std::clamp((clamped_t - lhs.position) / denom, 0.f, 1.f);
    float lhs_weight = 1.f - local_t;
    float rhs_weight = local_t;
    const float max_weight = std::max(lhs_weight, rhs_weight);
    if (max_weight > 1e-6f) {
        lhs_weight /= max_weight;
        rhs_weight /= max_weight;
    }
    add_weight(lhs.filament_id, lhs_weight);
    add_weight(rhs.filament_id, rhs_weight);
    return weights;
}

std::vector<unsigned int> TextureMappingManager::selected_component_ids(const TextureMappingZone &zone, size_t num_physical)
{
    if (zone.is_linear_gradient())
        return linear_gradient_component_ids_from_stops(zone, num_physical);

    std::vector<unsigned int> ids = decode_component_ids(zone.component_ids, num_physical);
    if (!ids.empty()) {
        return ids;
    }

    if (zone.component_a >= 1 && zone.component_a <= num_physical)
        ids.emplace_back(zone.component_a);
    if (zone.component_b >= 1 && zone.component_b <= num_physical && zone.component_b != zone.component_a)
        ids.emplace_back(zone.component_b);
    return ids;
}

std::vector<unsigned int> TextureMappingManager::effective_texture_component_ids(const TextureMappingZone      &zone,
                                                                                 size_t                         num_physical,
                                                                                 const std::vector<std::string> &filament_colours)
{
    std::vector<unsigned int> selected = selected_component_ids(zone, num_physical);
    const size_t expected = expected_component_count(zone.texture_mapping_mode, zone.filament_color_mode);
    if (expected == 0)
        return selected;

    const std::vector<RGB> roles = semantic_colors(zone.filament_color_mode);
    if (roles.size() != expected)
        return selected;

    std::vector<unsigned int> result;
    result.reserve(expected);
    std::vector<bool> used(num_physical + 1, false);

    auto choose_unused_physical = [&](const RGB &target) {
        unsigned int best_id = 0;
        float best_distance = std::numeric_limits<float>::max();
        for (unsigned int id = 1; id <= num_physical; ++id) {
            if (id < used.size() && used[id])
                continue;
            const float distance = perceptual_color_distance_sq(filament_color(id, filament_colours), target);
            if (distance < best_distance) {
                best_distance = distance;
                best_id = id;
            }
        }
        if (best_id != 0 && best_id < used.size())
            used[best_id] = true;
        return best_id;
    };

    if (zone.force_sequential_filaments) {
        for (const unsigned int id : selected) {
            if (result.size() >= expected)
                break;
            if (id >= 1 && id <= num_physical && id < used.size() && !used[id]) {
                used[id] = true;
                result.emplace_back(id);
            }
        }
        for (size_t role_idx = result.size(); role_idx < expected; ++role_idx) {
            const unsigned int id = choose_unused_physical(roles[role_idx]);
            if (id != 0)
                result.emplace_back(id);
        }
        return result;
    }

    std::vector<bool> selected_used(selected.size(), false);
    for (const RGB &role : roles) {
        size_t best_selected = selected.size();
        float best_distance = std::numeric_limits<float>::max();
        for (size_t i = 0; i < selected.size(); ++i) {
            const unsigned int id = selected[i];
            if (selected_used[i] || id < 1 || id > num_physical)
                continue;
            const float distance = perceptual_color_distance_sq(filament_color(id, filament_colours), role);
            if (distance < best_distance) {
                best_distance = distance;
                best_selected = i;
            }
        }

        unsigned int id = 0;
        if (best_selected < selected.size()) {
            id = selected[best_selected];
            selected_used[best_selected] = true;
            if (id < used.size())
                used[id] = true;
        } else {
            id = choose_unused_physical(role);
        }
        if (id != 0)
            result.emplace_back(id);
    }

    return result;
}

bool TextureMappingManager::auto_adjust_texture_component_ids(TextureMappingZone            &zone,
                                                              size_t                         num_physical,
                                                              const std::vector<std::string> &filament_colours)
{
    if (!zone.enabled || zone.deleted || !zone.is_image_texture() || !zone.auto_adjust_filament_selection || num_physical < 2)
        return false;

    const size_t expected = expected_component_count(zone.texture_mapping_mode, zone.filament_color_mode);
    if (expected == 0)
        return false;

    const std::vector<unsigned int> adjusted = effective_texture_component_ids(zone, num_physical, filament_colours);
    if (adjusted.size() < 2)
        return false;

    const std::string encoded = encode_component_ids(adjusted);
    if (encoded.empty())
        return false;

    const unsigned int component_a = adjusted[0];
    const unsigned int component_b = adjusted.size() > 1 ? adjusted[1] : adjusted[0];
    if (zone.component_ids == encoded && zone.component_a == component_a && zone.component_b == component_b)
        return false;

    zone.component_a = component_a;
    zone.component_b = component_b;
    zone.component_ids = encoded;
    zone.component_weights = normalize_weights(zone.component_weights, adjusted.size());
    return true;
}

std::vector<TextureMappingColorMatch> TextureMappingManager::texture_component_color_matches(
    const TextureMappingZone      &zone,
    size_t                         num_physical,
    const std::vector<std::string> &filament_colours)
{
    if (!zone.enabled || zone.deleted || !zone.is_image_texture() || num_physical == 0)
        return {};

    const size_t expected = expected_component_count(zone.texture_mapping_mode, zone.filament_color_mode);
    if (expected == 0)
        return {};

    const int filament_color_mode = clamp_int(zone.filament_color_mode,
                                              int(TextureMappingZone::FilamentColorAny),
                                              int(TextureMappingZone::FilamentColorRGBKW));
    if (filament_color_mode == int(TextureMappingZone::FilamentColorBW))
        return {};

    const std::vector<RGB> roles = semantic_colors(zone.filament_color_mode);
    const std::vector<std::string> role_names = semantic_color_names(zone.filament_color_mode);
    if (roles.size() != expected || role_names.size() != expected)
        return {};

    const std::vector<unsigned int> component_ids = effective_texture_component_ids(zone, num_physical, filament_colours);
    const size_t count = std::min(expected, component_ids.size());
    std::vector<TextureMappingColorMatch> matches;
    matches.reserve(count);
    for (size_t idx = 0; idx < count; ++idx) {
        const unsigned int id = component_ids[idx];
        if (id < 1 || id > num_physical)
            continue;
        const float distance = std::sqrt(perceptual_color_distance_sq(filament_color(id, filament_colours), roles[idx]));
        matches.push_back({id, role_names[idx], distance});
    }
    return matches;
}

float TextureMappingManager::poor_color_match_distance()
{
    return TextureMappingPoorColorMatchDistance;
}

float TextureMappingManager::max_component_surface_offset_mm(float reference_width_mm)
{
    const float safe_reference = std::max(0.05f, std::abs(reference_width_mm));
    return std::clamp(safe_reference, 0.01f, 0.35f);
}

std::vector<float> TextureMappingManager::default_offset_distances(size_t component_count, float reference_width_mm)
{
    return std::vector<float>(component_count, max_component_surface_offset_mm(reference_width_mm));
}

std::vector<float> TextureMappingManager::default_offset_angles(size_t component_count)
{
    std::vector<float> angles(component_count, 0.f);
    for (size_t i = 0; i < component_count; ++i)
        angles[i] = (360.f * float(i)) / std::max<float>(1.f, float(component_count));
    return angles;
}

std::vector<float> TextureMappingManager::effective_offset_distances(const TextureMappingZone &zone,
                                                                     size_t                    component_count,
                                                                     float                     reference_width_mm)
{
    const float max_distance = max_component_surface_offset_mm(reference_width_mm);
    std::vector<float> distances = decode_offset_distances(zone.offset_distances, component_count, max_distance);
    return distances.size() == component_count ? distances : default_offset_distances(component_count, reference_width_mm);
}

std::vector<float> TextureMappingManager::effective_offset_angles(const TextureMappingZone &zone, size_t component_count)
{
    std::vector<float> angles = decode_offset_angles(zone.offset_angles, component_count);
    return angles.size() == component_count ? angles : default_offset_angles(component_count);
}

unsigned int TextureMappingManager::resolve_zone_component(const TextureMappingZone      &zone,
                                                           size_t                         num_physical,
                                                           const std::vector<std::string> &filament_colours,
                                                           int                            layer_index)
{
    if (!zone.enabled || zone.deleted || num_physical == 0)
        return 0;

    std::vector<unsigned int> component_ids = zone.is_image_texture() ?
        effective_texture_component_ids(zone, num_physical, filament_colours) :
        selected_component_ids(zone, num_physical);
    component_ids.erase(std::remove_if(component_ids.begin(),
                                       component_ids.end(),
                                       [num_physical](unsigned int id) { return id == 0 || id > num_physical; }),
                        component_ids.end());
    component_ids.erase(std::unique(component_ids.begin(), component_ids.end()), component_ids.end());
    if (component_ids.empty())
        return 0;

    const std::vector<int> weights = decode_weights(zone.component_weights, component_ids.size());
    const std::vector<unsigned int> sequence =
        build_balanced_component_sequence(component_ids,
                                          weights.empty() ? std::vector<int>(component_ids.size(), 1) : weights);
    if (sequence.empty())
        return component_ids.front();

    return sequence[size_t(safe_mod(layer_index, int(sequence.size())))];
}

std::string TextureMappingManager::blend_color_multi(const std::vector<std::pair<std::string, int>> &color_percents)
{
    if (color_percents.empty())
        return "#000000";

    std::vector<std::array<float, 3>> colors;
    std::vector<int> weights;
    int total_pct = 0;
    for (const auto &[hex, pct] : color_percents) {
        if (pct <= 0)
            continue;
        const RGB color = parse_hex_color(hex);
        colors.push_back({float(color.r) / 255.f, float(color.g) / 255.f, float(color.b) / 255.f});
        weights.emplace_back(pct);
        total_pct += pct;
    }
    if (colors.empty() || total_pct <= 0)
        return "#000000";

    const std::array<float, 3> rgb = mix_color_solver_components(colors, weights, ColorSolverMixModel::PigmentPainter);
    return rgb_to_hex({std::clamp(int(std::lround(rgb[0] * 255.f)), 0, 255),
                       std::clamp(int(std::lround(rgb[1] * 255.f)), 0, 255),
                       std::clamp(int(std::lround(rgb[2] * 255.f)), 0, 255)});
}

} // namespace Slic3r
