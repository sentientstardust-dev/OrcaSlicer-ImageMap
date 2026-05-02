#include "TextureMapping.hpp"
#include "filament_mixer.h"

#include <boost/log/trivial.hpp>
#include <cctype>
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

static float normalize_tone_gamma(float value)
{
    return (!std::isfinite(value) || value <= 0.f) ? 1.f : std::clamp(value, 0.5f, 3.f);
}

static float normalize_sagging_ratio(float value)
{
    return std::isfinite(value) ? std::clamp(value, 0.f, 6.f) : 0.f;
}

static RGB filament_color(unsigned int id, const std::vector<std::string> &filament_colours)
{
    if (id >= 1 && size_t(id - 1) < filament_colours.size())
        return parse_hex_color(filament_colours[size_t(id - 1)]);
    return {};
}

static float color_distance_sq(const RGB &lhs, const RGB &rhs)
{
    const float dr = float(lhs.r - rhs.r);
    const float dg = float(lhs.g - rhs.g);
    const float db = float(lhs.b - rhs.b);
    return dr * dr + dg * dg + db * db;
}

static std::vector<RGB> semantic_colors(int filament_color_mode)
{
    switch (clamp_int(filament_color_mode,
                      int(TextureMappingZone::FilamentColorAny),
                      int(TextureMappingZone::FilamentColorBW))) {
    case int(TextureMappingZone::FilamentColorRGB):  return {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
    case int(TextureMappingZone::FilamentColorCMY):  return {{0, 255, 255}, {255, 0, 255}, {255, 255, 0}};
    case int(TextureMappingZone::FilamentColorCMYK): return {{0, 255, 255}, {255, 0, 255}, {255, 255, 0}, {0, 0, 0}};
    case int(TextureMappingZone::FilamentColorCMYW): return {{0, 255, 255}, {255, 0, 255}, {255, 255, 0}, {255, 255, 255}};
    case int(TextureMappingZone::FilamentColorRGBK): return {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {0, 0, 0}};
    case int(TextureMappingZone::FilamentColorRGBW): return {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}};
    case int(TextureMappingZone::FilamentColorBW):   return {{0, 0, 0}, {255, 255, 255}};
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

static std::string color_model_name(int mode)
{
    switch (clamp_int(mode, int(TextureMappingZone::FilamentColorAny), int(TextureMappingZone::FilamentColorBW))) {
    case int(TextureMappingZone::FilamentColorRGB):  return "rgb";
    case int(TextureMappingZone::FilamentColorCMY):  return "cmy";
    case int(TextureMappingZone::FilamentColorCMYK): return "cmyk";
    case int(TextureMappingZone::FilamentColorCMYW): return "cmyw";
    case int(TextureMappingZone::FilamentColorRGBK): return "rgbk";
    case int(TextureMappingZone::FilamentColorRGBW): return "rgbw";
    case int(TextureMappingZone::FilamentColorBW):   return "bw";
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

static std::string normalized_prime_tower_color_mode_name(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (name == "auto" || name.empty())
        return "auto";
    if (name == "generic" || name == "generic_solver" || name == "solver")
        return "generic_solver";
    if (name == "rgb" || name == "cmy" || name == "cmyk" || name == "cmyw" ||
        name == "rgbk" || name == "rgbw" || name == "bw")
        return name;
    return "auto";
}

} // namespace

std::string TextureMappingGlobalSettings::serialize() const
{
    const std::string normalized_mode = normalize_color_mode_name(prime_tower_color_mode);
    if (!enabled &&
        std::abs((std::isfinite(angle_offset_deg) ? angle_offset_deg : 0.f)) <= 1e-6f &&
        normalized_mode == "auto" &&
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
    prime_tower_texture_mapping["prime_tower_color_mode"] = normalized_mode;
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
    prime_tower_color_mode = normalize_color_mode_name(prime_tower_texture_mapping.value("prime_tower_color_mode", std::string("auto")));
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
           compact_offset_mode == rhs.compact_offset_mode &&
           use_legacy_fixed_color_mode == rhs.use_legacy_fixed_color_mode &&
           generic_solver_lookup_mode == rhs.generic_solver_lookup_mode &&
           generic_solver_mode == rhs.generic_solver_mode &&
           std::abs(contrast_pct - rhs.contrast_pct) <= eps &&
           high_resolution_sampling == rhs.high_resolution_sampling &&
           std::abs(tone_gamma - rhs.tone_gamma) <= eps &&
           std::abs(sagging_ratio - rhs.sagging_ratio) <= eps &&
           std::abs(preview_opacity_pct - rhs.preview_opacity_pct) <= eps &&
           preview_simulate_colors == rhs.preview_simulate_colors &&
           preview_limit_resolution == rhs.preview_limit_resolution &&
           auto_adjust_filament_selection == rhs.auto_adjust_filament_selection &&
           floats_equal(filament_strengths_pct, rhs.filament_strengths_pct) &&
           floats_equal(filament_minimum_offsets_pct, rhs.filament_minimum_offsets_pct);
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
        if (new_physical_count < 2)
            zone.enabled = false;
        zone.component_ids = encode_component_ids(ids);

        auto remove_index = [deleted_filament_id](std::vector<float> &values) {
            if (deleted_filament_id >= 1 && size_t(deleted_filament_id - 1) < values.size())
                values.erase(values.begin() + ptrdiff_t(deleted_filament_id - 1));
        };
        remove_index(zone.filament_strengths_pct);
        remove_index(zone.filament_minimum_offsets_pct);
    }
    normalize_zone_ids(new_physical_count);
}

TextureMappingZone *TextureMappingManager::add_zone(size_t num_physical,
                                                    const std::vector<std::string> &filament_colours,
                                                    int surface_pattern)
{
    if (num_physical < 2)
        return nullptr;

    TextureMappingZone zone;
    zone.stable_id = allocate_stable_id();
    zone.zone_id = allocate_zone_id(num_physical);
    if (zone.zone_id == 0)
        return nullptr;
    zone.surface_pattern = surface_pattern == int(TextureMappingZone::Gradient2D) ?
        int(TextureMappingZone::Gradient2D) :
        int(TextureMappingZone::ImageTexture);

    std::vector<unsigned int> ids;
    for (size_t i = 1; i <= std::min<size_t>(num_physical, 9); ++i)
        ids.emplace_back(unsigned(i));
    zone.component_ids = encode_component_ids(ids);
    zone.component_a = ids[0];
    zone.component_b = ids.size() > 1 ? ids[1] : ids[0];
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
    if (zone_index >= m_zones.size() || num_physical < 2)
        return false;

    TextureMappingZone copy = m_zones[zone_index];
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

unsigned int TextureMappingManager::find_image_texture_zone_id(size_t) const
{
    for (const TextureMappingZone &zone : m_zones)
        if (zone.enabled && !zone.deleted && zone.is_image_texture() && zone.zone_id != 0)
            return zone.zone_id;
    return 0;
}

unsigned int TextureMappingManager::ensure_image_texture_zone(size_t num_physical,
                                                              const std::vector<std::string> &filament_colours)
{
    if (unsigned int existing = find_image_texture_zone_id(num_physical); existing != 0)
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

        std::vector<unsigned int> component_ids = decode_component_ids(zone.component_ids, 9);
        if (component_ids.size() < 2) {
            component_ids = {zone.component_a, zone.component_b};
            component_ids.erase(std::remove_if(component_ids.begin(), component_ids.end(), [](unsigned int id) {
                return id == 0 || id > 9;
            }), component_ids.end());
        }

        const std::string normalized_weights = normalize_weights(zone.component_weights, component_ids.size());
        const std::string normalized_distances =
            normalize_offset_distances(zone.offset_distances, component_ids.size(), max_component_surface_offset_mm());
        const std::string normalized_angles = normalize_offset_angles(zone.offset_angles, component_ids.size());
        const std::vector<float> normalized_strengths = normalize_strengths(zone.filament_strengths_pct);
        const std::vector<float> normalized_min_offsets = normalize_minimum_offsets(zone.filament_minimum_offsets_pct);

        nlohmann::json entry;
        entry["schema"] = 2;
        entry["uid"] = zone.stable_id;
        entry["zone_id"] = zone.zone_id;
        entry["enabled"] = zone.enabled;
        entry["surface_pattern"] = zone.is_2d_gradient() ? "2d_gradient" : "image_texture";
        entry["anchor_filaments"] = {zone.component_a, zone.component_b};
        entry["component_filaments"] = ids_to_json(component_ids);
        entry["component_weights_pct"] = weights_to_json(normalized_weights, component_ids.size());
        entry["display_color"] = zone.display_color;

        nlohmann::json texture;
        texture["mode"] = mapping_mode_name(clamp_int(zone.texture_mapping_mode,
                                                       int(TextureMappingZone::TextureMappingFilamentBlending),
                                                       int(TextureMappingZone::TextureMappingRawValues)));
        texture["color_model"] = color_model_name(zone.filament_color_mode);
        texture["ordered_roles"] = zone.force_sequential_filaments;
        texture["reduce_outer_surface_texture"] = zone.reduce_outer_surface_texture;
        texture["hide_seams"] = zone.seam_hiding;
        texture["nonlinear_offset_adjustment"] = zone.nonlinear_offset_adjustment;
        texture["compact_offset_mode"] = zone.compact_offset_mode;
        texture["use_legacy_fixed_color_mode"] = zone.use_legacy_fixed_color_mode;
        texture["generic_solver_lookup"] = generic_solver_lookup_mode_name(zone.generic_solver_lookup_mode);
        texture["generic_solver_mode"] = generic_solver_mode_name(zone.generic_solver_mode);
        texture["contrast_pct"] = std::clamp(finite_or(zone.contrast_pct, 100.f), 25.f, 300.f);
        texture["high_resolution_sampling"] = zone.high_resolution_sampling;
        texture["tone_gamma"] = normalize_tone_gamma(zone.tone_gamma);
        texture["sagging_ratio"] = normalize_sagging_ratio(zone.sagging_ratio);
        texture["preview_opacity_pct"] = std::clamp(finite_or(zone.preview_opacity_pct, TextureMappingZone::DefaultPreviewOpacityPct), 0.f, 100.f);
        texture["simulate_preview_colors"] = zone.preview_simulate_colors;
        texture["limit_preview_resolution"] = zone.preview_limit_resolution;
        texture["auto_adjust_filaments"] = zone.auto_adjust_filament_selection;
        texture["strength_pct"] = floats_to_json(normalized_strengths);
        texture["minimum_offset_pct"] = floats_to_json(normalized_min_offsets);
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
    if (serialized.empty() || n < 2)
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

        std::vector<unsigned int> component_ids = ids_from_json(entry.value("component_filaments", nlohmann::json::array()), n);
        if (component_ids.size() < 2) {
            component_ids.clear();
            for (size_t i = 1; i <= std::min<size_t>(n, 9); ++i)
                component_ids.emplace_back(unsigned(i));
        }
        if (component_ids.size() < 2) {
            ++skipped_rows;
            continue;
        }

        std::vector<unsigned int> anchors = ids_from_json(entry.value("anchor_filaments", nlohmann::json::array()), n);
        if (anchors.size() < 2)
            anchors = {component_ids[0], component_ids[1]};
        if (anchors[0] == anchors[1])
            anchors[1] = anchors[0] == 1 ? 2 : 1;

        TextureMappingZone zone;
        zone.component_a = anchors[0];
        zone.component_b = anchors[1];
        zone.stable_id = dedupe_stable_id(entry.value("uid", uint64_t(0)));
        zone.zone_id = entry.value("zone_id", entry.value("filament_id", 0u));
        zone.enabled = entry.value("enabled", true);
        zone.deleted = false;
        zone.surface_pattern =
            entry.value("surface_pattern", std::string("image_texture")) == "2d_gradient" ||
            entry.value("surface_pattern", std::string("image_texture")) == "surface_gradient" ?
                int(TextureMappingZone::Gradient2D) :
                int(TextureMappingZone::ImageTexture);
        zone.component_ids = encode_component_ids(component_ids);
        zone.component_weights =
            weights_from_json(entry.value("component_weights_pct", nlohmann::json::array()), component_ids.size());
        zone.display_color = entry.value("display_color", std::string());
        if (zone.display_color.empty() || zone.display_color[0] != '#')
            zone.display_color = random_display_color(zone.stable_id);

        const nlohmann::json texture = entry.value("texture_options", nlohmann::json::object());
        zone.texture_mapping_mode = mapping_mode_from_name(texture.value("mode", std::string("target_color")));
        zone.filament_color_mode = color_model_from_name(texture.value("color_model", std::string("cmyk")));
        zone.force_sequential_filaments = texture.value("ordered_roles", false);
        zone.reduce_outer_surface_texture = texture.value("reduce_outer_surface_texture", false);
        zone.seam_hiding = texture.value("hide_seams", false);
        zone.nonlinear_offset_adjustment = texture.value("nonlinear_offset_adjustment", false);
        zone.compact_offset_mode = texture.value("compact_offset_mode", TextureMappingZone::DefaultCompactOffsetMode);
        zone.use_legacy_fixed_color_mode =
            texture.value("use_legacy_fixed_color_mode", TextureMappingZone::DefaultUseLegacyFixedColorMode);
        zone.generic_solver_lookup_mode =
            generic_solver_lookup_mode_from_name(texture.value("generic_solver_lookup", std::string("closest_mix")));
        const auto generic_solver_mode_it = texture.find("generic_solver_mode");
        zone.generic_solver_mode =
            generic_solver_mode_it != texture.end() && generic_solver_mode_it->is_string() ?
                generic_solver_mode_from_name(generic_solver_mode_it->get<std::string>()) :
                (zone.filament_color_mode == int(TextureMappingZone::FilamentColorAny) ?
                     int(TextureMappingZone::GenericSolverLegacy) :
                     int(TextureMappingZone::GenericSolverV2));
        zone.contrast_pct = std::clamp(texture.value("contrast_pct", 100.f), 25.f, 300.f);
        zone.high_resolution_sampling = texture.value("high_resolution_sampling", true);
        zone.tone_gamma = normalize_tone_gamma(texture.value("tone_gamma", 1.f));
        zone.sagging_ratio = normalize_sagging_ratio(texture.value("sagging_ratio", 0.f));
        zone.preview_opacity_pct = std::clamp(texture.value("preview_opacity_pct", TextureMappingZone::DefaultPreviewOpacityPct), 0.f, 100.f);
        zone.preview_simulate_colors = texture.value("simulate_preview_colors", false);
        zone.preview_limit_resolution = texture.value("limit_preview_resolution", true);
        zone.auto_adjust_filament_selection = texture.value("auto_adjust_filaments", true);
        zone.filament_strengths_pct = normalize_strengths(floats_from_json(texture.value("strength_pct", nlohmann::json::array())));
        zone.filament_minimum_offsets_pct = normalize_minimum_offsets(floats_from_json(texture.value("minimum_offset_pct", nlohmann::json::array())));

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
                      int(TextureMappingZone::FilamentColorBW))) {
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
    default:
        return 0;
    }
}

bool TextureMappingManager::component_count_mismatch(const TextureMappingZone &zone, size_t num_physical)
{
    const size_t expected = expected_component_count(zone.texture_mapping_mode, zone.filament_color_mode);
    return expected != 0 && selected_component_ids(zone, num_physical).size() != expected;
}

std::vector<unsigned int> TextureMappingManager::selected_component_ids(const TextureMappingZone &zone, size_t num_physical)
{
    std::vector<unsigned int> ids = decode_component_ids(zone.component_ids, num_physical);
    if (!ids.empty())
        return ids;

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
            const float distance = color_distance_sq(filament_color(id, filament_colours), target);
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
            const float distance = color_distance_sq(filament_color(id, filament_colours), role);
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

    struct WeightedColor {
        RGB color;
        int pct = 0;
    };

    std::vector<WeightedColor> colors;
    int total_pct = 0;
    for (const auto &[hex, pct] : color_percents) {
        if (pct <= 0)
            continue;
        colors.push_back({parse_hex_color(hex), pct});
        total_pct += pct;
    }
    if (colors.empty() || total_pct <= 0)
        return "#000000";

    unsigned char r = static_cast<unsigned char>(colors.front().color.r);
    unsigned char g = static_cast<unsigned char>(colors.front().color.g);
    unsigned char b = static_cast<unsigned char>(colors.front().color.b);
    int accumulated = colors.front().pct;

    for (size_t i = 1; i < colors.size(); ++i) {
        const int new_total = accumulated + colors[i].pct;
        if (new_total <= 0)
            continue;
        const float t = float(colors[i].pct) / float(new_total);
        filament_mixer_lerp(r, g, b,
                            static_cast<unsigned char>(colors[i].color.r),
                            static_cast<unsigned char>(colors[i].color.g),
                            static_cast<unsigned char>(colors[i].color.b),
                            t, &r, &g, &b);
        accumulated = new_total;
    }

    return rgb_to_hex({int(r), int(g), int(b)});
}

} // namespace Slic3r
