// original author: sentientstardust

#include "ImageMapRawFilamentOffsetAtlas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

namespace Slic3r {

namespace {

constexpr const char *Magic = "imagemap_raw_filament_offset";
constexpr size_t MagicSize = 28;
constexpr size_t FixedHeaderSize = 38;

static void set_error(std::string *error, const std::string &message)
{
    if (error != nullptr)
        *error = message;
}

static uint32_t read_be_u32(const std::vector<uint8_t> &bytes, size_t offset)
{
    return (uint32_t(bytes[offset]) << 24) |
           (uint32_t(bytes[offset + 1]) << 16) |
           (uint32_t(bytes[offset + 2]) << 8) |
           uint32_t(bytes[offset + 3]);
}

static void append_be_u32(std::vector<uint8_t> &bytes, uint32_t value)
{
    bytes.emplace_back(uint8_t((value >> 24) & 0xFFu));
    bytes.emplace_back(uint8_t((value >> 16) & 0xFFu));
    bytes.emplace_back(uint8_t((value >> 8) & 0xFFu));
    bytes.emplace_back(uint8_t(value & 0xFFu));
}

static bool read_header_bytes(const std::vector<uint8_t> &rgba,
                              uint32_t width,
                              uint32_t height,
                              size_t byte_count,
                              std::vector<uint8_t> &bytes)
{
    if (byte_count > std::numeric_limits<size_t>::max() / 8)
        return false;
    if (size_t(width) * size_t(height) < byte_count * 8)
        return false;

    bytes.assign(byte_count, 0);
    for (size_t byte_idx = 0; byte_idx < byte_count; ++byte_idx) {
        uint8_t value = 0;
        for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
            const size_t pixel_idx = byte_idx * 8 + bit_idx;
            const size_t rgba_idx = pixel_idx * 4;
            if (rgba_idx + 2 >= rgba.size())
                return false;
            const unsigned int average = (unsigned(rgba[rgba_idx + 0]) + unsigned(rgba[rgba_idx + 1]) + unsigned(rgba[rgba_idx + 2])) / 3u;
            value = uint8_t((value << 1) | (average >= 128u ? 1u : 0u));
        }
        bytes[byte_idx] = value;
    }
    return true;
}

static void write_header_bytes(std::vector<uint8_t> &rgba, const std::vector<uint8_t> &bytes)
{
    for (size_t byte_idx = 0; byte_idx < bytes.size(); ++byte_idx) {
        const uint8_t value = bytes[byte_idx];
        for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
            const bool bit = (value & (uint8_t(1) << (7 - bit_idx))) != 0;
            const size_t rgba_idx = (byte_idx * 8 + bit_idx) * 4;
            const uint8_t channel = bit ? 255 : 0;
            rgba[rgba_idx + 0] = channel;
            rgba[rgba_idx + 1] = channel;
            rgba[rgba_idx + 2] = channel;
            rgba[rgba_idx + 3] = 255;
        }
    }
}

static std::string uppercase_ascii(std::string value)
{
    for (char &ch : value)
        if (ch >= 'a' && ch <= 'z')
            ch = char(ch - 'a' + 'A');
    return value;
}

static std::string standard_hex_for_color_code(const std::string &color)
{
    const std::string key = uppercase_ascii(color);
    if (key == "C") return "#00FFFF";
    if (key == "M") return "#FF00FF";
    if (key == "Y") return "#FFFF00";
    if (key == "K") return "#000000";
    if (key == "W") return "#FFFFFF";
    if (key == "R") return "#FF0000";
    if (key == "G") return "#00FF00";
    if (key == "B") return "#0000FF";
    return "#FFFFFF";
}

static int hex_digit_value(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    return -1;
}

static std::array<uint8_t, 3> rgb_from_hex_string(const std::string &hex)
{
    if (hex.size() != 7 || hex[0] != '#')
        return { 255, 255, 255 };
    std::array<uint8_t, 3> out { 255, 255, 255 };
    for (size_t channel = 0; channel < 3; ++channel) {
        const int hi = hex_digit_value(hex[1 + channel * 2]);
        const int lo = hex_digit_value(hex[2 + channel * 2]);
        if (hi < 0 || lo < 0)
            return { 255, 255, 255 };
        out[channel] = uint8_t((hi << 4) | lo);
    }
    return out;
}

static std::array<uint8_t, 3> preview_rgb_for_filament_slot(const std::vector<ImageMapRawFilament> &filaments,
                                                            unsigned int slot)
{
    for (const ImageMapRawFilament &filament : filaments) {
        if (filament.slot != slot)
            continue;
        if (!filament.hex.empty())
            return rgb_from_hex_string(filament.hex);
        return rgb_from_hex_string(standard_hex_for_color_code(filament.color));
    }
    return { 255, 255, 255 };
}

static ImageMapRawExpectedLineWidth expected_line_width_from_json(const nlohmann::json &root)
{
    ImageMapRawExpectedLineWidth expected;
    const auto it = root.find("expected_line_width_mm");
    if (it == root.end() || !it->is_object())
        return expected;
    const nlohmann::json &entry = *it;
    const auto min_it = entry.find("min");
    const auto max_it = entry.find("max");
    if (min_it == entry.end() || max_it == entry.end() || !min_it->is_number() || !max_it->is_number())
        return expected;
    expected.min_mm = min_it->get<double>();
    expected.max_mm = max_it->get<double>();
    if (!std::isfinite(expected.min_mm) || !std::isfinite(expected.max_mm) || expected.min_mm <= 0.0 || expected.max_mm <= 0.0 ||
        expected.max_mm < expected.min_mm) {
        expected = {};
        return expected;
    }
    const auto warn_it = entry.find("warn_if_differs");
    expected.warn_if_differs = warn_it != entry.end() && warn_it->is_boolean() ? warn_it->get<bool>() : false;
    expected.valid = true;
    return expected;
}

static ImageMapRawExpectedLineWidth expected_line_width_from_metadata_json(const std::string &metadata_json)
{
    try {
        const nlohmann::json root = nlohmann::json::parse(metadata_json);
        if (root.is_object())
            return expected_line_width_from_json(root);
    } catch (...) {
    }
    return {};
}

static bool top_surface_layer_valid(const ImageMapRawTopSurfaceLayer &layer, uint32_t width, uint32_t height)
{
    return layer.depth >= 0 &&
           layer.filament_slots.size() >= size_t(width) * size_t(height);
}

static bool has_valid_offset_payload(const ImageMapRawFilamentOffsetAtlas &atlas)
{
    return atlas.channels > 0 &&
           atlas.offsets.size() >= size_t(atlas.width) * size_t(atlas.height) * size_t(atlas.channels);
}

static bool has_valid_top_surface_payload(const ImageMapRawFilamentOffsetAtlas &atlas)
{
    if (atlas.top_surface_layers.empty())
        return false;
    for (const ImageMapRawTopSurfaceLayer &layer : atlas.top_surface_layers)
        if (!top_surface_layer_valid(layer, atlas.width, atlas.height))
            return false;
    return true;
}

static std::vector<ImageMapRawTopSurfaceFilamentValue> top_surface_filament_values_from_metadata_json(
    const std::string &metadata_json)
{
    std::vector<ImageMapRawTopSurfaceFilamentValue> out;
    try {
        const nlohmann::json root = nlohmann::json::parse(metadata_json);
        const nlohmann::json top_surface = root.value("top_surface", nlohmann::json::object());
        const nlohmann::json values = top_surface.value("filament_values", nlohmann::json::array());
        if (!values.is_array())
            return out;
        for (const nlohmann::json &entry : values) {
            if (!entry.is_object())
                continue;
            const int value = entry.value("value", -1);
            const int slot = entry.value("slot", 0);
            if (value < 0 || value > 255 || slot <= 0)
                continue;
            out.push_back({ uint8_t(value), unsigned(slot) });
        }
    } catch (...) {
        out.clear();
    }
    return out;
}

static std::vector<ImageMapRawTopSurfaceFilamentValue> normalized_top_surface_filament_values(
    const ImageMapRawFilamentOffsetAtlas &atlas)
{
    std::vector<ImageMapRawTopSurfaceFilamentValue> values = atlas.top_surface_filament_values;
    if (values.empty())
        values = top_surface_filament_values_from_metadata_json(atlas.metadata_json);

    std::set<unsigned int> used_slots;
    std::vector<ImageMapRawTopSurfaceFilamentValue> normalized;
    for (const ImageMapRawTopSurfaceFilamentValue &entry : values) {
        if (entry.slot == 0 || used_slots.find(entry.slot) != used_slots.end())
            continue;
        normalized.emplace_back(entry);
        used_slots.insert(entry.slot);
    }
    if (!normalized.empty())
        return normalized;

    for (const ImageMapRawTopSurfaceLayer &layer : atlas.top_surface_layers) {
        const size_t pixel_count = size_t(atlas.width) * size_t(atlas.height);
        const size_t count = std::min(pixel_count, layer.filament_slots.size());
        for (size_t idx = 0; idx < count; ++idx) {
            const unsigned int slot = layer.filament_slots[idx];
            if (slot > 0)
                used_slots.insert(slot);
        }
    }
    if (used_slots.empty()) {
        for (const ImageMapRawFilament &filament : atlas.filaments)
            if (filament.slot > 0)
                used_slots.insert(filament.slot);
    }
    if (used_slots.empty())
        return normalized;

    const size_t count = used_slots.size();
    size_t idx = 0;
    for (unsigned int slot : used_slots) {
        const uint8_t value =
            count <= 1 ? uint8_t(0) : uint8_t(std::lround(double(idx) * 255.0 / double(count - 1)));
        normalized.push_back({ value, slot });
        ++idx;
    }
    return normalized;
}

static uint8_t top_surface_value_for_slot(unsigned int slot,
                                          const std::vector<ImageMapRawTopSurfaceFilamentValue> &values)
{
    for (const ImageMapRawTopSurfaceFilamentValue &entry : values)
        if (entry.slot == slot)
            return entry.value;
    return 0;
}

static nlohmann::json atlas_metadata_json(const ImageMapRawFilamentOffsetAtlas &atlas, uint32_t header_rows)
{
    const uint32_t region_count = (atlas.channels + 2u) / 3u;
    const uint32_t top_surface_region_count = uint32_t((atlas.top_surface_layers.size() + 2u) / 3u);
    nlohmann::json root;
    root["format"] = "raw_filament_offset_atlas";
    root["image"] = {
        { "width", atlas.width },
        { "height", atlas.height },
        { "channels", atlas.channels },
        { "offset_channels", atlas.channels }
    };
    root["filaments"] = nlohmann::json::array();
    for (const ImageMapRawFilament &filament : atlas.filaments) {
        nlohmann::json entry;
        entry["slot"] = filament.slot;
        entry["color"] = filament.color.empty() ? "custom" : filament.color;
        if (!filament.hex.empty())
            entry["hex"] = filament.hex;
        root["filaments"].push_back(std::move(entry));
    }
    const ImageMapRawExpectedLineWidth expected =
        atlas.expected_line_width_mm.valid ? atlas.expected_line_width_mm : expected_line_width_from_metadata_json(atlas.metadata_json);
    if (expected.valid) {
        root["expected_line_width_mm"] = {
            { "min", expected.min_mm },
            { "max", expected.max_mm },
            { "warn_if_differs", expected.warn_if_differs }
        };
    }
    root["regions"] = nlohmann::json::array();
    for (uint32_t region_idx = 0; region_idx < region_count; ++region_idx) {
        nlohmann::json channels = nlohmann::json::object();
        const uint32_t first_channel = region_idx * 3u + 1u;
        if (first_channel <= atlas.channels)
            channels["r"] = first_channel;
        if (first_channel + 1u <= atlas.channels)
            channels["g"] = first_channel + 1u;
        if (first_channel + 2u <= atlas.channels)
            channels["b"] = first_channel + 2u;

        nlohmann::json region;
        region["x"] = region_idx * atlas.width;
        region["y"] = header_rows;
        region["width"] = atlas.width;
        region["height"] = atlas.height;
        region["offset_channels"] = std::move(channels);
        if (region_idx == 0)
            region["alpha"] = "projection_mask";
        root["regions"].push_back(std::move(region));
    }
    if (!atlas.top_surface_layers.empty()) {
        nlohmann::json top_surface;
        top_surface["encoding"] = "surface_infill_layer_labels";
        top_surface["filament_values"] = nlohmann::json::array();
        const std::vector<ImageMapRawTopSurfaceFilamentValue> filament_values =
            normalized_top_surface_filament_values(atlas);
        for (const ImageMapRawTopSurfaceFilamentValue &entry : filament_values)
            top_surface["filament_values"].push_back({ { "value", int(entry.value) }, { "slot", entry.slot } });

        for (uint32_t region_idx = 0; region_idx < top_surface_region_count; ++region_idx) {
            nlohmann::json region;
            const std::string id = "top_" + std::to_string(region_idx + 1u);
            region["id"] = id;
            region["x"] = (region_count + region_idx) * atlas.width;
            region["y"] = header_rows;
            region["width"] = atlas.width;
            region["height"] = atlas.height;
            if (region_count == 0 && region_idx == 0)
                region["alpha"] = "projection_mask";
            root["regions"].push_back(std::move(region));
        }

        top_surface["layers"] = nlohmann::json::array();
        for (size_t layer_idx = 0; layer_idx < atlas.top_surface_layers.size(); ++layer_idx) {
            const uint32_t region_idx = uint32_t(layer_idx / 3u);
            const size_t channel_idx = layer_idx % 3u;
            const char *channel = channel_idx == 0 ? "r" : (channel_idx == 1 ? "g" : "b");
            top_surface["layers"].push_back({
                { "depth", atlas.top_surface_layers[layer_idx].depth },
                { "region", "top_" + std::to_string(region_idx + 1u) },
                { "channel", channel }
            });
        }
        root["top_surface"] = std::move(top_surface);
    }
    return root;
}

} // namespace

bool ImageMapRawFilamentOffsetAtlas::valid() const
{
    return width > 0 &&
           height > 0 &&
           (has_valid_offset_payload(*this) || has_valid_top_surface_payload(*this));
}

bool image_map_raw_filament_is_standard_color(const std::string &color)
{
    const std::string key = uppercase_ascii(color);
    return key == "C" || key == "M" || key == "Y" || key == "K" ||
           key == "W" || key == "R" || key == "G" || key == "B";
}

std::string image_map_raw_filament_channel_key(const ImageMapRawFilament &filament, size_t channel_idx)
{
    const std::string color = uppercase_ascii(filament.color);
    if (image_map_raw_filament_is_standard_color(color))
        return color;

    std::string hex = uppercase_ascii(filament.hex);
    const unsigned int slot = filament.slot != 0 ? filament.slot : unsigned(channel_idx + 1);
    if (!hex.empty())
        return "CUSTOM:" + std::to_string(slot) + ":" + hex;
    if (!color.empty())
        return color + ":" + std::to_string(slot);
    return "SLOT:" + std::to_string(slot);
}

std::vector<ImageMapRawFilament> image_map_raw_filaments_for_channels(const std::vector<ImageMapRawFilament> &filaments,
                                                                      uint32_t channels)
{
    std::vector<ImageMapRawFilament> normalized(static_cast<size_t>(channels));
    std::vector<uint8_t> filled(static_cast<size_t>(channels), 0);
    for (uint32_t channel = 0; channel < channels; ++channel) {
        normalized[size_t(channel)].slot = channel + 1;
        normalized[size_t(channel)].color = "custom";
        normalized[size_t(channel)].hex = "#FFFFFF";
    }

    size_t next_empty = 0;
    for (ImageMapRawFilament filament : filaments) {
        size_t target = size_t(channels);
        if (filament.slot >= 1 && filament.slot <= channels)
            target = size_t(filament.slot - 1);
        else {
            while (next_empty < filled.size() && filled[next_empty] != 0)
                ++next_empty;
            if (next_empty < filled.size())
                target = next_empty;
        }
        if (target >= normalized.size() || filled[target] != 0)
            continue;
        if (filament.slot == 0)
            filament.slot = unsigned(target + 1);
        if (filament.color.empty())
            filament.color = "custom";
        if (filament.hex.empty() && !image_map_raw_filament_is_standard_color(filament.color))
            filament.hex = "#FFFFFF";
        normalized[target] = std::move(filament);
        filled[target] = 1;
    }
    return normalized;
}

std::vector<ImageMapRawFilament> image_map_raw_filaments_from_metadata_json(const std::string &metadata_json)
{
    std::vector<ImageMapRawFilament> filaments;
    try {
        const nlohmann::json root = nlohmann::json::parse(metadata_json);
        const nlohmann::json entries = root.value("filaments", nlohmann::json::array());
        if (entries.is_array()) {
            for (const nlohmann::json &entry : entries) {
                if (!entry.is_object())
                    continue;
                ImageMapRawFilament filament;
                filament.slot = unsigned(std::max(0, entry.value("slot", 0)));
                filament.color = entry.value("color", std::string());
                filament.hex = entry.value("hex", std::string());
                if (filament.hex.empty() && image_map_raw_filament_is_standard_color(filament.color))
                    filament.hex = standard_hex_for_color_code(filament.color);
                filaments.emplace_back(std::move(filament));
            }
        }
    } catch (...) {
        filaments.clear();
    }
    return filaments;
}

std::vector<ImageMapRawFilament> image_map_raw_filaments_from_metadata_json(const std::string &metadata_json,
                                                                            uint32_t channels)
{
    std::vector<ImageMapRawFilament> filaments = image_map_raw_filaments_from_metadata_json(metadata_json);
    return image_map_raw_filaments_for_channels(filaments, channels);
}

ImageMapRawExpectedLineWidth image_map_raw_expected_line_width_from_metadata_json(const std::string &metadata_json)
{
    return expected_line_width_from_metadata_json(metadata_json);
}

std::vector<std::string> image_map_raw_filament_channel_keys(const std::vector<ImageMapRawFilament> &filaments)
{
    std::vector<std::string> keys;
    keys.reserve(filaments.size());
    for (size_t idx = 0; idx < filaments.size(); ++idx)
        keys.emplace_back(image_map_raw_filament_channel_key(filaments[idx], idx));
    return keys;
}

bool decode_image_map_raw_filament_offset_atlas(const std::vector<uint8_t> &rgba,
                                                uint32_t atlas_width,
                                                uint32_t atlas_height,
                                                ImageMapRawFilamentOffsetAtlas &out,
                                                std::string *error)
{
    out = {};
    if (atlas_width == 0 || atlas_height == 0 || rgba.size() < size_t(atlas_width) * size_t(atlas_height) * 4) {
        set_error(error, "Invalid image dimensions.");
        return false;
    }

    std::vector<uint8_t> fixed;
    if (!read_header_bytes(rgba, atlas_width, atlas_height, FixedHeaderSize, fixed)) {
        set_error(error, "The image is too small for an ImageMap raw filament offset header.");
        return false;
    }
    if (!std::equal(fixed.begin(), fixed.begin() + MagicSize, Magic)) {
        set_error(error, "The image is not an ImageMap raw filament offset atlas.");
        return false;
    }
    if (fixed[MagicSize] != 1u || (fixed[MagicSize + 1] & 1u) == 0u) {
        set_error(error, "Unsupported ImageMap raw filament offset atlas header.");
        return false;
    }

    const uint32_t header_rows = read_be_u32(fixed, MagicSize + 2);
    const uint32_t metadata_length = read_be_u32(fixed, MagicSize + 6);
    if (header_rows == 0 || header_rows > atlas_height) {
        set_error(error, "Invalid ImageMap raw filament offset header row count.");
        return false;
    }
    const size_t total_header_bytes = FixedHeaderSize + size_t(metadata_length);
    if (total_header_bytes > std::numeric_limits<size_t>::max() / 8 ||
        total_header_bytes * 8 > size_t(header_rows) * size_t(atlas_width)) {
        set_error(error, "ImageMap raw filament offset metadata exceeds the declared header rows.");
        return false;
    }

    std::vector<uint8_t> header;
    if (!read_header_bytes(rgba, atlas_width, atlas_height, total_header_bytes, header)) {
        set_error(error, "Could not read ImageMap raw filament offset metadata.");
        return false;
    }

    std::string metadata(reinterpret_cast<const char *>(header.data() + FixedHeaderSize), metadata_length);
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(metadata);
    } catch (...) {
        set_error(error, "ImageMap raw filament offset metadata is not valid JSON.");
        return false;
    }
    if (!root.is_object() || root.value("format", std::string()) != "raw_filament_offset_atlas") {
        set_error(error, "ImageMap raw filament offset metadata has an unsupported format.");
        return false;
    }

    const nlohmann::json image = root.value("image", nlohmann::json::object());
    const int logical_width = image.value("width", 0);
    const int logical_height = image.value("height", 0);
    const int logical_channels = image.value("offset_channels", image.value("channels", 0));
    if (logical_width <= 0 || logical_height <= 0 || logical_channels < 0) {
        set_error(error, "ImageMap raw filament offset metadata has invalid image dimensions.");
        return false;
    }

    ImageMapRawFilamentOffsetAtlas decoded;
    decoded.width = uint32_t(logical_width);
    decoded.height = uint32_t(logical_height);
    decoded.channels = uint32_t(logical_channels);
    if (decoded.channels > 0)
        decoded.offsets.assign(size_t(decoded.width) * size_t(decoded.height) * size_t(decoded.channels), 0);
    decoded.mask.assign(size_t(decoded.width) * size_t(decoded.height), 255);
    decoded.metadata_json = metadata;
    decoded.expected_line_width_mm = expected_line_width_from_json(root);

    const nlohmann::json filaments = root.value("filaments", nlohmann::json::array());
    if (filaments.is_array()) {
        for (const nlohmann::json &entry : filaments) {
            if (!entry.is_object())
                continue;
            ImageMapRawFilament filament;
            filament.slot = unsigned(std::max(0, entry.value("slot", 0)));
            filament.color = entry.value("color", std::string());
            filament.hex = entry.value("hex", std::string());
            if (filament.hex.empty())
                filament.hex = standard_hex_for_color_code(filament.color);
            decoded.filaments.emplace_back(std::move(filament));
        }
    }

    const nlohmann::json regions = root.value("regions", nlohmann::json::array());
    if (decoded.channels > 0 && (!regions.is_array() || regions.empty())) {
        set_error(error, "ImageMap raw filament offset metadata does not contain regions.");
        return false;
    }

    struct TopRegion {
        int x { 0 };
        int y { 0 };
        int width { 0 };
        int height { 0 };
        std::array<bool, 3> offset_channels { { false, false, false } };
    };
    std::map<std::string, TopRegion> top_regions_by_id;
    bool decoded_offset_payload = false;
    const std::array<std::pair<const char *, size_t>, 3> rgb_channels = {
        std::make_pair("r", size_t(0)),
        std::make_pair("g", size_t(1)),
        std::make_pair("b", size_t(2))
    };
    auto copy_projection_mask = [&](int rx, int ry, int rw, int rh) {
        const uint32_t copy_width = std::min<uint32_t>(decoded.width, uint32_t(rw));
        const uint32_t copy_height = std::min<uint32_t>(decoded.height, uint32_t(rh));
        for (uint32_t y = 0; y < copy_height; ++y) {
            for (uint32_t x = 0; x < copy_width; ++x) {
                const size_t src_idx = (size_t(ry + int(y)) * size_t(atlas_width) + size_t(rx + int(x))) * 4 + 3;
                decoded.mask[size_t(y) * size_t(decoded.width) + size_t(x)] = rgba[src_idx];
            }
        }
    };

    if (regions.is_array()) for (const nlohmann::json &region : regions) {
        if (!region.is_object())
            continue;
        const int rx = region.value("x", -1);
        const int ry = region.value("y", -1);
        const int rw = region.value("width", 0);
        const int rh = region.value("height", 0);
        if (rx < 0 || ry < 0 || rw <= 0 || rh <= 0 ||
            uint64_t(rx) + uint64_t(rw) > atlas_width ||
            uint64_t(ry) + uint64_t(rh) > atlas_height) {
            set_error(error, "ImageMap raw filament offset region exceeds the atlas image.");
            return false;
        }

        TopRegion top_region { rx, ry, rw, rh, { { false, false, false } } };
        const nlohmann::json offset_channels =
            region.value("offset_channels", region.value("channels", nlohmann::json::object()));
        if (offset_channels.is_object()) {
            for (const auto &rgb_channel : rgb_channels) {
                const int logical_channel = offset_channels.value(rgb_channel.first, 0);
                if (logical_channel <= 0 || logical_channel > logical_channels)
                    continue;
                decoded_offset_payload = true;
                top_region.offset_channels[rgb_channel.second] = true;
                const size_t dst_channel = size_t(logical_channel - 1);
                const uint32_t copy_width = std::min<uint32_t>(decoded.width, uint32_t(rw));
                const uint32_t copy_height = std::min<uint32_t>(decoded.height, uint32_t(rh));
                for (uint32_t y = 0; y < copy_height; ++y) {
                    for (uint32_t x = 0; x < copy_width; ++x) {
                        const size_t src_idx = (size_t(ry + int(y)) * size_t(atlas_width) + size_t(rx + int(x))) * 4 + rgb_channel.second;
                        const size_t dst_idx = (size_t(y) * size_t(decoded.width) + size_t(x)) * size_t(decoded.channels) + dst_channel;
                        decoded.offsets[dst_idx] = rgba[src_idx];
                    }
                }
            }
        }

        if (region.value("alpha", std::string()) == "projection_mask")
            copy_projection_mask(rx, ry, rw, rh);

        const std::string id = region.value("id", std::string());
        if (!id.empty() && !top_regions_by_id.emplace(id, top_region).second) {
            set_error(error, "ImageMap raw filament offset metadata has duplicate region ids.");
            return false;
        }
    }

    const nlohmann::json top_surface = root.value("top_surface", nlohmann::json::object());
    if (top_surface.is_object() && !top_surface.empty()) {
        if (top_surface.value("encoding", std::string()) != "surface_infill_layer_labels") {
            set_error(error, "ImageMap raw filament offset metadata has unsupported top-surface encoding.");
            return false;
        }

        std::map<int, unsigned int> slot_by_value;
        const nlohmann::json filament_values = top_surface.value("filament_values", nlohmann::json::array());
        if (!filament_values.is_array() || filament_values.empty()) {
            set_error(error, "ImageMap raw filament offset metadata has invalid top-surface filament values.");
            return false;
        }
        for (const nlohmann::json &entry : filament_values) {
            if (!entry.is_object())
                continue;
            const int value = entry.value("value", -1);
            const int slot = entry.value("slot", 0);
            if (value < 0 || value > 255 || slot <= 0 || slot > int(std::numeric_limits<uint16_t>::max())) {
                set_error(error, "ImageMap raw filament offset metadata has invalid top-surface filament value.");
                return false;
            }
            if (!slot_by_value.emplace(value, unsigned(slot)).second) {
                set_error(error, "ImageMap raw filament offset metadata has duplicate top-surface filament values.");
                return false;
            }
            decoded.top_surface_filament_values.push_back({ uint8_t(value), unsigned(slot) });
        }
        if (slot_by_value.empty()) {
            set_error(error, "ImageMap raw filament offset metadata has no usable top-surface filament values.");
            return false;
        }

        const nlohmann::json top_regions = top_surface.value("regions", nlohmann::json::array());
        if (top_regions.is_array()) for (const nlohmann::json &region : top_regions) {
            if (!region.is_object())
                continue;
            const std::string id = region.value("id", std::string());
            const int rx = region.value("x", -1);
            const int ry = region.value("y", -1);
            const int rw = region.value("width", 0);
            const int rh = region.value("height", 0);
            if (id.empty() || rx < 0 || ry < 0 || rw <= 0 || rh <= 0 ||
                uint64_t(rx) + uint64_t(rw) > atlas_width ||
                uint64_t(ry) + uint64_t(rh) > atlas_height) {
                set_error(error, "ImageMap raw filament offset top-surface region exceeds the atlas image.");
                return false;
            }
            if (!top_regions_by_id.emplace(id, TopRegion{ rx, ry, rw, rh, { { false, false, false } } }).second) {
                set_error(error, "ImageMap raw filament offset metadata has duplicate top-surface region ids.");
                return false;
            }
            if (region.value("alpha", std::string()) == "projection_mask")
                copy_projection_mask(rx, ry, rw, rh);
        }

        const nlohmann::json layers = top_surface.value("layers", nlohmann::json::array());
        if (!layers.is_array() || layers.empty()) {
            set_error(error, "ImageMap raw filament offset metadata has invalid top-surface layers.");
            return false;
        }
        std::set<int> used_depths;
        std::set<std::pair<std::string, std::string>> used_storage;
        for (const nlohmann::json &layer : layers) {
            if (!layer.is_object())
                continue;
            const int depth = layer.value("depth", -1);
            const std::string region_id = layer.value("region", std::string());
            const std::string channel = layer.value("channel", std::string());
            const auto region_it = top_regions_by_id.find(region_id);
            size_t channel_idx = size_t(-1);
            if (channel == "r")
                channel_idx = 0;
            else if (channel == "g")
                channel_idx = 1;
            else if (channel == "b")
                channel_idx = 2;
            if (depth < 0 || region_it == top_regions_by_id.end() || channel_idx >= 3) {
                set_error(error, "ImageMap raw filament offset metadata has invalid top-surface layer mapping.");
                return false;
            }
            if (region_it->second.offset_channels[channel_idx]) {
                set_error(error, "ImageMap raw filament offset metadata reuses a raw offset channel for a top-surface layer.");
                return false;
            }
            if (!used_depths.insert(depth).second ||
                !used_storage.insert(std::make_pair(region_id, channel)).second) {
                set_error(error, "ImageMap raw filament offset metadata has duplicate top-surface layer mappings.");
                return false;
            }

            const TopRegion &region = region_it->second;
            ImageMapRawTopSurfaceLayer decoded_layer;
            decoded_layer.depth = depth;
            decoded_layer.filament_slots.assign(size_t(decoded.width) * size_t(decoded.height), 0);
            const uint32_t copy_width = std::min<uint32_t>(decoded.width, uint32_t(region.width));
            const uint32_t copy_height = std::min<uint32_t>(decoded.height, uint32_t(region.height));
            for (uint32_t y = 0; y < copy_height; ++y) {
                for (uint32_t x = 0; x < copy_width; ++x) {
                    const size_t src_idx = (size_t(region.y + int(y)) * size_t(atlas_width) + size_t(region.x + int(x))) * 4 + channel_idx;
                    const auto value_it = slot_by_value.find(int(rgba[src_idx]));
                    if (value_it == slot_by_value.end()) {
                        set_error(error, "ImageMap raw filament offset top-surface layer has an unknown filament value.");
                        return false;
                    }
                    decoded_layer.filament_slots[size_t(y) * size_t(decoded.width) + size_t(x)] = uint16_t(value_it->second);
                }
            }
            decoded.top_surface_layers.emplace_back(std::move(decoded_layer));
        }
        std::sort(decoded.top_surface_layers.begin(), decoded.top_surface_layers.end(),
                  [](const ImageMapRawTopSurfaceLayer &lhs, const ImageMapRawTopSurfaceLayer &rhs) {
                      return lhs.depth < rhs.depth;
                  });
    }

    if (!decoded_offset_payload && !has_valid_top_surface_payload(decoded)) {
        set_error(error, "ImageMap raw filament offset metadata does not contain usable offset or top-surface data.");
        return false;
    }

    out = std::move(decoded);
    return true;
}

bool encode_image_map_raw_filament_offset_atlas(const ImageMapRawFilamentOffsetAtlas &atlas,
                                                std::vector<uint8_t> &rgba,
                                                uint32_t &atlas_width,
                                                uint32_t &atlas_height,
                                                std::string *error)
{
    rgba.clear();
    atlas_width = 0;
    atlas_height = 0;
    if (!atlas.valid()) {
        set_error(error, "Invalid ImageMap raw filament offset atlas data.");
        return false;
    }

    const uint32_t region_count = (atlas.channels + 2u) / 3u;
    const uint32_t top_surface_region_count = uint32_t((atlas.top_surface_layers.size() + 2u) / 3u);
    const uint32_t total_region_count = region_count + top_surface_region_count;
    if (total_region_count == 0 || atlas.width > std::numeric_limits<uint32_t>::max() / total_region_count) {
        set_error(error, "ImageMap raw filament offset atlas is too wide.");
        return false;
    }

    atlas_width = atlas.width * total_region_count;
    uint32_t header_rows = 1;
    std::string metadata;
    for (int iter = 0; iter < 8; ++iter) {
        metadata = atlas_metadata_json(atlas, header_rows).dump();
        const size_t bits = (FixedHeaderSize + metadata.size()) * 8;
        const uint32_t needed_rows = uint32_t((bits + size_t(atlas_width) - 1) / size_t(atlas_width));
        if (needed_rows == header_rows)
            break;
        header_rows = std::max<uint32_t>(needed_rows, 1);
    }
    metadata = atlas_metadata_json(atlas, header_rows).dump();
    if ((FixedHeaderSize + metadata.size()) * 8 > size_t(header_rows) * size_t(atlas_width)) {
        set_error(error, "Could not fit ImageMap raw filament offset metadata header.");
        return false;
    }
    if (header_rows > std::numeric_limits<uint32_t>::max() - atlas.height) {
        set_error(error, "ImageMap raw filament offset atlas is too tall.");
        return false;
    }

    atlas_height = header_rows + atlas.height;
    rgba.assign(size_t(atlas_width) * size_t(atlas_height) * 4, 255);
    for (uint32_t y = 0; y < header_rows; ++y) {
        for (uint32_t x = 0; x < atlas_width; ++x) {
            const size_t idx = (size_t(y) * size_t(atlas_width) + size_t(x)) * 4;
            rgba[idx + 0] = 0;
            rgba[idx + 1] = 0;
            rgba[idx + 2] = 0;
            rgba[idx + 3] = 255;
        }
    }

    for (uint32_t region_idx = 0; region_idx < region_count; ++region_idx) {
        const uint32_t region_x = region_idx * atlas.width;
        const uint32_t first_channel = region_idx * 3u;
        for (uint32_t y = 0; y < atlas.height; ++y) {
            for (uint32_t x = 0; x < atlas.width; ++x) {
                const size_t dst = (size_t(header_rows + y) * size_t(atlas_width) + size_t(region_x + x)) * 4;
                for (uint32_t c = 0; c < 3; ++c) {
                    const uint32_t channel = first_channel + c;
                    rgba[dst + c] = channel < atlas.channels ?
                        atlas.offsets[(size_t(y) * size_t(atlas.width) + size_t(x)) * size_t(atlas.channels) + size_t(channel)] :
                        0;
                }
                rgba[dst + 3] = region_idx == 0 && atlas.mask.size() >= size_t(atlas.width) * size_t(atlas.height) ?
                    atlas.mask[size_t(y) * size_t(atlas.width) + size_t(x)] :
                    255;
            }
        }
    }
    if (top_surface_region_count > 0) {
        const std::vector<ImageMapRawTopSurfaceFilamentValue> filament_values =
            normalized_top_surface_filament_values(atlas);
        for (uint32_t region_idx = 0; region_idx < top_surface_region_count; ++region_idx) {
            const uint32_t region_x = (region_count + region_idx) * atlas.width;
            const uint32_t first_layer = region_idx * 3u;
            for (uint32_t y = 0; y < atlas.height; ++y) {
                for (uint32_t x = 0; x < atlas.width; ++x) {
                    const size_t dst = (size_t(header_rows + y) * size_t(atlas_width) + size_t(region_x + x)) * 4;
                    for (uint32_t c = 0; c < 3; ++c) {
                        const uint32_t layer_idx = first_layer + c;
                        if (layer_idx < atlas.top_surface_layers.size()) {
                            const ImageMapRawTopSurfaceLayer &layer = atlas.top_surface_layers[size_t(layer_idx)];
                            const size_t src = size_t(y) * size_t(atlas.width) + size_t(x);
                            const unsigned int slot = src < layer.filament_slots.size() ? layer.filament_slots[src] : 0;
                            rgba[dst + c] = top_surface_value_for_slot(slot, filament_values);
                        } else {
                            rgba[dst + c] = 0;
                        }
                    }
                    rgba[dst + 3] =
                        region_count == 0 && region_idx == 0 && atlas.mask.size() >= size_t(atlas.width) * size_t(atlas.height) ?
                        atlas.mask[size_t(y) * size_t(atlas.width) + size_t(x)] :
                        255;
                }
            }
        }
    }

    std::vector<uint8_t> header;
    header.reserve(FixedHeaderSize + metadata.size());
    header.insert(header.end(), Magic, Magic + MagicSize);
    header.emplace_back(1);
    header.emplace_back(1);
    append_be_u32(header, header_rows);
    append_be_u32(header, uint32_t(metadata.size()));
    header.insert(header.end(), metadata.begin(), metadata.end());
    write_header_bytes(rgba, header);
    return true;
}

std::vector<uint8_t> image_map_raw_filament_offset_preview_rgba(const ImageMapRawFilamentOffsetAtlas &atlas)
{
    std::vector<uint8_t> preview;
    if (!atlas.valid())
        return preview;

    preview.assign(size_t(atlas.width) * size_t(atlas.height) * 4, 255);
    const bool has_offsets = has_valid_offset_payload(atlas);
    const bool grayscale = atlas.channels == 1;
    const ImageMapRawTopSurfaceLayer *top_surface_layer =
        !has_offsets && !atlas.top_surface_layers.empty() ? &atlas.top_surface_layers.front() : nullptr;
    for (uint32_t y = 0; y < atlas.height; ++y) {
        for (uint32_t x = 0; x < atlas.width; ++x) {
            const size_t pixel = size_t(y) * size_t(atlas.width) + size_t(x);
            const size_t src = pixel * size_t(atlas.channels);
            const size_t dst = pixel * 4;
            if (has_offsets && grayscale) {
                preview[dst + 0] = atlas.offsets[src];
                preview[dst + 1] = atlas.offsets[src];
                preview[dst + 2] = atlas.offsets[src];
            } else if (has_offsets) {
                preview[dst + 0] = atlas.channels > 0 ? atlas.offsets[src + 0] : 0;
                preview[dst + 1] = atlas.channels > 1 ? atlas.offsets[src + 1] : 0;
                preview[dst + 2] = atlas.channels > 2 ? atlas.offsets[src + 2] : 0;
            } else if (top_surface_layer != nullptr && pixel < top_surface_layer->filament_slots.size()) {
                const std::array<uint8_t, 3> rgb =
                    preview_rgb_for_filament_slot(atlas.filaments, top_surface_layer->filament_slots[pixel]);
                preview[dst + 0] = rgb[0];
                preview[dst + 1] = rgb[1];
                preview[dst + 2] = rgb[2];
            } else {
                preview[dst + 0] = 0;
                preview[dst + 1] = 0;
                preview[dst + 2] = 0;
            }
            preview[dst + 3] = atlas.mask.size() > pixel ? atlas.mask[pixel] : 255;
        }
    }
    return preview;
}

} // namespace Slic3r
