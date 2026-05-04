#include "ImageMapRawFilamentOffsetAtlas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
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

static nlohmann::json atlas_metadata_json(const ImageMapRawFilamentOffsetAtlas &atlas, uint32_t header_rows)
{
    const uint32_t region_count = (atlas.channels + 2u) / 3u;
    nlohmann::json root;
    root["format"] = "raw_filament_offset_atlas";
    root["image"] = {
        { "width", atlas.width },
        { "height", atlas.height },
        { "channels", atlas.channels }
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
        region["channels"] = std::move(channels);
        if (region_idx == 0)
            region["alpha"] = "projection_mask";
        root["regions"].push_back(std::move(region));
    }
    return root;
}

} // namespace

bool ImageMapRawFilamentOffsetAtlas::valid() const
{
    return width > 0 &&
           height > 0 &&
           channels > 0 &&
           offsets.size() >= size_t(width) * size_t(height) * size_t(channels);
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

std::vector<ImageMapRawFilament> image_map_raw_filaments_from_metadata_json(const std::string &metadata_json,
                                                                            uint32_t channels)
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
                filaments.emplace_back(std::move(filament));
            }
        }
    } catch (...) {
        filaments.clear();
    }
    return image_map_raw_filaments_for_channels(filaments, channels);
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
    const int logical_channels = image.value("channels", 0);
    if (logical_width <= 0 || logical_height <= 0 || logical_channels <= 0) {
        set_error(error, "ImageMap raw filament offset metadata has invalid image dimensions.");
        return false;
    }

    ImageMapRawFilamentOffsetAtlas decoded;
    decoded.width = uint32_t(logical_width);
    decoded.height = uint32_t(logical_height);
    decoded.channels = uint32_t(logical_channels);
    decoded.offsets.assign(size_t(decoded.width) * size_t(decoded.height) * size_t(decoded.channels), 0);
    decoded.mask.assign(size_t(decoded.width) * size_t(decoded.height), 255);
    decoded.metadata_json = metadata;

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
    if (!regions.is_array() || regions.empty()) {
        set_error(error, "ImageMap raw filament offset metadata does not contain regions.");
        return false;
    }

    for (const nlohmann::json &region : regions) {
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

        const nlohmann::json channels = region.value("channels", nlohmann::json::object());
        if (channels.is_object()) {
            const std::array<std::pair<const char *, size_t>, 3> rgb_channels = {
                std::make_pair("r", size_t(0)),
                std::make_pair("g", size_t(1)),
                std::make_pair("b", size_t(2))
            };
            for (const auto &rgb_channel : rgb_channels) {
                const int logical_channel = channels.value(rgb_channel.first, 0);
                if (logical_channel <= 0 || logical_channel > logical_channels)
                    continue;
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

        if (region.value("alpha", std::string()) == "projection_mask") {
            const uint32_t copy_width = std::min<uint32_t>(decoded.width, uint32_t(rw));
            const uint32_t copy_height = std::min<uint32_t>(decoded.height, uint32_t(rh));
            for (uint32_t y = 0; y < copy_height; ++y) {
                for (uint32_t x = 0; x < copy_width; ++x) {
                    const size_t src_idx = (size_t(ry + int(y)) * size_t(atlas_width) + size_t(rx + int(x))) * 4 + 3;
                    decoded.mask[size_t(y) * size_t(decoded.width) + size_t(x)] = rgba[src_idx];
                }
            }
        }
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
    if (region_count == 0 || atlas.width > std::numeric_limits<uint32_t>::max() / region_count) {
        set_error(error, "ImageMap raw filament offset atlas is too wide.");
        return false;
    }

    atlas_width = atlas.width * region_count;
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
    const bool grayscale = atlas.channels == 1;
    for (uint32_t y = 0; y < atlas.height; ++y) {
        for (uint32_t x = 0; x < atlas.width; ++x) {
            const size_t pixel = size_t(y) * size_t(atlas.width) + size_t(x);
            const size_t src = pixel * size_t(atlas.channels);
            const size_t dst = pixel * 4;
            if (grayscale) {
                preview[dst + 0] = atlas.offsets[src];
                preview[dst + 1] = atlas.offsets[src];
                preview[dst + 2] = atlas.offsets[src];
            } else {
                preview[dst + 0] = atlas.channels > 0 ? atlas.offsets[src + 0] : 0;
                preview[dst + 1] = atlas.channels > 1 ? atlas.offsets[src + 1] : 0;
                preview[dst + 2] = atlas.channels > 2 ? atlas.offsets[src + 2] : 0;
            }
            preview[dst + 3] = atlas.mask.size() > pixel ? atlas.mask[pixel] : 255;
        }
    }
    return preview;
}

} // namespace Slic3r
