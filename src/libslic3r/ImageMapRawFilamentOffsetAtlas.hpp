#ifndef slic3r_ImageMapRawFilamentOffsetAtlas_hpp_
#define slic3r_ImageMapRawFilamentOffsetAtlas_hpp_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Slic3r {

struct ImageMapRawFilament
{
    unsigned int slot { 0 };
    std::string  color;
    std::string  hex;
};

struct ImageMapRawFilamentOffsetAtlas
{
    uint32_t width { 0 };
    uint32_t height { 0 };
    uint32_t channels { 0 };
    std::vector<ImageMapRawFilament> filaments;
    std::vector<uint8_t> offsets;
    std::vector<uint8_t> mask;
    std::string metadata_json;

    bool valid() const;
};

bool decode_image_map_raw_filament_offset_atlas(const std::vector<uint8_t> &rgba,
                                                uint32_t atlas_width,
                                                uint32_t atlas_height,
                                                ImageMapRawFilamentOffsetAtlas &out,
                                                std::string *error = nullptr);

bool encode_image_map_raw_filament_offset_atlas(const ImageMapRawFilamentOffsetAtlas &atlas,
                                                std::vector<uint8_t> &rgba,
                                                uint32_t &atlas_width,
                                                uint32_t &atlas_height,
                                                std::string *error = nullptr);

bool image_map_raw_filament_is_standard_color(const std::string &color);

std::string image_map_raw_filament_channel_key(const ImageMapRawFilament &filament, size_t channel_idx);

std::vector<ImageMapRawFilament> image_map_raw_filaments_for_channels(const std::vector<ImageMapRawFilament> &filaments,
                                                                      uint32_t channels);

std::vector<ImageMapRawFilament> image_map_raw_filaments_from_metadata_json(const std::string &metadata_json,
                                                                            uint32_t channels);

std::vector<std::string> image_map_raw_filament_channel_keys(const std::vector<ImageMapRawFilament> &filaments);

std::vector<uint8_t> image_map_raw_filament_offset_preview_rgba(const ImageMapRawFilamentOffsetAtlas &atlas);

} // namespace Slic3r

#endif
