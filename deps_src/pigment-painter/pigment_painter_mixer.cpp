#include "pigment_painter_mixer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

#include <png.h>

extern "C" const char lut_wide_png_data[10295238];

namespace pigment_painter {
namespace {

constexpr int k_lut_dimen = 256;
constexpr size_t k_lut_png_size = sizeof(lut_wide_png_data);

struct PngMemoryReader
{
    const std::uint8_t *data = nullptr;
    size_t size = 0;
    size_t offset = 0;
};

struct LutImage
{
    int width = 0;
    int height = 0;
    int row_size = 0;
    std::vector<std::uint8_t> rgba;
    bool loaded = false;
};

float clamp01(float value)
{
    return std::max(0.f, std::min(1.f, value));
}

float srgb_to_linear(float value)
{
    const float x = clamp01(value);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(float value)
{
    const float x = clamp01(value);
    return x <= 0.0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
}

void png_memory_read(png_structp png_ptr, png_bytep out, png_size_t bytes)
{
    PngMemoryReader *reader = static_cast<PngMemoryReader *>(png_get_io_ptr(png_ptr));
    if (reader == nullptr || reader->offset + bytes > reader->size)
        png_error(png_ptr, "pigment painter lut read failed");

    std::memcpy(out, reader->data + reader->offset, bytes);
    reader->offset += bytes;
}

bool decode_lut_png(LutImage &image)
{
    PngMemoryReader reader {
        reinterpret_cast<const std::uint8_t *>(lut_wide_png_data),
        k_lut_png_size,
        0
    };

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png == nullptr)
        return false;

    png_infop info = png_create_info_struct(png);
    if (info == nullptr) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        return false;
    }

    png_set_read_fn(png, &reader, png_memory_read);
    png_read_info(png, info);

    png_uint_32 width = png_get_image_width(png, info);
    png_uint_32 height = png_get_image_height(png, info);
    int bit_depth = png_get_bit_depth(png, info);
    int color_type = png_get_color_type(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xff, PNG_FILLER_AFTER);

    png_read_update_info(png, info);

    if (png_get_channels(png, info) != 4 || width % k_lut_dimen != 0 || height % k_lut_dimen != 0) {
        png_destroy_read_struct(&png, &info, nullptr);
        return false;
    }

    image.width = int(width);
    image.height = int(height);
    image.row_size = image.width / k_lut_dimen;
    image.rgba.resize(size_t(width) * size_t(height) * 4);

    std::vector<png_bytep> rows(height);
    for (png_uint_32 row = 0; row < height; ++row)
        rows[row] = image.rgba.data() + size_t(row) * size_t(width) * 4;

    png_read_image(png, rows.data());
    png_read_end(png, nullptr);
    png_destroy_read_struct(&png, &info, nullptr);

    image.loaded = image.width == k_lut_dimen * image.row_size &&
                   image.height >= k_lut_dimen * ((k_lut_dimen * 2 + image.row_size - 1) / image.row_size);
    return image.loaded;
}

const LutImage *lut_image()
{
    static LutImage image;
    static std::once_flag once;
    std::call_once(once, []() { decode_lut_png(image); });
    return image.loaded ? &image : nullptr;
}

std::array<float, 3> sample_lut(const LutImage &image, const std::array<float, 3> &color, int z_offset)
{
    const int x = int(std::floor(clamp01(color[0]) * float(k_lut_dimen - 1)));
    const int y = int(std::floor(clamp01(color[1]) * float(k_lut_dimen - 1)));
    const int z = int(std::floor(clamp01(color[2]) * float(k_lut_dimen - 1))) + z_offset;
    const int col = z % image.row_size;
    const int row = z / image.row_size;
    const int ix = x + col * k_lut_dimen;
    const int iy = y + row * k_lut_dimen;
    const size_t idx = (size_t(iy) * size_t(image.width) + size_t(ix)) * 4;

    return {
        float(image.rgba[idx]) / 255.f,
        float(image.rgba[idx + 2]) / 255.f,
        float(image.rgba[idx + 1]) / 255.f
    };
}

std::array<float, 4> color_to_pigment(const LutImage &image, const std::array<float, 3> &color)
{
    const std::array<float, 3> sampled = sample_lut(image, color, 0);
    std::array<float, 4> pigment {
        sampled[0],
        sampled[1],
        sampled[2],
        1.f - sampled[0] - sampled[1] - sampled[2]
    };

    const float total = pigment[0] + pigment[1] + pigment[2] + pigment[3];
    if (total > 0.0001f) {
        const float inv_total = 1.f / total;
        for (float &value : pigment)
            value *= inv_total;
    }

    return pigment;
}

std::array<float, 3> pigment_to_color(const LutImage &image, const std::array<float, 4> &pigment)
{
    return sample_lut(image, { pigment[0], pigment[1], pigment[2] }, k_lut_dimen);
}

std::array<float, 3> mix_with_lut(const std::vector<std::array<float, 3>> &colors,
                                  const std::vector<float>                &weights,
                                  const LutImage                          &image)
{
    std::array<float, 4> mixed_pigment { 0.f, 0.f, 0.f, 0.f };
    std::array<float, 3> error { 0.f, 0.f, 0.f };
    float total_weight = 0.f;

    for (const float weight : weights) {
        if (std::isfinite(weight) && weight > 0.f)
            total_weight += weight;
    }

    if (total_weight <= 0.f)
        return colors.front();

    const float inv_total_weight = 1.f / total_weight;
    for (size_t idx = 0; idx < colors.size(); ++idx) {
        const float raw_weight = weights[idx];
        if (!std::isfinite(raw_weight) || raw_weight <= 0.f)
            continue;

        const float weight = raw_weight * inv_total_weight;
        const std::array<float, 4> pigment = color_to_pigment(image, colors[idx]);
        const std::array<float, 3> reconstructed = pigment_to_color(image, pigment);

        for (size_t channel = 0; channel < 4; ++channel)
            mixed_pigment[channel] += pigment[channel] * weight;
        for (size_t channel = 0; channel < 3; ++channel)
            error[channel] += (clamp01(colors[idx][channel]) - reconstructed[channel]) * weight;
    }

    const float pigment_total = mixed_pigment[0] + mixed_pigment[1] + mixed_pigment[2] + mixed_pigment[3];
    if (pigment_total > 0.0001f) {
        const float inv_pigment_total = 1.f / pigment_total;
        for (float &value : mixed_pigment)
            value *= inv_pigment_total;
    }

    std::array<float, 3> mixed_color = pigment_to_color(image, mixed_pigment);
    for (size_t channel = 0; channel < 3; ++channel)
        mixed_color[channel] = clamp01(mixed_color[channel] + error[channel]);

    return mixed_color;
}

float reflectance_to_ks(float reflectance)
{
    constexpr float k_min_reflectance = 0.02f;
    const float r = std::clamp(reflectance, k_min_reflectance, 1.f);
    return ((1.f - r) * (1.f - r)) / (2.f * r);
}

float ks_to_reflectance(float ks)
{
    const float ratio = std::max(0.f, ks);
    return clamp01(1.f + ratio - std::sqrt(std::max(0.f, ratio * ratio + 2.f * ratio)));
}

std::array<float, 3> mix_linear_reflectance(const std::vector<std::array<float, 3>> &colors,
                                            const std::vector<float>                &weights)
{
    if (colors.empty() || colors.size() != weights.size())
        return { 0.f, 0.f, 0.f };

    if (const LutImage *image = lut_image(); image != nullptr)
        return mix_with_lut(colors, weights, *image);

    std::array<float, 3> accumulated_ks { 0.f, 0.f, 0.f };
    float total_weight = 0.f;
    for (size_t idx = 0; idx < colors.size(); ++idx) {
        const float weight = std::max(0.f, weights[idx]);
        if (!std::isfinite(weight) || weight <= 0.f)
            continue;

        accumulated_ks[0] += reflectance_to_ks(srgb_to_linear(colors[idx][0])) * weight;
        accumulated_ks[1] += reflectance_to_ks(srgb_to_linear(colors[idx][1])) * weight;
        accumulated_ks[2] += reflectance_to_ks(srgb_to_linear(colors[idx][2])) * weight;
        total_weight += weight;
    }

    if (total_weight <= 0.f)
        return colors.front();

    const float inv_total = 1.f / total_weight;
    return {
        linear_to_srgb(ks_to_reflectance(accumulated_ks[0] * inv_total)),
        linear_to_srgb(ks_to_reflectance(accumulated_ks[1] * inv_total)),
        linear_to_srgb(ks_to_reflectance(accumulated_ks[2] * inv_total))
    };
}

} // namespace

std::array<float, 3> mix_srgb(const std::vector<std::array<float, 3>> &colors,
                              const std::vector<float>                &weights)
{
    return mix_linear_reflectance(colors, weights);
}

std::array<float, 3> mix_srgb(const std::vector<std::array<float, 3>> &colors,
                              const std::vector<int>                  &weights)
{
    std::vector<float> float_weights;
    float_weights.reserve(weights.size());
    for (const int weight : weights)
        float_weights.emplace_back(float(std::max(0, weight)));
    return mix_linear_reflectance(colors, float_weights);
}

} // namespace pigment_painter
