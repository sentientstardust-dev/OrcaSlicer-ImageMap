#include "ImportedTexture.hpp"

#include "libslic3r/PNGReadWrite.hpp"

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <iterator>
#include <limits>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/nowide/fstream.hpp>

#include <jpeglib.h>

namespace Slic3r {

bool checked_rgba_buffer_size(size_t width, size_t height, size_t &buffer_size)
{
    buffer_size = 0;
    if (width == 0 || height == 0)
        return false;
    if (width > std::numeric_limits<size_t>::max() / height)
        return false;
    const size_t pixel_count = width * height;
    if (pixel_count > std::numeric_limits<size_t>::max() / 4)
        return false;
    buffer_size = pixel_count * 4;
    return true;
}

static bool decode_png_texture_rgba_from_memory(const uint8_t *data,
                                                size_t data_size,
                                                std::vector<uint8_t> &out_rgba,
                                                uint32_t &out_width,
                                                uint32_t &out_height)
{
    out_rgba.clear();
    out_width  = 0;
    out_height = 0;

    if (data == nullptr || data_size == 0)
        return false;

    png::ReadBuf      rb{reinterpret_cast<const char *>(data), data_size};
    png::ImageColorscale img;
    if (!png::decode_colored_png(rb, img))
        return false;

    if (img.cols == 0 || img.rows == 0 || (img.bytes_per_pixel != 3 && img.bytes_per_pixel != 4))
        return false;

    size_t rgba_size = 0;
    if (!checked_rgba_buffer_size(img.cols, img.rows, rgba_size))
        return false;

    const size_t row_stride = img.cols * size_t(img.bytes_per_pixel);
    if (img.buf.size() < img.rows * row_stride)
        return false;

    out_rgba.assign(rgba_size, 255);
    for (size_t y = 0; y < img.rows; ++y) {
        const size_t src_row_off = y * row_stride;
        const size_t dst_row_off = y * img.cols * 4;
        for (size_t x = 0; x < img.cols; ++x) {
            const size_t src = src_row_off + x * size_t(img.bytes_per_pixel);
            const size_t dst = dst_row_off + x * 4;
            out_rgba[dst + 0] = img.buf[src + 0];
            out_rgba[dst + 1] = img.buf[src + 1];
            out_rgba[dst + 2] = img.buf[src + 2];
            out_rgba[dst + 3] = (img.bytes_per_pixel == 4) ? img.buf[src + 3] : uint8_t(255);
        }
    }

    out_width  = uint32_t(img.cols);
    out_height = uint32_t(img.rows);
    return true;
}

struct JpegDecodeErrorManager
{
    jpeg_error_mgr pub;
    jmp_buf        setjmp_buffer;
};

static void jpeg_decode_error_exit(j_common_ptr cinfo)
{
    auto *err = reinterpret_cast<JpegDecodeErrorManager *>(cinfo->err);
    longjmp(err->setjmp_buffer, 1);
}

static bool decode_jpeg_texture_rgba_from_memory(const uint8_t *data,
                                                 size_t data_size,
                                                 std::vector<uint8_t> &out_rgba,
                                                 uint32_t &out_width,
                                                 uint32_t &out_height)
{
    out_rgba.clear();
    out_width  = 0;
    out_height = 0;

    if (data == nullptr || data_size == 0)
        return false;

    jpeg_decompress_struct cinfo{};
    JpegDecodeErrorManager jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_decode_error_exit;
    bool jpeg_created = false;
    auto destroy_jpeg = [&cinfo, &jpeg_created]() {
        if (jpeg_created) {
            jpeg_destroy_decompress(&cinfo);
            jpeg_created = false;
        }
    };

    if (setjmp(jerr.setjmp_buffer)) {
        destroy_jpeg();
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_created = true;
    jpeg_mem_src(&cinfo, data, static_cast<unsigned long>(data_size));

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        destroy_jpeg();
        return false;
    }

    if (!jpeg_start_decompress(&cinfo)) {
        destroy_jpeg();
        return false;
    }

    const uint32_t width = cinfo.output_width;
    const uint32_t height = cinfo.output_height;
    const int components = cinfo.output_components;
    size_t rgba_size = 0;
    const size_t scanline_stride = size_t(width) * size_t(std::max(components, 0));
    if (!checked_rgba_buffer_size(width, height, rgba_size) ||
        components <= 0 ||
        scanline_stride > std::numeric_limits<JDIMENSION>::max()) {
        jpeg_finish_decompress(&cinfo);
        destroy_jpeg();
        return false;
    }

    out_rgba.assign(rgba_size, uint8_t(255));
    JSAMPARRAY scanline = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo,
                                                     JPOOL_IMAGE,
                                                     JDIMENSION(scanline_stride),
                                                     1);

    uint32_t y = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, scanline, 1);
        const unsigned char *src = scanline[0];
        const uint32_t dst_y = height - 1 - y;
        for (uint32_t x = 0; x < width; ++x) {
            const size_t dst = (size_t(dst_y) * size_t(width) + size_t(x)) * 4;
            if (components >= 3) {
                const size_t s = size_t(x) * size_t(components);
                out_rgba[dst + 0] = src[s + 0];
                out_rgba[dst + 1] = src[s + 1];
                out_rgba[dst + 2] = src[s + 2];
            } else {
                const unsigned char g = src[x];
                out_rgba[dst + 0] = g;
                out_rgba[dst + 1] = g;
                out_rgba[dst + 2] = g;
            }
            out_rgba[dst + 3] = 255;
        }
        ++y;
    }

    if (!jpeg_finish_decompress(&cinfo)) {
        destroy_jpeg();
        return false;
    }
    destroy_jpeg();

    out_width  = width;
    out_height = height;
    return true;
}

static bool has_png_signature(const uint8_t *data, size_t data_size)
{
    static constexpr uint8_t signature[] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    return data_size >= sizeof(signature) && std::equal(std::begin(signature), std::end(signature), data);
}

static bool has_jpeg_signature(const uint8_t *data, size_t data_size)
{
    return data_size >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff;
}

bool decode_image_texture_rgba_from_file(const std::string &texture_path,
                                         std::vector<uint8_t> &out_rgba,
                                         uint32_t &out_width,
                                         uint32_t &out_height)
{
    out_rgba.clear();
    out_width  = 0;
    out_height = 0;

    if (!is_supported_image_texture_path(texture_path))
        return false;

    boost::nowide::ifstream ifs(texture_path, std::ios::binary);
    if (!ifs.is_open())
        return false;

    std::string encoded_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (encoded_data.empty())
        return false;

    return decode_image_texture_rgba_from_memory(reinterpret_cast<const uint8_t *>(encoded_data.data()),
                                                 encoded_data.size(),
                                                 texture_path,
                                                 out_rgba,
                                                 out_width,
                                                 out_height);
}

bool decode_image_texture_rgba_from_memory(const uint8_t *data,
                                           size_t data_size,
                                           const std::string &mime_type_or_name,
                                           std::vector<uint8_t> &out_rgba,
                                           uint32_t &out_width,
                                           uint32_t &out_height)
{
    out_rgba.clear();
    out_width  = 0;
    out_height = 0;

    if (data == nullptr || data_size == 0)
        return false;

    if (boost::algorithm::iequals(mime_type_or_name, "image/png") ||
        boost::algorithm::iends_with(mime_type_or_name, ".png") ||
        has_png_signature(data, data_size))
        return decode_png_texture_rgba_from_memory(data, data_size, out_rgba, out_width, out_height);

    if (boost::algorithm::iequals(mime_type_or_name, "image/jpeg") ||
        boost::algorithm::iequals(mime_type_or_name, "image/jpg") ||
        boost::algorithm::iends_with(mime_type_or_name, ".jpg") ||
        boost::algorithm::iends_with(mime_type_or_name, ".jpeg") ||
        has_jpeg_signature(data, data_size))
        return decode_jpeg_texture_rgba_from_memory(data, data_size, out_rgba, out_width, out_height);

    return false;
}

bool is_supported_image_texture_path(const std::string &texture_path)
{
    return boost::algorithm::iends_with(texture_path, ".png") ||
           boost::algorithm::iends_with(texture_path, ".jpg") ||
           boost::algorithm::iends_with(texture_path, ".jpeg");
}

struct ImportedTextureAtlasEntry
{
    uint32_t x_offset{0};
    uint32_t width{0};
    uint32_t height{0};
};

bool build_imported_texture_atlas(const std::vector<ImportedTextureImage> &textures,
                                  const std::vector<int> &triangle_texture_indices,
                                  std::vector<std::array<Vec2f, 3>> &triangle_uvs,
                                  std::vector<uint8_t> &triangle_uv_valid,
                                  std::vector<uint8_t> &atlas_rgba,
                                  uint32_t &atlas_width,
                                  uint32_t &atlas_height)
{
    atlas_rgba.clear();
    atlas_width  = 0;
    atlas_height = 0;

    if (textures.empty() || triangle_uvs.size() != triangle_uv_valid.size() || triangle_uvs.size() != triangle_texture_indices.size())
        return false;

    std::vector<ImportedTextureAtlasEntry> placements(textures.size());
    for (size_t i = 0; i < textures.size(); ++i) {
        const ImportedTextureImage &texture = textures[i];
        if (texture.width == 0 || texture.height == 0 || texture.rgba.empty())
            return false;
        size_t texture_rgba_size = 0;
        if (!checked_rgba_buffer_size(texture.width, texture.height, texture_rgba_size) ||
            texture.rgba.size() < texture_rgba_size ||
            texture.width > std::numeric_limits<uint32_t>::max() - atlas_width)
            return false;

        placements[i].x_offset = atlas_width;
        placements[i].width    = texture.width;
        placements[i].height   = texture.height;
        atlas_width += texture.width;
        atlas_height = std::max(atlas_height, texture.height);
    }

    size_t atlas_rgba_size = 0;
    if (atlas_width == 0 || atlas_height == 0 || !checked_rgba_buffer_size(atlas_width, atlas_height, atlas_rgba_size))
        return false;

    atlas_rgba.assign(atlas_rgba_size, uint8_t(0));
    for (size_t i = 0; i < textures.size(); ++i) {
        const ImportedTextureImage       &texture = textures[i];
        const ImportedTextureAtlasEntry  &entry   = placements[i];
        for (uint32_t y = 0; y < texture.height; ++y) {
            const size_t src_off = size_t(y) * size_t(texture.width) * 4;
            const size_t dst_off = (size_t(y) * size_t(atlas_width) + size_t(entry.x_offset)) * 4;
            std::copy(texture.rgba.begin() + src_off,
                      texture.rgba.begin() + src_off + size_t(texture.width) * 4,
                      atlas_rgba.begin() + dst_off);
        }
    }

    auto wrap_uv = [](float value) {
        if (!std::isfinite(value))
            return 0.f;

        constexpr float k_uv_epsilon = 1e-6f;
        if (value >= -k_uv_epsilon && value <= 1.f + k_uv_epsilon)
            return std::clamp(value, 0.f, 1.f);

        const float wrapped = value - std::floor(value);
        return wrapped < 0.f ? wrapped + 1.f : wrapped;
    };

    auto remap_uv = [&wrap_uv, &atlas_width, &atlas_height](const Vec2f &uv, const ImportedTextureAtlasEntry &entry) {
        const float u = wrap_uv(uv.x());
        const float v = wrap_uv(uv.y());
        return Vec2f((float(entry.x_offset) + u * float(entry.width)) / float(atlas_width),
                     v * float(entry.height) / float(atlas_height));
    };

    bool has_any_textured_triangle = false;
    for (size_t tri_idx = 0; tri_idx < triangle_uvs.size(); ++tri_idx) {
        triangle_uv_valid[tri_idx] = 0;
        const int texture_idx = triangle_texture_indices[tri_idx];
        if (texture_idx < 0 || size_t(texture_idx) >= textures.size())
            continue;

        const ImportedTextureAtlasEntry &entry = placements[size_t(texture_idx)];
        triangle_uvs[tri_idx][0] = remap_uv(triangle_uvs[tri_idx][0], entry);
        triangle_uvs[tri_idx][1] = remap_uv(triangle_uvs[tri_idx][1], entry);
        triangle_uvs[tri_idx][2] = remap_uv(triangle_uvs[tri_idx][2], entry);
        triangle_uv_valid[tri_idx] = 1;
        has_any_textured_triangle = true;
    }

    if (!has_any_textured_triangle) {
        atlas_rgba.clear();
        atlas_width  = 0;
        atlas_height = 0;
        return false;
    }

    return true;
}

} // namespace Slic3r
