// original author: sentientstardust

#include "ModelTextureDataRemap.hpp"

#include "AABBMesh.hpp"
#include "ImageMapRawFilamentOffsetAtlas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

namespace Slic3r {
namespace {

constexpr uint32_t GeneratedImageTextureSize = 4096;
constexpr uint32_t MaxSourceImageTextureSize = 4096;
constexpr int GeneratedImageTextureUvMapVersion = 1;
constexpr float RemapEpsilon = 1e-6f;

struct GeneratedImageTextureIsland
{
    size_t tri_idx = 0;
    std::array<Vec2f, 3> local_uvs;
    float width = 0.f;
    float height = 0.f;
    int rect_width = 0;
    int rect_height = 0;
    int x = 0;
    int y = 0;
};

struct GeneratedImageTextureAtlas
{
    int padding_px = 0;
    float scale = 0.f;
    std::vector<GeneratedImageTextureIsland> islands;
    std::vector<int> island_by_triangle;
};

struct RgbaFacetLookup
{
    std::unordered_map<int, std::vector<size_t>> by_triangle;
};

struct RegionFacetLookup
{
    std::vector<std::unordered_map<int, std::vector<size_t>>> by_state_and_triangle;
};

struct FastTriangleProjection
{
    bool enabled = false;
    size_t source_triangle = 0;
    std::array<Vec3f, 3> source_barycentrics;
};

static void check_cancel(const SimplifyTextureCancelFn &throw_on_cancel)
{
    if (throw_on_cancel)
        throw_on_cancel();
}

static void set_progress(const SimplifyTextureProgressFn &status_fn, int percent)
{
    if (status_fn)
        status_fn(std::clamp(percent, 0, 100));
}

static uint32_t pack_rgba(const ColorRGBA &color)
{
    auto to_u8 = [](float value) -> uint32_t {
        return uint32_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    const uint32_t r = to_u8(color.r());
    const uint32_t g = to_u8(color.g());
    const uint32_t b = to_u8(color.b());
    const uint32_t a = to_u8(color.a());
    return (r << 24) | (g << 16) | (b << 8) | a;
}

static ColorRGBA unpack_rgba(uint32_t packed)
{
    return ColorRGBA(float((packed >> 24) & 0xFFu) / 255.f,
                     float((packed >> 16) & 0xFFu) / 255.f,
                     float((packed >> 8) & 0xFFu) / 255.f,
                     float(packed & 0xFFu) / 255.f);
}

static unsigned int effective_region_state(unsigned int state, unsigned int base_filament_id)
{
    if (state == 0 || state == base_filament_id || state > unsigned(EnforcerBlockerType::ExtruderMax))
        return 0;
    return state;
}

static unsigned int volume_base_region_filament_id(const ModelVolume &volume)
{
    return unsigned(std::max(volume.extruder_id(), 1));
}

static bool valid_triangle_vertices(const indexed_triangle_set &its, size_t tri_idx, std::array<Vec3f, 3> &vertices)
{
    if (tri_idx >= its.indices.size())
        return false;
    const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
    if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
        return false;
    if (size_t(tri[0]) >= its.vertices.size() || size_t(tri[1]) >= its.vertices.size() || size_t(tri[2]) >= its.vertices.size())
        return false;
    vertices = { its.vertices[size_t(tri[0])].cast<float>(),
                 its.vertices[size_t(tri[1])].cast<float>(),
                 its.vertices[size_t(tri[2])].cast<float>() };
    return true;
}

static Vec3f normalized_nonnegative_barycentric(Vec3f weights)
{
    weights.x() = std::max(weights.x(), 0.f);
    weights.y() = std::max(weights.y(), 0.f);
    weights.z() = std::max(weights.z(), 0.f);
    const float sum = weights.x() + weights.y() + weights.z();
    if (sum <= RemapEpsilon)
        return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
    weights /= sum;
    return weights;
}

static Vec3f barycentric_weights_3d(const Vec3f &point, const std::array<Vec3f, 3> &vertices)
{
    const Vec3f v0 = vertices[1] - vertices[0];
    const Vec3f v1 = vertices[2] - vertices[0];
    const Vec3f v2 = point - vertices[0];
    const float d00 = v0.dot(v0);
    const float d01 = v0.dot(v1);
    const float d11 = v1.dot(v1);
    const float d20 = v2.dot(v0);
    const float d21 = v2.dot(v1);
    const float denom = d00 * d11 - d01 * d01;
    if (!std::isfinite(denom) || std::abs(denom) <= RemapEpsilon)
        return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    return Vec3f(1.f - v - w, v, w);
}

static Vec3f barycentric_weights_2d(const Vec2f &point, const std::array<Vec2f, 3> &vertices)
{
    const Vec2f v0 = vertices[1] - vertices[0];
    const Vec2f v1 = vertices[2] - vertices[0];
    const Vec2f v2 = point - vertices[0];
    const float d00 = v0.dot(v0);
    const float d01 = v0.dot(v1);
    const float d11 = v1.dot(v1);
    const float d20 = v2.dot(v0);
    const float d21 = v2.dot(v1);
    const float denom = d00 * d11 - d01 * d01;
    if (!std::isfinite(denom) || std::abs(denom) <= RemapEpsilon)
        return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    return Vec3f(1.f - v - w, v, w);
}

static float wrap_texture_uv(float uv)
{
    if (!std::isfinite(uv))
        return 0.f;
    float wrapped = uv - std::floor(uv);
    if (wrapped < 0.f)
        wrapped += 1.f;
    return wrapped;
}

static ColorRGBA sample_rgba_bilinear_wrapped(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height, const Vec2f &uv)
{
    if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    const float u = wrap_texture_uv(uv.x());
    const float v = wrap_texture_uv(uv.y());
    const float x = u * float(width > 1 ? width - 1 : 0);
    const float y = v * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto sample_channel = [&rgba, width](size_t sx, size_t sy, size_t channel) {
        const size_t idx = (sy * size_t(width) + sx) * 4 + channel;
        return float(rgba[idx]) / 255.f;
    };
    auto blend_channel = [&](size_t channel) {
        const float c00 = sample_channel(x0, y0, channel);
        const float c10 = sample_channel(x1, y0, channel);
        const float c01 = sample_channel(x0, y1, channel);
        const float c11 = sample_channel(x1, y1, channel);
        const float cx0 = c00 + (c10 - c00) * tx;
        const float cx1 = c01 + (c11 - c01) * tx;
        return std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 1.f);
    };

    return ColorRGBA(blend_channel(0), blend_channel(1), blend_channel(2), blend_channel(3));
}

static std::vector<uint8_t> sample_raw_offsets_bilinear_wrapped(const std::vector<uint8_t> &offsets,
                                                                uint32_t                    width,
                                                                uint32_t                    height,
                                                                uint32_t                    channels,
                                                                const Vec2f                &uv)
{
    std::vector<uint8_t> out(size_t(channels), 0);
    if (width == 0 || height == 0 || channels == 0 || offsets.size() < size_t(width) * size_t(height) * size_t(channels))
        return out;

    const float u = wrap_texture_uv(uv.x());
    const float v = wrap_texture_uv(uv.y());
    const float x = u * float(width > 1 ? width - 1 : 0);
    const float y = v * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto sample_channel = [&offsets, width, channels](size_t sx, size_t sy, size_t channel) {
        const size_t idx = (sy * size_t(width) + sx) * size_t(channels) + channel;
        return float(offsets[idx]);
    };
    for (size_t channel = 0; channel < size_t(channels); ++channel) {
        const float c00 = sample_channel(x0, y0, channel);
        const float c10 = sample_channel(x1, y0, channel);
        const float c01 = sample_channel(x0, y1, channel);
        const float c11 = sample_channel(x1, y1, channel);
        const float cx0 = c00 + (c10 - c00) * tx;
        const float cx1 = c01 + (c11 - c01) * tx;
        out[channel] = uint8_t(std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 255.f) + 0.5f);
    }
    return out;
}

static std::vector<uint8_t> resized_rgba_bilinear(const std::vector<uint8_t> &rgba,
                                                  uint32_t                    width,
                                                  uint32_t                    height,
                                                  uint32_t                    resized_width,
                                                  uint32_t                    resized_height)
{
    std::vector<uint8_t> resized(size_t(resized_width) * size_t(resized_height) * 4, 255);
    if (width == 0 || height == 0 || resized_width == 0 || resized_height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return resized;

    for (uint32_t y = 0; y < resized_height; ++y) {
        for (uint32_t x = 0; x < resized_width; ++x) {
            const Vec2f uv((float(x) + 0.5f) / float(resized_width), (float(y) + 0.5f) / float(resized_height));
            const ColorRGBA color = sample_rgba_bilinear_wrapped(rgba, width, height, uv);
            const size_t idx = (size_t(y) * size_t(resized_width) + size_t(x)) * 4;
            resized[idx + 0] = uint8_t(std::clamp(color.r(), 0.f, 1.f) * 255.f + 0.5f);
            resized[idx + 1] = uint8_t(std::clamp(color.g(), 0.f, 1.f) * 255.f + 0.5f);
            resized[idx + 2] = uint8_t(std::clamp(color.b(), 0.f, 1.f) * 255.f + 0.5f);
            resized[idx + 3] = uint8_t(std::clamp(color.a(), 0.f, 1.f) * 255.f + 0.5f);
        }
    }
    return resized;
}

static std::vector<uint8_t> resized_raw_offsets_bilinear(const std::vector<uint8_t> &offsets,
                                                         uint32_t                    width,
                                                         uint32_t                    height,
                                                         uint32_t                    channels,
                                                         uint32_t                    resized_width,
                                                         uint32_t                    resized_height)
{
    std::vector<uint8_t> resized(size_t(resized_width) * size_t(resized_height) * size_t(channels), 0);
    if (width == 0 || height == 0 || channels == 0 || resized_width == 0 || resized_height == 0 ||
        offsets.size() < size_t(width) * size_t(height) * size_t(channels))
        return resized;

    for (uint32_t y = 0; y < resized_height; ++y) {
        for (uint32_t x = 0; x < resized_width; ++x) {
            const Vec2f uv((float(x) + 0.5f) / float(resized_width), (float(y) + 0.5f) / float(resized_height));
            const float old_x = wrap_texture_uv(uv.x()) * float(width > 1 ? width - 1 : 0);
            const float old_y = wrap_texture_uv(uv.y()) * float(height > 1 ? height - 1 : 0);
            const size_t x0 = std::min<size_t>(size_t(std::floor(old_x)), size_t(width - 1));
            const size_t y0 = std::min<size_t>(size_t(std::floor(old_y)), size_t(height - 1));
            const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
            const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
            const float tx = old_x - float(x0);
            const float ty = old_y - float(y0);
            const size_t idx = (size_t(y) * size_t(resized_width) + size_t(x)) * size_t(channels);
            for (size_t channel = 0; channel < size_t(channels); ++channel) {
                auto sample_channel = [&offsets, width, channels, channel](size_t sx, size_t sy) {
                    return float(offsets[(sy * size_t(width) + sx) * size_t(channels) + channel]);
                };
                const float c00 = sample_channel(x0, y0);
                const float c10 = sample_channel(x1, y0);
                const float c01 = sample_channel(x0, y1);
                const float c11 = sample_channel(x1, y1);
                const float cx0 = c00 + (c10 - c00) * tx;
                const float cx1 = c01 + (c11 - c01) * tx;
                resized[idx + channel] = uint8_t(std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 255.f) + 0.5f);
            }
        }
    }
    return resized;
}

static Vec3f texture_barycentric_for_bleed_safe_sampling(const Vec3f &barycentric,
                                                         const Vec2f &uv0,
                                                         const Vec2f &uv1,
                                                         const Vec2f &uv2,
                                                         uint32_t     width,
                                                         uint32_t     height)
{
    Vec3f safe = normalized_nonnegative_barycentric(barycentric);
    if (width == 0 || height == 0)
        return safe;

    auto pixel_edge_length = [width, height](const Vec2f &a, const Vec2f &b) {
        const Vec2f delta = b - a;
        return std::sqrt(Slic3r::sqr(delta.x() * float(width)) + Slic3r::sqr(delta.y() * float(height)));
    };
    const float max_edge = std::max({ pixel_edge_length(uv0, uv1), pixel_edge_length(uv1, uv2), pixel_edge_length(uv2, uv0) });
    if (!std::isfinite(max_edge) || max_edge <= RemapEpsilon)
        return safe;

    const float min_barycentric = std::min(0.08f, 0.75f / max_edge);
    if (min_barycentric <= 0.f)
        return safe;

    safe.x() = std::max(safe.x(), min_barycentric);
    safe.y() = std::max(safe.y(), min_barycentric);
    safe.z() = std::max(safe.z(), min_barycentric);
    return normalized_nonnegative_barycentric(safe);
}

static void unwrap_uvs(std::array<Vec2f, 3> &uvs)
{
    for (size_t idx = 1; idx < uvs.size(); ++idx) {
        while (uvs[idx].x() - uvs[0].x() > 0.5f)
            uvs[idx].x() -= 1.f;
        while (uvs[idx].x() - uvs[0].x() < -0.5f)
            uvs[idx].x() += 1.f;
        while (uvs[idx].y() - uvs[0].y() > 0.5f)
            uvs[idx].y() -= 1.f;
        while (uvs[idx].y() - uvs[0].y() < -0.5f)
            uvs[idx].y() += 1.f;
    }
}

static bool snapshot_has_valid_raw_atlas(const SimplifyTextureDataSnapshot &snapshot)
{
    return snapshot.texture_width > 0 &&
           snapshot.texture_height > 0 &&
           snapshot.texture_raw_channels > 0 &&
           snapshot.texture_raw_filament_offsets.size() >=
               size_t(snapshot.texture_width) * size_t(snapshot.texture_height) * size_t(snapshot.texture_raw_channels);
}

static bool snapshot_has_valid_rgba_texture(const SimplifyTextureDataSnapshot &snapshot)
{
    return snapshot.texture_width > 0 &&
           snapshot.texture_height > 0 &&
           snapshot.texture_rgba.size() >= size_t(snapshot.texture_width) * size_t(snapshot.texture_height) * 4;
}

static void limit_snapshot_texture_resolution(SimplifyTextureDataSnapshot &snapshot)
{
    if (snapshot.texture_width <= MaxSourceImageTextureSize && snapshot.texture_height <= MaxSourceImageTextureSize)
        return;
    if (snapshot.texture_width == 0 || snapshot.texture_height == 0)
        return;

    const uint32_t old_width = snapshot.texture_width;
    const uint32_t old_height = snapshot.texture_height;
    const float scale = std::min(float(MaxSourceImageTextureSize) / float(old_width),
                                 float(MaxSourceImageTextureSize) / float(old_height));
    const uint32_t resized_width = std::max<uint32_t>(1, uint32_t(std::round(float(old_width) * scale)));
    const uint32_t resized_height = std::max<uint32_t>(1, uint32_t(std::round(float(old_height) * scale)));

    if (snapshot_has_valid_rgba_texture(snapshot))
        snapshot.texture_rgba = resized_rgba_bilinear(snapshot.texture_rgba, old_width, old_height, resized_width, resized_height);
    if (snapshot_has_valid_raw_atlas(snapshot))
        snapshot.texture_raw_filament_offsets =
            resized_raw_offsets_bilinear(snapshot.texture_raw_filament_offsets,
                                         old_width,
                                         old_height,
                                         snapshot.texture_raw_channels,
                                         resized_width,
                                         resized_height);

    snapshot.texture_width = resized_width;
    snapshot.texture_height = resized_height;
}

static bool volume_has_valid_raw_atlas(const ModelVolume &volume)
{
    return volume.imported_texture_width > 0 &&
           volume.imported_texture_height > 0 &&
           volume.imported_texture_raw_channels > 0 &&
           volume.imported_texture_raw_filament_offsets.size() >=
               size_t(volume.imported_texture_width) * size_t(volume.imported_texture_height) * size_t(volume.imported_texture_raw_channels);
}

static bool volume_has_valid_image_texture(const ModelVolume &volume)
{
    const indexed_triangle_set &its = volume.mesh().its;
    if (its.vertices.empty() || its.indices.empty())
        return false;
    if (volume.imported_texture_uv_valid.size() != its.indices.size() || volume.imported_texture_uvs_per_face.size() < its.indices.size() * 6)
        return false;
    if (!std::any_of(volume.imported_texture_uv_valid.begin(), volume.imported_texture_uv_valid.end(), [](uint8_t valid) { return valid != 0; }))
        return false;
    const bool valid_rgba = volume.imported_texture_width > 0 &&
                            volume.imported_texture_height > 0 &&
                            volume.imported_texture_rgba.size() >=
                                size_t(volume.imported_texture_width) * size_t(volume.imported_texture_height) * 4;
    return valid_rgba || volume_has_valid_raw_atlas(volume);
}

static bool region_painting_data_needs_remap(const ModelVolume &volume)
{
    if (volume.mmu_segmentation_facets.empty())
        return false;

    const unsigned int base_filament_id = volume_base_region_filament_id(volume);
    const auto &used_states = volume.mmu_segmentation_facets.get_data().used_states;
    for (size_t state_idx = 0; state_idx < used_states.size(); ++state_idx)
        if (used_states[state_idx] && effective_region_state(unsigned(state_idx), base_filament_id) != 0)
            return true;
    return false;
}

static std::array<Vec2f, 3> source_triangle_uvs(const SimplifyTextureDataSnapshot &snapshot, size_t tri_idx, bool *valid = nullptr)
{
    std::array<Vec2f, 3> uvs = { Vec2f::Zero(), Vec2f::Zero(), Vec2f::Zero() };
    bool ok = tri_idx < snapshot.texture_uv_valid.size() &&
              snapshot.texture_uv_valid[tri_idx] != 0 &&
              tri_idx * 6 + 5 < snapshot.texture_uvs_per_face.size();
    if (ok) {
        const size_t uv_offset = tri_idx * 6;
        uvs = { Vec2f(snapshot.texture_uvs_per_face[uv_offset + 0], snapshot.texture_uvs_per_face[uv_offset + 1]),
                Vec2f(snapshot.texture_uvs_per_face[uv_offset + 2], snapshot.texture_uvs_per_face[uv_offset + 3]),
                Vec2f(snapshot.texture_uvs_per_face[uv_offset + 4], snapshot.texture_uvs_per_face[uv_offset + 5]) };
        unwrap_uvs(uvs);
    }
    if (valid != nullptr)
        *valid = ok;
    return uvs;
}

static ColorRGBA sample_image_rgba_at_source(const SimplifyTextureDataSnapshot &snapshot, size_t tri_idx, const Vec3f &barycentric)
{
    bool uv_valid = false;
    const std::array<Vec2f, 3> uvs = source_triangle_uvs(snapshot, tri_idx, &uv_valid);
    if (!uv_valid)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    const Vec3f safe = texture_barycentric_for_bleed_safe_sampling(
        barycentric, uvs[0], uvs[1], uvs[2], snapshot.texture_width, snapshot.texture_height);
    const Vec2f uv = uvs[0] * safe.x() + uvs[1] * safe.y() + uvs[2] * safe.z();
    return sample_rgba_bilinear_wrapped(snapshot.texture_rgba, snapshot.texture_width, snapshot.texture_height, uv);
}

static std::vector<uint8_t> sample_image_raw_at_source(const SimplifyTextureDataSnapshot &snapshot, size_t tri_idx, const Vec3f &barycentric)
{
    bool uv_valid = false;
    const std::array<Vec2f, 3> uvs = source_triangle_uvs(snapshot, tri_idx, &uv_valid);
    if (!uv_valid)
        return std::vector<uint8_t>(size_t(snapshot.texture_raw_channels), 0);
    const Vec3f safe = texture_barycentric_for_bleed_safe_sampling(
        barycentric, uvs[0], uvs[1], uvs[2], snapshot.texture_width, snapshot.texture_height);
    const Vec2f uv = uvs[0] * safe.x() + uvs[1] * safe.y() + uvs[2] * safe.z();
    return sample_raw_offsets_bilinear_wrapped(snapshot.texture_raw_filament_offsets,
                                               snapshot.texture_width,
                                               snapshot.texture_height,
                                               snapshot.texture_raw_channels,
                                               uv);
}

static bool project_to_source(const SimplifyTextureDataSnapshot &snapshot,
                              const AABBMesh                   &aabb,
                              const Vec3f                      &point,
                              size_t                           &source_triangle,
                              Vec3f                            &source_barycentric)
{
    int tri_idx = -1;
    Vec3d closest = Vec3d::Zero();
    const double squared_distance = aabb.squared_distance(point.cast<double>(), tri_idx, closest);
    if (tri_idx < 0 || !std::isfinite(squared_distance) || size_t(tri_idx) >= snapshot.source_mesh.indices.size())
        return false;

    std::array<Vec3f, 3> vertices;
    if (!valid_triangle_vertices(snapshot.source_mesh, size_t(tri_idx), vertices))
        return false;

    source_triangle = size_t(tri_idx);
    source_barycentric = normalized_nonnegative_barycentric(barycentric_weights_3d(closest.cast<float>(), vertices));
    return true;
}

static FastTriangleProjection make_fast_triangle_projection(const SimplifyTextureDataSnapshot &snapshot,
                                                            const AABBMesh                   &aabb,
                                                            const std::array<Vec3f, 3>       &target_vertices)
{
    FastTriangleProjection projection;
    size_t first_source_triangle = 0;
    for (size_t corner = 0; corner < 3; ++corner) {
        size_t source_triangle = 0;
        Vec3f source_barycentric = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
        if (!project_to_source(snapshot, aabb, target_vertices[corner], source_triangle, source_barycentric))
            return projection;
        if (corner == 0)
            first_source_triangle = source_triangle;
        else if (source_triangle != first_source_triangle)
            return projection;
        projection.source_barycentrics[corner] = source_barycentric;
    }

    projection.enabled = true;
    projection.source_triangle = first_source_triangle;
    return projection;
}

static bool projected_source_from_fast_triangle(const FastTriangleProjection &projection,
                                                const Vec3f                  &target_barycentric,
                                                size_t                       &source_triangle,
                                                Vec3f                        &source_barycentric)
{
    if (!projection.enabled)
        return false;
    source_triangle = projection.source_triangle;
    source_barycentric = normalized_nonnegative_barycentric(projection.source_barycentrics[0] * target_barycentric.x() +
                                                            projection.source_barycentrics[1] * target_barycentric.y() +
                                                            projection.source_barycentrics[2] * target_barycentric.z());
    return true;
}

static bool make_generated_image_texture_island(const indexed_triangle_set &its,
                                                size_t                      tri_idx,
                                                GeneratedImageTextureIsland &island)
{
    std::array<Vec3f, 3> vertices;
    if (!valid_triangle_vertices(its, tri_idx, vertices))
        return false;

    const float edge_01 = (vertices[1] - vertices[0]).norm();
    const float edge_12 = (vertices[2] - vertices[1]).norm();
    const float edge_20 = (vertices[0] - vertices[2]).norm();

    int a = 0;
    int b = 1;
    int c = 2;
    float base_length = edge_01;
    if (edge_12 > base_length) {
        a = 1;
        b = 2;
        c = 0;
        base_length = edge_12;
    }
    if (edge_20 > base_length) {
        a = 2;
        b = 0;
        c = 1;
        base_length = edge_20;
    }
    if (!std::isfinite(base_length) || base_length <= RemapEpsilon)
        return false;

    const Vec3f base = vertices[b] - vertices[a];
    const Vec3f side = vertices[c] - vertices[a];
    const float projected = side.dot(base) / base_length;
    const float height_sq = std::max(0.f, side.squaredNorm() - projected * projected);
    const float height = std::sqrt(height_sq);

    std::array<Vec2f, 3> local_uvs;
    local_uvs[size_t(a)] = Vec2f(0.f, 0.f);
    local_uvs[size_t(b)] = Vec2f(base_length, 0.f);
    local_uvs[size_t(c)] = Vec2f(projected, height);

    const float min_x = std::min({ local_uvs[0].x(), local_uvs[1].x(), local_uvs[2].x() });
    const float min_y = std::min({ local_uvs[0].y(), local_uvs[1].y(), local_uvs[2].y() });
    const float max_x = std::max({ local_uvs[0].x(), local_uvs[1].x(), local_uvs[2].x() });
    const float max_y = std::max({ local_uvs[0].y(), local_uvs[1].y(), local_uvs[2].y() });
    for (Vec2f &uv : local_uvs)
        uv -= Vec2f(min_x, min_y);

    island.tri_idx = tri_idx;
    island.local_uvs = local_uvs;
    island.width = std::max(max_x - min_x, 1e-4f);
    island.height = std::max(max_y - min_y, 1e-4f);
    return std::isfinite(island.width) && std::isfinite(island.height);
}

static bool pack_generated_image_texture_islands(std::vector<GeneratedImageTextureIsland> &islands,
                                                 uint32_t                                  texture_size,
                                                 int                                       padding_px,
                                                 float                                     scale)
{
    if (islands.empty() || texture_size == 0 || !std::isfinite(scale) || scale <= 0.f)
        return false;

    const int atlas_size = int(texture_size);
    for (GeneratedImageTextureIsland &island : islands) {
        const int content_width = std::max(1, int(std::ceil(island.width * scale))) + 1;
        const int content_height = std::max(1, int(std::ceil(island.height * scale))) + 1;
        island.rect_width = content_width + padding_px * 2;
        island.rect_height = content_height + padding_px * 2;
        if (island.rect_width > atlas_size || island.rect_height > atlas_size)
            return false;
    }

    std::vector<size_t> order(islands.size());
    for (size_t idx = 0; idx < order.size(); ++idx)
        order[idx] = idx;
    std::sort(order.begin(), order.end(), [&islands](size_t lhs, size_t rhs) {
        const GeneratedImageTextureIsland &a = islands[lhs];
        const GeneratedImageTextureIsland &b = islands[rhs];
        if (a.rect_height != b.rect_height)
            return a.rect_height > b.rect_height;
        if (a.rect_width != b.rect_width)
            return a.rect_width > b.rect_width;
        return a.tri_idx < b.tri_idx;
    });

    int x = 0;
    int y = 0;
    int row_height = 0;
    for (const size_t island_idx : order) {
        GeneratedImageTextureIsland &island = islands[island_idx];
        if (x + island.rect_width > atlas_size) {
            y += row_height;
            x = 0;
            row_height = 0;
        }
        if (y + island.rect_height > atlas_size)
            return false;
        island.x = x;
        island.y = y;
        x += island.rect_width;
        row_height = std::max(row_height, island.rect_height);
    }

    return true;
}

static bool pack_generated_image_texture_atlas(GeneratedImageTextureAtlas &atlas, uint32_t texture_size)
{
    if (atlas.islands.empty())
        return false;

    float max_dimension = 0.f;
    for (const GeneratedImageTextureIsland &island : atlas.islands)
        max_dimension = std::max({ max_dimension, island.width, island.height });
    if (!std::isfinite(max_dimension) || max_dimension <= RemapEpsilon)
        return false;

    const std::array<int, 3> padding_options = { 2, 1, 0 };
    for (const int padding_px : padding_options) {
        const float available = float(int(texture_size) - padding_px * 2 - 1);
        if (available <= 0.f)
            continue;

        float high = available / max_dimension;
        if (!std::isfinite(high) || high <= 0.f)
            continue;

        bool found = false;
        std::vector<GeneratedImageTextureIsland> best;
        float best_scale = 0.f;
        float upper = high;
        for (int attempt = 0; attempt < 12; ++attempt) {
            std::vector<GeneratedImageTextureIsland> candidate = atlas.islands;
            if (pack_generated_image_texture_islands(candidate, texture_size, padding_px, high)) {
                found = true;
                best = std::move(candidate);
                best_scale = high;
                break;
            }
            upper = high;
            high *= 0.5f;
        }
        if (!found)
            continue;

        float low = best_scale;
        high = upper;
        for (int iter = 0; iter < 24; ++iter) {
            const float mid = (low + high) * 0.5f;
            std::vector<GeneratedImageTextureIsland> candidate = atlas.islands;
            if (pack_generated_image_texture_islands(candidate, texture_size, padding_px, mid)) {
                low = mid;
                best = std::move(candidate);
                best_scale = mid;
            } else {
                high = mid;
            }
        }

        atlas.padding_px = padding_px;
        atlas.scale = best_scale;
        atlas.islands = std::move(best);
        atlas.island_by_triangle.assign(atlas.island_by_triangle.size(), -1);
        for (size_t island_idx = 0; island_idx < atlas.islands.size(); ++island_idx)
            if (atlas.islands[island_idx].tri_idx < atlas.island_by_triangle.size())
                atlas.island_by_triangle[atlas.islands[island_idx].tri_idx] = int(island_idx);
        return true;
    }

    return false;
}

static bool initialize_generated_image_texture(const indexed_triangle_set       &its,
                                               SimplifyTextureDataResult       &result,
                                               GeneratedImageTextureAtlas      &atlas,
                                               const ColorRGBA                 &background)
{
    if (its.vertices.empty() || its.indices.empty())
        return false;

    atlas.island_by_triangle.assign(its.indices.size(), -1);
    atlas.islands.reserve(its.indices.size());
    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        GeneratedImageTextureIsland island;
        if (make_generated_image_texture_island(its, tri_idx, island))
            atlas.islands.emplace_back(island);
    }
    if (!pack_generated_image_texture_atlas(atlas, GeneratedImageTextureSize))
        return false;

    const uint8_t r = uint8_t(std::clamp(background.r(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t g = uint8_t(std::clamp(background.g(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t b = uint8_t(std::clamp(background.b(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t a = uint8_t(std::clamp(background.a(), 0.f, 1.f) * 255.f + 0.5f);
    const size_t pixel_count = size_t(GeneratedImageTextureSize) * size_t(GeneratedImageTextureSize);

    result.texture_width = GeneratedImageTextureSize;
    result.texture_height = GeneratedImageTextureSize;
    result.uv_map_generator_version = GeneratedImageTextureUvMapVersion;
    result.texture_rgba.assign(pixel_count * 4, 0);
    for (size_t idx = 0; idx < pixel_count; ++idx) {
        result.texture_rgba[idx * 4 + 0] = r;
        result.texture_rgba[idx * 4 + 1] = g;
        result.texture_rgba[idx * 4 + 2] = b;
        result.texture_rgba[idx * 4 + 3] = a;
    }
    result.texture_uv_valid.assign(its.indices.size(), 0);
    result.texture_uvs_per_face.assign(its.indices.size() * 6, 0.f);

    const float texture_size = float(GeneratedImageTextureSize);
    for (const GeneratedImageTextureIsland &island : atlas.islands) {
        if (island.tri_idx >= its.indices.size())
            continue;
        result.texture_uv_valid[island.tri_idx] = 1;
        const size_t uv_offset = island.tri_idx * 6;
        for (size_t corner = 0; corner < 3; ++corner) {
            const Vec2f pixel(float(island.x + atlas.padding_px) + 0.5f + island.local_uvs[corner].x() * atlas.scale,
                              float(island.y + atlas.padding_px) + 0.5f + island.local_uvs[corner].y() * atlas.scale);
            result.texture_uvs_per_face[uv_offset + corner * 2 + 0] = std::clamp(pixel.x() / texture_size, 0.f, 1.f);
            result.texture_uvs_per_face[uv_offset + corner * 2 + 1] = std::clamp(pixel.y() / texture_size, 0.f, 1.f);
        }
    }

    return true;
}

static bool write_rgba_pixel(std::vector<uint8_t> &rgba, uint32_t width, uint32_t x, uint32_t y, const ColorRGBA &color)
{
    if (width == 0)
        return false;
    const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4;
    if (idx + 3 >= rgba.size())
        return false;
    rgba[idx + 0] = uint8_t(std::clamp(color.r(), 0.f, 1.f) * 255.f + 0.5f);
    rgba[idx + 1] = uint8_t(std::clamp(color.g(), 0.f, 1.f) * 255.f + 0.5f);
    rgba[idx + 2] = uint8_t(std::clamp(color.b(), 0.f, 1.f) * 255.f + 0.5f);
    rgba[idx + 3] = uint8_t(std::clamp(color.a(), 0.f, 1.f) * 255.f + 0.5f);
    return true;
}

static bool write_raw_offset_pixel(std::vector<uint8_t>       &offsets,
                                   uint32_t                    width,
                                   uint32_t                    channels,
                                   uint32_t                    x,
                                   uint32_t                    y,
                                   const std::vector<uint8_t> &values)
{
    if (width == 0 || channels == 0 || values.empty())
        return false;
    const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * size_t(channels);
    if (idx + size_t(channels) > offsets.size())
        return false;
    for (size_t channel = 0; channel < size_t(channels); ++channel)
        offsets[idx + channel] = channel < values.size() ? values[channel] : 0;
    return true;
}

static RgbaFacetLookup make_rgba_facet_lookup(const SimplifyTextureDataSnapshot &snapshot)
{
    RgbaFacetLookup lookup;
    lookup.by_triangle.reserve(snapshot.rgba_facets.size());
    for (size_t idx = 0; idx < snapshot.rgba_facets.size(); ++idx)
        lookup.by_triangle[snapshot.rgba_facets[idx].source_triangle].emplace_back(idx);
    return lookup;
}

static RegionFacetLookup make_region_facet_lookup(const SimplifyTextureDataSnapshot &snapshot)
{
    RegionFacetLookup lookup;
    lookup.by_state_and_triangle.resize(snapshot.region_facets_per_type.size());
    for (size_t state_idx = 0; state_idx < snapshot.region_facets_per_type.size(); ++state_idx) {
        const auto &facets = snapshot.region_facets_per_type[state_idx];
        auto &by_triangle = lookup.by_state_and_triangle[state_idx];
        by_triangle.reserve(facets.size());
        for (size_t facet_idx = 0; facet_idx < facets.size(); ++facet_idx)
            by_triangle[facets[facet_idx].source_triangle].emplace_back(facet_idx);
    }
    return lookup;
}

static ColorRGBA rgb_metadata_background_color(const std::string &metadata)
{
    const std::string key = "\"background_color\":\"#";
    const size_t start = metadata.find(key);
    if (start == std::string::npos || start + key.size() + 8 > metadata.size())
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    uint32_t packed = 0;
    for (size_t idx = 0; idx < 8; ++idx) {
        const char ch = metadata[start + key.size() + idx];
        const int value = ch >= '0' && ch <= '9' ? ch - '0' :
                          ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 :
                          ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
        if (value < 0)
            return ColorRGBA(1.f, 1.f, 1.f, 1.f);
        packed = (packed << 4) | uint32_t(value);
    }
    return unpack_rgba(packed);
}

static std::string rgb_metadata_json(const ColorRGBA &background)
{
    const uint32_t packed = pack_rgba(background);
    char buffer[48];
    std::snprintf(buffer,
                  sizeof(buffer),
                  "{\"background_color\":\"#%02X%02X%02X%02X\"}",
                  unsigned((packed >> 24) & 0xFFu),
                  unsigned((packed >> 16) & 0xFFu),
                  unsigned((packed >> 8) & 0xFFu),
                  unsigned(packed & 0xFFu));
    return buffer;
}

static ColorRGBA sample_rgba_facets_at_source(const SimplifyTextureDataSnapshot &snapshot,
                                              const RgbaFacetLookup            &lookup,
                                              size_t                            source_triangle,
                                              const Vec3f                      &point)
{
    const auto range_it = lookup.by_triangle.find(int(source_triangle));
    const ColorRGBA background = rgb_metadata_background_color(snapshot.rgba_metadata_json);
    if (range_it == lookup.by_triangle.end())
        return background;

    float best_distance = std::numeric_limits<float>::max();
    uint32_t best_rgba = pack_rgba(background);
    for (const size_t facet_idx : range_it->second) {
        const ColorFacetTriangle &facet = snapshot.rgba_facets[facet_idx];
        const Vec3f bary = barycentric_weights_3d(point, facet.vertices);
        const float min_weight = std::min({ bary.x(), bary.y(), bary.z() });
        if (min_weight >= -1e-4f)
            return unpack_rgba(facet.rgba);

        const Vec3f safe = normalized_nonnegative_barycentric(bary);
        const Vec3f closest = facet.vertices[0] * safe.x() + facet.vertices[1] * safe.y() + facet.vertices[2] * safe.z();
        const float distance = (closest - point).squaredNorm();
        if (distance < best_distance) {
            best_distance = distance;
            best_rgba = facet.rgba;
        }
    }
    return unpack_rgba(best_rgba);
}

static unsigned int sample_region_state_at_source(const SimplifyTextureDataSnapshot &snapshot,
                                                  const RegionFacetLookup           &lookup,
                                                  size_t                             source_triangle,
                                                  const Vec3f                       &point)
{
    float best_distance = std::numeric_limits<float>::max();
    unsigned int best_state = 0;
    bool found_candidate = false;
    for (size_t state_idx = 0; state_idx < snapshot.region_facets_per_type.size(); ++state_idx) {
        if (state_idx >= lookup.by_state_and_triangle.size())
            continue;
        const auto range_it = lookup.by_state_and_triangle[state_idx].find(int(source_triangle));
        if (range_it == lookup.by_state_and_triangle[state_idx].end())
            continue;

        const auto &facets = snapshot.region_facets_per_type[state_idx];
        for (const size_t facet_idx : range_it->second) {
            if (facet_idx >= facets.size())
                continue;
            const TriangleSelector::FacetStateTriangle &facet = facets[facet_idx];
            const Vec3f bary = barycentric_weights_3d(point, facet.vertices);
            const float min_weight = std::min({ bary.x(), bary.y(), bary.z() });
            const unsigned int state = effective_region_state(unsigned(state_idx), unsigned(snapshot.region_base_filament_id));
            if (min_weight >= -1e-4f)
                return state;

            const Vec3f safe = normalized_nonnegative_barycentric(bary);
            const Vec3f closest = facet.vertices[0] * safe.x() + facet.vertices[1] * safe.y() + facet.vertices[2] * safe.z();
            const float distance = (closest - point).squaredNorm();
            if (distance < best_distance) {
                best_distance = distance;
                best_state = state;
                found_candidate = true;
            }
        }
    }
    return found_candidate ? best_state : 0;
}

static ColorRGBA sample_vertex_color_at_source(const SimplifyTextureDataSnapshot &snapshot, size_t source_triangle, const Vec3f &barycentric)
{
    if (source_triangle >= snapshot.source_mesh.indices.size())
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    const stl_triangle_vertex_indices &tri = snapshot.source_mesh.indices[source_triangle];
    if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    if (size_t(tri[0]) >= snapshot.vertex_colors_rgba.size() ||
        size_t(tri[1]) >= snapshot.vertex_colors_rgba.size() ||
        size_t(tri[2]) >= snapshot.vertex_colors_rgba.size())
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    const ColorRGBA c0 = unpack_rgba(snapshot.vertex_colors_rgba[size_t(tri[0])]);
    const ColorRGBA c1 = unpack_rgba(snapshot.vertex_colors_rgba[size_t(tri[1])]);
    const ColorRGBA c2 = unpack_rgba(snapshot.vertex_colors_rgba[size_t(tri[2])]);
    const Vec3f safe = normalized_nonnegative_barycentric(barycentric);
    return ColorRGBA(c0.r() * safe.x() + c1.r() * safe.y() + c2.r() * safe.z(),
                     c0.g() * safe.x() + c1.g() * safe.y() + c2.g() * safe.z(),
                     c0.b() * safe.x() + c1.b() * safe.y() + c2.b() * safe.z(),
                     c0.a() * safe.x() + c1.a() * safe.y() + c2.a() * safe.z());
}

static ColorRGBA sample_snapshot_color_at_source(const SimplifyTextureDataSnapshot &snapshot,
                                                 const RgbaFacetLookup            *rgba_lookup,
                                                 size_t                            source_triangle,
                                                 const Vec3f                      &source_point,
                                                 const Vec3f                      &source_barycentric)
{
    switch (snapshot.source) {
    case SimplifyColorSource::RgbaData:
        return rgba_lookup != nullptr ? sample_rgba_facets_at_source(snapshot, *rgba_lookup, source_triangle, source_point) :
                                        ColorRGBA(1.f, 1.f, 1.f, 1.f);
    case SimplifyColorSource::ImageTexture:
        return sample_image_rgba_at_source(snapshot, source_triangle, source_barycentric);
    case SimplifyColorSource::VertexColors:
        return sample_vertex_color_at_source(snapshot, source_triangle, source_barycentric);
    case SimplifyColorSource::None:
        break;
    }
    return ColorRGBA(1.f, 1.f, 1.f, 1.f);
}

static float triangle_max_edge_length(const std::array<Vec3f, 3> &vertices)
{
    return std::max({ (vertices[1] - vertices[0]).norm(), (vertices[2] - vertices[1]).norm(), (vertices[0] - vertices[2]).norm() });
}

static float mesh_max_axis_span(const indexed_triangle_set &its)
{
    if (its.vertices.empty())
        return 1.f;

    Vec3f min_point = its.vertices.front().cast<float>();
    Vec3f max_point = min_point;
    for (const stl_vertex &vertex : its.vertices) {
        const Vec3f point = vertex.cast<float>();
        min_point = min_point.cwiseMin(point);
        max_point = max_point.cwiseMax(point);
    }

    const Vec3f span = max_point - min_point;
    return std::max({ span.x(), span.y(), span.z(), 1.f });
}

static int texture_mapping_depth_from_span(float span, float target_span, int max_depth)
{
    if (!std::isfinite(span) || !std::isfinite(target_span) || span <= target_span || target_span <= RemapEpsilon)
        return 0;
    return std::clamp(int(std::ceil(std::log2(span / target_span))), 0, max_depth);
}

static void region_append_nibble(std::vector<bool> &bits, unsigned int code)
{
    for (int bit = 0; bit < 4; ++bit)
        bits.emplace_back((code & (1u << bit)) != 0u);
}

static void region_append_leaf(TriangleSelector::TriangleSplittingData &data, unsigned int state)
{
    state = std::min<unsigned int>(state, unsigned(EnforcerBlockerType::ExtruderMax));
    if (state < data.used_states.size())
        data.used_states[state] = true;

    if (state >= 3u) {
        region_append_nibble(data.bitstream, 0b1100u);
        state -= 3u;
        while (state >= 15u) {
            region_append_nibble(data.bitstream, 0b1111u);
            state -= 15u;
        }
        region_append_nibble(data.bitstream, state);
    } else {
        region_append_nibble(data.bitstream, state << 2);
    }
}

static std::array<std::array<Vec3f, 3>, 4> region_split_triangle(const std::array<Vec3f, 3> &vertices)
{
    const Vec3f ab = 0.5f * (vertices[0] + vertices[1]);
    const Vec3f bc = 0.5f * (vertices[1] + vertices[2]);
    const Vec3f ca = 0.5f * (vertices[2] + vertices[0]);
    return { std::array<Vec3f, 3>{ vertices[0], ab, ca },
             std::array<Vec3f, 3>{ ab, vertices[1], bc },
             std::array<Vec3f, 3>{ bc, vertices[2], ca },
             std::array<Vec3f, 3>{ ab, bc, ca } };
}

using RegionPaintingSampler = std::function<unsigned int(size_t, const Vec3f &, const Vec3f &)>;

static bool region_append_sampled_triangle(TriangleSelector::TriangleSplittingData &data,
                                           const RegionPaintingSampler             &sampler,
                                           size_t                                   target_triangle,
                                           const std::array<Vec3f, 3>              &vertices,
                                           const std::array<Vec3f, 3>              &barycentrics,
                                           int                                      depth,
                                           int                                      min_depth,
                                           int                                      max_depth)
{
    const Vec3f centroid = (vertices[0] + vertices[1] + vertices[2]) / 3.f;
    const Vec3f centroid_bary = (barycentrics[0] + barycentrics[1] + barycentrics[2]) / 3.f;
    const unsigned int s0 = sampler(target_triangle, vertices[0], barycentrics[0]);
    const unsigned int s1 = sampler(target_triangle, vertices[1], barycentrics[1]);
    const unsigned int s2 = sampler(target_triangle, vertices[2], barycentrics[2]);
    const unsigned int sc = sampler(target_triangle, centroid, centroid_bary);
    const bool states_differ = s0 != s1 || s1 != s2 || s2 != sc;

    if (depth < max_depth && (depth < min_depth || states_differ)) {
        region_append_nibble(data.bitstream, 3u);
        const std::array<std::array<Vec3f, 3>, 4> child_vertices = region_split_triangle(vertices);
        const std::array<std::array<Vec3f, 3>, 4> child_barycentrics = region_split_triangle(barycentrics);
        bool has_non_base = false;
        for (int child_idx = 3; child_idx >= 0; --child_idx) {
            has_non_base |= region_append_sampled_triangle(data,
                                                           sampler,
                                                           target_triangle,
                                                           child_vertices[size_t(child_idx)],
                                                           child_barycentrics[size_t(child_idx)],
                                                           depth + 1,
                                                           min_depth,
                                                           max_depth);
        }
        return has_non_base;
    }

    region_append_leaf(data, sc);
    return sc != 0;
}

static SimplifyTextureDataResult remap_rgba_from_snapshot(const SimplifyTextureDataSnapshot &snapshot,
                                                          const indexed_triangle_set       &simplified_mesh,
                                                          const SimplifyTextureCancelFn    &throw_on_cancel,
                                                          const SimplifyTextureProgressFn  &status_fn)
{
    SimplifyTextureDataResult result;
    result.source = SimplifyColorSource::RgbaData;
    if (simplified_mesh.indices.empty() || snapshot.source_mesh.indices.empty())
        return result;

    set_progress(status_fn, 0);
    AABBMesh source_aabb(snapshot.source_mesh);
    std::vector<FastTriangleProjection> projection_cache(simplified_mesh.indices.size());
    std::vector<uint8_t> projection_cache_valid(simplified_mesh.indices.size(), 0);
    RgbaFacetLookup rgba_lookup;
    if (snapshot.source == SimplifyColorSource::RgbaData)
        rgba_lookup = make_rgba_facet_lookup(snapshot);

    auto annotation = ColorFacetsAnnotation::make_temporary();
    annotation->set_metadata_json(snapshot.source == SimplifyColorSource::RgbaData ?
                                      snapshot.rgba_metadata_json :
                                      rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));

    const int max_depth = snapshot.source == SimplifyColorSource::ImageTexture ? 5 : 4;
    const float target_span = std::max(mesh_max_axis_span(simplified_mesh) / 120.f, 0.3f);
    TextureMappingColorSubdivisionDepths subdivision_depths =
        [target_span, max_depth](size_t, const std::array<Vec3f, 3> &vertices) {
            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_span, max_depth);
            return std::make_pair(depth, max_depth);
        };

    size_t sample_counter = 0;
    TextureMappingColorSampler sampler = [&](size_t target_triangle, const Vec3f &point, const Vec3f &target_barycentric) {
        if ((++sample_counter & 1023u) == 0)
            check_cancel(throw_on_cancel);

        size_t source_triangle = 0;
        Vec3f source_barycentric = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
        bool source_projected = false;
        if (target_triangle < projection_cache.size()) {
            if (projection_cache_valid[target_triangle] == 0) {
                std::array<Vec3f, 3> target_vertices;
                if (valid_triangle_vertices(simplified_mesh, target_triangle, target_vertices))
                    projection_cache[target_triangle] = make_fast_triangle_projection(snapshot, source_aabb, target_vertices);
                projection_cache_valid[target_triangle] = 1;
            }
            source_projected = projected_source_from_fast_triangle(projection_cache[target_triangle],
                                                                   target_barycentric,
                                                                   source_triangle,
                                                                   source_barycentric);
        }
        if (!source_projected && !project_to_source(snapshot, source_aabb, point, source_triangle, source_barycentric))
            return pack_rgba(ColorRGBA(1.f, 1.f, 1.f, 1.f));

        std::array<Vec3f, 3> source_vertices;
        if (!valid_triangle_vertices(snapshot.source_mesh, source_triangle, source_vertices))
            return pack_rgba(ColorRGBA(1.f, 1.f, 1.f, 1.f));
        const Vec3f source_point = source_vertices[0] * source_barycentric.x() +
                                   source_vertices[1] * source_barycentric.y() +
                                   source_vertices[2] * source_barycentric.z();
        return pack_rgba(sample_snapshot_color_at_source(snapshot,
                                                         snapshot.source == SimplifyColorSource::RgbaData ? &rgba_lookup : nullptr,
                                                         source_triangle,
                                                         source_point,
                                                         source_barycentric));
    };

    const float split_threshold = snapshot.source == SimplifyColorSource::ImageTexture ? 0.025f : 0.04f;
    TextureMappingColorProgressFn progress_fn = [&status_fn](size_t completed, size_t total) {
        const int percent = total == 0 ? 100 : int((uint64_t(completed) * 100u) / uint64_t(total));
        set_progress(status_fn, percent);
    };
    annotation->set_from_triangle_sampler(simplified_mesh, sampler, max_depth, split_threshold, subdivision_depths, nullptr, {}, progress_fn);
    check_cancel(throw_on_cancel);
    set_progress(status_fn, 100);

    if (!annotation->empty()) {
        result.rgba_data = std::move(annotation);
    }
    return result;
}

static void remap_region_painting_from_snapshot(const SimplifyTextureDataSnapshot &snapshot,
                                                const indexed_triangle_set       &simplified_mesh,
                                                const SimplifyTextureCancelFn    &throw_on_cancel,
                                                const SimplifyTextureProgressFn  &status_fn,
                                                SimplifyTextureDataResult        &result)
{
    result.region_painting_touched = snapshot.region_painting_present;
    if (!snapshot.region_painting_present) {
        set_progress(status_fn, 100);
        return;
    }

    if (!snapshot.region_painting_transfer_needed) {
        set_progress(status_fn, 100);
        return;
    }

    if (simplified_mesh.indices.empty() || snapshot.source_mesh.indices.empty() || snapshot.region_facets_per_type.empty()) {
        result.region_painting_remap_failed = true;
        set_progress(status_fn, 100);
        return;
    }

    set_progress(status_fn, 0);
    AABBMesh source_aabb(snapshot.source_mesh);
    RegionFacetLookup region_lookup = make_region_facet_lookup(snapshot);
    std::vector<FastTriangleProjection> projection_cache(simplified_mesh.indices.size());
    std::vector<uint8_t> projection_cache_valid(simplified_mesh.indices.size(), 0);

    const int max_depth = 4;
    const float target_span = std::max(mesh_max_axis_span(simplified_mesh) / 120.f, 0.3f);
    const std::array<Vec3f, 3> root_barycentrics = {
        Vec3f(1.f, 0.f, 0.f),
        Vec3f(0.f, 1.f, 0.f),
        Vec3f(0.f, 0.f, 1.f)
    };

    RegionPaintingSampler sampler = [&](size_t target_triangle, const Vec3f &point, const Vec3f &target_barycentric) {
        size_t source_triangle = 0;
        Vec3f source_barycentric = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
        bool source_projected = false;
        if (target_triangle < projection_cache.size()) {
            if (projection_cache_valid[target_triangle] == 0) {
                std::array<Vec3f, 3> target_vertices;
                if (valid_triangle_vertices(simplified_mesh, target_triangle, target_vertices))
                    projection_cache[target_triangle] = make_fast_triangle_projection(snapshot, source_aabb, target_vertices);
                projection_cache_valid[target_triangle] = 1;
            }
            source_projected = projected_source_from_fast_triangle(projection_cache[target_triangle],
                                                                   target_barycentric,
                                                                   source_triangle,
                                                                   source_barycentric);
        }
        if (!source_projected && !project_to_source(snapshot, source_aabb, point, source_triangle, source_barycentric))
            return 0u;

        std::array<Vec3f, 3> source_vertices;
        if (!valid_triangle_vertices(snapshot.source_mesh, source_triangle, source_vertices))
            return 0u;
        const Vec3f source_point = source_vertices[0] * source_barycentric.x() +
                                   source_vertices[1] * source_barycentric.y() +
                                   source_vertices[2] * source_barycentric.z();
        return sample_region_state_at_source(snapshot, region_lookup, source_triangle, source_point);
    };

    TriangleSelector::TriangleSplittingData data;
    data.triangles_to_split.reserve(simplified_mesh.indices.size());
    size_t sample_counter = 0;
    for (size_t tri_idx = 0; tri_idx < simplified_mesh.indices.size(); ++tri_idx) {
        if ((++sample_counter & 255u) == 0) {
            check_cancel(throw_on_cancel);
            set_progress(status_fn, int((uint64_t(tri_idx) * 100u) / std::max<size_t>(simplified_mesh.indices.size(), 1)));
        }

        std::array<Vec3f, 3> vertices;
        if (!valid_triangle_vertices(simplified_mesh, tri_idx, vertices))
            continue;

        const size_t bitstream_start = data.bitstream.size();
        const int min_depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_span, max_depth);
        const bool has_non_base = region_append_sampled_triangle(data,
                                                                 sampler,
                                                                 tri_idx,
                                                                 vertices,
                                                                 root_barycentrics,
                                                                 0,
                                                                 min_depth,
                                                                 max_depth);
        if (has_non_base)
            data.triangles_to_split.emplace_back(int(tri_idx), int(bitstream_start));
        else
            data.bitstream.resize(bitstream_start);
    }

    check_cancel(throw_on_cancel);
    set_progress(status_fn, 100);
    data.triangles_to_split.shrink_to_fit();
    data.bitstream.shrink_to_fit();
    if (!data.triangles_to_split.empty()) {
        result.region_painting_data = std::move(data);
        result.region_painting_valid = true;
    }
}

static bool refresh_result_preview_from_raw(SimplifyTextureDataResult &result)
{
    ImageMapRawFilamentOffsetAtlas atlas;
    atlas.width = result.texture_width;
    atlas.height = result.texture_height;
    atlas.channels = result.texture_raw_channels;
    atlas.offsets.swap(result.texture_raw_filament_offsets);
    atlas.metadata_json = result.texture_raw_metadata_json;
    atlas.filaments = image_map_raw_filaments_from_metadata_json(atlas.metadata_json, atlas.channels);
    result.texture_rgba = image_map_raw_filament_offset_preview_rgba(atlas);
    result.texture_raw_filament_offsets.swap(atlas.offsets);
    return result.texture_rgba.size() >= size_t(result.texture_width) * size_t(result.texture_height) * 4;
}

static SimplifyTextureDataResult remap_image_texture_from_snapshot(const SimplifyTextureDataSnapshot &snapshot,
                                                                   const indexed_triangle_set       &simplified_mesh,
                                                                   const SimplifyTextureCancelFn    &throw_on_cancel,
                                                                   const SimplifyTextureProgressFn  &status_fn)
{
    SimplifyTextureDataResult result;
    std::unique_ptr<SimplifyTextureDataSnapshot> limited_snapshot;
    const SimplifyTextureDataSnapshot *sampling_snapshot = &snapshot;
    if (snapshot.texture_width > MaxSourceImageTextureSize || snapshot.texture_height > MaxSourceImageTextureSize) {
        limited_snapshot = std::make_unique<SimplifyTextureDataSnapshot>(snapshot);
        limit_snapshot_texture_resolution(*limited_snapshot);
        sampling_snapshot = limited_snapshot.get();
    }

    if ((!snapshot_has_valid_rgba_texture(*sampling_snapshot) && !snapshot_has_valid_raw_atlas(*sampling_snapshot)) ||
        simplified_mesh.indices.empty() ||
        sampling_snapshot->source_mesh.indices.empty())
        return result;

    set_progress(status_fn, 0);
    GeneratedImageTextureAtlas atlas;
    if (!initialize_generated_image_texture(simplified_mesh, result, atlas, ColorRGBA(1.f, 1.f, 1.f, 1.f)))
        return result;
    set_progress(status_fn, 8);
    check_cancel(throw_on_cancel);

    const bool remap_raw = snapshot_has_valid_raw_atlas(*sampling_snapshot);
    if (remap_raw) {
        result.texture_raw_channels = sampling_snapshot->texture_raw_channels;
        result.texture_raw_metadata_json = sampling_snapshot->texture_raw_metadata_json;
        result.texture_raw_filament_offsets.assign(size_t(result.texture_width) * size_t(result.texture_height) * size_t(result.texture_raw_channels), 0);
    }

    AABBMesh source_aabb(sampling_snapshot->source_mesh);
    const size_t island_count = atlas.islands.size();
    auto island_pixel_count = [&result](const GeneratedImageTextureIsland &island) -> uint64_t {
        const int x_begin = std::max(0, island.x);
        const int y_begin = std::max(0, island.y);
        const int x_end = std::min<int>(island.x + island.rect_width, int(result.texture_width));
        const int y_end = std::min<int>(island.y + island.rect_height, int(result.texture_height));
        if (x_end <= x_begin || y_end <= y_begin)
            return 0;
        return uint64_t(x_end - x_begin) * uint64_t(y_end - y_begin);
    };
    uint64_t total_pixels = 0;
    for (const GeneratedImageTextureIsland &island : atlas.islands)
        total_pixels += island_pixel_count(island);
    uint64_t processed_pixels = 0;
    int last_raster_progress = -1;
    auto report_raster_progress = [&status_fn, total_pixels, &processed_pixels, &last_raster_progress]() {
        const int progress = total_pixels == 0 ?
            96 :
            8 + int((processed_pixels * 88u) / total_pixels);
        if (progress != last_raster_progress) {
            last_raster_progress = progress;
            set_progress(status_fn, progress);
        }
    };
    report_raster_progress();

    size_t pixel_counter = 0;
    for (size_t island_idx = 0; island_idx < island_count; ++island_idx) {
        if ((island_idx & 31u) == 0)
            check_cancel(throw_on_cancel);

        const GeneratedImageTextureIsland &island = atlas.islands[island_idx];
        const uint64_t island_pixels = island_pixel_count(island);
        std::array<Vec3f, 3> target_vertices;
        if (!valid_triangle_vertices(simplified_mesh, island.tri_idx, target_vertices)) {
            processed_pixels += island_pixels;
            report_raster_progress();
            continue;
        }
        const FastTriangleProjection fast_projection = make_fast_triangle_projection(*sampling_snapshot, source_aabb, target_vertices);

        const size_t uv_offset = island.tri_idx * 6;
        if (uv_offset + 5 >= result.texture_uvs_per_face.size()) {
            processed_pixels += island_pixels;
            report_raster_progress();
            continue;
        }
        const float texture_size = float(result.texture_width);
        const std::array<Vec2f, 3> target_uv_pixels = {
            Vec2f(result.texture_uvs_per_face[uv_offset + 0] * texture_size, result.texture_uvs_per_face[uv_offset + 1] * texture_size),
            Vec2f(result.texture_uvs_per_face[uv_offset + 2] * texture_size, result.texture_uvs_per_face[uv_offset + 3] * texture_size),
            Vec2f(result.texture_uvs_per_face[uv_offset + 4] * texture_size, result.texture_uvs_per_face[uv_offset + 5] * texture_size)
        };

        const int x_end = std::min<int>(island.x + island.rect_width, int(result.texture_width));
        const int y_end = std::min<int>(island.y + island.rect_height, int(result.texture_height));
        for (int y = std::max(0, island.y); y < y_end; ++y) {
            for (int x = std::max(0, island.x); x < x_end; ++x) {
                ++processed_pixels;
                report_raster_progress();
                if ((++pixel_counter & 65535u) == 0)
                    check_cancel(throw_on_cancel);
                const Vec2f pixel(float(x) + 0.5f, float(y) + 0.5f);
                const Vec3f raw_target_bary = barycentric_weights_2d(pixel, target_uv_pixels);
                const Vec3f target_bary = normalized_nonnegative_barycentric(raw_target_bary);
                if (std::min({ raw_target_bary.x(), raw_target_bary.y(), raw_target_bary.z() }) < -1e-4f) {
                    const Vec2f closest = target_uv_pixels[0] * target_bary.x() +
                                          target_uv_pixels[1] * target_bary.y() +
                                          target_uv_pixels[2] * target_bary.z();
                    const float bleed_px = float(atlas.padding_px) + 1.5f;
                    if ((closest - pixel).squaredNorm() > bleed_px * bleed_px)
                        continue;
                }
                const Vec3f target_point = target_vertices[0] * target_bary.x() +
                                           target_vertices[1] * target_bary.y() +
                                           target_vertices[2] * target_bary.z();

                size_t source_triangle = 0;
                Vec3f source_barycentric = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
                if (!projected_source_from_fast_triangle(fast_projection, target_bary, source_triangle, source_barycentric) &&
                    !project_to_source(*sampling_snapshot, source_aabb, target_point, source_triangle, source_barycentric))
                    continue;

                if (remap_raw) {
                    write_raw_offset_pixel(result.texture_raw_filament_offsets,
                                           result.texture_width,
                                           result.texture_raw_channels,
                                           uint32_t(x),
                                           uint32_t(y),
                                           sample_image_raw_at_source(*sampling_snapshot, source_triangle, source_barycentric));
                } else {
                    write_rgba_pixel(result.texture_rgba,
                                     result.texture_width,
                                     uint32_t(x),
                                     uint32_t(y),
                                     sample_image_rgba_at_source(*sampling_snapshot, source_triangle, source_barycentric));
                }
            }
        }
    }

    if (remap_raw && !refresh_result_preview_from_raw(result))
        return SimplifyTextureDataResult();

    set_progress(status_fn, 100);
    result.source = SimplifyColorSource::ImageTexture;
    return result;
}

static SimplifyTextureDataResult remap_vertex_colors_from_snapshot(const SimplifyTextureDataSnapshot &snapshot,
                                                                   const indexed_triangle_set       &simplified_mesh,
                                                                   const SimplifyTextureCancelFn    &throw_on_cancel,
                                                                   const SimplifyTextureProgressFn  &status_fn)
{
    SimplifyTextureDataResult result;
    result.source = SimplifyColorSource::VertexColors;
    if (snapshot.vertex_colors_rgba.size() != snapshot.source_mesh.vertices.size() || simplified_mesh.vertices.empty())
        return result;

    set_progress(status_fn, 0);
    AABBMesh source_aabb(snapshot.source_mesh);
    result.vertex_colors_rgba.assign(simplified_mesh.vertices.size(), 0xFFFFFFFFu);
    for (size_t vertex_idx = 0; vertex_idx < simplified_mesh.vertices.size(); ++vertex_idx) {
        if ((vertex_idx & 1023u) == 0) {
            check_cancel(throw_on_cancel);
            set_progress(status_fn, int((uint64_t(vertex_idx) * 100u) / std::max<size_t>(simplified_mesh.vertices.size(), 1)));
        }

        size_t source_triangle = 0;
        Vec3f source_barycentric = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
        if (project_to_source(snapshot, source_aabb, simplified_mesh.vertices[vertex_idx].cast<float>(), source_triangle, source_barycentric))
            result.vertex_colors_rgba[vertex_idx] = pack_rgba(sample_vertex_color_at_source(snapshot, source_triangle, source_barycentric));
    }
    set_progress(status_fn, 100);
    return result;
}

struct PreparedMultiSourceTextureDataSource
{
    MultiSourceTextureDataSource source;
    RgbaFacetLookup rgba_lookup;
    RegionFacetLookup region_lookup;
};

static bool snapshot_has_color_data(const SimplifyTextureDataSnapshot &snapshot)
{
    return snapshot.source != SimplifyColorSource::None;
}

static ColorRGBA first_multi_source_fallback_color(const std::vector<PreparedMultiSourceTextureDataSource> &sources)
{
    return sources.empty() ? ColorRGBA(1.f, 1.f, 1.f, 1.f) : sources.front().source.fallback_color;
}

static bool multi_source_provenance_source(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                           const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                           size_t                                                   target_triangle,
                                           size_t                                                  &source_index,
                                           const MultiSourceTextureTriangleProvenance              *&entry)
{
    if (target_triangle >= provenance.size())
        return false;
    entry = &provenance[target_triangle];
    if (!entry->valid || entry->source_index >= sources.size())
        return false;
    source_index = entry->source_index;
    return true;
}

static Vec3f multi_source_source_barycentric(const MultiSourceTextureTriangleProvenance &entry, const Vec3f &target_barycentric)
{
    return normalized_nonnegative_barycentric(entry.source_barycentric[0] * target_barycentric.x() +
                                              entry.source_barycentric[1] * target_barycentric.y() +
                                              entry.source_barycentric[2] * target_barycentric.z());
}

static bool multi_source_source_point(const SimplifyTextureDataSnapshot &snapshot,
                                      size_t                             source_triangle,
                                      const Vec3f                       &source_barycentric,
                                      Vec3f                             &source_point)
{
    std::array<Vec3f, 3> source_vertices;
    if (!valid_triangle_vertices(snapshot.source_mesh, source_triangle, source_vertices))
        return false;
    source_point = source_vertices[0] * source_barycentric.x() +
                   source_vertices[1] * source_barycentric.y() +
                   source_vertices[2] * source_barycentric.z();
    return true;
}

static bool ensure_snapshot_rgba_preview_from_raw(SimplifyTextureDataSnapshot &snapshot)
{
    if (snapshot_has_valid_rgba_texture(snapshot))
        return true;
    if (!snapshot_has_valid_raw_atlas(snapshot))
        return false;

    ImageMapRawFilamentOffsetAtlas atlas;
    atlas.width = snapshot.texture_width;
    atlas.height = snapshot.texture_height;
    atlas.channels = snapshot.texture_raw_channels;
    atlas.offsets = snapshot.texture_raw_filament_offsets;
    atlas.metadata_json = snapshot.texture_raw_metadata_json;
    atlas.filaments = image_map_raw_filaments_from_metadata_json(atlas.metadata_json, atlas.channels);
    snapshot.texture_rgba = image_map_raw_filament_offset_preview_rgba(atlas);
    return snapshot_has_valid_rgba_texture(snapshot);
}

static std::vector<PreparedMultiSourceTextureDataSource> prepare_multi_source_texture_data_sources(const std::vector<MultiSourceTextureDataSource> &sources)
{
    std::vector<PreparedMultiSourceTextureDataSource> prepared;
    prepared.reserve(sources.size());
    for (const MultiSourceTextureDataSource &source : sources) {
        PreparedMultiSourceTextureDataSource item;
        item.source = source;
        if (item.source.snapshot.source == SimplifyColorSource::ImageTexture)
            limit_snapshot_texture_resolution(item.source.snapshot);
        if (item.source.snapshot.source == SimplifyColorSource::RgbaData)
            item.rgba_lookup = make_rgba_facet_lookup(item.source.snapshot);
        if (item.source.snapshot.region_painting_transfer_needed)
            item.region_lookup = make_region_facet_lookup(item.source.snapshot);
        prepared.emplace_back(std::move(item));
    }
    return prepared;
}

static ColorRGBA sample_multi_source_color_at_target(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                                     const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                     size_t                                                   target_triangle,
                                                     const Vec3f                                             &target_barycentric)
{
    size_t source_index = 0;
    const MultiSourceTextureTriangleProvenance *entry = nullptr;
    if (!multi_source_provenance_source(sources, provenance, target_triangle, source_index, entry))
        return first_multi_source_fallback_color(sources);

    const PreparedMultiSourceTextureDataSource &source = sources[source_index];
    const SimplifyTextureDataSnapshot &snapshot = source.source.snapshot;
    if (snapshot.source == SimplifyColorSource::None)
        return source.source.fallback_color;

    const Vec3f source_barycentric = multi_source_source_barycentric(*entry, target_barycentric);
    switch (snapshot.source) {
    case SimplifyColorSource::RgbaData: {
        Vec3f source_point = Vec3f::Zero();
        if (!multi_source_source_point(snapshot, entry->source_triangle, source_barycentric, source_point))
            return source.source.fallback_color;
        return sample_rgba_facets_at_source(snapshot, source.rgba_lookup, entry->source_triangle, source_point);
    }
    case SimplifyColorSource::ImageTexture:
        return snapshot_has_valid_rgba_texture(snapshot) ?
            sample_image_rgba_at_source(snapshot, entry->source_triangle, source_barycentric) :
            source.source.fallback_color;
    case SimplifyColorSource::VertexColors:
        return sample_vertex_color_at_source(snapshot, entry->source_triangle, source_barycentric);
    case SimplifyColorSource::None:
        break;
    }
    return source.source.fallback_color;
}

static std::vector<uint8_t> sample_multi_source_raw_at_target(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                                              const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                              size_t                                                   target_triangle,
                                                              const Vec3f                                             &target_barycentric,
                                                              uint32_t                                                 channels)
{
    size_t source_index = 0;
    const MultiSourceTextureTriangleProvenance *entry = nullptr;
    if (!multi_source_provenance_source(sources, provenance, target_triangle, source_index, entry))
        return std::vector<uint8_t>(size_t(channels), 0);

    const SimplifyTextureDataSnapshot &snapshot = sources[source_index].source.snapshot;
    if (snapshot.source != SimplifyColorSource::ImageTexture || !snapshot_has_valid_raw_atlas(snapshot))
        return std::vector<uint8_t>(size_t(channels), 0);
    return sample_image_raw_at_source(snapshot, entry->source_triangle, multi_source_source_barycentric(*entry, target_barycentric));
}

static unsigned int sample_multi_source_region_at_target(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                                         const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                         size_t                                                   target_triangle,
                                                         const Vec3f                                             &target_barycentric)
{
    size_t source_index = 0;
    const MultiSourceTextureTriangleProvenance *entry = nullptr;
    if (!multi_source_provenance_source(sources, provenance, target_triangle, source_index, entry))
        return 0;

    const PreparedMultiSourceTextureDataSource &source = sources[source_index];
    const SimplifyTextureDataSnapshot &snapshot = source.source.snapshot;
    if (!snapshot.region_painting_transfer_needed)
        return 0;

    const Vec3f source_barycentric = multi_source_source_barycentric(*entry, target_barycentric);
    Vec3f source_point = Vec3f::Zero();
    if (!multi_source_source_point(snapshot, entry->source_triangle, source_barycentric, source_point))
        return 0;
    return sample_region_state_at_source(snapshot, source.region_lookup, entry->source_triangle, source_point);
}

static bool multi_source_raw_compatible(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                        const std::vector<uint8_t>                              &participating,
                                        uint32_t                                                &channels,
                                        std::string                                             &metadata_json)
{
    channels = 0;
    metadata_json.clear();
    bool found = false;
    for (size_t source_idx = 0; source_idx < sources.size(); ++source_idx) {
        if (source_idx >= participating.size() || participating[source_idx] == 0)
            continue;
        const SimplifyTextureDataSnapshot &snapshot = sources[source_idx].source.snapshot;
        if (!snapshot_has_color_data(snapshot))
            continue;
        if (snapshot.source != SimplifyColorSource::ImageTexture || !snapshot_has_valid_raw_atlas(snapshot))
            return false;
        if (!found) {
            channels = snapshot.texture_raw_channels;
            metadata_json = snapshot.texture_raw_metadata_json;
            found = true;
        } else if (channels != snapshot.texture_raw_channels || metadata_json != snapshot.texture_raw_metadata_json)
            return false;
    }
    return found;
}

static SimplifyTextureDataResult remap_multi_source_rgba_from_provenance(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                                                         const indexed_triangle_set                             &target_mesh,
                                                                         const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                                         bool                                                    any_image_source,
                                                                         const SimplifyTextureCancelFn                          &throw_on_cancel,
                                                                         const SimplifyTextureProgressFn                        &status_fn)
{
    SimplifyTextureDataResult result;
    result.source = SimplifyColorSource::RgbaData;
    if (target_mesh.indices.empty() || provenance.size() != target_mesh.indices.size())
        return result;

    set_progress(status_fn, 0);
    auto annotation = ColorFacetsAnnotation::make_temporary();
    annotation->set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));

    const int max_depth = any_image_source ? 5 : 4;
    const float target_span = std::max(mesh_max_axis_span(target_mesh) / 120.f, 0.3f);
    TextureMappingColorSubdivisionDepths subdivision_depths =
        [target_span, max_depth](size_t, const std::array<Vec3f, 3> &vertices) {
            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_span, max_depth);
            return std::make_pair(depth, max_depth);
        };

    size_t sample_counter = 0;
    TextureMappingColorSampler sampler = [&](size_t target_triangle, const Vec3f &, const Vec3f &target_barycentric) {
        if ((++sample_counter & 1023u) == 0)
            check_cancel(throw_on_cancel);
        return pack_rgba(sample_multi_source_color_at_target(sources, provenance, target_triangle, target_barycentric));
    };

    const float split_threshold = any_image_source ? 0.025f : 0.04f;
    TextureMappingColorProgressFn progress_fn = [&status_fn](size_t completed, size_t total) {
        const int percent = total == 0 ? 100 : int((uint64_t(completed) * 100u) / uint64_t(total));
        set_progress(status_fn, percent);
    };
    annotation->set_from_triangle_sampler(target_mesh, sampler, max_depth, split_threshold, subdivision_depths, nullptr, {}, progress_fn);
    check_cancel(throw_on_cancel);
    set_progress(status_fn, 100);
    if (!annotation->empty())
        result.rgba_data = std::move(annotation);
    return result;
}

static SimplifyTextureDataResult remap_multi_source_image_from_provenance(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                                                          const indexed_triangle_set                             &target_mesh,
                                                                          const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                                          bool                                                    remap_raw,
                                                                          uint32_t                                                raw_channels,
                                                                          const std::string                                      &raw_metadata_json,
                                                                          const SimplifyTextureCancelFn                          &throw_on_cancel,
                                                                          const SimplifyTextureProgressFn                        &status_fn)
{
    SimplifyTextureDataResult result;
    if (target_mesh.indices.empty() || provenance.size() != target_mesh.indices.size())
        return result;

    set_progress(status_fn, 0);
    GeneratedImageTextureAtlas atlas;
    if (!initialize_generated_image_texture(target_mesh, result, atlas, first_multi_source_fallback_color(sources)))
        return result;
    set_progress(status_fn, 8);
    check_cancel(throw_on_cancel);

    if (remap_raw) {
        result.texture_raw_channels = raw_channels;
        result.texture_raw_metadata_json = raw_metadata_json;
        result.texture_raw_filament_offsets.assign(size_t(result.texture_width) * size_t(result.texture_height) * size_t(result.texture_raw_channels), 0);
    }

    auto island_pixel_count = [&result](const GeneratedImageTextureIsland &island) -> uint64_t {
        const int x_begin = std::max(0, island.x);
        const int y_begin = std::max(0, island.y);
        const int x_end = std::min<int>(island.x + island.rect_width, int(result.texture_width));
        const int y_end = std::min<int>(island.y + island.rect_height, int(result.texture_height));
        if (x_end <= x_begin || y_end <= y_begin)
            return 0;
        return uint64_t(x_end - x_begin) * uint64_t(y_end - y_begin);
    };

    uint64_t total_pixels = 0;
    for (const GeneratedImageTextureIsland &island : atlas.islands)
        total_pixels += island_pixel_count(island);

    uint64_t processed_pixels = 0;
    int last_raster_progress = -1;
    auto report_raster_progress = [&status_fn, total_pixels, &processed_pixels, &last_raster_progress]() {
        const int progress = total_pixels == 0 ? 96 : 8 + int((processed_pixels * 88u) / total_pixels);
        if (progress != last_raster_progress) {
            last_raster_progress = progress;
            set_progress(status_fn, progress);
        }
    };
    report_raster_progress();

    size_t pixel_counter = 0;
    for (size_t island_idx = 0; island_idx < atlas.islands.size(); ++island_idx) {
        if ((island_idx & 31u) == 0)
            check_cancel(throw_on_cancel);

        const GeneratedImageTextureIsland &island = atlas.islands[island_idx];
        const uint64_t island_pixels = island_pixel_count(island);
        const size_t uv_offset = island.tri_idx * 6;
        if (uv_offset + 5 >= result.texture_uvs_per_face.size()) {
            processed_pixels += island_pixels;
            report_raster_progress();
            continue;
        }

        const float texture_size = float(result.texture_width);
        const std::array<Vec2f, 3> target_uv_pixels = {
            Vec2f(result.texture_uvs_per_face[uv_offset + 0] * texture_size, result.texture_uvs_per_face[uv_offset + 1] * texture_size),
            Vec2f(result.texture_uvs_per_face[uv_offset + 2] * texture_size, result.texture_uvs_per_face[uv_offset + 3] * texture_size),
            Vec2f(result.texture_uvs_per_face[uv_offset + 4] * texture_size, result.texture_uvs_per_face[uv_offset + 5] * texture_size)
        };

        const int x_end = std::min<int>(island.x + island.rect_width, int(result.texture_width));
        const int y_end = std::min<int>(island.y + island.rect_height, int(result.texture_height));
        for (int y = std::max(0, island.y); y < y_end; ++y) {
            for (int x = std::max(0, island.x); x < x_end; ++x) {
                ++processed_pixels;
                report_raster_progress();
                if ((++pixel_counter & 65535u) == 0)
                    check_cancel(throw_on_cancel);

                const Vec2f pixel(float(x) + 0.5f, float(y) + 0.5f);
                const Vec3f raw_target_bary = barycentric_weights_2d(pixel, target_uv_pixels);
                const Vec3f target_bary = normalized_nonnegative_barycentric(raw_target_bary);
                if (std::min({ raw_target_bary.x(), raw_target_bary.y(), raw_target_bary.z() }) < -1e-4f) {
                    const Vec2f closest = target_uv_pixels[0] * target_bary.x() +
                                          target_uv_pixels[1] * target_bary.y() +
                                          target_uv_pixels[2] * target_bary.z();
                    const float bleed_px = float(atlas.padding_px) + 1.5f;
                    if ((closest - pixel).squaredNorm() > bleed_px * bleed_px)
                        continue;
                }

                if (remap_raw) {
                    write_raw_offset_pixel(result.texture_raw_filament_offsets,
                                           result.texture_width,
                                           result.texture_raw_channels,
                                           uint32_t(x),
                                           uint32_t(y),
                                           sample_multi_source_raw_at_target(sources, provenance, island.tri_idx, target_bary, raw_channels));
                } else {
                    write_rgba_pixel(result.texture_rgba,
                                     result.texture_width,
                                     uint32_t(x),
                                     uint32_t(y),
                                     sample_multi_source_color_at_target(sources, provenance, island.tri_idx, target_bary));
                }
            }
        }
    }

    if (remap_raw && !refresh_result_preview_from_raw(result))
        return SimplifyTextureDataResult();

    set_progress(status_fn, 100);
    result.source = SimplifyColorSource::ImageTexture;
    return result;
}

static void remap_multi_source_region_painting_from_provenance(const std::vector<PreparedMultiSourceTextureDataSource> &sources,
                                                               const indexed_triangle_set                             &target_mesh,
                                                               const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                               bool                                                    region_painting_present,
                                                               bool                                                    region_painting_transfer_needed,
                                                               const SimplifyTextureCancelFn                          &throw_on_cancel,
                                                               const SimplifyTextureProgressFn                        &status_fn,
                                                               SimplifyTextureDataResult                              &result)
{
    result.region_painting_touched = region_painting_present;
    if (!region_painting_present) {
        set_progress(status_fn, 100);
        return;
    }

    if (!region_painting_transfer_needed) {
        set_progress(status_fn, 100);
        return;
    }

    if (target_mesh.indices.empty() || provenance.size() != target_mesh.indices.size()) {
        result.region_painting_remap_failed = true;
        set_progress(status_fn, 100);
        return;
    }

    set_progress(status_fn, 0);
    const int max_depth = 4;
    const float target_span = std::max(mesh_max_axis_span(target_mesh) / 120.f, 0.3f);
    const std::array<Vec3f, 3> root_barycentrics = {
        Vec3f(1.f, 0.f, 0.f),
        Vec3f(0.f, 1.f, 0.f),
        Vec3f(0.f, 0.f, 1.f)
    };

    RegionPaintingSampler sampler = [&](size_t target_triangle, const Vec3f &, const Vec3f &target_barycentric) {
        return sample_multi_source_region_at_target(sources, provenance, target_triangle, target_barycentric);
    };

    TriangleSelector::TriangleSplittingData data;
    data.triangles_to_split.reserve(target_mesh.indices.size());
    size_t sample_counter = 0;
    for (size_t tri_idx = 0; tri_idx < target_mesh.indices.size(); ++tri_idx) {
        if ((++sample_counter & 255u) == 0) {
            check_cancel(throw_on_cancel);
            set_progress(status_fn, int((uint64_t(tri_idx) * 100u) / std::max<size_t>(target_mesh.indices.size(), 1)));
        }

        std::array<Vec3f, 3> vertices;
        if (!valid_triangle_vertices(target_mesh, tri_idx, vertices))
            continue;

        const size_t bitstream_start = data.bitstream.size();
        const int min_depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_span, max_depth);
        const bool has_non_base = region_append_sampled_triangle(data,
                                                                 sampler,
                                                                 tri_idx,
                                                                 vertices,
                                                                 root_barycentrics,
                                                                 0,
                                                                 min_depth,
                                                                 max_depth);
        if (has_non_base)
            data.triangles_to_split.emplace_back(int(tri_idx), int(bitstream_start));
        else
            data.bitstream.resize(bitstream_start);
    }

    check_cancel(throw_on_cancel);
    set_progress(status_fn, 100);
    data.triangles_to_split.shrink_to_fit();
    data.bitstream.shrink_to_fit();
    if (!data.triangles_to_split.empty()) {
        result.region_painting_data = std::move(data);
        result.region_painting_valid = true;
    }
}

static void clear_image_texture(ModelVolume &volume)
{
    volume.imported_texture_uvs_per_face.clear();
    volume.imported_texture_uv_valid.clear();
    volume.imported_texture_rgba.clear();
    volume.imported_texture_raw_filament_offsets.clear();
    volume.imported_texture_width = 0;
    volume.imported_texture_height = 0;
    volume.imported_texture_raw_channels = 0;
    volume.imported_texture_raw_metadata_json.clear();
    volume.uv_map_generator_version = 0;
}

static void touch_imported_texture(ModelVolume &volume)
{
    volume.imported_texture_uvs_per_face.set_new_unique_id();
    volume.imported_texture_uv_valid.set_new_unique_id();
    volume.imported_texture_rgba.set_new_unique_id();
    volume.imported_texture_raw_filament_offsets.set_new_unique_id();
}

} // namespace

bool model_volume_region_painting_needs_remap(const ModelVolume &volume)
{
    return region_painting_data_needs_remap(volume);
}

SimplifyTextureDataSnapshot snapshot_simplify_texture_data(const ModelVolume &volume)
{
    SimplifyTextureDataSnapshot snapshot;
    const indexed_triangle_set &its = volume.mesh().its;
    if (its.vertices.empty() || its.indices.empty())
        return snapshot;

    bool found_color_source = false;
    if (!volume.texture_mapping_color_facets.empty()) {
        snapshot.source = SimplifyColorSource::RgbaData;
        snapshot.source_mesh = its;
        snapshot.rgba_metadata_json = volume.texture_mapping_color_facets.metadata_json();
        volume.texture_mapping_color_facets.get_facet_triangles(volume, snapshot.rgba_facets);
        if (!snapshot.rgba_facets.empty())
            found_color_source = true;
        else {
            snapshot.source = SimplifyColorSource::None;
            snapshot.source_mesh = indexed_triangle_set();
            snapshot.rgba_metadata_json.clear();
        }
    }

    if (!found_color_source && volume_has_valid_image_texture(volume)) {
        snapshot.source = SimplifyColorSource::ImageTexture;
        snapshot.source_mesh = its;
        snapshot.texture_rgba.assign(volume.imported_texture_rgba.begin(), volume.imported_texture_rgba.end());
        snapshot.texture_uvs_per_face.assign(volume.imported_texture_uvs_per_face.begin(), volume.imported_texture_uvs_per_face.end());
        snapshot.texture_uv_valid.assign(volume.imported_texture_uv_valid.begin(), volume.imported_texture_uv_valid.end());
        snapshot.texture_raw_filament_offsets.assign(volume.imported_texture_raw_filament_offsets.begin(),
                                                     volume.imported_texture_raw_filament_offsets.end());
        snapshot.texture_width = volume.imported_texture_width;
        snapshot.texture_height = volume.imported_texture_height;
        snapshot.texture_raw_channels = volume.imported_texture_raw_channels;
        snapshot.texture_raw_metadata_json = volume.imported_texture_raw_metadata_json;
        snapshot.uv_map_generator_version = volume.uv_map_generator_version;
        if (snapshot_has_valid_rgba_texture(snapshot) || snapshot_has_valid_raw_atlas(snapshot))
            found_color_source = true;
        else {
            snapshot.source = SimplifyColorSource::None;
            snapshot.source_mesh = indexed_triangle_set();
            snapshot.texture_rgba.clear();
            snapshot.texture_uvs_per_face.clear();
            snapshot.texture_uv_valid.clear();
            snapshot.texture_raw_filament_offsets.clear();
            snapshot.texture_width = 0;
            snapshot.texture_height = 0;
            snapshot.texture_raw_channels = 0;
            snapshot.texture_raw_metadata_json.clear();
            snapshot.uv_map_generator_version = 0;
        }
    }

    if (!found_color_source &&
        !volume.imported_vertex_colors_rgba.empty() &&
        volume.imported_vertex_colors_rgba.size() == its.vertices.size()) {
        snapshot.source = SimplifyColorSource::VertexColors;
        snapshot.source_mesh = its;
        snapshot.vertex_colors_rgba.assign(volume.imported_vertex_colors_rgba.begin(), volume.imported_vertex_colors_rgba.end());
        found_color_source = true;
    }

    snapshot.region_painting_present = !volume.mmu_segmentation_facets.empty();
    snapshot.region_base_filament_id = int(volume_base_region_filament_id(volume));
    snapshot.region_painting_transfer_needed = region_painting_data_needs_remap(volume);
    if (snapshot.region_painting_transfer_needed) {
        if (!found_color_source)
            snapshot.source_mesh = its;
        volume.mmu_segmentation_facets.get_facet_triangles(volume, snapshot.region_facets_per_type);
    }
    return snapshot;
}

void transform_simplify_texture_data_snapshot(SimplifyTextureDataSnapshot &snapshot, const Transform3d &transform)
{
    if (snapshot.source == SimplifyColorSource::None && !snapshot.region_painting_transfer_needed)
        return;

    if (!snapshot.source_mesh.vertices.empty() && !snapshot.source_mesh.indices.empty()) {
        TriangleMesh mesh(std::move(snapshot.source_mesh));
        mesh.transform(transform, false);
        snapshot.source_mesh = std::move(mesh.its);
    }

    if (!snapshot.rgba_facets.empty()) {
        for (ColorFacetTriangle &facet : snapshot.rgba_facets)
            for (Vec3f &vertex : facet.vertices)
                vertex = (transform * vertex.cast<double>()).cast<float>();
    }

    for (auto &facets : snapshot.region_facets_per_type)
        for (TriangleSelector::FacetStateTriangle &facet : facets)
            for (Vec3f &vertex : facet.vertices)
                vertex = (transform * vertex.cast<double>()).cast<float>();
}

SimplifyTextureDataResult remap_simplify_texture_data(const SimplifyTextureDataSnapshot &snapshot,
                                                       const indexed_triangle_set       &simplified_mesh,
                                                       const SimplifyTextureCancelFn    &throw_on_cancel,
                                                       const SimplifyTextureProgressFn  &status_fn,
                                                       const SimplifyTextureDataRemapOptions &options)
{
    const bool has_color_data = snapshot.source != SimplifyColorSource::None;
    const bool remap_color = has_color_data && options.remap_color_data && !simplified_mesh.indices.empty();
    const bool remap_region = snapshot.region_painting_present &&
                              options.remap_region_painting &&
                              snapshot.region_painting_transfer_needed &&
                              !simplified_mesh.indices.empty();

    if (!remap_color && !remap_region) {
        set_progress(status_fn, 100);
        SimplifyTextureDataResult result;
        result.source = options.remap_color_data ? snapshot.source : SimplifyColorSource::None;
        result.region_painting_touched = snapshot.region_painting_present;
        return result;
    }

    auto scaled_progress = [&status_fn](int offset, int span) {
        return [status_fn, offset, span](int percent) {
            set_progress(status_fn, offset + int(std::round(float(std::clamp(percent, 0, 100)) * float(span) / 100.f)));
        };
    };

    const bool split_progress = remap_color && snapshot.region_painting_present;
    SimplifyTextureDataResult result;
    if (remap_color) {
        const SimplifyTextureProgressFn color_progress = split_progress ? scaled_progress(0, 80) : status_fn;
        switch (snapshot.source) {
        case SimplifyColorSource::RgbaData:
            result = remap_rgba_from_snapshot(snapshot, simplified_mesh, throw_on_cancel, color_progress);
            break;
        case SimplifyColorSource::ImageTexture: {
            result = remap_image_texture_from_snapshot(snapshot, simplified_mesh, throw_on_cancel, color_progress);
            if (result.source == SimplifyColorSource::ImageTexture)
                break;
            SimplifyTextureDataSnapshot fallback_snapshot = snapshot;
            limit_snapshot_texture_resolution(fallback_snapshot);
            if (!snapshot_has_valid_rgba_texture(fallback_snapshot) && snapshot_has_valid_raw_atlas(fallback_snapshot)) {
                ImageMapRawFilamentOffsetAtlas atlas;
                atlas.width = fallback_snapshot.texture_width;
                atlas.height = fallback_snapshot.texture_height;
                atlas.channels = fallback_snapshot.texture_raw_channels;
                atlas.offsets = fallback_snapshot.texture_raw_filament_offsets;
                atlas.metadata_json = fallback_snapshot.texture_raw_metadata_json;
                atlas.filaments = image_map_raw_filaments_from_metadata_json(atlas.metadata_json, atlas.channels);
                fallback_snapshot.texture_rgba = image_map_raw_filament_offset_preview_rgba(atlas);
            }
            result = remap_rgba_from_snapshot(fallback_snapshot, simplified_mesh, throw_on_cancel, color_progress);
            result.remap_failed = true;
            result.used_fallback_rgba = result.source == SimplifyColorSource::RgbaData && result.rgba_data != nullptr;
            break;
        }
        case SimplifyColorSource::VertexColors:
            result = remap_vertex_colors_from_snapshot(snapshot, simplified_mesh, throw_on_cancel, color_progress);
            break;
        case SimplifyColorSource::None:
            break;
        }
    } else {
        result.source = SimplifyColorSource::None;
    }

    if (snapshot.region_painting_present) {
        SimplifyTextureProgressFn region_progress = status_fn;
        if (split_progress)
            region_progress = scaled_progress(80, 20);
        if (options.remap_region_painting)
            remap_region_painting_from_snapshot(snapshot, simplified_mesh, throw_on_cancel, region_progress, result);
        else {
            result.region_painting_touched = true;
            set_progress(region_progress, 100);
        }
    }

    set_progress(status_fn, 100);
    return result;
}

SimplifyTextureDataResult remap_multi_source_texture_data(const std::vector<MultiSourceTextureDataSource> &sources,
                                                          const indexed_triangle_set                      &target_mesh,
                                                          const std::vector<MultiSourceTextureTriangleProvenance> &provenance,
                                                          const SimplifyTextureCancelFn                   &throw_on_cancel,
                                                          const SimplifyTextureProgressFn                 &status_fn,
                                                          const SimplifyTextureDataRemapOptions           &options)
{
    std::vector<PreparedMultiSourceTextureDataSource> prepared = prepare_multi_source_texture_data_sources(sources);
    std::vector<uint8_t> participating(prepared.size(), 0);
    for (const MultiSourceTextureTriangleProvenance &entry : provenance)
        if (entry.valid && entry.source_index < participating.size())
            participating[entry.source_index] = 1;

    bool has_color_data = false;
    bool any_image_source = false;
    bool region_painting_present = false;
    bool region_painting_transfer_needed = false;
    for (size_t source_idx = 0; source_idx < prepared.size(); ++source_idx) {
        if (source_idx >= participating.size() || participating[source_idx] == 0)
            continue;
        const SimplifyTextureDataSnapshot &snapshot = prepared[source_idx].source.snapshot;
        has_color_data = true;
        any_image_source |= snapshot.source == SimplifyColorSource::ImageTexture;
        region_painting_present |= snapshot.region_painting_present;
        region_painting_transfer_needed |= snapshot.region_painting_transfer_needed;
    }

    const bool remap_color = has_color_data && options.remap_color_data && !target_mesh.indices.empty();
    const bool remap_region = region_painting_present &&
                              options.remap_region_painting &&
                              region_painting_transfer_needed &&
                              !target_mesh.indices.empty();
    if (!remap_color && !remap_region) {
        set_progress(status_fn, 100);
        SimplifyTextureDataResult result;
        result.source = SimplifyColorSource::None;
        result.region_painting_touched = region_painting_present;
        return result;
    }

    if (provenance.size() != target_mesh.indices.size()) {
        SimplifyTextureDataResult result;
        result.source = SimplifyColorSource::None;
        result.remap_failed = remap_color;
        result.region_painting_touched = region_painting_present;
        result.region_painting_remap_failed = remap_region;
        set_progress(status_fn, 100);
        return result;
    }

    uint32_t raw_channels = 0;
    std::string raw_metadata_json;
    const bool remap_raw = any_image_source && multi_source_raw_compatible(prepared, participating, raw_channels, raw_metadata_json);
    if (any_image_source && !remap_raw) {
        for (PreparedMultiSourceTextureDataSource &source : prepared)
            if (source.source.snapshot.source == SimplifyColorSource::ImageTexture)
                ensure_snapshot_rgba_preview_from_raw(source.source.snapshot);
    }

    auto scaled_progress = [&status_fn](int offset, int span) {
        return [status_fn, offset, span](int percent) {
            set_progress(status_fn, offset + int(std::round(float(std::clamp(percent, 0, 100)) * float(span) / 100.f)));
        };
    };

    const bool split_progress = remap_color && region_painting_present;
    SimplifyTextureDataResult result;
    if (remap_color) {
        const SimplifyTextureProgressFn color_progress = split_progress ? scaled_progress(0, 80) : status_fn;
        if (any_image_source) {
            result = remap_multi_source_image_from_provenance(prepared,
                                                              target_mesh,
                                                              provenance,
                                                              remap_raw,
                                                              raw_channels,
                                                              raw_metadata_json,
                                                              throw_on_cancel,
                                                              color_progress);
            if (result.source != SimplifyColorSource::ImageTexture) {
                result = remap_multi_source_rgba_from_provenance(prepared, target_mesh, provenance, true, throw_on_cancel, color_progress);
                result.remap_failed = true;
                result.used_fallback_rgba = result.source == SimplifyColorSource::RgbaData && result.rgba_data != nullptr;
            }
        } else
            result = remap_multi_source_rgba_from_provenance(prepared, target_mesh, provenance, false, throw_on_cancel, color_progress);
    } else
        result.source = SimplifyColorSource::None;

    if (region_painting_present) {
        SimplifyTextureProgressFn region_progress = status_fn;
        if (split_progress)
            region_progress = scaled_progress(80, 20);
        if (options.remap_region_painting)
            remap_multi_source_region_painting_from_provenance(prepared,
                                                               target_mesh,
                                                               provenance,
                                                               region_painting_present,
                                                               region_painting_transfer_needed,
                                                               throw_on_cancel,
                                                               region_progress,
                                                               result);
        else {
            result.region_painting_touched = true;
            set_progress(region_progress, 100);
        }
    }

    set_progress(status_fn, 100);
    return result;
}

void apply_simplify_texture_data_result(ModelVolume &volume, SimplifyTextureDataResult &&result)
{
    switch (result.source) {
    case SimplifyColorSource::RgbaData:
        if (result.rgba_data)
            volume.texture_mapping_color_facets.assign(std::move(*result.rgba_data));
        else
            volume.texture_mapping_color_facets.reset();
        clear_image_texture(volume);
        volume.imported_vertex_colors_rgba.clear();
        volume.imported_vertex_colors_rgba.set_new_unique_id();
        touch_imported_texture(volume);
        break;
    case SimplifyColorSource::ImageTexture:
        volume.texture_mapping_color_facets.reset();
        volume.imported_vertex_colors_rgba.clear();
        volume.imported_vertex_colors_rgba.set_new_unique_id();
        volume.imported_texture_uvs_per_face = std::move(result.texture_uvs_per_face);
        volume.imported_texture_uv_valid = std::move(result.texture_uv_valid);
        volume.imported_texture_rgba = std::move(result.texture_rgba);
        volume.imported_texture_raw_filament_offsets = std::move(result.texture_raw_filament_offsets);
        volume.imported_texture_width = result.texture_width;
        volume.imported_texture_height = result.texture_height;
        volume.imported_texture_raw_channels = result.texture_raw_channels;
        volume.imported_texture_raw_metadata_json = std::move(result.texture_raw_metadata_json);
        volume.uv_map_generator_version = result.uv_map_generator_version;
        touch_imported_texture(volume);
        break;
    case SimplifyColorSource::VertexColors:
        volume.texture_mapping_color_facets.reset();
        clear_image_texture(volume);
        volume.imported_vertex_colors_rgba = std::move(result.vertex_colors_rgba);
        volume.imported_vertex_colors_rgba.set_new_unique_id();
        touch_imported_texture(volume);
        break;
    case SimplifyColorSource::None:
        volume.texture_mapping_color_facets.reset();
        clear_image_texture(volume);
        volume.imported_vertex_colors_rgba.clear();
        volume.imported_vertex_colors_rgba.set_new_unique_id();
        touch_imported_texture(volume);
        break;
    }

    if (result.region_painting_touched) {
        if (result.region_painting_valid && !result.region_painting_data.triangles_to_split.empty()) {
            TriangleSelector selector(volume.mesh());
            selector.deserialize(result.region_painting_data);
            volume.mmu_segmentation_facets.set(selector);
        } else {
            volume.mmu_segmentation_facets.reset();
        }
    }
}

} // namespace Slic3r
