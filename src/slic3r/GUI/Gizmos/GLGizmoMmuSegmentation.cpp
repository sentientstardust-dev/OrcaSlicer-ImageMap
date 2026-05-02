#include "GLGizmoMmuSegmentation.hpp"

#include "slic3r/GUI/GLCanvas3D.hpp"
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/ImGuiWrapper.hpp"
#include "slic3r/GUI/Camera.hpp"
#include "slic3r/GUI/Plater.hpp"
#include "slic3r/GUI/BitmapCache.hpp"
#include "slic3r/GUI/format.hpp"
#include "slic3r/GUI/GUI_ObjectList.hpp"
#include "slic3r/GUI/NotificationManager.hpp"
#include "slic3r/GUI/GUI.hpp"
#include "slic3r/GUI/ObjColorDialog.hpp"
#include "slic3r/GUI/MainFrame.hpp"
#include "slic3r/GUI/Tab.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/TextureMapping.hpp"
#include "slic3r/Utils/UndoRedo.hpp"
#include "GLGizmoUtils.hpp"


#include <glad/gl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <boost/log/trivial.hpp>
#include <wx/filedlg.h>
#include <wx/image.h>

namespace Slic3r::GUI {

static inline void show_notification_extruders_limit_exceeded()
{
    wxGetApp()
        .plater()
        ->get_notification_manager()
        ->push_notification(NotificationType::MmSegmentationExceededExtrudersLimit, NotificationManager::NotificationLevel::PrintInfoNotificationLevel,
                            GUI::format(_L("Filament count exceeds the maximum number that painting tool supports. Only the "
                                           "first %1% filaments will be available in painting tool."), GLGizmoMmuSegmentation::EXTRUDERS_LIMIT));
}

void GLGizmoMmuSegmentation::on_opening()
{
    if (get_extruders_colors().size() > GLGizmoMmuSegmentation::EXTRUDERS_LIMIT)
        show_notification_extruders_limit_exceeded();
}

void GLGizmoMmuSegmentation::on_shutdown()
{
    m_parent.use_slope(false);
    m_parent.toggle_model_objects_visibility(true);
}

std::string GLGizmoMmuSegmentation::on_get_name() const
{
    return _u8L("Color Region Painting");
}

bool GLGizmoMmuSegmentation::on_is_selectable() const
{
    return (wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF
            && /*wxGetApp().get_mode() != comSimple && */wxGetApp().filaments_cnt() > 1);
}

bool GLGizmoMmuSegmentation::on_is_activable() const
{
    const Selection& selection = m_parent.get_selection();
    return !selection.is_empty() && (selection.is_single_full_instance() || selection.is_any_volume()) && wxGetApp().filaments_cnt() > 1;
}

//BBS: use the global one in 3DScene.cpp
/*static std::vector<ColorRGBA> get_extruders_colors()
{
    unsigned char                     rgb_color[3] = {};
    std::vector<std::string>          colors       = Slic3r::GUI::wxGetApp().plater()->get_extruder_colors_from_plater_config();
    std::vector<ColorRGBA> colors_out(colors.size());
    for (const std::string &color : colors) {
        Slic3r::GUI::BitmapCache::parse_color(color, rgb_color);
        size_t color_idx      = &color - &colors.front();
        colors_out[color_idx] = {float(rgb_color[0]) / 255.f, float(rgb_color[1]) / 255.f, float(rgb_color[2]) / 255.f, 1.f};
    }

    return colors_out;
}*/

static std::vector<int> get_extruder_id_for_volumes(const ModelObject &model_object)
{
    std::vector<int> extruders_idx;
    extruders_idx.reserve(model_object.volumes.size());
    for (const ModelVolume *model_volume : model_object.volumes) {
        if (!model_volume->is_model_part())
            continue;

        extruders_idx.emplace_back(model_volume->extruder_id());
    }

    return extruders_idx;
}

static std::vector<unsigned int> get_display_filament_ids(size_t total_filaments)
{
    std::vector<unsigned int> ordered_filament_ids;
    if (wxGetApp().plater() != nullptr)
        ordered_filament_ids = wxGetApp().plater()->sidebar().get_ui_ordered_filament_ids();

    std::vector<unsigned int> sanitized_filament_ids;
    sanitized_filament_ids.reserve(total_filaments);
    std::vector<bool> used_filament_ids(total_filaments + 1, false);
    const size_t physical_count = size_t(std::max(wxGetApp().filaments_cnt(), 0));
    auto real_filament_id = [physical_count](unsigned int filament_id) {
        if (filament_id >= 1 && filament_id <= physical_count)
            return true;
        return wxGetApp().preset_bundle != nullptr &&
               wxGetApp().preset_bundle->texture_mapping_zones.is_texture_mapping_zone_id(filament_id);
    };
    for (const unsigned int filament_id : ordered_filament_ids) {
        if (filament_id == 0 || filament_id > total_filaments || used_filament_ids[filament_id] || !real_filament_id(filament_id))
            continue;
        used_filament_ids[filament_id] = true;
        sanitized_filament_ids.emplace_back(filament_id);
    }

    for (unsigned int filament_id = 1; filament_id <= total_filaments; ++filament_id) {
        if (!used_filament_ids[filament_id] && real_filament_id(filament_id))
            sanitized_filament_ids.emplace_back(filament_id);
    }

    return sanitized_filament_ids;
}

static unsigned int ensure_texture_mapping_zone()
{
    if (wxGetApp().preset_bundle == nullptr || wxGetApp().plater() == nullptr)
        return 0;

    TextureMappingManager &mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    const size_t num_physical = static_cast<size_t>(std::max(wxGetApp().filaments_cnt(), 0));
    std::vector<std::string> physical_colors = wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, false);
    physical_colors.resize(num_physical, "#26A69A");

    if (unsigned int existing_id = mgr.find_image_texture_zone_id(num_physical); existing_id != 0) {
        if (TextureMappingZone *zone = mgr.zone_from_id(existing_id);
            zone == nullptr || !TextureMappingManager::auto_adjust_texture_component_ids(*zone, num_physical, physical_colors)) {
            return existing_id;
        }
    } else if (num_physical < 2) {
        return 0;
    } else {
        mgr.ensure_image_texture_zone(num_physical, physical_colors);
    }

    const std::string texture_serialized = mgr.serialize_entries();
    DynamicPrintConfig *print_cfg = &wxGetApp().preset_bundle->prints.get_edited_preset().config;
    if (ConfigOptionString *opt = print_cfg->option<ConfigOptionString>("texture_mapping_definitions"))
        opt->value = texture_serialized;
    else
        print_cfg->set_key_value("texture_mapping_definitions", new ConfigOptionString(texture_serialized));

    if (ConfigOptionString *opt = wxGetApp().preset_bundle->project_config.option<ConfigOptionString>("texture_mapping_definitions"))
        opt->value = texture_serialized;
    else
        wxGetApp().preset_bundle->project_config.set_key_value("texture_mapping_definitions", new ConfigOptionString(texture_serialized));

    wxGetApp().sidebar().update_texture_mapping_panel(false);
    wxGetApp().sidebar().update_dynamic_filament_list();
    if (auto *print_tab = wxGetApp().get_tab(Preset::TYPE_PRINT))
        print_tab->update_dirty();
    if (wxGetApp().mainframe != nullptr)
        wxGetApp().mainframe->on_config_changed(print_cfg);

    return mgr.find_image_texture_zone_id(num_physical);
}

static bool model_volume_has_imported_image_texture_data(const ModelVolume *volume)
{
    return volume != nullptr &&
           !volume->imported_texture_rgba.empty() &&
           volume->imported_texture_width > 0 &&
           volume->imported_texture_height > 0;
}

static bool model_volume_has_bakeable_image_texture_data(const ModelVolume *volume)
{
    if (!model_volume_has_imported_image_texture_data(volume))
        return false;

    const indexed_triangle_set &its = volume->mesh().its;
    return !its.vertices.empty() &&
           !its.indices.empty() &&
           volume->imported_texture_uv_valid.size() == its.indices.size() &&
           volume->imported_texture_uvs_per_face.size() >= its.indices.size() * 6 &&
           volume->imported_texture_rgba.size() >=
               size_t(volume->imported_texture_width) * size_t(volume->imported_texture_height) * 4 &&
           std::any_of(volume->imported_texture_uv_valid.begin(), volume->imported_texture_uv_valid.end(), [](uint8_t valid) {
               return valid != 0;
           });
}

static float wrap_texture_uv_for_vertex_bake(float uv)
{
    if (!std::isfinite(uv))
        return 0.f;

    float wrapped = uv - std::floor(uv);
    if (wrapped < 0.f)
        wrapped += 1.f;
    return wrapped;
}

static ColorRGBA sample_texture_rgba_for_vertex_bake(const std::vector<uint8_t> &rgba,
                                                     uint32_t                    width,
                                                     uint32_t                    height,
                                                     const Vec2f                &uv)
{
    if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    const float u = wrap_texture_uv_for_vertex_bake(uv.x());
    const float v = wrap_texture_uv_for_vertex_bake(uv.y());
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

    return ColorRGBA(blend_channel(0), blend_channel(1), blend_channel(2), 1.f);
}

static uint32_t pack_vertex_color_rgba(const ColorRGBA &color)
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

static ColorRGBA unpack_vertex_color_rgba_for_conversion(uint32_t packed)
{
    return ColorRGBA(float((packed >> 24) & 0xFFu) / 255.f,
                     float((packed >> 16) & 0xFFu) / 255.f,
                     float((packed >> 8) & 0xFFu) / 255.f,
                     float(packed & 0xFFu) / 255.f);
}

static float triangle_max_edge_length(const std::array<Vec3f, 3> &vertices)
{
    return std::max({ (vertices[1] - vertices[0]).norm(),
                      (vertices[2] - vertices[1]).norm(),
                      (vertices[0] - vertices[2]).norm() });
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
    if (!std::isfinite(span) || !std::isfinite(target_span) || span <= target_span || target_span <= EPSILON)
        return 0;

    return std::clamp(int(std::ceil(std::log2(span / target_span))), 0, max_depth);
}

static int texture_mapping_depth_for_budget(size_t triangle_count, int requested_max_depth, size_t max_leaf_triangles)
{
    int depth = std::clamp(requested_max_depth, 0, 7);
    while (depth > 0) {
        double leaf_count = double(std::max<size_t>(triangle_count, 1));
        for (int idx = 0; idx < depth; ++idx)
            leaf_count *= 4.0;
        if (leaf_count <= double(max_leaf_triangles))
            break;
        --depth;
    }
    return depth;
}

static void normalize_color_mix_weights(std::vector<float> &weights)
{
    float sum = 0.f;
    for (float &weight : weights) {
        if (!std::isfinite(weight) || weight < 0.f)
            weight = 0.f;
        sum += weight;
    }

    if (sum <= EPSILON) {
        const float uniform = weights.empty() ? 0.f : 1.f / float(weights.size());
        for (float &weight : weights)
            weight = uniform;
        return;
    }

    const float inv_sum = 1.f / sum;
    for (float &weight : weights)
        weight *= inv_sum;
}

static ColorRGBA color_mix_from_weights(const std::vector<ColorRGBA> &colors,
                                        const std::vector<float>     &weights,
                                        const ColorRGBA              &fallback)
{
    if (colors.empty() || weights.empty())
        return fallback;

    float sum = 0.f;
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    for (size_t idx = 0; idx < colors.size() && idx < weights.size(); ++idx) {
        const float weight = std::max(weights[idx], 0.f);
        sum += weight;
        r += colors[idx].r() * weight;
        g += colors[idx].g() * weight;
        b += colors[idx].b() * weight;
    }

    if (sum <= EPSILON)
        return fallback;

    const float inv_sum = 1.f / sum;
    return ColorRGBA(r * inv_sum, g * inv_sum, b * inv_sum, fallback.a());
}

static float color_mix_error_squared(const std::vector<ColorRGBA> &colors, const std::vector<float> &weights, const ColorRGBA &target)
{
    const ColorRGBA mix = color_mix_from_weights(colors, weights, target);
    return Slic3r::sqr(mix.r() - target.r()) + Slic3r::sqr(mix.g() - target.g()) + Slic3r::sqr(mix.b() - target.b());
}

static std::vector<float> closest_color_mix_weights(const std::vector<ColorRGBA> &colors, const ColorRGBA &target)
{
    std::vector<float> weights(colors.size(), 0.f);
    if (colors.empty())
        return weights;

    auto improve = [&colors, &target](std::vector<float> candidate) {
        normalize_color_mix_weights(candidate);
        float step = 0.28f;
        for (int iter = 0; iter < 140; ++iter) {
            const ColorRGBA mix = color_mix_from_weights(colors, candidate, target);
            const float err_r = mix.r() - target.r();
            const float err_g = mix.g() - target.g();
            const float err_b = mix.b() - target.b();
            std::vector<float> next = candidate;
            for (size_t idx = 0; idx < colors.size(); ++idx) {
                const float grad = 2.f * (err_r * colors[idx].r() + err_g * colors[idx].g() + err_b * colors[idx].b());
                next[idx] -= step * grad;
            }
            normalize_color_mix_weights(next);
            candidate = std::move(next);
            step *= 0.985f;
        }
        return candidate;
    };

    std::vector<float> uniform(colors.size(), 1.f / float(colors.size()));
    weights = improve(uniform);
    float best_error = color_mix_error_squared(colors, weights, target);

    for (size_t idx = 0; idx < colors.size(); ++idx) {
        std::vector<float> single(colors.size(), 0.f);
        single[idx] = 1.f;
        single = improve(std::move(single));
        const float error = color_mix_error_squared(colors, single, target);
        if (error < best_error) {
            best_error = error;
            weights = std::move(single);
        }
    }

    return weights;
}

static float texture_triangle_uv_pixel_span(const ModelVolume *volume, size_t tri_idx)
{
    if (volume == nullptr ||
        tri_idx >= volume->imported_texture_uv_valid.size() ||
        volume->imported_texture_uv_valid[tri_idx] == 0)
        return 0.f;

    const size_t uv_offset = tri_idx * 6;
    if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
        return 0.f;

    const Vec2f uv0(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]);
    const Vec2f uv1(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]);
    const Vec2f uv2(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5]);
    const float width = float(std::max<uint32_t>(volume->imported_texture_width, 1));
    const float height = float(std::max<uint32_t>(volume->imported_texture_height, 1));
    auto pixel_edge_length = [width, height](const Vec2f &a, const Vec2f &b) {
        const Vec2f delta = b - a;
        return std::sqrt(Slic3r::sqr(delta.x() * width) + Slic3r::sqr(delta.y() * height));
    };

    return std::max({ pixel_edge_length(uv0, uv1),
                      pixel_edge_length(uv1, uv2),
                      pixel_edge_length(uv2, uv0) });
}

static bool barycentric_weights_for_region_vertex_colors(const Vec3f &point,
                                                         const Vec3f &p0,
                                                         const Vec3f &p1,
                                                         const Vec3f &p2,
                                                         Vec3f       &weights)
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
    if (std::abs(denom) <= EPSILON)
        return false;

    weights.y() = (d11 * d20 - d01 * d21) / denom;
    weights.z() = (d00 * d21 - d01 * d20) / denom;
    weights.x() = 1.f - weights.y() - weights.z();
    return std::isfinite(weights.x()) && std::isfinite(weights.y()) && std::isfinite(weights.z());
}

static std::string rgb_metadata_json(const ColorRGBA &background)
{
    const uint32_t packed = pack_vertex_color_rgba(background);
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

static ColorRGBA rgb_metadata_background_color(const ColorFacetsAnnotation &annotation)
{
    const std::string &metadata = annotation.metadata_json();
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
    return unpack_vertex_color_rgba_for_conversion(packed);
}

static void refresh_imported_texture_storage(ModelVolume &volume)
{
    std::vector<uint8_t> refreshed(volume.imported_texture_rgba.begin(), volume.imported_texture_rgba.end());
    volume.imported_texture_rgba.swap(refreshed);
}

static std::optional<ColorRGBA> sample_rgb_color_facets(const std::vector<ColorFacetTriangle>                 &facets,
                                                        const std::unordered_map<int, std::vector<size_t>>    &facets_by_source_triangle,
                                                        int                                                    source_triangle,
                                                        const Vec3f                                           &point)
{
    auto found = facets_by_source_triangle.find(source_triangle);
    if (found == facets_by_source_triangle.end())
        return std::nullopt;

    const float tolerance = -1e-4f;
    for (const size_t facet_idx : found->second) {
        if (facet_idx >= facets.size())
            continue;

        const ColorFacetTriangle &facet = facets[facet_idx];
        Vec3f weights = Vec3f::Zero();
        if (!barycentric_weights_for_region_vertex_colors(point, facet.vertices[0], facet.vertices[1], facet.vertices[2], weights))
            continue;
        if (weights.x() >= tolerance && weights.y() >= tolerance && weights.z() >= tolerance)
            return unpack_vertex_color_rgba_for_conversion(facet.rgba);
    }

    if (found->second.empty() || found->second.front() >= facets.size())
        return std::nullopt;
    return unpack_vertex_color_rgba_for_conversion(facets[found->second.front()].rgba);
}

struct RGBStrokeVertexKey
{
    long long x = 0;
    long long y = 0;
    long long z = 0;
};

struct RGBStrokeEdgeKey
{
    RGBStrokeVertexKey a;
    RGBStrokeVertexKey b;
};

struct RGBStrokeBoundaryEdge
{
    int   source_triangle = -1;
    Vec3f a = Vec3f::Zero();
    Vec3f b = Vec3f::Zero();
};

struct RGBStrokeEdgeData
{
    int                   count = 0;
    RGBStrokeBoundaryEdge edge;
};

struct RGBStrokeEdgeKeyHash
{
    size_t operator()(const RGBStrokeEdgeKey &key) const
    {
        size_t hash = 1469598103934665603ull;
        auto mix = [&hash](long long value) {
            hash ^= std::hash<long long>{}(value) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
        };
        mix(key.a.x);
        mix(key.a.y);
        mix(key.a.z);
        mix(key.b.x);
        mix(key.b.y);
        mix(key.b.z);
        return hash;
    }
};

static bool operator==(const RGBStrokeVertexKey &lhs, const RGBStrokeVertexKey &rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

static bool operator==(const RGBStrokeEdgeKey &lhs, const RGBStrokeEdgeKey &rhs)
{
    return lhs.a == rhs.a && lhs.b == rhs.b;
}

static bool rgb_stroke_vertex_key_less(const RGBStrokeVertexKey &lhs, const RGBStrokeVertexKey &rhs)
{
    if (lhs.x != rhs.x)
        return lhs.x < rhs.x;
    if (lhs.y != rhs.y)
        return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

static RGBStrokeVertexKey rgb_stroke_vertex_key(const Vec3f &point)
{
    auto key = [](float value) {
        return std::isfinite(value) ? static_cast<long long>(std::llround(double(value) * 100000.0)) : 0ll;
    };
    return { key(point.x()), key(point.y()), key(point.z()) };
}

struct RGBStrokeBoundaryEdges
{
    std::unordered_map<int, std::vector<RGBStrokeBoundaryEdge>> by_source_triangle;
    std::vector<RGBStrokeBoundaryEdge> all;
};

static RGBStrokeBoundaryEdges build_rgb_stroke_boundary_edges(
    const std::vector<TriangleSelector::FacetStateTriangle> &stroke_facets)
{
    std::unordered_map<RGBStrokeEdgeKey, RGBStrokeEdgeData, RGBStrokeEdgeKeyHash> edges;
    edges.reserve(stroke_facets.size() * 3);

    auto add_edge = [&edges](const TriangleSelector::FacetStateTriangle &facet, const Vec3f &a, const Vec3f &b) {
        RGBStrokeEdgeKey key { rgb_stroke_vertex_key(a), rgb_stroke_vertex_key(b) };
        if (rgb_stroke_vertex_key_less(key.b, key.a))
            std::swap(key.a, key.b);

        RGBStrokeEdgeData &data = edges[key];
        ++data.count;
        if (data.count == 1)
            data.edge = { facet.source_triangle, a, b };
    };

    for (const TriangleSelector::FacetStateTriangle &facet : stroke_facets) {
        add_edge(facet, facet.vertices[0], facet.vertices[1]);
        add_edge(facet, facet.vertices[1], facet.vertices[2]);
        add_edge(facet, facet.vertices[2], facet.vertices[0]);
    }

    RGBStrokeBoundaryEdges boundary_edges;
    boundary_edges.all.reserve(edges.size());
    for (const auto &edge : edges) {
        if (edge.second.count != 1)
            continue;
        boundary_edges.by_source_triangle[edge.second.edge.source_triangle].emplace_back(edge.second.edge);
        boundary_edges.all.emplace_back(edge.second.edge);
    }
    return boundary_edges;
}

static float distance_to_segment(const Vec3f &point, const Vec3f &a, const Vec3f &b)
{
    const Vec3f ab = b - a;
    const float len2 = ab.squaredNorm();
    if (len2 <= EPSILON)
        return (point - a).norm();
    const float t = std::clamp((point - a).dot(ab) / len2, 0.f, 1.f);
    return (point - (a + ab * t)).norm();
}

static float sample_rgb_stroke_alpha(const std::vector<TriangleSelector::FacetStateTriangle>      &stroke_facets,
                                     const std::unordered_map<int, std::vector<size_t>>           &stroke_by_source_triangle,
                                     const RGBStrokeBoundaryEdges                                 &stroke_boundary_edges,
                                     int                                                           source_triangle,
                                     const Vec3f                                                  &point,
                                     float                                                         hardness,
                                     float                                                         opacity,
                                     float                                                         brush_radius)
{
    if (opacity <= 0.f)
        return 0.f;

    auto found = stroke_by_source_triangle.find(source_triangle);
    if (found == stroke_by_source_triangle.end())
        return 0.f;

    const float tolerance = -1e-4f;
    bool inside_stroke = false;
    for (const size_t facet_idx : found->second) {
        if (facet_idx >= stroke_facets.size())
            continue;

        const TriangleSelector::FacetStateTriangle &facet = stroke_facets[facet_idx];
        Vec3f weights = Vec3f::Zero();
        if (!barycentric_weights_for_region_vertex_colors(point, facet.vertices[0], facet.vertices[1], facet.vertices[2], weights))
            continue;
        if (weights.x() < tolerance || weights.y() < tolerance || weights.z() < tolerance)
            continue;
        inside_stroke = true;
        break;
    }

    if (!inside_stroke)
        return 0.f;

    hardness = std::clamp(hardness, 0.f, 1.f);
    opacity = std::clamp(opacity, 0.f, 1.f);
    const float fade_width = brush_radius * (1.f - hardness);
    if (fade_width <= EPSILON)
        return opacity;

    if (stroke_boundary_edges.all.empty())
        return opacity;

    float boundary_distance = std::numeric_limits<float>::max();
    auto boundary_found = stroke_boundary_edges.by_source_triangle.find(source_triangle);
    if (boundary_found != stroke_boundary_edges.by_source_triangle.end())
        for (const RGBStrokeBoundaryEdge &edge : boundary_found->second)
            boundary_distance = std::min(boundary_distance, distance_to_segment(point, edge.a, edge.b));

    if (!std::isfinite(boundary_distance) || boundary_distance > fade_width) {
        for (const RGBStrokeBoundaryEdge &edge : stroke_boundary_edges.all)
            boundary_distance = std::min(boundary_distance, distance_to_segment(point, edge.a, edge.b));
    }

    if (!std::isfinite(boundary_distance))
        return opacity;

    const float t = std::clamp(boundary_distance / fade_width, 0.f, 1.f);
    const float soft_alpha = t * t * (3.f - 2.f * t);
    return opacity * soft_alpha;
}

static bool apply_rgb_stroke_to_volume(ModelVolume                                           &volume,
                                       const std::vector<TriangleSelector::FacetStateTriangle> &stroke_facets,
                                       const ColorRGBA                                      &brush_color,
                                       float                                                hardness,
                                       float                                                opacity,
                                       float                                                brush_radius)
{
    if (stroke_facets.empty())
        return false;

    std::vector<ColorFacetTriangle> existing_facets;
    volume.texture_mapping_color_facets.get_facet_triangles(volume, existing_facets);
    std::unordered_map<int, std::vector<size_t>> existing_by_source_triangle;
    existing_by_source_triangle.reserve(existing_facets.size());
    for (size_t idx = 0; idx < existing_facets.size(); ++idx)
        existing_by_source_triangle[existing_facets[idx].source_triangle].emplace_back(idx);

    std::unordered_map<int, std::vector<size_t>> stroke_by_source_triangle;
    stroke_by_source_triangle.reserve(stroke_facets.size());
    for (size_t idx = 0; idx < stroke_facets.size(); ++idx)
        stroke_by_source_triangle[stroke_facets[idx].source_triangle].emplace_back(idx);
    RGBStrokeBoundaryEdges stroke_boundary_edges = build_rgb_stroke_boundary_edges(stroke_facets);

    const ColorRGBA background = rgb_metadata_background_color(volume.texture_mapping_color_facets);
    const uint32_t brush_packed = pack_vertex_color_rgba(brush_color);
    TextureMappingColorSampler sampler = [&existing_facets,
                                          &existing_by_source_triangle,
                                          &stroke_facets,
                                          &stroke_by_source_triangle,
                                          &stroke_boundary_edges,
                                          background,
                                          brush_color,
                                          brush_packed,
                                          hardness,
                                          opacity,
                                          brush_radius](size_t tri_idx, const Vec3f &point, const Vec3f &) {
        ColorRGBA source_color = background;
        if (std::optional<ColorRGBA> sampled =
                sample_rgb_color_facets(existing_facets, existing_by_source_triangle, int(tri_idx), point)) {
            source_color = *sampled;
        }

        const float alpha = sample_rgb_stroke_alpha(stroke_facets,
                                                    stroke_by_source_triangle,
                                                    stroke_boundary_edges,
                                                    int(tri_idx),
                                                    point,
                                                    hardness,
                                                    opacity,
                                                    brush_radius);
        if (alpha <= 0.f)
            return pack_vertex_color_rgba(source_color);
        if (alpha >= 1.f)
            return brush_packed;

        return pack_vertex_color_rgba(ColorRGBA(source_color.r() * (1.f - alpha) + brush_color.r() * alpha,
                                                source_color.g() * (1.f - alpha) + brush_color.g() * alpha,
                                                source_color.b() * (1.f - alpha) + brush_color.b() * alpha,
                                                source_color.a() * (1.f - alpha) + brush_color.a() * alpha));
    };

    const float mesh_span = mesh_max_axis_span(volume.mesh().its);
    const int safe_max_depth = texture_mapping_depth_for_budget(volume.mesh().its.indices.size(), 7, 1800000);
    TextureMappingColorSubdivisionDepths subdivision_depths =
        [mesh_span, safe_max_depth, &stroke_by_source_triangle](size_t tri_idx, const std::array<Vec3f, 3> &vertices) {
        const int base_depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices),
                                                               std::max(mesh_span / 220.f, 0.18f),
                                                               std::min(6, safe_max_depth));
        if (stroke_by_source_triangle.find(int(tri_idx)) != stroke_by_source_triangle.end())
            return std::make_pair(std::min(std::max(base_depth, 4), safe_max_depth), safe_max_depth);
        return std::make_pair(base_depth, safe_max_depth);
    };

    return volume.texture_mapping_color_facets.set_from_triangle_sampler(volume, sampler, safe_max_depth, 0.012f, subdivision_depths);
}

static bool initialize_volume_rgb_data(ModelVolume &volume, const ColorRGBA &background)
{
    if (volume.mesh().its.indices.empty() || volume.mesh().its.vertices.empty())
        return false;

    const uint32_t packed = pack_vertex_color_rgba(background);
    TextureMappingColorSampler sampler = [packed](size_t, const Vec3f &, const Vec3f &) { return packed; };
    const bool changed = volume.texture_mapping_color_facets.set_from_triangle_sampler(volume, sampler, 0, 0.f);
    volume.texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(background));
    return changed;
}

struct ProjectionContext
{
    Matrix4d                     view_projection = Matrix4d::Identity();
    int                          canvas_width = 1;
    int                          canvas_height = 1;
    float                        overlay_left = 0.f;
    float                        overlay_top = 0.f;
    float                        overlay_width = 0.f;
    float                        overlay_height = 0.f;
    const std::vector<uint8_t>  *image_rgba = nullptr;
    uint32_t                     image_width = 0;
    uint32_t                     image_height = 0;
    float                        image_opacity = 1.f;
    bool                         apply_transparency_as_background = false;
};

struct VolumeColorSource
{
    std::vector<ColorFacetTriangle>              rgb_facets;
    std::unordered_map<int, std::vector<size_t>> rgb_by_source_triangle;
};

static ColorRGBA sample_rgba_bilinear_clamped(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height, float u, float v)
{
    if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    u = std::clamp(u, 0.f, 1.f);
    v = std::clamp(v, 0.f, 1.f);
    const float x = u * float(width > 1 ? width - 1 : 0);
    const float y = v * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto channel = [&rgba, width](size_t sx, size_t sy, size_t ch) {
        return float(rgba[(sy * size_t(width) + sx) * 4 + ch]) / 255.f;
    };
    auto blend_channel = [&](size_t ch) {
        const float c00 = channel(x0, y0, ch);
        const float c10 = channel(x1, y0, ch);
        const float c01 = channel(x0, y1, ch);
        const float c11 = channel(x1, y1, ch);
        return std::clamp((c00 + (c10 - c00) * tx) + ((c01 + (c11 - c01) * tx) - (c00 + (c10 - c00) * tx)) * ty, 0.f, 1.f);
    };
    auto blend_premultiplied_channel = [&](size_t ch) {
        const float c00 = channel(x0, y0, ch) * channel(x0, y0, 3);
        const float c10 = channel(x1, y0, ch) * channel(x1, y0, 3);
        const float c01 = channel(x0, y1, ch) * channel(x0, y1, 3);
        const float c11 = channel(x1, y1, ch) * channel(x1, y1, 3);
        return std::clamp((c00 + (c10 - c00) * tx) + ((c01 + (c11 - c01) * tx) - (c00 + (c10 - c00) * tx)) * ty, 0.f, 1.f);
    };

    const float a = blend_channel(3);
    if (a <= 0.f)
        return ColorRGBA(blend_channel(0), blend_channel(1), blend_channel(2), 0.f);
    return ColorRGBA(std::clamp(blend_premultiplied_channel(0) / a, 0.f, 1.f),
                     std::clamp(blend_premultiplied_channel(1) / a, 0.f, 1.f),
                     std::clamp(blend_premultiplied_channel(2) / a, 0.f, 1.f),
                     a);
}

static ColorRGBA blend_projection_color(const ColorRGBA &base, const ColorRGBA &overlay, float opacity)
{
    const float alpha = std::clamp(overlay.a(), 0.f, 1.f) * std::clamp(opacity, 0.f, 1.f);
    if (alpha <= 0.f)
        return base;
    const float out_alpha = std::clamp(alpha + base.a() * (1.f - alpha), 0.f, 1.f);
    if (out_alpha <= EPSILON)
        return ColorRGBA(overlay.r(), overlay.g(), overlay.b(), 0.f);
    return ColorRGBA(base.r() * (1.f - alpha) + overlay.r() * alpha,
                     base.g() * (1.f - alpha) + overlay.g() * alpha,
                     base.b() * (1.f - alpha) + overlay.b() * alpha,
                     out_alpha);
}

static float projection_overlay_alpha(const ColorRGBA &overlay, const ProjectionContext &context)
{
    return std::clamp(overlay.a(), 0.f, 1.f) * std::clamp(context.image_opacity, 0.f, 1.f);
}

static bool projection_overlay_has_paintable_alpha(const ColorRGBA &overlay, const ProjectionContext &context)
{
    return projection_overlay_alpha(overlay, context) > 0.5f / 255.f;
}

static ColorRGBA apply_projection_color(const ColorRGBA &base, const ColorRGBA &overlay, const ProjectionContext &context, bool image_texture_target)
{
    const float opacity = std::clamp(context.image_opacity, 0.f, 1.f);
    if (!context.apply_transparency_as_background)
        return blend_projection_color(base, overlay, opacity);

    const float alpha = std::clamp(overlay.a(), 0.f, 1.f) * opacity;
    if (image_texture_target)
        return ColorRGBA(overlay.r() * alpha, overlay.g() * alpha, overlay.b() * alpha, 1.f);
    return ColorRGBA(overlay.r(), overlay.g(), overlay.b(), alpha);
}

static bool wx_image_to_rgba(const wxImage &image, std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height)
{
    if (!image.IsOk() || image.GetWidth() <= 0 || image.GetHeight() <= 0)
        return false;

    width = uint32_t(image.GetWidth());
    height = uint32_t(image.GetHeight());
    rgba.assign(size_t(width) * size_t(height) * 4, 255);

    const unsigned char *rgb = image.GetData();
    const unsigned char *alpha = image.HasAlpha() ? image.GetAlpha() : nullptr;
    const bool has_mask = image.HasMask();
    const int mask_r = has_mask ? image.GetMaskRed() : -1;
    const int mask_g = has_mask ? image.GetMaskGreen() : -1;
    const int mask_b = has_mask ? image.GetMaskBlue() : -1;
    if (rgb == nullptr)
        return false;

    for (size_t idx = 0; idx < size_t(width) * size_t(height); ++idx) {
        rgba[idx * 4 + 0] = rgb[idx * 3 + 0];
        rgba[idx * 4 + 1] = rgb[idx * 3 + 1];
        rgba[idx * 4 + 2] = rgb[idx * 3 + 2];
        rgba[idx * 4 + 3] =
            has_mask &&
            int(rgb[idx * 3 + 0]) == mask_r &&
            int(rgb[idx * 3 + 1]) == mask_g &&
            int(rgb[idx * 3 + 2]) == mask_b ?
                0 :
                (alpha == nullptr ? 255 : alpha[idx]);
    }
    return true;
}

static bool project_point_to_screen(const ProjectionContext &context, const Vec3d &world_point, Vec2f &screen, float *ndc_z = nullptr)
{
    const Vec4d clip = context.view_projection * Vec4d(world_point.x(), world_point.y(), world_point.z(), 1.0);
    if (clip.w() <= 0.0)
        return false;

    const Vec3d ndc = clip.head<3>() / clip.w();
    if (ndc.x() < -1.0 || ndc.x() > 1.0 || ndc.y() < -1.0 || ndc.y() > 1.0 || ndc.z() < -1.0 || ndc.z() > 1.0)
        return false;

    screen.x() = float((ndc.x() * 0.5 + 0.5) * double(context.canvas_width));
    screen.y() = float((1.0 - (ndc.y() * 0.5 + 0.5)) * double(context.canvas_height));
    if (ndc_z != nullptr)
        *ndc_z = float(ndc.z());
    return true;
}

static std::optional<ColorRGBA> projected_image_color_at_point(const ProjectionContext &context,
                                                               const Transform3d      &world_matrix,
                                                               const Vec3f            &point)
{
    if (context.image_rgba == nullptr || context.overlay_width <= 0.f || context.overlay_height <= 0.f)
        return std::nullopt;

    Vec2f screen = Vec2f::Zero();
    if (!project_point_to_screen(context, world_matrix * point.cast<double>(), screen))
        return std::nullopt;
    if (screen.x() < context.overlay_left ||
        screen.y() < context.overlay_top ||
        screen.x() > context.overlay_left + context.overlay_width ||
        screen.y() > context.overlay_top + context.overlay_height)
        return std::nullopt;

    const float u = (screen.x() - context.overlay_left) / context.overlay_width;
    const float v = (screen.y() - context.overlay_top) / context.overlay_height;
    return sample_rgba_bilinear_clamped(*context.image_rgba, context.image_width, context.image_height, u, v);
}

static bool projection_triangle_intersects_overlay(const ProjectionContext      &context,
                                                   const Transform3d           &world_matrix,
                                                   const std::array<Vec3f, 3>  &vertices)
{
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    bool any_projected = false;

    for (const Vec3f &vertex : vertices) {
        Vec2f screen = Vec2f::Zero();
        if (!project_point_to_screen(context, world_matrix * vertex.cast<double>(), screen))
            continue;
        min_x = std::min(min_x, screen.x());
        min_y = std::min(min_y, screen.y());
        max_x = std::max(max_x, screen.x());
        max_y = std::max(max_y, screen.y());
        any_projected = true;
    }

    if (!any_projected)
        return false;
    return max_x >= context.overlay_left &&
           min_x <= context.overlay_left + context.overlay_width &&
           max_y >= context.overlay_top &&
           min_y <= context.overlay_top + context.overlay_height;
}

static bool barycentric_weights_2d(const Vec2f &point, const Vec2f &a, const Vec2f &b, const Vec2f &c, Vec3f &weights)
{
    const Vec2f v0 = b - a;
    const Vec2f v1 = c - a;
    const Vec2f v2 = point - a;
    const float d00 = v0.dot(v0);
    const float d01 = v0.dot(v1);
    const float d11 = v1.dot(v1);
    const float d20 = v2.dot(v0);
    const float d21 = v2.dot(v1);
    const float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) <= EPSILON)
        return false;

    weights.y() = (d11 * d20 - d01 * d21) / denom;
    weights.z() = (d00 * d21 - d01 * d20) / denom;
    weights.x() = 1.f - weights.y() - weights.z();
    return std::isfinite(weights.x()) && std::isfinite(weights.y()) && std::isfinite(weights.z());
}

static Vec3f normalized_nonnegative_barycentric(Vec3f weights)
{
    weights.x() = std::max(weights.x(), 0.f);
    weights.y() = std::max(weights.y(), 0.f);
    weights.z() = std::max(weights.z(), 0.f);
    const float sum = weights.x() + weights.y() + weights.z();
    if (sum <= EPSILON)
        return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
    weights /= sum;
    return weights;
}

static float distance_to_segment_2d(const Vec2f &point, const Vec2f &a, const Vec2f &b)
{
    const Vec2f ab = b - a;
    const float len2 = ab.squaredNorm();
    if (len2 <= EPSILON)
        return (point - a).norm();
    const float t = std::clamp((point - a).dot(ab) / len2, 0.f, 1.f);
    return (point - (a + ab * t)).norm();
}

static bool conservative_barycentric_weights_2d(const Vec2f &point,
                                                const Vec2f &a,
                                                const Vec2f &b,
                                                const Vec2f &c,
                                                float         tolerance,
                                                Vec3f        &weights)
{
    if (!barycentric_weights_2d(point, a, b, c, weights))
        return false;
    if (weights.x() >= -1e-4f && weights.y() >= -1e-4f && weights.z() >= -1e-4f)
        return true;

    const float distance = std::min({ distance_to_segment_2d(point, a, b),
                                      distance_to_segment_2d(point, b, c),
                                      distance_to_segment_2d(point, c, a) });
    if (distance > tolerance)
        return false;

    weights = normalized_nonnegative_barycentric(weights);
    return true;
}

static std::array<Vec2f, 3> unwrap_projection_uvs(std::array<Vec2f, 3> uvs)
{
    auto unwrap_axis = [&uvs](bool use_u_axis) mutable {
        std::array<float, 3> values = {
            use_u_axis ? uvs[0].x() : uvs[0].y(),
            use_u_axis ? uvs[1].x() : uvs[1].y(),
            use_u_axis ? uvs[2].x() : uvs[2].y()
        };

        const float min_value = std::min({ values[0], values[1], values[2] });
        const float max_value = std::max({ values[0], values[1], values[2] });
        if (max_value - min_value <= 0.5f)
            return;

        for (float &value : values)
            if (value < 0.5f)
                value += 1.f;

        if (use_u_axis) {
            uvs[0].x() = values[0];
            uvs[1].x() = values[1];
            uvs[2].x() = values[2];
        } else {
            uvs[0].y() = values[0];
            uvs[1].y() = values[1];
            uvs[2].y() = values[2];
        }
    };

    unwrap_axis(true);
    unwrap_axis(false);

    const float min_u = std::min({ uvs[0].x(), uvs[1].x(), uvs[2].x() });
    const float min_v = std::min({ uvs[0].y(), uvs[1].y(), uvs[2].y() });
    const Vec2f offset(std::floor(min_u), std::floor(min_v));
    for (Vec2f &uv : uvs)
        uv -= offset;

    return uvs;
}

static uint32_t wrapped_texture_pixel(int value, uint32_t size)
{
    if (size == 0)
        return 0;
    int wrapped = value % int(size);
    if (wrapped < 0)
        wrapped += int(size);
    return uint32_t(wrapped);
}

static VolumeColorSource build_volume_color_source(const ModelVolume &volume)
{
    VolumeColorSource source;
    if (!volume.texture_mapping_color_facets.empty()) {
        volume.texture_mapping_color_facets.get_facet_triangles(volume, source.rgb_facets);
        source.rgb_by_source_triangle.reserve(source.rgb_facets.size());
        for (size_t idx = 0; idx < source.rgb_facets.size(); ++idx)
            source.rgb_by_source_triangle[source.rgb_facets[idx].source_triangle].emplace_back(idx);
    }
    return source;
}

static ColorRGBA sample_volume_color_source(const ModelVolume       &volume,
                                            const VolumeColorSource &source,
                                            size_t                   tri_idx,
                                            const Vec3f             &point,
                                            const Vec3f             &barycentric,
                                            bool                     use_image_texture = true,
                                            const ColorRGBA         *fallback_color = nullptr)
{
    if (!volume.texture_mapping_color_facets.empty()) {
        if (std::optional<ColorRGBA> color = sample_rgb_color_facets(source.rgb_facets,
                                                                     source.rgb_by_source_triangle,
                                                                     int(tri_idx),
                                                                     point))
            return *color;
        return rgb_metadata_background_color(volume.texture_mapping_color_facets);
    }

    const indexed_triangle_set &its = volume.mesh().its;
    if (use_image_texture && model_volume_has_bakeable_image_texture_data(&volume) && tri_idx < volume.imported_texture_uv_valid.size()) {
        const size_t uv_offset = tri_idx * 6;
        if (volume.imported_texture_uv_valid[tri_idx] != 0 && uv_offset + 5 < volume.imported_texture_uvs_per_face.size()) {
            const Vec2f uv0(volume.imported_texture_uvs_per_face[uv_offset + 0], volume.imported_texture_uvs_per_face[uv_offset + 1]);
            const Vec2f uv1(volume.imported_texture_uvs_per_face[uv_offset + 2], volume.imported_texture_uvs_per_face[uv_offset + 3]);
            const Vec2f uv2(volume.imported_texture_uvs_per_face[uv_offset + 4], volume.imported_texture_uvs_per_face[uv_offset + 5]);
            const Vec2f uv = uv0 * barycentric.x() + uv1 * barycentric.y() + uv2 * barycentric.z();
            return sample_texture_rgba_for_vertex_bake(volume.imported_texture_rgba,
                                                       volume.imported_texture_width,
                                                       volume.imported_texture_height,
                                                       uv);
        }
    }

    if (volume.imported_vertex_colors_rgba.size() == its.vertices.size() && tri_idx < its.indices.size()) {
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        if (tri[0] >= 0 && tri[1] >= 0 && tri[2] >= 0 &&
            size_t(tri[0]) < volume.imported_vertex_colors_rgba.size() &&
            size_t(tri[1]) < volume.imported_vertex_colors_rgba.size() &&
            size_t(tri[2]) < volume.imported_vertex_colors_rgba.size()) {
            const ColorRGBA c0 = unpack_vertex_color_rgba_for_conversion(volume.imported_vertex_colors_rgba[size_t(tri[0])]);
            const ColorRGBA c1 = unpack_vertex_color_rgba_for_conversion(volume.imported_vertex_colors_rgba[size_t(tri[1])]);
            const ColorRGBA c2 = unpack_vertex_color_rgba_for_conversion(volume.imported_vertex_colors_rgba[size_t(tri[2])]);
            return ColorRGBA(c0.r() * barycentric.x() + c1.r() * barycentric.y() + c2.r() * barycentric.z(),
                             c0.g() * barycentric.x() + c1.g() * barycentric.y() + c2.g() * barycentric.z(),
                             c0.b() * barycentric.x() + c1.b() * barycentric.y() + c2.b() * barycentric.z(),
                             c0.a() * barycentric.x() + c1.a() * barycentric.y() + c2.a() * barycentric.z());
        }
    }

    return fallback_color != nullptr ? *fallback_color : ColorRGBA(1.f, 1.f, 1.f, 1.f);
}

static bool initialize_volume_rgb_data_from_current_surface_color(ModelVolume &volume, const ColorRGBA &fallback_color)
{
    const indexed_triangle_set &its = volume.mesh().its;
    if (its.indices.empty() || its.vertices.empty())
        return false;

    const bool has_image_texture = model_volume_has_bakeable_image_texture_data(&volume);
    const bool has_vertex_colors = volume.imported_vertex_colors_rgba.size() == its.vertices.size();
    if (!has_image_texture && !has_vertex_colors)
        return initialize_volume_rgb_data(volume, fallback_color);

    const VolumeColorSource source = build_volume_color_source(volume);
    TextureMappingColorSampler sampler = [&volume, source, fallback_color](size_t tri_idx, const Vec3f &point, const Vec3f &barycentric) {
        return pack_vertex_color_rgba(sample_volume_color_source(volume, source, tri_idx, point, barycentric, true, &fallback_color));
    };

    bool changed = false;
    if (has_image_texture) {
        const int safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 3200000);
        TextureMappingColorSubdivisionDepths subdivision_depths = [&volume, safe_max_depth](size_t tri_idx, const std::array<Vec3f, 3> &) {
            const int depth = texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(&volume, tri_idx), 8.f, safe_max_depth);
            return std::make_pair(depth, depth);
        };
        changed = volume.texture_mapping_color_facets.set_from_triangle_sampler(volume,
                                                                                sampler,
                                                                                safe_max_depth,
                                                                                0.015f,
                                                                                subdivision_depths);
    } else {
        const float target_edge = std::max(mesh_max_axis_span(its) / 160.f, 0.25f);
        TextureMappingColorSubdivisionDepths subdivision_depths = [target_edge](size_t, const std::array<Vec3f, 3> &vertices) {
            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_edge, 5);
            return std::make_pair(depth, depth);
        };
        changed = volume.texture_mapping_color_facets.set_from_triangle_sampler(volume, sampler, 5, 0.025f, subdivision_depths);
    }

    if (changed && volume.texture_mapping_color_facets.metadata_json().empty())
        volume.texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(fallback_color));
    return changed;
}

static ColorRGBA projection_base_color_for_volume(const ModelVolume &volume)
{
    std::vector<ColorRGBA> colors = get_extruders_colors();
    if (!colors.empty()) {
        int extruder_idx = volume.extruder_id() > 0 ? volume.extruder_id() - 1 : 0;
        extruder_idx = std::clamp(extruder_idx, 0, int(colors.size() - 1));
        ColorRGBA color = colors[size_t(extruder_idx)];
        color.a(1.f);
        return color;
    }
    return ColorRGBA(0.15f, 0.65f, 0.6f, 1.f);
}

static uint32_t projection_texture_size_for_triangles(size_t triangle_count)
{
    const uint32_t grid = uint32_t(std::max<size_t>(1, size_t(std::ceil(std::sqrt(double(std::max<size_t>(triangle_count, 1)))))));
    uint32_t size = 256;
    while (size < grid * 8 && size < 4096)
        size *= 2;
    return std::clamp<uint32_t>(size, 256, 4096);
}

static bool write_rgba_pixel(std::vector<uint8_t> &rgba, uint32_t width, uint32_t x, uint32_t y, const ColorRGBA &color)
{
    if (width == 0)
        return false;
    const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4;
    if (idx + 3 >= rgba.size())
        return false;
    const uint8_t r = uint8_t(std::clamp(color.r(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t g = uint8_t(std::clamp(color.g(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t b = uint8_t(std::clamp(color.b(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t a = uint8_t(std::clamp(color.a(), 0.f, 1.f) * 255.f + 0.5f);
    if (rgba[idx + 0] == r && rgba[idx + 1] == g && rgba[idx + 2] == b && rgba[idx + 3] == a)
        return false;
    rgba[idx + 0] = r;
    rgba[idx + 1] = g;
    rgba[idx + 2] = b;
    rgba[idx + 3] = a;
    return true;
}

static ColorRGBA read_rgba_pixel(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t x, uint32_t y)
{
    if (width == 0)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4;
    if (idx + 3 >= rgba.size())
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    return ColorRGBA(float(rgba[idx + 0]) / 255.f,
                     float(rgba[idx + 1]) / 255.f,
                     float(rgba[idx + 2]) / 255.f,
                     float(rgba[idx + 3]) / 255.f);
}

static Transform3d projection_world_matrix_for_volume(const GLCanvas3D &parent,
                                                      const ModelObject *object,
                                                      const ModelVolume *volume,
                                                      int                instance_idx)
{
    if (object == nullptr || volume == nullptr || object->instances.empty())
        return Transform3d::Identity();

    instance_idx = std::clamp(instance_idx, 0, int(object->instances.size() - 1));
    const ModelInstance *instance = object->instances[size_t(instance_idx)];
    if (parent.get_canvas_type() == GLCanvas3D::CanvasAssembleView)
        return instance->get_assemble_transformation().get_matrix() * volume->get_matrix();
    return instance->get_transformation().get_matrix() * volume->get_matrix();
}

struct ProjectionVisibility
{
    int                width = 0;
    int                height = 0;
    float              left = 0.f;
    float              top = 0.f;
    float              scale = 1.f;
    std::vector<float> depth;
};

static bool projection_visibility_valid(const ProjectionVisibility &visibility)
{
    return visibility.width > 0 &&
           visibility.height > 0 &&
           visibility.depth.size() == size_t(visibility.width) * size_t(visibility.height);
}

static ProjectionVisibility build_projection_visibility(const ProjectionContext &context,
                                                        const GLCanvas3D       &parent,
                                                        const ModelObject      *object,
                                                        int                     instance_idx)
{
    ProjectionVisibility visibility;
    if (object == nullptr || context.overlay_width <= 0.f || context.overlay_height <= 0.f)
        return visibility;

    const float max_dim = std::max(context.overlay_width, context.overlay_height);
    visibility.scale = max_dim > 2048.f ? 2048.f / max_dim : 1.f;
    visibility.width = std::max(1, int(std::ceil(context.overlay_width * visibility.scale)));
    visibility.height = std::max(1, int(std::ceil(context.overlay_height * visibility.scale)));
    visibility.left = context.overlay_left;
    visibility.top = context.overlay_top;
    visibility.depth.assign(size_t(visibility.width) * size_t(visibility.height), std::numeric_limits<float>::max());

    auto rasterize_triangle = [&visibility](const std::array<Vec2f, 3> &screen, const std::array<float, 3> &depths) {
        const float min_screen_x = std::min({ screen[0].x(), screen[1].x(), screen[2].x() });
        const float max_screen_x = std::max({ screen[0].x(), screen[1].x(), screen[2].x() });
        const float min_screen_y = std::min({ screen[0].y(), screen[1].y(), screen[2].y() });
        const float max_screen_y = std::max({ screen[0].y(), screen[1].y(), screen[2].y() });
        const int min_x = std::clamp(int(std::floor((min_screen_x - visibility.left) * visibility.scale)) - 1, 0, visibility.width - 1);
        const int max_x = std::clamp(int(std::ceil((max_screen_x - visibility.left) * visibility.scale)) + 1, 0, visibility.width - 1);
        const int min_y = std::clamp(int(std::floor((min_screen_y - visibility.top) * visibility.scale)) - 1, 0, visibility.height - 1);
        const int max_y = std::clamp(int(std::ceil((max_screen_y - visibility.top) * visibility.scale)) + 1, 0, visibility.height - 1);

        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                const Vec2f pixel(visibility.left + (float(x) + 0.5f) / visibility.scale,
                                  visibility.top + (float(y) + 0.5f) / visibility.scale);
                Vec3f weights = Vec3f::Zero();
                if (!barycentric_weights_2d(pixel, screen[0], screen[1], screen[2], weights))
                    continue;
                if (weights.x() < -1e-4f || weights.y() < -1e-4f || weights.z() < -1e-4f)
                    continue;

                const float depth = depths[0] * weights.x() + depths[1] * weights.y() + depths[2] * weights.z();
                const size_t idx = size_t(y) * size_t(visibility.width) + size_t(x);
                visibility.depth[idx] = std::min(visibility.depth[idx], depth);
            }
        }
    };

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const Transform3d world_matrix = projection_world_matrix_for_volume(parent, object, volume, instance_idx);
        for (const stl_triangle_vertex_indices &tri : its.indices) {
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            if (!projection_triangle_intersects_overlay(context, world_matrix, vertices))
                continue;

            std::array<Vec2f, 3> screen;
            std::array<float, 3> depths;
            bool projected = true;
            for (size_t idx = 0; idx < vertices.size(); ++idx) {
                if (!project_point_to_screen(context, world_matrix * vertices[idx].cast<double>(), screen[idx], &depths[idx])) {
                    projected = false;
                    break;
                }
            }
            if (projected)
                rasterize_triangle(screen, depths);
        }
    }

    return visibility;
}

static bool projection_point_is_visible(const ProjectionVisibility &visibility,
                                        const ProjectionContext    &context,
                                        const Transform3d         &world_matrix,
                                        const Vec3f               &point)
{
    if (!projection_visibility_valid(visibility))
        return true;

    Vec2f screen = Vec2f::Zero();
    float depth = 0.f;
    if (!project_point_to_screen(context, world_matrix * point.cast<double>(), screen, &depth))
        return false;

    const int x = int(std::floor((screen.x() - visibility.left) * visibility.scale));
    const int y = int(std::floor((screen.y() - visibility.top) * visibility.scale));
    if (x < 0 || y < 0 || x >= visibility.width || y >= visibility.height)
        return false;

    const float nearest = visibility.depth[size_t(y) * size_t(visibility.width) + size_t(x)];
    if (!std::isfinite(nearest))
        return false;
    return depth <= nearest + 2e-3f;
}

void GLGizmoMmuSegmentation::init_extruders_data(const std::vector<ColorRGBA> &extruder_colors)
{
    const unsigned int old_selected_filament_id =
        m_selected_extruder_idx < m_display_filament_ids.size() ? m_display_filament_ids[m_selected_extruder_idx] :
        (m_selected_extruder_idx < m_extruders_colors.size() ? unsigned(m_selected_extruder_idx + 1) : 0);

    m_extruders_colors = extruder_colors;
    m_display_filament_ids = get_display_filament_ids(m_extruders_colors.size());

    m_selected_extruder_idx = 0;
    if (!m_display_filament_ids.empty()) {
        auto selected_it = std::find(m_display_filament_ids.begin(), m_display_filament_ids.end(), old_selected_filament_id);
        if (selected_it != m_display_filament_ids.end())
            m_selected_extruder_idx = size_t(std::distance(m_display_filament_ids.begin(), selected_it));
    }

    // keep remap table consistent with current extruder count
    m_extruder_remap.resize(m_display_filament_ids.size());
    for (size_t i = 0; i < m_extruder_remap.size(); ++i)
        m_extruder_remap[i] = i;
}

void GLGizmoMmuSegmentation::init_extruders_data()
{
    init_extruders_data(get_extruders_colors());
}

bool GLGizmoMmuSegmentation::on_init()
{
    // BBS
    m_shortcut_key = WXK_CONTROL_N;

    // FIXME: maybe should be using GUI::shortkey_ctrl_prefix() or equivalent?
    const wxString ctrl  = _L("Ctrl+");
    // FIXME: maybe should be using GUI::shortkey_alt_prefix() or equivalent?
    const wxString alt   = _L("Alt+");
    const wxString shift = _L("Shift+");

    m_desc["clipping_of_view_caption"] = alt + _L("Mouse wheel");
    m_desc["clipping_of_view"]     = _L("Section view");
    m_desc["reset_direction"]     = _L("Reset direction");
    m_desc["cursor_size_caption"]  = ctrl + _L("Mouse wheel");
    m_desc["cursor_size"]          = _L("Pen size");
    m_desc["cursor_type"]          = _L("Pen shape");

    m_desc["paint_caption"]        = _L("Left mouse button");
    m_desc["paint"]                = _L("Paint");
    m_desc["erase_caption"]        = shift + _L("Left mouse button");
    m_desc["erase"]                = _L("Erase");
    m_desc["shortcut_key_caption"] = _L("Key 1~9");
    m_desc["shortcut_key"]         = _L("Choose filament");
    m_desc["edge_detection"]       = _L("Edge detection");
    m_desc["gap_area_caption"]     = ctrl + _L("Mouse wheel");
    m_desc["gap_area"]             = _L("Gap area");
    m_desc["perform"]              = _L("Perform");

    m_desc["remove_all"]           = _L("Erase all painting");
    m_desc["circle"]               = _L("Circle");
    m_desc["sphere"]               = _L("Sphere");
    m_desc["pointer"]              = _L("Triangles");

    m_desc["filaments"]            = _L("Filaments");
    m_desc["tool_type"]            = _L("Tool type");
    m_desc["tool_brush"]           = _L("Brush");
    m_desc["tool_smart_fill"]      = _L("Smart fill");
    m_desc["tool_bucket_fill"]     = _L("Bucket fill");

    m_desc["smart_fill_angle_caption"] = ctrl + _L("Mouse wheel");
    m_desc["smart_fill_angle"]     = _L("Smart fill angle");

    m_desc["height_range_caption"] = ctrl + _L("Mouse wheel");
    m_desc["height_range"]         = _L("Height range");

    //add toggle wire frame hint
    m_desc["toggle_wireframe_caption"]        = alt + shift + _L("Enter");
    m_desc["toggle_wireframe"]                = _L("Toggle Wireframe");

    // Filament remapping descriptions
    m_desc["perform_remap"]                   = _L("Remap filaments");
    m_desc["remap"]                           = _L("Remap");
    m_desc["cancel_remap"]                    = _L("Cancel");

    init_extruders_data();

    return true;
}

GLGizmoMmuSegmentation::GLGizmoMmuSegmentation(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id)
    : GLGizmoPainterBase(parent, icon_filename, sprite_id), m_current_tool(ImGui::CircleButtonIcon)
{
}

void GLGizmoMmuSegmentation::render_painter_gizmo()
{
    const Selection& selection = m_parent.get_selection();

    glsafe(::glEnable(GL_BLEND));
    glsafe(::glEnable(GL_DEPTH_TEST));

    render_triangles(selection);

    m_c->object_clipper()->render_cut();
    m_c->instances_hider()->render_cut();
    render_cursor();

    glsafe(::glDisable(GL_BLEND));
}

void GLGizmoMmuSegmentation::data_changed(bool is_serializing)
{
    GLGizmoPainterBase::data_changed(is_serializing);
    if (m_state != On || wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() != ptFFF || wxGetApp().extruders_edited_cnt() <= 1)
        return;

    ModelObject* model_object = m_c->selection_info()->model_object();
    const std::vector<ColorRGBA> current_extruder_colors = get_extruders_colors();
    const int prev_extruders_count = int(m_extruders_colors.size());
    const int current_extruders_count = int(current_extruder_colors.size());
    const std::vector<unsigned int> current_display_filament_ids = get_display_filament_ids(current_extruder_colors.size());
    if (prev_extruders_count != current_extruders_count) {
        if (current_extruder_colors.size() > GLGizmoMmuSegmentation::EXTRUDERS_LIMIT)
            show_notification_extruders_limit_exceeded();

        this->init_extruders_data(current_extruder_colors);
        // Reinitialize triangle selectors because of change of extruder count need also change the size of GLIndexedVertexArray
        if (prev_extruders_count != current_extruders_count)
            this->init_model_triangle_selectors();
    }
    else if (current_extruder_colors != m_extruders_colors) {
        this->init_extruders_data(current_extruder_colors);
        this->update_triangle_selectors_colors();
    }
    else if (current_display_filament_ids != m_display_filament_ids) {
        this->init_extruders_data(current_extruder_colors);
    }
    else if (model_object != nullptr && get_extruder_id_for_volumes(*model_object) != m_volumes_extruder_idxs) {
        this->init_model_triangle_selectors();
    }
}

// BBS
bool GLGizmoMmuSegmentation::on_number_key_down(int number)
{
    int extruder_idx = number - 1;
    if (extruder_idx >= 0 && size_t(extruder_idx) < m_display_filament_ids.size())
        m_selected_extruder_idx = extruder_idx;

    return true;
}

bool GLGizmoMmuSegmentation::on_key_down_select_tool_type(int keyCode) {
    switch (keyCode)
    {
    case 'F':
        m_current_tool = ImGui::FillButtonIcon;
        break;
    case 'T':
        m_current_tool = ImGui::TriangleButtonIcon;
        break;
    case 'S':
        m_current_tool = ImGui::SphereButtonIcon;
        break;
    case 'C':
        m_current_tool = ImGui::CircleButtonIcon;
        break;
    case 'H':
        m_current_tool = ImGui::HeightRangeIcon;
        break;
    case 'G':
        m_current_tool = ImGui::GapFillIcon;
        break;
    default:
        return false;
        break;
    }
    return true;
}

static void render_extruders_combo(const std::string& label,
                                   const std::vector<std::string>& extruders,
                                   const std::vector<ColorRGBA>& extruders_colors,
                                   size_t& selection_idx)
{
    assert(!extruders_colors.empty());
    assert(extruders.size() == extruders_colors.size());

    size_t selection_out = selection_idx;
    // It is necessary to use BeginGroup(). Otherwise, when using SameLine() is called, then other items will be drawn inside the combobox.
    ImGui::BeginGroup();
    ImVec2 combo_pos = ImGui::GetCursorScreenPos();
    if (ImGui::BeginCombo(label.c_str(), "")) {
        for (size_t extruder_idx = 0; extruder_idx < extruders.size(); ++extruder_idx) {
            ImGui::PushID(int(extruder_idx));
            ImVec2 start_position = ImGui::GetCursorScreenPos();

            if (ImGui::Selectable("", extruder_idx == selection_idx))
                selection_out = extruder_idx;

            ImGui::SameLine();
            ImGuiStyle &style  = ImGui::GetStyle();
            float       height = ImGui::GetTextLineHeight();
            ImGui::GetWindowDrawList()->AddRectFilled(start_position, ImVec2(start_position.x + height + height / 2, start_position.y + height), ImGuiWrapper::to_ImU32(extruders_colors[extruder_idx]));
            ImGui::GetWindowDrawList()->AddRect(start_position, ImVec2(start_position.x + height + height / 2, start_position.y + height), IM_COL32_BLACK);

            ImGui::SetCursorScreenPos(ImVec2(start_position.x + height + height / 2 + style.FramePadding.x, start_position.y));
            ImGui::Text("%s", extruders[extruder_idx].c_str());
            ImGui::PopID();
        }

        ImGui::EndCombo();
    }

    ImVec2      backup_pos = ImGui::GetCursorScreenPos();
    ImGuiStyle &style      = ImGui::GetStyle();

    ImGui::SetCursorScreenPos(ImVec2(combo_pos.x + style.FramePadding.x, combo_pos.y + style.FramePadding.y));
    ImVec2 p      = ImGui::GetCursorScreenPos();
    float  height = ImGui::GetTextLineHeight();

    ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + height + height / 2, p.y + height), ImGuiWrapper::to_ImU32(extruders_colors[selection_idx]));
    ImGui::GetWindowDrawList()->AddRect(p, ImVec2(p.x + height + height / 2, p.y + height), IM_COL32_BLACK);

    ImGui::SetCursorScreenPos(ImVec2(p.x + height + height / 2 + style.FramePadding.x, p.y));
    ImGui::Text("%s", extruders[selection_out].c_str());
    ImGui::SetCursorScreenPos(backup_pos);
    ImGui::EndGroup();

    selection_idx = selection_out;
}

void GLGizmoMmuSegmentation::show_tooltip_information(float caption_max, float x, float y)
{
    ImTextureID normal_id = m_parent.get_gizmos_manager().get_icon_texture_id(GLGizmosManager::MENU_ICON_NAME::IC_TOOLBAR_TOOLTIP);
    ImTextureID hover_id  = m_parent.get_gizmos_manager().get_icon_texture_id(GLGizmosManager::MENU_ICON_NAME::IC_TOOLBAR_TOOLTIP_HOVER);

    caption_max += m_imgui->calc_text_size(std::string_view{": "}).x + 15.f;

    float  scale       = m_parent.get_scale();
    ImVec2 button_size = ImVec2(25 * scale, 25 * scale); // ORCA: Use exact resolution will prevent blur on icon
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0}); // ORCA: Dont add padding
    ImGui::ImageButton3(normal_id, hover_id, button_size);

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip2(ImVec2(x, y));
        auto draw_text_with_caption = [this, &caption_max](const wxString &caption, const wxString &text) {
            m_imgui->text_colored(ImGuiWrapper::COL_ACTIVE, caption);
            ImGui::SameLine(caption_max);
            m_imgui->text_colored(ImGuiWrapper::COL_WINDOW_BG, text);
        };

        std::vector<std::string> tip_items;
        switch (m_tool_type) {
            case ToolType::BRUSH:
                tip_items = {"paint", "erase", "cursor_size", "clipping_of_view", "toggle_wireframe"};
                break;
            case ToolType::BUCKET_FILL:
                tip_items = {"paint", "erase", "smart_fill_angle", "clipping_of_view", "toggle_wireframe"};
                break;
            case ToolType::SMART_FILL:
                // TODO:
                break;
            case ToolType::GAP_FILL:
                tip_items = {"gap_area", "toggle_wireframe"};
                break;
            default:
                break;
        }
        for (const auto &t : tip_items) draw_text_with_caption(m_desc.at(t + "_caption") + ": ", m_desc.at(t));
        ImGui::EndTooltip();
    }
    ImGui::PopStyleVar(2);
}

void GLGizmoMmuSegmentation::on_render_input_window(float x, float y, float bottom_limit)
{
    if (!m_c->selection_info()->model_object()) return;

    const float approx_height = m_imgui->scaled(22.0f);
    y = std::min(y, bottom_limit - approx_height);
    GizmoImguiSetNextWIndowPos(x, y, ImGuiCond_Always);

    wchar_t old_tool = m_current_tool;

    // BBS
    ImGuiWrapper::push_toolbar_style(m_parent.get_scale());
    GizmoImguiBegin(get_name(), ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    // First calculate width of all the texts that are could possibly be shown. We will decide set the dialog width based on that:
    const float space_size = m_imgui->get_style_scaling() * 8;
    const float clipping_slider_left  = std::max(m_imgui->calc_text_size(m_desc.at("clipping_of_view")).x + m_imgui->scaled(1.5f),
        m_imgui->calc_text_size(m_desc.at("reset_direction")).x + m_imgui->scaled(1.5f) + ImGui::GetStyle().FramePadding.x * 2);
    const float cursor_slider_left = m_imgui->calc_text_size(m_desc.at("cursor_size")).x + m_imgui->scaled(1.5f);
    const float smart_fill_slider_left = m_imgui->calc_text_size(m_desc.at("smart_fill_angle")).x + m_imgui->scaled(1.5f);
    const float edge_detect_slider_left = m_imgui->calc_text_size(m_desc.at("edge_detection")).x + m_imgui->scaled(1.f);
    const float gap_area_slider_left = m_imgui->calc_text_size(m_desc.at("gap_area")).x + m_imgui->scaled(1.5f) + space_size;
    const float height_range_slider_left = m_imgui->calc_text_size(m_desc.at("height_range")).x + m_imgui->scaled(2.f);

    const float remove_btn_width = m_imgui->calc_text_size(m_desc.at("remove_all")).x + m_imgui->scaled(1.f);
    const float filter_btn_width = m_imgui->calc_text_size(m_desc.at("perform")).x + m_imgui->scaled(1.f);
    const float remap_btn_width = m_imgui->calc_text_size(m_desc.at("perform_remap")).x + m_imgui->scaled(1.f);
    const float buttons_width = remove_btn_width + filter_btn_width + remap_btn_width + m_imgui->scaled(2.f);
    const float minimal_slider_width = m_imgui->scaled(4.f);
    const float color_button_width = m_imgui->calc_text_size(std::string_view{""}).x + m_imgui->scaled(1.75f);
    const size_t total_filament_count = m_extruders_colors.size();
    const std::string max_filament_label = std::to_string(std::max<size_t>(total_filament_count, 1));
    const ImVec2 max_filament_label_size = ImGui::CalcTextSize(max_filament_label.c_str(), NULL, true);

    float caption_max = 0.f;
    float total_text_max = 0.f;
    for (const auto &t : std::array<std::string, 6>{"paint", "erase", "cursor_size", "smart_fill_angle", "height_range", "clipping_of_view"}) {
        caption_max = std::max(caption_max, m_imgui->calc_text_size(m_desc[t + "_caption"]).x);
        total_text_max = std::max(total_text_max, m_imgui->calc_text_size(m_desc[t]).x);
    }
    total_text_max += caption_max + m_imgui->scaled(1.f);
    caption_max += m_imgui->scaled(1.f);

    const float circle_max_width = std::max(clipping_slider_left,cursor_slider_left);
    const float height_max_width = std::max(clipping_slider_left,height_range_slider_left);
    const float sliders_left_width = std::max(smart_fill_slider_left,
                                         std::max(cursor_slider_left, std::max(edge_detect_slider_left, std::max(gap_area_slider_left, std::max(height_range_slider_left,
                                                                                                                                              clipping_slider_left))))) + space_size;
    const float slider_icon_width = m_imgui->get_slider_icon_size().x;
    float window_width = minimal_slider_width + sliders_left_width + slider_icon_width;
    const int max_filament_items_per_line = 8;
    const float empty_button_width = m_imgui->calc_button_size("").x;
    const float filament_item_width = std::max(empty_button_width, max_filament_label_size.x + m_imgui->scaled(1.4f)) + m_imgui->scaled(1.5f);

    window_width = std::max(window_width, total_text_max);
    window_width = std::max(window_width, buttons_width);
    window_width = std::max(window_width, max_filament_items_per_line * filament_item_width + +m_imgui->scaled(0.5f));

    const float sliders_width = m_imgui->scaled(7.0f);
    const float drag_left_width = ImGui::GetStyle().WindowPadding.x + sliders_width - space_size;

    const float max_tooltip_width = ImGui::GetFontSize() * 20.0f;
    ImDrawList * draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    static float color_button_high  = 25.0;
    draw_list->AddRectFilled({pos.x - 10.0f, pos.y - 7.0f}, {pos.x + window_width + ImGui::GetFrameHeight(), pos.y + color_button_high}, ImGui::GetColorU32(ImGuiCol_FrameBgActive, 1.0f), 5.0f);

    float color_button = ImGui::GetCursorPos().y;

    m_imgui->text(m_desc.at("filaments"));

    float start_pos_x = ImGui::GetCursorPos().x;
    size_t n_extruder_colors = std::min(GLGizmoMmuSegmentation::EXTRUDERS_LIMIT, m_display_filament_ids.size());
    for (size_t extruder_idx = 0; extruder_idx < n_extruder_colors; ++extruder_idx) {
        const unsigned int actual_filament_id = m_display_filament_ids[extruder_idx];
        if (actual_filament_id == 0 || actual_filament_id > m_extruders_colors.size())
            continue;
        const ColorRGBA &extruder_color = m_extruders_colors[actual_filament_id - 1];
        ImVec4           color_vec      = ImGuiWrapper::to_ImVec4(extruder_color);
        std::string color_label = std::string("##extruder color ") + std::to_string(extruder_idx);
        std::string item_text = std::to_string(extruder_idx + 1);
        const ImVec2 label_size = ImGui::CalcTextSize(item_text.c_str(), NULL, true);

        const ImVec2 button_size(max_filament_label_size.x + m_imgui->scaled(0.5f), 0.f);

        float button_offset = start_pos_x;
        if (extruder_idx % max_filament_items_per_line != 0) {
            button_offset += filament_item_width * (extruder_idx % max_filament_items_per_line);
            ImGui::SameLine(button_offset);
        }

        // draw filament background
        ImGuiColorEditFlags flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip;
        if (m_selected_extruder_idx != extruder_idx) flags |= ImGuiColorEditFlags_NoBorder;
        #ifdef __APPLE__
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA); // ORCA use orca color for selected filament border
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0);
            bool color_picked = ImGui::ColorButton(color_label.c_str(), color_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #else
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA); // ORCA use orca color for selected filament border
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0);
            bool color_picked = ImGui::ColorButton(color_label.c_str(), color_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #endif
        color_button_high = ImGui::GetCursorPos().y - color_button - 2.0;
        if (color_picked) { m_selected_extruder_idx = extruder_idx; }

        if (ImGui::IsItemHovered()) {
            if (extruder_idx < 9)
                m_imgui->tooltip(_L("Shortcut Key ") + std::to_string(extruder_idx + 1), max_tooltip_width);
            else
                m_imgui->tooltip(wxString::Format(_L("Filament %d"), int(extruder_idx + 1)), max_tooltip_width);
        }

        // draw filament id
        float gray = 0.299 * extruder_color.r() + 0.587 * extruder_color.g() + 0.114 * extruder_color.b();
        ImGui::SameLine(button_offset + (button_size.x - label_size.x) / 2.f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {10.0,15.0});
        if (gray * 255.f < 80.f)
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "%s", item_text.c_str());
        else
            ImGui::TextColored(ImVec4(0.0f, 0.0f, 0.0f, 1.0f), "%s", item_text.c_str());

        ImGui::PopStyleVar();
    }
    //ImGui::NewLine();
    ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.1));

    if (n_extruder_colors > 0) {
        int selected_filament = int(m_selected_extruder_idx) + 1;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(_L("Selected filament"));
        ImGui::SameLine();
        ImGui::PushItemWidth(m_imgui->scaled(4.5f));
        if (ImGui::InputInt("##selected_filament", &selected_filament, 1, 10, ImGuiInputTextFlags_CharsDecimal)) {
            selected_filament = std::clamp(selected_filament, 1, int(n_extruder_colors));
            m_selected_extruder_idx = size_t(selected_filament - 1);
        }
        ImGui::SameLine();
        m_imgui->text(wxString::Format(_L("/ %d"), int(n_extruder_colors)));
        ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.1));
    }

    m_imgui->text(m_desc.at("tool_type"));

    std::array<wchar_t, 6> tool_ids;
    tool_ids = { ImGui::CircleButtonIcon, ImGui::SphereButtonIcon, ImGui::TriangleButtonIcon, ImGui::HeightRangeIcon, ImGui::FillButtonIcon, ImGui::GapFillIcon };
    std::array<wchar_t, 6> icons;
    if (m_is_dark_mode)
        icons = { ImGui::CircleButtonDarkIcon, ImGui::SphereButtonDarkIcon, ImGui::TriangleButtonDarkIcon, ImGui::HeightRangeDarkIcon, ImGui::FillButtonDarkIcon, ImGui::GapFillDarkIcon };
    else
        icons = { ImGui::CircleButtonIcon, ImGui::SphereButtonIcon, ImGui::TriangleButtonIcon, ImGui::HeightRangeIcon, ImGui::FillButtonIcon, ImGui::GapFillIcon };
    std::array<wxString, 6> tool_tips = { _L("Circle"), _L("Sphere"), _L("Triangle"), _L("Height Range"), _L("Fill"), _L("Gap Fill") };
    for (int i = 0; i < tool_ids.size(); i++) {
        std::string  str_label = std::string("");
        std::wstring btn_name  = icons[i] + boost::nowide::widen(str_label);

        if (i != 0) ImGui::SameLine((empty_button_width + m_imgui->scaled(1.75f)) * i + m_imgui->scaled(1.5f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));                     // ORCA Removes button background on dark mode
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.f));                       // ORCA Fixes icon rendered without colors while using Light theme
        if (m_current_tool == tool_ids[i]) {
            ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.f, 0.59f, 0.53f, 0.25f));  // ORCA use orca color for selected tool / brush
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.f, 0.59f, 0.53f, 0.25f));  // ORCA use orca color for selected tool / brush
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.f, 0.59f, 0.53f, 0.30f));  // ORCA use orca color for selected tool / brush
            ImGui::PushStyleColor(ImGuiCol_Border,          ImGuiWrapper::COL_ORCA);            // ORCA use orca color for border on selected tool / brush
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 1.0);
        }
        bool btn_clicked = ImGui::Button(into_u8(btn_name).c_str());
        if (m_current_tool == tool_ids[i])
        {
            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar(2);
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(1);

        if (btn_clicked && m_current_tool != tool_ids[i]) {
            m_current_tool = tool_ids[i];
            for (auto &triangle_selector : m_triangle_selectors) {
                triangle_selector->seed_fill_unselect_all_triangles();
                triangle_selector->request_update_render_data();
            }
        }

        if (ImGui::IsItemHovered()) {
            m_imgui->tooltip(tool_tips[i], max_tooltip_width);
        }
    }

    ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.1));

    if (m_current_tool != old_tool)
        this->tool_changed(old_tool, m_current_tool);

    if (m_current_tool == ImGui::CircleButtonIcon || m_current_tool == ImGui::SphereButtonIcon) {
        if (m_current_tool == ImGui::CircleButtonIcon)
            m_cursor_type = TriangleSelector::CursorType::CIRCLE;
        else
             m_cursor_type = TriangleSelector::CursorType::SPHERE;
        m_tool_type = ToolType::BRUSH;

        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc.at("cursor_size"));
        ImGui::SameLine(circle_max_width);
        ImGui::PushItemWidth(sliders_width);
        m_imgui->bbl_slider_float_style("##cursor_radius", &m_cursor_radius, CursorRadiusMin, CursorRadiusMax, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + circle_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##cursor_radius_input", &m_cursor_radius, 0.05f, 0.0f, 0.0f, "%.2f");

        ImGui::Separator();
        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(circle_max_width);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + circle_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true); }

    } else if (m_current_tool == ImGui::TriangleButtonIcon) {
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        m_tool_type   = ToolType::BRUSH;

        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(clipping_slider_left);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + clipping_slider_left);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true); }

    } else if (m_current_tool == ImGui::FillButtonIcon) {
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        m_imgui->bbl_checkbox(m_desc["edge_detection"], m_detect_geometry_edge);
        m_tool_type = ToolType::BUCKET_FILL;

        if (m_detect_geometry_edge) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc["smart_fill_angle"]);
            std::string format_str = std::string("%.f") + I18N::translate_utf8("°", "Face angle threshold,"
                                                                                    "placed after the number with no whitespace in between.");
            ImGui::SameLine(sliders_left_width);
            ImGui::PushItemWidth(sliders_width);
            if (m_imgui->bbl_slider_float_style("##smart_fill_angle", &m_smart_fill_angle, SmartFillAngleMin, SmartFillAngleMax, format_str.data(), 1.0f, true))
                for (auto &triangle_selector : m_triangle_selectors) {
                    triangle_selector->seed_fill_unselect_all_triangles();
                    triangle_selector->request_update_render_data();
                }
            ImGui::SameLine(drag_left_width + sliders_left_width);
            ImGui::PushItemWidth(1.5 * slider_icon_width);
            ImGui::BBLDragFloat("##smart_fill_angle_input", &m_smart_fill_angle, 0.05f, 0.0f, 0.0f, "%.2f");
        } else {
            // set to negative value to disable edge detection
            m_smart_fill_angle = -1.f;
        }
        ImGui::Separator();
        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(sliders_left_width);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + sliders_left_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true);}

    } else if (m_current_tool == ImGui::HeightRangeIcon) {
        m_tool_type   = ToolType::BRUSH;
        m_cursor_type = TriangleSelector::CursorType::HEIGHT_RANGE;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc["height_range"] + ":");
        ImGui::SameLine(height_max_width);
        ImGui::PushItemWidth(sliders_width);
        std::string format_str = std::string("%.2f") + I18N::translate_utf8("mm", "Heigh range," "Facet in [cursor z, cursor z + height] will be selected.");
        m_imgui->bbl_slider_float_style("##cursor_height", &m_cursor_height, CursorHeightMin, CursorHeightMax, format_str.data(), 1.0f, true);
        ImGui::SameLine(drag_left_width + height_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##cursor_height_input", &m_cursor_height, 0.05f, 0.0f, 0.0f, "%.2f");

        ImGui::Separator();
        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(height_max_width);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + height_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true); }
    }
    else if (m_current_tool == ImGui::GapFillIcon) {
        m_tool_type = ToolType::GAP_FILL;
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc["gap_area"] + ":");
        ImGui::SameLine(gap_area_slider_left);
        ImGui::PushItemWidth(sliders_width);
        std::string format_str = std::string("%.2f") + I18N::translate_utf8("", "Triangle patch area threshold,""triangle patch will be merged to neighbor if its area is less than threshold");
        m_imgui->bbl_slider_float_style("##gap_area", &TriangleSelectorPatch::gap_area, TriangleSelectorPatch::GapAreaMin, TriangleSelectorPatch::GapAreaMax, format_str.data(), 1.0f, true);
        ImGui::SameLine(drag_left_width + gap_area_slider_left);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##gap_area_input", &TriangleSelectorPatch::gap_area, 0.05f, 0.0f, 0.0f, "%.2f");
    }

    ImGui::Separator();
    if(m_imgui->bbl_checkbox(_L("Vertical"), m_vertical_only)){
        if(m_vertical_only){
            m_horizontal_only = false;
        }
    }
    if(m_imgui->bbl_checkbox(_L("Horizontal"), m_horizontal_only)){
        if(m_horizontal_only){
            m_vertical_only = false;
        }
    }

    ImGui::Separator();

    const bool can_convert_regions_to_vertex_colors = selected_object_has_painted_regions();
    m_imgui->disabled_begin(!can_convert_regions_to_vertex_colors);
    if (m_imgui->button(_L("Convert regions to vertex colors")))
        convert_selected_regions_to_vertex_colors();
    if (ImGui::IsItemHovered()) {
        if (can_convert_regions_to_vertex_colors)
            m_imgui->tooltip(_L("Convert painted color regions into imported vertex color data, clear the regions, and assign a texture mapping zone."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have painted color regions."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    const bool can_bake_image_texture_data = selected_object_has_bakeable_image_texture_data();
    m_imgui->disabled_begin(!can_bake_image_texture_data);
    if (m_imgui->button(_L("Bake image texture to vertex colors")))
        bake_selected_object_image_texture_to_vertex_colors();
    if (ImGui::IsItemHovered()) {
        if (can_bake_image_texture_data)
            m_imgui->tooltip(_L("Sample imported image texture UVs into stored vertex colors, then discard the baked image texture data."),
                             max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have imported image texture data with UVs."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    const bool can_convert_vertex_colors_to_texture_mapping_colors = selected_object_has_imported_vertex_colors();
    m_imgui->disabled_begin(!can_convert_vertex_colors_to_texture_mapping_colors);
    if (m_imgui->button(_L("Convert vertex colors to RGB data")))
        convert_selected_object_vertex_colors_to_texture_mapping_colors();
    if (ImGui::IsItemHovered()) {
        if (can_convert_vertex_colors_to_texture_mapping_colors)
            m_imgui->tooltip(_L("Convert imported vertex colors into RGB data for texture mapping zones."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have stored imported vertex colors."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    const bool can_convert_image_texture_to_texture_mapping_colors = selected_object_has_bakeable_image_texture_data();
    m_imgui->disabled_begin(!can_convert_image_texture_to_texture_mapping_colors);
    if (m_imgui->button(_L("Convert image texture to RGB data")))
        convert_selected_object_image_texture_to_texture_mapping_colors();
    if (ImGui::IsItemHovered()) {
        if (can_convert_image_texture_to_texture_mapping_colors)
            m_imgui->tooltip(_L("Sample imported image texture UVs into RGB data for texture mapping zones."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have imported image texture data with UVs."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    const bool can_clear_image_texture_data = selected_object_has_imported_texture_data();
    m_imgui->disabled_begin(!can_clear_image_texture_data);
    if (m_imgui->button(_L("Clear Image Texture Data")))
        clear_selected_object_image_texture_data();
    if (ImGui::IsItemHovered()) {
        if (can_clear_image_texture_data)
            m_imgui->tooltip(_L("Discard imported image texture data from the selected object."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have imported image texture data."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    const bool can_clear_texture_mapping_color_data = selected_object_has_texture_mapping_color_data();
    m_imgui->disabled_begin(!can_clear_texture_mapping_color_data);
    if (m_imgui->button(_L("Clear RGB Data")))
        clear_selected_object_texture_mapping_color_data();
    if (ImGui::IsItemHovered()) {
        if (can_clear_texture_mapping_color_data)
            m_imgui->tooltip(_L("Discard RGB data from the selected object."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have RGB data."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    ImGui::Separator();

    const bool can_apply_stored_vertex_colors = selected_object_has_imported_vertex_colors();
    m_imgui->disabled_begin(!can_apply_stored_vertex_colors);
    if (m_imgui->button(_L("Convert vertex colors to regions (will erase painting)")))
        open_obj_vertex_color_mapping_dialog();
    if (ImGui::IsItemHovered()) {
        if (can_apply_stored_vertex_colors)
            m_imgui->tooltip(_L("Open OBJ color mapping dialog using stored imported vertex colors."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have stored imported vertex colors."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    ImGui::Separator();


    if (m_imgui->button(m_desc.at("perform_remap"))) {
        m_show_filament_remap_ui = !m_show_filament_remap_ui;
        if (m_show_filament_remap_ui) {
            // reset remap to identity on opening
            m_extruder_remap.resize(m_extruders_colors.size());
            for (size_t i = 0; i < m_extruder_remap.size(); ++i)
                m_extruder_remap[i] = i;
        }
    }

    // Render filament swap UI if enabled
    if (m_show_filament_remap_ui) {
        ImGui::Separator();
        render_filament_remap_ui(window_width, max_tooltip_width);
    }
    ImGui::Separator();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 10.0f));
    float get_cur_y = ImGui::GetContentRegionMax().y + ImGui::GetFrameHeight() + y;
    show_tooltip_information(caption_max, x, get_cur_y);

    float f_scale =m_parent.get_gizmos_manager().get_layout_scale();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 4.0f * f_scale));

    ImGui::SameLine();

    if (m_current_tool == ImGui::GapFillIcon) {
        if (m_imgui->button(m_desc.at("perform"))) {
            Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Gap fill", UndoRedo::SnapshotType::GizmoAction);

            for (int i = 0; i < m_triangle_selectors.size(); i++) {
                TriangleSelectorPatch* ts_mm = dynamic_cast<TriangleSelectorPatch*>(m_triangle_selectors[i].get());
                ts_mm->update_selector_triangles();
                ts_mm->request_update_render_data(true);
            }
            update_model_object();
            m_parent.set_as_dirty();
        }

        ImGui::SameLine();
    }

    if (m_imgui->button(m_desc.at("remove_all"))) {
        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Reset selection", UndoRedo::SnapshotType::GizmoAction);
        ModelObject *        mo  = m_c->selection_info()->model_object();
        int                  idx = -1;
        for (ModelVolume *mv : mo->volumes)
            if (mv->is_model_part()) {
                ++idx;
                m_triangle_selectors[idx]->reset();
                m_triangle_selectors[idx]->request_update_render_data(true);
            }

        update_model_object();
        m_parent.set_as_dirty();
    }
    ImGui::PopStyleVar(2);
    GizmoImguiEnd();

    // BBS
    ImGuiWrapper::pop_toolbar_style();
}


void GLGizmoMmuSegmentation::update_model_object()
{
    bool updated = false;
    ModelObject* mo = m_c->selection_info()->model_object();
    int idx = -1;
    for (ModelVolume* mv : mo->volumes) {
        if (! mv->is_model_part())
            continue;
        ++idx;
        updated |= mv->mmu_segmentation_facets.set(*m_triangle_selectors[idx].get());
    }

    if (updated) {
        const size_t num_physical = static_cast<size_t>(std::max(wxGetApp().filaments_cnt(), 0));
        size_t       num_total    = num_physical;
        if (wxGetApp().preset_bundle != nullptr)
            num_total = wxGetApp().preset_bundle->texture_mapping_zones.total_filaments(num_physical);

        size_t max_used_state = 0;
        for (const ModelVolume *mv : mo->volumes) {
            if (!mv->is_model_part())
                continue;
            const auto &used_states = mv->mmu_segmentation_facets.get_data().used_states;
            for (size_t state_idx = static_cast<size_t>(EnforcerBlockerType::Extruder1); state_idx < used_states.size(); ++state_idx) {
                if (used_states[state_idx])
                    max_used_state = std::max(max_used_state, state_idx);
            }
        }

        if (max_used_state > num_physical) {
            BOOST_LOG_TRIVIAL(warning) << "GLGizmoMmuSegmentation::update_model_object painted virtual extruder state detected"
                                       << " max_used_state=" << max_used_state
                                       << " physical_filaments=" << num_physical
                                       << " total_filaments=" << num_total;
        }

        const ModelObjectPtrs &mos = wxGetApp().model().objects;
        size_t obj_idx = std::find(mos.begin(), mos.end(), mo) - mos.begin();
        wxGetApp().obj_list()->update_info_items(obj_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(obj_idx, 0);
        m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
    }
}

void GLGizmoMmuSegmentation::init_model_triangle_selectors()
{
    const ModelObject *mo = m_c->selection_info()->model_object();
    m_triangle_selectors.clear();
    m_volumes_extruder_idxs.clear();

    // Don't continue when extruders colors are not initialized
    if(m_extruders_colors.empty())
        return;

    // BBS: Don't continue when model object is null
    if (mo == nullptr)
        return;

    for (const ModelVolume *mv : mo->volumes) {
        if (!mv->is_model_part())
            continue;

        int extruder_idx = (mv->extruder_id() > 0) ? mv->extruder_id() - 1 : 0;
        std::vector<ColorRGBA> ebt_colors;
        ebt_colors.push_back(m_extruders_colors[size_t(extruder_idx)]);
        ebt_colors.insert(ebt_colors.end(), m_extruders_colors.begin(), m_extruders_colors.end());

        // This mesh does not account for the possible Z up SLA offset.
        const TriangleMesh* mesh = &mv->mesh();
        m_triangle_selectors.emplace_back(std::make_unique<TriangleSelectorPatch>(*mesh, mv, ebt_colors, 0.2));
        // Reset of TriangleSelector is done inside TriangleSelectorMmGUI's constructor, so we don't need it to perform it again in deserialize().
        EnforcerBlockerType max_ebt = (EnforcerBlockerType)std::min(m_extruders_colors.size(), (size_t)EnforcerBlockerType::ExtruderMax);
        m_triangle_selectors.back()->deserialize(mv->mmu_segmentation_facets.get_data(), false, max_ebt);
        m_triangle_selectors.back()->request_update_render_data();
        m_triangle_selectors.back()->set_wireframe_needed(true);
        m_volumes_extruder_idxs.push_back(mv->extruder_id());
    }
}

void GLGizmoMmuSegmentation::update_triangle_selectors_colors()
{
    for (int i = 0; i < m_triangle_selectors.size(); i++) {
        TriangleSelectorPatch* selector = dynamic_cast<TriangleSelectorPatch*>(m_triangle_selectors[i].get());
        int extruder_idx = m_volumes_extruder_idxs[i];
        int extruder_color_idx = std::max(0, extruder_idx - 1);
        std::vector<ColorRGBA> ebt_colors;
        ebt_colors.push_back(m_extruders_colors[extruder_color_idx]);
        ebt_colors.insert(ebt_colors.end(), m_extruders_colors.begin(), m_extruders_colors.end());
        selector->set_ebt_colors(ebt_colors);
    }
}

void GLGizmoMmuSegmentation::update_from_model_object(bool first_update)
{
    wxBusyCursor wait;

    // Extruder colors need to be reloaded before calling init_model_triangle_selectors to render painted triangles
    // using colors from loaded 3MF and not from printer profile in Slicer.
    const std::vector<ColorRGBA> current_extruder_colors = get_extruders_colors();
    if (int prev_extruders_count = int(m_extruders_colors.size());
        prev_extruders_count != int(current_extruder_colors.size()) || current_extruder_colors != m_extruders_colors)
        this->init_extruders_data(current_extruder_colors);

    this->init_model_triangle_selectors();
}

void GLGizmoMmuSegmentation::tool_changed(wchar_t old_tool, wchar_t new_tool)
{
    if ((old_tool == ImGui::GapFillIcon && new_tool == ImGui::GapFillIcon) ||
        (old_tool != ImGui::GapFillIcon && new_tool != ImGui::GapFillIcon))
        return;

    for (auto& selector_ptr : m_triangle_selectors) {
        TriangleSelectorPatch* tsp = dynamic_cast<TriangleSelectorPatch*>(selector_ptr.get());
        tsp->set_filter_state(new_tool == ImGui::GapFillIcon);
    }
}

bool GLGizmoMmuSegmentation::selected_object_has_imported_vertex_colors() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!volume->imported_vertex_colors_rgba.empty())
            return true;
    }
    return false;
}

bool GLGizmoMmuSegmentation::selected_object_has_imported_texture_data() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (model_volume_has_imported_image_texture_data(volume))
            return true;
    }
    return false;
}

bool GLGizmoMmuSegmentation::selected_object_has_bakeable_image_texture_data() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (model_volume_has_bakeable_image_texture_data(volume))
            return true;
    }
    return false;
}

bool GLGizmoMmuSegmentation::selected_object_has_texture_mapping_color_data() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!volume->texture_mapping_color_facets.empty())
            return true;
    }
    return false;
}

bool GLGizmoMmuSegmentation::selected_object_has_painted_regions() const
{
    for (const auto &selector : m_triangle_selectors) {
        if (selector == nullptr)
            continue;
        const TriangleSelector::TriangleSplittingData data = selector->serialize();
        for (size_t state_idx = static_cast<size_t>(EnforcerBlockerType::Extruder1); state_idx < data.used_states.size(); ++state_idx)
            if (data.used_states[state_idx])
                return true;
    }

    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume != nullptr && volume->is_model_part() && !volume->mmu_segmentation_facets.empty())
            return true;
    }
    return false;
}

void GLGizmoMmuSegmentation::open_obj_vertex_color_mapping_dialog()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    ModelVolume *target_volume = nullptr;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!volume->imported_vertex_colors_rgba.empty()) {
            target_volume = volume;
            break;
        }
    }
    if (target_volume == nullptr)
        return;

    if (target_volume->mesh().its.vertices.size() != target_volume->imported_vertex_colors_rgba.size())
        return;

    std::vector<RGBA> input_colors;
    input_colors.reserve(target_volume->imported_vertex_colors_rgba.size());
    for (const uint32_t packed : target_volume->imported_vertex_colors_rgba) {
        const float r = float((packed >> 24) & 0xFF) / 255.f;
        const float g = float((packed >> 16) & 0xFF) / 255.f;
        const float b = float((packed >> 8) & 0xFF) / 255.f;
        const float a = float(packed & 0xFF) / 255.f;
        input_colors.emplace_back(RGBA{r, g, b, a});
    }

    if (input_colors.empty())
        return;

    bool is_single_color = true;
    const RGBA first_color = input_colors.front();
    for (const RGBA &color : input_colors) {
        if (color != first_color) {
            is_single_color = false;
            break;
        }
    }

    std::vector<unsigned char> filament_ids;
    unsigned char first_extruder_id = 1;
    const std::vector<std::string> extruder_colours = wxGetApp().plater()->get_extruder_colors_from_plater_config();
    ObjColorDialog color_dlg(nullptr, input_colors, is_single_color, extruder_colours, filament_ids, first_extruder_id);
    if (color_dlg.ShowModal() != wxID_OK)
        return;
    if (filament_ids.empty())
        return;

    if (!Model::obj_import_vertex_color_deal_for_object(filament_ids, first_extruder_id, object))
        return;

    update_from_model_object();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::bake_selected_object_image_texture_to_vertex_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool baked = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Bake image texture to vertex colors", UndoRedo::SnapshotType::GizmoAction);

    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_bakeable_image_texture_data(volume))
            continue;

        const indexed_triangle_set &its = volume->mesh().its;

        struct VertexColorAccumulator
        {
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            double a = 0.0;
            double weight = 0.0;
        };

        std::vector<VertexColorAccumulator> accumulators(its.vertices.size());
        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            if (volume->imported_texture_uv_valid[tri_idx] == 0)
                continue;

            const auto &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const size_t uv_offset = tri_idx * 6;
            const std::array<Vec2f, 3> uvs = {
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5])
            };

            const Vec3f p0 = its.vertices[size_t(tri[0])].cast<float>();
            const Vec3f p1 = its.vertices[size_t(tri[1])].cast<float>();
            const Vec3f p2 = its.vertices[size_t(tri[2])].cast<float>();
            const float area = 0.5f * (p1 - p0).cross(p2 - p0).norm();
            const double weight = std::isfinite(area) && area > EPSILON ? double(area) : 1.0;
            const std::array<int, 3> vertex_indices = { tri[0], tri[1], tri[2] };

            for (size_t corner = 0; corner < 3; ++corner) {
                const ColorRGBA color = sample_texture_rgba_for_vertex_bake(volume->imported_texture_rgba,
                                                                            volume->imported_texture_width,
                                                                            volume->imported_texture_height,
                                                                            uvs[corner]);
                VertexColorAccumulator &acc = accumulators[size_t(vertex_indices[corner])];
                acc.r += double(color.r()) * weight;
                acc.g += double(color.g()) * weight;
                acc.b += double(color.b()) * weight;
                acc.a += double(color.a()) * weight;
                acc.weight += weight;
            }
        }

        std::vector<uint32_t> vertex_colors;
        vertex_colors.reserve(its.vertices.size());
        for (size_t vertex_idx = 0; vertex_idx < its.vertices.size(); ++vertex_idx) {
            const VertexColorAccumulator &acc = accumulators[vertex_idx];
            if (acc.weight > 0.0) {
                vertex_colors.emplace_back(pack_vertex_color_rgba(ColorRGBA(float(acc.r / acc.weight),
                                                                            float(acc.g / acc.weight),
                                                                            float(acc.b / acc.weight),
                                                                            float(acc.a / acc.weight))));
            } else if (vertex_idx < volume->imported_vertex_colors_rgba.size()) {
                vertex_colors.emplace_back(volume->imported_vertex_colors_rgba[vertex_idx]);
            } else {
                vertex_colors.emplace_back(pack_vertex_color_rgba(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
            }
        }

        if (vertex_colors.size() != its.vertices.size())
            continue;

        volume->imported_vertex_colors_rgba = std::move(vertex_colors);
        volume->imported_texture_uvs_per_face.clear();
        volume->imported_texture_uv_valid.clear();
        volume->imported_texture_rgba.clear();
        volume->imported_texture_width = 0;
        volume->imported_texture_height = 0;
        baked = true;
    }

    if (!baked)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();
    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::convert_selected_object_vertex_colors_to_texture_mapping_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert vertex colors to RGB data", UndoRedo::SnapshotType::GizmoAction);

    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() ||
            its.indices.empty() ||
            volume->imported_vertex_colors_rgba.size() != its.vertices.size())
            continue;

        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size())
                return 0xFFFFFFFFu;

            const auto &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                return 0xFFFFFFFFu;
            if (size_t(tri[0]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[1]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[2]) >= volume->imported_vertex_colors_rgba.size())
                return 0xFFFFFFFFu;

            const ColorRGBA c0 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[0])]);
            const ColorRGBA c1 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[1])]);
            const ColorRGBA c2 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[2])]);
            return pack_vertex_color_rgba(ColorRGBA(c0.r() * barycentric.x() + c1.r() * barycentric.y() + c2.r() * barycentric.z(),
                                                    c0.g() * barycentric.x() + c1.g() * barycentric.y() + c2.g() * barycentric.z(),
                                                    c0.b() * barycentric.x() + c1.b() * barycentric.y() + c2.b() * barycentric.z(),
                                                    c0.a() * barycentric.x() + c1.a() * barycentric.y() + c2.a() * barycentric.z()));
        };

        const float target_edge = std::max(mesh_max_axis_span(its) / 160.f, 0.25f);
        TextureMappingColorSubdivisionDepths subdivision_depths = [target_edge](size_t, const std::array<Vec3f, 3> &vertices) {
            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_edge, 5);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, 5, 0.025f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::convert_selected_object_image_texture_to_texture_mapping_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert image texture to RGB data", UndoRedo::SnapshotType::GizmoAction);

    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_bakeable_image_texture_data(volume))
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size() ||
                tri_idx >= volume->imported_texture_uv_valid.size() ||
                volume->imported_texture_uv_valid[tri_idx] == 0)
                return 0xFFFFFFFFu;

            const size_t uv_offset = tri_idx * 6;
            if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
                return 0xFFFFFFFFu;

            const Vec2f uv0(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]);
            const Vec2f uv1(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]);
            const Vec2f uv2(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5]);
            const Vec2f uv = uv0 * barycentric.x() + uv1 * barycentric.y() + uv2 * barycentric.z();
            return pack_vertex_color_rgba(sample_texture_rgba_for_vertex_bake(volume->imported_texture_rgba,
                                                                              volume->imported_texture_width,
                                                                              volume->imported_texture_height,
                                                                              uv));
        };

        const int safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 3200000);
        TextureMappingColorSubdivisionDepths subdivision_depths = [volume, safe_max_depth](size_t tri_idx, const std::array<Vec3f, 3> &) {
            const int depth = texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(volume, tri_idx), 8.f, safe_max_depth);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, safe_max_depth, 0.015f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::convert_selected_regions_to_vertex_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr || m_triangle_selectors.empty())
        return;

    std::vector<std::string> color_strings;
    if (wxGetApp().plater() != nullptr)
        color_strings = wxGetApp().plater()->get_extruder_colors_from_plater_config();

    std::vector<ColorRGBA> filament_colors;
    filament_colors.reserve(color_strings.size());
    for (const std::string &color_string : color_strings) {
        unsigned char rgba[4] = { 38, 166, 154, 255 };
        BitmapCache::parse_color4(color_string, rgba);
        filament_colors.emplace_back(float(rgba[0]) / 255.f,
                                     float(rgba[1]) / 255.f,
                                     float(rgba[2]) / 255.f,
                                     float(rgba[3]) / 255.f);
    }
    if (filament_colors.empty())
        filament_colors.emplace_back(0.15f, 0.65f, 0.6f, 1.f);

    auto color_for_filament_id = [&filament_colors](unsigned int filament_id) {
        if (filament_id >= 1 && filament_id <= filament_colors.size())
            return filament_colors[filament_id - 1];
        return filament_colors.front();
    };

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert regions to vertex colors", UndoRedo::SnapshotType::GizmoAction);

    bool converted = false;
    int selector_idx = -1;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        ++selector_idx;
        if (selector_idx < 0 || size_t(selector_idx) >= m_triangle_selectors.size() || m_triangle_selectors[size_t(selector_idx)] == nullptr)
            continue;

        const auto &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        std::vector<std::vector<TriangleSelector::FacetStateTriangle>> triangles_per_type;
        m_triangle_selectors[size_t(selector_idx)]->get_facet_triangles(triangles_per_type);
        if (triangles_per_type.empty())
            continue;

        struct VertexColorAccumulator
        {
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            double a = 0.0;
            double weight = 0.0;
        };

        std::vector<VertexColorAccumulator> accumulators(its.vertices.size());
        bool accumulated_any = false;
        const unsigned int base_filament_id = volume->extruder_id() > 0 ? unsigned(volume->extruder_id()) : 1u;

        for (size_t state_idx = 0; state_idx < triangles_per_type.size(); ++state_idx) {
            const unsigned int filament_id = state_idx == 0 ? base_filament_id : unsigned(state_idx);
            ColorRGBA state_color = color_for_filament_id(filament_id);
            state_color.a(1.f);

            for (const TriangleSelector::FacetStateTriangle &triangle : triangles_per_type[state_idx]) {
                if (triangle.source_triangle < 0)
                    continue;
                const size_t source_triangle = size_t(triangle.source_triangle);
                if (source_triangle >= its.indices.size())
                    continue;

                const auto &source_indices = its.indices[source_triangle];
                if (source_indices[0] < 0 || source_indices[1] < 0 || source_indices[2] < 0)
                    continue;
                if (size_t(source_indices[0]) >= its.vertices.size() ||
                    size_t(source_indices[1]) >= its.vertices.size() ||
                    size_t(source_indices[2]) >= its.vertices.size())
                    continue;

                const Vec3f source_p0 = its.vertices[size_t(source_indices[0])].cast<float>();
                const Vec3f source_p1 = its.vertices[size_t(source_indices[1])].cast<float>();
                const Vec3f source_p2 = its.vertices[size_t(source_indices[2])].cast<float>();
                const Vec3f centroid = (triangle.vertices[0] + triangle.vertices[1] + triangle.vertices[2]) / 3.f;
                Vec3f weights(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
                if (!barycentric_weights_for_region_vertex_colors(centroid, source_p0, source_p1, source_p2, weights))
                    weights = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);

                weights.x() = std::max(0.f, weights.x());
                weights.y() = std::max(0.f, weights.y());
                weights.z() = std::max(0.f, weights.z());
                const float weights_sum = weights.x() + weights.y() + weights.z();
                if (weights_sum > EPSILON)
                    weights /= weights_sum;
                else
                    weights = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);

                const float area = 0.5f * (triangle.vertices[1] - triangle.vertices[0]).cross(triangle.vertices[2] - triangle.vertices[0]).norm();
                const double area_weight = std::max(double(area), 1e-6);
                const std::array<float, 3> bary = { weights.x(), weights.y(), weights.z() };

                for (size_t corner = 0; corner < 3; ++corner) {
                    VertexColorAccumulator &acc = accumulators[size_t(source_indices[corner])];
                    const double weight = area_weight * double(bary[corner]);
                    acc.r += double(state_color.r()) * weight;
                    acc.g += double(state_color.g()) * weight;
                    acc.b += double(state_color.b()) * weight;
                    acc.a += double(state_color.a()) * weight;
                    acc.weight += weight;
                }
                accumulated_any = true;
            }
        }

        if (!accumulated_any)
            continue;

        const ColorRGBA fallback_color = color_for_filament_id(base_filament_id);
        std::vector<uint32_t> vertex_colors;
        vertex_colors.reserve(its.vertices.size());
        for (const VertexColorAccumulator &acc : accumulators) {
            if (acc.weight > 0.0) {
                vertex_colors.emplace_back(pack_vertex_color_rgba(ColorRGBA(float(acc.r / acc.weight),
                                                                            float(acc.g / acc.weight),
                                                                            float(acc.b / acc.weight),
                                                                            float(acc.a / acc.weight))));
            } else {
                vertex_colors.emplace_back(pack_vertex_color_rgba(fallback_color));
            }
        }

        volume->imported_vertex_colors_rgba = std::move(vertex_colors);
        volume->mmu_segmentation_facets.reset();
        m_triangle_selectors[size_t(selector_idx)]->reset();
        m_triangle_selectors[size_t(selector_idx)]->request_update_render_data(true);
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::clear_selected_object_image_texture_data()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool cleared = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Clear image texture data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_imported_image_texture_data(volume))
            continue;

        volume->imported_texture_uvs_per_face.clear();
        volume->imported_texture_uv_valid.clear();
        volume->imported_texture_rgba.clear();
        volume->imported_texture_width = 0;
        volume->imported_texture_height = 0;
        cleared = true;
    }

    if (!cleared)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();
    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::clear_selected_object_texture_mapping_color_data()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool cleared = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Clear RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || volume->texture_mapping_color_facets.empty())
            continue;

        volume->texture_mapping_color_facets.reset();
        cleared = true;
    }

    if (!cleared)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();
    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

PainterGizmoType GLGizmoMmuSegmentation::get_painter_type() const
{
    return PainterGizmoType::MM_SEGMENTATION;
}

// BBS
ColorRGBA GLGizmoMmuSegmentation::get_cursor_hover_color() const
{
    if (m_selected_extruder_idx < m_display_filament_ids.size()) {
        const unsigned int actual_filament_id = m_display_filament_ids[m_selected_extruder_idx];
        if (actual_filament_id >= 1 && actual_filament_id <= m_extruders_colors.size())
            return m_extruders_colors[actual_filament_id - 1];
    }
    return m_extruders_colors.empty() ? ColorRGBA() : m_extruders_colors[0];
}

void GLGizmoMmuSegmentation::on_set_state()
{
    GLGizmoPainterBase::on_set_state();

    if (get_state() == Off) {
        ModelObject* mo = m_c->selection_info()->model_object();
        if (mo) Slic3r::save_object_mesh(*mo);
        m_parent.post_event(SimpleEvent(EVT_GLCANVAS_FORCE_UPDATE));
    }
}

wxString GLGizmoMmuSegmentation::handle_snapshot_action_name(bool shift_down, GLGizmoPainterBase::Button button_down) const
{
    wxString action_name;
    if (shift_down)
        action_name = _L("Remove painted color");
    else {
        action_name        = GUI::format(_L("Painted using: Filament %1%"), m_selected_extruder_idx + 1);
    }
    return action_name;
}

GLGizmoTrueColorPainting::GLGizmoTrueColorPainting(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id)
    : GLGizmoPainterBase(parent, icon_filename, sprite_id)
{
}

bool GLGizmoTrueColorPainting::on_init()
{
    m_cursor_type = TriangleSelector::CursorType::SPHERE;
    m_tool_type = ToolType::BRUSH;
    m_triangle_splitting_enabled = true;
    m_cursor_radius = 1.f;
    sync_active_color_mode_from_rgb(true);
    return true;
}

void GLGizmoTrueColorPainting::on_opening()
{
    update_selected_object_color_state();
    if (m_color_input_mode == ColorInputMode::FilamentColors)
        sync_active_color_mode_from_rgb(true);
}

void GLGizmoTrueColorPainting::on_shutdown()
{
    m_color_picker_active = false;
    m_color_picker_source_cache.clear();
    m_parent.use_slope(false);
    m_parent.toggle_model_objects_visibility(true);
}

PainterGizmoType GLGizmoTrueColorPainting::get_painter_type() const
{
    return PainterGizmoType::TRUE_COLOR;
}

std::string GLGizmoTrueColorPainting::on_get_name() const
{
    return _u8L("True Color Painting");
}

bool GLGizmoTrueColorPainting::on_is_selectable() const
{
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF;
}

bool GLGizmoTrueColorPainting::on_is_activable() const
{
    const Selection& selection = m_parent.get_selection();
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF &&
           !selection.is_empty() &&
           (selection.is_single_full_instance() || selection.is_any_volume());
}

ColorRGBA GLGizmoTrueColorPainting::get_cursor_hover_color() const
{
    if (m_color_picker_active) {
        ColorRGBA color;
        if (sample_color_from_model(m_parent.get_local_mouse_position(), color))
            return ColorRGBA(std::clamp(color.r(), 0.f, 1.f),
                             std::clamp(color.g(), 0.f, 1.f),
                             std::clamp(color.b(), 0.f, 1.f),
                             1.f);
    }
    return ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f);
}

ColorRGBA GLGizmoTrueColorPainting::get_cursor_sphere_left_button_color() const
{
    return ColorRGBA(m_rgb_color[0],
                     m_rgb_color[1],
                     m_rgb_color[2],
                     0.15f + 0.35f * std::clamp(m_opacity, 0.f, 1.f));
}

void GLGizmoTrueColorPainting::render_painter_gizmo()
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;
    if (object->id() != m_selected_color_state_object_id)
        update_selected_object_color_state();

    const Selection& selection = m_parent.get_selection();
    glsafe(::glEnable(GL_BLEND));
    glsafe(::glEnable(GL_DEPTH_TEST));

    render_triangles(selection);
    m_c->object_clipper()->render_cut();
    m_c->instances_hider()->render_cut();
    render_cursor();

    glsafe(::glDisable(GL_BLEND));
}

bool GLGizmoTrueColorPainting::gizmo_event(SLAGizmoEventType action,
                                           const Vec2d& mouse_position,
                                           bool shift_down,
                                           bool alt_down,
                                           bool control_down)
{
    const bool painting_event =
        action == SLAGizmoEventType::LeftDown ||
        action == SLAGizmoEventType::RightDown ||
        action == SLAGizmoEventType::Dragging ||
        action == SLAGizmoEventType::LeftUp ||
        action == SLAGizmoEventType::RightUp ||
        action == SLAGizmoEventType::Moving;
    const ModelObject *object = selected_model_object();
    if (object == nullptr || object->id() != m_selected_color_state_object_id)
        update_selected_object_color_state();

    if (m_color_picker_active) {
        if (action == SLAGizmoEventType::LeftDown) {
            if (pick_color_from_model(mouse_position))
                m_color_picker_active = false;
            m_parent.set_as_dirty();
            return true;
        }
        if (action == SLAGizmoEventType::RightDown) {
            m_color_picker_active = false;
            m_parent.set_as_dirty();
            return true;
        }
        if (painting_event)
            return true;
    }

    return GLGizmoPainterBase::gizmo_event(action, mouse_position, shift_down, alt_down, control_down);
}

void GLGizmoTrueColorPainting::init_model_triangle_selectors()
{
    const ModelObject *object = selected_model_object();
    m_triangle_selectors.clear();
    if (object == nullptr)
        return;

    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 1.f, 1.f, 0.f),
        ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f)
    };
    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        m_triangle_selectors.emplace_back(std::make_unique<TriangleSelectorPatch>(volume->mesh(), volume, colors, 0.2f));
        if (TriangleSelectorPatch *patch = dynamic_cast<TriangleSelectorPatch *>(m_triangle_selectors.back().get())) {
            patch->set_none_state_rendered(false);
            patch->set_texture_preview_needed(true);
            patch->set_texture_preview_opaque(true);
        }
        m_triangle_selectors.back()->set_wireframe_needed(true);
        m_triangle_selectors.back()->request_update_render_data(true);
    }
}

void GLGizmoTrueColorPainting::update_triangle_selectors_color()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 1.f, 1.f, 0.f),
        ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f)
    };
    for (std::unique_ptr<TriangleSelectorGUI> &selector : m_triangle_selectors) {
        TriangleSelectorPatch *patch = dynamic_cast<TriangleSelectorPatch *>(selector.get());
        if (patch == nullptr)
            continue;
        patch->set_ebt_colors(colors);
    }
    m_parent.set_as_dirty();
}

void GLGizmoTrueColorPainting::update_from_model_object(bool first_update)
{
    (void)first_update;
    wxBusyCursor wait;
    update_selected_object_color_state();
    init_model_triangle_selectors();
}

void GLGizmoTrueColorPainting::update_model_object()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    bool updated = false;
    int selector_idx = -1;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        ++selector_idx;
        if (selector_idx < 0 ||
            size_t(selector_idx) >= m_triangle_selectors.size() ||
            m_triangle_selectors[size_t(selector_idx)] == nullptr)
            continue;

        std::vector<std::vector<TriangleSelector::FacetStateTriangle>> triangles_per_type;
        m_triangle_selectors[size_t(selector_idx)]->get_facet_triangles(triangles_per_type);
        const size_t paint_state = size_t(EnforcerBlockerType::ENFORCER);
        if (triangles_per_type.size() <= paint_state || triangles_per_type[paint_state].empty())
            continue;

        if (volume->texture_mapping_color_facets.empty())
            initialize_volume_rgb_data_from_current_surface_color(*volume, ColorRGBA(1.f, 1.f, 1.f, 1.f));

        const ColorRGBA brush_color(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f);
        updated |= apply_rgb_stroke_to_volume(*volume,
                                              triangles_per_type[paint_state],
                                              brush_color,
                                              m_brush_hardness,
                                              m_opacity,
                                              m_cursor_radius);
        m_triangle_selectors[size_t(selector_idx)]->reset();
        m_triangle_selectors[size_t(selector_idx)]->request_update_render_data(true);
    }

    if (!updated)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    refresh_selected_object_after_rgb_change(object);
}

ModelObject *GLGizmoTrueColorPainting::selected_model_object() const
{
    if (m_c == nullptr)
        return nullptr;
    const auto *selection_info = m_c->selection_info();
    return selection_info != nullptr ? selection_info->model_object() : nullptr;
}

void GLGizmoTrueColorPainting::update_selected_object_color_state()
{
    m_selected_has_rgb_data = false;
    m_selected_has_imported_color_data = false;
    m_selected_can_convert_vertex = false;
    m_selected_can_convert_image = false;

    const ModelObject *object = selected_model_object();
    const ObjectID previous_object_id = m_selected_color_state_object_id;
    m_selected_color_state_object_id = object != nullptr ? object->id() : ObjectID();
    if (m_selected_color_state_object_id != previous_object_id)
        m_color_picker_source_cache.clear();
    if (object == nullptr)
        return;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        m_selected_has_rgb_data |= !volume->texture_mapping_color_facets.empty();
        m_selected_can_convert_vertex |= !volume->imported_vertex_colors_rgba.empty();
        m_selected_can_convert_image |= model_volume_has_bakeable_image_texture_data(volume);
    }
    m_selected_has_imported_color_data = m_selected_can_convert_vertex || m_selected_can_convert_image;
}

bool GLGizmoTrueColorPainting::selected_object_has_rgb_data() const
{
    return m_selected_has_rgb_data;
}

bool GLGizmoTrueColorPainting::selected_object_has_imported_color_data() const
{
    return m_selected_has_imported_color_data;
}

void GLGizmoTrueColorPainting::initialize_selected_object_rgb_data()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    bool initialized = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Create blank RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        initialized |= initialize_volume_rgb_data(*volume, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    }

    if (!initialized)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    refresh_selected_object_after_rgb_change(object);
}

void GLGizmoTrueColorPainting::convert_selected_object_vertex_colors_to_rgb_data()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert vertex colors to RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() ||
            its.indices.empty() ||
            volume->imported_vertex_colors_rgba.size() != its.vertices.size())
            continue;

        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size())
                return 0xFFFFFFFFu;

            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                return 0xFFFFFFFFu;
            if (size_t(tri[0]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[1]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[2]) >= volume->imported_vertex_colors_rgba.size())
                return 0xFFFFFFFFu;

            const ColorRGBA c0 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[0])]);
            const ColorRGBA c1 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[1])]);
            const ColorRGBA c2 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[2])]);
            return pack_vertex_color_rgba(ColorRGBA(c0.r() * barycentric.x() + c1.r() * barycentric.y() + c2.r() * barycentric.z(),
                                                    c0.g() * barycentric.x() + c1.g() * barycentric.y() + c2.g() * barycentric.z(),
                                                    c0.b() * barycentric.x() + c1.b() * barycentric.y() + c2.b() * barycentric.z(),
                                                    c0.a() * barycentric.x() + c1.a() * barycentric.y() + c2.a() * barycentric.z()));
        };

        const float target_edge = std::max(mesh_max_axis_span(its) / 160.f, 0.25f);
        TextureMappingColorSubdivisionDepths subdivision_depths = [target_edge](size_t, const std::array<Vec3f, 3> &vertices) {
            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_edge, 5);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, 5, 0.025f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    refresh_selected_object_after_rgb_change(object);
}

void GLGizmoTrueColorPainting::convert_selected_object_image_texture_to_rgb_data()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert image texture to RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_bakeable_image_texture_data(volume))
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size() ||
                tri_idx >= volume->imported_texture_uv_valid.size() ||
                volume->imported_texture_uv_valid[tri_idx] == 0)
                return 0xFFFFFFFFu;

            const size_t uv_offset = tri_idx * 6;
            if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
                return 0xFFFFFFFFu;

            const Vec2f uv0(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]);
            const Vec2f uv1(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]);
            const Vec2f uv2(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5]);
            const Vec2f uv = uv0 * barycentric.x() + uv1 * barycentric.y() + uv2 * barycentric.z();
            return pack_vertex_color_rgba(sample_texture_rgba_for_vertex_bake(volume->imported_texture_rgba,
                                                                              volume->imported_texture_width,
                                                                              volume->imported_texture_height,
                                                                              uv));
        };

        const int safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 3200000);
        TextureMappingColorSubdivisionDepths subdivision_depths = [volume, safe_max_depth](size_t tri_idx, const std::array<Vec3f, 3> &) {
            const int depth = texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(volume, tri_idx), 8.f, safe_max_depth);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, safe_max_depth, 0.015f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    refresh_selected_object_after_rgb_change(object);
}

void GLGizmoTrueColorPainting::refresh_selected_object_after_rgb_change(ModelObject *object)
{
    update_selected_object_color_state();
    init_model_triangle_selectors();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

bool GLGizmoTrueColorPainting::pick_color_from_model(const Vec2d &mouse_position)
{
    ColorRGBA color;
    if (!sample_color_from_model(mouse_position, color))
        return false;

    set_active_color_from_sample(color);
    return true;
}

bool GLGizmoTrueColorPainting::sample_color_from_model(const Vec2d &mouse_position, ColorRGBA &color) const
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;

    int mesh_idx = -1;
    Vec3f hit = Vec3f::Zero();
    size_t tri_idx = 0;
    if (!raycast_to_selected_mesh(mouse_position, mesh_idx, hit, tri_idx) || mesh_idx < 0)
        return false;

    ModelVolume *volume = nullptr;
    int part_idx = -1;
    for (ModelVolume *candidate : object->volumes) {
        if (candidate == nullptr || !candidate->is_model_part())
            continue;
        ++part_idx;
        if (part_idx == mesh_idx) {
            volume = candidate;
            break;
        }
    }
    if (volume == nullptr)
        return false;

    const indexed_triangle_set &its = volume->mesh().its;
    if (tri_idx >= its.indices.size())
        return false;

    const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
    if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
        return false;
    if (size_t(tri[0]) >= its.vertices.size() ||
        size_t(tri[1]) >= its.vertices.size() ||
        size_t(tri[2]) >= its.vertices.size())
        return false;

    Vec3f barycentric = Vec3f::Zero();
    if (!barycentric_weights_for_region_vertex_colors(hit,
                                                      its.vertices[size_t(tri[0])].cast<float>(),
                                                      its.vertices[size_t(tri[1])].cast<float>(),
                                                      its.vertices[size_t(tri[2])].cast<float>(),
                                                      barycentric))
        return false;

    if (!volume->texture_mapping_color_facets.empty()) {
        const ColorPickerVolumeSourceCache &source = cached_volume_color_source(*volume);
        if (std::optional<ColorRGBA> sampled = sample_rgb_color_facets(source.rgb_facets,
                                                                       source.rgb_by_source_triangle,
                                                                       int(tri_idx),
                                                                       hit)) {
            color = *sampled;
        } else {
            color = rgb_metadata_background_color(volume->texture_mapping_color_facets);
        }
        return true;
    }

    const VolumeColorSource source;
    color = sample_volume_color_source(*volume, source, tri_idx, hit, barycentric);
    return true;
}

const GLGizmoTrueColorPainting::ColorPickerVolumeSourceCache &
GLGizmoTrueColorPainting::cached_volume_color_source(const ModelVolume &volume) const
{
    const ObjectID volume_id = volume.id();
    const ObjectBase::Timestamp timestamp = volume.texture_mapping_color_facets.timestamp();
    auto cache_it = std::find_if(m_color_picker_source_cache.begin(),
                                 m_color_picker_source_cache.end(),
                                 [volume_id](const ColorPickerVolumeSourceCache &cache) {
                                     return cache.volume_id == volume_id;
                                 });
    if (cache_it == m_color_picker_source_cache.end()) {
        m_color_picker_source_cache.emplace_back();
        cache_it = m_color_picker_source_cache.end() - 1;
        cache_it->volume_id = volume_id;
    }
    if (cache_it->timestamp != timestamp) {
        cache_it->timestamp = timestamp;
        cache_it->rgb_facets.clear();
        cache_it->rgb_by_source_triangle.clear();
        volume.texture_mapping_color_facets.get_facet_triangles(volume, cache_it->rgb_facets);
        cache_it->rgb_by_source_triangle.reserve(cache_it->rgb_facets.size());
        for (size_t idx = 0; idx < cache_it->rgb_facets.size(); ++idx)
            cache_it->rgb_by_source_triangle[cache_it->rgb_facets[idx].source_triangle].emplace_back(idx);
    }
    return *cache_it;
}

void GLGizmoTrueColorPainting::set_active_color_from_sample(const ColorRGBA &color)
{
    m_rgb_color[0] = std::clamp(color.r(), 0.f, 1.f);
    m_rgb_color[1] = std::clamp(color.g(), 0.f, 1.f);
    m_rgb_color[2] = std::clamp(color.b(), 0.f, 1.f);
    m_rgb_color[3] = 1.f;
    sync_active_color_mode_from_rgb(true);
    update_triangle_selectors_color();
}

void GLGizmoTrueColorPainting::sync_cmy_from_rgb()
{
    m_cmy_color[0] = 1.f - std::clamp(m_rgb_color[0], 0.f, 1.f);
    m_cmy_color[1] = 1.f - std::clamp(m_rgb_color[1], 0.f, 1.f);
    m_cmy_color[2] = 1.f - std::clamp(m_rgb_color[2], 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_rgb_from_cmy()
{
    m_rgb_color[0] = 1.f - std::clamp(m_cmy_color[0], 0.f, 1.f);
    m_rgb_color[1] = 1.f - std::clamp(m_cmy_color[1], 0.f, 1.f);
    m_rgb_color[2] = 1.f - std::clamp(m_cmy_color[2], 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_cmyk_from_rgb()
{
    const float r = std::clamp(m_rgb_color[0], 0.f, 1.f);
    const float g = std::clamp(m_rgb_color[1], 0.f, 1.f);
    const float b = std::clamp(m_rgb_color[2], 0.f, 1.f);
    const float k = 1.f - std::max({ r, g, b });
    m_cmyk_color[3] = std::clamp(k, 0.f, 1.f);
    if (k >= 1.f - EPSILON) {
        m_cmyk_color[0] = 0.f;
        m_cmyk_color[1] = 0.f;
        m_cmyk_color[2] = 0.f;
        return;
    }

    const float denom = 1.f - k;
    m_cmyk_color[0] = std::clamp((1.f - r - k) / denom, 0.f, 1.f);
    m_cmyk_color[1] = std::clamp((1.f - g - k) / denom, 0.f, 1.f);
    m_cmyk_color[2] = std::clamp((1.f - b - k) / denom, 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_rgb_from_cmyk()
{
    const float c = std::clamp(m_cmyk_color[0], 0.f, 1.f);
    const float m = std::clamp(m_cmyk_color[1], 0.f, 1.f);
    const float y = std::clamp(m_cmyk_color[2], 0.f, 1.f);
    const float k = std::clamp(m_cmyk_color[3], 0.f, 1.f);
    m_rgb_color[0] = std::clamp((1.f - c) * (1.f - k), 0.f, 1.f);
    m_rgb_color[1] = std::clamp((1.f - m) * (1.f - k), 0.f, 1.f);
    m_rgb_color[2] = std::clamp((1.f - y) * (1.f - k), 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_cmyw_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(0.f, 1.f, 1.f, 1.f),
        ColorRGBA(1.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_cmyw_color.size() && idx < weights.size(); ++idx)
        m_cmyw_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_cmyw()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(0.f, 1.f, 1.f, 1.f),
        ColorRGBA(1.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights(m_cmyw_color.begin(), m_cmyw_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_rgbk_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_rgbk_color.size() && idx < weights.size(); ++idx)
        m_rgbk_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_rgbk()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f)
    };
    const std::vector<float> weights(m_rgbk_color.begin(), m_rgbk_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(0.f, 0.f, 0.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_rgbw_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_rgbw_color.size() && idx < weights.size(); ++idx)
        m_rgbw_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_rgbw()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights(m_rgbw_color.begin(), m_rgbw_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_bw_from_rgb()
{
    const float r = std::clamp(m_rgb_color[0], 0.f, 1.f);
    const float g = std::clamp(m_rgb_color[1], 0.f, 1.f);
    const float b = std::clamp(m_rgb_color[2], 0.f, 1.f);
    const float luminance = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.f, 1.f);
    m_bw_color[0] = 1.f - luminance;
    m_bw_color[1] = luminance;
}

void GLGizmoTrueColorPainting::sync_rgb_from_bw()
{
    const float black = std::clamp(m_bw_color[0], 0.f, 1.f);
    const float white = std::clamp(m_bw_color[1], 0.f, 1.f);
    const float sum = black + white;
    const float value = sum <= EPSILON ? 1.f : white / sum;
    m_rgb_color[0] = value;
    m_rgb_color[1] = value;
    m_rgb_color[2] = value;
}

void GLGizmoTrueColorPainting::ensure_filament_mix_colors()
{
    std::vector<ColorRGBA> colors = get_extruders_colors();
    const size_t physical_count = size_t(std::max(wxGetApp().filaments_cnt(), 0));
    if (physical_count > 0 && colors.size() > physical_count)
        colors.resize(physical_count);

    bool changed = colors.size() != m_filament_mix_colors.size();
    if (!changed) {
        for (size_t idx = 0; idx < colors.size(); ++idx) {
            if (colors[idx] != m_filament_mix_colors[idx]) {
                changed = true;
                break;
            }
        }
    }

    if (changed) {
        m_filament_mix_colors = std::move(colors);
        m_filament_mix.clear();
    }

    if (m_filament_mix.size() != m_filament_mix_colors.size())
        sync_filament_mix_from_rgb();
}

void GLGizmoTrueColorPainting::sync_filament_mix_from_rgb()
{
    m_filament_mix = closest_color_mix_weights(m_filament_mix_colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
}

void GLGizmoTrueColorPainting::sync_rgb_from_filament_mix()
{
    const ColorRGBA mixed = color_mix_from_weights(m_filament_mix_colors,
                                                  m_filament_mix,
                                                  ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_active_color_mode_from_rgb(bool update_filament_mix)
{
    switch (m_color_input_mode) {
    case ColorInputMode::FilamentColors:
        ensure_filament_mix_colors();
        if (update_filament_mix)
            sync_filament_mix_from_rgb();
        break;
    case ColorInputMode::RGB:
        break;
    case ColorInputMode::CMY:
        sync_cmy_from_rgb();
        break;
    case ColorInputMode::CMYK:
        sync_cmyk_from_rgb();
        break;
    case ColorInputMode::CMYW:
        sync_cmyw_from_rgb();
        break;
    case ColorInputMode::RGBK:
        sync_rgbk_from_rgb();
        break;
    case ColorInputMode::RGBW:
        sync_rgbw_from_rgb();
        break;
    case ColorInputMode::BW:
        sync_bw_from_rgb();
        break;
    }
}

bool GLGizmoTrueColorPainting::render_rgb_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    changed |= ImGui::ColorEdit3("##true_color_rgb_visual", m_rgb_color.data(), flags);
    changed |= ImGui::SliderFloat("Red", &m_rgb_color[0], 0.f, 1.f, "%.2f");
    changed |= ImGui::SliderFloat("Green", &m_rgb_color[1], 0.f, 1.f, "%.2f");
    changed |= ImGui::SliderFloat("Blue", &m_rgb_color[2], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();
    return changed;
}

bool GLGizmoTrueColorPainting::render_cmy_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    if (ImGui::ColorEdit3("##true_color_cmy_visual", m_rgb_color.data(), flags)) {
        sync_cmy_from_rgb();
        changed = true;
    }

    bool cmy_changed = false;
    cmy_changed |= ImGui::SliderFloat("Cyan", &m_cmy_color[0], 0.f, 1.f, "%.2f");
    cmy_changed |= ImGui::SliderFloat("Magenta", &m_cmy_color[1], 0.f, 1.f, "%.2f");
    cmy_changed |= ImGui::SliderFloat("Yellow", &m_cmy_color[2], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (cmy_changed) {
        sync_rgb_from_cmy();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_cmyk_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    if (ImGui::ColorEdit3("##true_color_cmyk_visual", m_rgb_color.data(), flags)) {
        sync_cmyk_from_rgb();
        changed = true;
    }

    bool cmyk_changed = false;
    cmyk_changed |= ImGui::SliderFloat("Cyan", &m_cmyk_color[0], 0.f, 1.f, "%.2f");
    cmyk_changed |= ImGui::SliderFloat("Magenta", &m_cmyk_color[1], 0.f, 1.f, "%.2f");
    cmyk_changed |= ImGui::SliderFloat("Yellow", &m_cmyk_color[2], 0.f, 1.f, "%.2f");
    cmyk_changed |= ImGui::SliderFloat("Key", &m_cmyk_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (cmyk_changed) {
        sync_rgb_from_cmyk();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_cmyw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    if (ImGui::ColorEdit3("##true_color_cmyw_visual", m_rgb_color.data(), flags)) {
        sync_cmyw_from_rgb();
        changed = true;
    }

    bool cmyw_changed = false;
    cmyw_changed |= ImGui::SliderFloat("Cyan", &m_cmyw_color[0], 0.f, 1.f, "%.2f");
    cmyw_changed |= ImGui::SliderFloat("Magenta", &m_cmyw_color[1], 0.f, 1.f, "%.2f");
    cmyw_changed |= ImGui::SliderFloat("Yellow", &m_cmyw_color[2], 0.f, 1.f, "%.2f");
    cmyw_changed |= ImGui::SliderFloat("White", &m_cmyw_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (cmyw_changed) {
        sync_rgb_from_cmyw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_rgbk_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    if (ImGui::ColorEdit3("##true_color_rgbk_visual", m_rgb_color.data(), flags)) {
        sync_rgbk_from_rgb();
        changed = true;
    }

    bool rgbk_changed = false;
    rgbk_changed |= ImGui::SliderFloat("Red", &m_rgbk_color[0], 0.f, 1.f, "%.2f");
    rgbk_changed |= ImGui::SliderFloat("Green", &m_rgbk_color[1], 0.f, 1.f, "%.2f");
    rgbk_changed |= ImGui::SliderFloat("Blue", &m_rgbk_color[2], 0.f, 1.f, "%.2f");
    rgbk_changed |= ImGui::SliderFloat("Black", &m_rgbk_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (rgbk_changed) {
        sync_rgb_from_rgbk();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_rgbw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    if (ImGui::ColorEdit3("##true_color_rgbw_visual", m_rgb_color.data(), flags)) {
        sync_rgbw_from_rgb();
        changed = true;
    }

    bool rgbw_changed = false;
    rgbw_changed |= ImGui::SliderFloat("Red", &m_rgbw_color[0], 0.f, 1.f, "%.2f");
    rgbw_changed |= ImGui::SliderFloat("Green", &m_rgbw_color[1], 0.f, 1.f, "%.2f");
    rgbw_changed |= ImGui::SliderFloat("Blue", &m_rgbw_color[2], 0.f, 1.f, "%.2f");
    rgbw_changed |= ImGui::SliderFloat("White", &m_rgbw_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (rgbw_changed) {
        sync_rgb_from_rgbw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_bw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    if (ImGui::ColorEdit3("##true_color_bw_visual", m_rgb_color.data(), flags)) {
        sync_bw_from_rgb();
        changed = true;
    }

    float value = std::clamp(m_bw_color[1], 0.f, 1.f);
    const bool slider_changed = ImGui::SliderFloat("Black / White", &value, 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (slider_changed) {
        m_bw_color[0] = 1.f - value;
        m_bw_color[1] = value;
        sync_rgb_from_bw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_filament_colors_picker(float item_width)
{
    ensure_filament_mix_colors();
    if (m_filament_mix_colors.empty()) {
        m_imgui->text(_L("No real filaments are available."));
        return false;
    }

    bool changed = false;
    ImGui::PushItemWidth(item_width);
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    if (ImGui::ColorEdit3("##true_color_filament_visual", m_rgb_color.data(), flags))
        changed = true;
    if (ImGui::IsItemDeactivatedAfterEdit())
        sync_filament_mix_from_rgb();

    bool mix_changed = false;
    for (size_t idx = 0; idx < m_filament_mix.size(); ++idx) {
        const std::string label = GUI::format(_u8L("Filament %1%"), idx + 1);
        mix_changed |= ImGui::SliderFloat(label.c_str(), &m_filament_mix[idx], 0.f, 1.f, "%.2f");
    }
    ImGui::PopItemWidth();

    if (mix_changed) {
        sync_rgb_from_filament_mix();
        changed = true;
    }
    return changed;
}

void GLGizmoTrueColorPainting::on_render_input_window(float x, float y, float bottom_limit)
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;
    if (object->id() != m_selected_color_state_object_id)
        update_selected_object_color_state();

    const float approx_height = m_imgui->scaled(22.0f);
    y = std::min(y, bottom_limit - approx_height);
    GizmoImguiSetNextWIndowPos(x, y, ImGuiCond_Always);

    ImGuiWrapper::push_toolbar_style(m_parent.get_scale());
    GizmoImguiBegin(get_name(), ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                                    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    const float slider_width = m_imgui->scaled(8.f);
    const float max_tooltip_width = ImGui::GetFontSize() * 20.f;
    const char *mode_labels[] = {
        "Filament colors",
        "RGB",
        "CMY",
        "CMYK",
        "CMYW",
        "RGBK",
        "RGBW",
        "BW"
    };
    const int mode_count = int(sizeof(mode_labels) / sizeof(mode_labels[0]));
    int mode = std::clamp(int(m_color_input_mode), 0, mode_count - 1);
    if (ImGui::BeginCombo("##true_color_mode", mode_labels[mode])) {
        for (int idx = 0; idx < mode_count; ++idx) {
            const bool selected = idx == mode;
            if (ImGui::Selectable(mode_labels[idx], selected)) {
                mode = idx;
                m_color_input_mode = ColorInputMode(mode);
                sync_active_color_mode_from_rgb(true);
                update_triangle_selectors_color();
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    const wxString picker_label = m_color_picker_active ? _L("Cancel color picker") : _L("Pick color from model");
    if (m_imgui->button(picker_label)) {
        m_color_picker_active = !m_color_picker_active;
        m_parent.set_as_dirty();
    }

    bool color_changed = false;
    switch (m_color_input_mode) {
    case ColorInputMode::FilamentColors:
        color_changed = render_filament_colors_picker(slider_width);
        break;
    case ColorInputMode::RGB:
        color_changed = render_rgb_picker(slider_width);
        break;
    case ColorInputMode::CMY:
        color_changed = render_cmy_picker(slider_width);
        break;
    case ColorInputMode::CMYK:
        color_changed = render_cmyk_picker(slider_width);
        break;
    case ColorInputMode::CMYW:
        color_changed = render_cmyw_picker(slider_width);
        break;
    case ColorInputMode::RGBK:
        color_changed = render_rgbk_picker(slider_width);
        break;
    case ColorInputMode::RGBW:
        color_changed = render_rgbw_picker(slider_width);
        break;
    case ColorInputMode::BW:
        color_changed = render_bw_picker(slider_width);
        break;
    }
    if (color_changed)
        update_triangle_selectors_color();

    ImGui::Separator();
    m_imgui->text(_L("Pen size"));
    ImGui::PushItemWidth(slider_width);
    m_imgui->bbl_slider_float_style("##true_color_cursor_radius", &m_cursor_radius, CursorRadiusMin, CursorRadiusMax, "%.2f", 1.f, true);
    ImGui::PopItemWidth();

    float softness_pct = (1.f - m_brush_hardness) * 100.f;
    m_imgui->text(_L("Brush softness"));
    ImGui::PushItemWidth(slider_width);
    if (m_imgui->bbl_slider_float_style("##true_color_softness", &softness_pct, 0.f, 100.f, "%.0f%%", 1.f, true))
        m_brush_hardness = 1.f - std::clamp(softness_pct / 100.f, 0.f, 1.f);
    ImGui::PopItemWidth();

    float opacity_pct = m_opacity * 100.f;
    m_imgui->text(_L("Opacity"));
    ImGui::PushItemWidth(slider_width);
    if (m_imgui->bbl_slider_float_style("##true_color_opacity", &opacity_pct, 0.f, 100.f, "%.0f%%", 1.f, true)) {
        m_opacity = std::clamp(opacity_pct / 100.f, 0.f, 1.f);
        m_parent.set_as_dirty();
    }
    ImGui::PopItemWidth();

    ImGui::Separator();
    if (m_c->object_clipper()->get_position() == 0.f) {
        m_imgui->text(_L("Section view"));
    } else if (m_imgui->button(_L("Reset direction"))) {
        wxGetApp().CallAfter([this]() {
            m_c->object_clipper()->set_position_by_ratio(-1., false);
        });
    }

    float clp_dist = float(m_c->object_clipper()->get_position());
    ImGui::PushItemWidth(slider_width);
    if (m_imgui->bbl_slider_float_style("##true_color_clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.f, true))
        m_c->object_clipper()->set_position_by_ratio(clp_dist, true);
    ImGui::PopItemWidth();

    if (ImGui::IsItemHovered())
        m_imgui->tooltip(_L("Section view"), max_tooltip_width);

    GizmoImguiEnd();
    ImGuiWrapper::pop_toolbar_style();
}

wxString GLGizmoTrueColorPainting::handle_snapshot_action_name(bool shift_down, GLGizmoPainterBase::Button button_down) const
{
    (void)shift_down;
    (void)button_down;
    return _L("Paint RGB color");
}

GLGizmoImageProjection::GLGizmoImageProjection(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id)
    : GLGizmoBase(parent, icon_filename, sprite_id)
{
}

bool GLGizmoImageProjection::on_init()
{
    return true;
}

void GLGizmoImageProjection::on_render()
{
}

std::string GLGizmoImageProjection::on_get_name() const
{
    return _u8L("Project image to model surface");
}

void GLGizmoImageProjection::on_set_state()
{
    if (get_state() == On) {
        m_parent.enable_picking(false);
        m_projection_mode_initialized = false;
    } else if (get_state() == Off) {
        m_parent.enable_picking(true);
        m_parent.toggle_model_objects_visibility(true);
    }
}

bool GLGizmoImageProjection::on_is_selectable() const
{
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF;
}

bool GLGizmoImageProjection::on_is_activable() const
{
    const Selection& selection = m_parent.get_selection();
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF &&
           !selection.is_empty() &&
           (selection.is_single_full_instance() || selection.is_any_volume());
}

CommonGizmosDataID GLGizmoImageProjection::on_get_requirements() const
{
    return CommonGizmosDataID(int(CommonGizmosDataID::SelectionInfo) | int(CommonGizmosDataID::InstancesHider));
}

bool GLGizmoImageProjection::load_projection_image()
{
    m_image_error.clear();
    wxFileDialog dialog(wxGetApp().mainframe,
                        _L("Load projection image"),
                        "",
                        "",
                        _L("Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp|All files (*.*)|*.*"),
                        wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    if (dialog.ShowModal() != wxID_OK)
        return false;

    wxImage image(dialog.GetPath(), wxBITMAP_TYPE_ANY);
    std::vector<uint8_t> rgba;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!wx_image_to_rgba(image, rgba, width, height)) {
        m_image_error = _u8L("Unable to load the selected image.");
        return false;
    }

    m_image_path = into_u8(dialog.GetPath());
    m_image_rgba = std::move(rgba);
    m_image_width = width;
    m_image_height = height;
    m_overlay_texture_dirty = true;
    m_parent.set_as_dirty();
    return true;
}

void GLGizmoImageProjection::clear_projection_image()
{
    m_image_path.clear();
    m_image_error.clear();
    m_image_rgba.clear();
    m_image_width = 0;
    m_image_height = 0;
    m_overlay_texture.reset();
    m_overlay_texture_dirty = false;
    m_parent.set_as_dirty();
}

bool GLGizmoImageProjection::ensure_overlay_texture()
{
    if (m_overlay_texture.get_id() != 0 && !m_overlay_texture_dirty)
        return true;
    if (m_image_rgba.empty() || m_image_width == 0 || m_image_height == 0)
        return false;

    std::vector<unsigned char> raw(m_image_rgba.begin(), m_image_rgba.end());
    if (!m_overlay_texture.load_from_raw_data(std::move(raw), m_image_width, m_image_height)) {
        m_image_error = _u8L("Unable to display the selected image.");
        return false;
    }
    m_overlay_texture_dirty = false;
    return true;
}

GLGizmoImageProjection::OverlayRect GLGizmoImageProjection::overlay_rect() const
{
    OverlayRect rect;
    if (m_image_width == 0 || m_image_height == 0)
        return rect;

    const Size canvas_size = m_parent.get_canvas_size();
    const float canvas_w = float(std::max(1, canvas_size.get_width()));
    const float canvas_h = float(std::max(1, canvas_size.get_height()));
    const float max_w = canvas_w * 0.48f;
    const float max_h = canvas_h * 0.48f;
    float scale = std::min(max_w / float(m_image_width), max_h / float(m_image_height));
    if (!std::isfinite(scale) || scale <= 0.f)
        scale = 1.f;

    rect.width = float(m_image_width) * scale;
    rect.height = float(m_image_height) * scale;
    rect.left = (canvas_w - rect.width) * 0.5f;
    rect.top = (canvas_h - rect.height) * 0.5f;
    return rect;
}

ModelObject *GLGizmoImageProjection::selected_model_object() const
{
    if (m_c == nullptr)
        return nullptr;
    const auto *selection_info = m_c->selection_info();
    return selection_info != nullptr ? selection_info->model_object() : nullptr;
}

void GLGizmoImageProjection::update_default_projection_mode()
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    if (m_projection_mode_initialized &&
        object->id() == m_projection_mode_object_id &&
        projection_mode_allowed(m_projection_mode))
        return;

    m_projection_mode = default_projection_mode();
    m_projection_mode_initialized = true;
    m_projection_mode_object_id = object->id();
}

GLGizmoImageProjection::ProjectionMode GLGizmoImageProjection::default_projection_mode() const
{
    if (selected_object_has_rgb_data())
        return ProjectionMode::RGBData;
    if (selected_object_has_image_texture_data())
        return ProjectionMode::ImageTexture;
    return ProjectionMode::RGBData;
}

bool GLGizmoImageProjection::projection_mode_allowed(ProjectionMode mode) const
{
    if (selected_object_has_rgb_data())
        return mode == ProjectionMode::RGBData;
    if (selected_object_has_image_texture_data())
        return mode == ProjectionMode::ImageTexture || mode == ProjectionMode::RGBData;
    return true;
}

bool GLGizmoImageProjection::selected_object_has_image_texture_data() const
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && model_volume_has_bakeable_image_texture_data(volume))
            return true;
    return false;
}

bool GLGizmoImageProjection::selected_object_has_vertex_color_data() const
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->imported_vertex_colors_rgba.empty())
            return true;
    return false;
}

bool GLGizmoImageProjection::selected_object_has_rgb_data() const
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->texture_mapping_color_facets.empty())
            return true;
    return false;
}

void GLGizmoImageProjection::on_render_input_window(float x, float y, float bottom_limit)
{
    update_default_projection_mode();

    if (ensure_overlay_texture()) {
        const OverlayRect rect = overlay_rect();
        if (rect.width > 0.f && rect.height > 0.f) {
            ImGui::SetNextWindowPos(ImVec2(rect.left, rect.top), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(rect.width, rect.height), ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
            ImGui::Begin("##image_projection_overlay",
                         nullptr,
                         ImGuiWindowFlags_NoDecoration |
                         ImGuiWindowFlags_NoInputs |
                         ImGuiWindowFlags_NoBackground |
                         ImGuiWindowFlags_NoSavedSettings);
            ImGui::Image((void *)(intptr_t)m_overlay_texture.get_id(),
                         ImVec2(rect.width, rect.height),
                         ImVec2(0.f, 0.f),
                         ImVec2(1.f, 1.f),
                         ImVec4(1.f, 1.f, 1.f, 0.72f * std::clamp(m_projection_opacity, 0.f, 1.f)));
            ImGui::End();
            ImGui::PopStyleVar(2);
        }
    }

    const float approx_height = m_imgui->scaled(12.0f);
    y = std::min(y, bottom_limit - approx_height);
    GizmoImguiSetNextWIndowPos(x, y, ImGuiCond_Always);

    ImGuiWrapper::push_toolbar_style(m_parent.get_scale());
    GizmoImguiBegin(get_name(), ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    if (m_imgui->button(_L("Load image")))
        load_projection_image();

    ImGui::SameLine();
    m_imgui->disabled_begin(m_image_rgba.empty());
    if (m_imgui->button(_L("Clear image")))
        clear_projection_image();
    m_imgui->disabled_end();

    if (!m_image_path.empty()) {
        const size_t slash = m_image_path.find_last_of("/\\");
        ImGui::SameLine();
        m_imgui->text(from_u8(slash == std::string::npos ? m_image_path : m_image_path.substr(slash + 1)));
    }

    m_imgui->text(_L("Apply to:"));
    ImGui::SameLine();
    const char *mode_labels[] = { "Vertex colors", "Image Texture", "RGB data" };
    int mode = std::clamp(int(m_projection_mode), 0, 2);
    if (ImGui::BeginCombo("##projection_mode", mode_labels[mode])) {
        for (int idx = 0; idx < 3; ++idx) {
            const ProjectionMode candidate = ProjectionMode(idx);
            if (!projection_mode_allowed(candidate))
                continue;
            const bool selected = m_projection_mode == candidate;
            if (ImGui::Selectable(mode_labels[idx], selected)) {
                mode = idx;
                m_projection_mode = candidate;
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    float opacity_pct = m_projection_opacity * 100.f;
    m_imgui->text(_L("Opacity"));
    ImGui::PushItemWidth(m_imgui->scaled(8.f));
    if (m_imgui->bbl_slider_float_style("##image_projection_opacity", &opacity_pct, 0.f, 100.f, "%.0f%%", 1.f, true)) {
        m_projection_opacity = std::clamp(opacity_pct / 100.f, 0.f, 1.f);
        m_parent.set_as_dirty();
    }
    ImGui::PopItemWidth();

    ImGui::Checkbox("Apply transparent regions as background color", &m_apply_transparency_as_background);
    ImGui::Checkbox("Pass through model", &m_pass_through_model);

    m_imgui->disabled_begin(m_image_rgba.empty());
    if (m_imgui->button(_L("Project image onto model")))
        project_image_to_selected_object();
    m_imgui->disabled_end();

    if (!m_image_error.empty())
        m_imgui->warning_text(from_u8(m_image_error));

    GizmoImguiEnd();
    ImGuiWrapper::pop_toolbar_style();
}

bool GLGizmoImageProjection::project_image_to_selected_object()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr || m_image_rgba.empty())
        return false;

    update_default_projection_mode();
    if (!projection_mode_allowed(m_projection_mode))
        return false;

    bool changed = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Project image onto model", UndoRedo::SnapshotType::GizmoAction);
    switch (m_projection_mode) {
    case ProjectionMode::VertexColors:
        changed = project_to_vertex_colors(object);
        break;
    case ProjectionMode::ImageTexture:
        changed = project_to_image_texture(object);
        break;
    case ProjectionMode::RGBData:
        changed = project_to_rgb_data(object);
        break;
    }

    if (!changed)
        return false;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    refresh_projected_object(object);
    m_projection_mode_initialized = true;
    m_projection_mode_object_id = object->id();
    return true;
}

bool GLGizmoImageProjection::project_to_vertex_colors(ModelObject *object)
{
    const Selection &selection = m_parent.get_selection();
    const int instance_idx = selection.get_instance_idx();
    const Camera &camera = wxGetApp().plater()->get_camera();
    const std::array<int, 4> &viewport = camera.get_viewport();
    const OverlayRect rect = overlay_rect();

    ProjectionContext context;
    context.view_projection = (camera.get_projection_matrix() * camera.get_view_matrix()).matrix();
    context.canvas_width = std::max(1, viewport[2]);
    context.canvas_height = std::max(1, viewport[3]);
    context.overlay_left = rect.left;
    context.overlay_top = rect.top;
    context.overlay_width = rect.width;
    context.overlay_height = rect.height;
    context.image_rgba = &m_image_rgba;
    context.image_width = m_image_width;
    context.image_height = m_image_height;
    context.image_opacity = m_projection_opacity;
    context.apply_transparency_as_background = m_apply_transparency_as_background;

    const ProjectionVisibility visibility = m_pass_through_model ?
        ProjectionVisibility() :
        build_projection_visibility(context, m_parent, object, instance_idx);

    bool changed = false;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const VolumeColorSource source = build_volume_color_source(*volume);
        const ColorRGBA fallback_color = projection_base_color_for_volume(*volume);
        std::vector<std::array<float, 4>> base_accum(its.vertices.size(), { 0.f, 0.f, 0.f, 0.f });
        std::vector<unsigned int> base_counts(its.vertices.size(), 0);

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            for (int corner = 0; corner < 3; ++corner) {
                if (tri[corner] < 0 || size_t(tri[corner]) >= its.vertices.size())
                    continue;
                Vec3f barycentric = Vec3f::Zero();
                barycentric[corner] = 1.f;
                const ColorRGBA color = sample_volume_color_source(*volume,
                                                                   source,
                                                                   tri_idx,
                                                                   its.vertices[size_t(tri[corner])].cast<float>(),
                                                                   barycentric,
                                                                   true,
                                                                   &fallback_color);
                std::array<float, 4> &accum = base_accum[size_t(tri[corner])];
                accum[0] += color.r();
                accum[1] += color.g();
                accum[2] += color.b();
                accum[3] += color.a();
                ++base_counts[size_t(tri[corner])];
            }
        }

        std::vector<ColorRGBA> base_colors(its.vertices.size(), ColorRGBA(1.f, 1.f, 1.f, 1.f));
        for (size_t idx = 0; idx < base_colors.size(); ++idx) {
            if (base_counts[idx] == 0)
                continue;
            const float inv = 1.f / float(base_counts[idx]);
            base_colors[idx] = ColorRGBA(base_accum[idx][0] * inv,
                                         base_accum[idx][1] * inv,
                                         base_accum[idx][2] * inv,
                                         base_accum[idx][3] * inv);
        }

        std::vector<std::array<float, 4>> projected_accum(its.vertices.size(), { 0.f, 0.f, 0.f, 0.f });
        std::vector<unsigned int> projected_counts(its.vertices.size(), 0);
        const Transform3d world_matrix = projection_world_matrix_for_volume(m_parent, object, volume, instance_idx);

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            for (int corner = 0; corner < 3; ++corner) {
                const size_t vertex_idx = size_t(tri[corner]);
                if (!m_pass_through_model && !projection_point_is_visible(visibility, context, world_matrix, vertices[size_t(corner)]))
                    continue;
                if (std::optional<ColorRGBA> projected = projected_image_color_at_point(context, world_matrix, vertices[size_t(corner)])) {
                    if (!context.apply_transparency_as_background && !projection_overlay_has_paintable_alpha(*projected, context))
                        continue;
                    const ColorRGBA color = apply_projection_color(base_colors[vertex_idx], *projected, context, false);
                    std::array<float, 4> &accum = projected_accum[vertex_idx];
                    accum[0] += color.r();
                    accum[1] += color.g();
                    accum[2] += color.b();
                    accum[3] += color.a();
                    ++projected_counts[vertex_idx];
                }
            }
        }

        volume->imported_vertex_colors_rgba.assign(its.vertices.size(), 0xFFFFFFFFu);
        for (size_t idx = 0; idx < its.vertices.size(); ++idx) {
            ColorRGBA color = base_colors[idx];
            if (projected_counts[idx] > 0) {
                const float inv = 1.f / float(projected_counts[idx]);
                color = ColorRGBA(projected_accum[idx][0] * inv,
                                  projected_accum[idx][1] * inv,
                                  projected_accum[idx][2] * inv,
                                  projected_accum[idx][3] * inv);
            }
            volume->imported_vertex_colors_rgba[idx] = pack_vertex_color_rgba(color);
        }
        changed = true;
    }
    return changed;
}

bool GLGizmoImageProjection::project_to_image_texture(ModelObject *object)
{
    const Selection &selection = m_parent.get_selection();
    const int instance_idx = selection.get_instance_idx();
    const Camera &camera = wxGetApp().plater()->get_camera();
    const std::array<int, 4> &viewport = camera.get_viewport();
    const OverlayRect rect = overlay_rect();

    ProjectionContext context;
    context.view_projection = (camera.get_projection_matrix() * camera.get_view_matrix()).matrix();
    context.canvas_width = std::max(1, viewport[2]);
    context.canvas_height = std::max(1, viewport[3]);
    context.overlay_left = rect.left;
    context.overlay_top = rect.top;
    context.overlay_width = rect.width;
    context.overlay_height = rect.height;
    context.image_rgba = &m_image_rgba;
    context.image_width = m_image_width;
    context.image_height = m_image_height;
    context.image_opacity = m_projection_opacity;
    context.apply_transparency_as_background = m_apply_transparency_as_background;

    const ProjectionVisibility visibility = m_pass_through_model ?
        ProjectionVisibility() :
        build_projection_visibility(context, m_parent, object, instance_idx);

    bool changed = false;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const bool generated_texture = !model_volume_has_bakeable_image_texture_data(volume);
        if (generated_texture) {
            const uint32_t texture_size = projection_texture_size_for_triangles(its.indices.size());
            const uint32_t grid = uint32_t(std::ceil(std::sqrt(double(std::max<size_t>(its.indices.size(), 1)))));
            const float tile = float(texture_size) / float(std::max<uint32_t>(grid, 1));
            volume->imported_texture_width = texture_size;
            volume->imported_texture_height = texture_size;
            volume->imported_texture_rgba.assign(size_t(texture_size) * size_t(texture_size) * 4, 255);
            volume->imported_texture_uv_valid.assign(its.indices.size(), 1);
            volume->imported_texture_uvs_per_face.assign(its.indices.size() * 6, 0.f);

            for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
                const uint32_t cell_x = uint32_t(tri_idx % grid);
                const uint32_t cell_y = uint32_t(tri_idx / grid);
                const float left = float(cell_x) * tile + 0.5f;
                const float top = float(cell_y) * tile + 0.5f;
                const float right = std::min(float(texture_size) - 0.5f, float(cell_x + 1) * tile - 0.5f);
                const float bottom = std::min(float(texture_size) - 0.5f, float(cell_y + 1) * tile - 0.5f);
                const size_t uv = tri_idx * 6;
                volume->imported_texture_uvs_per_face[uv + 0] = left / float(texture_size);
                volume->imported_texture_uvs_per_face[uv + 1] = top / float(texture_size);
                volume->imported_texture_uvs_per_face[uv + 2] = right / float(texture_size);
                volume->imported_texture_uvs_per_face[uv + 3] = top / float(texture_size);
                volume->imported_texture_uvs_per_face[uv + 4] = left / float(texture_size);
                volume->imported_texture_uvs_per_face[uv + 5] = bottom / float(texture_size);
            }
        }

        const VolumeColorSource source = build_volume_color_source(*volume);
        const ColorRGBA fallback_color = projection_base_color_for_volume(*volume);
        const Transform3d world_matrix = projection_world_matrix_for_volume(m_parent, object, volume, instance_idx);
        const bool rewrite_texture_base = generated_texture || !volume->texture_mapping_color_facets.empty();
        const std::vector<uint8_t> source_texture_rgba(volume->imported_texture_rgba.begin(), volume->imported_texture_rgba.end());
        bool volume_changed = generated_texture;

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;
            if (tri_idx >= volume->imported_texture_uv_valid.size() ||
                volume->imported_texture_uv_valid[tri_idx] == 0)
                continue;

            const size_t uv_offset = tri_idx * 6;
            if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
                continue;

            const std::array<Vec2f, 3> uvs = unwrap_projection_uvs(std::array<Vec2f, 3>{
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5])
            });
            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            const float texture_width = float(volume->imported_texture_width);
            const float texture_height = float(volume->imported_texture_height);
            if (texture_width <= 0.f || texture_height <= 0.f)
                continue;
            const std::array<Vec2f, 3> pixel_uvs = {
                Vec2f(uvs[0].x() * texture_width, uvs[0].y() * texture_height),
                Vec2f(uvs[1].x() * texture_width, uvs[1].y() * texture_height),
                Vec2f(uvs[2].x() * texture_width, uvs[2].y() * texture_height)
            };
            const float min_u = std::min({ uvs[0].x(), uvs[1].x(), uvs[2].x() });
            const float max_u = std::max({ uvs[0].x(), uvs[1].x(), uvs[2].x() });
            const float min_v = std::min({ uvs[0].y(), uvs[1].y(), uvs[2].y() });
            const float max_v = std::max({ uvs[0].y(), uvs[1].y(), uvs[2].y() });
            int min_x = int(std::floor(min_u * texture_width)) - 1;
            int max_x = int(std::ceil(max_u * texture_width)) + 1;
            int min_y = int(std::floor(min_v * texture_height)) - 1;
            int max_y = int(std::ceil(max_v * texture_height)) + 1;
            const bool uv_raster_too_large =
                max_x - min_x > int(volume->imported_texture_width) * 2 ||
                max_y - min_y > int(volume->imported_texture_height) * 2;

            if (!uv_raster_too_large) {
                for (int y_px = min_y; y_px <= max_y; ++y_px) {
                    for (int x_px = min_x; x_px <= max_x; ++x_px) {
                        const Vec2f pixel(float(x_px) + 0.5f, float(y_px) + 0.5f);
                        Vec3f barycentric = Vec3f::Zero();
                        if (!conservative_barycentric_weights_2d(pixel, pixel_uvs[0], pixel_uvs[1], pixel_uvs[2], 0.7072f, barycentric))
                            continue;

                        const Vec3f point = vertices[0] * barycentric.x() +
                                            vertices[1] * barycentric.y() +
                                            vertices[2] * barycentric.z();
                        ColorRGBA color = rewrite_texture_base ?
                            sample_volume_color_source(*volume, source, tri_idx, point, barycentric, false, &fallback_color) :
                            read_rgba_pixel(source_texture_rgba,
                                            volume->imported_texture_width,
                                            wrapped_texture_pixel(x_px, volume->imported_texture_width),
                                            wrapped_texture_pixel(y_px, volume->imported_texture_height));
                        if (m_pass_through_model || projection_point_is_visible(visibility, context, world_matrix, point)) {
                            if (std::optional<ColorRGBA> projected = projected_image_color_at_point(context, world_matrix, point)) {
                                const bool transparent_sample =
                                    !context.apply_transparency_as_background &&
                                    !projection_overlay_has_paintable_alpha(*projected, context);
                                if (transparent_sample) {
                                    if (!rewrite_texture_base) {
                                        continue;
                                    }
                                } else {
                                    color = apply_projection_color(color, *projected, context, true);
                                }
                            } else if (!rewrite_texture_base) {
                                continue;
                            }
                        }
                        volume_changed |= write_rgba_pixel(volume->imported_texture_rgba,
                                                           volume->imported_texture_width,
                                                           wrapped_texture_pixel(x_px, volume->imported_texture_width),
                                                           wrapped_texture_pixel(y_px, volume->imported_texture_height),
                                                           color);
                    }
                }
            }
        }
        if (volume_changed) {
            refresh_imported_texture_storage(*volume);
            changed = true;
        }
    }
    return changed;
}

bool GLGizmoImageProjection::project_to_rgb_data(ModelObject *object)
{
    const Selection &selection = m_parent.get_selection();
    const int instance_idx = selection.get_instance_idx();
    const Camera &camera = wxGetApp().plater()->get_camera();
    const std::array<int, 4> &viewport = camera.get_viewport();
    const OverlayRect rect = overlay_rect();

    ProjectionContext context;
    context.view_projection = (camera.get_projection_matrix() * camera.get_view_matrix()).matrix();
    context.canvas_width = std::max(1, viewport[2]);
    context.canvas_height = std::max(1, viewport[3]);
    context.overlay_left = rect.left;
    context.overlay_top = rect.top;
    context.overlay_width = rect.width;
    context.overlay_height = rect.height;
    context.image_rgba = &m_image_rgba;
    context.image_width = m_image_width;
    context.image_height = m_image_height;
    context.image_opacity = m_projection_opacity;
    context.apply_transparency_as_background = m_apply_transparency_as_background;

    const ProjectionVisibility visibility = m_pass_through_model ?
        ProjectionVisibility() :
        build_projection_visibility(context, m_parent, object, instance_idx);

    bool changed = false;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const VolumeColorSource source = build_volume_color_source(*volume);
        const ColorRGBA fallback_color = projection_base_color_for_volume(*volume);
        const Transform3d world_matrix = projection_world_matrix_for_volume(m_parent, object, volume, instance_idx);
        std::vector<bool> projected_triangles(its.indices.size(), false);

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            projected_triangles[tri_idx] = projection_triangle_intersects_overlay(context, world_matrix, vertices);
        }

        TextureMappingColorSampler sampler = [this, volume, source, context, world_matrix, fallback_color, &projected_triangles, &visibility](size_t tri_idx,
                                                                                                                                             const Vec3f &point,
                                                                                                                                             const Vec3f &barycentric) {
            ColorRGBA color = sample_volume_color_source(*volume, source, tri_idx, point, barycentric, true, &fallback_color);
            if (tri_idx < projected_triangles.size() && projected_triangles[tri_idx]) {
                if (m_pass_through_model || projection_point_is_visible(visibility, context, world_matrix, point)) {
                    if (std::optional<ColorRGBA> projected = projected_image_color_at_point(context, world_matrix, point)) {
                        if (!context.apply_transparency_as_background && !projection_overlay_has_paintable_alpha(*projected, context))
                            return pack_vertex_color_rgba(color);
                        color = apply_projection_color(color, *projected, context, false);
                    }
                }
            }
            return pack_vertex_color_rgba(color);
        };

        const float mesh_span = mesh_max_axis_span(its);
        const int safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 2200000);
        const float split_threshold = safe_max_depth < 5 ? 0.018f : 0.012f;
        TextureMappingColorSubdivisionDepths subdivision_depths =
            [volume, mesh_span, safe_max_depth, &projected_triangles](size_t tri_idx, const std::array<Vec3f, 3> &vertices) {
            int base_depth = model_volume_has_bakeable_image_texture_data(volume) ?
                texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(volume, tri_idx), 8.f, safe_max_depth) :
                texture_mapping_depth_from_span(triangle_max_edge_length(vertices), std::max(mesh_span / 180.f, 0.18f), std::min(6, safe_max_depth));
            if (tri_idx < projected_triangles.size() && projected_triangles[tri_idx])
                base_depth = std::min(std::max(base_depth, 4), safe_max_depth);
            return std::make_pair(base_depth, safe_max_depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, safe_max_depth, split_threshold, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        changed = true;
    }
    return changed;
}

void GLGizmoImageProjection::refresh_projected_object(ModelObject *object)
{
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLMmSegmentationGizmo3DScene::release_geometry() {
    if (this->vertices_VBO_id) {
        glsafe(::glDeleteBuffers(1, &this->vertices_VBO_id));
        this->vertices_VBO_id = 0;
    }
    for(auto &triangle_indices_VBO_id : triangle_indices_VBO_ids) {
        glsafe(::glDeleteBuffers(1, &triangle_indices_VBO_id));
        triangle_indices_VBO_id = 0;
    }

    this->clear();
}

void GLMmSegmentationGizmo3DScene::render(size_t triangle_indices_idx) const
{
    assert(triangle_indices_idx < this->triangle_indices_VBO_ids.size());
    assert(this->triangle_patches.size() == this->triangle_indices_VBO_ids.size());
    assert(this->vertices_VBO_id != 0);
    assert(this->triangle_indices_VBO_ids[triangle_indices_idx] != 0);

    GLShaderProgram* shader = wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

    // the following binding is needed to set the vertex attributes
    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_VBO_id));
    const GLint position_id = shader->get_attrib_location("v_position");
    if (position_id != -1) {
        glsafe(::glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (GLvoid*)0));
        glsafe(::glEnableVertexAttribArray(position_id));
    }

    // Render using the Vertex Buffer Objects.
    if (this->triangle_indices_VBO_ids[triangle_indices_idx] != 0 &&
        this->triangle_indices_sizes[triangle_indices_idx] > 0) {
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices_VBO_ids[triangle_indices_idx]));
        glsafe(::glDrawElements(GL_TRIANGLES, GLsizei(this->triangle_indices_sizes[triangle_indices_idx]), GL_UNSIGNED_INT, nullptr));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }

    if (position_id != -1)
        glsafe(::glDisableVertexAttribArray(position_id));

    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void GLMmSegmentationGizmo3DScene::finalize_vertices()
{
    assert(this->vertices_VBO_id == 0);
    if (!this->vertices.empty()) {
        glsafe(::glGenBuffers(1, &this->vertices_VBO_id));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_VBO_id));
        glsafe(::glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(float), this->vertices.data(), GL_STATIC_DRAW));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
        this->vertices.clear();
    }
}

void GLMmSegmentationGizmo3DScene::finalize_triangle_indices()
{
    triangle_indices_VBO_ids.resize(this->triangle_patches.size());
    triangle_indices_sizes.resize(this->triangle_patches.size());
    assert(std::all_of(triangle_indices_VBO_ids.cbegin(), triangle_indices_VBO_ids.cend(), [](const auto &ti_VBO_id) { return ti_VBO_id == 0; }));

    for (size_t buffer_idx = 0; buffer_idx < this->triangle_patches.size(); ++buffer_idx) {
        std::vector<int>& triangle_indices = this->triangle_patches[buffer_idx].triangle_indices;
        triangle_indices_sizes[buffer_idx] = triangle_indices.size();
        if (!triangle_indices.empty()) {
            glsafe(::glGenBuffers(1, &this->triangle_indices_VBO_ids[buffer_idx]));
            glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices_VBO_ids[buffer_idx]));
            glsafe(::glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangle_indices.size() * sizeof(int), triangle_indices.data(), GL_STATIC_DRAW));
            glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
            triangle_indices.clear();
        }
    }
}

void GLGizmoMmuSegmentation::render_filament_remap_ui(float window_width, float max_tooltip_width)
{
    size_t n_extr = std::min((size_t)EnforcerBlockerType::ExtruderMax, m_display_filament_ids.size());

    const std::string max_label = std::to_string(std::max<size_t>(n_extr, 1));
    const ImVec2 max_label_size = ImGui::CalcTextSize(max_label.c_str(), NULL, true);
    const ImVec2 button_size(max_label_size.x + m_imgui->scaled(0.5f), 0.f);
    const int max_items_per_line = 8;
    const float item_width = button_size.x + m_imgui->scaled(1.5f);
    const float start_pos_x = ImGui::GetCursorPosX();

    for (int src = 0; src < (int)n_extr; ++src) {
        const unsigned int dst_filament_id = m_extruder_remap[src] < m_display_filament_ids.size() ? m_display_filament_ids[m_extruder_remap[src]] : 0;
        if (dst_filament_id == 0 || dst_filament_id > m_extruders_colors.size())
            continue;
        const ColorRGBA &dst_col = m_extruders_colors[dst_filament_id - 1];
        ImVec4 col_vec = ImGuiWrapper::to_ImVec4(dst_col);

        if (src % max_items_per_line != 0) {
            ImGui::SameLine(start_pos_x + item_width * (src % max_items_per_line));
        }
        std::string btn_id = "##remap_src_" + std::to_string(src);
        
        ImGuiColorEditFlags flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_NoLabel  | ImGuiColorEditFlags_NoPicker |
                                    ImGuiColorEditFlags_NoTooltip;
        if (m_selected_extruder_idx != src) flags |= ImGuiColorEditFlags_NoBorder;
        
        #ifdef __APPLE__
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0);
            bool clicked = ImGui::ColorButton(btn_id.c_str(), col_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #else
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0);
            bool clicked = ImGui::ColorButton(btn_id.c_str(), col_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #endif

        // overlay destination number with proper contrast calculation
        std::string dst_txt = std::to_string(m_extruder_remap[src] + 1);
        float gray = 0.299f * dst_col.r() + 0.587f * dst_col.g() + 0.114f * dst_col.b();
        ImVec2 txt_sz = ImGui::CalcTextSize(dst_txt.c_str());
        ImVec2 pos = ImGui::GetItemRectMin();
        ImVec2 size = ImGui::GetItemRectSize();
        
        if (gray * 255.f < 80.f)
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(pos.x + (size.x - txt_sz.x) * 0.5f, pos.y + (size.y - txt_sz.y) * 0.5f),
                IM_COL32(255,255,255,255), dst_txt.c_str());
        else
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(pos.x + (size.x - txt_sz.x) * 0.5f, pos.y + (size.y - txt_sz.y) * 0.5f),
                IM_COL32(0,0,0,255), dst_txt.c_str());

        // popup with possible destinations
        std::string pop_id = "popup_" + std::to_string(src);
        if (clicked) {
            // Calculate popup position centered below the current button
            ImVec2 button_pos = ImGui::GetItemRectMin();
            ImVec2 button_size = ImGui::GetItemRectSize();
            ImVec2 popup_pos(button_pos.x + button_size.x * 0.5f, button_pos.y + button_size.y);
            
            // Set popup styling BEFORE opening popup
            ImGui::SetNextWindowPos(popup_pos, ImGuiCond_Appearing, ImVec2(0.5f, -0.1f));
            ImGui::SetNextWindowBgAlpha(1.0f); // Ensure full opacity
            ImGui::OpenPopup(pop_id.c_str());
        }
        
        // Apply popup styling before BeginPopup using standard Orca colors
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, m_is_dark_mode ? ImGuiWrapper::COL_WINDOW_BG_DARK : ImGuiWrapper::COL_WINDOW_BG);
        ImGui::PushStyleColor(ImGuiCol_Border, m_is_dark_mode ? ImVec4(0.5f, 0.5f, 0.5f, 1.0f) : ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        
        if (ImGui::BeginPopup(pop_id.c_str())) {
            const float popup_start_pos_x = ImGui::GetCursorPosX();
            
            for (int dst = 0; dst < (int)n_extr; ++dst) {
                const unsigned int popup_filament_id = m_display_filament_ids[dst];
                if (popup_filament_id == 0 || popup_filament_id > m_extruders_colors.size())
                    continue;
                const ColorRGBA &dst_col_popup = m_extruders_colors[popup_filament_id - 1];
                ImVec4 dst_vec = ImGuiWrapper::to_ImVec4(dst_col_popup);
                if (dst % max_items_per_line != 0)
                    ImGui::SameLine(popup_start_pos_x + item_width * (dst % max_items_per_line));
                std::string dst_btn = "##dst_" + std::to_string(src) + "_" + std::to_string(dst);
                
                // Apply same styling to destination buttons
                ImGuiColorEditFlags dst_flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs |
                                               ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoPicker |
                                               ImGuiColorEditFlags_NoTooltip;
                // Show border for currently selected destination filament
                if (m_extruder_remap[src] != dst) dst_flags |= ImGuiColorEditFlags_NoBorder;
                
                #ifdef __APPLE__
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0);
                    bool dst_clicked = ImGui::ColorButton(dst_btn.c_str(), dst_vec, dst_flags, button_size);
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor(1);
                #else
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0);
                    bool dst_clicked = ImGui::ColorButton(dst_btn.c_str(), dst_vec, dst_flags, button_size);
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor(1);
                #endif
                
                // overlay destination number on popup buttons
                std::string dst_num_txt = std::to_string(dst + 1);
                float dst_gray = 0.299f * dst_col_popup.r() + 0.587f * dst_col_popup.g() + 0.114f * dst_col_popup.b();
                ImVec2 dst_txt_sz = ImGui::CalcTextSize(dst_num_txt.c_str());
                ImVec2 dst_pos = ImGui::GetItemRectMin();
                ImVec2 dst_size = ImGui::GetItemRectSize();
                
                if (dst_gray * 255.f < 80.f)
                    ImGui::GetWindowDrawList()->AddText(
                        ImVec2(dst_pos.x + (dst_size.x - dst_txt_sz.x) * 0.5f, dst_pos.y + (dst_size.y - dst_txt_sz.y) * 0.5f),
                        IM_COL32(255,255,255,255), dst_num_txt.c_str());
                else
                    ImGui::GetWindowDrawList()->AddText(
                        ImVec2(dst_pos.x + (dst_size.x - dst_txt_sz.x) * 0.5f, dst_pos.y + (dst_size.y - dst_txt_sz.y) * 0.5f),
                        IM_COL32(0,0,0,255), dst_num_txt.c_str());
                
                if (dst_clicked)
                {
                    m_extruder_remap[src] = dst;
                    // update the source button color immediately
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::EndPopup();
        }
        
        // Clean up popup styling (always pop, whether popup was open or not)
        ImGui::PopStyleColor(2); // PopupBg and Border
        ImGui::PopStyleVar(2);   // PopupRounding and PopupBorderSize
    }

    ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.3f));

    if (m_imgui->button(m_desc.at("remap"))) {
        remap_filament_assignments();
        m_show_filament_remap_ui = false;
    }

    ImGui::SameLine();
    if (m_imgui->button(m_desc.at("cancel_remap")))
        m_show_filament_remap_ui = false;
}

void GLGizmoMmuSegmentation::remap_filament_assignments()
{
    if (m_extruder_remap.empty())
        return;

    constexpr size_t MAX_EBT = (size_t)EnforcerBlockerType::ExtruderMax;
    EnforcerBlockerStateMap state_map;

    // identity mapping by default
    for (size_t i = 0; i <= MAX_EBT; ++i)
        state_map[i] = static_cast<EnforcerBlockerType>(i);

    size_t n_extr = std::min({m_extruder_remap.size(), m_display_filament_ids.size(), MAX_EBT});
    bool   any_change = false;
    for (size_t src = 0; src < n_extr; ++src) {
        const size_t dst = m_extruder_remap[src];
        if (dst >= m_display_filament_ids.size())
            continue;

        const unsigned int src_state = m_display_filament_ids[src];
        const unsigned int dst_state = m_display_filament_ids[dst];
        if (src_state == 0 || dst_state == 0 || src_state == dst_state)
            continue;

        state_map[src_state] = static_cast<EnforcerBlockerType>(dst_state);
        if (src_state == 1)
            state_map[0] = static_cast<EnforcerBlockerType>(dst_state);

        any_change = true;
    }
    if (!any_change)
        return;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(),
                                  "Remap filament assignments",
                                  UndoRedo::SnapshotType::GizmoAction);

    bool updated = false;
    int idx = -1;
    ModelObject* mo = m_c->selection_info()->model_object();
    if (!mo) return;

    for (ModelVolume* mv : mo->volumes) {
        if (!mv->is_model_part()) continue;
        ++idx;
        TriangleSelectorGUI* ts = m_triangle_selectors[idx].get();
        if (!ts) continue;
        ts->remap_triangle_state(state_map);
        ts->request_update_render_data(true);
        updated = true;
    }

    if (updated) {
        wxGetApp().plater()->get_notification_manager()->push_notification(
            _L("Filament remapping finished.").ToStdString());
        update_model_object();
        m_parent.set_as_dirty();
    }
}

} // namespace Slic3r
