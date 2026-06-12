#include <glad/gl.h>

#include "3DScene.hpp"
#include "GLShader.hpp"
#include "MMUPaintedTexturePreview.hpp"
#include "GUI_App.hpp"
#include "GUI_Colors.hpp"
#include "Plater.hpp"
#include "BitmapCache.hpp"
#include "Camera.hpp"

#include "libslic3r/BuildVolume.hpp"
#include "libslic3r/Color.hpp"
#include "ColorSolver.hpp"
#include "libslic3r/ExtrusionEntity.hpp"
#include "libslic3r/ExtrusionEntityCollection.hpp"
#include "libslic3r/Geometry.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/SLAPrint.hpp"
#include "libslic3r/Slicing.hpp"
#include "libslic3r/Format/STL.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/AppConfig.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/Tesselate.hpp"
#include "libslic3r/PrintConfig.hpp"
#include "libslic3r/TextureMapping.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <Eigen/Dense>

#ifdef HAS_GLSAFE
void glAssertRecentCallImpl(const char* file_name, unsigned int line, const char* function_name)
{
#if defined(NDEBUG)
    // In release mode, only show OpenGL errors if sufficiently high loglevel.
    if (Slic3r::get_logging_level() < 5)
        return;
#endif // NDEBUG

    GLenum err = glGetError();
    if (err == GL_NO_ERROR)
        return;
    const char* sErr = 0;
    switch (err) {
    case GL_INVALID_ENUM:       sErr = "Invalid Enum";      break;
    case GL_INVALID_VALUE:      sErr = "Invalid Value";     break;
    // be aware that GL_INVALID_OPERATION is generated if glGetError is executed between the execution of glBegin and the corresponding execution of glEnd
    case GL_INVALID_OPERATION:  sErr = "Invalid Operation"; break;
    case GL_STACK_OVERFLOW:     sErr = "Stack Overflow";    break;
    case GL_STACK_UNDERFLOW:    sErr = "Stack Underflow";   break;
    case GL_OUT_OF_MEMORY:      sErr = "Out Of Memory";     break;
    default:                    sErr = "Unknown";           break;
    }
    BOOST_LOG_TRIVIAL(error) << "OpenGL error in " << file_name << ":" << line << ", function " << function_name << "() : " << (int)err << " - " << sErr;
    assert(false);
}
#endif // HAS_GLSAFE

float FullyTransparentMaterialThreshold  = 0.1f;
float FullTransparentModdifiedToFixAlpha = 0.3f;
// Be careful changing this value because it could break thumbnail color due to rounding error!
// The color rendering on BambuLab's "send to printer" screen relies on the assumption that this color can be accurately rendered by OpenGL,
// value like 0.18f could not because in C++ (int)(0.18f * 255) == 45 however in OpenGL it renders this as 46
// which breaks the `SelectMachineDialog::record_edge_pixels_data()` function!
float FULL_BLACK_THRESHOLD = 0.2f;

Slic3r::ColorRGBA adjust_color_for_rendering(const Slic3r::ColorRGBA &colors)
{
    if (colors.a() < FullyTransparentMaterialThreshold) { // completely transparent
        return {1, 1, 1, FullTransparentModdifiedToFixAlpha};
    }
    else if(colors.r() < FULL_BLACK_THRESHOLD && colors.g() < FULL_BLACK_THRESHOLD && colors.b() < FULL_BLACK_THRESHOLD) { // black
        return {FULL_BLACK_THRESHOLD, FULL_BLACK_THRESHOLD, FULL_BLACK_THRESHOLD, colors.a()};
    }
    else
        return colors;
}

namespace Slic3r {

namespace {

std::vector<TriangleSelector::FacetStateTriangle> build_full_mesh_texture_preview_triangles(const ModelVolume &model_volume)
{
    std::vector<TriangleSelector::FacetStateTriangle> out;
    const indexed_triangle_set &its = model_volume.mesh().its;
    out.reserve(its.indices.size());
    for (size_t triangle_idx = 0; triangle_idx < its.indices.size(); ++triangle_idx) {
        const stl_triangle_vertex_indices &triangle = its.indices[triangle_idx];
        if (triangle[0] < 0 || triangle[1] < 0 || triangle[2] < 0)
            continue;
        if (size_t(triangle[0]) >= its.vertices.size() ||
            size_t(triangle[1]) >= its.vertices.size() ||
            size_t(triangle[2]) >= its.vertices.size())
            continue;

        TriangleSelector::FacetStateTriangle facet;
        facet.source_triangle = int(triangle_idx);
        facet.vertices[0] = its.vertices[size_t(triangle[0])].cast<float>();
        facet.vertices[1] = its.vertices[size_t(triangle[1])].cast<float>();
        facet.vertices[2] = its.vertices[size_t(triangle[2])].cast<float>();
        out.emplace_back(std::move(facet));
    }
    return out;
}

const TextureMappingZone *texture_preview_zone_for_filament(unsigned int filament_id,
                                                            size_t num_physical,
                                                            const TextureMappingManager *texture_mgr)
{
    const TextureMappingZone *zone = texture_mgr != nullptr && filament_id > num_physical ?
        texture_mgr->zone_from_id(filament_id) : nullptr;
    return zone != nullptr && zone->enabled && !zone->deleted && (zone->is_image_texture() || zone->is_surface_gradient()) ?
        zone : nullptr;
}

bool filament_state_uses_texture_preview(unsigned int filament_id,
                                         size_t num_physical,
                                         const TextureMappingManager *texture_mgr)
{
    return texture_preview_zone_for_filament(filament_id, num_physical, texture_mgr) != nullptr;
}

bool filament_state_uses_surface_gradient_preview(unsigned int filament_id,
                                                  size_t num_physical,
                                                  const TextureMappingManager *texture_mgr)
{
    const TextureMappingZone *zone = texture_preview_zone_for_filament(filament_id, num_physical, texture_mgr);
    return zone != nullptr && zone->is_surface_gradient();
}

bool texture_preview_used_states_have_surface_gradient(const std::vector<bool> &used_states,
                                                       size_t num_physical,
                                                       const TextureMappingManager *texture_mgr)
{
    for (size_t state_id = 1; state_id < used_states.size(); ++state_id) {
        if (used_states[state_id] && filament_state_uses_surface_gradient_preview(unsigned(state_id), num_physical, texture_mgr))
            return true;
    }

    return false;
}

bool linear_gradient_vec3_is_finite(const Vec3f &point)
{
    return std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z());
}

Vec3f linear_gradient_anchor_array_point(const std::array<float, 3> &point)
{
    return Vec3f(point[0], point[1], point[2]);
}

int linear_gradient_model_object_backup_id(const ModelObject *object)
{
    const Model *model = object != nullptr ? object->get_model() : nullptr;
    return model != nullptr ? model->find_object_backup_id(*object) : -1;
}

const ModelObject *linear_gradient_anchor_model_object(const Model &model,
                                                       const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (anchor.object_backup_id >= 0) {
        for (const ModelObject *object : model.objects)
            if (linear_gradient_model_object_backup_id(object) == anchor.object_backup_id)
                return object;
    }
    if (anchor.object_id != 0) {
        for (const ModelObject *object : model.objects)
            if (object != nullptr && object->id().id == anchor.object_id)
                return object;
    }
    if (anchor.object_index_valid && anchor.object_index < model.objects.size())
        return model.objects[anchor.object_index];
    return nullptr;
}

bool linear_gradient_anchor_has_object_reference(const TextureMappingZone::LinearGradientAnchor &anchor)
{
    return anchor.object_backup_id >= 0 || anchor.object_id != 0 || anchor.object_index_valid;
}

bool linear_gradient_anchor_object_resolves(const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (!anchor.valid || !linear_gradient_anchor_has_object_reference(anchor))
        return true;
    return linear_gradient_anchor_model_object(GUI::wxGetApp().model(), anchor) != nullptr;
}

bool linear_gradient_anchor_matches_model_object(const Model &model,
                                                const ModelObject *object,
                                                const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (object == nullptr)
        return false;
    if (anchor.object_backup_id >= 0 && linear_gradient_model_object_backup_id(object) == anchor.object_backup_id)
        return true;
    if (anchor.object_id != 0 && object->id().id == anchor.object_id)
        return true;
    return anchor.object_index_valid && anchor.object_index < model.objects.size() && model.objects[anchor.object_index] == object;
}

const ModelInstance *linear_gradient_anchor_model_instance(const ModelObject *object,
                                                          const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (object == nullptr)
        return nullptr;
    if (anchor.instance_loaded_id != 0) {
        for (const ModelInstance *instance : object->instances)
            if (instance != nullptr && instance->loaded_id == anchor.instance_loaded_id)
                return instance;
    }
    if (anchor.instance_id != 0) {
        for (const ModelInstance *instance : object->instances)
            if (instance != nullptr && instance->id().id == anchor.instance_id)
                return instance;
    }
    if (anchor.instance_index_valid && anchor.instance_index < object->instances.size())
        return object->instances[anchor.instance_index];
    return object->instances.size() == 1 ? object->instances.front() : nullptr;
}

std::optional<Vec3f> linear_gradient_anchor_global_point(const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (!anchor.valid)
        return std::nullopt;

    const Vec3f local = linear_gradient_anchor_array_point(anchor.local_point);
    const Model &model = GUI::wxGetApp().model();
    const ModelObject *object = linear_gradient_anchor_model_object(model, anchor);
    const ModelInstance *instance = linear_gradient_anchor_model_instance(object, anchor);
    if (instance != nullptr) {
        const Vec3f global = (instance->get_matrix() * local.cast<double>()).cast<float>();
        if (linear_gradient_vec3_is_finite(global))
            return global;
    }

    const Vec3f fallback = linear_gradient_anchor_array_point(anchor.global_point);
    return linear_gradient_vec3_is_finite(fallback) ? std::optional<Vec3f>(fallback) : std::nullopt;
}

std::optional<Vec3f> linear_gradient_anchor_global_point(const GLVolumePtrs &volumes,
                                                         const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (!anchor.valid)
        return std::nullopt;

    const Vec3f local = linear_gradient_anchor_array_point(anchor.local_point);
    const Model &model = GUI::wxGetApp().model();
    const ModelObject *anchor_object = linear_gradient_anchor_model_object(model, anchor);
    const ModelInstance *anchor_instance = linear_gradient_anchor_model_instance(anchor_object, anchor);
    for (const GLVolume *volume : volumes) {
        if (volume == nullptr ||
            volume->object_idx() < 0 ||
            volume->instance_idx() < 0 ||
            volume->object_idx() >= int(model.objects.size()))
            continue;

        const ModelObject *object = model.objects[size_t(volume->object_idx())];
        if (object == nullptr || !linear_gradient_anchor_matches_model_object(model, object, anchor))
            continue;
        if (size_t(volume->instance_idx()) >= object->instances.size())
            continue;

        const ModelInstance *instance = object->instances[size_t(volume->instance_idx())];
        if (anchor_instance != nullptr && instance != anchor_instance)
            continue;
        if (anchor_instance == nullptr && object->instances.size() != 1)
            continue;

        const Vec3f global = (volume->get_instance_transformation().get_matrix() * local.cast<double>()).cast<float>();
        if (linear_gradient_vec3_is_finite(global))
            return global;
    }

    return linear_gradient_anchor_global_point(anchor);
}

std::optional<float> linear_gradient_anchor_object_radius(const GLVolumePtrs &volumes,
                                                         const TextureMappingZone::LinearGradientAnchor &anchor)
{
    if (!anchor.valid)
        return std::nullopt;

    const Model &model = GUI::wxGetApp().model();
    const ModelObject *anchor_object = linear_gradient_anchor_model_object(model, anchor);
    const ModelInstance *anchor_instance = linear_gradient_anchor_model_instance(anchor_object, anchor);
    BoundingBoxf3 bbox;
    bool defined = false;
    for (const GLVolume *volume : volumes) {
        if (volume == nullptr ||
            volume->object_idx() < 0 ||
            volume->instance_idx() < 0 ||
            volume->object_idx() >= int(model.objects.size()))
            continue;

        const ModelObject *object = model.objects[size_t(volume->object_idx())];
        if (object == nullptr || !linear_gradient_anchor_matches_model_object(model, object, anchor))
            continue;
        if (size_t(volume->instance_idx()) >= object->instances.size())
            continue;

        const ModelInstance *instance = object->instances[size_t(volume->instance_idx())];
        if (anchor_instance != nullptr && instance != anchor_instance)
            continue;
        if (anchor_instance == nullptr && object->instances.size() != 1)
            continue;

        bbox.merge(volume->transformed_convex_hull_bounding_box());
        defined = true;
    }

    if (!defined)
        return std::nullopt;
    const float radius = 0.5f * (bbox.max.cast<float>() - bbox.min.cast<float>()).norm();
    return std::isfinite(radius) && radius > 0.f ? std::optional<float>(radius) : std::nullopt;
}

float linear_gradient_radial_radius_mm(const GLVolumePtrs &volumes,
                                       const TextureMappingZone &zone,
                                       float default_radius_mm)
{
    if (!zone.linear_gradient_radius_percent) {
        const float radius_mm = std::isfinite(zone.linear_gradient_radius_mm) ? zone.linear_gradient_radius_mm : default_radius_mm;
        return std::max(0.01f, radius_mm);
    }

    const float radius_value = std::isfinite(zone.linear_gradient_radius_pct) ?
        std::max(0.f, zone.linear_gradient_radius_pct) :
        TextureMappingZone::DefaultLinearGradientRadiusPct;
    if (zone.linear_gradient_start.valid) {
        const std::optional<float> anchor_radius = linear_gradient_anchor_object_radius(volumes, zone.linear_gradient_start);
        if (anchor_radius)
            return std::max(0.01f, *anchor_radius * radius_value / 100.f);
        return std::max(0.01f, radius_value);
    }

    return std::max(0.01f, default_radius_mm * radius_value / 100.f);
}

bool model_volume_uses_texture_mapping_zone(const ModelVolume &model_volume, unsigned int zone_id)
{
    if (zone_id == 0)
        return false;
    if (model_volume.extruder_id() > 0 && unsigned(model_volume.extruder_id()) == zone_id)
        return true;
    if (model_volume.mmu_segmentation_facets.empty())
        return false;
    const std::vector<bool> &used_states = model_volume.mmu_segmentation_facets.get_data().used_states;
    return zone_id < used_states.size() && used_states[zone_id];
}

std::vector<unsigned int> linear_gradient_component_ids_for_arrow(const TextureMappingZone &zone, size_t num_physical)
{
    std::vector<unsigned int> ids = TextureMappingManager::selected_component_ids(zone, num_physical);
    ids.erase(std::remove_if(ids.begin(), ids.end(), [num_physical](unsigned int id) {
        return id == 0 || id > num_physical;
    }), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.empty() && num_physical >= 1)
        ids = {1};
    return ids;
}

std::array<float, 3> linear_gradient_filament_color(unsigned int filament_id, const std::vector<std::string> &filament_colors)
{
    ColorRGB decoded;
    if (filament_id >= 1 &&
        size_t(filament_id - 1) < filament_colors.size() &&
        decode_color(filament_colors[size_t(filament_id - 1)], decoded))
        return { decoded.r(), decoded.g(), decoded.b() };
    return { 0.5f, 0.5f, 0.5f };
}

ColorRGBA linear_gradient_arrow_color(const std::vector<std::array<float, 3>> &component_colors,
                                      const std::vector<TextureMappingZone::LinearGradientStop> &stops,
                                      const std::vector<unsigned int> &component_ids,
                                      ColorSolverMixModel mix_model,
                                      float t)
{
    const std::vector<float> weights = TextureMappingManager::linear_gradient_compact_weights(t, stops, component_ids);
    const std::array<float, 3> mixed = mix_color_solver_components(component_colors, weights, mix_model);
    return { std::clamp(mixed[0], 0.f, 1.f), std::clamp(mixed[1], 0.f, 1.f), std::clamp(mixed[2], 0.f, 1.f), 1.f };
}

float linear_gradient_arrow_t(const Vec3f &point,
                              const Vec3f &arrow_start,
                              const Vec3f &axis,
                              float arrow_length)
{
    return std::clamp((point - arrow_start).dot(axis) / std::max(0.0001f, arrow_length), 0.f, 1.f);
}

struct LinearGradientArrowUsage {
    bool any { false };
    Vec3f default_start { Vec3f::Zero() };
    Vec3f default_end { Vec3f::UnitZ() };
    Vec3f radial_center { Vec3f::Zero() };
    float radial_radius { 1.f };
};

LinearGradientArrowUsage linear_gradient_arrow_usage(const GLVolumePtrs &volumes, unsigned int zone_id)
{
    LinearGradientArrowUsage usage;
    const Model &model = GUI::wxGetApp().model();
    std::map<std::pair<int, int>, BoundingBoxf3> object_boxes;
    for (const GLVolume *volume : volumes) {
        if (volume == nullptr ||
            !volume->is_active ||
            volume->disabled ||
            volume->is_wipe_tower ||
            volume->is_modifier ||
            volume->is_extrusion_path ||
            volume->object_idx() < 0 ||
            volume->volume_idx() < 0 ||
            volume->instance_idx() < 0)
            continue;
        if (size_t(volume->object_idx()) >= model.objects.size())
            continue;
        const ModelObject *object = model.objects[size_t(volume->object_idx())];
        if (object == nullptr || size_t(volume->volume_idx()) >= object->volumes.size())
            continue;
        const ModelVolume *model_volume = object->volumes[size_t(volume->volume_idx())];
        if (model_volume == nullptr || !model_volume_uses_texture_mapping_zone(*model_volume, zone_id))
            continue;

        const BoundingBoxf3 bbox = volume->transformed_convex_hull_bounding_box();
        if (!bbox.defined)
            continue;
        const std::pair<int, int> key { volume->object_idx(), volume->instance_idx() };
        auto it = object_boxes.find(key);
        if (it == object_boxes.end())
            object_boxes.emplace(key, bbox);
        else
            it->second.merge(bbox);
    }

    if (object_boxes.empty())
        return usage;

    Vec3f center_sum = Vec3f::Zero();
    float max_z = 0.f;
    float max_radius = 0.01f;
    size_t count = 0;
    for (const auto &item : object_boxes) {
        if (!item.second.defined)
            continue;
        const Vec3f center = item.second.center().cast<float>();
        center_sum += center;
        max_z = std::max(max_z, float(item.second.max.z()));
        max_radius = std::max(max_radius, 0.5f * (item.second.max.cast<float>() - item.second.min.cast<float>()).norm());
        ++count;
    }
    if (count == 0)
        return usage;

    const Vec3f avg = center_sum / float(count);
    usage.any = true;
    usage.default_start = Vec3f(avg.x(), avg.y(), 0.f);
    usage.default_end = Vec3f(avg.x(), avg.y(), std::max(0.01f, max_z));
    usage.radial_center = avg;
    usage.radial_radius = std::max(0.01f, max_radius);
    return usage;
}

unsigned int linear_gradient_add_vertex(GUI::GLModel::Geometry &geometry,
                                        const Vec3f &position,
                                        const Vec3f &normal,
                                        const ColorRGBA &color)
{
    const unsigned int id = unsigned(geometry.vertices_count());
    geometry.add_vertex(position, normal, color);
    return id;
}

void linear_gradient_arrow_basis(const Vec3f &direction, Vec3f &u, Vec3f &v)
{
    const Vec3f reference = std::abs(direction.z()) < 0.9f ? Vec3f::UnitZ() : Vec3f::UnitX();
    u = direction.cross(reference).normalized();
    if (!linear_gradient_vec3_is_finite(u) || u.squaredNorm() <= 1e-8f)
        u = Vec3f::UnitX();
    v = direction.cross(u).normalized();
    if (!linear_gradient_vec3_is_finite(v) || v.squaredNorm() <= 1e-8f)
        v = Vec3f::UnitY();
}

void append_linear_gradient_cylinder(GUI::GLModel::Geometry &geometry,
                                     const Vec3f &start,
                                     const Vec3f &end,
                                     const Vec3f &arrow_start,
                                     float arrow_length,
                                     const Vec3f &axis,
                                     const Vec3f &u,
                                     const Vec3f &v,
                                     float radius,
                                     const std::vector<std::array<float, 3>> &component_colors,
                                     const std::vector<TextureMappingZone::LinearGradientStop> &stops,
                                     const std::vector<unsigned int> &component_ids,
                                     ColorSolverMixModel mix_model,
                                     bool black)
{
    constexpr int sectors = 32;
    constexpr int axial_steps = 72;
    const ColorRGBA black_color = ColorRGBA::BLACK();
    std::vector<std::vector<unsigned int>> ring_ids(size_t(axial_steps + 1), std::vector<unsigned int>(size_t(sectors + 1), 0));
    for (int step = 0; step <= axial_steps; ++step) {
        const float step_t = float(step) / float(axial_steps);
        const Vec3f center = start + (end - start) * step_t;
        const float arrow_t = linear_gradient_arrow_t(center, arrow_start, axis, arrow_length);
        const ColorRGBA color = black ? black_color : linear_gradient_arrow_color(component_colors, stops, component_ids, mix_model, arrow_t);
        for (int sector = 0; sector <= sectors; ++sector) {
            const float a = 2.f * float(PI) * float(sector) / float(sectors);
            const Vec3f radial = std::cos(a) * u + std::sin(a) * v;
            ring_ids[size_t(step)][size_t(sector)] =
                linear_gradient_add_vertex(geometry, center + radius * radial, radial, color);
        }
    }

    const unsigned int start_center =
        linear_gradient_add_vertex(geometry,
                                   start,
                                   -axis,
                                   black ? black_color : linear_gradient_arrow_color(component_colors, stops, component_ids, mix_model, linear_gradient_arrow_t(start, arrow_start, axis, arrow_length)));
    for (int i = 0; i < sectors; ++i) {
        geometry.add_triangle(start_center, ring_ids[0][size_t(i + 1)], ring_ids[0][size_t(i)]);
        for (int step = 0; step < axial_steps; ++step) {
            const unsigned int i0 = ring_ids[size_t(step)][size_t(i)];
            const unsigned int i1 = ring_ids[size_t(step)][size_t(i + 1)];
            const unsigned int i2 = ring_ids[size_t(step + 1)][size_t(i)];
            const unsigned int i3 = ring_ids[size_t(step + 1)][size_t(i + 1)];
            geometry.add_triangle(i0, i2, i1);
            geometry.add_triangle(i1, i2, i3);
        }
    }
}

void append_linear_gradient_cone(GUI::GLModel::Geometry &geometry,
                                 const Vec3f &base,
                                 const Vec3f &tip,
                                 const Vec3f &arrow_start,
                                 float arrow_length,
                                 const Vec3f &axis,
                                 const Vec3f &u,
                                 const Vec3f &v,
                                 float radius,
                                 const std::vector<std::array<float, 3>> &component_colors,
                                 const std::vector<TextureMappingZone::LinearGradientStop> &stops,
                                 const std::vector<unsigned int> &component_ids,
                                 ColorSolverMixModel mix_model,
                                 bool black)
{
    constexpr int sectors = 32;
    constexpr int axial_steps = 24;
    const ColorRGBA black_color = ColorRGBA::BLACK();
    const ColorRGBA tip_color = black ? black_color : linear_gradient_arrow_color(component_colors, stops, component_ids, mix_model, linear_gradient_arrow_t(tip, arrow_start, axis, arrow_length));
    std::vector<std::vector<unsigned int>> ring_ids(size_t(axial_steps), std::vector<unsigned int>(size_t(sectors + 1), 0));
    const Vec3f cone_axis = tip - base;
    for (int step = 0; step < axial_steps; ++step) {
        const float step_t = float(step) / float(axial_steps);
        const Vec3f center = base + cone_axis * step_t;
        const float ring_radius = radius * (1.f - step_t);
        const ColorRGBA color = black ? black_color : linear_gradient_arrow_color(component_colors, stops, component_ids, mix_model, linear_gradient_arrow_t(center, arrow_start, axis, arrow_length));
        for (int sector = 0; sector <= sectors; ++sector) {
            const float a = 2.f * float(PI) * float(sector) / float(sectors);
            const Vec3f radial = std::cos(a) * u + std::sin(a) * v;
            const Vec3f normal = (radial + axis * 0.35f).normalized();
            ring_ids[size_t(step)][size_t(sector)] =
                linear_gradient_add_vertex(geometry, center + ring_radius * radial, normal, color);
        }
    }

    for (int i = 0; i < sectors; ++i) {
        for (int step = 0; step + 1 < axial_steps; ++step) {
            const unsigned int i0 = ring_ids[size_t(step)][size_t(i)];
            const unsigned int i1 = ring_ids[size_t(step)][size_t(i + 1)];
            const unsigned int i2 = ring_ids[size_t(step + 1)][size_t(i)];
            const unsigned int i3 = ring_ids[size_t(step + 1)][size_t(i + 1)];
            geometry.add_triangle(i0, i2, i1);
            geometry.add_triangle(i1, i2, i3);
        }
        const unsigned int i0 = ring_ids[size_t(axial_steps - 1)][size_t(i)];
        const unsigned int i1 = ring_ids[size_t(axial_steps - 1)][size_t(i + 1)];
        const unsigned int tip_idx = linear_gradient_add_vertex(geometry, tip, axis, tip_color);
        geometry.add_triangle(i0, tip_idx, i1);
    }
}

void append_linear_gradient_sphere(GUI::GLModel::Geometry &geometry,
                                   const Vec3f &center,
                                   float radius,
                                   const ColorRGBA &color)
{
    constexpr int sectors = 32;
    constexpr int stacks = 16;
    std::vector<std::vector<unsigned int>> ids(size_t(stacks + 1), std::vector<unsigned int>(size_t(sectors + 1), 0));
    for (int stack = 0; stack <= stacks; ++stack) {
        const float phi = 0.5f * float(PI) - float(PI) * float(stack) / float(stacks);
        const float z = std::sin(phi);
        const float xy = std::cos(phi);
        for (int sector = 0; sector <= sectors; ++sector) {
            const float theta = 2.f * float(PI) * float(sector) / float(sectors);
            const Vec3f normal(xy * std::cos(theta), xy * std::sin(theta), z);
            ids[size_t(stack)][size_t(sector)] =
                linear_gradient_add_vertex(geometry, center + radius * normal, normal, color);
        }
    }
    for (int stack = 0; stack < stacks; ++stack) {
        for (int sector = 0; sector < sectors; ++sector) {
            const unsigned int a = ids[size_t(stack)][size_t(sector)];
            const unsigned int b = ids[size_t(stack + 1)][size_t(sector)];
            const unsigned int c = ids[size_t(stack)][size_t(sector + 1)];
            const unsigned int d = ids[size_t(stack + 1)][size_t(sector + 1)];
            if (stack != 0)
                geometry.add_triangle(a, b, c);
            if (stack + 1 != stacks)
                geometry.add_triangle(c, b, d);
        }
    }
}

void append_linear_gradient_dashed_ring(GUI::GLModel::Geometry &geometry,
                                        const Vec3f &center,
                                        const Vec3f &u,
                                        const Vec3f &v,
                                        float radius,
                                        float tube_radius,
                                        const ColorRGBA &color)
{
    constexpr int dash_count = 20;
    constexpr int arc_steps = 4;
    constexpr int tube_sectors = 8;
    constexpr float dash_fraction = 0.42f;
    if (radius <= 1e-5f || tube_radius <= 1e-6f)
        return;
    Vec3f normal = u.cross(v).normalized();
    if (!linear_gradient_vec3_is_finite(normal) || normal.squaredNorm() <= 1e-8f)
        normal = Vec3f::UnitZ();
    for (int dash = 0; dash < dash_count; ++dash) {
        const float a_start = 2.f * float(PI) * float(dash) / float(dash_count);
        const float a_end = a_start + 2.f * float(PI) * dash_fraction / float(dash_count);
        std::vector<std::vector<unsigned int>> ids(size_t(arc_steps + 1), std::vector<unsigned int>(size_t(tube_sectors + 1), 0));
        for (int step = 0; step <= arc_steps; ++step) {
            const float t = float(step) / float(arc_steps);
            const float a = a_start + (a_end - a_start) * t;
            const Vec3f radial = std::cos(a) * u + std::sin(a) * v;
            const Vec3f p = center + radius * radial;
            for (int sector = 0; sector <= tube_sectors; ++sector) {
                const float b = 2.f * float(PI) * float(sector) / float(tube_sectors);
                const Vec3f tube_normal = std::cos(b) * radial + std::sin(b) * normal;
                ids[size_t(step)][size_t(sector)] =
                    linear_gradient_add_vertex(geometry, p + tube_radius * tube_normal, tube_normal, color);
            }
        }
        for (int step = 0; step < arc_steps; ++step) {
            for (int sector = 0; sector < tube_sectors; ++sector) {
                const unsigned int a = ids[size_t(step)][size_t(sector)];
                const unsigned int b = ids[size_t(step + 1)][size_t(sector)];
                const unsigned int c = ids[size_t(step)][size_t(sector + 1)];
                const unsigned int d = ids[size_t(step + 1)][size_t(sector + 1)];
                geometry.add_triangle(a, b, c);
                geometry.add_triangle(c, b, d);
            }
        }
    }
}

void append_linear_gradient_dashed_sphere(GUI::GLModel::Geometry &geometry,
                                          const Vec3f &center,
                                          float radius,
                                          float tube_radius,
                                          const ColorRGBA &color)
{
    append_linear_gradient_dashed_ring(geometry, center, Vec3f::UnitX(), Vec3f::UnitY(), radius, tube_radius, color);
    append_linear_gradient_dashed_ring(geometry, center, Vec3f::UnitX(), Vec3f::UnitZ(), radius, tube_radius, color);
    append_linear_gradient_dashed_ring(geometry, center, Vec3f::UnitY(), Vec3f::UnitZ(), radius, tube_radius, color);
}

void append_linear_gradient_arrow(GUI::GLModel::Geometry &geometry,
                                  const Vec3f &start,
                                  const Vec3f &end,
                                  float radius,
                                  float head_radius,
                                  float head_length,
                                  const std::vector<std::array<float, 3>> &component_colors,
                                  const std::vector<TextureMappingZone::LinearGradientStop> &stops,
                                  const std::vector<unsigned int> &component_ids,
                                  ColorSolverMixModel mix_model,
                                  bool black)
{
    const Vec3f delta = end - start;
    const float length = delta.norm();
    if (length <= 1e-6f)
        return;
    const Vec3f axis = delta / length;
    Vec3f u;
    Vec3f v;
    linear_gradient_arrow_basis(axis, u, v);
    float clamped_head_length = std::min(head_length, length * 0.75f);
    clamped_head_length = std::max(clamped_head_length, std::min(radius * 2.f, length * 0.5f));
    const Vec3f shaft_end = start + axis * std::max(radius, length - clamped_head_length);
    append_linear_gradient_cylinder(geometry, start, shaft_end, start, length, axis, u, v, radius, component_colors, stops, component_ids, mix_model, black);
    append_linear_gradient_cone(geometry, shaft_end, end, start, length, axis, u, v, head_radius, component_colors, stops, component_ids, mix_model, black);
}

} // namespace


const float GLVolume::SinkingContours::HalfWidth = 0.25f;

void GLVolume::SinkingContours::render()
{
    update();

    GLShaderProgram* shader = GUI::wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

    const GUI::Camera& camera = GUI::wxGetApp().plater()->get_camera();
    shader->set_uniform("view_model_matrix", camera.get_view_matrix() * Geometry::assemble_transform(m_shift));
    shader->set_uniform("projection_matrix", camera.get_projection_matrix());
    m_model.render();
}

void GLVolume::SinkingContours::update()
{
    const int object_idx = m_parent.object_idx();
    const Model& model = GUI::wxGetApp().plater()->model();

    if (0 <= object_idx && object_idx < int(model.objects.size()) && m_parent.is_sinking() && !m_parent.is_below_printbed()) {
        const BoundingBoxf3& box = m_parent.transformed_convex_hull_bounding_box();
        if (!m_old_box.size().isApprox(box.size()) || m_old_box.min.z() != box.min.z()) {
            m_old_box = box;
            m_shift = Vec3d::Zero();

            const TriangleMesh& mesh = model.objects[object_idx]->volumes[m_parent.volume_idx()]->mesh();

            m_model.reset();
            GUI::GLModel::Geometry init_data;
            init_data.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3 };
            init_data.color = ColorRGBA::WHITE();
            unsigned int vertices_counter = 0;
            MeshSlicingParams slicing_params;
            slicing_params.trafo = m_parent.world_matrix();
            const Polygons polygons = union_(slice_mesh(mesh.its, 0.0f, slicing_params));
            for (const ExPolygon& expoly : diff_ex(expand(polygons, float(scale_(HalfWidth))), shrink(polygons, float(scale_(HalfWidth))))) {
                const std::vector<Vec3d> triangulation = triangulate_expolygon_3d(expoly);
                init_data.reserve_vertices(init_data.vertices_count() + triangulation.size());
                init_data.reserve_indices(init_data.indices_count() + triangulation.size());
                for (const Vec3d& v : triangulation) {
                    init_data.add_vertex((Vec3f)(v.cast<float>() + 0.015f * Vec3f::UnitZ())); // add a small positive z to avoid z-fighting
                    ++vertices_counter;
                    if (vertices_counter % 3 == 0)
                        init_data.add_triangle(vertices_counter - 3, vertices_counter - 2, vertices_counter - 1);
                }
            }
            m_model.init_from(std::move(init_data));
        }
        else
            m_shift = box.center() - m_old_box.center();
    }
    else
        m_model.reset();
}

ColorRGBA GLVolume::DISABLED_COLOR    = ColorRGBA::DARK_GRAY();
ColorRGBA GLVolume::SLA_SUPPORT_COLOR = ColorRGBA::LIGHT_GRAY();
ColorRGBA GLVolume::SLA_PAD_COLOR     = { 0.0f, 0.2f, 0.0f, 1.0f };
// BBS
ColorRGBA GLVolume::NEUTRAL_COLOR     = { 0.8f, 0.8f, 0.8f, 1.0f };
ColorRGBA GLVolume::UNPRINTABLE_COLOR = { 0.0f, 0.0f, 0.0f, 0.5f };

ColorRGBA GLVolume::MODEL_MIDIFIER_COL   = {1.0f, 1.0f, 0.0f, 0.6f};
ColorRGBA GLVolume::MODEL_NEGTIVE_COL    = {0.3f, 0.3f, 0.3f, 0.4f};
ColorRGBA GLVolume::SUPPORT_ENFORCER_COL = {0.3f, 0.3f, 1.0f, 0.4f};
ColorRGBA GLVolume::SUPPORT_BLOCKER_COL  = {1.0f, 0.3f, 0.3f, 0.4f};

ColorRGBA GLVolume::MODEL_HIDDEN_COL  = {0.f, 0.f, 0.f, 0.3f};

std::array<ColorRGBA, 5> GLVolume::MODEL_COLOR = { {
    { 1.0f, 1.0f, 0.0f, 1.f },
    { 1.0f, 0.5f, 0.5f, 1.f },
    { 0.5f, 1.0f, 0.5f, 1.f },
    { 0.5f, 0.5f, 1.0f, 1.f },
    { 1.0f, 1.0f, 0.0f, 1.f }
} };

void GLVolume::update_render_colors()
{
    GLVolume::DISABLED_COLOR    = GUI::ImGuiWrapper::from_ImVec4(RenderColor::colors[RenderCol_Model_Disable]);
    GLVolume::NEUTRAL_COLOR     = GUI::ImGuiWrapper::from_ImVec4(RenderColor::colors[RenderCol_Model_Neutral]);
    GLVolume::MODEL_COLOR[0]    = GUI::ImGuiWrapper::from_ImVec4(RenderColor::colors[RenderCol_Modifier]);
    GLVolume::MODEL_COLOR[1]    = GUI::ImGuiWrapper::from_ImVec4(RenderColor::colors[RenderCol_Negtive_Volume]);
    GLVolume::MODEL_COLOR[2]    = GUI::ImGuiWrapper::from_ImVec4(RenderColor::colors[RenderCol_Support_Enforcer]);
    GLVolume::MODEL_COLOR[3]    = GUI::ImGuiWrapper::from_ImVec4(RenderColor::colors[RenderCol_Support_Blocker]);
    GLVolume::UNPRINTABLE_COLOR = GUI::ImGuiWrapper::from_ImVec4(RenderColor::colors[RenderCol_Model_Unprintable]);

}

void GLVolume::load_render_colors()
{
    RenderColor::colors[RenderCol_Model_Disable]    = GUI::ImGuiWrapper::to_ImVec4(GLVolume::DISABLED_COLOR);
    RenderColor::colors[RenderCol_Model_Neutral]    = GUI::ImGuiWrapper::to_ImVec4(GLVolume::NEUTRAL_COLOR);
    RenderColor::colors[RenderCol_Modifier]         = GUI::ImGuiWrapper::to_ImVec4(GLVolume::MODEL_COLOR[0]);
    RenderColor::colors[RenderCol_Negtive_Volume]   = GUI::ImGuiWrapper::to_ImVec4(GLVolume::MODEL_COLOR[1]);
    RenderColor::colors[RenderCol_Support_Enforcer] = GUI::ImGuiWrapper::to_ImVec4(GLVolume::MODEL_COLOR[2]);
    RenderColor::colors[RenderCol_Support_Blocker]   = GUI::ImGuiWrapper::to_ImVec4(GLVolume::MODEL_COLOR[3]);
    RenderColor::colors[RenderCol_Model_Unprintable] = GUI::ImGuiWrapper::to_ImVec4(GLVolume::UNPRINTABLE_COLOR);
}

ColorRGBA GLVolume::brighten_color(const ColorRGBA& color, float multiplier)
{
    // Convert RGB to HSL, increase lightness, convert back

    float r = color.r(), g = color.g(), b = color.b();

    // RGB to HSL conversion
    float max_val = std::max({r, g, b});
    float min_val = std::min({r, g, b});
    float l = (max_val + min_val) / 2.0f;
    float h = 0.0f, s = 0.0f;

    if (max_val != min_val) {
        float delta = max_val - min_val;
        s = l > 0.5f ? delta / (2.0f - max_val - min_val) : delta / (max_val + min_val);

        if (max_val == r)
            h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if (max_val == g)
            h = (b - r) / delta + 2.0f;
        else
            h = (r - g) / delta + 4.0f;
        h /= 6.0f;
    }

    // Increase lightness by a fixed amount (0.25)
    // Ensures even saturated colors become visibly brighter
    l = std::min(l + 0.25f, 1.0f);

    // HSL to RGB conversion
    auto hue_to_rgb = [](float p, float q, float t) {
        if (t < 0.0f) t += 1.0f;
        if (t > 1.0f) t -= 1.0f;
        if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
        if (t < 1.0f / 2.0f) return q;
        if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
        return p;
    };

    if (s == 0.0f) {
        r = g = b = l; // achromatic (gray)
    } else {
        float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
        float p = 2.0f * l - q;
        r = hue_to_rgb(p, q, h + 1.0f / 3.0f);
        g = hue_to_rgb(p, q, h);
        b = hue_to_rgb(p, q, h - 1.0f / 3.0f);
    }

    return ColorRGBA(r, g, b, color.a());
}

GLVolume::GLVolume(float r, float g, float b, float a)
    : m_sla_shift_z(0.0)
    , m_sinking_contours(*this)
    // geometry_id == 0 -> invalid
    , geometry_id(std::pair<size_t, size_t>(0, 0))
    , extruder_id(0)
    , selected(false)
    , disabled(false)
    , printable(true)
    , visible(true)
    , is_active(true)
    , zoom_to_volumes(true)
    , shader_outside_printer_detection_enabled(false)
    , is_outside(false)
    , partly_inside(false)
    , hover(HS_None)
    , is_modifier(false)
    , slice_error(false)
    , is_wipe_tower(false)
    , is_extrusion_path(false)
    , force_transparent(false)
    , force_native_color(false)
    , force_neutral_color(false)
    , force_sinking_contours(false)
    , picking(false)
    , tverts_range(0, size_t(-1))
{
    color = { r, g, b, a };
    set_render_color(color);
    mmuseg_ts = 0;
}


// BBS
float GLVolume::explosion_ratio = 1.0;
float GLVolume::last_explosion_ratio = 1.0;

void GLVolume::set_render_color()
{
    bool outside = is_outside || is_below_printbed();

    if (force_native_color || force_neutral_color) {
#ifdef ENABBLE_OUTSIDE_COLOR
        if (outside && shader_outside_printer_detection_enabled)
            set_render_color(OUTSIDE_COLOR);
        else {
#endif
            if (force_native_color)
                set_render_color(color);
            else
                set_render_color(NEUTRAL_COLOR);
#ifdef ENABLE_OUTSIDE_COLOR
        }
#endif
    }
    else {
        /* BBS
        if (hover == HS_Select)
            set_render_color(HOVER_SELECT_COLOR);
        else if (hover == HS_Deselect)
            set_render_color(HOVER_DESELECT_COLOR);
        else if (selected)
            set_render_color(outside ? SELECTED_OUTSIDE_COLOR : SELECTED_COLOR);
        else if (disabled)
        */
        // Determine base color first
        ColorRGBA base_color;

        if (disabled) {
            base_color = DISABLED_COLOR;
        }
#ifdef ENABLE_OUTSIDE_COLOR
        else if (is_outside && shader_outside_printer_detection_enabled) {
            base_color = OUTSIDE_COLOR;
        }
#endif
        else {
            // to make black not too hard too see
            base_color = adjust_color_for_rendering(color);
        }

        // Apply selection brightening AFTER determining base color
        if (selected && !disabled) {
            set_render_color(brighten_color(base_color, 1.25f));
        }
        else {
            set_render_color(base_color);
        }
    }

    if (force_transparent) {
        if (color.a() < FullyTransparentMaterialThreshold) {
            render_color.a(FullTransparentModdifiedToFixAlpha);
        } else {
            render_color.a(color.a());
        }
    }

    //BBS set unprintable color
    if (!printable) {
        if (selected) {
            render_color = brighten_color(UNPRINTABLE_COLOR, 1.25f);
        } else {
            render_color = UNPRINTABLE_COLOR;
        }
    }

    //BBS set invisible color
    if (!visible) {
        render_color = MODEL_HIDDEN_COL;
    }
}

ColorRGBA color_from_model_volume(const ModelVolume& model_volume)
{
    ColorRGBA color;
    if (model_volume.is_negative_volume())
        return GLVolume::MODEL_NEGTIVE_COL;
    else if (model_volume.is_modifier())
#if ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        return GLVolume::MODEL_MIDIFIER_COL;
#else
		color = { 0.2f, 1.0f, 0.2f, 1.0f };
#endif // ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
    else if (model_volume.is_support_blocker())
        return GLVolume::SUPPORT_BLOCKER_COL;
    else if (model_volume.is_support_enforcer())
        return GLVolume::SUPPORT_ENFORCER_COL;
    return color;
}

Transform3d GLVolume::world_matrix() const
{
    Transform3d m = m_instance_transformation.get_matrix() * m_volume_transformation.get_matrix();
    Vec3d ofs2ass = m_offset_to_assembly * (GLVolume::explosion_ratio - 1.0);
    Vec3d volofs2obj = m_volume_transformation.get_offset() * (GLVolume::explosion_ratio - 1.0);

    m.translation()(2) += m_sla_shift_z;
    m.translate(ofs2ass + volofs2obj);
    return m;
}

bool GLVolume::is_left_handed() const
{
    const Vec3d &m1 = m_instance_transformation.get_mirror();
    const Vec3d &m2 = m_volume_transformation.get_mirror();
    return m1.x() * m1.y() * m1.z() * m2.x() * m2.y() * m2.z() < 0.;
}

const BoundingBoxf3& GLVolume::transformed_bounding_box() const
{
    if (!m_transformed_bounding_box.has_value() || last_explosion_ratio != explosion_ratio) {
        const BoundingBoxf3& box = bounding_box();
        assert(box.defined || box.min.x() >= box.max.x() || box.min.y() >= box.max.y() || box.min.z() >= box.max.z());
        std::optional<BoundingBoxf3>* trans_box = const_cast<std::optional<BoundingBoxf3>*>(&m_transformed_bounding_box);
        *trans_box = box.transformed(world_matrix());
        last_explosion_ratio = explosion_ratio;
    }
    return *m_transformed_bounding_box;
}

const BoundingBoxf3& GLVolume::transformed_convex_hull_bounding_box() const
{
    if (!m_transformed_convex_hull_bounding_box.has_value()) {
        std::optional<BoundingBoxf3>* trans_box = const_cast<std::optional<BoundingBoxf3>*>(&m_transformed_convex_hull_bounding_box);
        *trans_box = transformed_convex_hull_bounding_box(world_matrix());
    }
    return *m_transformed_convex_hull_bounding_box;
}

BoundingBoxf3 GLVolume::transformed_convex_hull_bounding_box(const Transform3d &trafo) const
{
	return (m_convex_hull && ! m_convex_hull->empty()) ?
		m_convex_hull->transformed_bounding_box(trafo) :
        bounding_box().transformed(trafo);
}

BoundingBoxf3 GLVolume::transformed_non_sinking_bounding_box(const Transform3d& trafo) const
{
    return GUI::wxGetApp().plater()->model().objects[object_idx()]->volumes[volume_idx()]->mesh().transformed_bounding_box(trafo, 0.0);
}

const BoundingBoxf3& GLVolume::transformed_non_sinking_bounding_box() const
{
    if (!m_transformed_non_sinking_bounding_box.has_value()) {
        std::optional<BoundingBoxf3>* trans_box = const_cast<std::optional<BoundingBoxf3>*>(&m_transformed_non_sinking_bounding_box);
        const Transform3d& trafo = world_matrix();
        *trans_box = transformed_non_sinking_bounding_box(trafo);
    }
    return *m_transformed_non_sinking_bounding_box;
}

void GLVolume::set_range(double min_z, double max_z)
{
    this->tverts_range.first = 0;
    this->tverts_range.second = this->model.indices_count();

    if (!this->print_zs.empty()) {
        // The Z layer range is specified.
        // First test whether the Z span of this object is not out of (min_z, max_z) completely.
        if (this->print_zs.front() > max_z || this->print_zs.back() < min_z)
            this->tverts_range.second = 0;
        else {
            // Then find the lowest layer to be displayed.
            size_t i = 0;
            for (; i < this->print_zs.size() && this->print_zs[i] < min_z; ++i);
            if (i == this->print_zs.size())
                // This shall not happen.
                this->tverts_range.second = 0;
            else {
                // Remember start of the layer.
                this->tverts_range.first = this->offsets[i];
                // Some layers are above $min_z. Which?
                for (; i < this->print_zs.size() && this->print_zs[i] <= max_z; ++i);
                if (i < this->print_zs.size())
                    this->tverts_range.second = this->offsets[i];
            }
        }
    }
}

void GLVolume::render()
{
    if (!is_active)
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

    ModelObjectPtrs &model_objects = GUI::wxGetApp().model().objects;
    std::vector<ColorRGBA> colors = GUI::wxGetApp().plater()->get_extruders_colors();

    simple_render(shader, model_objects, colors);
}

//BBS: add outline related logic
void GLVolume::render_with_outline(const GUI::Size& cnv_size)
{
    if (!is_active)
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

    ModelObjectPtrs &model_objects = GUI::wxGetApp().model().objects;
    std::vector<ColorRGBA> colors = GUI::wxGetApp().plater()->get_extruders_colors();

    const GUI::OpenGLManager::EFramebufferType framebuffers_type = GUI::OpenGLManager::get_framebuffers_type();
    if (framebuffers_type == GUI::OpenGLManager::EFramebufferType::Unknown) {
        // No supported, degrade to normal rendering
        simple_render(shader, model_objects, colors);
        return;
    }

    // 1st. render pass, render the model into a separate render target that has only depth buffer
    GLuint depth_fbo   = 0;
    GLuint depth_tex = 0;
    if (framebuffers_type == GUI::OpenGLManager::EFramebufferType::Arb) {
        glsafe(::glGenFramebuffers(1, &depth_fbo));
        glsafe(::glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo));

        glActiveTexture(GL_TEXTURE0);
        glsafe(::glGenTextures(1, &depth_tex));
        glsafe(::glBindTexture(GL_TEXTURE_2D, depth_tex));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        glsafe(::glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, cnv_size.get_width(), cnv_size.get_height(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));

        glsafe(::glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0));
    } else {
        glsafe(::glGenFramebuffersEXT(1, &depth_fbo));
        glsafe(::glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, depth_fbo));

        glActiveTexture(GL_TEXTURE0);
        glsafe(::glGenTextures(1, &depth_tex));
        glsafe(::glBindTexture(GL_TEXTURE_2D, depth_tex));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        glsafe(::glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, cnv_size.get_width(), cnv_size.get_height(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr));

        glsafe(::glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, depth_tex, 0));
    }
    glsafe(::glClear(GL_DEPTH_BUFFER_BIT));
    if (tverts_range == std::make_pair<size_t, size_t>(0, -1))
        model.render(shader);
    else
        model.render(this->tverts_range, shader);
    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));

    // 2nd. render pass, just a normal render with the depth buffer passed as a texture
    if (framebuffers_type == GUI::OpenGLManager::EFramebufferType::Arb) {
        glsafe(::glBindFramebuffer(GL_FRAMEBUFFER, 0));
    } else if (framebuffers_type == GUI::OpenGLManager::EFramebufferType::Ext) {
        glsafe(::glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));
    }
    shader->set_uniform("is_outline", true);
    shader->set_uniform("screen_size", Vec2f{cnv_size.get_width(), cnv_size.get_height()});
    glActiveTexture(GL_TEXTURE0);
    glsafe(::glBindTexture(GL_TEXTURE_2D, depth_tex));
    shader->set_uniform("depth_tex", 0);
    simple_render(shader, model_objects, colors);

    // Some clean up to do
    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
    shader->set_uniform("is_outline", false);
    if (framebuffers_type == GUI::OpenGLManager::EFramebufferType::Arb) {
        glsafe(::glBindFramebuffer(GL_FRAMEBUFFER, 0));
        if (depth_fbo != 0)
            glsafe(::glDeleteFramebuffers(1, &depth_fbo));
    } else if (framebuffers_type == GUI::OpenGLManager::EFramebufferType::Ext) {
        glsafe(::glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));
        if (depth_fbo != 0)
            glsafe(::glDeleteFramebuffersEXT(1, &depth_fbo));
    }
    if (depth_tex != 0)
        glsafe(::glDeleteTextures(1, &depth_tex));
}

//BBS add render for simple case
void GLVolume::simple_render(GLShaderProgram* shader,
                             ModelObjectPtrs& model_objects,
                             std::vector<ColorRGBA>& extruder_colors,
                             bool ban_light,
                             bool suppress_texture_preview_base)
{
    if (this->is_left_handed())
        glFrontFace(GL_CW);
    glsafe(::glCullFace(GL_BACK));

    bool color_volume = false;
    ModelObject* model_object = nullptr;
    ModelVolume* model_volume = nullptr;
    unsigned int base_filament_id = 0;
    bool base_uses_texture_preview = false;
    bool use_original_mesh_texture_preview = false;
    bool texture_preview_base_suppressed = false;
    do {
        if ((!printable) || object_idx() >= model_objects.size())
            break;
        model_object = model_objects[object_idx()];

        if (volume_idx() >=  model_object->volumes.size())
            break;
        model_volume = model_object->volumes[volume_idx()];
        const size_t num_physical = std::max(0, GUI::wxGetApp().filaments_cnt());
        const TextureMappingManager *texture_mgr = GUI::wxGetApp().preset_bundle != nullptr ?
            &GUI::wxGetApp().preset_bundle->texture_mapping_zones : nullptr;
        base_filament_id = model_volume->extruder_id() > 0 ? unsigned(model_volume->extruder_id()) : 0u;
        const bool has_mmu_segmentation = !model_volume->mmu_segmentation_facets.empty();
        base_uses_texture_preview = filament_state_uses_texture_preview(base_filament_id, num_physical, texture_mgr);
        const bool base_uses_surface_gradient_preview =
            filament_state_uses_surface_gradient_preview(base_filament_id, num_physical, texture_mgr);
        const bool base_uses_image_texture_preview = base_uses_texture_preview && !base_uses_surface_gradient_preview;
        const bool base_uses_halftone_texture_preview =
            texture_preview_halftone_simulation_enabled_for_filament(base_filament_id, num_physical, texture_mgr);
        const bool has_surface_gradient_preview_state =
            base_uses_surface_gradient_preview ||
            (has_mmu_segmentation &&
             texture_preview_used_states_have_surface_gradient(model_volume->mmu_segmentation_facets.get_data().used_states,
                                                               num_physical,
                                                               texture_mgr));
        const bool has_texture_mapping_color_data =
            model_volume_has_texture_mapping_color_preview_data(*model_volume);
        const bool has_texture_mapping_color_preview_data =
            base_uses_texture_preview && has_texture_mapping_color_data;
        const bool has_texture_preview_data = model_volume_has_texture_preview_data(*model_volume);
        const bool has_vertex_color_preview_data = model_volume_has_vertex_color_preview_data(*model_volume);
        const Transform3d preview_world_matrix = this->world_matrix();
        use_original_mesh_texture_preview =
            !has_mmu_segmentation &&
            !has_texture_mapping_color_preview_data &&
            base_uses_image_texture_preview &&
            !base_uses_halftone_texture_preview &&
            model_volume_has_complete_texture_preview_data(*model_volume) &&
            GUI::GLModel::Geometry::has_tex_coord(model.get_geometry().format);
        if (!has_mmu_segmentation && !base_uses_texture_preview)
        {
            mmuseg_models.clear();
            mmuseg_texture_preview_models.clear();
            mmuseg_texture_preview_colors.clear();
            mmuseg_texture_preview_filament_ids.clear();
            mmuseg_vertex_color_preview_models.clear();
            mmuseg_vertex_color_preview_colors.clear();
            mmuseg_vertex_color_preview_filament_ids.clear();
            mmuseg_texture_preview.reset();
            mmuseg_texture_preview_signature = 0;
            mmuseg_texture_preview_visual_signature = 0;
            mmuseg_ts = model_volume->mmu_segmentation_facets.timestamp();
            break;
        }

        color_volume = has_mmu_segmentation;
        const std::vector<bool> *texture_preview_used_states = has_mmu_segmentation ?
            &model_volume->mmu_segmentation_facets.get_data().used_states : nullptr;
        size_t preview_visual_signature = texture_preview_model_settings_signature(num_physical,
                                                                                   texture_mgr,
                                                                                   base_filament_id,
                                                                                   texture_preview_used_states,
                                                                                   has_texture_preview_data,
                                                                                   has_vertex_color_preview_data,
                                                                                   has_texture_mapping_color_data);
        preview_visual_signature ^= size_t(base_filament_id) + 0x9e3779b97f4a7c15ull +
                                    (preview_visual_signature << 6) + (preview_visual_signature >> 2);
        preview_visual_signature ^= texture_preview_simulation_generation_signature() + 0x9e3779b97f4a7c15ull +
                                    (preview_visual_signature << 6) + (preview_visual_signature >> 2);
        preview_visual_signature ^= model_volume_texture_preview_signature(*model_volume) + 0x9e3779b97f4a7c15ull +
                                    (preview_visual_signature << 6) + (preview_visual_signature >> 2);
        preview_visual_signature ^= model_volume->imported_vertex_colors_rgba.id().id + 0x9e3779b97f4a7c15ull +
                                    (preview_visual_signature << 6) + (preview_visual_signature >> 2);
        preview_visual_signature ^= model_volume->imported_vertex_colors_rgba.size() + 0x9e3779b97f4a7c15ull +
                                    (preview_visual_signature << 6) + (preview_visual_signature >> 2);
        preview_visual_signature ^= reinterpret_cast<size_t>(model_volume->imported_vertex_colors_rgba.data()) + 0x9e3779b97f4a7c15ull +
                                    (preview_visual_signature << 6) + (preview_visual_signature >> 2);
        preview_visual_signature ^= model_volume_texture_mapping_color_preview_signature(*model_volume) + 0x9e3779b97f4a7c15ull +
                                    (preview_visual_signature << 6) + (preview_visual_signature >> 2);
        if (has_surface_gradient_preview_state || has_texture_preview_data) {
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    preview_visual_signature ^= std::hash<int>{}(int(std::lround(preview_world_matrix(row, col) * 1000000.0))) +
                                                0x9e3779b97f4a7c15ull + (preview_visual_signature << 6) +
                                                (preview_visual_signature >> 2);
                }
            }
        }
        if (model_volume->mmu_segmentation_facets.timestamp() != mmuseg_ts ||
            preview_visual_signature != mmuseg_texture_preview_visual_signature) {
            mmuseg_models.clear();
            mmuseg_texture_preview_models.clear();
            mmuseg_texture_preview_colors.clear();
            mmuseg_texture_preview_filament_ids.clear();
            mmuseg_vertex_color_preview_models.clear();
            mmuseg_vertex_color_preview_colors.clear();
            mmuseg_vertex_color_preview_filament_ids.clear();

            std::vector<std::vector<TriangleSelector::FacetStateTriangle>> triangles_per_type;
            bool has_texture_preview_state = base_uses_texture_preview;
            if (has_mmu_segmentation) {
                std::vector<indexed_triangle_set> its_per_color;
                model_volume->mmu_segmentation_facets.get_facets(*model_volume, its_per_color);
                mmuseg_models.resize(its_per_color.size());
                for (int idx = 0; idx < its_per_color.size(); idx++) {
                    mmuseg_models[idx].init_from(its_per_color[idx]);
                    if (!its_per_color[idx].indices.empty())
                        has_texture_preview_state = has_texture_preview_state ||
                            filament_state_uses_texture_preview(unsigned(idx), num_physical, texture_mgr);
                }
                model_volume->mmu_segmentation_facets.get_facet_triangles(*model_volume, triangles_per_type);
            } else {
                triangles_per_type.resize(1);
                triangles_per_type[0] = build_full_mesh_texture_preview_triangles(*model_volume);
            }

            std::vector<ColorRGBA> state_colors;
            const int extruder_id = model_volume->extruder_id();
            const ColorRGBA fallback_color = extruder_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : extruder_colors.front();
            state_colors.emplace_back(extruder_id > 0 && size_t(extruder_id - 1) < extruder_colors.size() ?
                                      extruder_colors[size_t(extruder_id - 1)] :
                                      fallback_color);
            state_colors.insert(state_colors.end(), extruder_colors.begin(), extruder_colors.end());

            const bool has_active_texture_mapping_color_preview_data =
                has_texture_preview_state && model_volume_has_texture_mapping_color_preview_data(*model_volume);
            if (has_texture_preview_state) {
                build_mmu_vertex_color_preview_models(*model_volume,
                                                      triangles_per_type,
                                                      state_colors,
                                                      base_filament_id,
                                                      num_physical,
                                                      texture_mgr,
                                                      preview_world_matrix,
                                                      mmuseg_vertex_color_preview_models,
                                                      mmuseg_vertex_color_preview_colors,
                                                      mmuseg_vertex_color_preview_filament_ids);
            }
            if (has_texture_preview_state && !use_original_mesh_texture_preview && !has_active_texture_mapping_color_preview_data && has_texture_preview_data) {
                build_mmu_texture_preview_models(*model_volume,
                                                 triangles_per_type,
                                                 state_colors,
                                                 base_filament_id,
                                                 num_physical,
                                                 texture_mgr,
                                                 mmuseg_texture_preview_models,
                                                 mmuseg_texture_preview_colors,
                                                 mmuseg_texture_preview_filament_ids,
                                                 &mmuseg_vertex_color_preview_filament_ids);
            }
            mmuseg_ts = model_volume->mmu_segmentation_facets.timestamp();
            mmuseg_texture_preview_visual_signature = preview_visual_signature;
        }
        texture_preview_base_suppressed = suppress_texture_preview_base &&
            !picking &&
            base_uses_texture_preview &&
            (use_original_mesh_texture_preview || !mmuseg_texture_preview_models.empty() || !mmuseg_vertex_color_preview_models.empty());
    } while (0);

    if (texture_preview_base_suppressed) {
        if (this->is_left_handed())
            glFrontFace(GL_CCW);
        return;
    } else if (color_volume && !picking) {
        // when force_transparent, we need to keep the alpha
        if (force_native_color && render_color.is_transparent()) {
            for (auto &extruder_color : extruder_colors)
                extruder_color.a(render_color.a());
        }

        if (mmuseg_models.empty()) {
            if (tverts_range == std::make_pair<size_t, size_t>(0, -1))
                model.render(shader);
            else
                model.render(this->tverts_range, shader);
        } else {
            for (int idx = 0; idx < mmuseg_models.size(); idx++) {
                GUI::GLModel &m = mmuseg_models[idx];
                if (!m.is_initialized())
                    continue;

                if (shader) {
                    if (idx == 0) {
                        int extruder_id = model_volume->extruder_id();
                        ColorRGBA new_color = extruder_id > 0 && size_t(extruder_id - 1) < extruder_colors.size() ?
                            adjust_color_for_rendering(extruder_colors[size_t(extruder_id - 1)]) :
                            (extruder_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : adjust_color_for_rendering(extruder_colors.front()));
                        if (ban_light) {
                            new_color[3] = (255 - std::max(0, extruder_id - 1))/255.0f;
                        }
                        m.set_color(new_color);
                    }
                    else {
                        if (idx <= extruder_colors.size()) {
                            ColorRGBA new_color = adjust_color_for_rendering(extruder_colors[idx - 1]);
                            if (ban_light) {
                                new_color[3] = (255 - (idx - 1))/255.0f;
                            }
                            m.set_color(new_color);
                        }
                        else {
                            ColorRGBA new_color = extruder_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : adjust_color_for_rendering(extruder_colors[0]);
                            if (ban_light) {
                                new_color[3] = (255 - 0) / 255.0f;
                            }
                            m.set_color(new_color);
                        }
                    }
                }
                if (tverts_range == std::make_pair<size_t, size_t>(0, -1))
                    m.render(shader);
                else
                    m.render(this->tverts_range, shader);
            }
        }
    } else {
        if (tverts_range == std::make_pair<size_t, size_t>(0, -1))
            model.render(shader);
        else
            model.render(this->tverts_range, shader);
    }
    if (this->is_left_handed())
        glFrontFace(GL_CCW);
}

void GLVolume::render_mmu_texture_preview(const Transform3d &view_matrix,
                                          const Transform3d &projection_matrix,
                                          const std::array<float, 2> &z_range,
                                          const std::array<float, 4> &clipping_plane,
                                          int print_volume_type,
                                          const std::array<float, 4> &print_volume_xy,
                                          const std::array<float, 2> &print_volume_z,
                                          bool opaque,
                                          const SurfaceGradientAnchorResolver *surface_gradient_anchor_resolver,
                                          const SurfaceGradientAnchorRadiusResolver *surface_gradient_anchor_radius_resolver)
{
    if (picking || !printable || object_idx() < 0 || volume_idx() < 0)
        return;

    ModelObjectPtrs &model_objects = GUI::wxGetApp().model().objects;
    if (size_t(object_idx()) >= model_objects.size())
        return;

    const ModelObject *model_object = model_objects[size_t(object_idx())];
    if (model_object == nullptr || size_t(volume_idx()) >= model_object->volumes.size())
        return;

    const ModelVolume *model_volume = model_object->volumes[size_t(volume_idx())];
    if (model_volume == nullptr)
        return;

    const size_t num_physical = std::max(0, GUI::wxGetApp().filaments_cnt());
    const TextureMappingManager *texture_mgr = GUI::wxGetApp().preset_bundle != nullptr ?
        &GUI::wxGetApp().preset_bundle->texture_mapping_zones : nullptr;
    const unsigned int base_filament_id = model_volume->extruder_id() > 0 ? unsigned(model_volume->extruder_id()) : 0u;
    const bool base_uses_texture_preview = filament_state_uses_texture_preview(base_filament_id, num_physical, texture_mgr);
    const bool base_uses_surface_gradient_preview =
        filament_state_uses_surface_gradient_preview(base_filament_id, num_physical, texture_mgr);
    const bool base_uses_image_texture_preview = base_uses_texture_preview && !base_uses_surface_gradient_preview;
    const bool base_uses_halftone_texture_preview =
        texture_preview_halftone_simulation_enabled_for_filament(base_filament_id, num_physical, texture_mgr);

    const bool has_mmu_segmentation = !model_volume->mmu_segmentation_facets.empty();
    const bool has_texture_mapping_color_preview_data = model_volume_has_texture_mapping_color_preview_data(*model_volume);
    const bool use_original_mesh_texture_preview =
        !has_mmu_segmentation &&
        !has_texture_mapping_color_preview_data &&
        base_uses_image_texture_preview &&
        !base_uses_halftone_texture_preview &&
        model_volume_has_complete_texture_preview_data(*model_volume) &&
        GUI::GLModel::Geometry::has_tex_coord(model.get_geometry().format);

    if (!use_original_mesh_texture_preview && mmuseg_texture_preview_models.empty()) {
        mmuseg_texture_preview.reset();
        mmuseg_texture_preview_signature = 0;
    }

    if (!use_original_mesh_texture_preview && mmuseg_texture_preview_models.empty() && mmuseg_vertex_color_preview_models.empty())
        return;

    if (GUI::wxGetApp().plater() == nullptr)
        return;

    if (this->is_left_handed())
        glFrontFace(GL_CW);
    glsafe(::glCullFace(GL_BACK));

    const Transform3d model_matrix = this->world_matrix();
    const std::vector<ColorRGBA> extruder_colors = GUI::wxGetApp().plater()->get_extruders_colors();
    auto adjusted_preview_colors = [this, &extruder_colors](const std::vector<unsigned int> &filament_ids, const std::vector<ColorRGBA> &colors) {
        std::vector<ColorRGBA> preview_colors = colors;
        for (size_t idx = 0; idx < preview_colors.size(); ++idx) {
            ColorRGBA &preview_color = preview_colors[idx];
            const unsigned int filament_id = idx < filament_ids.size() ? filament_ids[idx] : 0u;
            if (filament_id > 0 && size_t(filament_id - 1) < extruder_colors.size())
                preview_color = extruder_colors[size_t(filament_id - 1)];
            preview_color = adjust_color_for_rendering(preview_color);
            if (force_native_color && render_color.is_transparent())
                preview_color.a(render_color.a());
        }
        return preview_colors;
    };

    if ((use_original_mesh_texture_preview || !mmuseg_texture_preview_models.empty()) &&
        ensure_model_volume_texture_preview(*model_volume, mmuseg_texture_preview, mmuseg_texture_preview_signature)) {
        if (use_original_mesh_texture_preview) {
            const int extruder_id = model_volume->extruder_id();
            const ColorRGBA fallback_color = extruder_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : extruder_colors.front();
            ColorRGBA original_mesh_preview_color = extruder_id > 0 && size_t(extruder_id - 1) < extruder_colors.size() ?
                extruder_colors[size_t(extruder_id - 1)] :
                fallback_color;
            original_mesh_preview_color = adjust_color_for_rendering(original_mesh_preview_color);
            if (force_native_color && render_color.is_transparent())
                original_mesh_preview_color.a(render_color.a());
            render_model_texture_preview_model(model,
                                               original_mesh_preview_color,
                                               base_filament_id,
                                               num_physical,
                                               texture_mgr,
                                               *model_volume,
                                               mmuseg_texture_preview,
                                               model_matrix,
                                               view_matrix,
                                               projection_matrix,
                                               z_range,
                                               clipping_plane,
                                               this->tverts_range,
                                               print_volume_type,
                                               print_volume_xy,
                                               print_volume_z,
                                               opaque);
        } else {
            render_model_texture_preview_models(mmuseg_texture_preview_models,
                                                adjusted_preview_colors(mmuseg_texture_preview_filament_ids, mmuseg_texture_preview_colors),
                                                mmuseg_texture_preview_filament_ids,
                                                num_physical,
                                                texture_mgr,
                                                *model_volume,
                                                mmuseg_texture_preview,
                                                model_matrix,
                                                view_matrix,
                                                projection_matrix,
                                                z_range,
                                                clipping_plane,
                                                print_volume_type,
                                                print_volume_xy,
                                                print_volume_z,
                                                opaque);
        }
    }

    if (!mmuseg_vertex_color_preview_models.empty()) {
        render_model_vertex_color_preview_models(mmuseg_vertex_color_preview_models,
                                                 adjusted_preview_colors(mmuseg_vertex_color_preview_filament_ids, mmuseg_vertex_color_preview_colors),
                                                 mmuseg_vertex_color_preview_filament_ids,
                                                 num_physical,
                                                 texture_mgr,
                                                 model_matrix,
                                                 view_matrix,
                                                 projection_matrix,
                                                 z_range,
                                                 clipping_plane,
                                                 print_volume_type,
                                                 print_volume_xy,
                                                 print_volume_z,
                                                 opaque,
                                                 model_volume,
                                                 surface_gradient_anchor_resolver,
                                                 surface_gradient_anchor_radius_resolver);
    }

    if (this->is_left_handed())
        glFrontFace(GL_CCW);
}

void GLVolume::invalidate_texture_mapping_preview()
{
    mmuseg_models.clear();
    mmuseg_texture_preview_models.clear();
    mmuseg_texture_preview_colors.clear();
    mmuseg_texture_preview_filament_ids.clear();
    mmuseg_vertex_color_preview_models.clear();
    mmuseg_vertex_color_preview_colors.clear();
    mmuseg_vertex_color_preview_filament_ids.clear();
    mmuseg_texture_preview.reset();
    mmuseg_texture_preview_signature = 0;
    mmuseg_texture_preview_visual_signature = 0;
    mmuseg_ts = 0;
}

bool GLVolume::is_sla_support() const { return this->composite_id.volume_id == -int(slaposSupportTree); }
bool GLVolume::is_sla_pad() const { return this->composite_id.volume_id == -int(slaposPad); }

bool GLVolume::is_sinking() const
{
    if (is_modifier || GUI::wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptSLA)
        return false;
    const BoundingBoxf3& box = transformed_convex_hull_bounding_box();
    return box.min.z() < SINKING_Z_THRESHOLD && box.max.z() >= SINKING_Z_THRESHOLD;
}

bool GLVolume::is_below_printbed() const
{
    return transformed_convex_hull_bounding_box().max.z() < 0.0;
}

void GLVolume::render_sinking_contours()
{
    m_sinking_contours.render();
}

static constexpr float prime_tower_preview_epsilon = 1e-6f;
static constexpr float prime_tower_preview_offset = 0.001f;
static constexpr float prime_tower_preview_coord_epsilon = 1e-4f;

template<class Points>
float prime_tower_preview_anchor_distance(const Points &points, const Vec2f &center, float angle_deg)
{
    float travelled = 0.f;
    float fallback_distance = 0.f;
    float fallback_dist = std::numeric_limits<float>::max();
    float best_distance = std::numeric_limits<float>::max();
    float best_projection = -std::numeric_limits<float>::max();
    const float angle = angle_deg * float(M_PI / 180.);
    const Vec2f ray_dir(std::cos(angle), std::sin(angle));

    for (size_t i = 0; i < points.size(); ++i) {
        const Vec2f a = points[i];
        const Vec2f b = points[(i + 1) % points.size()];
        const Vec2f delta = b - a;
        const float len = delta.norm();
        if (len <= prime_tower_preview_epsilon)
            continue;

        const Vec2f from_center = a - center;
        const float denom = ray_dir.x() * delta.y() - ray_dir.y() * delta.x();
        if (std::abs(denom) > prime_tower_preview_epsilon) {
            const float projection = (from_center.x() * delta.y() - from_center.y() * delta.x()) / denom;
            const float segment_t = (from_center.x() * ray_dir.y() - from_center.y() * ray_dir.x()) / denom;
            if (projection >= -prime_tower_preview_epsilon &&
                segment_t >= -prime_tower_preview_epsilon &&
                segment_t <= 1.f + prime_tower_preview_epsilon &&
                projection > best_projection) {
                best_projection = projection;
                best_distance = travelled + len * std::clamp(segment_t, 0.f, 1.f);
            }
        } else if (std::abs(from_center.x() * ray_dir.y() - from_center.y() * ray_dir.x()) <= prime_tower_preview_epsilon) {
            const float projection_a = from_center.dot(ray_dir);
            const float projection_b = (b - center).dot(ray_dir);
            if (projection_a >= -prime_tower_preview_epsilon || projection_b >= -prime_tower_preview_epsilon) {
                const bool use_b = projection_b > projection_a;
                const float projection = use_b ? projection_b : projection_a;
                if (projection > best_projection) {
                    best_projection = projection;
                    best_distance = travelled + (use_b ? len : 0.f);
                }
            }
        }

        const float fallback_t = std::clamp((center - a).dot(delta) / (len * len), 0.f, 1.f);
        const Vec2f fallback_point = a + delta * fallback_t - center;
        const float projection = fallback_point.dot(ray_dir);
        const float perpendicular = fallback_point.x() * ray_dir.y() - fallback_point.y() * ray_dir.x();
        const float fallback_score = perpendicular * perpendicular + (projection < 0.f ? projection * projection : 0.f);
        if (fallback_score < fallback_dist - prime_tower_preview_epsilon) {
            fallback_dist = fallback_score;
            fallback_distance = travelled + len * fallback_t;
        }
        travelled += len;
    }

    return best_projection > -std::numeric_limits<float>::max() ? best_distance : fallback_distance;
}

float prime_tower_preview_preserved_texture_u(float u,
                                              bool preserve_aspect_ratio,
                                              unsigned int image_width,
                                              unsigned int image_height,
                                              float surface_width,
                                              float surface_height)
{
    if (!preserve_aspect_ratio || image_width == 0 || image_height == 0 || surface_width <= prime_tower_preview_epsilon ||
        surface_height <= prime_tower_preview_epsilon)
        return u;

    const float image_aspect = float(image_width) / float(image_height);
    const float target_aspect = surface_width / surface_height;
    if (!std::isfinite(image_aspect) || !std::isfinite(target_aspect) || image_aspect <= prime_tower_preview_epsilon ||
        target_aspect <= prime_tower_preview_epsilon || image_aspect <= target_aspect + prime_tower_preview_epsilon)
        return u;

    const float visible_width = std::clamp(target_aspect / image_aspect, 0.f, 1.f);
    return std::clamp(0.5f * (1.f - visible_width) + u * visible_width, 0.f, 1.f);
}

float prime_tower_preview_preserved_source_v(float source_v,
                                             bool preserve_aspect_ratio,
                                             unsigned int image_width,
                                             unsigned int image_height,
                                             float surface_width,
                                             float surface_height)
{
    if (!preserve_aspect_ratio || image_width == 0 || image_height == 0 || surface_width <= prime_tower_preview_epsilon ||
        surface_height <= prime_tower_preview_epsilon)
        return source_v;

    const float image_aspect = float(image_width) / float(image_height);
    const float target_aspect = surface_width / surface_height;
    if (!std::isfinite(image_aspect) || !std::isfinite(target_aspect) || image_aspect <= prime_tower_preview_epsilon ||
        target_aspect <= prime_tower_preview_epsilon || target_aspect <= image_aspect + prime_tower_preview_epsilon)
        return source_v;

    const float visible_height = std::clamp(image_aspect / target_aspect, 0.f, 1.f);
    return std::clamp(source_v * visible_height, 0.f, 1.f);
}

float prime_tower_preview_texture_v(float z,
                                    float texture_z_min,
                                    float texture_z_max,
                                    bool preserve_aspect_ratio,
                                    unsigned int image_width,
                                    unsigned int image_height,
                                    float surface_width)
{
    const float v = texture_z_max > texture_z_min + prime_tower_preview_epsilon ?
        std::clamp((z - texture_z_min) / (texture_z_max - texture_z_min), 0.f, 1.f) :
        0.f;
    const float surface_height = texture_z_max > texture_z_min + prime_tower_preview_epsilon ? texture_z_max - texture_z_min : 0.f;
    return 1.f - prime_tower_preview_preserved_source_v(v,
                                                        preserve_aspect_ratio,
                                                        image_width,
                                                        image_height,
                                                        surface_width,
                                                        surface_height);
}

template<class Points>
float prime_tower_preview_anchor_angle(const Points &points, float angle_deg)
{
    Vec2f min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    for (const Vec2f &point : points) {
        min_pt.x() = std::min(min_pt.x(), point.x());
        min_pt.y() = std::min(min_pt.y(), point.y());
        max_pt.x() = std::max(max_pt.x(), point.x());
        max_pt.y() = std::max(max_pt.y(), point.y());
    }
    float angle = std::clamp(angle_deg, 0.f, 360.f);
    if (max_pt.y() - min_pt.y() > max_pt.x() - min_pt.x() + prime_tower_preview_epsilon)
        angle += 90.f;
    return angle >= 360.f ? angle - 360.f : angle;
}

GUI::GLModel::Geometry prime_tower_image_preview_geometry(float width,
                                                          float depth,
                                                          float height,
                                                          float angle_offset_deg,
                                                          float texture_z_min,
                                                          float texture_z_max,
                                                          int image_slot,
                                                          bool preserve_aspect_ratio,
                                                          unsigned int image_width,
                                                          unsigned int image_height)
{
    GUI::GLModel::Geometry data;
    data.format = {GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3T2};

    if (width <= prime_tower_preview_epsilon || depth <= prime_tower_preview_epsilon || height <= prime_tower_preview_epsilon)
        return data;

    const std::array<Vec2f, 4> points = {Vec2f(0.f, 0.f), Vec2f(width, 0.f), Vec2f(width, depth), Vec2f(0.f, depth)};
    const std::array<float, 4> distances = {0.f, width, width + depth, 2.f * width + depth};
    const float total_length = 2.f * (width + depth);
    const float texture_surface_width = image_slot == 0 ? total_length : 0.5f * total_length;
    const float texture_surface_height =
        texture_z_max > texture_z_min + prime_tower_preview_epsilon ? texture_z_max - texture_z_min : 0.f;
    const float anchor_distance = prime_tower_preview_anchor_distance(points,
                                                                      Vec2f(width * 0.5f, depth * 0.5f),
                                                                      prime_tower_preview_anchor_angle(points, angle_offset_deg));

    std::vector<float> z_levels = {0.f, height};
    if (texture_z_min > prime_tower_preview_epsilon && texture_z_min < height - prime_tower_preview_epsilon)
        z_levels.emplace_back(texture_z_min);
    if (texture_z_max > prime_tower_preview_epsilon && texture_z_max < height - prime_tower_preview_epsilon)
        z_levels.emplace_back(texture_z_max);
    std::sort(z_levels.begin(), z_levels.end());
    z_levels.erase(std::unique(z_levels.begin(), z_levels.end(), [](float lhs, float rhs) {
        return std::abs(lhs - rhs) <= prime_tower_preview_epsilon;
    }), z_levels.end());

    data.reserve_vertices(16 * points.size() * (z_levels.size() - 1));
    data.reserve_indices(24 * points.size() * (z_levels.size() - 1));

    auto texture_u = [anchor_distance,
                      total_length,
                      image_slot,
                      preserve_aspect_ratio,
                      image_width,
                      image_height,
                      texture_surface_width,
                      texture_surface_height](float distance, float mid_distance) {
        const float raw_u = (distance - anchor_distance) / total_length;
        const float mid_raw_u = (mid_distance - anchor_distance) / total_length;
        const float base = std::floor(mid_raw_u);
        if (image_slot == 0)
            return prime_tower_preview_preserved_texture_u(std::clamp(raw_u - base, 0.f, 1.f),
                                                           preserve_aspect_ratio,
                                                           image_width,
                                                           image_height,
                                                           texture_surface_width,
                                                           texture_surface_height);
        const float u = image_slot == 1 ?
            std::clamp(2.f * (raw_u - base), 0.f, 1.f) :
            std::clamp(2.f * (raw_u - base - 0.5f), 0.f, 1.f);
        return prime_tower_preview_preserved_texture_u(u,
                                                       preserve_aspect_ratio,
                                                       image_width,
                                                       image_height,
                                                       texture_surface_width,
                                                       texture_surface_height);
    };

    for (size_t side_idx = 0; side_idx < points.size(); ++side_idx) {
        const size_t next_idx = (side_idx + 1) % points.size();
        const Vec2f a = points[side_idx];
        const Vec2f b = points[next_idx];
        const Vec2f delta = b - a;
        const float len = delta.norm();
        if (len <= prime_tower_preview_epsilon)
            continue;

        const Vec2f dir = delta / len;
        const Vec2f outward(dir.y(), -dir.x());
        const Vec3f normal(outward.x(), outward.y(), 0.f);
        const float side_start = distances[side_idx];
        const float side_end = next_idx == 0 ? total_length : distances[next_idx];

        std::vector<float> cuts = {side_start, side_end};
        const int first = int(std::floor((side_start - anchor_distance) / total_length)) - 1;
        const int last = int(std::ceil((side_end - anchor_distance) / total_length)) + 1;
        for (int wrap = first; wrap <= last; ++wrap) {
            const float wrap_cut = anchor_distance + total_length * float(wrap);
            if (wrap_cut > side_start + prime_tower_preview_epsilon && wrap_cut < side_end - prime_tower_preview_epsilon)
                cuts.emplace_back(wrap_cut);
            if (image_slot != 0) {
                const float half_cut = anchor_distance + total_length * (float(wrap) + 0.5f);
                if (half_cut > side_start + prime_tower_preview_epsilon && half_cut < side_end - prime_tower_preview_epsilon)
                    cuts.emplace_back(half_cut);
            }
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end(), [](float lhs, float rhs) {
            return std::abs(lhs - rhs) <= prime_tower_preview_epsilon;
        }), cuts.end());

        for (size_t cut_idx = 0; cut_idx + 1 < cuts.size(); ++cut_idx) {
            const float d0 = cuts[cut_idx];
            const float d1 = cuts[cut_idx + 1];
            if (d1 - d0 <= prime_tower_preview_epsilon)
                continue;

            if (image_slot != 0) {
                const float mid_raw_u = (0.5f * (d0 + d1) - anchor_distance) / total_length;
                const float mid_u = mid_raw_u - std::floor(mid_raw_u);
                if ((mid_u >= 0.5f ? 2 : 1) != image_slot)
                    continue;
            }

            const float t0 = std::clamp((d0 - side_start) / len, 0.f, 1.f);
            const float t1 = std::clamp((d1 - side_start) / len, 0.f, 1.f);
            const Vec2f p0 = a + delta * t0 + outward * prime_tower_preview_offset;
            const Vec2f p1 = a + delta * t1 + outward * prime_tower_preview_offset;
            const float mid_distance = 0.5f * (d0 + d1);
            const float u0 = texture_u(d0, mid_distance);
            const float u1 = texture_u(d1, mid_distance);

            for (size_t z_idx = 0; z_idx + 1 < z_levels.size(); ++z_idx) {
                const float z0 = z_levels[z_idx];
                const float z1 = z_levels[z_idx + 1];
                const float v0 = prime_tower_preview_texture_v(z0,
                                                               texture_z_min,
                                                               texture_z_max,
                                                               preserve_aspect_ratio,
                                                               image_width,
                                                               image_height,
                                                               texture_surface_width);
                const float v1 = prime_tower_preview_texture_v(z1,
                                                               texture_z_min,
                                                               texture_z_max,
                                                               preserve_aspect_ratio,
                                                               image_width,
                                                               image_height,
                                                               texture_surface_width);
                const unsigned int base = unsigned(data.vertices_count());

                data.add_vertex(Vec3f(p0.x(), p0.y(), z0), normal, Vec2f(u0, v0));
                data.add_vertex(Vec3f(p1.x(), p1.y(), z0), normal, Vec2f(u1, v0));
                data.add_vertex(Vec3f(p1.x(), p1.y(), z1), normal, Vec2f(u1, v1));
                data.add_vertex(Vec3f(p0.x(), p0.y(), z1), normal, Vec2f(u0, v1));
                data.add_triangle(base, base + 1, base + 2);
                data.add_triangle(base, base + 2, base + 3);
            }
        }
    }

    return data;
}

struct PrimeTowerPreviewRingEdgeGroup
{
    float z = 0.f;
    std::vector<Vec2f> points;
    std::vector<std::pair<int, int>> edges;
};

struct PrimeTowerPreviewRing
{
    float z = 0.f;
    std::vector<Vec2f> points;
    std::vector<float> distances;
    float total_length = 0.f;
    float anchor_distance = 0.f;
};

int prime_tower_preview_find_z_group(std::vector<PrimeTowerPreviewRingEdgeGroup> &groups, float z)
{
    for (size_t i = 0; i < groups.size(); ++i) {
        if (std::abs(groups[i].z - z) <= prime_tower_preview_coord_epsilon)
            return int(i);
    }
    groups.emplace_back();
    groups.back().z = z;
    return int(groups.size() - 1);
}

int prime_tower_preview_find_point(std::vector<Vec2f> &points, const Vec2f &point)
{
    for (size_t i = 0; i < points.size(); ++i) {
        if ((points[i] - point).norm() <= prime_tower_preview_coord_epsilon)
            return int(i);
    }
    points.emplace_back(point);
    return int(points.size() - 1);
}

void prime_tower_preview_add_edge(PrimeTowerPreviewRingEdgeGroup &group, int a, int b)
{
    if (a == b)
        return;
    for (const std::pair<int, int> &edge : group.edges)
        if ((edge.first == a && edge.second == b) || (edge.first == b && edge.second == a))
            return;
    group.edges.emplace_back(a, b);
}

size_t prime_tower_preview_lowest_point_index(const std::vector<Vec2f> &points)
{
    size_t best = 0;
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].y() < points[best].y() - prime_tower_preview_coord_epsilon ||
            (std::abs(points[i].y() - points[best].y()) <= prime_tower_preview_coord_epsilon &&
             points[i].x() < points[best].x())) {
            best = i;
        }
    }
    return best;
}

float prime_tower_preview_polygon_area(const std::vector<Vec2f> &points)
{
    double area = 0.;
    for (size_t i = 0; i < points.size(); ++i) {
        const Vec2f &a = points[i];
        const Vec2f &b = points[(i + 1) % points.size()];
        area += double(a.x()) * double(b.y()) - double(b.x()) * double(a.y());
    }
    return float(0.5 * area);
}

void prime_tower_preview_rotate_to_lowest_point(std::vector<Vec2f> &points)
{
    if (points.empty())
        return;
    const size_t start = prime_tower_preview_lowest_point_index(points);
    std::rotate(points.begin(), points.begin() + start, points.end());
}

bool prime_tower_preview_build_loop(const PrimeTowerPreviewRingEdgeGroup &group, std::vector<Vec2f> &loop)
{
    loop.clear();
    if (group.points.size() < 3 || group.edges.size() < 3)
        return false;

    std::vector<std::vector<int>> adjacency(group.points.size());
    for (const std::pair<int, int> &edge : group.edges) {
        if (edge.first < 0 || edge.second < 0 || edge.first >= int(group.points.size()) || edge.second >= int(group.points.size()))
            continue;
        adjacency[size_t(edge.first)].emplace_back(edge.second);
        adjacency[size_t(edge.second)].emplace_back(edge.first);
    }

    const int start = int(prime_tower_preview_lowest_point_index(group.points));
    int prev = -1;
    int current = start;
    bool closed = false;
    for (size_t guard = 0; guard <= group.points.size(); ++guard) {
        loop.emplace_back(group.points[size_t(current)]);
        const std::vector<int> &neighbors = adjacency[size_t(current)];
        if (neighbors.empty())
            return false;

        int next = -1;
        for (const int candidate : neighbors) {
            if (candidate != prev) {
                next = candidate;
                break;
            }
        }
        if (next < 0)
            return false;
        if (next == start) {
            closed = true;
            break;
        }
        prev = current;
        current = next;
    }

    if (!closed || loop.size() < 3)
        return false;
    if (prime_tower_preview_polygon_area(loop) < 0.f)
        std::reverse(loop.begin(), loop.end());
    prime_tower_preview_rotate_to_lowest_point(loop);
    return true;
}

void prime_tower_preview_prepare_ring(PrimeTowerPreviewRing &ring, float angle_offset_deg)
{
    ring.distances.assign(ring.points.size(), 0.f);
    ring.total_length = 0.f;
    for (size_t i = 0; i < ring.points.size(); ++i) {
        ring.distances[i] = ring.total_length;
        ring.total_length += (ring.points[(i + 1) % ring.points.size()] - ring.points[i]).norm();
    }

    Vec2f min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    for (const Vec2f &point : ring.points) {
        min_pt.x() = std::min(min_pt.x(), point.x());
        min_pt.y() = std::min(min_pt.y(), point.y());
        max_pt.x() = std::max(max_pt.x(), point.x());
        max_pt.y() = std::max(max_pt.y(), point.y());
    }
    const Vec2f center = (min_pt + max_pt) * 0.5f;
    ring.anchor_distance =
        prime_tower_preview_anchor_distance(ring.points, center, prime_tower_preview_anchor_angle(ring.points, angle_offset_deg));
}

std::vector<PrimeTowerPreviewRing> prime_tower_preview_extract_mesh_rings(const TriangleMesh &mesh, float angle_offset_deg)
{
    std::vector<PrimeTowerPreviewRingEdgeGroup> groups;
    for (const stl_triangle_vertex_indices &face : mesh.its.indices) {
        const Vec3f vertices[3] = {
            mesh.its.vertices[size_t(face[0])],
            mesh.its.vertices[size_t(face[1])],
            mesh.its.vertices[size_t(face[2])]
        };
        const float z_min = std::min({vertices[0].z(), vertices[1].z(), vertices[2].z()});
        const float z_max = std::max({vertices[0].z(), vertices[1].z(), vertices[2].z()});
        if (z_max - z_min <= prime_tower_preview_coord_epsilon)
            continue;

        for (int edge_idx = 0; edge_idx < 3; ++edge_idx) {
            const Vec3f &a = vertices[edge_idx];
            const Vec3f &b = vertices[(edge_idx + 1) % 3];
            if (std::abs(a.z() - b.z()) > prime_tower_preview_coord_epsilon)
                continue;
            PrimeTowerPreviewRingEdgeGroup &group = groups[size_t(prime_tower_preview_find_z_group(groups, 0.5f * (a.z() + b.z())))];
            const int a_idx = prime_tower_preview_find_point(group.points, Vec2f(a.x(), a.y()));
            const int b_idx = prime_tower_preview_find_point(group.points, Vec2f(b.x(), b.y()));
            prime_tower_preview_add_edge(group, a_idx, b_idx);
        }
    }

    std::sort(groups.begin(), groups.end(), [](const PrimeTowerPreviewRingEdgeGroup &lhs, const PrimeTowerPreviewRingEdgeGroup &rhs) {
        return lhs.z < rhs.z;
    });

    std::vector<PrimeTowerPreviewRing> rings;
    rings.reserve(groups.size());
    for (const PrimeTowerPreviewRingEdgeGroup &group : groups) {
        PrimeTowerPreviewRing ring;
        ring.z = group.z;
        if (!prime_tower_preview_build_loop(group, ring.points))
            continue;
        prime_tower_preview_prepare_ring(ring, angle_offset_deg);
        if (ring.total_length > prime_tower_preview_epsilon)
            rings.emplace_back(std::move(ring));
    }
    return rings;
}

PrimeTowerPreviewRing prime_tower_preview_interpolate_ring(const PrimeTowerPreviewRing &lower,
                                                           const PrimeTowerPreviewRing &upper,
                                                           float z,
                                                           float angle_offset_deg)
{
    PrimeTowerPreviewRing ring;
    ring.z = z;
    const float t = std::clamp((z - lower.z) / (upper.z - lower.z), 0.f, 1.f);
    ring.points.reserve(lower.points.size());
    for (size_t i = 0; i < lower.points.size(); ++i)
        ring.points.emplace_back(lower.points[i] + (upper.points[i] - lower.points[i]) * t);
    prime_tower_preview_prepare_ring(ring, angle_offset_deg);
    return ring;
}

std::vector<PrimeTowerPreviewRing> prime_tower_preview_insert_z_cuts(const std::vector<PrimeTowerPreviewRing> &rings,
                                                                     float texture_z_min,
                                                                     float texture_z_max,
                                                                     float angle_offset_deg)
{
    if (rings.size() < 2)
        return rings;

    std::vector<PrimeTowerPreviewRing> out;
    for (size_t i = 0; i + 1 < rings.size(); ++i) {
        const PrimeTowerPreviewRing &lower = rings[i];
        const PrimeTowerPreviewRing &upper = rings[i + 1];
        if (lower.points.size() != upper.points.size() || upper.z - lower.z <= prime_tower_preview_epsilon)
            return rings;

        if (out.empty())
            out.emplace_back(lower);

        std::vector<float> cuts;
        if (texture_z_min > lower.z + prime_tower_preview_coord_epsilon && texture_z_min < upper.z - prime_tower_preview_coord_epsilon)
            cuts.emplace_back(texture_z_min);
        if (texture_z_max > lower.z + prime_tower_preview_coord_epsilon && texture_z_max < upper.z - prime_tower_preview_coord_epsilon)
            cuts.emplace_back(texture_z_max);
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end(), [](float lhs, float rhs) {
            return std::abs(lhs - rhs) <= prime_tower_preview_coord_epsilon;
        }), cuts.end());

        for (const float cut : cuts)
            out.emplace_back(prime_tower_preview_interpolate_ring(lower, upper, cut, angle_offset_deg));
        out.emplace_back(upper);
    }
    return out;
}

void prime_tower_preview_add_raw_u_cuts(std::vector<float> &cuts, float raw0, float raw1, int image_slot)
{
    if (std::abs(raw1 - raw0) <= prime_tower_preview_epsilon)
        return;
    const float boundary_step = image_slot == 0 ? 1.f : 0.5f;
    const float raw_min = std::min(raw0, raw1);
    const float raw_max = std::max(raw0, raw1);
    const int first = int(std::floor(raw_min / boundary_step)) - 1;
    const int last = int(std::ceil(raw_max / boundary_step)) + 1;
    for (int boundary_idx = first; boundary_idx <= last; ++boundary_idx) {
        const float boundary = float(boundary_idx) * boundary_step;
        const float t = (boundary - raw0) / (raw1 - raw0);
        if (t > prime_tower_preview_epsilon && t < 1.f - prime_tower_preview_epsilon)
            cuts.emplace_back(t);
    }
}

float prime_tower_preview_texture_u_from_raw(float raw_u, float mid_raw_u, int image_slot)
{
    const float base = std::floor(mid_raw_u);
    if (image_slot == 0)
        return std::clamp(raw_u - base, 0.f, 1.f);
    return image_slot == 1 ? std::clamp(2.f * (raw_u - base), 0.f, 1.f) :
                             std::clamp(2.f * (raw_u - base - 0.5f), 0.f, 1.f);
}

GUI::GLModel::Geometry prime_tower_mesh_image_preview_geometry(const TriangleMesh &mesh,
                                                               float angle_offset_deg,
                                                               float texture_z_min,
                                                               float texture_z_max,
                                                               int image_slot,
                                                               bool preserve_aspect_ratio,
                                                               unsigned int image_width,
                                                               unsigned int image_height)
{
    GUI::GLModel::Geometry data;
    data.format = {GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3T2};

    std::vector<PrimeTowerPreviewRing> rings =
        prime_tower_preview_insert_z_cuts(prime_tower_preview_extract_mesh_rings(mesh, angle_offset_deg),
                                          texture_z_min,
                                          texture_z_max,
                                          angle_offset_deg);
    if (rings.size() < 2)
        return data;

    data.reserve_vertices(mesh.its.indices.size() * 6);
    data.reserve_indices(mesh.its.indices.size() * 6);

    for (size_t ring_idx = 0; ring_idx + 1 < rings.size(); ++ring_idx) {
        const PrimeTowerPreviewRing &lower = rings[ring_idx];
        const PrimeTowerPreviewRing &upper = rings[ring_idx + 1];
        if (lower.points.size() != upper.points.size() || lower.points.size() < 3)
            continue;

        const size_t point_count = lower.points.size();
        for (size_t point_idx = 0; point_idx < point_count; ++point_idx) {
            const size_t next_idx = (point_idx + 1) % point_count;
            const Vec2f lower_a = lower.points[point_idx];
            const Vec2f lower_b = lower.points[next_idx];
            const Vec2f upper_a = upper.points[point_idx];
            const Vec2f upper_b = upper.points[next_idx];
            const float lower_len = (lower_b - lower_a).norm();
            const float upper_len = (upper_b - upper_a).norm();
            if (lower_len <= prime_tower_preview_epsilon && upper_len <= prime_tower_preview_epsilon)
                continue;

            const float lower_raw0 = (lower.distances[point_idx] - lower.anchor_distance) / lower.total_length;
            const float lower_raw1 = (lower.distances[point_idx] + lower_len - lower.anchor_distance) / lower.total_length;
            const float upper_raw0 = (upper.distances[point_idx] - upper.anchor_distance) / upper.total_length;
            const float upper_raw1 = (upper.distances[point_idx] + upper_len - upper.anchor_distance) / upper.total_length;
            const float lower_surface_width = image_slot == 0 ? lower.total_length : 0.5f * lower.total_length;
            const float upper_surface_width = image_slot == 0 ? upper.total_length : 0.5f * upper.total_length;
            const float texture_surface_height =
                texture_z_max > texture_z_min + prime_tower_preview_epsilon ? texture_z_max - texture_z_min : 0.f;

            std::vector<float> cuts = {0.f, 1.f};
            prime_tower_preview_add_raw_u_cuts(cuts, lower_raw0, lower_raw1, image_slot);
            prime_tower_preview_add_raw_u_cuts(cuts, upper_raw0, upper_raw1, image_slot);
            std::sort(cuts.begin(), cuts.end());
            cuts.erase(std::unique(cuts.begin(), cuts.end(), [](float lhs, float rhs) {
                return std::abs(lhs - rhs) <= prime_tower_preview_epsilon;
            }), cuts.end());

            for (size_t cut_idx = 0; cut_idx + 1 < cuts.size(); ++cut_idx) {
                const float t0 = cuts[cut_idx];
                const float t1 = cuts[cut_idx + 1];
                if (t1 - t0 <= prime_tower_preview_epsilon)
                    continue;

                const float mid_t = 0.5f * (t0 + t1);
                const float lower_mid_raw = lower_raw0 + (lower_raw1 - lower_raw0) * mid_t;
                const float upper_mid_raw = upper_raw0 + (upper_raw1 - upper_raw0) * mid_t;
                const float mid_raw = 0.5f * (lower_mid_raw + upper_mid_raw);
                if (image_slot != 0) {
                    const float mid_u = mid_raw - std::floor(mid_raw);
                    if ((mid_u >= 0.5f ? 2 : 1) != image_slot)
                        continue;
                }

                const Vec2f lower_p0 = lower_a + (lower_b - lower_a) * t0;
                const Vec2f lower_p1 = lower_a + (lower_b - lower_a) * t1;
                const Vec2f upper_p0 = upper_a + (upper_b - upper_a) * t0;
                const Vec2f upper_p1 = upper_a + (upper_b - upper_a) * t1;

                Vec3f p0(lower_p0.x(), lower_p0.y(), lower.z);
                Vec3f p1(lower_p1.x(), lower_p1.y(), lower.z);
                Vec3f p2(upper_p1.x(), upper_p1.y(), upper.z);
                Vec3f p3(upper_p0.x(), upper_p0.y(), upper.z);
                Vec3f normal = (p1 - p0).cross(p3 - p0);
                if (normal.norm() <= prime_tower_preview_epsilon)
                    continue;
                normal.normalize();
                const Vec3f offset = normal * prime_tower_preview_offset;
                p0 += offset;
                p1 += offset;
                p2 += offset;
                p3 += offset;

                const float lower_raw_t0 = lower_raw0 + (lower_raw1 - lower_raw0) * t0;
                const float lower_raw_t1 = lower_raw0 + (lower_raw1 - lower_raw0) * t1;
                const float upper_raw_t0 = upper_raw0 + (upper_raw1 - upper_raw0) * t0;
                const float upper_raw_t1 = upper_raw0 + (upper_raw1 - upper_raw0) * t1;
                const float u0 = prime_tower_preview_preserved_texture_u(prime_tower_preview_texture_u_from_raw(lower_raw_t0,
                                                                                                                mid_raw,
                                                                                                                image_slot),
                                                                         preserve_aspect_ratio,
                                                                         image_width,
                                                                         image_height,
                                                                         lower_surface_width,
                                                                         texture_surface_height);
                const float u1 = prime_tower_preview_preserved_texture_u(prime_tower_preview_texture_u_from_raw(lower_raw_t1,
                                                                                                                mid_raw,
                                                                                                                image_slot),
                                                                         preserve_aspect_ratio,
                                                                         image_width,
                                                                         image_height,
                                                                         lower_surface_width,
                                                                         texture_surface_height);
                const float u2 = prime_tower_preview_preserved_texture_u(prime_tower_preview_texture_u_from_raw(upper_raw_t1,
                                                                                                                mid_raw,
                                                                                                                image_slot),
                                                                         preserve_aspect_ratio,
                                                                         image_width,
                                                                         image_height,
                                                                         upper_surface_width,
                                                                         texture_surface_height);
                const float u3 = prime_tower_preview_preserved_texture_u(prime_tower_preview_texture_u_from_raw(upper_raw_t0,
                                                                                                                mid_raw,
                                                                                                                image_slot),
                                                                         preserve_aspect_ratio,
                                                                         image_width,
                                                                         image_height,
                                                                         upper_surface_width,
                                                                         texture_surface_height);
                const float v0 = prime_tower_preview_texture_v(lower.z,
                                                               texture_z_min,
                                                               texture_z_max,
                                                               preserve_aspect_ratio,
                                                               image_width,
                                                               image_height,
                                                               lower_surface_width);
                const float v1 = prime_tower_preview_texture_v(upper.z,
                                                               texture_z_min,
                                                               texture_z_max,
                                                               preserve_aspect_ratio,
                                                               image_width,
                                                               image_height,
                                                               upper_surface_width);
                const unsigned int base = unsigned(data.vertices_count());

                data.add_vertex(p0, normal, Vec2f(u0, v0));
                data.add_vertex(p1, normal, Vec2f(u1, v0));
                data.add_vertex(p2, normal, Vec2f(u2, v1));
                data.add_vertex(p3, normal, Vec2f(u3, v1));
                data.add_triangle(base, base + 1, base + 2);
                data.add_triangle(base, base + 2, base + 3);
            }
        }
    }

    return data;
}

void set_prime_tower_preview_uniforms(GLShaderProgram &shader,
                                      const Transform3d &model_matrix,
                                      const Transform3d &view_matrix,
                                      const Transform3d &projection_matrix,
                                      const std::array<float, 2> &z_range,
                                      const std::array<double, 4> &clipping_plane,
                                      int print_volume_type,
                                      const std::array<float, 4> &print_volume_xy,
                                      const std::array<float, 2> &print_volume_z)
{
    const Transform3d view_model_matrix = view_matrix * model_matrix;
    const Matrix3d view_normal_matrix = view_matrix.matrix().block(0, 0, 3, 3) *
                                        model_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();

    shader.set_uniform("view_model_matrix", view_model_matrix);
    shader.set_uniform("projection_matrix", projection_matrix);
    shader.set_uniform("view_normal_matrix", view_normal_matrix);
    shader.set_uniform("volume_world_matrix", model_matrix);
    shader.set_uniform("z_range", z_range);
    shader.set_uniform("clipping_plane", clipping_plane);
    shader.set_uniform("print_volume.type", print_volume_type);
    shader.set_uniform("print_volume.xy_data", print_volume_xy);
    shader.set_uniform("print_volume.z_data", print_volume_z);
}

GLWipeTowerVolume::GLWipeTowerVolume(const std::vector<ColorRGBA>& colors)
    : GLVolume()
{
    m_colors = colors;
}

void GLWipeTowerVolume::set_prime_tower_image_preview(std::vector<unsigned char> image_rgba,
                                                       unsigned int image_width,
                                                       unsigned int image_height,
                                                       std::vector<unsigned char> image_rgba_back,
                                                       unsigned int image_width_back,
                                                       unsigned int image_height_back,
                                                       float angle_offset_deg,
                                                       bool preserve_aspect_ratio,
                                                       float width,
                                                       float depth,
                                                       float height,
                                                       float texture_z_min,
                                                       float texture_z_max)
{
    auto reset_image = [](PrimeTowerPreviewImage &image) {
        image.model.reset();
        image.texture.reset();
        image.rgba.clear();
        image.width = 0;
        image.height = 0;
    };
    auto valid_image = [](const std::vector<unsigned char> &rgba, unsigned int width, unsigned int height) {
        return width > 0 && height > 0 && rgba.size() >= size_t(width) * size_t(height) * 4;
    };
    auto assign_image = [&](PrimeTowerPreviewImage &target,
                            std::vector<unsigned char> rgba,
                            unsigned int image_w,
                            unsigned int image_h,
                            int image_slot) {
        if (!valid_image(rgba, image_w, image_h))
            return;

        GUI::GLModel::Geometry image_geometry =
            prime_tower_image_preview_geometry(width,
                                               depth,
                                               height,
                                               angle_offset_deg,
                                               texture_z_min,
                                               texture_z_max,
                                               image_slot,
                                               preserve_aspect_ratio,
                                               image_w,
                                               image_h);
        if (image_geometry.is_empty())
            return;

        target.rgba = std::move(rgba);
        target.width = image_w;
        target.height = image_h;
        target.model.init_from(std::move(image_geometry));
    };

    reset_image(m_prime_tower_image);
    reset_image(m_prime_tower_image_back);

    const bool front_valid = valid_image(image_rgba, image_width, image_height);
    const bool back_valid = valid_image(image_rgba_back, image_width_back, image_height_back);
    if (front_valid && back_valid) {
        assign_image(m_prime_tower_image, std::move(image_rgba), image_width, image_height, 1);
        assign_image(m_prime_tower_image_back, std::move(image_rgba_back), image_width_back, image_height_back, 2);
    } else if (front_valid) {
        assign_image(m_prime_tower_image, std::move(image_rgba), image_width, image_height, 0);
    } else if (back_valid) {
        assign_image(m_prime_tower_image_back, std::move(image_rgba_back), image_width_back, image_height_back, 0);
    }
}

void GLWipeTowerVolume::set_prime_tower_image_preview(std::vector<unsigned char> image_rgba,
                                                       unsigned int image_width,
                                                       unsigned int image_height,
                                                       std::vector<unsigned char> image_rgba_back,
                                                       unsigned int image_width_back,
                                                       unsigned int image_height_back,
                                                       float angle_offset_deg,
                                                       bool preserve_aspect_ratio,
                                                       const TriangleMesh &mesh,
                                                       float texture_z_min,
                                                       float texture_z_max)
{
    auto reset_image = [](PrimeTowerPreviewImage &image) {
        image.model.reset();
        image.texture.reset();
        image.rgba.clear();
        image.width = 0;
        image.height = 0;
    };
    auto valid_image = [](const std::vector<unsigned char> &rgba, unsigned int width, unsigned int height) {
        return width > 0 && height > 0 && rgba.size() >= size_t(width) * size_t(height) * 4;
    };
    auto assign_image = [&](PrimeTowerPreviewImage &target,
                            std::vector<unsigned char> rgba,
                            unsigned int image_w,
                            unsigned int image_h,
                            int image_slot) {
        if (!valid_image(rgba, image_w, image_h))
            return;

        GUI::GLModel::Geometry image_geometry =
            prime_tower_mesh_image_preview_geometry(mesh,
                                                    angle_offset_deg,
                                                    texture_z_min,
                                                    texture_z_max,
                                                    image_slot,
                                                    preserve_aspect_ratio,
                                                    image_w,
                                                    image_h);
        if (image_geometry.is_empty())
            return;

        target.rgba = std::move(rgba);
        target.width = image_w;
        target.height = image_h;
        target.model.init_from(std::move(image_geometry));
    };

    reset_image(m_prime_tower_image);
    reset_image(m_prime_tower_image_back);

    const bool front_valid = valid_image(image_rgba, image_width, image_height);
    const bool back_valid = valid_image(image_rgba_back, image_width_back, image_height_back);
    if (front_valid && back_valid) {
        assign_image(m_prime_tower_image, std::move(image_rgba), image_width, image_height, 1);
        assign_image(m_prime_tower_image_back, std::move(image_rgba_back), image_width_back, image_height_back, 2);
    } else if (front_valid) {
        assign_image(m_prime_tower_image, std::move(image_rgba), image_width, image_height, 0);
    } else if (back_valid) {
        assign_image(m_prime_tower_image_back, std::move(image_rgba_back), image_width_back, image_height_back, 0);
    }
}

void GLWipeTowerVolume::render()
{
    if (!is_active)
        return;

    if (m_colors.size() == 0 || m_colors.size() != model_per_colors.size())
        return;

    if (this->is_left_handed())
        glFrontFace(GL_CW);
    glsafe(::glCullFace(GL_BACK));

    for (int i = 0; i < m_colors.size(); i++) {
        if (!picking) {
            ColorRGBA new_color = adjust_color_for_rendering(m_colors[i]);
            this->model_per_colors[i].set_color(new_color);
        } else {
            this->model_per_colors[i].set_color(model.get_color());
        }
        this->model_per_colors[i].render();
    }
    
    if (this->is_left_handed())
        glFrontFace(GL_CCW);
}

void GLWipeTowerVolume::render_prime_tower_image_preview(const Transform3d& view_matrix,
                                                         const Transform3d& projection_matrix,
                                                         const std::array<float, 2>& z_range,
                                                         const std::array<double, 4>& clipping_plane,
                                                         int print_volume_type,
                                                         const std::array<float, 4>& print_volume_xy,
                                                         const std::array<float, 2>& print_volume_z)
{
    if (!is_active || picking || (!m_prime_tower_image.model.is_initialized() && !m_prime_tower_image_back.model.is_initialized()))
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_shader("painted_texture_preview");
    if (shader == nullptr)
        return;

    GLboolean blend_enabled = glIsEnabled(GL_BLEND);
    GLboolean cull_face_enabled = glIsEnabled(GL_CULL_FACE);
    GLboolean depth_mask = GL_TRUE;
    GLint cull_face_mode = GL_BACK;
    GLint depth_func = GL_LESS;
    glsafe(::glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask));
    glsafe(::glGetIntegerv(GL_CULL_FACE_MODE, &cull_face_mode));
    glsafe(::glGetIntegerv(GL_DEPTH_FUNC, &depth_func));

    if (this->is_left_handed())
        glFrontFace(GL_CW);

    glsafe(::glEnable(GL_BLEND));
    glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    glsafe(::glEnable(GL_CULL_FACE));
    glsafe(::glCullFace(GL_BACK));
    glsafe(::glDepthMask(GL_FALSE));
    glsafe(::glDepthFunc(GL_LEQUAL));

    shader->start_using();
    set_prime_tower_preview_uniforms(*shader,
                                     world_matrix(),
                                     view_matrix,
                                     projection_matrix,
                                     z_range,
                                     clipping_plane,
                                     print_volume_type,
                                     print_volume_xy,
                                     print_volume_z);

    glsafe(::glActiveTexture(GL_TEXTURE0));
    shader->set_uniform("uniform_texture", 0);
    shader->set_uniform("texture_preview_mix", 1.f);
    shader->set_uniform("invalid_texture_mapping", false);
    shader->set_uniform("raw_atlas_surface_filter_enabled", false);
    shader->set_uniform("raw_atlas_side_texture_enabled", false);
    shader->set_uniform("raw_atlas_flat_texture_enabled", false);

    auto render_image = [](PrimeTowerPreviewImage &image) {
        if (!image.model.is_initialized())
            return;
        if (image.texture.get_id() == 0) {
            if (image.width == 0 || image.height == 0 || image.rgba.empty())
                return;

            std::vector<unsigned char> texture_data = image.rgba;
            if (!image.texture.load_from_raw_data(std::move(texture_data), image.width, image.height))
                return;

            glsafe(::glBindTexture(GL_TEXTURE_2D, image.texture.get_id()));
            glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            image.rgba.clear();
        }

        glsafe(::glBindTexture(GL_TEXTURE_2D, image.texture.get_id()));
        image.model.set_color(ColorRGBA(1.f, 1.f, 1.f, 0.82f));
        image.model.render();
    };

    render_image(m_prime_tower_image);
    render_image(m_prime_tower_image_back);

    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
    shader->stop_using();

    glsafe(::glDepthFunc(depth_func));
    glsafe(::glDepthMask(depth_mask));
    glsafe(::glCullFace(cull_face_mode));
    if (cull_face_enabled)
        glsafe(::glEnable(GL_CULL_FACE));
    else
        glsafe(::glDisable(GL_CULL_FACE));
    if (blend_enabled)
        glsafe(::glEnable(GL_BLEND));
    else
        glsafe(::glDisable(GL_BLEND));

    if (this->is_left_handed())
        glFrontFace(GL_CCW);
}

bool GLWipeTowerVolume::IsTransparent() { 
    if (m_prime_tower_image.model.is_initialized() || m_prime_tower_image_back.model.is_initialized())
        return true;

    for (size_t i = 0; i < m_colors.size(); i++) {
        if (m_colors[i].is_transparent()) { 
            return true;
        }
    }
    return false; 
}

void GLVolumeCollection::invalidate_texture_mapping_preview_for_object(int object_idx)
{
    bool changed = false;
    for (GLVolume *volume : volumes) {
        if (volume == nullptr || volume->object_idx() != object_idx)
            continue;
        volume->invalidate_texture_mapping_preview();
        changed = true;
    }
    if (changed)
        clear_texture_preview_simulation_cache();
}

std::vector<int> GLVolumeCollection::load_object(
    const ModelObject       *model_object,
    int                      obj_idx,
    const std::vector<int>  &instance_idxs,
    const std::string       &color_by,
    bool 					 opengl_initialized,
    bool                    need_raycaster)
{
    std::vector<int> volumes_idx;
    for (int volume_idx = 0; volume_idx < int(model_object->volumes.size()); ++volume_idx)
        for (int instance_idx : instance_idxs)
            volumes_idx.emplace_back(this->GLVolumeCollection::load_object_volume(model_object, obj_idx, volume_idx, instance_idx, color_by, opengl_initialized, false, false, need_raycaster));
    return volumes_idx;
}


int GLVolumeCollection::load_object_volume(
    const ModelObject   *model_object,
    int                  obj_idx,
    int                  volume_idx,
    int                  instance_idx,
    const std::string   &color_by,
    bool 				 opengl_initialized,
    bool                 in_assemble_view,
    bool                 use_loaded_id,
    bool                 need_raycaster)
{
    const ModelVolume   *model_volume = model_object->volumes[volume_idx];
    const int            extruder_id  = model_volume->extruder_id();
    const ModelInstance *instance 	  = model_object->instances[instance_idx];
    auto color = GLVolume::MODEL_COLOR[((color_by == "volume") ? volume_idx : obj_idx) % 4];
    color.a(model_volume->is_model_part() ? 0.7f : 0.4f);

    std::shared_ptr<const TriangleMesh> mesh = model_volume->mesh_ptr();
    this->volumes.emplace_back(new GLVolume(color));
    GLVolume& v = *this->volumes.back();
    v.set_color(color_from_model_volume(*model_volume));
    v.name = model_volume->name;
	
#if ENABLE_SMOOTH_NORMALS
    v.model.init_from(mesh, true);
#else
    v.model.init_from(*mesh);
    if (need_raycaster) { v.mesh_raycaster = std::make_unique<GUI::MeshRaycaster>(mesh); }
#endif // ENABLE_SMOOTH_NORMALS
    v.composite_id = GLVolume::CompositeID(obj_idx, volume_idx, instance_idx);

    if (model_volume->is_model_part())
    {
        // GLVolume will reference a convex hull from model_volume!
        v.set_convex_hull(model_volume->get_convex_hull_shared_ptr());
        if (extruder_id != -1)
            v.extruder_id = extruder_id;
    }
    v.is_modifier = !model_volume->is_model_part();
    v.shader_outside_printer_detection_enabled = model_volume->is_model_part();
    if (in_assemble_view) {
        v.set_instance_transformation(instance->get_assemble_transformation());
        v.set_offset_to_assembly(instance->get_offset_to_assembly());
    }
    else
        v.set_instance_transformation(instance->get_transformation());
    v.set_volume_transformation(model_volume->get_transformation());
    //use object's instance id
    if (use_loaded_id && (instance->loaded_id > 0))
        v.model_object_ID = instance->loaded_id;
    else
        v.model_object_ID = instance->id().id;

    return int(this->volumes.size() - 1);
}

// Load SLA auxiliary GLVolumes (for support trees or pad).
// This function produces volumes for multiple instances in a single shot,
// as some object specific mesh conversions may be expensive.
void GLVolumeCollection::load_object_auxiliary(
    const SLAPrintObject* print_object,
    int                             obj_idx,
    // pairs of <instance_idx, print_instance_idx>
    const std::vector<std::pair<size_t, size_t>>& instances,
    SLAPrintObjectStep              milestone,
    // Timestamp of the last change of the milestone
    size_t                          timestamp)
{
    assert(print_object->is_step_done(milestone));
    Transform3d  mesh_trafo_inv = print_object->trafo().inverse();
    // Get the support mesh.
    TriangleMesh mesh = print_object->get_mesh(milestone);
    mesh.transform(mesh_trafo_inv);
    // Convex hull is required for out of print bed detection.
    TriangleMesh convex_hull = mesh.convex_hull_3d();
    for (const std::pair<size_t, size_t>& instance_idx : instances) {
        const ModelInstance& model_instance = *print_object->model_object()->instances[instance_idx.first];
        this->volumes.emplace_back(new GLVolume((milestone == slaposPad) ? GLVolume::SLA_PAD_COLOR : GLVolume::SLA_SUPPORT_COLOR));
        GLVolume& v = *this->volumes.back();
#if ENABLE_SMOOTH_NORMALS
        v.model.init_from(mesh, true);
#else
        v.model.init_from(mesh);
        v.model.set_color((milestone == slaposPad) ? GLVolume::SLA_PAD_COLOR : GLVolume::SLA_SUPPORT_COLOR);
        v.mesh_raycaster = std::make_unique<GUI::MeshRaycaster>(std::make_shared<const TriangleMesh>(mesh));
#endif // ENABLE_SMOOTH_NORMALS
        v.composite_id = GLVolume::CompositeID(obj_idx, -int(milestone), (int)instance_idx.first);
        v.geometry_id = std::pair<size_t, size_t>(timestamp, model_instance.id().id);
        // Create a copy of the convex hull mesh for each instance. Use a move operator on the last instance.
        if (&instance_idx == &instances.back())
            v.set_convex_hull(std::move(convex_hull));
        else
            v.set_convex_hull(convex_hull);
        v.is_modifier = false;
        v.shader_outside_printer_detection_enabled = (milestone == slaposSupportTree);
        v.set_instance_transformation(model_instance.get_transformation());
        // Leave the volume transformation at identity.
        // v.set_volume_transformation(model_volume->get_transformation());
    }
}

int GLVolumeCollection::load_wipe_tower_preview(
    int obj_idx, float pos_x, float pos_y, float width, float depth, float height,
    float rotation_angle, bool size_unknown, float brim_width, float texture_z_min, float texture_z_max)
{
    int plate_idx = obj_idx - 1000;

    if (depth < 0.01f)
        return int(this->volumes.size() - 1);
    if (height == 0.0f)
        height = 0.1f;

    std::vector<ColorRGBA> extruder_colors = GUI::wxGetApp().plater()->get_extruders_colors();
    std::vector<ColorRGBA> colors;
    GUI::PartPlateList& ppl = GUI::wxGetApp().plater()->get_partplate_list();
    std::vector<int> plate_extruders = ppl.get_plate(plate_idx)->get_wipe_tower_extruders(true);
    TriangleMesh wipe_tower_shell = make_cube(width, depth, height);
    for (int extruder_id : plate_extruders) {
        if (extruder_id > 0 && extruder_id <= extruder_colors.size())
            colors.push_back(extruder_colors[extruder_id - 1]);
        else if (!extruder_colors.empty())
            colors.push_back(extruder_colors[0]);
    }
    if (colors.empty())
        colors.emplace_back(ColorRGBA::WHITE());

    // Orca: make it transparent
    for(auto& color : colors)
        color.a(0.66f);
    volumes.emplace_back(new GLWipeTowerVolume(colors));
    GLWipeTowerVolume& v = *dynamic_cast<GLWipeTowerVolume*>(volumes.back());
    v.model_per_colors.resize(colors.size());
    for (int i = 0; i < colors.size(); i++) {
        TriangleMesh color_part = make_cube(width, depth / colors.size(), height);
        color_part.translate({ 0.f, depth * i / colors.size(), 0. });
        v.model_per_colors[i].init_from(color_part);
    }
    const TextureMappingGlobalSettings *texture_mapping_global_settings = GUI::wxGetApp().preset_bundle != nullptr ?
        &GUI::wxGetApp().preset_bundle->texture_mapping_global_settings :
        nullptr;
    const TextureMappingPrimeTowerImage &prime_tower_image = GUI::wxGetApp().model().texture_mapping_prime_tower_image;
    const TextureMappingPrimeTowerImage &prime_tower_image_back = GUI::wxGetApp().model().texture_mapping_prime_tower_image_back;
    if (texture_mapping_global_settings != nullptr &&
        texture_mapping_global_settings->effective_enabled(prime_tower_image, prime_tower_image_back)) {
        std::vector<unsigned char> texture_data(prime_tower_image.rgba.begin(), prime_tower_image.rgba.end());
        std::vector<unsigned char> texture_data_back(prime_tower_image_back.rgba.begin(), prime_tower_image_back.rgba.end());
        v.set_prime_tower_image_preview(std::move(texture_data),
                                        prime_tower_image.width,
                                        prime_tower_image.height,
                                        std::move(texture_data_back),
                                        prime_tower_image_back.width,
                                        prime_tower_image_back.height,
                                        texture_mapping_global_settings->angle_offset_deg,
                                        texture_mapping_global_settings->preserve_aspect_ratio,
                                        width,
                                        depth,
                                        height,
                                        texture_z_min,
                                        texture_z_max);
    }
    v.model.init_from(wipe_tower_shell);
    v.mesh_raycaster = std::make_unique<GUI::MeshRaycaster>(std::make_shared<const TriangleMesh>(wipe_tower_shell));
    v.set_convex_hull(wipe_tower_shell);
    v.set_volume_offset(Vec3d(pos_x, pos_y, 0.0));
    v.set_volume_rotation(Vec3d(0., 0., (M_PI / 180.) * rotation_angle));
    v.composite_id = GLVolume::CompositeID(obj_idx, 0, 0);
    v.geometry_id.first = 0;
    v.geometry_id.second = wipe_tower_instance_id().id + (obj_idx - 1000);
    v.is_wipe_tower = true;
    v.shader_outside_printer_detection_enabled = !size_unknown;
    return int(volumes.size() - 1);
}

int GLVolumeCollection::load_real_wipe_tower_preview(int                 obj_idx,
                                                     float               pos_x,
                                                     float               pos_y,
                                                     const TriangleMesh &wt_mesh,
                                                     const TriangleMesh &brim_mesh,
                                                     bool                render_brim,
                                                     float               rotation_angle,
                                                     bool                size_unknown,
                                                     bool                opengl_initialized,
                                                     float               texture_z_min,
                                                     float               texture_z_max)
{
    int plate_idx = obj_idx - 1000;
    if (wt_mesh.its.vertices.empty()) return int(this->volumes.size() - 1);

    std::vector<Slic3r::ColorRGBA> extruder_colors = GUI::wxGetApp().plater()->get_extruders_colors();
    GUI::PartPlateList               &ppl              = GUI::wxGetApp().plater()->get_partplate_list();
    std::vector<int>                  plate_extruders  = ppl.get_plate(plate_idx)->get_wipe_tower_extruders(true);
    std::vector<Slic3r::ColorRGBA>    colors;
    if (!plate_extruders.empty()) {
        if (plate_extruders.front() <= extruder_colors.size())
            colors.push_back(extruder_colors[plate_extruders.front() - 1]);
        else
            colors.push_back(extruder_colors[0]);
    }
    if (colors.empty()) return int(this->volumes.size() - 1);
    volumes.emplace_back(new GLWipeTowerVolume({colors}));
    GLWipeTowerVolume &v = *dynamic_cast<GLWipeTowerVolume *>(volumes.back());
    auto mesh = wt_mesh;
    if (render_brim) {
        mesh.merge(brim_mesh);
    }
    if (!colors.empty()) {
        v.model_per_colors.resize(1);
        v.model_per_colors[0].init_from(mesh);
    }
    const TextureMappingGlobalSettings *texture_mapping_global_settings = GUI::wxGetApp().preset_bundle != nullptr ?
        &GUI::wxGetApp().preset_bundle->texture_mapping_global_settings :
        nullptr;
    const TextureMappingPrimeTowerImage &prime_tower_image = GUI::wxGetApp().model().texture_mapping_prime_tower_image;
    const TextureMappingPrimeTowerImage &prime_tower_image_back = GUI::wxGetApp().model().texture_mapping_prime_tower_image_back;
    if (texture_mapping_global_settings != nullptr &&
        texture_mapping_global_settings->effective_enabled(prime_tower_image, prime_tower_image_back)) {
        const BoundingBoxf3 mesh_bbox = wt_mesh.bounding_box();
        const bool valid_texture_z_range = texture_z_max > texture_z_min + prime_tower_preview_epsilon;
        const float resolved_texture_z_min = valid_texture_z_range ? texture_z_min : mesh_bbox.min.z();
        const float resolved_texture_z_max = valid_texture_z_range ? texture_z_max : mesh_bbox.max.z();
        std::vector<unsigned char> texture_data(prime_tower_image.rgba.begin(), prime_tower_image.rgba.end());
        std::vector<unsigned char> texture_data_back(prime_tower_image_back.rgba.begin(), prime_tower_image_back.rgba.end());
        v.set_prime_tower_image_preview(std::move(texture_data),
                                        prime_tower_image.width,
                                        prime_tower_image.height,
                                        std::move(texture_data_back),
                                        prime_tower_image_back.width,
                                        prime_tower_image_back.height,
                                        texture_mapping_global_settings->angle_offset_deg,
                                        texture_mapping_global_settings->preserve_aspect_ratio,
                                        wt_mesh,
                                        resolved_texture_z_min,
                                        resolved_texture_z_max);
    }
    TriangleMesh wipe_tower_shell = mesh.convex_hull_3d();
    v.model.init_from(wipe_tower_shell);
    v.mesh_raycaster = std::make_unique<GUI::MeshRaycaster>(std::make_shared<const TriangleMesh>(wipe_tower_shell));
    v.set_convex_hull(wipe_tower_shell);
    v.set_volume_offset(Vec3d(pos_x, pos_y, 0.0));
    v.set_volume_rotation(Vec3d(0., 0., (M_PI / 180.) * rotation_angle));
    v.composite_id                             = GLVolume::CompositeID(obj_idx, 0, 0);
    v.geometry_id.first                        = 0;
    v.geometry_id.second                       = wipe_tower_instance_id().id + (obj_idx - 1000);
    v.is_wipe_tower                            = true;
    v.shader_outside_printer_detection_enabled = !size_unknown;
    return int(volumes.size() - 1);
}


GLVolume* GLVolumeCollection::new_toolpath_volume(const ColorRGBA& rgba)
{
    GLVolume* out = new_nontoolpath_volume(rgba);
    out->is_extrusion_path = true;
    return out;
}

GLVolume* GLVolumeCollection::new_nontoolpath_volume(const ColorRGBA& rgba)
{
    GLVolume* out = new GLVolume(rgba);
    out->is_extrusion_path = false;
    this->volumes.emplace_back(out);
    return out;
}

GLVolumeWithIdAndZList volumes_to_render(const GLVolumePtrs& volumes, GLVolumeCollection::ERenderType type, const Transform3d& view_matrix, std::function<bool(const GLVolume&)> filter_func)
{
    GLVolumeWithIdAndZList list;
    list.reserve(volumes.size());

    for (unsigned int i = 0; i < (unsigned int)volumes.size(); ++i) {
        GLVolume* volume = volumes[i];
        if (!volume->is_active)
            continue;
        bool is_transparent = volume->render_color.is_transparent();
        if (volume->is_wipe_tower) {
            GLWipeTowerVolume *wipe_tower_volume = static_cast<GLWipeTowerVolume *>(volume);
            is_transparent = wipe_tower_volume->IsTransparent();
        }
        if (((type == GLVolumeCollection::ERenderType::Opaque && !is_transparent) || 
            (type == GLVolumeCollection::ERenderType::Transparent && is_transparent) ||
             type == GLVolumeCollection::ERenderType::All) &&
            (! filter_func || filter_func(*volume)))
            list.emplace_back(std::make_pair(volume, std::make_pair(i, 0.0)));
    }

    if (type == GLVolumeCollection::ERenderType::Transparent && list.size() > 1) {
        for (GLVolumeWithIdAndZ& volume : list) {
            volume.second.second = volume.first->bounding_box().transformed(view_matrix * volume.first->world_matrix()).max(2);
        }

        std::sort(list.begin(), list.end(),
            [](const GLVolumeWithIdAndZ& v1, const GLVolumeWithIdAndZ& v2) -> bool { return v1.second.second < v2.second.second; }
        );
    }
    else if (type == GLVolumeCollection::ERenderType::Opaque && list.size() > 1) {
        std::sort(list.begin(), list.end(),
            [](const GLVolumeWithIdAndZ& v1, const GLVolumeWithIdAndZ& v2) -> bool { return v1.first->selected && !v2.first->selected; }
        );
    }

    return list;
}

int GLVolumeCollection::get_selection_support_threshold_angle(bool &enable_support) const
{
    const DynamicPrintConfig& glb_cfg        = GUI::wxGetApp().preset_bundle->prints.get_edited_preset().config;
    enable_support =  glb_cfg.opt_bool("enable_support");
    int support_threshold_angle =  glb_cfg.opt_int("support_threshold_angle");
    return  support_threshold_angle ;
}

//BBS: add outline drawing logic
void GLVolumeCollection::render(GLVolumeCollection::ERenderType       type,
                                bool                                  disable_cullface,
                                const Transform3d &                   view_matrix,
                                const Transform3d&                    projection_matrix,
                                const GUI::Size&                      cnv_size,
                                std::function<bool(const GLVolume &)> filter_func,
                                bool                                  partly_inside_enable) const
{
    GLVolumeWithIdAndZList to_render = volumes_to_render(volumes, type, view_matrix, filter_func);
    if (to_render.empty())
        return;

    GLShaderProgram* shader = GUI::wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

    GLShaderProgram* sink_shader = GUI::wxGetApp().get_shader("flat");
#if SLIC3R_OPENGL_ES
    GLShaderProgram* edges_shader = GUI::wxGetApp().get_shader("dashed_lines");
#else
    GLShaderProgram* edges_shader = GUI::OpenGLManager::get_gl_info().is_core_profile() ? GUI::wxGetApp().get_shader("dashed_thick_lines") : GUI::wxGetApp().get_shader("flat");
#endif // SLIC3R_OPENGL_ES

    if (type == ERenderType::Transparent) {
        glsafe(::glEnable(GL_BLEND));
        glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    }

    glsafe(::glCullFace(GL_BACK));
    if (disable_cullface)
        glsafe(::glDisable(GL_CULL_FACE));

    for (GLVolumeWithIdAndZ& volume : to_render) {
#if ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        if (type == ERenderType::Transparent) {
            volume.first->force_transparent = true;
            //BOOST_LOG_TRIVIAL(info) << boost::format("transparent rendering...");
        }
        //else
        //    BOOST_LOG_TRIVIAL(info) << boost::format("opaque rendering...");
#endif // ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        volume.first->set_render_color();
#if ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        if (type == ERenderType::Transparent)
            volume.first->force_transparent = false;
#endif // ENABLE_MODIFIERS_ALWAYS_TRANSPARENT

        // render sinking contours of non-hovered volumes
        shader->stop_using();
        if (sink_shader != nullptr) {
            sink_shader->start_using();
            if (m_show_sinking_contours) {
                if (volume.first->is_sinking() && !volume.first->is_below_printbed() &&
                    volume.first->hover == GLVolume::HS_None && !volume.first->force_sinking_contours) {
                    volume.first->render_sinking_contours();
                }
            }
            sink_shader->stop_using();
        }
        shader->start_using();

        if (!volume.first->model.is_initialized())
            shader->set_uniform("uniform_color", volume.first->render_color);
        shader->set_uniform("z_range", m_z_range);
        shader->set_uniform("clipping_plane", m_clipping_plane);
        shader->set_uniform("use_color_clip_plane", m_use_color_clip_plane);
        shader->set_uniform("color_clip_plane", m_color_clip_plane);
        shader->set_uniform("uniform_color_clip_plane_1", m_color_clip_plane_colors[0]);
        shader->set_uniform("uniform_color_clip_plane_2", m_color_clip_plane_colors[1]);
        //BOOST_LOG_TRIVIAL(info) << boost::format("set uniform_color to {%1%, %2%, %3%, %4%}, with_outline=%5%, selected %6%")
        //    %volume.first->render_color[0]%volume.first->render_color[1]%volume.first->render_color[2]%volume.first->render_color[3]
        //    %with_outline%volume.first->selected;

        //BBS set print_volume to render volume
        //shader->set_uniform("print_volume.type", static_cast<int>(m_render_volume.type));
        //shader->set_uniform("print_volume.xy_data", m_render_volume.data);
        //shader->set_uniform("print_volume.z_data", m_render_volume.zs);

        if (volume.first->partly_inside && partly_inside_enable) {
            //only partly inside volume need to be painted with boundary check
            shader->set_uniform("print_volume.type", static_cast<int>(m_print_volume.type));
            shader->set_uniform("print_volume.xy_data", m_print_volume.data);
            shader->set_uniform("print_volume.z_data", m_print_volume.zs);
        }
        else {
            //use -1 ad a invalid type
            shader->set_uniform("print_volume.type", -1);
        }
        
        bool  enable_support;
        int   support_threshold_angle = get_selection_support_threshold_angle(enable_support);
    
        float normal_z  = -::cos(Geometry::deg2rad((float) support_threshold_angle));
  
        shader->set_uniform("volume_world_matrix", volume.first->world_matrix());
        shader->set_uniform("slope.actived", m_slope.isGlobalActive && !volume.first->is_modifier && !volume.first->is_wipe_tower);
        shader->set_uniform("slope.volume_world_normal_matrix", static_cast<Matrix3f>(volume.first->world_matrix().matrix().block(0, 0, 3, 3).inverse().transpose().cast<float>()));
        shader->set_uniform("slope.normal_z", normal_z);

#if ENABLE_ENVIRONMENT_MAP
        unsigned int environment_texture_id = GUI::wxGetApp().plater()->get_environment_texture_id();
        bool use_environment_texture = environment_texture_id > 0 && GUI::wxGetApp().app_config->get("use_environment_map") == "1";
        shader->set_uniform("use_environment_tex", use_environment_texture);
        if (use_environment_texture)
            glsafe(::glBindTexture(GL_TEXTURE_2D, environment_texture_id));
#endif // ENABLE_ENVIRONMENT_MAP
        glcheck();

		auto red_color = ColorRGBA{1.0f, 0.0f, 0.0f, 1.0f};//slice_error
        volume.first->model.set_color(volume.first->slice_error ? red_color : volume.first->render_color);
        const Transform3d model_matrix = volume.first->world_matrix();
        shader->set_uniform("view_model_matrix", view_matrix * model_matrix);
        shader->set_uniform("projection_matrix", projection_matrix);
        const Matrix3d view_normal_matrix = view_matrix.matrix().block(0, 0, 3, 3) * model_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();
        shader->set_uniform("view_normal_matrix", view_normal_matrix);
		//BBS: add outline related logic
        if (volume.first->selected && GUI::wxGetApp().show_outline())
            volume.first->render_with_outline(cnv_size);
        else
            volume.first->render();

#if ENABLE_ENVIRONMENT_MAP
        if (use_environment_texture)
            glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
#endif // ENABLE_ENVIRONMENT_MAP

        const int texture_preview_print_volume_type =
            volume.first->partly_inside && partly_inside_enable ? static_cast<int>(m_print_volume.type) : -1;
        const std::array<float, 4> texture_preview_clipping_plane = {
            float(m_clipping_plane[0]),
            float(m_clipping_plane[1]),
            float(m_clipping_plane[2]),
            float(m_clipping_plane[3])
        };
        const SurfaceGradientAnchorResolver surface_gradient_anchor_resolver = [this](const TextureMappingZone::LinearGradientAnchor &anchor) {
            return linear_gradient_anchor_global_point(this->volumes, anchor);
        };
        const SurfaceGradientAnchorRadiusResolver surface_gradient_anchor_radius_resolver = [this](const TextureMappingZone::LinearGradientAnchor &anchor) {
            return linear_gradient_anchor_object_radius(this->volumes, anchor);
        };
        const bool render_model_texture_preview =
            volume.first->object_idx() >= 0 && volume.first->volume_idx() >= 0 && !volume.first->is_wipe_tower &&
            !volume.first->is_modifier && !volume.first->is_extrusion_path;
        shader->stop_using();
        if (render_model_texture_preview)
            volume.first->render_mmu_texture_preview(view_matrix,
                                                     projection_matrix,
                                                     m_z_range,
                                                     texture_preview_clipping_plane,
                                                     texture_preview_print_volume_type,
                                                     m_print_volume.data,
                                                     m_print_volume.zs,
                                                     false,
                                                     &surface_gradient_anchor_resolver,
                                                     &surface_gradient_anchor_radius_resolver);
        if (volume.first->is_wipe_tower) {
            GLWipeTowerVolume *wipe_tower_volume = static_cast<GLWipeTowerVolume *>(volume.first);
            wipe_tower_volume->render_prime_tower_image_preview(view_matrix,
                                                                projection_matrix,
                                                                m_z_range,
                                                                m_clipping_plane,
                                                                texture_preview_print_volume_type,
                                                                m_print_volume.data,
                                                                m_print_volume.zs);
        }
        shader->start_using();

        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }

    if (m_show_sinking_contours) {
        shader->stop_using();
        if (sink_shader != nullptr) {
            sink_shader->start_using();
            for (GLVolumeWithIdAndZ& volume : to_render) {
                // render sinking contours of hovered/displaced volumes
                if (volume.first->is_sinking() && !volume.first->is_below_printbed() &&
                    (volume.first->hover != GLVolume::HS_None || volume.first->force_sinking_contours)) {
                    glsafe(::glDepthFunc(GL_ALWAYS));
                    volume.first->render_sinking_contours();
                    glsafe(::glDepthFunc(GL_LESS));
                }
            }
            sink_shader->start_using();
        }
        shader->start_using();
    }

    if (disable_cullface)
        glsafe(::glEnable(GL_CULL_FACE));

    if (type == ERenderType::Transparent)
        glsafe(::glDisable(GL_BLEND));
}

void GLVolumeCollection::render_linear_gradient_direction_arrows(const Transform3d &view_matrix,
                                                                 const Transform3d &projection_matrix) const
{
    const PresetBundle *bundle = GUI::wxGetApp().preset_bundle;
    if (bundle == nullptr)
        return;
    const ConfigOptionStrings *colors_opt = bundle->project_config.option<ConfigOptionStrings>("filament_colour");
    if (colors_opt == nullptr || colors_opt->values.empty())
        return;

    const std::vector<std::string> &filament_colors = colors_opt->values;
    const size_t num_physical = filament_colors.size();
    const TextureMappingManager &texture_mgr = bundle->texture_mapping_zones;

    GUI::GLModel::Geometry black_geometry;
    black_geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };
    GUI::GLModel::Geometry color_geometry;
    color_geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };
    GUI::GLModel::Geometry radius_geometry;
    radius_geometry.format = { GUI::GLModel::Geometry::EPrimitiveType::Triangles, GUI::GLModel::Geometry::EVertexLayout::P3N3C4 };

    for (const TextureMappingZone &zone : texture_mgr.zones()) {
        if (!zone.enabled || zone.deleted || !zone.is_linear_gradient() || !zone.show_linear_gradient_direction_arrow)
            continue;

        const std::vector<unsigned int> component_ids = linear_gradient_component_ids_for_arrow(zone, num_physical);
        if (component_ids.empty())
            continue;
        const std::vector<TextureMappingZone::LinearGradientStop> gradient_stops =
            TextureMappingManager::normalized_linear_gradient_stops(zone, num_physical);
        std::vector<std::array<float, 3>> component_colors;
        component_colors.reserve(component_ids.size());
        for (const unsigned int component_id : component_ids)
            component_colors.emplace_back(linear_gradient_filament_color(component_id, filament_colors));
        const ColorSolverMixModel mix_model = color_solver_mix_model_from_index(zone.generic_solver_mix_model);

        const LinearGradientArrowUsage usage = linear_gradient_arrow_usage(volumes, zone.zone_id);
        const bool stale_unused_start_anchor =
            !usage.any && zone.linear_gradient_start.valid && !linear_gradient_anchor_object_resolves(zone.linear_gradient_start);
        const bool stale_unused_end_anchor =
            !usage.any && zone.linear_gradient_end.valid && !linear_gradient_anchor_object_resolves(zone.linear_gradient_end);
        const std::optional<Vec3f> start_anchor = stale_unused_start_anchor ?
            std::nullopt :
            linear_gradient_anchor_global_point(volumes, zone.linear_gradient_start);
        const bool radial_mode = zone.linear_gradient_mode == int(TextureMappingZone::LinearGradientRadial);
        const std::optional<Vec3f> end_anchor = radial_mode ?
            std::nullopt :
            (stale_unused_end_anchor ? std::nullopt : linear_gradient_anchor_global_point(volumes, zone.linear_gradient_end));
        if (!usage.any && !start_anchor && !end_anchor)
            continue;

        Vec3f start = start_anchor ? *start_anchor : (radial_mode ? usage.radial_center : usage.default_start);
        Vec3f end = usage.default_end;
        if (radial_mode)
            end = start + Vec3f::UnitX();
        if (end_anchor)
            end = *end_anchor;
        if (!start_anchor && !usage.any)
            start = radial_mode ? end - Vec3f::UnitX() : end - Vec3f::UnitZ();
        if (!end_anchor && !usage.any)
            end = start + (radial_mode ? Vec3f::UnitX() : Vec3f::UnitZ());
        if (!linear_gradient_vec3_is_finite(start) || !linear_gradient_vec3_is_finite(end))
            continue;

        Vec3f delta = end - start;
        float length = delta.norm();
        if (length <= 1e-5f) {
            if (radial_mode) {
                delta = Vec3f::UnitX();
                length = 1.f;
            } else if (usage.any) {
                start = usage.default_start;
                end = usage.default_end;
                delta = end - start;
                length = delta.norm();
            }
            if (length <= 1e-5f)
                continue;
        }
        if (radial_mode) {
            length = linear_gradient_radial_radius_mm(volumes, zone, usage.radial_radius);
            end = start + Vec3f::UnitX() * length;
            delta = end - start;
        }

        const float shaft_radius = std::clamp(length * 0.018f, 0.35f, 2.2f);
        const float head_length = std::min(length * 0.24f, std::max(shaft_radius * 4.f, length * 0.12f));
        const float head_radius = shaft_radius * 3.1f;
        append_linear_gradient_arrow(black_geometry,
                                     start,
                                     end,
                                     shaft_radius * 1.55f,
                                     head_radius * 1.28f,
                                     head_length,
                                     component_colors,
                                     gradient_stops,
                                     component_ids,
                                     mix_model,
                                     true);
        append_linear_gradient_arrow(color_geometry,
                                     start,
                                     end,
                                     shaft_radius,
                                     head_radius,
                                     head_length,
                                     component_colors,
                                     gradient_stops,
                                     component_ids,
                                     mix_model,
                                     false);

        const float sphere_radius = shaft_radius * 2.45f;
        if (start_anchor)
            append_linear_gradient_sphere(black_geometry, *start_anchor, sphere_radius, ColorRGBA::BLACK());
        if (!radial_mode && end_anchor)
            append_linear_gradient_sphere(black_geometry, *end_anchor, sphere_radius, ColorRGBA::BLACK());
        if (radial_mode) {
            const float tube_radius = 0.5f;
            append_linear_gradient_dashed_sphere(radius_geometry, start, length, tube_radius, ColorRGBA(0.f, 0.f, 0.f, 0.4f));
        }
    }

    if (black_geometry.is_empty() && color_geometry.is_empty() && radius_geometry.is_empty())
        return;

    GLShaderProgram *shader = GUI::wxGetApp().get_shader("flat_vertex_color");
    if (shader == nullptr)
        return;

    GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blend_enabled = glIsEnabled(GL_BLEND);
    GLboolean cull_face_enabled = glIsEnabled(GL_CULL_FACE);
    GLboolean depth_mask = GL_TRUE;
    GLint depth_func = GL_LESS;
    GLint cull_face_mode = GL_BACK;
    GLenum blend_src_rgb = GL_ONE;
    GLenum blend_dst_rgb = GL_ZERO;
    GLenum blend_src_alpha = GL_ONE;
    GLenum blend_dst_alpha = GL_ZERO;
    glsafe(::glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask));
    glsafe(::glGetIntegerv(GL_DEPTH_FUNC, &depth_func));
    glsafe(::glGetIntegerv(GL_CULL_FACE_MODE, &cull_face_mode));
    glsafe(::glGetIntegerv(GL_BLEND_SRC_RGB, reinterpret_cast<GLint *>(&blend_src_rgb)));
    glsafe(::glGetIntegerv(GL_BLEND_DST_RGB, reinterpret_cast<GLint *>(&blend_dst_rgb)));
    glsafe(::glGetIntegerv(GL_BLEND_SRC_ALPHA, reinterpret_cast<GLint *>(&blend_src_alpha)));
    glsafe(::glGetIntegerv(GL_BLEND_DST_ALPHA, reinterpret_cast<GLint *>(&blend_dst_alpha)));

    glsafe(::glDisable(GL_DEPTH_TEST));
    glsafe(::glDepthMask(GL_FALSE));
    glsafe(::glDisable(GL_CULL_FACE));
    glsafe(::glEnable(GL_BLEND));
    glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

    shader->start_using();
    shader->set_uniform("view_model_matrix", view_matrix);
    shader->set_uniform("projection_matrix", projection_matrix);

    if (!radius_geometry.is_empty()) {
        glsafe(::glEnable(GL_DEPTH_TEST));
        glsafe(::glDepthFunc(GL_LEQUAL));
        glsafe(::glDepthMask(GL_FALSE));
        GUI::GLModel radius_model;
        radius_model.init_from(std::move(radius_geometry));
        radius_model.render(shader);
    }

    glsafe(::glDisable(GL_DEPTH_TEST));
    glsafe(::glDepthMask(GL_FALSE));

    if (!black_geometry.is_empty()) {
        GUI::GLModel black_model;
        black_model.init_from(std::move(black_geometry));
        black_model.render(shader);
    }
    if (!color_geometry.is_empty()) {
        GUI::GLModel color_model;
        color_model.init_from(std::move(color_geometry));
        color_model.render(shader);
    }

    shader->stop_using();
    glsafe(::glBlendFuncSeparate(blend_src_rgb, blend_dst_rgb, blend_src_alpha, blend_dst_alpha));
    glsafe(::glCullFace(cull_face_mode));
    if (cull_face_enabled)
        glsafe(::glEnable(GL_CULL_FACE));
    else
        glsafe(::glDisable(GL_CULL_FACE));
    glsafe(::glDepthFunc(depth_func));
    glsafe(::glDepthMask(depth_mask));
    if (depth_test_enabled)
        glsafe(::glEnable(GL_DEPTH_TEST));
    else
        glsafe(::glDisable(GL_DEPTH_TEST));
    if (blend_enabled)
        glsafe(::glEnable(GL_BLEND));
    else
        glsafe(::glDisable(GL_BLEND));
}

bool GLVolumeCollection::check_wipe_tower_outside_state(const Slic3r::BuildVolume &build_volume, int plate_id) const
{
    for (GLVolume *volume : this->volumes) {
        if (volume->is_wipe_tower) {
            int wipe_tower_plate_id = volume->composite_id.object_id - 1000;
            if (wipe_tower_plate_id != plate_id)
                continue;
            const std::vector<Vec2d>& printable_area = build_volume.printable_area();
            Polygon printable_poly = Polygon::new_scale(printable_area);

            // multi-extruder
            Polygons extruder_polys;
            const std::vector<std::vector<Vec2d>> & extruder_areas = build_volume.extruder_areas();
            if (!extruder_areas.empty()) {
                for (size_t i = 0; i < extruder_areas.size(); ++i) {
                    extruder_polys.emplace_back(Polygon::new_scale(extruder_areas[i]));
                }
                extruder_polys = union_(extruder_polys);
                if (extruder_polys.empty())
                    return false;

                printable_poly = extruder_polys[0];
            }

            const BoundingBoxf3 &bbox = volume->transformed_convex_hull_bounding_box();
            Polygon wipe_tower_polygon = bbox.polygon(true);

            Polygons diff_res = diff(wipe_tower_polygon, printable_poly);
            return diff_res.empty();
        }
    }
    return true;
}

bool GLVolumeCollection::check_outside_state(const BuildVolume &build_volume, ModelInstanceEPrintVolumeState *out_state, ObjectFilamentResults* object_results) const
{
    if (GUI::wxGetApp().plater() == NULL)
    {
        if (out_state != nullptr)
            *out_state = ModelInstancePVS_Inside;
        return false;
    }

    const Model&        model              = GUI::wxGetApp().plater()->model();
    auto                volume_below       = [](GLVolume& volume) -> bool
        { return volume.object_idx() != -1 && volume.volume_idx() != -1 && volume.is_below_printbed(); };
    // Volume is partially below the print bed, thus a pre-calculated convex hull cannot be used.
    auto                volume_sinking     = [](GLVolume& volume) -> bool
        { return volume.object_idx() != -1 && volume.volume_idx() != -1 && volume.is_sinking(); };
    // Cached bounding box of a volume above the print bed.
    auto                volume_bbox        = [volume_sinking](GLVolume& volume) -> BoundingBoxf3
        { return volume_sinking(volume) ? volume.transformed_non_sinking_bounding_box() : volume.transformed_convex_hull_bounding_box(); };
    // Cached 3D convex hull of a volume above the print bed.
    auto                volume_convex_mesh = [volume_sinking, &model](GLVolume& volume) -> const TriangleMesh&
        { return volume_sinking(volume) ? model.objects[volume.object_idx()]->volumes[volume.volume_idx()]->mesh() : *volume.convex_hull(); };

    ModelInstanceEPrintVolumeState overall_state = ModelInstancePVS_Fully_Outside;
    bool contained_min_one = false;

    //BBS: add instance judge logic, besides to original volume judge logic
    //std::map<int64_t, ModelInstanceEPrintVolumeState> model_state;

    GUI::PartPlate* curr_plate = GUI::wxGetApp().plater()->get_partplate_list().get_selected_plate();
    const Pointfs& pp_bed_shape = curr_plate->get_shape();
    BuildVolume plate_build_volume(pp_bed_shape, build_volume.printable_height(), build_volume.extruder_areas(), build_volume.extruder_heights());
    const std::vector<BoundingBoxf3>& exclude_areas = curr_plate->get_exclude_areas();

    std::map<ModelObject*, std::map<int, std::set<int>>> objects_unprintable_filaments;
    int extruder_count = build_volume.get_extruder_area_count();
    std::vector<std::set<int>> unprintable_filament_ids(extruder_count, std::set<int>());
    std::set<ModelObject*> partly_objects_set;
    const ModelObjectPtrs &model_objects = model.objects;
    for (GLVolume* volume : this->volumes)
    {
        std::vector<bool> inside_extruders;
        if (! volume->is_modifier && (volume->shader_outside_printer_detection_enabled || (! volume->is_wipe_tower && volume->composite_id.volume_id >= 0))) {
            BuildVolume::ObjectState state;
            if (volume_below(*volume))
                state = BuildVolume::ObjectState::Below;
            else {
                switch (plate_build_volume.type()) {
                case BuildVolume_Type::Rectangle: {
                    //FIXME this test does not evaluate collision of a build volume bounding box with non-convex objects.
                    const BoundingBoxf3& bb = volume_bbox(*volume);
                    state = plate_build_volume.volume_state_bbox(bb);
                    if ((state == BuildVolume::ObjectState::Inside) && (extruder_count > 1))
                    {
                        state = plate_build_volume.check_volume_bbox_state_with_extruder_areas(bb, inside_extruders);
                    }
                    break;
                }
                case BuildVolume_Type::Circle:
                case BuildVolume_Type::Convex:
                //FIXME doing test on convex hull until we learn to do test on non-convex polygons efficiently.
                case BuildVolume_Type::Custom:
                {
                    const indexed_triangle_set& convex_mesh_it = volume_convex_mesh(*volume).its;
                    const Transform3f trafo = volume->world_matrix().cast<float>();
                    state = plate_build_volume.object_state(convex_mesh_it, trafo, volume_sinking(*volume));
                    if ((state == BuildVolume::ObjectState::Inside) && (extruder_count > 1))
                    {
                        state = plate_build_volume.check_object_state_with_extruder_areas(convex_mesh_it, trafo, inside_extruders);
                    }
                    break;
                }
                default:
                    // Ignore, don't produce any collision.
                    state = BuildVolume::ObjectState::Inside;
                    break;
                }
                assert(state != BuildVolume::ObjectState::Below);

                if (state == BuildVolume::ObjectState::Limited) {
                    //unprintable_filament_ids.resize(inside_extruders.size());
                    ModelObject *model_object = model_objects[volume->object_idx()];
                    ModelVolume *model_volume = model_object->volumes[volume->volume_idx()];
                    for (size_t i = 0; i < inside_extruders.size(); ++i) {
                        if (!inside_extruders[i]) {
                            std::vector<int> filaments = model_volume->get_extruders();
                            unprintable_filament_ids[i].insert(filaments.begin(), filaments.end());
                            if (object_results) {
                                std::map<int, std::set<int>>& obj_extruder_filament_maps = objects_unprintable_filaments[model_object];
                                std::set<int>& obj_extruder_filaments = obj_extruder_filament_maps[i+1];
                                obj_extruder_filaments.insert(filaments.begin(), filaments.end());
                            }
                        }
                    }
                }
            }

            //int64_t comp_id = ((int64_t)volume->composite_id.object_id << 32) | ((int64_t)volume->composite_id.instance_id);
            volume->is_outside = (state != BuildVolume::ObjectState::Inside && state != BuildVolume::ObjectState::Limited);
            volume->partly_inside = (state == BuildVolume::ObjectState::Colliding);
            if (volume->printable) {
                if (state == BuildVolume::ObjectState::Colliding)
                {
                    overall_state = ModelInstancePVS_Partly_Outside;
                    partly_objects_set.emplace(model_objects[volume->object_idx()]);
                }
                else if ((state == BuildVolume::ObjectState::Limited) && (overall_state != ModelInstancePVS_Partly_Outside))
                    overall_state = ModelInstancePVS_Limited;
                else if ((state == BuildVolume::ObjectState::Inside) && (overall_state == ModelInstancePVS_Fully_Outside)) {
                    overall_state = ModelInstancePVS_Fully_Outside;
                }
                contained_min_one |= !volume->is_outside;
            }

            /*ModelInstanceEPrintVolumeState volume_state;
            //if (volume->is_outside && (plate_build_volume.bounding_volume().intersects(volume->bounding_box())))
            if (volume->is_outside && (state == BuildVolume::ObjectState::Colliding))
                volume_state = ModelInstancePVS_Partly_Outside;
            else if (volume->is_outside)
                volume_state = ModelInstancePVS_Fully_Outside;
            else
                volume_state = ModelInstancePVS_Inside;

            if (model_state.find(comp_id) != model_state.end())
            {
                if (model_state[comp_id] != ModelInstancePVS_Partly_Outside)
                {
                    if (volume_state == ModelInstancePVS_Partly_Outside)
                        model_state[comp_id] = ModelInstancePVS_Partly_Outside;
                    else if (model_state[comp_id] != volume_state)
                    {
                        model_state[comp_id] = ModelInstancePVS_Partly_Outside;
                    }
                }
            }
            else
            {
                model_state[comp_id] = volume_state;
            }

            if (model_state[comp_id] == ModelInstancePVS_Partly_Outside) {
                overall_state = ModelInstancePVS_Partly_Outside;
                BOOST_LOG_TRIVIAL(debug) << "instance includes " << volume->name << " is partially outside of bed";
            }*/
        }
    }

    std::vector<std::vector<int>> unprintable_filament_vec;
    for (const std::set<int>& filamnt_ids : unprintable_filament_ids) {
        unprintable_filament_vec.emplace_back(std::vector<int>(filamnt_ids.begin(), filamnt_ids.end()));
    }

    if (object_results && !partly_objects_set.empty()) {
        object_results->partly_outside_objects = std::vector<ModelObject*>(partly_objects_set.begin(), partly_objects_set.end());
    }

    //check per-object error for extruder areas
    if (object_results && (extruder_count > 1))
    {
        const auto& project_config = Slic3r::GUI::wxGetApp().preset_bundle->project_config;
        object_results->mode = curr_plate->get_real_filament_map_mode(project_config);
        if (object_results->mode < FilamentMapMode::fmmManual)
        {
            std::vector<int> conflict_filament_vector;
            for (int index = 0; index < extruder_count; index++ )
            {
                if (!unprintable_filament_vec[index].empty())
                {
                    std::sort (unprintable_filament_vec[index].begin(), unprintable_filament_vec[index].end());
                    if (index == 0)
                        conflict_filament_vector = unprintable_filament_vec[index];
                    else
                    {
                        std::vector<int> result_filaments;
                        //result_filaments.reserve(conflict_filaments.size());
                        std::set_intersection (conflict_filament_vector.begin(), conflict_filament_vector.end(), unprintable_filament_vec[index].begin(), unprintable_filament_vec[index].end(), insert_iterator<vector<int>>(result_filaments, result_filaments.begin()));
                        conflict_filament_vector = result_filaments;
                    }
                }
                else
                {
                    conflict_filament_vector.clear();
                    break;
                }
            }

            if (!conflict_filament_vector.empty())
            {
                std::set<int> conflict_filaments_set(conflict_filament_vector.begin(), conflict_filament_vector.end());
                object_results->filaments = conflict_filament_vector;

                for (auto& object_map: objects_unprintable_filaments)
                {
                    ModelObject *model_object = object_map.first;
                    std::map<int, std::set<int>>& obj_extruder_filament_maps = object_map.second;
                    std::set<int> obj_filaments_set;
                    ObjectFilamentInfo object_filament_info;
                    object_filament_info.object = model_object;

                    for (std::map<int, std::set<int>>::iterator extruder_map_iter = obj_extruder_filament_maps.begin(); extruder_map_iter != obj_extruder_filament_maps.end(); extruder_map_iter++ )
                    {
                        int extruder_id = extruder_map_iter->first;
                        std::set<int>& filaments_set = extruder_map_iter->second;

                        for (int filament: filaments_set)
                        {
                            if (conflict_filaments_set.find(filament) != conflict_filaments_set.end())
                            {
                                obj_filaments_set.emplace(filament);
                            }
                        }
                    }
                    if (!obj_filaments_set.empty()) {
                        object_filament_info.auto_filaments = std::vector<int>(obj_filaments_set.begin(), obj_filaments_set.end());
                        object_results->object_filaments.push_back(std::move(object_filament_info));
                    }
                }
            }
        }
        else
        {
            std::set<int> conflict_filaments_set;
            const auto& project_config = Slic3r::GUI::wxGetApp().preset_bundle->project_config;
            std::vector<int> filament_maps = curr_plate->get_real_filament_maps(project_config);
            for (auto& object_map: objects_unprintable_filaments)
            {
                ModelObject *model_object = object_map.first;
                std::map<int, std::set<int>>& obj_extruder_filament_maps = object_map.second;
                ObjectFilamentInfo object_filament_info;
                object_filament_info.object = model_object;

                for (std::map<int, std::set<int>>::iterator extruder_map_iter = obj_extruder_filament_maps.begin(); extruder_map_iter != obj_extruder_filament_maps.end(); extruder_map_iter++ )
                {
                    int extruder_id = extruder_map_iter->first;
                    std::set<int>& filaments_set = extruder_map_iter->second;

                    for (int filament: filaments_set)
                    {
                        if (filament_maps[filament - 1] == extruder_id)
                        {
                            object_filament_info.manual_filaments.emplace(filament, extruder_id);
                            object_results->filament_maps[filament] = extruder_id;
                            conflict_filaments_set.emplace(filament);
                        }
                    }
                }
                if (!object_filament_info.manual_filaments.empty())
                {
                    object_results->object_filaments.push_back(std::move(object_filament_info));
                }
            }
            if (!conflict_filaments_set.empty()) {
                object_results->filaments = std::vector<int>(conflict_filaments_set.begin(), conflict_filaments_set.end());
            }
        }
    }

    /*for (GLVolume* volume : this->volumes)
    {
        if (! volume->is_modifier && (volume->shader_outside_printer_detection_enabled || (! volume->is_wipe_tower && volume->composite_id.volume_id >= 0)))
        {
            int64_t comp_id = ((int64_t)volume->composite_id.object_id << 32) | ((int64_t)volume->composite_id.instance_id);
            if (model_state.find(comp_id) != model_state.end())
            {
                if (model_state[comp_id] == ModelInstancePVS_Partly_Outside) {
                    volume->partly_inside = true;
                }
                else
                    volume->partly_inside = false;
            }
        }
    }*/

    if (out_state != nullptr)
        *out_state = overall_state;

    return contained_min_one;
}

void GLVolumeCollection::reset_outside_state()
{
    for (GLVolume* volume : this->volumes)
    {
        if (volume != nullptr) {
            volume->is_outside = false;
            volume->partly_inside = false;
        }
    }
}

void GLVolumeCollection::update_colors_by_extruder(const DynamicPrintConfig *config, bool is_update_alpha)
{
    if (config == nullptr)
        return;

    
    using ColorItem = std::pair<std::string, ColorRGBA>;
    std::vector<ColorItem> colors;

    if (config->has("printer_technology") && static_cast<PrinterTechnology>(config->opt_int("printer_technology")) == ptSLA) {
        const std::string& txt_color = config->opt_string("material_colour").empty() ?
                                       print_config_def.get("material_colour")->get_default_value<ConfigOptionString>()->value :
                                       config->opt_string("material_colour");
        ColorRGBA rgba;
        if (decode_color(txt_color, rgba))
            colors.push_back({ txt_color, rgba });
    }
    else {
		if (!config->has("filament_colour")) {
            	return;
        }
        const ConfigOptionStrings* filamemts_opt = dynamic_cast<const ConfigOptionStrings*>(config->option("filament_colour"));
        if (filamemts_opt == nullptr)
            return;

        std::vector<std::string> filament_colors = filamemts_opt->values;
        if (filament_colors.empty())
            return;

        if (GUI::wxGetApp().preset_bundle != nullptr) {
            const size_t physical_count = filament_colors.size();
            const TextureMappingManager &texture_mgr = GUI::wxGetApp().preset_bundle->texture_mapping_zones;
            const size_t total_count = texture_mgr.total_filaments(physical_count);
            filament_colors.resize(total_count, "#8C8C8C");
            for (const TextureMappingZone &zone : texture_mgr.zones()) {
                if (zone.enabled && !zone.deleted && zone.zone_id >= 1 && zone.zone_id <= filament_colors.size() && !zone.display_color.empty())
                    filament_colors[zone.zone_id - 1] = zone.display_color;
            }
        }

        colors.resize(filament_colors.size());

        for (size_t i = 0; i < filament_colors.size(); ++i) {
            ColorRGBA rgba;
            const std::string& fil_color = filament_colors[i];
            if (decode_color(fil_color, rgba))
                colors[i] = { fil_color, rgba };
        }
    }

    for (GLVolume* volume : volumes) {
        if (volume == nullptr || volume->is_modifier || volume->is_wipe_tower || volume->volume_idx() < 0)
            continue;

        int extruder_id = volume->extruder_id - 1;
        if (extruder_id < 0 || (int)colors.size() <= extruder_id)
            extruder_id = 0;

        const ColorItem& color = colors[extruder_id];
        if (!color.first.empty()) {
            if (!is_update_alpha) {
                float old_a   = volume->color.a();
                volume->color = color.second;
                volume->color.a(old_a);
            } else {
                volume->color = color.second;
            }
        }
    }
}

void GLVolumeCollection::set_transparency(float alpha)
{
    for (GLVolume *volume : volumes) {
        if (volume == nullptr || volume->is_modifier || volume->is_wipe_tower || (volume->volume_idx() < 0))
            continue;

        volume->color.a(alpha);
    }
}

std::vector<double> GLVolumeCollection::get_current_print_zs(bool active_only) const
{
    // Collect layer top positions of all volumes.
    std::vector<double> print_zs;
    for (GLVolume *vol : this->volumes)
    {
        if (!active_only || vol->is_active)
            append(print_zs, vol->print_zs);
    }
    std::sort(print_zs.begin(), print_zs.end());

    // Replace intervals of layers with similar top positions with their average value.
    int n = int(print_zs.size());
    int k = 0;
    for (int i = 0; i < n;) {
        int j = i + 1;
        coordf_t zmax = print_zs[i] + EPSILON;
        for (; j < n && print_zs[j] <= zmax; ++ j) ;
        print_zs[k ++] = (j > i + 1) ? (0.5 * (print_zs[i] + print_zs[j - 1])) : print_zs[i];
        i = j;
    }
    if (k < n)
        print_zs.erase(print_zs.begin() + k, print_zs.end());

    return print_zs;
}

size_t GLVolumeCollection::cpu_memory_used() const
{
	size_t memsize = sizeof(*this) + this->volumes.capacity() * sizeof(GLVolume);
	for (const GLVolume *volume : this->volumes)
		memsize += volume->cpu_memory_used();
	return memsize;
}

size_t GLVolumeCollection::gpu_memory_used() const
{
	size_t memsize = 0;
	for (const GLVolume *volume : this->volumes)
		memsize += volume->gpu_memory_used();
	return memsize;
}

std::string GLVolumeCollection::log_memory_info() const
{
	return " (GLVolumeCollection RAM: " + format_memsize_MB(this->cpu_memory_used()) + " GPU: " + format_memsize_MB(this->gpu_memory_used()) + " Both: " + format_memsize_MB(this->gpu_memory_used()) + ")";
}

static void thick_lines_to_geometry(
    const Lines&               lines,
    const std::vector<double>& widths,
    const std::vector<double>& heights,
    bool                       closed,
    double                     top_z,
    GUI::GLModel::Geometry&    geometry)
{
    assert(!lines.empty());
    if (lines.empty())
        return;

    enum Direction : unsigned char
    {
        Left,
        Right,
        Top,
        Bottom
    };

    // right, left, top, bottom
    std::array<int, 4> idx_prev    = { -1, -1, -1, -1 };
    std::array<int, 4> idx_initial = { -1, -1, -1, -1 };

    double bottom_z_prev = 0.0;
    Vec2d  b1_prev(Vec2d::Zero());
    Vec2d  v_prev(Vec2d::Zero());
    double len_prev = 0.0;
    double width_initial = 0.0;
    double bottom_z_initial = 0.0;

    // loop once more in case of closed loops
    const size_t lines_end = closed ? (lines.size() + 1) : lines.size();
    for (size_t ii = 0; ii < lines_end; ++ii) {
        const size_t i = (ii == lines.size()) ? 0 : ii;
        const Line& line = lines[i];
        const double bottom_z = top_z - heights[i];
        const double middle_z = 0.5 * (top_z + bottom_z);
        const double width = widths[i];

        const bool is_first = (ii == 0);
        const bool is_last = (ii == lines_end - 1);
        const bool is_closing = closed && is_last;

        const Vec2d v = unscale(line.vector()).normalized();
        const double len = unscale<double>(line.length());

        const Vec2d a = unscale(line.a);
        const Vec2d b = unscale(line.b);
        Vec2d a1 = a;
        Vec2d a2 = a;
        Vec2d b1 = b;
        Vec2d b2 = b;
        {
            const double dist = 0.5 * width;  // scaled
            const double dx = dist * v.x();
            const double dy = dist * v.y();
            a1 += Vec2d(+dy, -dx);
            a2 += Vec2d(-dy, +dx);
            b1 += Vec2d(+dy, -dx);
            b2 += Vec2d(-dy, +dx);
        }

        // calculate new XY normals
        const Vec2d xy_right_normal = unscale(line.normal()).normalized();

        std::array<int, 4> idx_a = { 0, 0, 0, 0 };
        std::array<int, 4> idx_b = { 0, 0, 0, 0 };
        int idx_last = int(geometry.vertices_count());

        const bool bottom_z_different = bottom_z_prev != bottom_z;
        bottom_z_prev = bottom_z;

        if (!is_first && bottom_z_different) {
            // Found a change of the layer thickness -> Add a cap at the end of the previous segment.
            geometry.add_triangle(idx_b[Bottom], idx_b[Left], idx_b[Top]);
            geometry.add_triangle(idx_b[Bottom], idx_b[Top], idx_b[Right]);
        }

        // Share top / bottom vertices if possible.
        if (is_first) {
            idx_a[Top] = idx_last++;
            geometry.add_vertex(Vec3f(a.x(), a.y(), top_z), Vec3f(0.0f, 0.0f, 1.0f));
        }
        else
            idx_a[Top] = idx_prev[Top];

        if (is_first || bottom_z_different) {
            // Start of the 1st line segment or a change of the layer thickness while maintaining the print_z.
            idx_a[Bottom] = idx_last++;
            geometry.add_vertex(Vec3f(a.x(), a.y(), bottom_z), Vec3f(0.0f, 0.0f, -1.0f));
            idx_a[Left] = idx_last++;
            geometry.add_vertex(Vec3f(a2.x(), a2.y(), middle_z), Vec3f(-xy_right_normal.x(), -xy_right_normal.y(), 0.0f));
            idx_a[Right] = idx_last++;
            geometry.add_vertex(Vec3f(a1.x(), a1.y(), middle_z), Vec3f(xy_right_normal.x(), xy_right_normal.y(), 0.0f));
        }
        else
            idx_a[Bottom] = idx_prev[Bottom];

        if (is_first) {
            // Start of the 1st line segment.
            width_initial = width;
            bottom_z_initial = bottom_z;
            idx_initial = idx_a;
        }
        else {
            // Continuing a previous segment.
            // Share left / right vertices if possible.
            const double v_dot = v_prev.dot(v);
            // To reduce gpu memory usage, we try to reuse vertices
            // To reduce the visual artifacts, due to averaged normals, we allow to reuse vertices only when any of two adjacent edges 
            // is longer than a fixed threshold.
            // The following value is arbitrary, it comes from tests made on a bunch of models showing the visual artifacts
            const double len_threshold = 2.5;

            // Generate new vertices if the angle between adjacent edges is greater than 45 degrees or thresholds conditions are met
            const bool sharp = (v_dot < 0.707) || (len_prev > len_threshold) || (len > len_threshold);
            if (sharp) {
                if (!bottom_z_different) {
                    // Allocate new left / right points for the start of this segment as these points will receive their own normals to indicate a sharp turn.
                    idx_a[Right] = idx_last++;
                    geometry.add_vertex(Vec3f(a1.x(), a1.y(), middle_z), Vec3f(xy_right_normal.x(), xy_right_normal.y(), 0.0f));
                    idx_a[Left] = idx_last++;
                    geometry.add_vertex(Vec3f(a2.x(), a2.y(), middle_z), Vec3f(-xy_right_normal.x(), -xy_right_normal.y(), 0.0f));
                    if (cross2(v_prev, v) > 0.0) {
                        // Right turn. Fill in the right turn wedge.
                        geometry.add_triangle(idx_prev[Right], idx_a[Right], idx_prev[Top]);
                        geometry.add_triangle(idx_prev[Right], idx_prev[Bottom], idx_a[Right]);
                    }
                    else {
                        // Left turn. Fill in the left turn wedge.
                        geometry.add_triangle(idx_prev[Left], idx_prev[Top], idx_a[Left]);
                        geometry.add_triangle(idx_prev[Left], idx_a[Left], idx_prev[Bottom]);
                    }
                }
            }
            else {
                if (!bottom_z_different) {
                    // The two successive segments are nearly collinear.
                    idx_a[Left]  = idx_prev[Left];
                    idx_a[Right] = idx_prev[Right];
                }
            }
            if (is_closing) {
                if (!sharp) {
                    if (!bottom_z_different) {
                        // Closing a loop with smooth transition. Unify the closing left / right vertices.
                        geometry.set_vertex(idx_initial[Left], geometry.extract_position_3(idx_prev[Left]), geometry.extract_normal_3(idx_prev[Left]));
                        geometry.set_vertex(idx_initial[Right], geometry.extract_position_3(idx_prev[Right]), geometry.extract_normal_3(idx_prev[Right]));
                        geometry.remove_vertex(geometry.vertices_count() - 1);
                        geometry.remove_vertex(geometry.vertices_count() - 1);
                        // Replace the left / right vertex indices to point to the start of the loop.
                        const size_t indices_count = geometry.indices_count();
                        for (size_t u = indices_count - 24; u < indices_count; ++u) {
                            const unsigned int id = geometry.extract_index(u);
                            if (id == (unsigned int)idx_prev[Left])
                                geometry.set_index(u, (unsigned int)idx_initial[Left]);
                            else if (id == (unsigned int)idx_prev[Right])
                                geometry.set_index(u, (unsigned int)idx_initial[Right]);
                        }
                    }
                }
                // This is the last iteration, only required to solve the transition.
                break;
            }
        }

        // Only new allocate top / bottom vertices, if not closing a loop.
        if (is_closing)
            idx_b[Top] = idx_initial[Top];
        else {
            idx_b[Top] = idx_last++;
            geometry.add_vertex(Vec3f(b.x(), b.y(), top_z), Vec3f(0.0f, 0.0f, 1.0f));
        }

        if (is_closing && width == width_initial && bottom_z == bottom_z_initial)
            idx_b[Bottom] = idx_initial[Bottom];
        else {
            idx_b[Bottom] = idx_last++;
            geometry.add_vertex(Vec3f(b.x(), b.y(), bottom_z), Vec3f(0.0f, 0.0f, -1.0f));
        }
        // Generate new vertices for the end of this line segment.
        idx_b[Left] = idx_last++;
        geometry.add_vertex(Vec3f(b2.x(), b2.y(), middle_z), Vec3f(-xy_right_normal.x(), -xy_right_normal.y(), 0.0f));
        idx_b[Right] = idx_last++;
        geometry.add_vertex(Vec3f(b1.x(), b1.y(), middle_z), Vec3f(xy_right_normal.x(), xy_right_normal.y(), 0.0f));

        idx_prev = idx_b;
        bottom_z_prev = bottom_z;
        b1_prev = b1;
        v_prev = v;
        len_prev = len;

        if (bottom_z_different && (closed || (!is_first && !is_last))) {
            // Found a change of the layer thickness -> Add a cap at the beginning of this segment.
            geometry.add_triangle(idx_a[Bottom], idx_a[Right], idx_a[Top]);
            geometry.add_triangle(idx_a[Bottom], idx_a[Top], idx_a[Left]);
        }

        if (!closed) {
            // Terminate open paths with caps.
            if (is_first) {
                geometry.add_triangle(idx_a[Bottom], idx_a[Right], idx_a[Top]);
                geometry.add_triangle(idx_a[Bottom], idx_a[Top], idx_a[Left]);
            }
            // We don't use 'else' because both cases are true if we have only one line.
            if (is_last) {
                geometry.add_triangle(idx_b[Bottom], idx_b[Left], idx_b[Top]);
                geometry.add_triangle(idx_b[Bottom], idx_b[Top], idx_b[Right]);
            }
        }

        // Add quads for a straight hollow tube-like segment.
        // bottom-right face
        geometry.add_triangle(idx_a[Bottom], idx_b[Bottom], idx_b[Right]);
        geometry.add_triangle(idx_a[Bottom], idx_b[Right], idx_a[Right]);
        // top-right face
        geometry.add_triangle(idx_a[Right], idx_b[Right], idx_b[Top]);
        geometry.add_triangle(idx_a[Right], idx_b[Top], idx_a[Top]);
        // top-left face
        geometry.add_triangle(idx_a[Top], idx_b[Top], idx_b[Left]);
        geometry.add_triangle(idx_a[Top], idx_b[Left], idx_a[Left]);
        // bottom-left face
        geometry.add_triangle(idx_a[Left], idx_b[Left], idx_b[Bottom]);
        geometry.add_triangle(idx_a[Left], idx_b[Bottom], idx_a[Bottom]);
    }
}

// caller is responsible for supplying NO lines with zero length
static void thick_lines_to_geometry(
    const Lines3&              lines,
    const std::vector<double>& widths,
    const std::vector<double>& heights,
    bool                       closed,
    GUI::GLModel::Geometry&    geometry)
{
    assert(!lines.empty());
    if (lines.empty())
        return;

    enum Direction : unsigned char
    {
        Left,
        Right,
        Top,
        Bottom
    };

    // left, right, top, bottom
    std::array<int, 4> idx_prev    = { -1, -1, -1, -1 };
    std::array<int, 4> idx_initial = { -1, -1, -1, -1 };

    double z_prev = 0.0;
    double len_prev = 0.0;
    Vec3d  n_right_prev = Vec3d::Zero();
    Vec3d  n_top_prev = Vec3d::Zero();
    Vec3d  unit_v_prev = Vec3d::Zero();
    double width_initial = 0.0;

    // new vertices around the line endpoints
    // left, right, top, bottom
    std::array<Vec3d, 4> a = { Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero() };
    std::array<Vec3d, 4> b = { Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero() };

    // loop once more in case of closed loops
    const size_t lines_end = closed ? (lines.size() + 1) : lines.size();
    for (size_t ii = 0; ii < lines_end; ++ii) {
        const size_t i = (ii == lines.size()) ? 0 : ii;

        const Line3& line = lines[i];
        const double height = heights[i];
        const double width = widths[i];

        const Vec3d unit_v = unscale(line.vector()).normalized();
        const double len = unscale<double>(line.length());

        Vec3d n_top = Vec3d::Zero();
        Vec3d n_right = Vec3d::Zero();

        if (line.a.x() == line.b.x() && line.a.y() == line.b.y()) {
            // vertical segment
            n_top = Vec3d::UnitY();
            n_right = Vec3d::UnitX();
            if (line.a.z() < line.b.z())
                n_right = -n_right;
        }
        else {
            // horizontal segment
            n_right = unit_v.cross(Vec3d::UnitZ()).normalized();
            n_top = n_right.cross(unit_v).normalized();
        }

        const Vec3d rl_displacement = 0.5 * width * n_right;
        const Vec3d tb_displacement = 0.5 * height * n_top;
        const Vec3d l_a = unscale(line.a);
        const Vec3d l_b = unscale(line.b);

        a[Right]  = l_a + rl_displacement;
        a[Left]   = l_a - rl_displacement;
        a[Top]    = l_a + tb_displacement;
        a[Bottom] = l_a - tb_displacement;
        b[Right]  = l_b + rl_displacement;
        b[Left]   = l_b - rl_displacement;
        b[Top]    = l_b + tb_displacement;
        b[Bottom] = l_b - tb_displacement;

        const Vec3d n_bottom = -n_top;
        const Vec3d n_left = -n_right;

        std::array<int, 4> idx_a = { 0, 0, 0, 0};
        std::array<int, 4> idx_b = { 0, 0, 0, 0 };
        int idx_last = int(geometry.vertices_count());

        const bool z_different = (z_prev != l_a.z());
        z_prev = l_b.z();

        // Share top / bottom vertices if possible.
        if (ii == 0) {
            idx_a[Top] = idx_last++;
            geometry.add_vertex((Vec3f)a[Top].cast<float>(), (Vec3f)n_top.cast<float>());
        }
        else
            idx_a[Top] = idx_prev[Top];

        if (ii == 0 || z_different) {
            // Start of the 1st line segment or a change of the layer thickness while maintaining the print_z.
            idx_a[Bottom] = idx_last++;
            geometry.add_vertex((Vec3f)a[Bottom].cast<float>(), (Vec3f)n_bottom.cast<float>());
            idx_a[Left] = idx_last++;
            geometry.add_vertex((Vec3f)a[Left].cast<float>(), (Vec3f)n_left.cast<float>());
            idx_a[Right] = idx_last++;
            geometry.add_vertex((Vec3f)a[Right].cast<float>(), (Vec3f)n_right.cast<float>());
        }
        else
            idx_a[Bottom] = idx_prev[Bottom];

        if (ii == 0) {
            // Start of the 1st line segment.
            width_initial = width;
            idx_initial =  idx_a;
        }
        else {
            // Continuing a previous segment.
            // Share left / right vertices if possible.
            const double v_dot = unit_v_prev.dot(unit_v);
            const bool is_right_turn = n_top_prev.dot(unit_v_prev.cross(unit_v)) > 0.0;

            // To reduce gpu memory usage, we try to reuse vertices
            // To reduce the visual artifacts, due to averaged normals, we allow to reuse vertices only when any of two adjacent edges 
            // is longer than a fixed threshold.
            // The following value is arbitrary, it comes from tests made on a bunch of models showing the visual artifacts
            const double len_threshold = 2.5;

            // Generate new vertices if the angle between adjacent edges is greater than 45 degrees or thresholds conditions are met
            const bool is_sharp = v_dot < 0.707 || len_prev > len_threshold || len > len_threshold;
            if (is_sharp) {
                // Allocate new left / right points for the start of this segment as these points will receive their own normals to indicate a sharp turn.
                idx_a[Right] = idx_last++;
                geometry.add_vertex((Vec3f)a[Right].cast<float>(), (Vec3f)n_right.cast<float>());
                idx_a[Left] = idx_last++;
                geometry.add_vertex((Vec3f)a[Left].cast<float>(), (Vec3f)n_left.cast<float>());

                if (is_right_turn) {
                    // Right turn. Fill in the right turn wedge.
                    geometry.add_triangle(idx_prev[Right], idx_a[Right], idx_prev[Top]);
                    geometry.add_triangle(idx_prev[Right], idx_prev[Bottom], idx_a[Right]);
                }
                else {
                    // Left turn. Fill in the left turn wedge.
                    geometry.add_triangle(idx_prev[Left], idx_prev[Top], idx_a[Left]);
                    geometry.add_triangle(idx_prev[Left], idx_a[Left], idx_prev[Bottom]);
                }
            }
            else {
                // The two successive segments are nearly collinear.
                idx_a[Left] = idx_prev[Left];
                idx_a[Right] = idx_prev[Right];
            }

            if (ii == lines.size()) {
                if (!is_sharp) {
                    // Closing a loop with smooth transition. Unify the closing left / right vertices.
                    geometry.set_vertex(idx_initial[Left], geometry.extract_position_3(idx_prev[Left]), geometry.extract_normal_3(idx_prev[Left]));
                    geometry.set_vertex(idx_initial[Right], geometry.extract_position_3(idx_prev[Right]), geometry.extract_normal_3(idx_prev[Right]));
                    geometry.remove_vertex(geometry.vertices_count() - 1);
                    geometry.remove_vertex(geometry.vertices_count() - 1);
                    // Replace the left / right vertex indices to point to the start of the loop.
                    const size_t indices_count = geometry.indices_count();
                    for (size_t u = indices_count - 24; u < indices_count; ++u) {
                        const unsigned int id = geometry.extract_index(u);
                        if (id == (unsigned int)idx_prev[Left])
                            geometry.set_index(u, (unsigned int)idx_initial[Left]);
                        else if (id == (unsigned int)idx_prev[Right])
                            geometry.set_index(u, (unsigned int)idx_initial[Right]);
                    }
                }

                // This is the last iteration, only required to solve the transition.
                break;
            }
        }

        // Only new allocate top / bottom vertices, if not closing a loop.
        if (closed && ii + 1 == lines.size())
            idx_b[Top] = idx_initial[Top];
        else {
            idx_b[Top] = idx_last++;
            geometry.add_vertex((Vec3f)b[Top].cast<float>(), (Vec3f)n_top.cast<float>());
        }

        if (closed && ii + 1 == lines.size() && width == width_initial)
            idx_b[Bottom] = idx_initial[Bottom];
        else {
            idx_b[Bottom] = idx_last++;
            geometry.add_vertex((Vec3f)b[Bottom].cast<float>(), (Vec3f)n_bottom.cast<float>());
        }

        // Generate new vertices for the end of this line segment.
        idx_b[Left] = idx_last++;
        geometry.add_vertex((Vec3f)b[Left].cast<float>(), (Vec3f)n_left.cast<float>());
        idx_b[Right] = idx_last++;
        geometry.add_vertex((Vec3f)b[Right].cast<float>(), (Vec3f)n_right.cast<float>());

        idx_prev = idx_b;
        n_right_prev = n_right;
        n_top_prev = n_top;
        unit_v_prev = unit_v;
        len_prev = len;

        if (!closed) {
            // Terminate open paths with caps.
            if (i == 0) {
                geometry.add_triangle(idx_a[Bottom], idx_a[Right], idx_a[Top]);
                geometry.add_triangle(idx_a[Bottom], idx_a[Top], idx_a[Left]);
            }

            // We don't use 'else' because both cases are true if we have only one line.
            if (i + 1 == lines.size()) {
                geometry.add_triangle(idx_b[Bottom], idx_b[Left], idx_b[Top]);
                geometry.add_triangle(idx_b[Bottom], idx_b[Top], idx_b[Right]);
            }
        }

        // Add quads for a straight hollow tube-like segment.
        // bottom-right face
        geometry.add_triangle(idx_a[Bottom], idx_b[Bottom], idx_b[Right]);
        geometry.add_triangle(idx_a[Bottom], idx_b[Right], idx_a[Right]);
        // top-right face
        geometry.add_triangle(idx_a[Right], idx_b[Right], idx_b[Top]);
        geometry.add_triangle(idx_a[Right], idx_b[Top], idx_a[Top]);
        // top-left face
        geometry.add_triangle(idx_a[Top], idx_b[Top], idx_b[Left]);
        geometry.add_triangle(idx_a[Top], idx_b[Left], idx_a[Left]);
        // bottom-left face
        geometry.add_triangle(idx_a[Left], idx_b[Left], idx_b[Bottom]);
        geometry.add_triangle(idx_a[Left], idx_b[Bottom], idx_a[Bottom]);
    }
}

void _3DScene::thick_lines_to_verts(
    const Lines&               lines,
    const std::vector<double>& widths,
    const std::vector<double>& heights,
    bool                       closed,
    double                     top_z,
    GUI::GLModel::Geometry&    geometry)
{
    thick_lines_to_geometry(lines, widths, heights, closed, top_z, geometry);
}

void _3DScene::thick_lines_to_verts(
    const Lines3&              lines,
    const std::vector<double>& widths,
    const std::vector<double>& heights,
    bool                       closed,
    GUI::GLModel::Geometry&    geometry)
{
    thick_lines_to_geometry(lines, widths, heights, closed, geometry);
}

// Fill in the qverts and tverts with quads and triangles for the extrusion_path.
void _3DScene::extrusionentity_to_verts(const ExtrusionPath& extrusion_path, float print_z, const Point& copy, GUI::GLModel::Geometry& geometry)
{
    Polyline            polyline = extrusion_path.polyline;
    polyline.remove_duplicate_points();
    polyline.translate(copy);
    const Lines               lines = polyline.lines();
    std::vector<double> widths(lines.size(), extrusion_path.width);
    std::vector<double> heights(lines.size(), extrusion_path.height);
    thick_lines_to_verts(lines, widths, heights, false, print_z, geometry);
}

// Fill in the qverts and tverts with quads and triangles for the extrusion_loop.
void _3DScene::extrusionentity_to_verts(const ExtrusionLoop& extrusion_loop, float print_z, const Point& copy, GUI::GLModel::Geometry& geometry)
{
    Lines               lines;
    std::vector<double> widths;
    std::vector<double> heights;
    for (const ExtrusionPath& extrusion_path : extrusion_loop.paths) {
        Polyline            polyline = extrusion_path.polyline;
        polyline.remove_duplicate_points();
        polyline.translate(copy);
        const Lines lines_this = polyline.lines();
        append(lines, lines_this);
        widths.insert(widths.end(), lines_this.size(), extrusion_path.width);
        heights.insert(heights.end(), lines_this.size(), extrusion_path.height);
    }
    thick_lines_to_verts(lines, widths, heights, true, print_z, geometry);
}

// Fill in the qverts and tverts with quads and triangles for the extrusion_multi_path.
void _3DScene::extrusionentity_to_verts(const ExtrusionMultiPath& extrusion_multi_path, float print_z, const Point& copy, GUI::GLModel::Geometry& geometry)
{
    Lines               lines;
    std::vector<double> widths;
    std::vector<double> heights;
    for (const ExtrusionPath& extrusion_path : extrusion_multi_path.paths) {
        Polyline            polyline = extrusion_path.polyline;
        polyline.remove_duplicate_points();
        polyline.translate(copy);
        const Lines lines_this = polyline.lines();
        append(lines, lines_this);
        widths.insert(widths.end(), lines_this.size(), extrusion_path.width);
        heights.insert(heights.end(), lines_this.size(), extrusion_path.height);
    }
    thick_lines_to_verts(lines, widths, heights, false, print_z, geometry);
}

void _3DScene::extrusionentity_to_verts(const ExtrusionEntityCollection& extrusion_entity_collection, float print_z, const Point& copy, GUI::GLModel::Geometry& geometry)
{
    for (const ExtrusionEntity* extrusion_entity : extrusion_entity_collection.entities)
        extrusionentity_to_verts(extrusion_entity, print_z, copy, geometry);
}

void _3DScene::extrusionentity_to_verts(const ExtrusionEntity* extrusion_entity, float print_z, const Point& copy, GUI::GLModel::Geometry& geometry)
{
    if (extrusion_entity != nullptr) {
        auto* extrusion_path = dynamic_cast<const ExtrusionPath*>(extrusion_entity);
        if (extrusion_path != nullptr)
            extrusionentity_to_verts(*extrusion_path, print_z, copy, geometry);
        else {
            auto* extrusion_loop = dynamic_cast<const ExtrusionLoop*>(extrusion_entity);
            if (extrusion_loop != nullptr)
                extrusionentity_to_verts(*extrusion_loop, print_z, copy, geometry);
            else {
                auto* extrusion_multi_path = dynamic_cast<const ExtrusionMultiPath*>(extrusion_entity);
                if (extrusion_multi_path != nullptr)
                    extrusionentity_to_verts(*extrusion_multi_path, print_z, copy, geometry);
                else {
                    auto* extrusion_entity_collection = dynamic_cast<const ExtrusionEntityCollection*>(extrusion_entity);
                    if (extrusion_entity_collection != nullptr)
                        extrusionentity_to_verts(*extrusion_entity_collection, print_z, copy, geometry);
                    else
                        throw Slic3r::RuntimeError("Unexpected extrusion_entity type in to_verts()");
                }
            }
        }
    }
}

} // namespace Slic3r
