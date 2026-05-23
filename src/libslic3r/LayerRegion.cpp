#include "Layer.hpp"
#include "BridgeDetector.hpp"
#include "ClipperUtils.hpp"
#include "Color.hpp"
#include "Geometry.hpp"
#include "PerimeterGenerator.hpp"
#include "Print.hpp"
#include "Surface.hpp"
#include "BoundingBox.hpp"
#include "SVG.hpp"
#include "TextureMapping.hpp"
#include "TextureMappingOffset.hpp"
#include "MultiMaterialSegmentation.hpp"
#include "Algorithm/RegionExpansion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <map>
#include <optional>

#include <boost/log/trivial.hpp>
#include <boost/algorithm/clamp.hpp>

namespace Slic3r {

struct PerimeterPathBoundarySample {
    Point  point;
    double inward_x { 0.0 };
    double inward_y { 0.0 };
    float  inset_mm { 0.f };
};

struct PerimeterTextureMaskIndex {
    const ExPolygons *expolygons { nullptr };
    std::vector<BoundingBox> bboxes;
    BoundingBox bbox;

    bool empty() const { return expolygons == nullptr || expolygons->empty() || bboxes.empty() || !bbox.defined; }
};

static ExPolygons perimeter_texture_expolygons_overlapping_bbox(const ExPolygons              &expolygons,
                                                                const std::vector<BoundingBox> *bboxes,
                                                                const BoundingBox             &bbox)
{
    ExPolygons out;
    if (expolygons.empty() || !bbox.defined)
        return out;

    for (size_t idx = 0; idx < expolygons.size(); ++idx) {
        BoundingBox expolygon_bbox = bboxes != nullptr && idx < bboxes->size() ?
            (*bboxes)[idx] :
            get_extents(expolygons[idx]);
        if (expolygon_bbox.defined && expolygon_bbox.overlap(bbox))
            out.emplace_back(expolygons[idx]);
    }
    return out;
}

static ExPolygons perimeter_texture_above_layer_slices_overlapping_bbox(const Layer *layer, const BoundingBox &bbox, int above_layer_count)
{
    ExPolygons out;
    const int clamped_count =
        std::clamp(above_layer_count,
                   TextureMappingZone::MinTopVisiblePerimeterRecolorAboveLayers,
                   TextureMappingZone::MaxTopVisiblePerimeterRecolorAboveLayers);
    const Layer *upper_layer = layer != nullptr ? layer->upper_layer : nullptr;
    for (int idx = 0; upper_layer != nullptr && idx < clamped_count; ++idx, upper_layer = upper_layer->upper_layer) {
        if (upper_layer->lslices.empty())
            continue;
        ExPolygons overlapping = bbox.defined ?
            perimeter_texture_expolygons_overlapping_bbox(upper_layer->lslices, &upper_layer->lslices_bboxes, bbox) :
            upper_layer->lslices;
        if (!overlapping.empty())
            append(out, std::move(overlapping));
    }
    return out.empty() ? ExPolygons() : union_ex(out);
}

static void perimeter_texture_collect_external_widths(const ExtrusionEntity &entity, std::vector<float> &widths)
{
    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity)) {
        if (is_external_perimeter(path->role()) && std::isfinite(path->width) && path->width > 0.f)
            widths.emplace_back(path->width);
    } else if (const ExtrusionMultiPath *multipath = dynamic_cast<const ExtrusionMultiPath *>(&entity)) {
        for (const ExtrusionPath &path : multipath->paths)
            if (is_external_perimeter(path.role()) && std::isfinite(path.width) && path.width > 0.f)
                widths.emplace_back(path.width);
    } else if (const ExtrusionLoop *loop = dynamic_cast<const ExtrusionLoop *>(&entity)) {
        for (const ExtrusionPath &path : loop->paths)
            if (is_external_perimeter(path.role()) && std::isfinite(path.width) && path.width > 0.f)
                widths.emplace_back(path.width);
    } else if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity)) {
        for (const ExtrusionEntity *child : collection->entities) {
            if (child != nullptr)
                perimeter_texture_collect_external_widths(*child, widths);
        }
    }
}

static std::optional<float> perimeter_texture_min_external_width(const ExtrusionEntity &entity)
{
    std::vector<float> widths;
    perimeter_texture_collect_external_widths(entity, widths);
    if (widths.empty())
        return std::nullopt;
    return *std::min_element(widths.begin(), widths.end());
}

static std::optional<float> perimeter_texture_min_external_width(const ExtrusionEntityCollection &collection)
{
    return perimeter_texture_min_external_width(static_cast<const ExtrusionEntity &>(collection));
}

static void perimeter_texture_collect_external_covered_polygons(const ExtrusionEntity &entity, Polygons &out)
{
    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity)) {
        if (is_external_perimeter(path->role()))
            path->polygons_covered_by_width(out, 0.f);
    } else if (const ExtrusionMultiPath *multipath = dynamic_cast<const ExtrusionMultiPath *>(&entity)) {
        for (const ExtrusionPath &path : multipath->paths)
            if (is_external_perimeter(path.role()))
                path.polygons_covered_by_width(out, 0.f);
    } else if (const ExtrusionLoop *loop = dynamic_cast<const ExtrusionLoop *>(&entity)) {
        for (const ExtrusionPath &path : loop->paths)
            if (is_external_perimeter(path.role()))
                path.polygons_covered_by_width(out, 0.f);
    } else if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity)) {
        for (const ExtrusionEntity *child : collection->entities)
            if (child != nullptr)
                perimeter_texture_collect_external_covered_polygons(*child, out);
    }
}

static ExPolygons perimeter_texture_external_visible_footprint(const LayerRegion &layer_region, const ExtrusionEntity &entity)
{
    Polygons covered;
    perimeter_texture_collect_external_covered_polygons(entity, covered);
    if (covered.empty())
        return {};

    ExPolygons footprint = union_ex(covered);
    if (footprint.empty())
        return {};

    const Layer *layer = layer_region.layer();
    if (layer == nullptr || layer->upper_layer == nullptr || layer->upper_layer->lslices.empty())
        return footprint;

    const BoundingBox footprint_bbox = get_extents(footprint);
    const ExPolygons upper_slices = footprint_bbox.defined ?
        perimeter_texture_expolygons_overlapping_bbox(layer->upper_layer->lslices,
                                                      &layer->upper_layer->lslices_bboxes,
                                                      footprint_bbox) :
        layer->upper_layer->lslices;
    if (upper_slices.empty())
        return footprint;

    ExPolygons visible = diff_ex(footprint, upper_slices);
    return visible.empty() ? footprint : visible;
}

template <typename Fn>
static bool perimeter_texture_sample_expolygons(const ExPolygons &expolygons, float pitch_mm, Fn fn)
{
    bool sampled = false;
    const double pitch_scaled = std::max(1.0, double(scale_(std::max(0.02f, pitch_mm))));
    for (const ExPolygon &expolygon : expolygons) {
        const BoundingBox bbox = get_extents(expolygon);
        if (!bbox.defined)
            continue;

        const double width = double(bbox.max.x()) - double(bbox.min.x());
        const double height = double(bbox.max.y()) - double(bbox.min.y());
        int nx = std::max(1, int(std::ceil(width / pitch_scaled)));
        int ny = std::max(1, int(std::ceil(height / pitch_scaled)));
        constexpr size_t max_samples = 4096;
        while (size_t(nx) * size_t(ny) > max_samples) {
            if (nx >= ny && nx > 1)
                nx = (nx + 1) / 2;
            else if (ny > 1)
                ny = (ny + 1) / 2;
            else
                break;
        }

        const double step_x = width / double(nx);
        const double step_y = height / double(ny);
        const double weight = std::max(1e-6, unscale<double>(step_x) * unscale<double>(step_y));
        for (int ix = 0; ix < nx; ++ix) {
            const double x = double(bbox.min.x()) + (double(ix) + 0.5) * step_x;
            for (int iy = 0; iy < ny; ++iy) {
                const double y = double(bbox.min.y()) + (double(iy) + 0.5) * step_y;
                const Point point(coord_t(std::llround(x)), coord_t(std::llround(y)));
                if (!expolygon.contains(point, true))
                    continue;
                fn(point, weight);
                sampled = true;
            }
        }
    }
    return sampled;
}

static bool perimeter_texture_accumulate_image_weights_at_point(const TextureMappingOffsetContext &context,
                                                                const Point                       &point,
                                                                double                             weight,
                                                                std::vector<double>               &accum,
                                                                double                            &total_weight)
{
    std::vector<float> weights = sample_weight_field_components(context.weight_field,
                                                                unscale<float>(point.x()),
                                                                unscale<float>(point.y()),
                                                                context.high_resolution_texture_sampling);
    if (weights.empty())
        return false;

    const size_t count = std::min(accum.size(), weights.size());
    if (count == 0)
        return false;

    for (size_t idx = 0; idx < count; ++idx)
        accum[idx] += double(std::clamp(weights[idx], 0.f, 1.f)) * weight;
    total_weight += weight;
    return true;
}

static bool perimeter_texture_accumulate_visible_image_weights(const ExPolygons                  &visible,
                                                               const TextureMappingOffsetContext &context,
                                                               std::vector<double>               &accum,
                                                               double                            &total_weight)
{
    bool accumulated = false;
    const float pitch_mm = context.high_resolution_texture_sampling ? 0.08f : 0.16f;
    perimeter_texture_sample_expolygons(visible, pitch_mm, [&](const Point &point, double weight) {
        accumulated |= perimeter_texture_accumulate_image_weights_at_point(context, point, weight, accum, total_weight);
    });
    return accumulated;
}

static bool perimeter_texture_accumulate_image_rgb_at_point(const TextureMappingOffsetContext &context,
                                                            const Point                       &point,
                                                            double                             weight,
                                                            std::array<double, 3>             &accum,
                                                            double                            &total_weight)
{
    std::optional<std::array<float, 3>> rgb = sample_weight_field_rgb(context.weight_field,
                                                                      unscale<float>(point.x()),
                                                                      unscale<float>(point.y()),
                                                                      context.high_resolution_texture_sampling);
    if (!rgb)
        return false;

    accum[0] += double((*rgb)[0]) * weight;
    accum[1] += double((*rgb)[1]) * weight;
    accum[2] += double((*rgb)[2]) * weight;
    total_weight += weight;
    return true;
}

static std::optional<std::array<float, 3>> perimeter_texture_average_visible_image_rgb(
    const ExPolygons                  &visible,
    const TextureMappingOffsetContext &context)
{
    if (visible.empty())
        return std::nullopt;

    std::array<double, 3> accum{ 0.0, 0.0, 0.0 };
    double total_weight = 0.0;
    const float pitch_mm = context.high_resolution_texture_sampling ? 0.08f : 0.16f;
    perimeter_texture_sample_expolygons(visible, pitch_mm, [&](const Point &point, double weight) {
        perimeter_texture_accumulate_image_rgb_at_point(context, point, weight, accum, total_weight);
    });
    if (total_weight <= EPSILON)
        return std::nullopt;
    return std::array<float, 3>{ float(std::clamp(accum[0] / total_weight, 0.0, 1.0)),
                                 float(std::clamp(accum[1] / total_weight, 0.0, 1.0)),
                                 float(std::clamp(accum[2] / total_weight, 0.0, 1.0)) };
}

static void perimeter_texture_accumulate_path_image_weights(const ExtrusionEntity                &entity,
                                                            const TextureMappingOffsetContext    &context,
                                                            std::vector<double>                  &accum,
                                                            double                               &total_weight);

static void perimeter_texture_accumulate_path_image_weights_for_path(const ExtrusionPath              &path,
                                                                     const TextureMappingOffsetContext &context,
                                                                     std::vector<double>             &accum,
                                                                     double                          &total_weight)
{
    if (!is_external_perimeter(path.role()) || path.polyline.points.size() < 2)
        return;

    const double pitch_scaled = std::max(1.0, double(scale_(context.high_resolution_texture_sampling ? 0.08 : 0.16)));
    for (size_t idx = 1; idx < path.polyline.points.size(); ++idx) {
        const Point &a = path.polyline.points[idx - 1];
        const Point &b = path.polyline.points[idx];
        const double dx = double(b.x()) - double(a.x());
        const double dy = double(b.y()) - double(a.y());
        const double len = std::hypot(dx, dy);
        if (!std::isfinite(len) || len <= EPSILON)
            continue;
        const int sample_count = std::clamp(int(std::ceil(len / pitch_scaled)), 1, 1024);
        const double weight = std::max(1e-6, unscale<double>(len) / double(sample_count));
        for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
            const double t = (double(sample_idx) + 0.5) / double(sample_count);
            const Point point(coord_t(std::llround(double(a.x()) + dx * t)),
                              coord_t(std::llround(double(a.y()) + dy * t)));
            perimeter_texture_accumulate_image_weights_at_point(context, point, weight, accum, total_weight);
        }
    }
}

static void perimeter_texture_accumulate_path_image_weights(const ExtrusionEntity             &entity,
                                                            const TextureMappingOffsetContext &context,
                                                            std::vector<double>               &accum,
                                                            double                            &total_weight)
{
    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity)) {
        perimeter_texture_accumulate_path_image_weights_for_path(*path, context, accum, total_weight);
    } else if (const ExtrusionMultiPath *multipath = dynamic_cast<const ExtrusionMultiPath *>(&entity)) {
        for (const ExtrusionPath &path : multipath->paths)
            perimeter_texture_accumulate_path_image_weights_for_path(path, context, accum, total_weight);
    } else if (const ExtrusionLoop *loop = dynamic_cast<const ExtrusionLoop *>(&entity)) {
        for (const ExtrusionPath &path : loop->paths)
            perimeter_texture_accumulate_path_image_weights_for_path(path, context, accum, total_weight);
    } else if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity)) {
        for (const ExtrusionEntity *child : collection->entities)
            if (child != nullptr)
                perimeter_texture_accumulate_path_image_weights(*child, context, accum, total_weight);
    }
}

static std::optional<unsigned int> perimeter_texture_nearest_component_color(
    const std::array<float, 3>                  &target,
    const std::vector<unsigned int>             &component_ids,
    const std::vector<std::array<float, 3>>     &component_colors)
{
    if (component_ids.empty() || component_ids.size() != component_colors.size())
        return std::nullopt;

    size_t best_idx = size_t(-1);
    double best_distance = std::numeric_limits<double>::max();
    for (size_t idx = 0; idx < component_ids.size(); ++idx) {
        const double dr = double(target[0]) - double(component_colors[idx][0]);
        const double dg = double(target[1]) - double(component_colors[idx][1]);
        const double db = double(target[2]) - double(component_colors[idx][2]);
        const double distance = dr * dr + dg * dg + db * db;
        if (distance < best_distance) {
            best_distance = distance;
            best_idx = idx;
        }
    }

    if (best_idx >= component_ids.size())
        return std::nullopt;
    return component_ids[best_idx];
}

static std::optional<unsigned int> perimeter_texture_choose_image_recolor_component(
    const ExPolygons                   &visible,
    const ExtrusionEntity              *fallback_entity,
    const TextureMappingOffsetContext  &context,
    const std::vector<std::array<float, 3>> &component_colors)
{
    if (context.component_ids.empty())
        return std::nullopt;

    if (!context.weight_field.raw_component_weights_from_texture) {
        std::optional<std::array<float, 3>> rgb =
            perimeter_texture_average_visible_image_rgb(visible, context);
        if (rgb) {
            std::optional<unsigned int> nearest =
                perimeter_texture_nearest_component_color(*rgb, context.component_ids, component_colors);
            if (nearest)
                return nearest;
        }
    }

    std::vector<double> accum(context.component_ids.size(), 0.0);
    double total_weight = 0.0;
    if (!visible.empty())
        perimeter_texture_accumulate_visible_image_weights(visible, context, accum, total_weight);
    if (total_weight <= EPSILON && fallback_entity != nullptr)
        perimeter_texture_accumulate_path_image_weights(*fallback_entity, context, accum, total_weight);
    if (total_weight > EPSILON) {
        const auto best_it = std::max_element(accum.begin(), accum.end());
        if (best_it != accum.end()) {
            const size_t best_idx = size_t(best_it - accum.begin());
            if (best_idx < context.component_ids.size())
                return context.component_ids[best_idx];
        }
    }

    return std::nullopt;
}

static bool perimeter_texture_accumulate_visible_gradient_scores(const ExPolygons                               &visible,
                                                                 const std::vector<TextureMappingOffsetContext> &contexts,
                                                                 std::vector<double>                           &scores,
                                                                 double                                        &total_weight)
{
    if (contexts.empty())
        return false;

    bool accumulated = false;
    const float pitch_mm = contexts.front().high_resolution_texture_sampling ? 0.08f : 0.16f;
    perimeter_texture_sample_expolygons(visible, pitch_mm, [&](const Point &point, double weight) {
        const double radial_x = double(point.x()) - double(contexts.front().object_center.x());
        const double radial_y = double(point.y()) - double(contexts.front().object_center.y());
        const double radial_len = std::hypot(radial_x, radial_y);
        if (!std::isfinite(radial_len) || radial_len <= EPSILON)
            return;
        const double inward_x = -radial_x / radial_len;
        const double inward_y = -radial_y / radial_len;
        for (size_t idx = 0; idx < contexts.size(); ++idx) {
            const float inset = texture_mapping_offset_surface_inset_mm(contexts[idx], point, inward_x, inward_y);
            const float denom = std::max(contexts[idx].max_width_delta_mm, float(EPSILON));
            scores[idx] += double(std::clamp(1.f - inset / denom, 0.f, 1.f)) * weight;
        }
        total_weight += weight;
        accumulated = true;
    });
    return accumulated;
}

static std::optional<unsigned int> perimeter_texture_choose_gradient_recolor_component(
    const ExPolygons                  &visible,
    const std::vector<unsigned int>   &component_ids,
    const std::vector<TextureMappingOffsetContext> &contexts)
{
    if (component_ids.empty() || component_ids.size() != contexts.size())
        return std::nullopt;

    std::vector<double> scores(component_ids.size(), 0.0);
    double total_weight = 0.0;
    if (!visible.empty())
        perimeter_texture_accumulate_visible_gradient_scores(visible, contexts, scores, total_weight);
    if (total_weight <= EPSILON)
        return std::nullopt;

    const auto best_it = std::max_element(scores.begin(), scores.end());
    if (best_it == scores.end())
        return std::nullopt;
    const size_t best_idx = size_t(best_it - scores.begin());
    if (best_idx >= component_ids.size())
        return std::nullopt;
    return component_ids[best_idx];
}

struct PerimeterTextureRecolorSampler {
    bool image_texture { false };
    size_t num_physical { 0 };
    std::optional<TextureMappingOffsetContext> image_context;
    std::vector<std::array<float, 3>> image_component_colors;
    std::vector<unsigned int> component_ids;
    std::vector<TextureMappingOffsetContext> gradient_contexts;
};

static std::optional<PerimeterTextureRecolorSampler> perimeter_texture_make_recolor_sampler(
    const LayerRegion        &layer_region,
    const TextureMappingZone &zone,
    unsigned int             texture_zone_id,
    float                    base_outer_width_mm)
{
    const Layer *layer = layer_region.layer();
    if (layer == nullptr)
        return std::nullopt;

    const PrintObject *print_object = layer->object();
    if (print_object == nullptr)
        return std::nullopt;

    const Print *print = print_object->print();
    if (print == nullptr || print->canceled())
        return std::nullopt;

    const PrintConfig &print_config = print->config();
    const size_t num_physical = print_config.filament_colour.values.size();
    if (num_physical == 0)
        return std::nullopt;

    std::optional<TextureMappingOffsetContext> context =
        build_texture_mapping_offset_context_for_layer(*print_object, *layer, zone, texture_zone_id, 0, base_outer_width_mm);
    if (!context || print->canceled() || context->component_ids.empty())
        return std::nullopt;

    PerimeterTextureRecolorSampler sampler;
    sampler.image_texture = zone.is_image_texture();
    sampler.num_physical = num_physical;
    if (sampler.image_texture) {
        sampler.image_context = std::move(*context);
        sampler.image_component_colors.reserve(sampler.image_context->component_ids.size());
        for (const unsigned int id : sampler.image_context->component_ids) {
            ColorRGB decoded;
            if (id >= 1 && id <= print_config.filament_colour.values.size() &&
                decode_color(print_config.filament_colour.get_at(size_t(id - 1)), decoded)) {
                sampler.image_component_colors.push_back({ decoded.r(), decoded.g(), decoded.b() });
            } else {
                sampler.image_component_colors.push_back({ 0.f, 0.f, 0.f });
            }
        }
        return sampler;
    }

    sampler.component_ids.reserve(context->component_ids.size());
    sampler.gradient_contexts.reserve(context->component_ids.size());
    for (const unsigned int component_id : context->component_ids) {
        if (print->canceled())
            return std::nullopt;
        if (component_id < 1 || component_id > sampler.num_physical)
            continue;
        std::optional<TextureMappingOffsetContext> component_context =
            build_texture_mapping_offset_context_for_layer(*print_object, *layer, zone, texture_zone_id, component_id, base_outer_width_mm);
        if (!component_context)
            continue;
        sampler.component_ids.emplace_back(component_id);
        sampler.gradient_contexts.emplace_back(std::move(*component_context));
    }
    if (sampler.component_ids.empty() || sampler.component_ids.size() != sampler.gradient_contexts.size())
        return std::nullopt;
    return sampler;
}

static std::optional<unsigned int> perimeter_texture_choose_recolor_component_with_sampler(
    const ExPolygons                       &visible,
    const ExtrusionEntity                  *fallback_entity,
    const PerimeterTextureRecolorSampler   &sampler)
{
    if (sampler.image_texture) {
        if (!sampler.image_context)
            return std::nullopt;
        std::optional<unsigned int> chosen =
            perimeter_texture_choose_image_recolor_component(visible,
                                                             fallback_entity,
                                                             *sampler.image_context,
                                                             sampler.image_component_colors);
        if (chosen && *chosen >= 1 && *chosen <= sampler.num_physical)
            return chosen;
        return std::nullopt;
    }

    return perimeter_texture_choose_gradient_recolor_component(visible,
                                                               sampler.component_ids,
                                                               sampler.gradient_contexts);
}

static void perimeter_texture_move_reusable_offset_context(PerimeterTextureRecolorSampler             &sampler,
                                                           unsigned int                                active_component_id,
                                                           std::optional<TextureMappingOffsetContext> *reusable_context)
{
    if (reusable_context == nullptr)
        return;

    reusable_context->reset();
    if (sampler.image_texture) {
        if (sampler.image_context)
            *reusable_context = std::move(*sampler.image_context);
        return;
    }

    for (TextureMappingOffsetContext &context : sampler.gradient_contexts) {
        if (context.active_component_id == active_component_id) {
            *reusable_context = std::move(context);
            return;
        }
    }
}

static std::optional<unsigned int> perimeter_texture_choose_recolor_component_for_visible(
    const LayerRegion        &layer_region,
    const ExPolygons         &visible,
    const ExtrusionEntity    *fallback_entity,
    const TextureMappingZone &zone,
    unsigned int             texture_zone_id,
    float                    base_outer_width_mm)
{
    const Layer *layer = layer_region.layer();
    if (layer == nullptr || layer->object() == nullptr || layer->object()->print() == nullptr)
        return std::nullopt;

    std::optional<PerimeterTextureRecolorSampler> sampler =
        perimeter_texture_make_recolor_sampler(layer_region, zone, texture_zone_id, base_outer_width_mm);
    if (!sampler)
        return std::nullopt;

    return perimeter_texture_choose_recolor_component_with_sampler(visible, fallback_entity, *sampler);
}

static std::optional<unsigned int> perimeter_texture_choose_recolor_component(const LayerRegion        &layer_region,
                                                                              const ExtrusionEntity    &entity,
                                                                              const TextureMappingZone &zone,
                                                                              unsigned int             texture_zone_id,
                                                                              float                    base_outer_width_mm)
{
    const ExPolygons visible = perimeter_texture_external_visible_footprint(layer_region, entity);
    return perimeter_texture_choose_recolor_component_for_visible(layer_region,
                                                                  visible,
                                                                  &entity,
                                                                  zone,
                                                                  texture_zone_id,
                                                                  base_outer_width_mm);
}

static bool perimeter_texture_apply_recolor_small_perimeter_loops(LayerRegion              &layer_region,
                                                                  const TextureMappingZone &zone,
                                                                  unsigned int             texture_zone_id,
                                                                  float                    texture_external_width_mm)
{
    bool saw_reduced_loop = false;
    bool all_reduced_loops_recolored = true;
    for (ExtrusionEntity *entity : layer_region.perimeters.entities) {
        ExtrusionEntityCollection *collection = dynamic_cast<ExtrusionEntityCollection *>(entity);
        if (collection == nullptr)
            continue;

        const std::optional<float> min_width = perimeter_texture_min_external_width(*collection);
        if (!min_width || *min_width >= texture_external_width_mm - float(EPSILON))
            continue;

        saw_reduced_loop = true;
        std::optional<unsigned int> component_id =
            perimeter_texture_choose_recolor_component(layer_region, *collection, zone, texture_zone_id, *min_width);
        if (component_id && *component_id > 0)
            collection->texture_mapping_extruder_override = int(*component_id) - 1;
        else
            all_reduced_loops_recolored = false;
    }
    return saw_reduced_loop && all_reduced_loops_recolored;
}

struct PerimeterTextureTopVisibleRecolorThresholds {
    float visible_fraction { 0.60f };
    float min_run_length_mm { 2.0f };
    float min_visible_area_mm2 { 1.0f };
    float merge_gap_mm { 0.4f };
};

struct PerimeterTextureVisiblePointSample {
    Point  point;
    double weight { 0.0 };
    double inward_x { 0.0 };
    double inward_y { 0.0 };
};

static PerimeterTextureTopVisibleRecolorThresholds perimeter_texture_top_visible_recolor_thresholds(int aggressiveness)
{
    switch (std::clamp(aggressiveness,
                       int(TextureMappingZone::TopVisibleRecolorConservative),
                       int(TextureMappingZone::TopVisibleRecolorAggressive))) {
    case int(TextureMappingZone::TopVisibleRecolorAggressive):
        return { 0.12f, 0.3f, 0.05f, 1.2f };
    case int(TextureMappingZone::TopVisibleRecolorBalanced):
        return { 0.40f, 1.2f, 0.5f, 0.6f };
    default:
        return {};
    }
}

static double perimeter_texture_scaled_area_mm2(double scaled_area)
{
    return std::abs(scaled_area) * SCALING_FACTOR * SCALING_FACTOR;
}

static ExPolygons perimeter_texture_top_visible_wall_band_mask(const LayerRegion       &layer_region,
                                                               const SurfaceCollection &slices,
                                                               float                    wall_depth_mm,
                                                               int                      above_layer_count,
                                                               ExPolygons              *wall_band_out = nullptr)
{
    const Layer *layer = layer_region.layer();
    if (layer == nullptr)
        return {};

    ExPolygons out;
    ExPolygons wall_band_all;
    const float wall_depth_scaled = float(scale_(std::max(0.02f, wall_depth_mm)));
    for (const Surface &surface : slices.surfaces) {
        if (surface.expolygon.empty())
            continue;

        ExPolygons visible;
        const ExPolygons upper_slices =
            perimeter_texture_above_layer_slices_overlapping_bbox(layer, get_extents(surface.expolygon), above_layer_count);
        if (upper_slices.empty())
            visible.emplace_back(surface.expolygon);
        else
            visible = diff_ex(surface.expolygon, upper_slices);
        if (visible.empty())
            continue;

        ExPolygons inner = offset_ex(surface.expolygon, -wall_depth_scaled);
        ExPolygons surface_only{ surface.expolygon };
        ExPolygons wall_band = inner.empty() ? surface_only : diff_ex(surface_only, inner);
        if (wall_band.empty())
            continue;

        append(wall_band_all, wall_band);
        ExPolygons clipped = intersection_ex(visible, wall_band);
        out.insert(out.end(), clipped.begin(), clipped.end());
    }

    if (wall_band_out != nullptr)
        *wall_band_out = wall_band_all.empty() ? ExPolygons() : union_ex(wall_band_all);
    if (!out.empty())
        out = union_ex(out);
    return out;
}

static PerimeterTextureMaskIndex perimeter_texture_make_mask_index(const ExPolygons *mask)
{
    PerimeterTextureMaskIndex out;
    if (mask == nullptr || mask->empty())
        return out;

    out.expolygons = mask;
    out.bboxes.reserve(mask->size());
    for (const ExPolygon &expolygon : *mask) {
        BoundingBox bbox = get_extents(expolygon);
        out.bboxes.emplace_back(bbox);
        if (bbox.defined)
            out.bbox.merge(bbox);
    }
    return out;
}

static bool perimeter_texture_mask_index_contains_point(const PerimeterTextureMaskIndex &mask, const Point &point)
{
    if (mask.empty() || !mask.bbox.contains(point))
        return false;

    for (size_t idx = 0; idx < mask.expolygons->size() && idx < mask.bboxes.size(); ++idx) {
        if (mask.bboxes[idx].defined &&
            mask.bboxes[idx].contains(point) &&
            (*mask.expolygons)[idx].contains(point, true))
            return true;
    }
    return false;
}

static ExPolygons perimeter_texture_mask_index_overlapping_expolygons(const PerimeterTextureMaskIndex &mask,
                                                                      const BoundingBox              &bbox)
{
    ExPolygons out;
    if (mask.empty() || !bbox.defined || !mask.bbox.overlap(bbox))
        return out;

    for (size_t idx = 0; idx < mask.expolygons->size() && idx < mask.bboxes.size(); ++idx) {
        if (mask.bboxes[idx].defined && mask.bboxes[idx].overlap(bbox))
            out.emplace_back((*mask.expolygons)[idx]);
    }
    return out;
}

static Polygons perimeter_texture_mask_index_clipped_polygons(const PerimeterTextureMaskIndex &mask,
                                                              const BoundingBox              &bbox)
{
    Polygons out;
    if (mask.empty() || !bbox.defined || !mask.bbox.overlap(bbox))
        return out;

    const BoundingBox clip_bbox = bbox.inflated(SCALED_EPSILON);
    for (size_t idx = 0; idx < mask.expolygons->size() && idx < mask.bboxes.size(); ++idx) {
        if (!mask.bboxes[idx].defined || !mask.bboxes[idx].overlap(bbox))
            continue;
        Polygons clipped = ClipperUtils::clip_clipper_polygons_with_subject_bbox((*mask.expolygons)[idx], clip_bbox);
        polygons_append(out, std::move(clipped));
    }
    return out;
}

static bool perimeter_texture_expolygons_contain_point(const ExPolygons &expolygons, const Point &point)
{
    return std::any_of(expolygons.begin(), expolygons.end(), [&point](const ExPolygon &expolygon) {
        return expolygon.contains(point, true);
    });
}

static bool perimeter_texture_sample_is_top_visible_recolor_protected(const PerimeterTextureMaskIndex               *mask,
                                                                      const PerimeterPathBoundarySample            &sample,
                                                                      double                                       tangent_x,
                                                                      double                                       tangent_y,
                                                                      float                                        max_inset_mm,
                                                                      const PerimeterTextureTopVisibleRecolorThresholds *thresholds)
{
    if (mask == nullptr || mask->empty() || thresholds == nullptr)
        return false;

    auto contains_at_inset = [mask, &sample](float inset_mm) {
        const double inset_scaled = scale_(double(inset_mm));
        const Point candidate(coord_t(std::llround(double(sample.point.x()) + sample.inward_x * inset_scaled)),
                              coord_t(std::llround(double(sample.point.y()) + sample.inward_y * inset_scaled)));
        return perimeter_texture_mask_index_contains_point(*mask, candidate);
    };

    if (!contains_at_inset(0.f) &&
        !contains_at_inset(std::max(0.02f, 0.5f * max_inset_mm)) &&
        !contains_at_inset(std::max(0.02f, max_inset_mm)))
        return false;

    const double depth_scaled = scale_(std::max(0.05f, max_inset_mm));
    const double half_span_scaled = scale_(0.5 * double(std::clamp(max_inset_mm, 0.20f, 0.80f)));
    Polygon footprint;
    footprint.points.reserve(4);
    footprint.points.emplace_back(Point(coord_t(std::llround(double(sample.point.x()) - tangent_x * half_span_scaled)),
                                        coord_t(std::llround(double(sample.point.y()) - tangent_y * half_span_scaled))));
    footprint.points.emplace_back(Point(coord_t(std::llround(double(sample.point.x()) + tangent_x * half_span_scaled)),
                                        coord_t(std::llround(double(sample.point.y()) + tangent_y * half_span_scaled))));
    footprint.points.emplace_back(Point(coord_t(std::llround(double(sample.point.x()) + tangent_x * half_span_scaled + sample.inward_x * depth_scaled)),
                                        coord_t(std::llround(double(sample.point.y()) + tangent_y * half_span_scaled + sample.inward_y * depth_scaled))));
    footprint.points.emplace_back(Point(coord_t(std::llround(double(sample.point.x()) - tangent_x * half_span_scaled + sample.inward_x * depth_scaled)),
                                        coord_t(std::llround(double(sample.point.y()) - tangent_y * half_span_scaled + sample.inward_y * depth_scaled))));
    remove_same_neighbor(footprint);
    if (!footprint.is_valid())
        return false;

    const double footprint_area_mm2 = perimeter_texture_scaled_area_mm2(footprint.area());
    if (footprint_area_mm2 <= EPSILON)
        return false;

    Polygons local_mask = perimeter_texture_mask_index_clipped_polygons(*mask, get_extents(footprint));
    if (local_mask.empty())
        return false;

    Polygons visible = intersection(Polygons{ footprint }, local_mask);
    const double visible_area_mm2 = visible.empty() ? 0.0 : perimeter_texture_scaled_area_mm2(area(visible));
    return visible_area_mm2 / footprint_area_mm2 >= thresholds->visible_fraction;
}

static Polygons perimeter_texture_segment_visible_footprint(float             depth,
                                                           const Point      &a,
                                                           const Point      &b,
                                                           const PerimeterTextureMaskIndex &top_visible_mask,
                                                           double           &footprint_area_mm2)
{
    footprint_area_mm2 = 0.0;
    if (a == b || top_visible_mask.empty() || !std::isfinite(depth) || depth <= 0.f)
        return {};

    const double dx = double(b.x()) - double(a.x());
    const double dy = double(b.y()) - double(a.y());
    const double len = std::hypot(dx, dy);
    if (!std::isfinite(len) || len <= EPSILON)
        return {};

    const double inward_x = -dy / len;
    const double inward_y = dx / len;
    const double depth_scaled = scale_(double(depth));
    Polygon footprint;
    footprint.points.reserve(4);
    footprint.points.emplace_back(a);
    footprint.points.emplace_back(b);
    footprint.points.emplace_back(Point(coord_t(std::llround(double(b.x()) + inward_x * depth_scaled)),
                                        coord_t(std::llround(double(b.y()) + inward_y * depth_scaled))));
    footprint.points.emplace_back(Point(coord_t(std::llround(double(a.x()) + inward_x * depth_scaled)),
                                        coord_t(std::llround(double(a.y()) + inward_y * depth_scaled))));
    remove_same_neighbor(footprint);
    if (!footprint.is_valid())
        return {};

    footprint_area_mm2 = perimeter_texture_scaled_area_mm2(footprint.area());
    if (footprint_area_mm2 <= EPSILON)
        return {};

    Polygons local_mask = perimeter_texture_mask_index_clipped_polygons(top_visible_mask, get_extents(footprint));
    if (local_mask.empty())
        return {};

    Polygons footprint_visible = intersection(Polygons{ footprint }, local_mask);
    if (footprint_visible.empty())
        return {};

    return footprint_visible;
}

static void perimeter_texture_sample_segment_visible_points(float                              depth,
                                                            const Point                       &a,
                                                            const Point                       &b,
                                                            const PerimeterTextureMaskIndex   &top_visible_mask,
                                                            std::vector<PerimeterTextureVisiblePointSample> &visible_samples,
                                                            double                            &footprint_area_mm2,
                                                            double                            &visible_area_mm2)
{
    footprint_area_mm2 = 0.0;
    visible_area_mm2 = 0.0;
    visible_samples.clear();
    if (a == b || top_visible_mask.empty() || !std::isfinite(depth) || depth <= 0.f)
        return;

    const double dx = double(b.x()) - double(a.x());
    const double dy = double(b.y()) - double(a.y());
    const double len_scaled = std::hypot(dx, dy);
    if (!std::isfinite(len_scaled) || len_scaled <= EPSILON)
        return;

    const double depth_mm = std::max(0.02, double(depth));
    const double len_mm = unscale<double>(len_scaled);
    footprint_area_mm2 = len_mm * depth_mm;
    if (footprint_area_mm2 <= EPSILON)
        return;

    const double tangent_x = dx / len_scaled;
    const double tangent_y = dy / len_scaled;
    const double inward_x = -tangent_y;
    const double inward_y = tangent_x;
    const double depth_scaled = scale_(depth_mm);
    const Point c(coord_t(std::llround(double(b.x()) + inward_x * depth_scaled)),
                  coord_t(std::llround(double(b.y()) + inward_y * depth_scaled)));
    const Point d(coord_t(std::llround(double(a.x()) + inward_x * depth_scaled)),
                  coord_t(std::llround(double(a.y()) + inward_y * depth_scaled)));
    BoundingBox footprint_bbox;
    footprint_bbox.merge(a);
    footprint_bbox.merge(b);
    footprint_bbox.merge(c);
    footprint_bbox.merge(d);
    if (!footprint_bbox.defined || !top_visible_mask.bbox.overlap(footprint_bbox))
        return;

    const int along_count = std::clamp(int(std::ceil(len_mm / 0.08)), 1, 4096);
    const int across_count = std::clamp(int(std::ceil(depth_mm / 0.15)), 1, 16);
    const double weight = footprint_area_mm2 / double(along_count * across_count);
    visible_samples.reserve(size_t(along_count * across_count));
    for (int along_idx = 0; along_idx < along_count; ++along_idx) {
        const double t = (double(along_idx) + 0.5) / double(along_count);
        const double base_x = double(a.x()) + dx * t;
        const double base_y = double(a.y()) + dy * t;
        for (int across_idx = 0; across_idx < across_count; ++across_idx) {
            const double u = (double(across_idx) + 0.5) / double(across_count);
            const Point sample(coord_t(std::llround(base_x + inward_x * depth_scaled * u)),
                               coord_t(std::llround(base_y + inward_y * depth_scaled * u)));
            if (!perimeter_texture_mask_index_contains_point(top_visible_mask, sample))
                continue;
            visible_area_mm2 += weight;
            visible_samples.push_back(PerimeterTextureVisiblePointSample{ sample, weight, inward_x, inward_y });
        }
    }
}

static std::optional<unsigned int> perimeter_texture_choose_recolor_component_from_point_samples(
    const std::vector<PerimeterTextureVisiblePointSample> &samples,
    const PerimeterTextureRecolorSampler                  &sampler)
{
    if (samples.empty())
        return std::nullopt;

    if (sampler.image_texture) {
        if (!sampler.image_context)
            return std::nullopt;
        if (!sampler.image_context->weight_field.raw_component_weights_from_texture) {
            std::array<double, 3> rgb_accum{ 0.0, 0.0, 0.0 };
            double rgb_total_weight = 0.0;
            for (const PerimeterTextureVisiblePointSample &sample : samples)
                perimeter_texture_accumulate_image_rgb_at_point(*sampler.image_context, sample.point, sample.weight, rgb_accum, rgb_total_weight);
            if (rgb_total_weight > EPSILON) {
                std::optional<unsigned int> nearest =
                    perimeter_texture_nearest_component_color(
                        std::array<float, 3>{ float(std::clamp(rgb_accum[0] / rgb_total_weight, 0.0, 1.0)),
                                              float(std::clamp(rgb_accum[1] / rgb_total_weight, 0.0, 1.0)),
                                              float(std::clamp(rgb_accum[2] / rgb_total_weight, 0.0, 1.0)) },
                        sampler.image_context->component_ids,
                        sampler.image_component_colors);
                if (nearest && *nearest >= 1 && *nearest <= sampler.num_physical)
                    return nearest;
            }
        }

        std::vector<double> accum(sampler.image_context->component_ids.size(), 0.0);
        double total_weight = 0.0;
        for (const PerimeterTextureVisiblePointSample &sample : samples)
            perimeter_texture_accumulate_image_weights_at_point(*sampler.image_context, sample.point, sample.weight, accum, total_weight);
        if (total_weight > EPSILON) {
            const auto best_it = std::max_element(accum.begin(), accum.end());
            if (best_it != accum.end()) {
                const size_t best_idx = size_t(best_it - accum.begin());
                if (best_idx < sampler.image_context->component_ids.size()) {
                    const unsigned int component_id = sampler.image_context->component_ids[best_idx];
                    if (component_id >= 1 && component_id <= sampler.num_physical)
                        return component_id;
                }
            }
        }
        return std::nullopt;
    }

    if (sampler.component_ids.empty() || sampler.component_ids.size() != sampler.gradient_contexts.size())
        return std::nullopt;
    std::vector<double> scores(sampler.component_ids.size(), 0.0);
    double total_weight = 0.0;
    for (const PerimeterTextureVisiblePointSample &sample : samples) {
        if (sample.weight <= EPSILON)
            continue;
        for (size_t idx = 0; idx < sampler.gradient_contexts.size(); ++idx) {
            const TextureMappingOffsetContext &context = sampler.gradient_contexts[idx];
            const float inset = texture_mapping_offset_surface_inset_mm(context, sample.point, sample.inward_x, sample.inward_y);
            const float denom = std::max(context.max_width_delta_mm, float(EPSILON));
            scores[idx] += double(std::clamp(1.f - inset / denom, 0.f, 1.f)) * sample.weight;
        }
        total_weight += sample.weight;
    }
    if (total_weight <= EPSILON)
        return std::nullopt;
    const auto best_it = std::max_element(scores.begin(), scores.end());
    if (best_it == scores.end())
        return std::nullopt;
    const size_t best_idx = size_t(best_it - scores.begin());
    if (best_idx >= sampler.component_ids.size())
        return std::nullopt;
    return sampler.component_ids[best_idx];
}

static void perimeter_texture_append_colored_line(ColoredLines &lines, const Line &line, int color)
{
    if (line.a == line.b)
        return;
    if (!lines.empty() && lines.back().color == color && lines.back().line.b == line.a) {
        const Point &a = lines.back().line.a;
        const Point &b = lines.back().line.b;
        const Point &c = line.b;
        if (int128::orient(a, b, c) == 0) {
            lines.back().line.b = c;
            return;
        }
    }
    lines.emplace_back(ColoredLine{ line, color });
}

static ColoredLines perimeter_texture_colored_lines_for_polygon(const Polygon                                  &polygon,
                                                                const PerimeterTextureMaskIndex                &top_visible_mask,
                                                                float                                           base_width_mm,
                                                                const PerimeterTextureTopVisibleRecolorThresholds &thresholds,
                                                                const PerimeterTextureRecolorSampler            &recolor_sampler,
                                                                bool                                            point_sample_visibility,
                                                                std::vector<ExPolygons>                       *recolor_footprint_masks = nullptr,
                                                                float                                           recolor_footprint_depth_mm = 0.f)
{
    if (polygon.points.size() < 3 || top_visible_mask.empty())
        return {};

    struct Segment {
        Line   line;
        int    color { 0 };
        bool   eligible { false };
        Polygons visible;
        std::vector<PerimeterTextureVisiblePointSample> visible_samples;
        double visible_area_mm2 { 0.0 };
    };

    struct Run {
        size_t start { 0 };
        size_t end { 0 };
        double length_scaled { 0.0 };
        Polygons visible;
        std::vector<PerimeterTextureVisiblePointSample> visible_samples;
        double visible_area_mm2 { 0.0 };
    };

    std::vector<Segment> segments;

    const double pitch_scaled = std::max(1.0, double(scale_(0.35)));
    for (size_t point_idx = 0; point_idx < polygon.points.size(); ++point_idx) {
        const Point &start = polygon.points[point_idx];
        const Point &end = polygon.points[(point_idx + 1) % polygon.points.size()];
        const double dx = double(end.x()) - double(start.x());
        const double dy = double(end.y()) - double(start.y());
        const double segment_len = std::hypot(dx, dy);
        if (!std::isfinite(segment_len) || segment_len <= EPSILON)
            continue;

        const int sample_count = std::clamp(int(std::ceil(segment_len / pitch_scaled)), 1, 2048);
        for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
            const double t0 = double(sample_idx) / double(sample_count);
            const double t1 = double(sample_idx + 1) / double(sample_count);
            const Point a(coord_t(std::llround(double(start.x()) + dx * t0)),
                          coord_t(std::llround(double(start.y()) + dy * t0)));
            const Point b(coord_t(std::llround(double(start.x()) + dx * t1)),
                          coord_t(std::llround(double(start.y()) + dy * t1)));
            if (a == b)
                continue;

            Polygons visible;
            std::vector<PerimeterTextureVisiblePointSample> visible_samples;
            double footprint_area_mm2 = 0.0;
            double visible_area_mm2 = 0.0;
            if (point_sample_visibility) {
                perimeter_texture_sample_segment_visible_points(base_width_mm,
                                                                a,
                                                                b,
                                                                top_visible_mask,
                                                                visible_samples,
                                                                footprint_area_mm2,
                                                                visible_area_mm2);
                if (recolor_footprint_masks != nullptr && visible_area_mm2 > EPSILON) {
                    double vector_footprint_area_mm2 = 0.0;
                    visible = perimeter_texture_segment_visible_footprint(base_width_mm, a, b, top_visible_mask, vector_footprint_area_mm2);
                }
            } else {
                visible = perimeter_texture_segment_visible_footprint(base_width_mm, a, b, top_visible_mask, footprint_area_mm2);
                visible_area_mm2 = visible.empty() ? 0.0 : perimeter_texture_scaled_area_mm2(area(visible));
            }
            const double visible_fraction = footprint_area_mm2 > EPSILON ? visible_area_mm2 / footprint_area_mm2 : 0.0;
            const bool eligible = visible_fraction >= thresholds.visible_fraction && visible_area_mm2 > EPSILON;
            if (!eligible) {
                visible.clear();
                visible_samples.clear();
            }
            segments.push_back(Segment{ Line(a, b), 0, eligible, std::move(visible), std::move(visible_samples), visible_area_mm2 });
        }
    }

    if (segments.empty())
        return {};

    auto segment_length = [&segments](size_t idx) {
        return segments[idx].line.length();
    };

    const double merge_gap_scaled = scale_(double(thresholds.merge_gap_mm));
    auto rotate_to_large_visibility_gap = [&segments, &segment_length, merge_gap_scaled]() {
        const size_t count = segments.size();
        if (count < 2)
            return;

        const bool has_eligible = std::any_of(segments.begin(), segments.end(), [](const Segment &segment) {
            return segment.eligible;
        });
        if (!has_eligible)
            return;

        for (size_t start = 0; start < count; ++start) {
            if (segments[start].eligible || !segments[(start + count - 1) % count].eligible)
                continue;
            double gap = 0.0;
            size_t idx = start;
            do {
                gap += segment_length(idx);
                idx = (idx + 1) % count;
            } while (idx != start && !segments[idx].eligible);
            if (gap > merge_gap_scaled) {
                if (idx != 0)
                    std::rotate(segments.begin(), segments.begin() + idx, segments.end());
                return;
            }
        }
    };

    auto build_runs = [&segments, &segment_length, merge_gap_scaled]() {
        std::vector<Run> runs;
        size_t idx = 0;
        while (idx < segments.size()) {
            while (idx < segments.size() && !segments[idx].eligible)
                ++idx;
            if (idx >= segments.size())
                break;

            Run run;
            run.start = idx;
            run.end = idx;
            double pending_gap_scaled = 0.0;
            while (idx < segments.size()) {
                if (segments[idx].eligible) {
                    if (pending_gap_scaled > merge_gap_scaled && run.end > run.start)
                        break;
                    for (size_t gap_idx = run.end; gap_idx < idx; ++gap_idx)
                        run.length_scaled += segment_length(gap_idx);
                    run.length_scaled += segment_length(idx);
                    append(run.visible, std::move(segments[idx].visible));
                    for (PerimeterTextureVisiblePointSample &sample : segments[idx].visible_samples)
                        run.visible_samples.emplace_back(std::move(sample));
                    segments[idx].visible_samples.clear();
                    run.visible_area_mm2 += segments[idx].visible_area_mm2;
                    run.end = idx + 1;
                    pending_gap_scaled = 0.0;
                    ++idx;
                } else {
                    pending_gap_scaled += segment_length(idx);
                    ++idx;
                }
            }
            runs.emplace_back(run);
        }
        return runs;
    };

    auto set_run_color = [&segments](const Run &run, unsigned int component_id) {
        for (size_t idx = run.start; idx < run.end; ++idx)
            segments[idx].color = int(component_id);
    };

    rotate_to_large_visibility_gap();
    for (const Run &run : build_runs()) {
        if (unscale<double>(run.length_scaled) < thresholds.min_run_length_mm ||
            run.visible_area_mm2 < thresholds.min_visible_area_mm2)
            continue;
        std::optional<unsigned int> component_id;
        if (point_sample_visibility) {
            component_id = perimeter_texture_choose_recolor_component_from_point_samples(run.visible_samples, recolor_sampler);
        } else {
            ExPolygons visible = union_ex(run.visible);
            if (visible.empty())
                continue;
            component_id = perimeter_texture_choose_recolor_component_with_sampler(visible,
                                                                                  nullptr,
                                                                                  recolor_sampler);
        }
        if (component_id && *component_id > 0) {
            if (recolor_footprint_masks != nullptr && *component_id < recolor_footprint_masks->size()) {
                Polygons path_mask;
                const float depth_mm = recolor_footprint_depth_mm > 0.f ? recolor_footprint_depth_mm : base_width_mm;
                for (size_t idx = run.start; idx < run.end; ++idx) {
                    double footprint_area_mm2 = 0.0;
                    Polygons visible = perimeter_texture_segment_visible_footprint(depth_mm,
                                                                                   segments[idx].line.a,
                                                                                   segments[idx].line.b,
                                                                                   top_visible_mask,
                                                                                   footprint_area_mm2);
                    if (!visible.empty())
                        append(path_mask, std::move(visible));
                }
                if (!path_mask.empty())
                    append((*recolor_footprint_masks)[*component_id], union_ex(path_mask));
            }
            set_run_color(run, *component_id);
        }
    }

    ColoredLines out;
    for (const Segment &segment : segments)
        perimeter_texture_append_colored_line(out, segment.line, segment.color);
    if (out.size() > 1 && out.front().color == out.back().color && out.back().line.b == out.front().line.a) {
        const Point &a = out.back().line.a;
        const Point &b = out.back().line.b;
        const Point &c = out.front().line.b;
        if (int128::orient(a, b, c) == 0) {
            out.front().line.a = a;
            out.pop_back();
        }
    }

    return out;
}

static std::vector<ExPolygons> perimeter_texture_top_visible_region_recolor_masks(
    const LayerRegion                              &layer_region,
    const SurfaceCollection                        &slices,
    const ExPolygons                               &wall_band,
    const ExPolygons                               &top_visible_mask,
    float                                           base_width_mm,
    float                                           recolor_path_depth_mm,
    const PerimeterTextureTopVisibleRecolorThresholds &thresholds,
    const PerimeterTextureRecolorSampler            &recolor_sampler,
    bool                                            point_sample_visibility)
{
    const Layer *layer = layer_region.layer();
    const size_t num_physical = layer != nullptr && layer->object() != nullptr && layer->object()->print() != nullptr ?
        layer->object()->print()->config().filament_colour.values.size() :
        0;
    std::vector<ExPolygons> out(num_physical + 1);
    if (num_physical == 0 || slices.empty() || wall_band.empty() || top_visible_mask.empty())
        return out;
    const PerimeterTextureMaskIndex wall_band_index =
        perimeter_texture_make_mask_index(&wall_band);
    const PerimeterTextureMaskIndex top_visible_mask_index =
        perimeter_texture_make_mask_index(&top_visible_mask);
    if (wall_band_index.empty() || top_visible_mask_index.empty())
        return out;

    for (const Surface &surface : slices.surfaces) {
        const ExPolygon &expolygon = surface.expolygon;
        if (expolygon.empty())
            continue;
        const BoundingBox island_bbox = get_extents(expolygon);
        ExPolygons local_wall_band = perimeter_texture_mask_index_overlapping_expolygons(wall_band_index, island_bbox);
        if (local_wall_band.empty())
            continue;
        ExPolygons island_source{ expolygon };
        ExPolygons island_wall_band = intersection_ex(island_source, local_wall_band);
        if (island_wall_band.empty())
            continue;
        ExPolygons local_top_visible_mask =
            perimeter_texture_mask_index_overlapping_expolygons(top_visible_mask_index, get_extents(island_wall_band));
        if (local_top_visible_mask.empty())
            continue;
        ExPolygons island_top_visible_mask = intersection_ex(island_wall_band, local_top_visible_mask);
        if (island_top_visible_mask.empty())
            continue;
        const PerimeterTextureMaskIndex island_top_visible_mask_index =
            perimeter_texture_make_mask_index(&island_top_visible_mask);
        if (island_top_visible_mask_index.empty())
            continue;

        std::vector<ExPolygons> direct_recolor_masks(num_physical + 1);
        std::vector<ColoredLines> colored_contours;
        ColoredLines contour = perimeter_texture_colored_lines_for_polygon(expolygon.contour,
                                                                           island_top_visible_mask_index,
                                                                           base_width_mm,
                                                                           thresholds,
                                                                           recolor_sampler,
                                                                           point_sample_visibility,
                                                                           &direct_recolor_masks,
                                                                           recolor_path_depth_mm);
        if (!contour.empty())
            colored_contours.emplace_back(std::move(contour));
        for (const Polygon &hole : expolygon.holes) {
            ColoredLines hole_lines = perimeter_texture_colored_lines_for_polygon(hole,
                                                                                  island_top_visible_mask_index,
                                                                                  base_width_mm,
                                                                                  thresholds,
                                                                                  recolor_sampler,
                                                                                  point_sample_visibility,
                                                                                  &direct_recolor_masks,
                                                                                  recolor_path_depth_mm);
            if (!hole_lines.empty())
                colored_contours.emplace_back(std::move(hole_lines));
        }

        const bool has_recolor = std::any_of(colored_contours.begin(), colored_contours.end(), [](const ColoredLines &lines) {
            return std::any_of(lines.begin(), lines.end(), [](const ColoredLine &line) { return line.color > 0; });
        });
        if (!has_recolor)
            continue;

        for (size_t idx = 1; idx < direct_recolor_masks.size(); ++idx) {
            if (direct_recolor_masks[idx].empty())
                continue;
            ExPolygons clipped = intersection_ex(union_ex(direct_recolor_masks[idx]), island_top_visible_mask);
            if (!clipped.empty())
                append(out[idx], std::move(clipped));
        }
    }

    for (size_t idx = 1; idx < out.size(); ++idx) {
        if (out[idx].empty())
            continue;
        out[idx] = union_ex(out[idx]);
        if (out[idx].empty() || perimeter_texture_scaled_area_mm2(area(out[idx])) < thresholds.min_visible_area_mm2)
            out[idx].clear();
    }
    return out;
}

static bool perimeter_texture_recolor_masks_have_color(const std::vector<ExPolygons> &masks)
{
    for (size_t idx = 1; idx < masks.size(); ++idx)
        if (!masks[idx].empty())
            return true;
    return false;
}

static void perimeter_texture_top_visible_recolor_data(const LayerRegion       &layer_region,
                                                       const SurfaceCollection &slices,
                                                       const TextureMappingZone &zone,
                                                       unsigned int             texture_zone_id,
                                                       float                    texture_external_width_mm,
                                                       std::vector<ExPolygons> &top_visible_recolor_masks,
                                                       ExPolygons              &top_visible_recolor_path_mask,
                                                       PerimeterTextureTopVisibleRecolorThresholds &top_visible_recolor_thresholds,
                                                       std::optional<TextureMappingOffsetContext> *reusable_offset_context = nullptr)
{
    top_visible_recolor_masks.clear();
    top_visible_recolor_path_mask.clear();
    if (reusable_offset_context != nullptr)
        reusable_offset_context->reset();
    top_visible_recolor_thresholds =
        perimeter_texture_top_visible_recolor_thresholds(zone.top_visible_perimeter_recolor_aggressiveness);
    const PrintRegionConfig &region_config = layer_region.region().config();
    const int wall_loops = std::max(1, region_config.wall_loops.value);
    const Flow perimeter_flow = layer_region.flow(frPerimeter);
    const float wall_depth_mm =
        0.5f * texture_external_width_mm +
        float(std::max(0, wall_loops - 1)) * float(perimeter_flow.spacing()) +
        0.5f * float(perimeter_flow.width()) + 0.05f;
    const int above_layer_count =
        std::clamp(zone.top_visible_perimeter_recolor_above_layers,
                   TextureMappingZone::MinTopVisiblePerimeterRecolorAboveLayers,
                   TextureMappingZone::MaxTopVisiblePerimeterRecolorAboveLayers);
    ExPolygons top_visible_recolor_wall_band;
    ExPolygons top_visible_recolor_mask =
        perimeter_texture_top_visible_wall_band_mask(layer_region, slices, wall_depth_mm, above_layer_count, &top_visible_recolor_wall_band);
    if (top_visible_recolor_mask.empty())
        return;

    std::optional<PerimeterTextureRecolorSampler> recolor_sampler =
        perimeter_texture_make_recolor_sampler(layer_region, zone, texture_zone_id, texture_external_width_mm);
    if (!recolor_sampler)
        return;

    top_visible_recolor_masks =
        perimeter_texture_top_visible_region_recolor_masks(layer_region,
                                                           slices,
                                                           top_visible_recolor_wall_band,
                                                           top_visible_recolor_mask,
                                                           texture_external_width_mm,
                                                           wall_depth_mm,
                                                           top_visible_recolor_thresholds,
                                                           *recolor_sampler,
                                                           zone.top_visible_perimeter_recolor_point_sampling &&
                                                               zone.uses_perimeter_path_modulation_v2());
    const Layer *layer = layer_region.layer();
    const Print *print = layer != nullptr && layer->object() != nullptr ? layer->object()->print() : nullptr;
    const size_t num_physical = print != nullptr ? print->config().filament_colour.values.size() : 0;
    const unsigned int active_component_id = print != nullptr && num_physical > 0 ?
        print->texture_mapping_manager().resolve_zone_component(texture_zone_id, num_physical, int(layer->id())) :
        0;
    perimeter_texture_move_reusable_offset_context(*recolor_sampler, active_component_id, reusable_offset_context);
    if (!perimeter_texture_recolor_masks_have_color(top_visible_recolor_masks)) {
        top_visible_recolor_masks.clear();
        return;
    }

    for (size_t idx = 1; idx < top_visible_recolor_masks.size(); ++idx)
        append(top_visible_recolor_path_mask, top_visible_recolor_masks[idx]);
    if (!top_visible_recolor_path_mask.empty())
        top_visible_recolor_path_mask = intersection_ex(union_ex(top_visible_recolor_path_mask), top_visible_recolor_mask);
    if (top_visible_recolor_path_mask.empty())
        top_visible_recolor_masks.clear();
}

struct PerimeterTextureRecolorEntityPiece {
    int              extruder_override { -1 };
    ExtrusionEntity *entity { nullptr };
};

struct PerimeterTexturePathRecolorContext {
    PerimeterTextureMaskIndex           path_mask;
    const PerimeterTextureRecolorSampler *sampler { nullptr };
};

struct PerimeterTexturePathSegment {
    Point a;
    Point b;
    int   extruder_override { -1 };
};

static bool perimeter_texture_path_role_can_top_visible_recolor(ExtrusionRole role)
{
    return is_perimeter(role);
}

static Point perimeter_texture_interpolate_point(const Point &a, const Point &b, double t)
{
    return Point(coord_t(std::llround(double(a.x()) + (double(b.x()) - double(a.x())) * t)),
                 coord_t(std::llround(double(a.y()) + (double(b.y()) - double(a.y())) * t)));
}

static double perimeter_texture_path_segment_length_scaled(const PerimeterTexturePathSegment &segment)
{
    return std::hypot(double(segment.b.x()) - double(segment.a.x()),
                      double(segment.b.y()) - double(segment.a.y()));
}

static double perimeter_texture_path_segments_length_scaled(const std::vector<PerimeterTexturePathSegment> &segments,
                                                            size_t                                          begin,
                                                            size_t                                          end)
{
    double out = 0.0;
    for (size_t idx = begin; idx < end && idx < segments.size(); ++idx)
        out += perimeter_texture_path_segment_length_scaled(segments[idx]);
    return out;
}

static Point perimeter_texture_point_at_segment_distance(const PerimeterTexturePathSegment &segment,
                                                         double                             distance_scaled)
{
    const double length_scaled = perimeter_texture_path_segment_length_scaled(segment);
    if (!std::isfinite(length_scaled) || length_scaled <= EPSILON)
        return segment.a;
    return perimeter_texture_interpolate_point(segment.a, segment.b, std::clamp(distance_scaled / length_scaled, 0.0, 1.0));
}

static double perimeter_texture_recolor_normal_length_before(const std::vector<PerimeterTexturePathSegment> &segments,
                                                             size_t                                          start)
{
    double out = 0.0;
    for (size_t idx = start; idx > 0;) {
        --idx;
        if (segments[idx].extruder_override >= 0)
            break;
        out += perimeter_texture_path_segment_length_scaled(segments[idx]);
    }
    return out;
}

static double perimeter_texture_recolor_normal_length_after(const std::vector<PerimeterTexturePathSegment> &segments,
                                                            size_t                                          end)
{
    double out = 0.0;
    for (size_t idx = end; idx < segments.size(); ++idx) {
        if (segments[idx].extruder_override >= 0)
            break;
        out += perimeter_texture_path_segment_length_scaled(segments[idx]);
    }
    return out;
}

static void perimeter_texture_expand_recolor_run_before(std::vector<PerimeterTexturePathSegment> &segments,
                                                        size_t                                  &start,
                                                        size_t                                  &end,
                                                        int                                      extruder_override,
                                                        double                                   length_scaled)
{
    const double eps = std::max<double>(1.0, double(SCALED_EPSILON));
    double remaining = length_scaled;
    while (remaining > eps && start > 0) {
        const size_t idx = start - 1;
        if (segments[idx].extruder_override >= 0)
            break;
        const double segment_length_scaled = perimeter_texture_path_segment_length_scaled(segments[idx]);
        if (segment_length_scaled <= eps) {
            segments[idx].extruder_override = extruder_override;
            --start;
            continue;
        }
        if (segment_length_scaled <= remaining + eps) {
            segments[idx].extruder_override = extruder_override;
            remaining -= segment_length_scaled;
            --start;
            continue;
        }

        const PerimeterTexturePathSegment segment = segments[idx];
        const Point split = perimeter_texture_point_at_segment_distance(segment, segment_length_scaled - remaining);
        if (split == segment.a) {
            segments[idx].extruder_override = extruder_override;
            remaining -= segment_length_scaled;
            --start;
            continue;
        }
        if (split == segment.b)
            break;
        segments[idx] = PerimeterTexturePathSegment{ segment.a, split, -1 };
        segments.insert(segments.begin() + idx + 1, PerimeterTexturePathSegment{ split, segment.b, extruder_override });
        start = idx + 1;
        ++end;
        remaining = 0.0;
    }
}

static void perimeter_texture_expand_recolor_run_after(std::vector<PerimeterTexturePathSegment> &segments,
                                                       size_t                                  &end,
                                                       int                                      extruder_override,
                                                       double                                   length_scaled)
{
    const double eps = std::max<double>(1.0, double(SCALED_EPSILON));
    double remaining = length_scaled;
    while (remaining > eps && end < segments.size()) {
        if (segments[end].extruder_override >= 0)
            break;
        const double segment_length_scaled = perimeter_texture_path_segment_length_scaled(segments[end]);
        if (segment_length_scaled <= eps) {
            segments[end].extruder_override = extruder_override;
            ++end;
            continue;
        }
        if (segment_length_scaled <= remaining + eps) {
            segments[end].extruder_override = extruder_override;
            remaining -= segment_length_scaled;
            ++end;
            continue;
        }

        const PerimeterTexturePathSegment segment = segments[end];
        const Point split = perimeter_texture_point_at_segment_distance(segment, remaining);
        if (split == segment.a)
            break;
        if (split == segment.b) {
            segments[end].extruder_override = extruder_override;
            remaining -= segment_length_scaled;
            ++end;
            continue;
        }
        segments[end] = PerimeterTexturePathSegment{ segment.a, split, extruder_override };
        segments.insert(segments.begin() + end + 1, PerimeterTexturePathSegment{ split, segment.b, -1 });
        ++end;
        remaining = 0.0;
    }
}

static void perimeter_texture_expand_short_recolor_runs(std::vector<PerimeterTexturePathSegment> &segments,
                                                        double                                    min_length_scaled)
{
    const double eps = std::max<double>(1.0, double(SCALED_EPSILON));
    if (segments.empty() || min_length_scaled <= eps)
        return;

    for (size_t start = 0; start < segments.size();) {
        const int extruder_override = segments[start].extruder_override;
        size_t end = start + 1;
        while (end < segments.size() && segments[end].extruder_override == extruder_override)
            ++end;

        if (extruder_override >= 0) {
            const double length_scaled = perimeter_texture_path_segments_length_scaled(segments, start, end);
            if (length_scaled < min_length_scaled - eps && length_scaled >= min_length_scaled * 0.5) {
                const double needed = min_length_scaled - length_scaled;
                const double available_before = perimeter_texture_recolor_normal_length_before(segments, start);
                const double available_after = perimeter_texture_recolor_normal_length_after(segments, end);
                if (available_before + available_after + eps >= needed) {
                    double before_take = std::min(available_before, needed * 0.5);
                    double after_take = std::min(available_after, needed - before_take);
                    double remaining = needed - before_take - after_take;
                    if (remaining > eps) {
                        const double extra_before = std::min(available_before - before_take, remaining);
                        before_take += extra_before;
                        remaining -= extra_before;
                    }
                    if (remaining > eps) {
                        const double extra_after = std::min(available_after - after_take, remaining);
                        after_take += extra_after;
                    }
                    perimeter_texture_expand_recolor_run_before(segments, start, end, extruder_override, before_take);
                    perimeter_texture_expand_recolor_run_after(segments, end, extruder_override, after_take);
                }
            }
        }

        start = end;
    }
}

static void perimeter_texture_clear_short_recolor_runs(std::vector<PerimeterTexturePathSegment> &segments,
                                                       double                                    min_length_scaled)
{
    const double eps = std::max<double>(1.0, double(SCALED_EPSILON));
    if (segments.empty() || min_length_scaled <= eps)
        return;

    for (size_t start = 0; start < segments.size();) {
        const int extruder_override = segments[start].extruder_override;
        size_t end = start + 1;
        while (end < segments.size() && segments[end].extruder_override == extruder_override)
            ++end;
        if (extruder_override >= 0 &&
            perimeter_texture_path_segments_length_scaled(segments, start, end) < min_length_scaled - eps)
            for (size_t idx = start; idx < end; ++idx)
                segments[idx].extruder_override = -1;
        start = end;
    }
}

static unsigned int perimeter_texture_recolor_component_for_segment(const PerimeterTexturePathRecolorContext &context,
                                                                    const Point                                  &a,
                                                                    const Point                                  &b)
{
    if (context.sampler == nullptr || context.path_mask.empty() || a == b)
        return 0;

    const double dx = double(b.x()) - double(a.x());
    const double dy = double(b.y()) - double(a.y());
    const double len = std::hypot(dx, dy);
    if (!std::isfinite(len) || len <= EPSILON)
        return 0;

    const double inward_x = -dy / len;
    const double inward_y = dx / len;
    std::vector<PerimeterTextureVisiblePointSample> visible_samples;
    visible_samples.reserve(3);
    const std::array<Point, 3> sample_points{
        perimeter_texture_interpolate_point(a, b, 0.25),
        perimeter_texture_interpolate_point(a, b, 0.50),
        perimeter_texture_interpolate_point(a, b, 0.75)
    };
    for (const Point &sample : sample_points)
        if (perimeter_texture_mask_index_contains_point(context.path_mask, sample))
            visible_samples.push_back(PerimeterTextureVisiblePointSample{ sample, 1.0, inward_x, inward_y });

    const std::optional<unsigned int> component =
        perimeter_texture_choose_recolor_component_from_point_samples(visible_samples, *context.sampler);
    return component && *component > 0 ? *component : 0;
}

static void perimeter_texture_delete_recolor_entity_pieces(std::vector<PerimeterTextureRecolorEntityPiece> &pieces)
{
    for (PerimeterTextureRecolorEntityPiece &piece : pieces)
        delete piece.entity;
    pieces.clear();
}

static void perimeter_texture_append_recolor_path_piece(std::vector<PerimeterTextureRecolorEntityPiece> &pieces,
                                                        const ExtrusionPath                             &source,
                                                        Points                                         &&points,
                                                        int                                              extruder_override)
{
    Polyline polyline;
    polyline.points = std::move(points);
    remove_same_neighbor(polyline);
    if (polyline.points.size() < 2 || polyline.length() <= SCALED_EPSILON)
        return;
    if (extruder_override >= 0 && std::isfinite(source.width) && source.width > 0.f &&
        unscale<double>(polyline.length()) + unscale<double>(std::max<double>(1.0, double(SCALED_EPSILON))) <
            2.0 * double(source.width))
        extruder_override = -1;
    pieces.push_back(PerimeterTextureRecolorEntityPiece{ extruder_override, new ExtrusionPath(std::move(polyline), source) });
}

static bool perimeter_texture_split_path_by_recolor_masks(const ExtrusionPath                           &path,
                                                          const PerimeterTexturePathRecolorContext      &context,
                                                          std::vector<PerimeterTextureRecolorEntityPiece> &pieces,
                                                          bool                                           emit_unchanged)
{
    if (!perimeter_texture_path_role_can_top_visible_recolor(path.role()) || path.polyline.points.size() < 2) {
        if (emit_unchanged)
            pieces.push_back(PerimeterTextureRecolorEntityPiece{ -1, path.clone() });
        return false;
    }

    std::vector<PerimeterTexturePathSegment> path_segments;
    path_segments.reserve(path.polyline.points.size() - 1);
    bool saw_segment = false;
    bool saw_override = false;

    for (size_t idx = 1; idx < path.polyline.points.size(); ++idx) {
        const Point &a = path.polyline.points[idx - 1];
        const Point &b = path.polyline.points[idx];
        if (a == b)
            continue;
        saw_segment = true;
        const unsigned int component = perimeter_texture_recolor_component_for_segment(context, a, b);
        const int extruder_override = component > 0 ? int(component) - 1 : -1;
        if (extruder_override >= 0)
            saw_override = true;
        path_segments.push_back(PerimeterTexturePathSegment{ a, b, extruder_override });
    }

    if (!saw_segment || !saw_override) {
        if (emit_unchanged)
            pieces.push_back(PerimeterTextureRecolorEntityPiece{ -1, path.clone() });
        return false;
    }

    if (std::isfinite(path.width) && path.width > 0.f) {
        const double min_recolor_length_scaled = double(scale_(2.f * path.width));
        perimeter_texture_expand_short_recolor_runs(path_segments, min_recolor_length_scaled);
        perimeter_texture_clear_short_recolor_runs(path_segments, min_recolor_length_scaled);
    }

    saw_override = std::any_of(path_segments.begin(), path_segments.end(), [](const PerimeterTexturePathSegment &segment) {
        return segment.extruder_override >= 0;
    });
    if (!saw_override) {
        if (emit_unchanged)
            pieces.push_back(PerimeterTextureRecolorEntityPiece{ -1, path.clone() });
        return false;
    }

    std::vector<PerimeterTextureRecolorEntityPiece> fragments;
    Points current_points;
    int current_override = -1;
    for (const PerimeterTexturePathSegment &segment : path_segments) {
        if (segment.a == segment.b)
            continue;
        if (current_points.empty()) {
            current_override = segment.extruder_override;
            current_points.emplace_back(segment.a);
            current_points.emplace_back(segment.b);
        } else if (current_override == segment.extruder_override && current_points.back() == segment.a) {
            current_points.emplace_back(segment.b);
        } else {
            perimeter_texture_append_recolor_path_piece(fragments, path, std::move(current_points), current_override);
            current_override = segment.extruder_override;
            current_points.clear();
            current_points.emplace_back(segment.a);
            current_points.emplace_back(segment.b);
        }
    }

    if (!current_points.empty())
        perimeter_texture_append_recolor_path_piece(fragments, path, std::move(current_points), current_override);

    for (PerimeterTextureRecolorEntityPiece &piece : fragments)
        pieces.push_back(piece);
    fragments.clear();
    return true;
}

static bool perimeter_texture_split_entity_by_recolor_masks(const ExtrusionEntity                         &entity,
                                                            const PerimeterTexturePathRecolorContext      &context,
                                                            std::vector<PerimeterTextureRecolorEntityPiece> &pieces,
                                                            bool                                           emit_unchanged);

static bool perimeter_texture_split_paths_by_recolor_masks(const ExtrusionPaths                          &paths,
                                                           const ExtrusionEntity                         &fallback_entity,
                                                           const PerimeterTexturePathRecolorContext      &context,
                                                           std::vector<PerimeterTextureRecolorEntityPiece> &pieces,
                                                           bool                                           emit_unchanged)
{
    std::vector<PerimeterTextureRecolorEntityPiece> path_pieces;
    bool changed = false;
    for (const ExtrusionPath &path : paths)
        changed |= perimeter_texture_split_path_by_recolor_masks(path, context, path_pieces, true);

    if (!changed) {
        perimeter_texture_delete_recolor_entity_pieces(path_pieces);
        if (emit_unchanged)
            pieces.push_back(PerimeterTextureRecolorEntityPiece{ -1, fallback_entity.clone() });
        return false;
    }

    for (PerimeterTextureRecolorEntityPiece &piece : path_pieces)
        pieces.push_back(piece);
    path_pieces.clear();
    return true;
}

static bool perimeter_texture_split_collection_children_by_recolor_masks(const ExtrusionEntityCollection              &collection,
                                                                         const PerimeterTexturePathRecolorContext      &context,
                                                                         std::vector<PerimeterTextureRecolorEntityPiece> &pieces,
                                                                         bool                                           emit_unchanged)
{
    std::vector<PerimeterTextureRecolorEntityPiece> child_pieces;
    bool changed = collection.texture_mapping_extruder_override >= 0;
    for (const ExtrusionEntity *child : collection.entities) {
        if (child == nullptr)
            continue;
        if (collection.texture_mapping_extruder_override >= 0) {
            child_pieces.push_back(PerimeterTextureRecolorEntityPiece{ collection.texture_mapping_extruder_override, child->clone() });
        } else {
            changed |= perimeter_texture_split_entity_by_recolor_masks(*child, context, child_pieces, true);
        }
    }

    if (!changed) {
        if (emit_unchanged) {
            for (PerimeterTextureRecolorEntityPiece &piece : child_pieces)
                pieces.push_back(piece);
            child_pieces.clear();
        } else {
            perimeter_texture_delete_recolor_entity_pieces(child_pieces);
        }
        return false;
    }

    for (PerimeterTextureRecolorEntityPiece &piece : child_pieces)
        pieces.push_back(piece);
    child_pieces.clear();
    return true;
}

static bool perimeter_texture_split_entity_by_recolor_masks(const ExtrusionEntity                         &entity,
                                                            const PerimeterTexturePathRecolorContext      &context,
                                                            std::vector<PerimeterTextureRecolorEntityPiece> &pieces,
                                                            bool                                           emit_unchanged)
{
    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity))
        return perimeter_texture_split_path_by_recolor_masks(*path, context, pieces, emit_unchanged);
    if (const ExtrusionMultiPath *multipath = dynamic_cast<const ExtrusionMultiPath *>(&entity))
        return perimeter_texture_split_paths_by_recolor_masks(multipath->paths, *multipath, context, pieces, emit_unchanged);
    if (const ExtrusionLoop *loop = dynamic_cast<const ExtrusionLoop *>(&entity))
        return perimeter_texture_split_paths_by_recolor_masks(loop->paths, *loop, context, pieces, emit_unchanged);
    if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity))
        return perimeter_texture_split_collection_children_by_recolor_masks(*collection, context, pieces, emit_unchanged);

    if (emit_unchanged)
        pieces.push_back(PerimeterTextureRecolorEntityPiece{ -1, entity.clone() });
    return false;
}

static void perimeter_texture_append_recolor_top_collection(ExtrusionEntitiesPtr &entities,
                                                            int                   extruder_override,
                                                            ExtrusionEntity      *entity,
                                                            bool                  force_new_collection)
{
    if (entity == nullptr)
        return;

    ExtrusionEntityCollection *collection = nullptr;
    if (!force_new_collection && !entities.empty())
        collection = dynamic_cast<ExtrusionEntityCollection *>(entities.back());
    if (collection == nullptr || collection->texture_mapping_extruder_override != extruder_override) {
        collection = new ExtrusionEntityCollection();
        collection->texture_mapping_extruder_override = extruder_override;
        entities.emplace_back(collection);
    }
    collection->entities.emplace_back(entity);
}

static void perimeter_texture_append_recolor_top_collections(ExtrusionEntitiesPtr &entities,
                                                             std::vector<PerimeterTextureRecolorEntityPiece> &pieces)
{
    bool force_new_collection = true;
    for (PerimeterTextureRecolorEntityPiece &piece : pieces) {
        perimeter_texture_append_recolor_top_collection(entities, piece.extruder_override, piece.entity, force_new_collection);
        piece.entity = nullptr;
        force_new_collection = false;
    }
    pieces.clear();
}

static void perimeter_texture_apply_top_visible_recolor_to_perimeters(const LayerRegion              &layer_region,
                                                                      ExtrusionEntityCollection     &perimeters,
                                                                      const ExPolygons              &path_mask,
                                                                      const TextureMappingZone      &zone,
                                                                      unsigned int                   texture_zone_id,
                                                                      float                          texture_external_width_mm)
{
    if (perimeters.entities.empty() || path_mask.empty())
        return;

    std::optional<PerimeterTextureRecolorSampler> sampler =
        perimeter_texture_make_recolor_sampler(layer_region, zone, texture_zone_id, texture_external_width_mm);
    if (!sampler)
        return;

    PerimeterTexturePathRecolorContext context;
    context.path_mask = perimeter_texture_make_mask_index(&path_mask);
    context.sampler = &*sampler;
    if (context.path_mask.empty())
        return;

    ExtrusionEntitiesPtr recolored_entities;
    bool changed = false;
    for (const ExtrusionEntity *entity : perimeters.entities) {
        if (entity == nullptr)
            continue;

        std::vector<PerimeterTextureRecolorEntityPiece> pieces;
        if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(entity)) {
            if (collection->texture_mapping_extruder_override >= 0) {
                recolored_entities.emplace_back(collection->clone());
                continue;
            }
            perimeter_texture_split_collection_children_by_recolor_masks(*collection, context, pieces, true);
        } else {
            perimeter_texture_split_entity_by_recolor_masks(*entity, context, pieces, true);
        }

        const bool entity_changed = std::any_of(pieces.begin(), pieces.end(), [](const PerimeterTextureRecolorEntityPiece &piece) {
            return piece.extruder_override >= 0;
        });
        if (entity_changed) {
            changed = true;
            perimeter_texture_append_recolor_top_collections(recolored_entities, pieces);
        } else {
            perimeter_texture_delete_recolor_entity_pieces(pieces);
            recolored_entities.emplace_back(entity->clone());
        }
    }

    if (!changed) {
        for (ExtrusionEntity *entity : recolored_entities)
            delete entity;
        return;
    }

    perimeters.clear();
    perimeters.entities = std::move(recolored_entities);
}

static std::vector<ExPolygons> perimeter_texture_build_erode_ladder(const ExPolygon &source,
                                                                    float            max_inset_mm,
                                                                    float           &step_mm)
{
    step_mm = std::clamp(max_inset_mm / 10.f, 0.025f, 0.08f);
    const int level_count = std::clamp(int(std::ceil(max_inset_mm / std::max(step_mm, 1e-4f))) + 1, 1, 96);
    std::vector<ExPolygons> ladder(static_cast<size_t>(level_count));
    ladder.front().push_back(source);
    for (int level = 1; level < level_count; ++level) {
        const float distance_scaled = float(scale_(double(step_mm) * double(level)));
        ladder[size_t(level)] = offset_ex(source, -distance_scaled);
        if (ladder[size_t(level)].empty()) {
            for (int fill = level + 1; fill < level_count; ++fill)
                ladder[size_t(fill)].clear();
            break;
        }
    }
    return ladder;
}

static float perimeter_texture_safe_inset_for_sample(const ExPolygon              &source,
                                                     const std::vector<ExPolygons> &erode_ladder,
                                                     float                         erode_step_mm,
                                                     const PerimeterPathBoundarySample &sample,
                                                     float                         desired_mm,
                                                     float                         max_inset_mm)
{
    const float clamped_desired = std::clamp(desired_mm, 0.f, max_inset_mm);
    if (clamped_desired <= EPSILON)
        return 0.f;

    auto point_at_inset = [&sample](float inset_mm) {
        const double inset_scaled = scale_(double(inset_mm));
        return Point(coord_t(std::llround(double(sample.point.x()) + sample.inward_x * inset_scaled)),
                     coord_t(std::llround(double(sample.point.y()) + sample.inward_y * inset_scaled)));
    };

    auto inset_allowed = [&](float inset_mm) {
        const Point candidate = point_at_inset(inset_mm);
        if (!source.contains(candidate, true))
            return false;
        const size_t level = std::min(erode_ladder.empty() ? size_t(0) :
                                          size_t(std::floor(std::max(0.f, inset_mm - 0.01f) / std::max(erode_step_mm, 1e-4f))),
                                      erode_ladder.empty() ? size_t(0) : erode_ladder.size() - 1);
        if (level > 0) {
            if (level >= erode_ladder.size() || erode_ladder[level].empty())
                return false;
            if (!perimeter_texture_expolygons_contain_point(erode_ladder[level], candidate))
                return false;
        }
        return true;
    };

    if (inset_allowed(clamped_desired))
        return clamped_desired;

    float low = 0.f;
    float high = clamped_desired;
    for (int i = 0; i < 10; ++i) {
        const float mid = 0.5f * (low + high);
        if (inset_allowed(mid))
            low = mid;
        else
            high = mid;
    }
    return low;
}

static std::vector<PerimeterPathBoundarySample> perimeter_texture_sample_polygon_boundary(
    const Polygon                    &polygon,
    const TextureMappingOffsetContext &context,
    const ExPolygon                  &source,
    const std::vector<ExPolygons>    &erode_ladder,
    float                             erode_step_mm,
    const PerimeterTextureMaskIndex  *top_visible_recolor_mask = nullptr,
    const PerimeterTextureTopVisibleRecolorThresholds *top_visible_recolor_thresholds = nullptr)
{
    std::vector<PerimeterPathBoundarySample> samples;
    const Points &points = polygon.points;
    if (points.size() < 3)
        return samples;

    const double pitch_scaled = scale_(context.high_resolution_texture_sampling ? 0.08 : 0.16);

    for (size_t idx = 0; idx < points.size(); ++idx) {
        const Point &a = points[idx];
        const Point &b = points[(idx + 1) % points.size()];
        const double dx = double(b.x()) - double(a.x());
        const double dy = double(b.y()) - double(a.y());
        const double len = std::hypot(dx, dy);
        if (!std::isfinite(len) || len <= EPSILON)
            continue;

        const double inward_x = -dy / len;
        const double inward_y = dx / len;
        const int sample_count = std::clamp(int(std::ceil(len / std::max(pitch_scaled, 1.0))), 1, 4096);
        for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
            const double t = double(sample_idx) / double(sample_count);
            PerimeterPathBoundarySample sample;
            sample.point = Point(coord_t(std::llround(double(a.x()) + dx * t)),
                                 coord_t(std::llround(double(a.y()) + dy * t)));
            sample.inward_x = inward_x;
            sample.inward_y = inward_y;
            const bool protected_sample =
                perimeter_texture_sample_is_top_visible_recolor_protected(top_visible_recolor_mask,
                                                                          sample,
                                                                          dx / len,
                                                                          dy / len,
                                                                          context.max_width_delta_mm,
                                                                          top_visible_recolor_thresholds);
            const float desired_mm = protected_sample ? 0.f : texture_mapping_offset_surface_inset_mm(context, sample.point, inward_x, inward_y);
            sample.inset_mm = perimeter_texture_safe_inset_for_sample(source,
                                                                      erode_ladder,
                                                                      erode_step_mm,
                                                                      sample,
                                                                      desired_mm,
                                                                      context.max_width_delta_mm);
            samples.emplace_back(sample);
        }
    }

    if (samples.size() < 3)
        return {};

    auto sample_distance_mm = [](const PerimeterPathBoundarySample &lhs, const PerimeterPathBoundarySample &rhs) {
        const double dx = double(lhs.point.x()) - double(rhs.point.x());
        const double dy = double(lhs.point.y()) - double(rhs.point.y());
        return unscale<float>(std::hypot(dx, dy));
    };

    auto smooth_insets = [&samples, &sample_distance_mm](float radius_mm) {
        if (samples.size() < 3)
            return;
        radius_mm = std::clamp(radius_mm, 0.05f, 1.5f);
        std::vector<float> smoothed(samples.size(), 0.f);
        for (size_t idx = 0; idx < samples.size(); ++idx) {
            float weighted_sum = samples[idx].inset_mm;
            float weight_sum = 1.f;
            float distance_mm = 0.f;
            for (size_t step = 1; step < samples.size(); ++step) {
                const size_t prev = (idx + samples.size() - step) % samples.size();
                const size_t next_prev = (prev + 1) % samples.size();
                distance_mm += sample_distance_mm(samples[prev], samples[next_prev]);
                if (distance_mm > radius_mm)
                    break;
                const float weight = 1.f - distance_mm / radius_mm;
                weighted_sum += samples[prev].inset_mm * weight;
                weight_sum += weight;
            }
            distance_mm = 0.f;
            for (size_t step = 1; step < samples.size(); ++step) {
                const size_t next = (idx + step) % samples.size();
                const size_t prev_next = next == 0 ? samples.size() - 1 : next - 1;
                distance_mm += sample_distance_mm(samples[prev_next], samples[next]);
                if (distance_mm > radius_mm)
                    break;
                const float weight = 1.f - distance_mm / radius_mm;
                weighted_sum += samples[next].inset_mm * weight;
                weight_sum += weight;
            }
            smoothed[idx] = weight_sum > EPSILON ? std::min(samples[idx].inset_mm, weighted_sum / weight_sum) : samples[idx].inset_mm;
        }
        for (size_t idx = 0; idx < samples.size(); ++idx)
            samples[idx].inset_mm = smoothed[idx];
    };

    smooth_insets(std::max(0.30f, 1.15f * context.base_outer_width_mm));

    for (int pass = 0; pass < 4; ++pass) {
        for (size_t idx = 0; idx < samples.size(); ++idx) {
            const size_t prev = idx == 0 ? samples.size() - 1 : idx - 1;
            const float limit = sample_distance_mm(samples[idx], samples[prev]) * 0.35f + 0.015f;
            if (samples[idx].inset_mm > samples[prev].inset_mm + limit)
                samples[idx].inset_mm = samples[prev].inset_mm + limit;
        }
        for (size_t idx = samples.size(); idx-- > 0;) {
            const size_t next = (idx + 1) % samples.size();
            const float limit = sample_distance_mm(samples[idx], samples[next]) * 0.35f + 0.015f;
            if (samples[idx].inset_mm > samples[next].inset_mm + limit)
                samples[idx].inset_mm = samples[next].inset_mm + limit;
        }
    }

    smooth_insets(std::max(0.20f, 0.45f * context.base_outer_width_mm));

    return samples;
}

static Polygon perimeter_texture_moved_polygon_from_samples(const std::vector<PerimeterPathBoundarySample> &samples)
{
    Polygon polygon;
    polygon.points.reserve(samples.size());
    Point last_point;
    bool has_last = false;
    for (const PerimeterPathBoundarySample &sample : samples) {
        const double inset_scaled = scale_(double(sample.inset_mm));
        Point moved(coord_t(std::llround(double(sample.point.x()) + sample.inward_x * inset_scaled)),
                    coord_t(std::llround(double(sample.point.y()) + sample.inward_y * inset_scaled)));
        if (!has_last || moved != last_point) {
            polygon.points.emplace_back(moved);
            last_point = moved;
            has_last = true;
        }
    }
    if (polygon.points.size() > 1 && polygon.points.front() == polygon.points.back())
        polygon.points.pop_back();
    remove_same_neighbor(polygon);
    return polygon;
}

static ExPolygons perimeter_texture_modulated_expolygon(const ExPolygon &source,
                                                        const TextureMappingOffsetContext &context,
                                                        const ExPolygons *top_visible_recolor_mask,
                                                        const PerimeterTextureTopVisibleRecolorThresholds *top_visible_recolor_thresholds,
                                                        bool &ok)
{
    ok = false;
    if (source.empty() || source.contour.points.size() < 3)
        return {};

    const double source_area = std::abs(source.area());
    if (!std::isfinite(source_area) || source_area <= 0.0)
        return {};

    const PerimeterTextureMaskIndex protection_mask =
        perimeter_texture_make_mask_index(top_visible_recolor_mask);
    const PerimeterTextureMaskIndex *protection_mask_ptr =
        protection_mask.empty() ? nullptr : &protection_mask;
    float erode_step_mm = 0.f;
    const std::vector<ExPolygons> erode_ladder =
        perimeter_texture_build_erode_ladder(source, context.max_width_delta_mm, erode_step_mm);

    const std::vector<PerimeterPathBoundarySample> contour_samples =
        perimeter_texture_sample_polygon_boundary(source.contour,
                                                  context,
                                                  source,
                                                  erode_ladder,
                                                  erode_step_mm,
                                                  protection_mask_ptr,
                                                  top_visible_recolor_thresholds);
    if (contour_samples.size() < 3)
        return {};

    bool has_meaningful_inset = false;
    for (const PerimeterPathBoundarySample &sample : contour_samples) {
        if (sample.inset_mm > 0.002f) {
            has_meaningful_inset = true;
            break;
        }
    }
    std::vector<std::vector<PerimeterPathBoundarySample>> hole_samples;
    hole_samples.reserve(source.holes.size());
    for (const Polygon &hole : source.holes) {
        hole_samples.emplace_back(perimeter_texture_sample_polygon_boundary(hole,
                                                                            context,
                                                                            source,
                                                                            erode_ladder,
                                                                            erode_step_mm,
                                                                            protection_mask_ptr,
                                                                            top_visible_recolor_thresholds));
        if (hole_samples.back().size() < 3)
            return {};
        for (const PerimeterPathBoundarySample &sample : hole_samples.back()) {
            if (sample.inset_mm > 0.002f) {
                has_meaningful_inset = true;
                break;
            }
        }
    }
    if (!has_meaningful_inset) {
        ok = true;
        return ExPolygons{ source };
    }

    ExPolygon moved;
    moved.contour = perimeter_texture_moved_polygon_from_samples(contour_samples);
    if (moved.contour.points.size() < 3)
        return {};
    moved.contour.make_counter_clockwise();
    moved.holes.reserve(hole_samples.size());
    for (const std::vector<PerimeterPathBoundarySample> &samples : hole_samples) {
        Polygon hole = perimeter_texture_moved_polygon_from_samples(samples);
        if (hole.points.size() < 3)
            return {};
        hole.make_clockwise();
        moved.holes.emplace_back(std::move(hole));
    }

    ExPolygons simplified = moved.simplify(scale_(0.006));
    if (simplified.empty())
        return {};

    ExPolygons source_only;
    source_only.emplace_back(source);
    ExPolygons clipped = intersection_ex(simplified, source_only);
    if (clipped.empty())
        return {};
    remove_same_neighbor(clipped);

    const double clipped_area = area(clipped);
    if (!std::isfinite(clipped_area) || clipped_area <= source_area * 0.10)
        return {};
    if (clipped_area > source_area * 1.002)
        return {};

    ok = true;
    return clipped;
}

static SurfaceCollection perimeter_path_modulated_surfaces(const LayerRegion      &layer_region,
                                                           const SurfaceCollection &slices,
                                                           const TextureMappingZone &zone,
                                                           unsigned int             texture_zone_id,
                                                           std::optional<float>     base_outer_width_mm = std::nullopt,
                                                           const ExPolygons        *top_visible_recolor_mask = nullptr,
                                                           const PerimeterTextureTopVisibleRecolorThresholds *top_visible_recolor_thresholds = nullptr,
                                                           const TextureMappingOffsetContext *prebuilt_context = nullptr)
{
    const Layer *layer = layer_region.layer();
    if (layer == nullptr || layer->object() == nullptr)
        return slices;

    std::optional<TextureMappingOffsetContext> built_context;
    const TextureMappingOffsetContext *context = prebuilt_context;
    if (context == nullptr) {
        built_context =
            build_texture_mapping_offset_context_for_layer(*layer->object(),
                                                           *layer,
                                                           zone,
                                                           texture_zone_id,
                                                           0,
                                                           base_outer_width_mm);
        if (!built_context)
            return slices;
        context = &*built_context;
    }

    SurfaceCollection out;
    out.surfaces.reserve(slices.surfaces.size());
    for (const Surface &surface : slices.surfaces) {
        bool ok = false;
        ExPolygons modulated =
            perimeter_texture_modulated_expolygon(surface.expolygon, *context, top_visible_recolor_mask, top_visible_recolor_thresholds, ok);
        if (ok && !modulated.empty())
            out.append(std::move(modulated), surface);
        else
            out.surfaces.emplace_back(surface);
    }
    return out;
}

static bool region_uses_overhang_texture_mapping(const Print &print, const PrintRegionConfig &region_config)
{
    const int filament_id = region_config.wall_filament.value;
    if (filament_id <= 0)
        return false;

    const TextureMappingZone *zone = print.texture_mapping_manager().zone_from_id(unsigned(filament_id));
    return zone != nullptr && zone->enabled && !zone->deleted && (zone->is_2d_gradient() || zone->is_image_texture());
}

static const TextureMappingZone *perimeter_path_modulation_zone_for_region(const Print              &print,
                                                                           const PrintRegionConfig &region_config,
                                                                           unsigned int            &texture_zone_id)
{
    const int filament_id = region_config.wall_filament.value;
    if (filament_id <= 0)
        return nullptr;

    texture_zone_id = unsigned(filament_id);
    const TextureMappingZone *zone = print.texture_mapping_manager().zone_from_id(texture_zone_id);
    if (zone == nullptr ||
        !zone->enabled ||
        zone->deleted ||
        !zone->uses_perimeter_path_modulation() ||
        (!zone->is_2d_gradient() && !zone->is_image_texture()))
        return nullptr;

    return zone;
}

static const TextureMappingZone *perimeter_path_modulation_v2_zone_for_region(const Print              &print,
                                                                              const PrintRegionConfig &region_config,
                                                                              unsigned int            &texture_zone_id)
{
    const TextureMappingZone *zone = perimeter_path_modulation_zone_for_region(print, region_config, texture_zone_id);
    return zone != nullptr && zone->uses_perimeter_path_modulation_v2() ? zone : nullptr;
}

void Layer::apply_perimeter_path_modulation_v2()
{
    PrintObject *print_object = this->object();
    if (print_object == nullptr || print_object->print() == nullptr)
        return;

    bool needs_geometry_update = false;
    for (LayerRegion *layerm : m_regions) {
        if (layerm == nullptr)
            continue;
        unsigned int texture_zone_id = 0;
        if (layerm->perimeter_path_modulation_v2_applied ||
            perimeter_path_modulation_v2_zone_for_region(*print_object->print(),
                                                         layerm->region().config(),
                                                         texture_zone_id) != nullptr) {
            needs_geometry_update = true;
            break;
        }
    }
    if (!needs_geometry_update)
        return;

    for (LayerRegion *layerm : m_regions) {
        if (layerm == nullptr)
            continue;
        if (layerm->unmodulated_raw_slices.empty() && !layerm->raw_slices.empty())
            layerm->unmodulated_raw_slices = layerm->raw_slices;
        if (layerm->unmodulated_raw_slices.empty() && !layerm->slices.empty())
            layerm->unmodulated_raw_slices = to_expolygons(layerm->slices.surfaces);
        if (!layerm->unmodulated_raw_slices.empty() || !layerm->slices.empty()) {
            layerm->slices.set(layerm->unmodulated_raw_slices, stInternal);
            layerm->raw_slices = layerm->unmodulated_raw_slices;
        } else {
            layerm->raw_slices.clear();
        }
        layerm->perimeter_path_modulation_v2_applied = false;
        layerm->perimeter_path_modulation_v2_fallback_slices.clear();
        layerm->perimeter_path_modulation_v2_has_fallback_slices = false;
        layerm->perimeter_path_modulation_v2_fallback_is_modulated = false;
    }

    for (LayerRegion *layerm : m_regions) {
        if (layerm == nullptr || layerm->slices.empty())
            continue;
        unsigned int texture_zone_id = 0;
        const TextureMappingZone *zone =
            perimeter_path_modulation_v2_zone_for_region(*print_object->print(),
                                                         layerm->region().config(),
                                                         texture_zone_id);
        if (zone == nullptr)
            continue;
        SurfaceCollection modulated_slices =
            perimeter_path_modulated_surfaces(*layerm,
                                              layerm->slices,
                                              *zone,
                                              texture_zone_id,
                                              std::nullopt,
                                              nullptr,
                                              nullptr,
                                              nullptr);
        layerm->slices = std::move(modulated_slices);
        layerm->raw_slices = to_expolygons(layerm->slices.surfaces);
        layerm->perimeter_path_modulation_v2_applied = true;
    }

    this->make_slices();
    this->lslices_bboxes.clear();
    this->lslices_bboxes.reserve(this->lslices.size());
    for (const ExPolygon &expoly : this->lslices)
        this->lslices_bboxes.emplace_back(get_extents(expoly));
}

Flow LayerRegion::flow(FlowRole role) const
{
    return this->flow(role, m_layer->height);
}

Flow LayerRegion::flow(FlowRole role, double layer_height) const
{
    return m_region->flow(*m_layer->object(), role, layer_height, m_layer->id() == 0);
}

Flow LayerRegion::bridging_flow(FlowRole role, bool thick_bridge) const
{
    const PrintRegion       &region         = this->region();
    const PrintRegionConfig &region_config  = region.config();
    const PrintObject       &print_object   = *this->layer()->object();
    Flow bridge_flow;
    auto nozzle_diameter = float(print_object.print()->config().nozzle_diameter.get_at(region.extruder(role) - 1));
    if (thick_bridge) {
        // The old Slic3r way (different from all other slicers): Use rounded extrusions.
        // Get the configured nozzle_diameter for the extruder associated to the flow role requested.
        // Here this->extruder(role) - 1 may underflow to MAX_INT, but then the get_at() will follback to zero'th element, so everything is all right.
        // Applies default bridge spacing.
        bridge_flow = Flow::bridging_flow(float(sqrt(region_config.bridge_flow)) * nozzle_diameter, nozzle_diameter);
    } else {
        // The same way as other slicers: Use normal extrusions. Apply bridge_flow while maintaining the original spacing.
        bridge_flow = this->flow(role).with_flow_ratio(region_config.bridge_flow);
    }
    return bridge_flow;

}

// Fill in layerm->fill_surfaces by trimming the layerm->slices by the cummulative layerm->fill_surfaces.
void LayerRegion::slices_to_fill_surfaces_clipped()
{
    // Note: this method should be idempotent, but fill_surfaces gets modified 
    // in place. However we're now only using its boundaries (which are invariant)
    // so we're safe. This guarantees idempotence of prepare_infill() also in case
    // that combine_infill() turns some fill_surface into VOID surfaces.
    // Collect polygons per surface type.
    std::array<SurfacesPtr, size_t(stCount)> by_surface;
    for (Surface &surface : this->slices.surfaces)
        by_surface[size_t(surface.surface_type)].emplace_back(&surface);
    // Trim surfaces by the fill_boundaries.
    this->fill_surfaces.surfaces.clear();
    for (size_t surface_type = 0; surface_type < size_t(stCount); ++ surface_type) {
        const SurfacesPtr &this_surfaces = by_surface[surface_type];
        if (! this_surfaces.empty())
            this->fill_surfaces.append(intersection_ex(this_surfaces, this->fill_expolygons), SurfaceType(surface_type));
    }
}

void LayerRegion::make_perimeters(const SurfaceCollection &slices, const LayerRegionPtrs &compatible_regions, SurfaceCollection* fill_surfaces, ExPolygons* fill_no_overlap)
{
    this->perimeters.clear();
    this->thin_fills.clear();

    const PrintConfig       &print_config  = this->layer()->object()->print()->config();
    const PrintRegionConfig &region_config = this->region().config();
    const PrintObjectConfig& object_config = this->layer()->object()->config();
    // This needs to be in sync with PrintObject::_slice() slicing_mode_normal_below_layer!
    bool spiral_mode = print_config.spiral_mode &&
        //FIXME account for raft layers.
        (this->layer()->id() >= size_t(region_config.bottom_shell_layers.value) &&
         this->layer()->print_z >= region_config.bottom_shell_thickness - EPSILON);

    const float texture_external_width_mm =
        std::max(0.05f, float(print_config.texture_mapping_outer_wall_gradient_max_line_width.value));

    SurfaceCollection modulated_slices;
    const SurfaceCollection *perimeter_slices = &slices;
    ExPolygons top_visible_recolor_path_mask;
    std::vector<ExPolygons> top_visible_recolor_masks;
    PerimeterTextureTopVisibleRecolorThresholds top_visible_recolor_thresholds;
    std::optional<TextureMappingOffsetContext> reusable_modulation_context;
    unsigned int perimeter_texture_zone_id = 0;
    bool use_perimeter_path_modulation = false;
    bool use_legacy_perimeter_path_modulation = false;
    bool use_perimeter_path_modulation_v2 = false;
    const TextureMappingZone *perimeter_path_zone =
        perimeter_path_modulation_zone_for_region(*this->layer()->object()->print(), region_config, perimeter_texture_zone_id);
    if (perimeter_path_zone != nullptr) {
        if (perimeter_path_zone->recolor_top_visible_perimeter_sections) {
            perimeter_texture_top_visible_recolor_data(*this,
                                                       slices,
                                                       *perimeter_path_zone,
                                                       perimeter_texture_zone_id,
                                                       texture_external_width_mm,
                                                       top_visible_recolor_masks,
                                                       top_visible_recolor_path_mask,
                                                       top_visible_recolor_thresholds,
                                                       perimeter_path_zone->uses_legacy_perimeter_path_modulation() ? &reusable_modulation_context : nullptr);
        }
        if (perimeter_path_zone->uses_legacy_perimeter_path_modulation()) {
            modulated_slices = perimeter_path_modulated_surfaces(*this,
                                                                 slices,
                                                                 *perimeter_path_zone,
                                                                 perimeter_texture_zone_id,
                                                                 std::nullopt,
                                                                 nullptr,
                                                                 nullptr,
                                                                 reusable_modulation_context ? &*reusable_modulation_context : nullptr);
            perimeter_slices = &modulated_slices;
            use_legacy_perimeter_path_modulation = true;
        } else if (perimeter_path_zone->uses_perimeter_path_modulation_v2()) {
            use_perimeter_path_modulation_v2 = true;
        }
        use_perimeter_path_modulation = true;
    }

    SurfaceCollection v2_original_slices;
    const SurfaceCollection *fallback_original_slices = &slices;
    if (use_perimeter_path_modulation_v2 && !this->unmodulated_raw_slices.empty()) {
        v2_original_slices.set(this->unmodulated_raw_slices, stInternal);
        fallback_original_slices = &v2_original_slices;
    }

    auto set_perimeter_path_modulation_v2_fallback_slices =
        [this, use_perimeter_path_modulation_v2](const SurfaceCollection &fallback_slices, bool is_modulated) {
            if (!use_perimeter_path_modulation_v2)
                return;
            this->perimeter_path_modulation_v2_fallback_slices = fallback_slices;
            this->perimeter_path_modulation_v2_has_fallback_slices = true;
            this->perimeter_path_modulation_v2_fallback_is_modulated = is_modulated;
        };

    const bool force_classic_wall_generator = region_uses_overhang_texture_mapping(*this->layer()->object()->print(), region_config);
    auto texture_external_flow = [&](float external_width_mm) {
        Flow out = this->flow(frExternalPerimeter);
        const float min_width_for_positive_spacing_mm =
            std::max(0.01f, float(this->layer()->height) * float(1. - 0.25 * PI) + 1e-4f);
        out = out.with_width(std::max(external_width_mm, min_width_for_positive_spacing_mm));
        out.set_spacing(std::min(out.spacing(), this->flow(frExternalPerimeter).spacing()));
        return out;
    };

    auto process_slices = [&](const SurfaceCollection *input_slices, std::optional<float> texture_external_width_mm) {
        PerimeterGenerator g(
            // input:
            input_slices,
            &compatible_regions,
            this->layer()->height,
            this->layer()->slice_z,
            this->flow(frPerimeter),
            &region_config,
            &this->layer()->object()->config(),
            &print_config,
            spiral_mode,

            // output:
            &this->perimeters,
            &this->thin_fills,
            fill_surfaces,
            //BBS
            fill_no_overlap
        );

        if (this->layer()->lower_layer != nullptr)
            // Cummulative sum of polygons over all the regions.
            g.lower_slices = &this->layer()->lower_layer->lslices;
        if (this->layer()->upper_layer != NULL)
            g.upper_slices = &this->layer()->upper_layer->lslices;

        int region_id = this->region().print_object_region_id();
        if (this->layer()->upper_layer != NULL)
            g.upper_slices_same_region = &this->layer()->upper_layer->get_region(region_id)->slices;

        g.layer_id              = (int)this->layer()->id();
        g.ext_perimeter_flow    = this->flow(frExternalPerimeter);
        if (texture_external_width_mm)
            g.ext_perimeter_flow = texture_external_flow(*texture_external_width_mm);
        g.overhang_flow         = this->bridging_flow(frPerimeter, object_config.thick_bridges);
        g.solid_infill_flow     = this->flow(frSolidInfill);

        if (this->layer()->object()->config().wall_generator.value == PerimeterGeneratorType::Arachne && !spiral_mode &&
            !force_classic_wall_generator)
            g.process_arachne();
        else
            g.process_classic();
    };

    auto process_slices_with_top_visible_recolor =
        [&](const SurfaceCollection *input_slices,
            std::optional<float> active_texture_external_width_mm) {
            process_slices(input_slices, active_texture_external_width_mm);
            if (perimeter_path_zone != nullptr)
                perimeter_texture_apply_top_visible_recolor_to_perimeters(*this,
                                                                          this->perimeters,
                                                                          top_visible_recolor_path_mask,
                                                                          *perimeter_path_zone,
                                                                          perimeter_texture_zone_id,
                                                                          active_texture_external_width_mm.value_or(texture_external_width_mm));
        };

    SurfaceCollection fill_surfaces_before;
    ExPolygons fill_no_overlap_before;
    if (use_legacy_perimeter_path_modulation || use_perimeter_path_modulation_v2) {
        fill_surfaces_before = *fill_surfaces;
        fill_no_overlap_before = *fill_no_overlap;
    }

    const std::optional<float> initial_texture_external_width_mm =
        use_perimeter_path_modulation ? std::optional<float>(texture_external_width_mm) : std::optional<float>();
    process_slices_with_top_visible_recolor(perimeter_slices, initial_texture_external_width_mm);

    if ((use_legacy_perimeter_path_modulation || use_perimeter_path_modulation_v2) &&
        this->perimeters.entities.empty() &&
        this->thin_fills.entities.empty()) {
        this->perimeters.clear();
        this->thin_fills.clear();
        *fill_surfaces = fill_surfaces_before;
        *fill_no_overlap = fill_no_overlap_before;
        fill_surfaces_before = *fill_surfaces;
        fill_no_overlap_before = *fill_no_overlap;
        process_slices(fallback_original_slices, std::optional<float>(texture_external_width_mm));
        const bool original_texture_has_extrusions =
            !this->perimeters.entities.empty() || !this->thin_fills.entities.empty();
        const std::optional<float> reduced_external_width_mm = perimeter_texture_min_external_width(this->perimeters);
        const bool has_reduced_external_width =
            reduced_external_width_mm &&
            *reduced_external_width_mm < texture_external_width_mm - float(EPSILON);
        if (has_reduced_external_width && perimeter_path_zone->recolor_small_perimeter_loops) {
            if (perimeter_texture_apply_recolor_small_perimeter_loops(*this,
                                                                      *perimeter_path_zone,
                                                                      perimeter_texture_zone_id,
                                                                      texture_external_width_mm)) {
                set_perimeter_path_modulation_v2_fallback_slices(*fallback_original_slices, false);
                return;
            }
        }
        if (has_reduced_external_width) {
            this->perimeters.clear();
            this->thin_fills.clear();
            *fill_surfaces = fill_surfaces_before;
            *fill_no_overlap = fill_no_overlap_before;
            SurfaceCollection reduced_modulated_slices =
                perimeter_path_modulated_surfaces(*this,
                                                  *fallback_original_slices,
                                                  *perimeter_path_zone,
                                                  perimeter_texture_zone_id,
                                                  reduced_external_width_mm,
                                                  nullptr,
                                                  nullptr);
            process_slices_with_top_visible_recolor(&reduced_modulated_slices, reduced_external_width_mm);
            if (!this->perimeters.entities.empty() || !this->thin_fills.entities.empty())
                set_perimeter_path_modulation_v2_fallback_slices(reduced_modulated_slices, true);
        }
        if (!has_reduced_external_width && (!this->perimeters.entities.empty() || !this->thin_fills.entities.empty()))
            set_perimeter_path_modulation_v2_fallback_slices(*fallback_original_slices, false);
        if (this->perimeters.entities.empty() && this->thin_fills.entities.empty()) {
            this->perimeters.clear();
            this->thin_fills.clear();
            *fill_surfaces = fill_surfaces_before;
            *fill_no_overlap = fill_no_overlap_before;
            const std::optional<float> fallback_texture_external_width_mm =
                original_texture_has_extrusions ? std::optional<float>(texture_external_width_mm) : std::optional<float>();
            process_slices(fallback_original_slices, fallback_texture_external_width_mm);
            set_perimeter_path_modulation_v2_fallback_slices(*fallback_original_slices, false);
        }
    }
}

#if 1

// Extract surfaces of given type from surfaces, extract fill (layer) thickness of one of the surfaces.
static ExPolygons fill_surfaces_extract_expolygons(Surfaces &surfaces, std::initializer_list<SurfaceType> surface_types, double &thickness)
{
    size_t cnt = 0;
    for (const Surface &surface : surfaces)
        if (std::find(surface_types.begin(), surface_types.end(), surface.surface_type) != surface_types.end()) {
            ++cnt;
            thickness = surface.thickness;
        }
    if (cnt == 0)
        return {};

    ExPolygons out;
    out.reserve(cnt);
    for (Surface &surface : surfaces)
        if (std::find(surface_types.begin(), surface_types.end(), surface.surface_type) != surface_types.end())
            out.emplace_back(std::move(surface.expolygon));
    return out;
}

struct ExpansionZone
{
    ExPolygons                           expolygons;
    Algorithm::RegionExpansionParameters parameters;
    bool                                 expanded_into = false;
};

// Cache for detecting bridge orientation and merging regions with overlapping expansions.
struct Bridge {
    ExPolygon expolygon;
    uint32_t group_id;
    std::vector<Algorithm::RegionExpansionEx>::const_iterator bridge_expansion_begin;
    std::optional<double> angle{std::nullopt};
};

// Group the bridge surfaces by overlaps.
uint32_t group_id(std::vector<Bridge> &bridges, uint32_t src_id) {
    uint32_t group_id = bridges[src_id].group_id;
    while (group_id != src_id) {
        src_id = group_id;
        group_id = bridges[src_id].group_id;
    }
    bridges[src_id].group_id = group_id;
    return group_id;
};

std::vector<Bridge> get_grouped_bridges(
    ExPolygons&& bridge_expolygons,
    const std::vector<Algorithm::RegionExpansionEx>& bridge_expansions
) {
    using namespace Algorithm;

    std::vector<Bridge> result;
    {
        result.reserve(bridge_expansions.size());
        uint32_t group_id = 0;
        using std::move_iterator;
        for (ExPolygon& expolygon : bridge_expolygons)
            result.push_back({ std::move(expolygon), group_id ++, bridge_expansions.end() });
    }


    // Detect overlaps of bridge anchors inside their respective shell regions.
    // bridge_expansions are sorted by boundary id and source id.
    for (auto expansion_iterator = bridge_expansions.begin(); expansion_iterator != bridge_expansions.end();) {
        auto boundary_region_begin = expansion_iterator;
        auto boundary_region_end = std::find_if(
            next(expansion_iterator),
            bridge_expansions.end(),
            [&](const RegionExpansionEx& expansion){
                return expansion.boundary_id != expansion_iterator->boundary_id;
            }
        );

        // Cache of bboxes per expansion boundary.
        std::vector<BoundingBox> bounding_boxes;
        bounding_boxes.reserve(std::distance(boundary_region_begin, boundary_region_end));
        std::transform(
            boundary_region_begin,
            boundary_region_end,
            std::back_inserter(bounding_boxes),
            [](const RegionExpansionEx& expansion){
                return get_extents(expansion.expolygon.contour);
            }
        );

        // For each bridge anchor of the current source:
        for (;expansion_iterator != boundary_region_end; ++expansion_iterator) {
            auto candidate_iterator = std::next(expansion_iterator);
            for (;candidate_iterator != boundary_region_end; ++candidate_iterator) {
                const BoundingBox& current_bounding_box{
                    bounding_boxes[expansion_iterator - boundary_region_begin]
                };
                const BoundingBox& candidate_bounding_box{
                    bounding_boxes[candidate_iterator - boundary_region_begin]
                };
                if (
                    expansion_iterator->src_id != candidate_iterator->src_id
                    && current_bounding_box.overlap(candidate_bounding_box)
                    // One may ignore holes, they are irrelevant for intersection test.
                    && !intersection(expansion_iterator->expolygon.contour, candidate_iterator->expolygon.contour).empty()
                ) {
                    // The two bridge regions intersect. Give them the same (lower) group id.
                    uint32_t id  = group_id(result, expansion_iterator->src_id);
                    uint32_t id2 = group_id(result, candidate_iterator->src_id);
                    if (id < id2)
                        result[id2].group_id = id;
                    else
                        result[id].group_id = id2;
                }
            }
        }
    }
    return result;
}

void detect_bridge_directions(
    const Algorithm::WaveSeeds& bridge_anchors,
    std::vector<Bridge>& bridges,
    const std::vector<ExpansionZone>& expansion_zones
) {
    if (expansion_zones.empty()) {
        throw std::runtime_error("At least one expansion zone must exist!");
    }
    auto it_bridge_anchor = bridge_anchors.begin();
    for (uint32_t bridge_id = 0; bridge_id < uint32_t(bridges.size()); ++ bridge_id) {
        Bridge &bridge = bridges[bridge_id];
        Polygons anchor_areas;
        int32_t last_anchor_id = -1;
        for (; it_bridge_anchor != bridge_anchors.end() && it_bridge_anchor->src == bridge_id; ++ it_bridge_anchor) {
            if (last_anchor_id != int(it_bridge_anchor->boundary)) {
                last_anchor_id = int(it_bridge_anchor->boundary);

                unsigned start_index{};
                unsigned end_index{};
                for (const ExpansionZone& expansion_zone: expansion_zones) {
                    end_index += expansion_zone.expolygons.size();
                    if (last_anchor_id < static_cast<int64_t>(end_index)) {
                        append(anchor_areas, to_polygons(expansion_zone.expolygons[last_anchor_id - start_index]));
                        break;
                    }
                    start_index += expansion_zone.expolygons.size();
                }
            }
        }
        Lines lines{to_lines(diff_pl(to_polylines(bridge.expolygon), expand(anchor_areas, float(SCALED_EPSILON))))};
        auto [bridging_dir, unsupported_dist] = detect_bridging_direction(lines, to_polygons(bridge.expolygon));
        bridge.angle = M_PI + std::atan2(bridging_dir.y(), bridging_dir.x());

        if constexpr (false) {
            coordf_t    stroke_width = scale_(0.06);
            BoundingBox bbox         = get_extents(anchor_areas);
            bbox.merge(get_extents(bridge.expolygon));
            bbox.offset(scale_(1.));
            ::Slic3r::SVG
                svg(debug_out_path(("bridge" + std::to_string(*bridge.angle) + "_" /* + std::to_string(this->layer()->bottom_z())*/).c_str()),
                bbox);
            svg.draw(bridge.expolygon, "cyan");
            svg.draw(lines, "green", stroke_width);
            svg.draw(anchor_areas, "red");
        }
    }
}

Surfaces merge_bridges(
    std::vector<Bridge>& bridges,
    const std::vector<Algorithm::RegionExpansionEx>& bridge_expansions,
    const float closing_radius
) {
    for (auto it = bridge_expansions.begin(); it != bridge_expansions.end(); ) {
        bridges[it->src_id].bridge_expansion_begin = it;
        uint32_t src_id = it->src_id;
        for (++ it; it != bridge_expansions.end() && it->src_id == src_id; ++ it) ;
    }

    Surfaces result;
    for (uint32_t bridge_id = 0; bridge_id < uint32_t(bridges.size()); ++ bridge_id) {
        if (group_id(bridges, bridge_id) == bridge_id) {
            // Head of the group.
            Polygons acc;
            for (uint32_t bridge_id2 = bridge_id; bridge_id2 < uint32_t(bridges.size()); ++ bridge_id2)
                if (group_id(bridges, bridge_id2) == bridge_id) {
                    append(acc, to_polygons(std::move(bridges[bridge_id2].expolygon)));
                    auto it_bridge_expansion = bridges[bridge_id2].bridge_expansion_begin;
                    assert(it_bridge_expansion == bridge_expansions.end() || it_bridge_expansion->src_id == bridge_id2);
                    for (; it_bridge_expansion != bridge_expansions.end() && it_bridge_expansion->src_id == bridge_id2; ++ it_bridge_expansion)
                        append(acc, to_polygons(it_bridge_expansion->expolygon));
                }
            //FIXME try to be smart and pick the best bridging angle for all?
            if (!bridges[bridge_id].angle) {
                assert(false && "Bridge angle must be pre-calculated!");
            }
            Surface templ{ stBottomBridge, {} };
            templ.bridge_angle = bridges[bridge_id].angle ? *bridges[bridge_id].angle : -1;
            //NOTE: The current regularization of the shells can create small unasigned regions in the object (E.G. benchy)
            // without the following closing operation, those regions will stay unfilled and cause small holes in the expanded surface.
            // look for narrow_ensure_vertical_wall_thickness_region_radius filter.
            ExPolygons final = closing_ex(acc, closing_radius);
            // without safety offset, artifacts are generated (GH #2494)
            // union_safety_offset_ex(acc)
            for (ExPolygon &ex : final)
                result.emplace_back(templ, std::move(ex));
        }
    }
    return result;
}

struct ExpansionResult {
    Algorithm::WaveSeeds anchors;
    std::vector<Algorithm::RegionExpansionEx> expansions;
};

ExpansionResult expand_expolygons(
    const ExPolygons& expolygons,
    std::vector<ExpansionZone>& expansion_zones
) {
    using namespace Algorithm;
    WaveSeeds bridge_anchors;
    std::vector<RegionExpansionEx> bridge_expansions;

    unsigned processed_bridges_count = 0;
    for (ExpansionZone& expansion_zone : expansion_zones) {
        WaveSeeds seeds{wave_seeds(
            expolygons,
            expansion_zone.expolygons,
            expansion_zone.parameters.tiny_expansion,
            true
        )};
        std::vector<RegionExpansionEx> expansions{propagate_waves_ex(
            seeds,
            expansion_zone.expolygons,
            expansion_zone.parameters
        )};

        for (WaveSeed &seed : seeds)
            seed.boundary += processed_bridges_count;
        for (RegionExpansionEx &expansion : expansions)
            expansion.boundary_id += processed_bridges_count;

        expansion_zone.expanded_into = ! expansions.empty();

        append(bridge_anchors, std::move(seeds));
        append(bridge_expansions, std::move(expansions));

        processed_bridges_count += expansion_zone.expolygons.size();
    }
    return {bridge_anchors, bridge_expansions};
}

// Extract bridging surfaces from "surfaces", expand them into "shells" using expansion_params,
// detect bridges.
// Trim "shells" by the expanded bridges.
Surfaces expand_bridges_detect_orientations(
    Surfaces &surfaces,
    std::vector<ExpansionZone>& expansion_zones,
    const float closing_radius
)
{
    using namespace Slic3r::Algorithm;

    double thickness;
    ExPolygons bridge_expolygons = fill_surfaces_extract_expolygons(surfaces, {stBottomBridge}, thickness);
    if (bridge_expolygons.empty())
        return {};

    // Calculate bridge anchors and their expansions in their respective shell region.
    ExpansionResult expansion_result{expand_expolygons(
        bridge_expolygons,
        expansion_zones
    )};

    std::vector<Bridge> bridges{get_grouped_bridges(
        std::move(bridge_expolygons),
        expansion_result.expansions
    )};
    bridge_expolygons.clear();

    std::sort(expansion_result.anchors.begin(), expansion_result.anchors.end(), Algorithm::lower_by_src_and_boundary);
    detect_bridge_directions(expansion_result.anchors, bridges, expansion_zones);

    // Merge the groups with the same group id, produce surfaces by merging source overhangs with their newly expanded anchors.
    std::sort(expansion_result.expansions.begin(), expansion_result.expansions.end(), [](auto &l, auto &r) {
        return l.src_id < r.src_id || (l.src_id == r.src_id && l.boundary_id < r.boundary_id);
    });
    Surfaces out{merge_bridges(bridges, expansion_result.expansions, closing_radius)};

    // Clip by the expanded bridges.
    for (ExpansionZone& expansion_zone : expansion_zones)
        if (expansion_zone.expanded_into)
            expansion_zone.expolygons = diff_ex(expansion_zone.expolygons, out);
    return out;
}

Surfaces expand_merge_surfaces(
    Surfaces &surfaces,
    SurfaceType surface_type,
    std::vector<ExpansionZone>& expansion_zones,
    const float closing_radius,
    const double bridge_angle = -1
)
{
    using namespace Slic3r::Algorithm;

    double thickness;
    ExPolygons src = fill_surfaces_extract_expolygons(surfaces, {surface_type}, thickness);
    if (src.empty())
        return {};

    unsigned processed_expolygons_count = 0;
    std::vector<RegionExpansion> expansions;
    for (ExpansionZone& expansion_zone : expansion_zones) {
        std::vector<RegionExpansion> zone_expansions = propagate_waves(src, expansion_zone.expolygons, expansion_zone.parameters);
        expansion_zone.expanded_into = !zone_expansions.empty();

        for (RegionExpansion &expansion : zone_expansions)
            expansion.boundary_id += processed_expolygons_count;

        processed_expolygons_count += expansion_zone.expolygons.size();
        append(expansions, std::move(zone_expansions));
    }

    std::vector<ExPolygon> expanded = merge_expansions_into_expolygons(std::move(src), std::move(expansions));
    //NOTE: The current regularization of the shells can create small unasigned regions in the object (E.G. benchy)
    // without the following closing operation, those regions will stay unfilled and cause small holes in the expanded surface.
    // look for narrow_ensure_vertical_wall_thickness_region_radius filter.
    expanded = closing_ex(expanded, closing_radius);
    // Trim the zones by the expanded expolygons.
    for (ExpansionZone& expansion_zone : expansion_zones)
        if (expansion_zone.expanded_into)
            expansion_zone.expolygons = diff_ex(expansion_zone.expolygons, expanded);

    Surface templ{ surface_type, {} };
    templ.bridge_angle = bridge_angle;
    Surfaces out;
    out.reserve(expanded.size());
    for (auto &expoly : expanded)
        out.emplace_back(templ, std::move(expoly));
    return out;
}

void LayerRegion::process_external_surfaces(const Layer *lower_layer, const Polygons *lower_layer_covered)
{
    using namespace Slic3r::Algorithm;

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
    export_region_fill_surfaces_to_svg_debug("4_process_external_surfaces-initial");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */

    // Width of the perimeters.
    float shell_width = 0;
    float expansion_min = 0;
    if (int num_perimeters = this->region().config().wall_loops; num_perimeters > 0) {
        Flow external_perimeter_flow = this->flow(frExternalPerimeter);
        Flow perimeter_flow          = this->flow(frPerimeter);
        shell_width  = 0.5f * external_perimeter_flow.scaled_width() + external_perimeter_flow.scaled_spacing();
        shell_width += perimeter_flow.scaled_spacing() * (num_perimeters - 1);
        expansion_min = perimeter_flow.scaled_spacing();
    } else {
        // TODO: Maybe there is better solution when printing with zero perimeters, but this works reasonably well, given the situation
        shell_width   = float(SCALED_EPSILON);
        expansion_min = float(SCALED_EPSILON);;
    }

    // Scaled expansions of the respective external surfaces.
    float                           expansion_top           = shell_width * sqrt(2.);
    float                           expansion_bottom        = expansion_top;
    float                           expansion_bottom_bridge = expansion_top;
    // Expand by waves of expansion_step size (expansion_step is scaled), but with no more steps than max_nr_expansion_steps.
    const float                     expansion_step          = scaled<float>(0.1);
    // Don't take more than max_nr_steps for small expansion_step.
    static constexpr const size_t   max_nr_expansion_steps  = 5;
    // Radius (with added epsilon) to absorb empty regions emering from regularization of ensuring, viz  const float narrow_ensure_vertical_wall_thickness_region_radius = 0.5f * 0.65f * min_perimeter_infill_spacing;
    const float closing_radius = 0.55f * 0.65f * 1.05f * this->flow(frSolidInfill).scaled_spacing();

    // Expand the top / bottom / bridge surfaces into the shell thickness solid infills.
    double     layer_thickness;
    ExPolygons shells = union_ex(fill_surfaces_extract_expolygons(this->fill_surfaces.surfaces, { stInternalSolid }, layer_thickness));
    ExPolygons sparse = union_ex(fill_surfaces_extract_expolygons(this->fill_surfaces.surfaces, {stInternal}, layer_thickness));
    ExPolygons top_expolygons = union_ex(fill_surfaces_extract_expolygons(this->fill_surfaces.surfaces, {stTop}, layer_thickness));
    const auto expansion_params_into_sparse_infill = RegionExpansionParameters::build(expansion_min, expansion_step, max_nr_expansion_steps);
    const auto expansion_params_into_solid_infill  = RegionExpansionParameters::build(expansion_bottom_bridge, expansion_step, max_nr_expansion_steps);

    std::vector<ExpansionZone> expansion_zones{
        ExpansionZone{std::move(shells), expansion_params_into_solid_infill},
        ExpansionZone{std::move(sparse), expansion_params_into_sparse_infill},
        ExpansionZone{std::move(top_expolygons), expansion_params_into_solid_infill},
    };

    SurfaceCollection bridges;
    {
        BOOST_LOG_TRIVIAL(trace) << "Processing external surface, detecting bridges. layer" << this->layer()->print_z;
        const double custom_angle = this->region().config().bridge_angle.value;
        bridges.surfaces = custom_angle > 0 ?
            expand_merge_surfaces(this->fill_surfaces.surfaces, stBottomBridge, expansion_zones, closing_radius, Geometry::deg2rad(custom_angle)) :
            expand_bridges_detect_orientations(this->fill_surfaces.surfaces, expansion_zones, closing_radius);
        BOOST_LOG_TRIVIAL(trace) << "Processing external surface, detecting bridges - done";
#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
        {
            static int iRun = 0;
            bridges.export_to_svg(debug_out_path("bridges-after-grouping-%d.svg", iRun++).c_str(), true);
        }
#endif
    }

    this->fill_surfaces.remove_types({stTop});
    {
        Surface top_templ(stTop, {});
        top_templ.thickness = layer_thickness;
        this->fill_surfaces.append(std::move(expansion_zones.back().expolygons), top_templ);
    }

    expansion_zones.pop_back();

    expansion_zones.at(0).parameters = RegionExpansionParameters::build(expansion_bottom, expansion_step, max_nr_expansion_steps);
    Surfaces bottoms = expand_merge_surfaces(this->fill_surfaces.surfaces, stBottom, expansion_zones, closing_radius);

    expansion_zones.at(0).parameters = RegionExpansionParameters::build(expansion_top, expansion_step, max_nr_expansion_steps);
    Surfaces tops = expand_merge_surfaces(this->fill_surfaces.surfaces, stTop, expansion_zones, closing_radius);

    // turn too small internal regions into solid regions according to the user setting
    if (!this->layer()->object()->print()->config().spiral_mode && this->region().config().sparse_infill_density.value > 0) {
        // scaling an area requires two calls!
        double min_area = scale_(scale_(this->region().config().minimum_sparse_infill_area.value));
        ExPolygons small_regions{};
        expansion_zones[1].expolygons.erase(std::remove_if(expansion_zones[1].expolygons.begin(), expansion_zones[1].expolygons.end(), [min_area, &small_regions](ExPolygon& ex_polygon) {
            if (ex_polygon.area() <= min_area) {
                small_regions.push_back(ex_polygon);
                return true;
            }
            return false;
        }), expansion_zones[1].expolygons.end());

        if (!small_regions.empty()) {
            expansion_zones[0].expolygons = union_ex(expansion_zones[0].expolygons, small_regions);
        }
    }

//    this->fill_surfaces.remove_types({ stBottomBridge, stBottom, stTop, stInternal, stInternalSolid });
    this->fill_surfaces.clear();
    unsigned zones_expolygons_count = 0;
    for (const ExpansionZone& zone : expansion_zones)
        zones_expolygons_count += zone.expolygons.size();
    reserve_more(this->fill_surfaces.surfaces, zones_expolygons_count + bridges.size() + bottoms.size() + tops.size());
    {
        Surface solid_templ(stInternalSolid, {});
        solid_templ.thickness = layer_thickness;
        this->fill_surfaces.append(std::move(expansion_zones[0].expolygons), solid_templ);
    }
    {
        Surface sparse_templ(stInternal, {});
        sparse_templ.thickness = layer_thickness;
        this->fill_surfaces.append(std::move(expansion_zones[1].expolygons), sparse_templ);
    }
    this->fill_surfaces.append(std::move(bridges.surfaces));
    this->fill_surfaces.append(std::move(bottoms));
    this->fill_surfaces.append(std::move(tops));

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
    export_region_fill_surfaces_to_svg_debug("4_process_external_surfaces-final");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */
}
#else

//#define EXTERNAL_SURFACES_OFFSET_PARAMETERS ClipperLib::jtMiter, 3.
//#define EXTERNAL_SURFACES_OFFSET_PARAMETERS ClipperLib::jtMiter, 1.5
#define EXTERNAL_SURFACES_OFFSET_PARAMETERS ClipperLib::jtSquare, 0.

void LayerRegion::process_external_surfaces(const Layer *lower_layer, const Polygons *lower_layer_covered)
{
    const bool      has_infill = this->region().config().sparse_infill_density.value > 0.;
    //BBS
    auto nozzle_diameter = this->region().nozzle_dmr_avg(this->layer()->object()->print()->config());
    const float margin = float(scale_(EXTERNAL_INFILL_MARGIN));
    const float bridge_margin = std::min(float(scale_(BRIDGE_INFILL_MARGIN)), float(scale_(nozzle_diameter * BRIDGE_INFILL_MARGIN / 0.4)));

    // BBS
    const PrintObjectConfig& object_config = this->layer()->object()->config();

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
    export_region_fill_surfaces_to_svg_debug("3_process_external_surfaces-initial");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */

    // 1) Collect bottom and bridge surfaces, each of them grown by a fixed 3mm offset
    // for better anchoring.
    // Bottom surfaces, grown.
    Surfaces                    bottom;
    // Bridge surfaces, initialy not grown.
    Surfaces                    bridges;
    // Top surfaces, grown.
    Surfaces                    top;
    // Internal surfaces, not grown.
    Surfaces                    internal;
    // Areas, where an infill of various types (top, bottom, bottom bride, sparse, void) could be placed.
    Polygons                    fill_boundaries = to_polygons(this->fill_expolygons);
    Polygons  					lower_layer_covered_tmp;

    // Collect top surfaces and internal surfaces.
    // Collect fill_boundaries: If we're slicing with no infill, we can't extend external surfaces over non-existent infill.
    // This loop destroys the surfaces (aliasing this->fill_surfaces.surfaces) by moving into top/internal/fill_boundaries!

    {
        // Voids are sparse infills if infill rate is zero.
        Polygons voids;

        double max_grid_area = -1;
        if (this->layer()->lower_layer != nullptr)
            max_grid_area = this->layer()->lower_layer->get_sparse_infill_max_void_area();
        for (const Surface &surface : this->fill_surfaces.surfaces) {
            if (surface.is_top()) {
                // Collect the top surfaces, inflate them and trim them by the bottom surfaces.
                // This gives the priority to bottom surfaces.
                if (max_grid_area < 0 || surface.expolygon.area() < max_grid_area)
                    surfaces_append(top, offset_ex(surface.expolygon, margin, EXTERNAL_SURFACES_OFFSET_PARAMETERS), surface);
                else
                    //BBS: Don't need to expand too much in this situation. Expand 3mm to eliminate hole and 1mm for contour
                    surfaces_append(top, intersection_ex(offset(surface.expolygon.contour, margin / 3.0, EXTERNAL_SURFACES_OFFSET_PARAMETERS),
                                                         offset_ex(surface.expolygon, margin, EXTERNAL_SURFACES_OFFSET_PARAMETERS)), surface);
            } else if (surface.surface_type == stBottom || (surface.surface_type == stBottomBridge && lower_layer == nullptr)) {
                // Grown by 3mm.
                surfaces_append(bottom, offset_ex(surface.expolygon, margin, EXTERNAL_SURFACES_OFFSET_PARAMETERS), surface);
            } else if (surface.surface_type == stBottomBridge) {
                if (! surface.empty())
                    bridges.emplace_back(surface);
            }
            if (surface.is_internal()) {
            	assert(surface.surface_type == stInternal || surface.surface_type == stInternalSolid);
            	if (! has_infill && lower_layer != nullptr)
            		polygons_append(voids, surface.expolygon);
            	internal.emplace_back(std::move(surface));
            }
        }
        if (! has_infill && lower_layer != nullptr && ! voids.empty()) {
        	// Remove voids from fill_boundaries, that are not supported by the layer below.
            if (lower_layer_covered == nullptr) {
            	lower_layer_covered = &lower_layer_covered_tmp;
            	lower_layer_covered_tmp = to_polygons(lower_layer->lslices);
            }
            if (! lower_layer_covered->empty())
            	voids = diff(voids, *lower_layer_covered);
            fill_boundaries = diff(fill_boundaries, voids);
        }
    }

#if 0
    {
        static int iRun = 0;
        bridges.export_to_svg(debug_out_path("bridges-before-grouping-%d.svg", iRun ++), true);
    }
#endif

    if (bridges.empty())
    {
        fill_boundaries = union_safety_offset(fill_boundaries);
    } else
    {
        // 1) Calculate the inflated bridge regions, each constrained to its island.
        ExPolygons               fill_boundaries_ex = union_safety_offset_ex(fill_boundaries);
        std::vector<Polygons>    bridges_grown;
        std::vector<BoundingBox> bridge_bboxes;

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
        {
            static int iRun = 0;
            SVG svg(debug_out_path("3_process_external_surfaces-fill_regions-%d.svg", iRun ++).c_str(), get_extents(fill_boundaries_ex));
            svg.draw(fill_boundaries_ex);
            svg.draw_outline(fill_boundaries_ex, "black", "blue", scale_(0.05)); 
            svg.Close();
        }

//        export_region_fill_surfaces_to_svg_debug("3_process_external_surfaces-initial");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */
 
        {
            // Bridge expolygons, grown, to be tested for intersection with other bridge regions.
            std::vector<BoundingBox> fill_boundaries_ex_bboxes = get_extents_vector(fill_boundaries_ex);
            bridges_grown.reserve(bridges.size());
            bridge_bboxes.reserve(bridges.size());
            for (size_t i = 0; i < bridges.size(); ++ i) {
                // Find the island of this bridge.
                const Point pt = bridges[i].expolygon.contour.points.front();
                int idx_island = -1;
                for (int j = 0; j < int(fill_boundaries_ex.size()); ++ j)
                    if (fill_boundaries_ex_bboxes[j].contains(pt) && 
                        fill_boundaries_ex[j].contains(pt)) {
                        idx_island = j;
                        break;
                    }
                // Grown by 3mm.
                //BBS: eliminate too narrow area to avoid generating bridge on top layer when wall loop is 1
                //Polygons polys = offset(bridges[i].expolygon, bridge_margin, EXTERNAL_SURFACES_OFFSET_PARAMETERS);
                Polygons polys = offset2({ bridges[i].expolygon }, -scale_(nozzle_diameter * 0.1), bridge_margin, EXTERNAL_SURFACES_OFFSET_PARAMETERS);
                if (idx_island == -1) {
				    BOOST_LOG_TRIVIAL(trace) << "Bridge did not fall into the source region!";
                } else {
                    // Found an island, to which this bridge region belongs. Trim it,
                    polys = intersection(polys, fill_boundaries_ex[idx_island]);
                }
                bridge_bboxes.push_back(get_extents(polys));
                bridges_grown.push_back(std::move(polys));
            }
        }

        // 2) Group the bridge surfaces by overlaps.
        std::vector<size_t> bridge_group(bridges.size(), (size_t)-1);
        size_t n_groups = 0; 
        for (size_t i = 0; i < bridges.size(); ++ i) {
            // A grup id for this bridge.
            size_t group_id = (bridge_group[i] == size_t(-1)) ? (n_groups ++) : bridge_group[i];
            bridge_group[i] = group_id;
            // For all possibly overlaping bridges:
            for (size_t j = i + 1; j < bridges.size(); ++ j) {
                if (! bridge_bboxes[i].overlap(bridge_bboxes[j]))
                    continue;
                if (intersection(bridges_grown[i], bridges_grown[j]).empty())
                    continue;
                // The two bridge regions intersect. Give them the same group id.
                if (bridge_group[j] != size_t(-1)) {
                    // The j'th bridge has been merged with some other bridge before.
                    size_t group_id_new = bridge_group[j];
                    for (size_t k = 0; k < j; ++ k)
                        if (bridge_group[k] == group_id)
                            bridge_group[k] = group_id_new;
                    group_id = group_id_new;
                }
                bridge_group[j] = group_id;
            }
        }

        // 3) Merge the groups with the same group id, detect bridges.
        {
			BOOST_LOG_TRIVIAL(trace) << "Processing external surface, detecting bridges. layer" << this->layer()->print_z << ", bridge groups: " << n_groups;
            for (size_t group_id = 0; group_id < n_groups; ++ group_id) {
                size_t n_bridges_merged = 0;
                size_t idx_last = (size_t)-1;
                for (size_t i = 0; i < bridges.size(); ++ i) {
                    if (bridge_group[i] == group_id) {
                        ++ n_bridges_merged;
                        idx_last = i;
                    }
                }
                if (n_bridges_merged == 0)
                    // This group has no regions assigned as these were moved into another group.
                    continue;
                // Collect the initial ungrown regions and the grown polygons.
                ExPolygons  initial;
                Polygons    grown;
                for (size_t i = 0; i < bridges.size(); ++ i) {
                    if (bridge_group[i] != group_id)
                        continue;
                    initial.push_back(std::move(bridges[i].expolygon));
                    polygons_append(grown, bridges_grown[i]);
                }
                // detect bridge direction before merging grown surfaces otherwise adjacent bridges
                // would get merged into a single one while they need different directions
                // also, supply the original expolygon instead of the grown one, because in case
                // of very thin (but still working) anchors, the grown expolygon would go beyond them
                double custom_angle = Geometry::deg2rad(this->region().config().bridge_angle.value);
                if (custom_angle > 0.0) {
                    bridges[idx_last].bridge_angle = custom_angle;
                } else {
                    auto [bridging_dir, unsupported_dist] = detect_bridging_direction(to_polygons(initial), to_polygons(lower_layer->lslices));
                    bridges[idx_last].bridge_angle = PI + std::atan2(bridging_dir.y(), bridging_dir.x());
                }

                /*
                BridgeDetector bd(initial, lower_layer->lslices, this->bridging_flow(frInfill, object_config.thick_bridges).scaled_width());
                #ifdef SLIC3R_DEBUG
                printf("Processing bridge at layer %zu:\n", this->layer()->id());
                #endif
                //BBS: use 0 as custom angle to enable auto detection all the time
                double custom_angle = Geometry::deg2rad(this->region().config().bridge_angle.value);
                if(custom_angle > 0)
                        bridges[idx_last].bridge_angle = custom_angle;
				else if (bd.detect_angle(custom_angle)) {
                    bridges[idx_last].bridge_angle = bd.angle;
                    if (this->layer()->object()->has_support()) {
//                        polygons_append(this->bridged, bd.coverage());
                        append(this->unsupported_bridge_edges, bd.unsupported_edges());
                    }
				} else if (custom_angle > 0) {
					// Bridge was not detected (likely it is only supported at one side). Still it is a surface filled in
					// using a bridging flow, therefore it makes sense to respect the custom bridging direction.
					bridges[idx_last].bridge_angle = custom_angle;
				}
                */
                // without safety offset, artifacts are generated (GH #2494)
                surfaces_append(bottom, union_safety_offset_ex(grown), bridges[idx_last]);
            }

            fill_boundaries = to_polygons(fill_boundaries_ex);
			BOOST_LOG_TRIVIAL(trace) << "Processing external surface, detecting bridges - done";
		}

    #if 0
        {
            static int iRun = 0;
            bridges.export_to_svg(debug_out_path("bridges-after-grouping-%d.svg", iRun ++), true);
        }
    #endif
    }

    Surfaces new_surfaces;
    {
        // Merge top and bottom in a single collection.
        surfaces_append(top, std::move(bottom));
        // Intersect the grown surfaces with the actual fill boundaries.
        Polygons bottom_polygons = to_polygons(bottom);
        for (size_t i = 0; i < top.size(); ++ i) {
            Surface &s1 = top[i];
            if (s1.empty())
                continue;
            Polygons polys;
            polygons_append(polys, to_polygons(std::move(s1)));
            for (size_t j = i + 1; j < top.size(); ++ j) {
                Surface &s2 = top[j];
                if (! s2.empty() && surfaces_could_merge(s1, s2)) {
                    polygons_append(polys, to_polygons(std::move(s2)));
                    s2.clear();
                }
            }
            if (s1.is_top())
                // Trim the top surfaces by the bottom surfaces. This gives the priority to the bottom surfaces.
                polys = diff(polys, bottom_polygons);
            surfaces_append(
                new_surfaces,
                // Don't use a safety offset as fill_boundaries were already united using the safety offset.
                intersection_ex(polys, fill_boundaries),
                s1);
        }
    }
    
    // Subtract the new top surfaces from the other non-top surfaces and re-add them.
    Polygons new_polygons = to_polygons(new_surfaces);
    for (size_t i = 0; i < internal.size(); ++ i) {
        Surface &s1 = internal[i];
        if (s1.empty())
            continue;
        Polygons polys;
        polygons_append(polys, to_polygons(std::move(s1)));
        for (size_t j = i + 1; j < internal.size(); ++ j) {
            Surface &s2 = internal[j];
            if (! s2.empty() && surfaces_could_merge(s1, s2)) {
                polygons_append(polys, to_polygons(std::move(s2)));
                s2.clear();
            }
        }
        ExPolygons new_expolys = diff_ex(polys, new_polygons);
        polygons_append(new_polygons, to_polygons(new_expolys));
        surfaces_append(new_surfaces, std::move(new_expolys), s1);
    }
    
    this->fill_surfaces.surfaces = std::move(new_surfaces);

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
    export_region_fill_surfaces_to_svg_debug("3_process_external_surfaces-final");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */
}
#endif

void LayerRegion::prepare_fill_surfaces()
{
#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
    export_region_slices_to_svg_debug("2_prepare_fill_surfaces-initial");
    export_region_fill_surfaces_to_svg_debug("2_prepare_fill_surfaces-initial");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */ 

    /*  Note: in order to make the psPrepareInfill step idempotent, we should never
        alter fill_surfaces boundaries on which our idempotency relies since that's
        the only meaningful information returned by psPerimeters. */
    
    bool spiral_mode = this->layer()->object()->print()->config().spiral_mode;

    // if no solid layers are requested, turn top/bottom surfaces to internal
    if (! spiral_mode && this->region().config().top_shell_layers == 0) {
        for (Surface &surface : this->fill_surfaces.surfaces)
            if (surface.is_top())
                //BBS
                //surface.surface_type = this->layer()->object()->config().infill_only_where_needed ? stInternalVoid : stInternal;
                surface.surface_type = PrintObject::infill_only_where_needed ? stInternalVoid : stInternal;
    }
    if (this->region().config().bottom_shell_layers == 0) {
        for (Surface &surface : this->fill_surfaces.surfaces)
            if (surface.is_bottom()) // (surface.surface_type == stBottom)
                surface.surface_type = stInternal;
    }

    if (!spiral_mode && fabs(this->region().config().sparse_infill_density.value - 100.) < EPSILON) {
        // Turn all internal sparse infill into solid infill, if sparse_infill_density is 100%
        for (Surface &surface : this->fill_surfaces.surfaces)
            if (surface.surface_type == stInternal)
                surface.surface_type = stInternalSolid;
    }

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
    export_region_slices_to_svg_debug("2_prepare_fill_surfaces-final");
    export_region_fill_surfaces_to_svg_debug("2_prepare_fill_surfaces-final");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */
}

double LayerRegion::infill_area_threshold() const
{
    double ss = this->flow(frSolidInfill).scaled_spacing();
    return ss*ss;
}

void LayerRegion::trim_surfaces(const Polygons &trimming_polygons)
{
#ifndef NDEBUG
    for (const Surface &surface : this->slices.surfaces)
        assert(surface.surface_type == stInternal);
#endif /* NDEBUG */
	this->slices.set(intersection_ex(this->slices.surfaces, trimming_polygons), stInternal);
}

void LayerRegion::elephant_foot_compensation_step(const float elephant_foot_compensation_perimeter_step, const Polygons &trimming_polygons)
{
#ifndef NDEBUG
    for (const Surface &surface : this->slices.surfaces)
        assert(surface.surface_type == stInternal);
#endif /* NDEBUG */
    Polygons tmp = intersection(this->slices.surfaces, trimming_polygons);
    append(tmp, diff(this->slices.surfaces, opening(this->slices.surfaces, elephant_foot_compensation_perimeter_step)));
    this->slices.set(union_ex(tmp), stInternal);
}

void LayerRegion::export_region_slices_to_svg(const char *path) const
{
    BoundingBox bbox;
    for (Surfaces::const_iterator surface = this->slices.surfaces.begin(); surface != this->slices.surfaces.end(); ++surface)
        bbox.merge(get_extents(surface->expolygon));
    Point legend_size = export_surface_type_legend_to_svg_box_size();
    Point legend_pos(bbox.min(0), bbox.max(1));
    bbox.merge(Point(std::max(bbox.min(0) + legend_size(0), bbox.max(0)), bbox.max(1) + legend_size(1)));

    SVG svg(path, bbox);
    const float transparency = 0.5f;
    for (Surfaces::const_iterator surface = this->slices.surfaces.begin(); surface != this->slices.surfaces.end(); ++surface)
        svg.draw(surface->expolygon, surface_type_to_color_name(surface->surface_type), transparency);
    for (Surfaces::const_iterator surface = this->fill_surfaces.surfaces.begin(); surface != this->fill_surfaces.surfaces.end(); ++surface)
        svg.draw(surface->expolygon.lines(), surface_type_to_color_name(surface->surface_type));
    export_surface_type_legend_to_svg(svg, legend_pos);
    svg.Close();
}

// Export to "out/LayerRegion-name-%d.svg" with an increasing index with every export.
void LayerRegion::export_region_slices_to_svg_debug(const char *name) const
{
    static std::map<std::string, size_t> idx_map;
    size_t &idx = idx_map[name];
    this->export_region_slices_to_svg(debug_out_path("LayerRegion-slices-%s-%d.svg", name, idx ++).c_str());
}

void LayerRegion::export_region_fill_surfaces_to_svg(const char *path) const
{
    BoundingBox bbox;
    for (Surfaces::const_iterator surface = this->fill_surfaces.surfaces.begin(); surface != this->fill_surfaces.surfaces.end(); ++surface)
        bbox.merge(get_extents(surface->expolygon));
    Point legend_size = export_surface_type_legend_to_svg_box_size();
    Point legend_pos(bbox.min(0), bbox.max(1));
    bbox.merge(Point(std::max(bbox.min(0) + legend_size(0), bbox.max(0)), bbox.max(1) + legend_size(1)));

    SVG svg(path, bbox);
    const float transparency = 0.5f;
    for (const Surface &surface : this->fill_surfaces.surfaces) {
        svg.draw(surface.expolygon, surface_type_to_color_name(surface.surface_type), transparency);
        svg.draw_outline(surface.expolygon, "black", "blue", scale_(0.05)); 
    }
    export_surface_type_legend_to_svg(svg, legend_pos);
    svg.Close();
}

// Export to "out/LayerRegion-name-%d.svg" with an increasing index with every export.
void LayerRegion::export_region_fill_surfaces_to_svg_debug(const char *name) const
{
    static std::map<std::string, size_t> idx_map;
    size_t &idx = idx_map[name];
    this->export_region_fill_surfaces_to_svg(debug_out_path("LayerRegion-fill_surfaces-%s-%d.svg", name, idx ++).c_str());
}

void LayerRegion::simplify_entity_collection(ExtrusionEntityCollection* entity_collection)
{
    for (size_t i = 0; i < entity_collection->entities.size(); i++) {
        if (ExtrusionEntityCollection* collection = dynamic_cast<ExtrusionEntityCollection*>(entity_collection->entities[i]))
            this->simplify_entity_collection(collection);
        else if (ExtrusionPath* path = dynamic_cast<ExtrusionPath*>(entity_collection->entities[i]))
            this->simplify_path(path);
        else if (ExtrusionMultiPath* multipath = dynamic_cast<ExtrusionMultiPath*>(entity_collection->entities[i]))
            this->simplify_multi_path(multipath);
        else if (ExtrusionLoop* loop = dynamic_cast<ExtrusionLoop*>(entity_collection->entities[i]))
            this->simplify_loop(loop);
        else
            throw Slic3r::InvalidArgument("Invalid extrusion entity supplied to simplify_entity_collection()");
    }
}

void LayerRegion::simplify_path(ExtrusionPath* path)
{
    const auto print_config = this->layer()->object()->print()->config();
    const bool spiral_mode = print_config.spiral_mode;
    const bool enable_arc_fitting = print_config.enable_arc_fitting;
    const auto scaled_resolution = scaled<double>(print_config.resolution.value);

    if (enable_arc_fitting &&
        !spiral_mode) {
        if (path->role() == erInternalInfill)
            path->simplify_by_fitting_arc(SCALED_SPARSE_INFILL_RESOLUTION);
        else
            path->simplify_by_fitting_arc(scaled_resolution);
    } else {
        path->simplify(scaled_resolution);
    }
}

void LayerRegion::simplify_multi_path(ExtrusionMultiPath* multipath)
{
    const auto print_config = this->layer()->object()->print()->config();
    const bool spiral_mode = print_config.spiral_mode;
    const bool enable_arc_fitting = print_config.enable_arc_fitting;
    const auto scaled_resolution = scaled<double>(print_config.resolution.value);

    for (size_t i = 0; i < multipath->paths.size(); ++i) {
        if (enable_arc_fitting &&
            !spiral_mode) {
            if (multipath->paths[i].role() == erInternalInfill)
                multipath->paths[i].simplify_by_fitting_arc(SCALED_SPARSE_INFILL_RESOLUTION);
            else
                multipath->paths[i].simplify_by_fitting_arc(scaled_resolution);
        } else {
            multipath->paths[i].simplify(scaled_resolution);
        }
    }
}

void LayerRegion::simplify_loop(ExtrusionLoop* loop)
{
    const auto print_config = this->layer()->object()->print()->config();
    const bool spiral_mode = print_config.spiral_mode;
    const bool enable_arc_fitting = print_config.enable_arc_fitting;
    const auto scaled_resolution = scaled<double>(print_config.resolution.value);

    for (size_t i = 0; i < loop->paths.size(); ++i) {
        if (enable_arc_fitting &&
            !spiral_mode) {
            if (loop->paths[i].role() == erInternalInfill)
                loop->paths[i].simplify_by_fitting_arc(SCALED_SPARSE_INFILL_RESOLUTION);
            else
                loop->paths[i].simplify_by_fitting_arc(scaled_resolution);
        } else {
            loop->paths[i].simplify(scaled_resolution);
        }
    }
}

}
 
