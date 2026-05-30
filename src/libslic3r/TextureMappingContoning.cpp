// original author: sentientstardust

#include "TextureMappingContoning.hpp"

#include "Color.hpp"
#include "ColorSolver.hpp"
#include "PrintConfig.hpp"
#include "libslic3r.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

namespace Slic3r {
namespace {

constexpr float OPAQUE_CONTONING_TD_THRESHOLD_MM = 0.5f;
constexpr float INFERRED_BLACK_TD_MM = 0.1f;
constexpr float CONTONING_SURFACE_SCATTER = 0.12f;
constexpr size_t MAX_TD_ORDERED_CONTONING_CANDIDATES = 2500000;
constexpr size_t MAX_TD_ORDERED_CONTONING_STACK_ITEMS = 20000000;

float clamp01(float value)
{
    if (!std::isfinite(value))
        return 0.f;
    return std::clamp(value, 0.f, 1.f);
}

float filament_luminance(const PrintConfig &config, unsigned int component_id)
{
    ColorRGB color;
    if (component_id == 0 || component_id > config.filament_colour.values.size() ||
        !decode_color(config.filament_colour.get_at(size_t(component_id - 1)), color))
        return std::numeric_limits<float>::max();
    return 0.2126f * color.r() + 0.7152f * color.g() + 0.0722f * color.b();
}

bool explicit_transmission_distance_mm(const TextureMappingZone &zone, unsigned int component_id, float &td_mm)
{
    if (component_id == 0)
        return false;
    const size_t idx = size_t(component_id - 1);
    if (idx >= zone.filament_transmission_distances_mm.size())
        return false;
    const float td = zone.filament_transmission_distances_mm[idx];
    if (!std::isfinite(td) || td <= 0.f)
        return false;
    td_mm = td;
    return true;
}

bool black_role_component(int filament_color_mode, size_t component_idx, size_t component_count)
{
    switch (std::clamp(filament_color_mode,
                       int(TextureMappingZone::FilamentColorAny),
                       int(TextureMappingZone::FilamentColorRGBKW))) {
    case int(TextureMappingZone::FilamentColorCMYK):
    case int(TextureMappingZone::FilamentColorRGBK):
        return component_count == 4 && component_idx == 3;
    case int(TextureMappingZone::FilamentColorBW):
        return component_count == 2 && component_idx == 0;
    case int(TextureMappingZone::FilamentColorCMYKW):
    case int(TextureMappingZone::FilamentColorRGBKW):
        return component_count == 5 && component_idx == 3;
    default:
        return false;
    }
}

bool is_black_color_filament(const PrintConfig &config, unsigned int component_id)
{
    ColorRGB color;
    if (component_id == 0 || component_id > config.filament_colour.values.size() ||
        !decode_color(config.filament_colour.get_at(size_t(component_id - 1)), color))
        return false;
    const float r = clamp01(color.r());
    const float g = clamp01(color.g());
    const float b = clamp01(color.b());
    const float max_channel = std::max({ r, g, b });
    const float min_channel = std::min({ r, g, b });
    const float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    return max_channel <= 0.20f && luminance <= 0.12f && max_channel - min_channel <= 0.08f;
}

float estimated_transmission_distance_mm(const PrintConfig &config, unsigned int component_id)
{
    const float luminance = std::clamp(filament_luminance(config, component_id), 0.f, 1.f);
    return std::clamp(0.1f + 2.9f * std::pow(luminance, 1.35f), 0.12f, 3.f);
}

std::vector<float> effective_transmission_distances_mm(const TextureMappingZone &zone,
                                                       const PrintConfig        &config,
                                                       const std::vector<unsigned int> &component_ids,
                                                       bool estimate_missing)
{
    std::vector<float> out(component_ids.size(), 0.f);
    for (size_t idx = 0; idx < component_ids.size(); ++idx) {
        float td_mm = 0.f;
        if (explicit_transmission_distance_mm(zone, component_ids[idx], td_mm)) {
            out[idx] = td_mm;
        } else if (black_role_component(zone.filament_color_mode, idx, component_ids.size()) ||
                   is_black_color_filament(config, component_ids[idx])) {
            out[idx] = INFERRED_BLACK_TD_MM;
        } else if (estimate_missing) {
            out[idx] = estimated_transmission_distance_mm(config, component_ids[idx]);
        }
    }
    return out;
}

float layer_opacity_from_td(float td_mm, float layer_height_mm)
{
    const float safe_td = std::clamp(td_mm, 0.01f, 50.f);
    const float safe_layer_height = std::clamp(layer_height_mm, 0.01f, 2.f);
    return std::clamp(1.f - std::pow(0.05f, 2.f * safe_layer_height / safe_td), 1e-4f, 0.9999f);
}

float effective_transmission_distance_for_component(const std::vector<unsigned int> &component_ids,
                                                    const std::vector<float>        &effective_tds,
                                                    unsigned int                     component_id)
{
    const auto it = std::find(component_ids.begin(), component_ids.end(), component_id);
    if (it == component_ids.end())
        return 0.f;
    const size_t idx = size_t(it - component_ids.begin());
    return idx < effective_tds.size() ? effective_tds[idx] : 0.f;
}

float perceptual_error(const std::array<float, 3> &lhs, const std::array<float, 3> &rhs)
{
    const float dl = lhs[0] - rhs[0];
    const float da = lhs[1] - rhs[1];
    const float db = lhs[2] - rhs[2];
    return dl * dl + 4.f * da * da + 4.f * db * db;
}

void enumerate_counts(size_t component_idx,
                      int remaining,
                      std::vector<int> &counts,
                      std::vector<std::vector<int>> &out)
{
    if (component_idx + 1 == counts.size()) {
        counts[component_idx] = remaining;
        out.emplace_back(counts);
        return;
    }
    for (int count = 0; count <= remaining; ++count) {
        counts[component_idx] = count;
        enumerate_counts(component_idx + 1, remaining - count, counts, out);
    }
}

bool can_finish_without_repeat(const std::vector<unsigned int> &ids,
                               const std::vector<int>          &counts,
                               unsigned int                     previous_id,
                               int                              remaining)
{
    for (size_t idx = 0; idx < ids.size() && idx < counts.size(); ++idx) {
        const int limit = ids[idx] == previous_id ? remaining / 2 : (remaining + 1) / 2;
        if (counts[idx] > limit)
            return false;
    }
    return true;
}

void spread_surface_repeats(std::vector<unsigned int> &surface_to_deep)
{
    if (surface_to_deep.size() < 3)
        return;

    std::vector<unsigned int> ids;
    std::vector<int> counts;
    std::vector<int> ranks;
    ids.reserve(surface_to_deep.size());
    counts.reserve(surface_to_deep.size());
    ranks.reserve(surface_to_deep.size());
    for (size_t pos = 0; pos < surface_to_deep.size(); ++pos) {
        const unsigned int id = surface_to_deep[pos];
        auto it = std::find(ids.begin(), ids.end(), id);
        if (it == ids.end()) {
            ids.emplace_back(id);
            counts.emplace_back(1);
            ranks.emplace_back(int(pos));
        } else {
            ++counts[size_t(it - ids.begin())];
        }
    }
    if (ids.size() < 2)
        return;

    std::vector<unsigned int> reordered;
    reordered.reserve(surface_to_deep.size());
    unsigned int previous_id = surface_to_deep.front();
    int remaining = int(surface_to_deep.size());
    const auto first_it = std::find(ids.begin(), ids.end(), previous_id);
    if (first_it != ids.end()) {
        const size_t first_idx = size_t(first_it - ids.begin());
        reordered.emplace_back(previous_id);
        --counts[first_idx];
        --remaining;
    }

    auto better_candidate = [&counts, &ranks, &ids](int lhs, int rhs) {
        if (rhs < 0)
            return true;
        if (ranks[size_t(lhs)] != ranks[size_t(rhs)])
            return ranks[size_t(lhs)] < ranks[size_t(rhs)];
        if (counts[size_t(lhs)] != counts[size_t(rhs)])
            return counts[size_t(lhs)] > counts[size_t(rhs)];
        return ids[size_t(lhs)] < ids[size_t(rhs)];
    };

    auto select_candidate = [&](bool require_feasible, bool allow_repeat) {
        int best = -1;
        for (size_t idx = 0; idx < ids.size(); ++idx) {
            if (counts[idx] <= 0 || (!allow_repeat && ids[idx] == previous_id))
                continue;
            --counts[idx];
            const bool feasible =
                can_finish_without_repeat(ids, counts, ids[idx], remaining - 1);
            ++counts[idx];
            if (require_feasible && !feasible)
                continue;
            if (better_candidate(int(idx), best))
                best = int(idx);
        }
        return best;
    };

    while (remaining > 0) {
        int selected = select_candidate(true, false);
        if (selected < 0)
            selected = select_candidate(false, false);
        if (selected < 0)
            selected = select_candidate(false, true);
        if (selected < 0)
            break;
        reordered.emplace_back(ids[size_t(selected)]);
        --counts[size_t(selected)];
        previous_id = ids[size_t(selected)];
        --remaining;
    }

    if (reordered.size() == surface_to_deep.size())
        surface_to_deep = std::move(reordered);
}

}

std::optional<std::array<float, 3>> texture_mapping_contoning_component_colors(
    const PrintConfig &config,
    const std::vector<unsigned int> &component_ids,
    std::vector<std::array<float, 3>> &out)
{
    out.clear();
    out.reserve(component_ids.size());
    for (const unsigned int id : component_ids) {
        if (id == 0 || id > config.filament_colour.values.size())
            return std::nullopt;
        ColorRGB color;
        if (!decode_color(config.filament_colour.get_at(size_t(id - 1)), color))
            return std::nullopt;
        out.push_back({ color.r(), color.g(), color.b() });
    }
    if (out.empty())
        return std::nullopt;
    return out.front();
}

std::vector<unsigned int> texture_mapping_contoning_components_bottom_to_top(
    const TextureMappingZone &zone,
    const PrintConfig &config,
    std::vector<unsigned int> component_ids)
{
    component_ids.erase(std::remove_if(component_ids.begin(), component_ids.end(), [](unsigned int id) {
        return id == 0;
    }), component_ids.end());
    component_ids.erase(std::unique(component_ids.begin(), component_ids.end()), component_ids.end());
    if (component_ids.empty())
        return component_ids;

    const std::vector<unsigned int> td_component_ids = component_ids;
    const std::vector<float> effective_tds =
        effective_transmission_distances_mm(zone, config, td_component_ids, zone.top_surface_contoning_td_adjustment_enabled);
    const bool has_complete_td =
        std::all_of(effective_tds.begin(), effective_tds.end(), [](float td) { return std::isfinite(td) && td > 0.f; });
    if (has_complete_td) {
        std::stable_sort(component_ids.begin(), component_ids.end(), [&effective_tds, &td_component_ids](unsigned int lhs, unsigned int rhs) {
            return effective_transmission_distance_for_component(td_component_ids, effective_tds, lhs) <
                   effective_transmission_distance_for_component(td_component_ids, effective_tds, rhs);
        });
        return component_ids;
    }

    std::stable_sort(component_ids.begin(), component_ids.end(), [&config, &effective_tds, &td_component_ids](unsigned int lhs, unsigned int rhs) {
        const float lhs_td = effective_transmission_distance_for_component(td_component_ids, effective_tds, lhs);
        const float rhs_td = effective_transmission_distance_for_component(td_component_ids, effective_tds, rhs);
        const bool lhs_opaque = lhs_td > 0.f && lhs_td < OPAQUE_CONTONING_TD_THRESHOLD_MM;
        const bool rhs_opaque = rhs_td > 0.f && rhs_td < OPAQUE_CONTONING_TD_THRESHOLD_MM;
        if (lhs_opaque != rhs_opaque)
            return lhs_opaque;
        if (lhs_opaque && std::abs(lhs_td - rhs_td) > 1e-6f)
            return lhs_td < rhs_td;
        return filament_luminance(config, lhs) < filament_luminance(config, rhs);
    });
    return component_ids;
}

float texture_mapping_contoning_min_feature_mm(const TextureMappingZone &zone,
                                               const PrintConfig &config,
                                               const std::vector<unsigned int> &component_ids,
                                               float external_width_mm)
{
    const float configured =
        std::clamp(zone.top_surface_contoning_min_feature_mm,
                   TextureMappingZone::MinTopSurfaceContoningMinFeatureMm,
                   TextureMappingZone::MaxTopSurfaceContoningMinFeatureMm);
    if (configured > 0.f)
        return configured;

    float nozzle = 0.4f;
    for (const unsigned int id : component_ids) {
        if (id >= 1 && size_t(id - 1) < config.nozzle_diameter.values.size())
            nozzle = std::max(nozzle, float(config.nozzle_diameter.get_at(size_t(id - 1))));
    }
    if (!config.nozzle_diameter.values.empty())
        nozzle = std::max(nozzle, float(config.nozzle_diameter.values.front()));

    const float width = std::isfinite(external_width_mm) && external_width_mm > 0.f ? external_width_mm : nozzle;
    return std::max({ 2.0f, 4.f * nozzle, 3.f * width });
}

bool texture_mapping_contoning_normal_eligible(float normal_z, float threshold_deg)
{
    if (!std::isfinite(threshold_deg) || threshold_deg >= TextureMappingZone::MaxTopSurfaceContoningAngleThresholdDeg - 1e-4f)
        return true;
    if (!std::isfinite(normal_z))
        return true;
    const float clamped_z = std::clamp(normal_z, -1.f, 1.f);
    const float angle = float(std::acos(clamped_z) * 180.0 / PI);
    return angle <= std::clamp(threshold_deg,
                               TextureMappingZone::MinTopSurfaceContoningAngleThresholdDeg,
                               TextureMappingZone::MaxTopSurfaceContoningAngleThresholdDeg);
}

TextureMappingContoningSolver::TextureMappingContoningSolver(const TextureMappingZone &zone,
                                                             const PrintConfig &config,
                                                             std::vector<unsigned int> component_ids,
                                                             float layer_height_mm)
{
    m_mix_model = color_solver_mix_model_from_index(zone.generic_solver_mix_model);
    m_td_adjustment_enabled = zone.top_surface_contoning_td_adjustment_enabled;
    m_beer_lambert_rgb_correction_enabled =
        m_td_adjustment_enabled && zone.top_surface_contoning_beer_lambert_rgb_correction_enabled;
    m_layer_height_mm = std::isfinite(layer_height_mm) && layer_height_mm > 0.f ? layer_height_mm : 0.2f;
    if (!std::isfinite(m_layer_height_mm) || m_layer_height_mm <= 0.f)
        m_layer_height_mm = 0.2f;
    m_surface_scatter =
        m_td_adjustment_enabled && zone.top_surface_contoning_surface_scatter_enabled ? CONTONING_SURFACE_SCATTER : 0.f;

    component_ids.erase(std::remove_if(component_ids.begin(), component_ids.end(), [&config](unsigned int id) {
        return id == 0 || id > config.filament_colour.values.size();
    }), component_ids.end());
    component_ids.erase(std::unique(component_ids.begin(), component_ids.end()), component_ids.end());
    m_component_ids = component_ids;
    if (!texture_mapping_contoning_component_colors(config, m_component_ids, m_component_colors))
        m_component_ids.clear();
    if (m_component_ids.empty())
        return;

    m_component_luminance.reserve(m_component_ids.size());
    for (const unsigned int id : m_component_ids)
        m_component_luminance.emplace_back(filament_luminance(config, id));
    std::vector<float> background_weights(m_component_colors.size(), 1.f / float(m_component_colors.size()));
    m_background_rgb = mix_color_solver_components(m_component_colors, background_weights, m_mix_model);
    m_effective_transmission_distances_mm =
        effective_transmission_distances_mm(zone, config, m_component_ids, m_td_adjustment_enabled);
    m_component_layer_opacity.reserve(m_effective_transmission_distances_mm.size());
    for (float td_mm : m_effective_transmission_distances_mm)
        m_component_layer_opacity.emplace_back(layer_opacity_from_td(td_mm > 0.f ? td_mm : 3.f, m_layer_height_mm));
    m_components_bottom_to_top = texture_mapping_contoning_components_bottom_to_top(zone, config, m_component_ids);
}

const std::vector<TextureMappingContoningSolver::Candidate>&
TextureMappingContoningSolver::candidates_for_depth(int stack_layers) const
{
    const int depth = std::clamp(stack_layers,
                                 TextureMappingZone::MinTopSurfaceContoningStackLayers,
                                 TextureMappingZone::MaxTopSurfaceContoningStackLayers);
    auto found = m_candidates_by_depth.find(depth);
    if (found != m_candidates_by_depth.end())
        return found->second;

    std::vector<Candidate> candidates;
    if (!valid())
        return m_candidates_by_depth.emplace(depth, std::move(candidates)).first->second;

    std::vector<std::vector<int>> counts_list;
    std::vector<int> counts(m_component_ids.size(), 0);
    enumerate_counts(0, depth, counts, counts_list);
    candidates.reserve(counts_list.size());
    for (std::vector<int> &candidate_counts : counts_list) {
        std::vector<float> weights(candidate_counts.size(), 0.f);
        for (size_t idx = 0; idx < candidate_counts.size(); ++idx)
            weights[idx] = float(candidate_counts[idx]) / float(std::max(1, depth));

        Candidate candidate;
        candidate.counts = std::move(candidate_counts);
        candidate.rgb = mix_color_solver_components(m_component_colors, weights, m_mix_model);
        candidate.oklab = color_solver_oklab_from_srgb(candidate.rgb);
        for (size_t idx = 0; idx < candidate.counts.size() && idx < m_component_luminance.size(); ++idx)
            candidate.dark_score += float(candidate.counts[idx]) * (1.f - clamp01(m_component_luminance[idx]));
        candidates.emplace_back(std::move(candidate));
    }

    return m_candidates_by_depth.emplace(depth, std::move(candidates)).first->second;
}

std::optional<size_t> TextureMappingContoningSolver::component_index(unsigned int component_id) const
{
    const auto it = std::find(m_component_ids.begin(), m_component_ids.end(), component_id);
    if (it == m_component_ids.end())
        return std::nullopt;
    return size_t(it - m_component_ids.begin());
}

std::optional<std::array<float, 3>> TextureMappingContoningSolver::stack_rgb(
    const std::vector<unsigned int> &bottom_to_top,
    bool lower_surface,
    int visible_stack_layers) const
{
    if (bottom_to_top.empty() || !valid())
        return std::nullopt;

    if (!m_td_adjustment_enabled) {
        std::vector<std::array<float, 3>> colors;
        std::vector<float> weights;
        colors.reserve(bottom_to_top.size());
        weights.reserve(bottom_to_top.size());
        const float weight = 1.f / float(bottom_to_top.size());
        for (unsigned int component_id : bottom_to_top) {
            const std::optional<size_t> idx = component_index(component_id);
            if (!idx || *idx >= m_component_colors.size())
                return std::nullopt;
            colors.emplace_back(m_component_colors[*idx]);
            weights.emplace_back(weight);
        }
        return mix_color_solver_components(colors, weights, m_mix_model);
    }

    const int visible_depth = visible_stack_layers > 0 ?
        std::clamp(visible_stack_layers,
                   TextureMappingZone::MinTopSurfaceContoningStackLayers,
                   TextureMappingZone::MaxTopSurfaceContoningStackLayers) :
        int(bottom_to_top.size());
    std::vector<uint16_t> surface_to_deep;
    surface_to_deep.reserve(size_t(visible_depth));
    auto append_component = [this, &surface_to_deep](unsigned int component_id) {
        const std::optional<size_t> idx = component_index(component_id);
        if (!idx || *idx > size_t(std::numeric_limits<uint16_t>::max()))
            return false;
        surface_to_deep.emplace_back(uint16_t(*idx));
        return true;
    };
    if (lower_surface) {
        for (unsigned int component_id : bottom_to_top)
            if (!append_component(component_id))
                return std::nullopt;
    } else {
        for (auto it = bottom_to_top.rbegin(); it != bottom_to_top.rend(); ++it)
            if (!append_component(*it))
                return std::nullopt;
    }
    if (visible_depth > 0 && int(surface_to_deep.size()) != visible_depth) {
        std::vector<uint16_t> repeated;
        repeated.reserve(size_t(visible_depth));
        for (int idx = 0; idx < visible_depth; ++idx)
            repeated.emplace_back(surface_to_deep[size_t(idx) % surface_to_deep.size()]);
        surface_to_deep = std::move(repeated);
    }

    return mix_color_solver_ordered_stack(m_component_colors,
                                          surface_to_deep,
                                          m_component_layer_opacity,
                                          m_background_rgb,
                                          m_mix_model,
                                          m_surface_scatter,
                                          m_beer_lambert_rgb_correction_enabled);
}

void TextureMappingContoningSolver::arrange_stack_for_light_path(std::vector<unsigned int> &bottom_to_top,
                                                                 const std::array<float, 3> &target_rgb) const
{
    if (bottom_to_top.size() < 2)
        return;

    const std::array<float, 3> target_oklab = color_solver_oklab_from_srgb(target_rgb);
    auto component_error = [this, &target_oklab](unsigned int component_id) {
        const auto it = std::find(m_component_ids.begin(), m_component_ids.end(), component_id);
        if (it == m_component_ids.end())
            return std::numeric_limits<float>::max();
        const size_t idx = size_t(it - m_component_ids.begin());
        if (idx >= m_component_colors.size())
            return std::numeric_limits<float>::max();
        return perceptual_error(color_solver_oklab_from_srgb(m_component_colors[idx]), target_oklab);
    };

    std::stable_sort(bottom_to_top.begin(), bottom_to_top.end(), [&component_error](unsigned int lhs, unsigned int rhs) {
        return component_error(lhs) > component_error(rhs);
    });

    std::vector<unsigned int> surface_to_deep;
    surface_to_deep.reserve(bottom_to_top.size());
    for (auto it = bottom_to_top.rbegin(); it != bottom_to_top.rend(); ++it)
        surface_to_deep.emplace_back(*it);

    struct OpaqueItem {
        unsigned int component_id = 0;
        float td_mm = 0.f;
        size_t order = 0;
    };

    std::vector<unsigned int> translucent;
    std::vector<OpaqueItem> opaque;
    translucent.reserve(surface_to_deep.size());
    opaque.reserve(surface_to_deep.size());
    for (size_t idx = 0; idx < surface_to_deep.size(); ++idx) {
        const unsigned int component_id = surface_to_deep[idx];
        const float td_mm =
            effective_transmission_distance_for_component(m_component_ids, m_effective_transmission_distances_mm, component_id);
        if (td_mm > 0.f && td_mm < OPAQUE_CONTONING_TD_THRESHOLD_MM)
            opaque.push_back({ component_id, td_mm, idx });
        else
            translucent.emplace_back(component_id);
    }

    spread_surface_repeats(translucent);
    std::stable_sort(opaque.begin(), opaque.end(), [](const OpaqueItem &lhs, const OpaqueItem &rhs) {
        if (std::abs(lhs.td_mm - rhs.td_mm) > 1e-6f)
            return lhs.td_mm > rhs.td_mm;
        return lhs.order < rhs.order;
    });

    surface_to_deep.clear();
    surface_to_deep.reserve(translucent.size() + opaque.size());
    surface_to_deep.insert(surface_to_deep.end(), translucent.begin(), translucent.end());
    for (const OpaqueItem &item : opaque)
        surface_to_deep.emplace_back(item.component_id);

    bottom_to_top.clear();
    bottom_to_top.reserve(surface_to_deep.size());
    for (auto it = surface_to_deep.rbegin(); it != surface_to_deep.rend(); ++it)
        bottom_to_top.emplace_back(*it);
}

TextureMappingContoningStack TextureMappingContoningSolver::solve(const std::array<float, 3> &target_rgb,
                                                                  int stack_layers,
                                                                  bool lower_surface,
                                                                  int visible_stack_layers) const
{
    TextureMappingContoningStack out;
    if (!valid())
        return out;

    const int requested_depth = std::clamp(stack_layers,
                                           TextureMappingZone::MinTopSurfaceContoningStackLayers,
                                           TextureMappingZone::MaxTopSurfaceContoningStackLayers);
    const int visible_depth = visible_stack_layers > 0 ?
        std::clamp(visible_stack_layers,
                   TextureMappingZone::MinTopSurfaceContoningStackLayers,
                   TextureMappingZone::MaxTopSurfaceContoningStackLayers) :
        requested_depth;
    const int depth = std::min(requested_depth, visible_depth);
    if (m_td_adjustment_enabled) {
        const ColorSolverOrderedStackCandidateSet *ordered_candidates = nullptr;
        {
            std::lock_guard<std::mutex> lock(*m_ordered_candidate_cache_mutex);
            ordered_candidates =
                &color_solver_ordered_stack_candidates(*m_ordered_candidate_cache,
                                                       m_component_colors,
                                                       m_component_layer_opacity,
                                                       m_background_rgb,
                                                       m_mix_model,
                                                       depth,
                                                       visible_depth,
                                                       MAX_TD_ORDERED_CONTONING_CANDIDATES,
                                                       MAX_TD_ORDERED_CONTONING_STACK_ITEMS,
                                                       m_surface_scatter,
                                                       m_beer_lambert_rgb_correction_enabled);
        }
        const std::vector<uint16_t> surface_to_deep =
            solve_color_solver_ordered_stack_for_target(*ordered_candidates, target_rgb, ColorSolverMode::V2);
        if (!surface_to_deep.empty()) {
            out.bottom_to_top.reserve(surface_to_deep.size());
            auto append_component = [this, &out](uint16_t component_idx) {
                if (size_t(component_idx) < m_component_ids.size())
                    out.bottom_to_top.emplace_back(m_component_ids[size_t(component_idx)]);
            };
            if (lower_surface) {
                for (uint16_t component_idx : surface_to_deep)
                    append_component(component_idx);
            } else {
                for (auto it = surface_to_deep.rbegin(); it != surface_to_deep.rend(); ++it)
                    append_component(*it);
            }
            if (!out.bottom_to_top.empty())
                return out;
        }
    }

    const std::vector<Candidate> &candidates = candidates_for_depth(depth);
    if (candidates.empty())
        return out;

    const std::array<float, 3> target_oklab = color_solver_oklab_from_srgb(target_rgb);
    size_t best_idx = size_t(-1);
    float best_error = std::numeric_limits<float>::max();
    float best_dark_score = -std::numeric_limits<float>::max();
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const Candidate &candidate = candidates[idx];
        const float error = perceptual_error(candidate.oklab, target_oklab);
        const bool near_tie = std::isfinite(best_error) && error <= best_error * 1.03f + 1e-6f;
        if (error < best_error || (near_tie && candidate.dark_score > best_dark_score)) {
            best_idx = idx;
            best_error = error;
            best_dark_score = candidate.dark_score;
        }
    }
    if (best_idx >= candidates.size())
        return out;

    const Candidate &best = candidates[best_idx];
    out.bottom_to_top.reserve(size_t(depth));
    for (const unsigned int ordered_id : m_components_bottom_to_top) {
        const auto component_it = std::find(m_component_ids.begin(), m_component_ids.end(), ordered_id);
        if (component_it == m_component_ids.end())
            continue;
        const size_t component_idx = size_t(component_it - m_component_ids.begin());
        const int count = component_idx < best.counts.size() ? best.counts[component_idx] : 0;
        for (int i = 0; i < count; ++i)
            out.bottom_to_top.emplace_back(ordered_id);
    }
    while (int(out.bottom_to_top.size()) < depth)
        out.bottom_to_top.emplace_back(m_components_bottom_to_top.empty() ? m_component_ids.front() : m_components_bottom_to_top.back());
    if (int(out.bottom_to_top.size()) > depth)
        out.bottom_to_top.resize(size_t(depth));
    arrange_stack_for_light_path(out.bottom_to_top, target_rgb);
    if (lower_surface && m_td_adjustment_enabled)
        std::reverse(out.bottom_to_top.begin(), out.bottom_to_top.end());
    return out;
}

unsigned int TextureMappingContoningSolver::component_for_depth(const std::array<float, 3> &target_rgb,
                                                                int stack_layers,
                                                                int depth_from_top,
                                                                bool lower_surface) const
{
    TextureMappingContoningStack stack = solve(target_rgb, stack_layers, lower_surface);
    if (stack.bottom_to_top.empty())
        return 0;
    const int depth = int(stack.bottom_to_top.size());
    const int idx = lower_surface ?
        std::clamp(depth_from_top, 0, depth - 1) :
        std::clamp(depth - 1 - depth_from_top, 0, depth - 1);
    return stack.bottom_to_top[size_t(idx)];
}

}
