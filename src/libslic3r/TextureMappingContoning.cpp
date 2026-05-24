// original author: sentientstardust

#include "TextureMappingContoning.hpp"

#include "Color.hpp"
#include "ColorSolver.hpp"
#include "PrintConfig.hpp"
#include "libslic3r.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace Slic3r {
namespace {

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

    bool has_complete_td = true;
    for (unsigned int component_id : component_ids) {
        const size_t idx = size_t(component_id - 1);
        const float td = idx < zone.filament_transmission_distances_mm.size() ?
            zone.filament_transmission_distances_mm[idx] :
            0.f;
        if (!std::isfinite(td) || td <= 0.f) {
            has_complete_td = false;
            break;
        }
    }
    if (has_complete_td) {
        std::stable_sort(component_ids.begin(), component_ids.end(), [&zone](unsigned int lhs, unsigned int rhs) {
            return zone.filament_transmission_distances_mm[size_t(lhs - 1)] <
                   zone.filament_transmission_distances_mm[size_t(rhs - 1)];
        });
        return component_ids;
    }

    std::stable_sort(component_ids.begin(), component_ids.end(), [&config](unsigned int lhs, unsigned int rhs) {
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
                                                             std::vector<unsigned int> component_ids)
{
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
        candidate.rgb = mix_color_solver_components(m_component_colors, weights, ColorSolverMixModel::PigmentPainter);
        candidate.oklab = color_solver_oklab_from_srgb(candidate.rgb);
        for (size_t idx = 0; idx < candidate.counts.size() && idx < m_component_luminance.size(); ++idx)
            candidate.dark_score += float(candidate.counts[idx]) * (1.f - clamp01(m_component_luminance[idx]));
        candidates.emplace_back(std::move(candidate));
    }

    return m_candidates_by_depth.emplace(depth, std::move(candidates)).first->second;
}

TextureMappingContoningStack TextureMappingContoningSolver::solve(const std::array<float, 3> &target_rgb, int stack_layers) const
{
    TextureMappingContoningStack out;
    if (!valid())
        return out;

    const int depth = std::clamp(stack_layers,
                                 TextureMappingZone::MinTopSurfaceContoningStackLayers,
                                 TextureMappingZone::MaxTopSurfaceContoningStackLayers);
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
    return out;
}

unsigned int TextureMappingContoningSolver::component_for_depth(const std::array<float, 3> &target_rgb,
                                                                int stack_layers,
                                                                int depth_from_top) const
{
    TextureMappingContoningStack stack = solve(target_rgb, stack_layers);
    if (stack.bottom_to_top.empty())
        return 0;
    const int depth = int(stack.bottom_to_top.size());
    const int idx = std::clamp(depth - 1 - depth_from_top, 0, depth - 1);
    return stack.bottom_to_top[size_t(idx)];
}

}
