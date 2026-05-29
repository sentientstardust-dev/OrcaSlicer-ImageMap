// ColorSolver
// Copyright (C) 2026 sentientstardust

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "ColorSolver.hpp"

#include "pigment_painter_mixer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <utility>

namespace Slic3r {
namespace {

float clamp01(float value)
{
    if (!std::isfinite(value))
        return 0.f;
    return std::clamp(value, 0.f, 1.f);
}

int mix_model_index(ColorSolverMixModel)
{
    return int(ColorSolverMixModel::PigmentPainter);
}

float srgb_to_linear_component(float value)
{
    const float x = clamp01(value);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb_component(float value)
{
    const float x = clamp01(value);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
}

std::array<float, 3> oklab_from_srgb(const std::array<float, 3> &rgb)
{
    const float r = srgb_to_linear_component(rgb[0]);
    const float g = srgb_to_linear_component(rgb[1]);
    const float b = srgb_to_linear_component(rgb[2]);

    const float l = std::cbrt(0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b);
    const float m = std::cbrt(0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b);
    const float s = std::cbrt(0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b);

    return {
        0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
        1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
        0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s
    };
}

std::array<float, 3> srgb_from_oklab(const std::array<float, 3> &oklab)
{
    const float l_ = oklab[0] + 0.3963377774f * oklab[1] + 0.2158037573f * oklab[2];
    const float m_ = oklab[0] - 0.1055613458f * oklab[1] - 0.0638541728f * oklab[2];
    const float s_ = oklab[0] - 0.0894841775f * oklab[1] - 1.2914855480f * oklab[2];

    const float l = l_ * l_ * l_;
    const float m = m_ * m_ * m_;
    const float s = s_ * s_ * s_;

    const float r = 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    const float g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    const float b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;

    return {
        linear_to_srgb_component(r),
        linear_to_srgb_component(g),
        linear_to_srgb_component(b)
    };
}

std::array<float, 3> color_solver_v2_axis_weights(const std::array<float, 3> &target_oklab)
{
    const float chroma = std::hypot(target_oklab[1], target_oklab[2]);
    const float chroma_factor = std::clamp((chroma - 0.015f) / 0.13f, 0.f, 1.f);
    return {
        1.f + (0.25f - 1.f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor
    };
}

int build_color_solver_kd_tree(const std::vector<float>                    &coords,
                               std::vector<ColorSolverCandidateSet::KdNode> &nodes,
                               std::vector<uint32_t>                       &indices,
                               size_t                                       begin,
                               size_t                                       end,
                               uint8_t                                      axis)
{
    if (begin >= end)
        return -1;

    const size_t mid = begin + (end - begin) / 2;
    auto axis_value = [&coords, axis](uint32_t candidate_idx) {
        return coords[size_t(candidate_idx) * 3 + size_t(axis)];
    };
    std::nth_element(indices.begin() + begin,
                     indices.begin() + mid,
                     indices.begin() + end,
                     [&axis_value](uint32_t lhs, uint32_t rhs) {
                         return axis_value(lhs) < axis_value(rhs);
                     });

    const int node_idx = int(nodes.size());
    ColorSolverCandidateSet::KdNode node;
    node.candidate_idx = indices[mid];
    node.axis = axis;
    nodes.emplace_back(node);

    const uint8_t next_axis = uint8_t((axis + 1) % 3);
    const int left = build_color_solver_kd_tree(coords, nodes, indices, begin, mid, next_axis);
    const int right = build_color_solver_kd_tree(coords, nodes, indices, mid + 1, end, next_axis);
    nodes[size_t(node_idx)].left = left;
    nodes[size_t(node_idx)].right = right;
    return node_idx;
}

int build_color_solver_kd_tree(const std::vector<float>                    &coords,
                               std::vector<ColorSolverCandidateSet::KdNode> &nodes)
{
    const size_t candidate_count = coords.size() / 3;
    nodes.clear();
    if (candidate_count == 0)
        return -1;

    std::vector<uint32_t> indices(candidate_count, 0);
    for (size_t idx = 0; idx < candidate_count; ++idx)
        indices[idx] = uint32_t(idx);

    nodes.reserve(candidate_count);
    return build_color_solver_kd_tree(coords, nodes, indices, 0, candidate_count, uint8_t(0));
}

template <class CandidateSet>
void build_color_solver_kd_trees(CandidateSet &candidates)
{
    candidates.kd_root = build_color_solver_kd_tree(candidates.rgbs, candidates.kd_nodes);
    if (candidates.perceptual_coords.size() == candidates.rgbs.size()) {
        candidates.perceptual_kd_root =
            build_color_solver_kd_tree(candidates.perceptual_coords, candidates.perceptual_kd_nodes);
    } else {
        candidates.perceptual_kd_nodes.clear();
        candidates.perceptual_kd_root = -1;
    }
}

struct ColorSolverNearestResult {
    size_t best_idx { size_t(-1) };
    size_t second_idx { size_t(-1) };
    float  best_error { std::numeric_limits<float>::max() };
    float  second_error { std::numeric_limits<float>::max() };
};

void update_color_solver_nearest_result(ColorSolverNearestResult &result, size_t candidate_idx, float error)
{
    if (candidate_idx == result.best_idx || candidate_idx == result.second_idx)
        return;

    if (error < result.best_error) {
        result.second_error = result.best_error;
        result.second_idx = result.best_idx;
        result.best_error = error;
        result.best_idx = candidate_idx;
    } else if (error < result.second_error) {
        result.second_error = error;
        result.second_idx = candidate_idx;
    }
}

template <class CandidateSet>
float color_solver_candidate_error(const CandidateSet             &candidates,
                                   size_t                         candidate_idx,
                                   const std::array<float, 3>    &target_rgb)
{
    const size_t rgb_idx = candidate_idx * 3;
    const float dr = candidates.rgbs[rgb_idx + 0] - target_rgb[0];
    const float dg = candidates.rgbs[rgb_idx + 1] - target_rgb[1];
    const float db = candidates.rgbs[rgb_idx + 2] - target_rgb[2];
    return dr * dr + dg * dg + db * db;
}

template <class CandidateSet>
ColorSolverNearestResult nearest_color_solver_candidates_linear(const CandidateSet             &candidates,
                                                                const std::array<float, 3>    &target_rgb)
{
    ColorSolverNearestResult result;
    const size_t candidate_count = candidates.rgbs.size() / 3;
    for (size_t candidate_idx = 0; candidate_idx < candidate_count; ++candidate_idx) {
        update_color_solver_nearest_result(
            result,
            candidate_idx,
            color_solver_candidate_error(candidates, candidate_idx, target_rgb));
    }
    return result;
}

template <class CandidateSet>
void query_color_solver_kd_tree(const CandidateSet             &candidates,
                                const std::array<float, 3>    &target_rgb,
                                int                            node_idx,
                                ColorSolverNearestResult      &result)
{
    if (node_idx < 0 || size_t(node_idx) >= candidates.kd_nodes.size())
        return;

    const size_t candidate_count = candidates.rgbs.size() / 3;
    const ColorSolverCandidateSet::KdNode &node = candidates.kd_nodes[size_t(node_idx)];
    if (size_t(node.candidate_idx) >= candidate_count) {
        query_color_solver_kd_tree(candidates, target_rgb, node.left, result);
        query_color_solver_kd_tree(candidates, target_rgb, node.right, result);
        return;
    }

    update_color_solver_nearest_result(
        result,
        size_t(node.candidate_idx),
        color_solver_candidate_error(candidates, size_t(node.candidate_idx), target_rgb));

    const size_t rgb_idx = size_t(node.candidate_idx) * 3;
    const size_t axis = std::min<size_t>(node.axis, 2);
    const float split_delta = target_rgb[axis] - candidates.rgbs[rgb_idx + axis];
    const int near_node = split_delta <= 0.f ? node.left : node.right;
    const int far_node = split_delta <= 0.f ? node.right : node.left;

    query_color_solver_kd_tree(candidates, target_rgb, near_node, result);
    if (split_delta * split_delta <= result.second_error)
        query_color_solver_kd_tree(candidates, target_rgb, far_node, result);
}

template <class CandidateSet>
ColorSolverNearestResult nearest_color_solver_candidates(const CandidateSet             &candidates,
                                                         const std::array<float, 3>    &target_rgb)
{
    ColorSolverNearestResult result;
    const size_t candidate_count = candidates.rgbs.size() / 3;
    if (candidates.kd_root >= 0 && !candidates.kd_nodes.empty())
        query_color_solver_kd_tree(candidates, target_rgb, candidates.kd_root, result);
    if (result.best_idx >= candidate_count)
        result = nearest_color_solver_candidates_linear(candidates, target_rgb);
    return result;
}

template <class CandidateSet>
float color_solver_candidate_perceptual_error(const CandidateSet             &candidates,
                                              size_t                         candidate_idx,
                                              const std::array<float, 3>    &target_oklab,
                                              const std::array<float, 3>    &axis_weights)
{
    const size_t coord_idx = candidate_idx * 3;
    const float dl = candidates.perceptual_coords[coord_idx + 0] - target_oklab[0];
    const float da = candidates.perceptual_coords[coord_idx + 1] - target_oklab[1];
    const float db = candidates.perceptual_coords[coord_idx + 2] - target_oklab[2];
    return axis_weights[0] * dl * dl + axis_weights[1] * da * da + axis_weights[2] * db * db;
}

template <class CandidateSet>
ColorSolverNearestResult nearest_color_solver_candidates_perceptual_linear(const CandidateSet             &candidates,
                                                                           const std::array<float, 3>    &target_oklab,
                                                                           const std::array<float, 3>    &axis_weights)
{
    ColorSolverNearestResult result;
    const size_t candidate_count = candidates.perceptual_coords.size() / 3;
    for (size_t candidate_idx = 0; candidate_idx < candidate_count; ++candidate_idx)
        update_color_solver_nearest_result(
            result,
            candidate_idx,
            color_solver_candidate_perceptual_error(candidates, candidate_idx, target_oklab, axis_weights));
    return result;
}

template <class CandidateSet>
void query_color_solver_perceptual_kd_tree(const CandidateSet             &candidates,
                                           const std::array<float, 3>    &target_oklab,
                                           const std::array<float, 3>    &axis_weights,
                                           int                            node_idx,
                                           ColorSolverNearestResult      &result)
{
    if (node_idx < 0 || size_t(node_idx) >= candidates.perceptual_kd_nodes.size())
        return;

    const size_t candidate_count = candidates.perceptual_coords.size() / 3;
    const ColorSolverCandidateSet::KdNode &node = candidates.perceptual_kd_nodes[size_t(node_idx)];
    if (size_t(node.candidate_idx) >= candidate_count) {
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, node.left, result);
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, node.right, result);
        return;
    }

    update_color_solver_nearest_result(
        result,
        size_t(node.candidate_idx),
        color_solver_candidate_perceptual_error(candidates, size_t(node.candidate_idx), target_oklab, axis_weights));

    const size_t coord_idx = size_t(node.candidate_idx) * 3;
    const size_t axis = std::min<size_t>(node.axis, 2);
    const float split_delta = target_oklab[axis] - candidates.perceptual_coords[coord_idx + axis];
    const int near_node = split_delta <= 0.f ? node.left : node.right;
    const int far_node = split_delta <= 0.f ? node.right : node.left;

    query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, near_node, result);
    if (axis_weights[axis] * split_delta * split_delta <= result.second_error)
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, far_node, result);
}

template <class CandidateSet>
ColorSolverNearestResult nearest_color_solver_candidates_perceptual(const CandidateSet             &candidates,
                                                                    const std::array<float, 3>    &target_rgb)
{
    ColorSolverNearestResult result;
    const size_t candidate_count = candidates.perceptual_coords.size() / 3;
    if (candidate_count == 0 || candidates.perceptual_coords.size() != candidates.rgbs.size())
        return result;

    const std::array<float, 3> target_oklab = oklab_from_srgb(target_rgb);
    const std::array<float, 3> axis_weights = color_solver_v2_axis_weights(target_oklab);
    if (candidates.perceptual_kd_root >= 0 && !candidates.perceptual_kd_nodes.empty())
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, candidates.perceptual_kd_root, result);
    if (result.best_idx >= candidate_count)
        result = nearest_color_solver_candidates_perceptual_linear(candidates, target_oklab, axis_weights);
    return result;
}

std::array<float, 3> mix_ordered_stack_with_buffers(const std::vector<std::array<float, 3>> &colors_with_background,
                                                    std::vector<float>                      &weights,
                                                    const std::vector<uint16_t>             &surface_to_deep,
                                                    const std::vector<float>                &layer_opacities)
{
    if (colors_with_background.empty() || surface_to_deep.empty())
        return colors_with_background.empty() ? std::array<float, 3>{ { 0.f, 0.f, 0.f } } : colors_with_background.back();

    const size_t component_count = colors_with_background.size() - 1;
    weights.assign(colors_with_background.size(), 0.f);
    float transmission = 1.f;
    for (uint16_t component_idx : surface_to_deep) {
        if (size_t(component_idx) >= component_count)
            continue;
        const float opacity =
            size_t(component_idx) < layer_opacities.size() && std::isfinite(layer_opacities[size_t(component_idx)]) ?
                std::clamp(layer_opacities[size_t(component_idx)], 1e-4f, 0.9999f) :
                0.5f;
        weights[size_t(component_idx)] += transmission * opacity;
        transmission *= 1.f - opacity;
        if (transmission <= 1e-5f)
            break;
    }
    weights.back() = std::max(0.f, transmission);
    return pigment_painter::mix_srgb(colors_with_background, weights);
}

size_t ordered_stack_candidate_count(size_t component_count, int stack_depth, size_t candidate_limit)
{
    if (component_count == 0 || stack_depth <= 0)
        return 0;
    size_t count = 1;
    for (int idx = 0; idx < stack_depth; ++idx) {
        if (candidate_limit > 0 && count > candidate_limit / component_count)
            return 0;
        if (count > std::numeric_limits<size_t>::max() / component_count)
            return 0;
        count *= component_count;
    }
    return candidate_limit > 0 && count > candidate_limit ? 0 : count;
}

size_t ordered_stack_candidate_storage_limit(int stack_depth, size_t candidate_limit, size_t stack_item_limit)
{
    size_t limit = candidate_limit > 0 ? candidate_limit : std::numeric_limits<size_t>::max();
    if (stack_item_limit > 0 && stack_depth > 0)
        limit = std::min(limit, stack_item_limit / size_t(stack_depth));
    return limit;
}

int ordered_stack_control_depth(size_t component_count, int stack_depth, size_t candidate_limit)
{
    if (component_count == 0 || stack_depth <= 0 || candidate_limit == 0)
        return 0;
    size_t count = 1;
    int depth = 0;
    for (int idx = 0; idx < stack_depth; ++idx) {
        if (count > candidate_limit / component_count)
            break;
        count *= component_count;
        ++depth;
    }
    return depth > 0 ? depth : 1;
}

std::vector<uint16_t> repeat_ordered_stack_to_depth(const std::vector<uint16_t> &surface_to_deep, int depth)
{
    std::vector<uint16_t> out;
    if (surface_to_deep.empty() || depth <= 0)
        return out;
    out.reserve(size_t(depth));
    for (int idx = 0; idx < depth; ++idx)
        out.emplace_back(surface_to_deep[size_t(idx) % surface_to_deep.size()]);
    return out;
}

void append_unique_ordered_stack_variant(std::vector<std::vector<uint16_t>> &variants, std::vector<uint16_t> candidate)
{
    if (candidate.empty())
        return;
    if (std::find(variants.begin(), variants.end(), candidate) == variants.end())
        variants.emplace_back(std::move(candidate));
}

std::vector<uint16_t> stretch_ordered_stack_tiled(const std::vector<uint16_t> &control, int stack_depth)
{
    return repeat_ordered_stack_to_depth(control, stack_depth);
}

std::vector<uint16_t> stretch_ordered_stack_proportional(const std::vector<uint16_t> &control, int stack_depth)
{
    std::vector<uint16_t> out;
    if (control.empty() || stack_depth <= 0)
        return out;
    out.reserve(size_t(stack_depth));
    const size_t control_depth = control.size();
    for (int idx = 0; idx < stack_depth; ++idx) {
        const size_t source_idx = std::min(control_depth - 1, size_t(idx) * control_depth / size_t(stack_depth));
        out.emplace_back(control[source_idx]);
    }
    return out;
}

std::vector<uint16_t> stretch_ordered_stack_by_scores(const std::vector<uint16_t> &control,
                                                      const std::vector<float>    &scores,
                                                      int                          stack_depth)
{
    std::vector<uint16_t> out;
    if (control.empty() || stack_depth <= 0)
        return out;
    if (int(control.size()) >= stack_depth) {
        out.assign(control.begin(), control.begin() + std::min(control.size(), size_t(stack_depth)));
        return out;
    }

    std::vector<int> duplicates(control.size(), 0);
    const int extra = stack_depth - int(control.size());
    for (int idx = 0; idx < extra; ++idx) {
        size_t best_idx = 0;
        float best_score = -std::numeric_limits<float>::max();
        for (size_t score_idx = 0; score_idx < control.size(); ++score_idx) {
            const float base_score =
                score_idx < scores.size() && std::isfinite(scores[score_idx]) ?
                    std::max(scores[score_idx], 0.f) :
                    0.f;
            const float score = base_score / float(duplicates[score_idx] + 1);
            if (score > best_score) {
                best_score = score;
                best_idx = score_idx;
            }
        }
        ++duplicates[best_idx];
    }

    out.reserve(size_t(stack_depth));
    for (size_t idx = 0; idx < control.size(); ++idx) {
        out.emplace_back(control[idx]);
        for (int duplicate_idx = 0; duplicate_idx < duplicates[idx]; ++duplicate_idx)
            out.emplace_back(control[idx]);
    }
    return out;
}

std::vector<std::vector<uint16_t>> stretched_ordered_stack_variants(const std::vector<uint16_t> &control,
                                                                    const std::vector<float>    &layer_opacities,
                                                                    int                          stack_depth)
{
    std::vector<std::vector<uint16_t>> variants;
    if (control.empty() || stack_depth <= 0)
        return variants;
    if (int(control.size()) >= stack_depth) {
        std::vector<uint16_t> variant(control.begin(), control.begin() + std::min(control.size(), size_t(stack_depth)));
        append_unique_ordered_stack_variant(variants, std::move(variant));
        return variants;
    }

    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_tiled(control, stack_depth));
    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_proportional(control, stack_depth));

    std::vector<float> even(control.size(), 1.f);
    std::vector<float> opacity(control.size(), 1.f);
    std::vector<float> surface(control.size(), 1.f);
    std::vector<float> deep(control.size(), 1.f);
    for (size_t idx = 0; idx < control.size(); ++idx) {
        const size_t component_idx = size_t(control[idx]);
        const float layer_opacity =
            component_idx < layer_opacities.size() && std::isfinite(layer_opacities[component_idx]) ?
                std::clamp(layer_opacities[component_idx], 1e-4f, 0.9999f) :
                0.5f;
        opacity[idx] = 1.f - layer_opacity;
        surface[idx] = float(control.size() - idx);
        deep[idx] = float(idx + 1);
    }

    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_by_scores(control, even, stack_depth));
    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_by_scores(control, opacity, stack_depth));
    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_by_scores(control, surface, stack_depth));
    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_by_scores(control, deep, stack_depth));
    for (size_t idx = 0; idx < control.size(); ++idx) {
        surface[idx] *= opacity[idx];
        deep[idx] *= opacity[idx];
    }
    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_by_scores(control, surface, stack_depth));
    append_unique_ordered_stack_variant(variants, stretch_ordered_stack_by_scores(control, deep, stack_depth));
    return variants;
}

void append_ordered_stack_candidate(ColorSolverOrderedStackCandidateSet       &candidates,
                                    const std::vector<std::array<float, 3>>  &colors_with_background,
                                    std::vector<float>                       &weights,
                                    const std::vector<uint16_t>              &surface_to_deep,
                                    const std::vector<float>                 &layer_opacities,
                                    int                                       simulated_stack_depth)
{
    std::vector<uint16_t> simulated_surface_to_deep;
    const std::vector<uint16_t> *mix_stack = &surface_to_deep;
    if (simulated_stack_depth > 0 && simulated_stack_depth != int(surface_to_deep.size())) {
        simulated_surface_to_deep = repeat_ordered_stack_to_depth(surface_to_deep, simulated_stack_depth);
        mix_stack = &simulated_surface_to_deep;
    }
    const std::array<float, 3> mixed =
        mix_ordered_stack_with_buffers(colors_with_background, weights, *mix_stack, layer_opacities);
    const std::array<float, 3> perceptual = oklab_from_srgb(mixed);
    candidates.rgbs.emplace_back(mixed[0]);
    candidates.rgbs.emplace_back(mixed[1]);
    candidates.rgbs.emplace_back(mixed[2]);
    candidates.perceptual_coords.emplace_back(perceptual[0]);
    candidates.perceptual_coords.emplace_back(perceptual[1]);
    candidates.perceptual_coords.emplace_back(perceptual[2]);
    candidates.stacks.insert(candidates.stacks.end(), surface_to_deep.begin(), surface_to_deep.end());
}

} // namespace

ColorSolverMixModel color_solver_mix_model_from_index(int model)
{
    (void) model;
    return ColorSolverMixModel::PigmentPainter;
}

ColorSolverLookupMode color_solver_lookup_mode_from_index(int mode)
{
    return ColorSolverLookupMode(std::clamp(mode, int(ColorSolverLookupMode::ClosestMix), int(ColorSolverLookupMode::BlendClosestTwo)));
}

ColorSolverMode color_solver_mode_from_index(int mode)
{
    return ColorSolverMode(std::clamp(mode, int(ColorSolverMode::Legacy), int(ColorSolverMode::V2)));
}

int color_solver_total_units_for_component_count(size_t component_count)
{
    return component_count <= 4 ? 40 : (component_count == 5 ? 24 : (component_count == 6 ? 20 : 12));
}

size_t color_solver_candidate_count(size_t component_count, int total_units)
{
    if (component_count == 0)
        return 0;

    const size_t n = size_t(total_units) + component_count - 1;
    size_t k = component_count - 1;
    k = std::min(k, n - k);

    size_t result = 1;
    for (size_t idx = 1; idx <= k; ++idx)
        result = (result * (n - k + idx)) / idx;
    return result;
}

std::array<float, 3> mix_color_solver_components(const std::vector<std::array<float, 3>> &component_colors,
                                                 const std::vector<int>                  &weights,
                                                 ColorSolverMixModel                       mix_model)
{
    (void) mix_model;
    return pigment_painter::mix_srgb(component_colors, weights);
}

std::array<float, 3> mix_color_solver_components(const std::vector<std::array<float, 3>> &component_colors,
                                                 const std::vector<float>                &weights,
                                                 ColorSolverMixModel                       mix_model)
{
    (void) mix_model;
    std::vector<float> safe_weights;
    safe_weights.reserve(weights.size());
    for (const float weight : weights)
        safe_weights.emplace_back(std::isfinite(weight) ? std::clamp(weight, 0.f, 1.f) : 0.f);
    return pigment_painter::mix_srgb(component_colors, safe_weights);
}

std::array<float, 3> mix_color_solver_ordered_stack(const std::vector<std::array<float, 3>> &component_colors,
                                                    const std::vector<uint16_t>             &surface_to_deep,
                                                    const std::vector<float>                &layer_opacities,
                                                    const std::array<float, 3>              &background_rgb,
                                                    ColorSolverMixModel                       mix_model)
{
    (void) mix_model;
    if (component_colors.empty() || surface_to_deep.empty())
        return background_rgb;

    std::vector<std::array<float, 3>> colors = component_colors;
    colors.emplace_back(background_rgb);
    std::vector<float> weights;
    return mix_ordered_stack_with_buffers(colors, weights, surface_to_deep, layer_opacities);
}

std::array<float, 3> color_solver_oklab_from_srgb(const std::array<float, 3> &rgb)
{
    return oklab_from_srgb(rgb);
}

std::array<float, 3> color_solver_srgb_from_oklab(const std::array<float, 3> &oklab)
{
    return srgb_from_oklab(oklab);
}

std::string color_solver_candidate_cache_key(const std::vector<std::array<float, 3>> &component_colors,
                                             ColorSolverMixModel                       mix_model,
                                             int                                       total_units)
{
    const int default_total_units = color_solver_total_units_for_component_count(component_colors.size());
    std::ostringstream key;
    key << component_colors.size();
    key << "|mx" << mix_model_index(mix_model);
    if (total_units > 0 && total_units != default_total_units)
        key << "|tu" << total_units;
    for (const std::array<float, 3> &color : component_colors) {
        key << '|'
            << int(std::lround(clamp01(color[0]) * 65535.f)) << ','
            << int(std::lround(clamp01(color[1]) * 65535.f)) << ','
            << int(std::lround(clamp01(color[2]) * 65535.f));
    }
    return key.str();
}

ColorSolverCandidateSet build_color_solver_candidates(const std::vector<std::array<float, 3>> &component_colors,
                                                      ColorSolverMixModel                       mix_model,
                                                      int                                       total_units)
{
    ColorSolverCandidateSet candidates;
    if (component_colors.empty())
        return candidates;

    const size_t component_count = component_colors.size();
    if (total_units <= 0)
        total_units = color_solver_total_units_for_component_count(component_count);
    std::vector<int> units(component_count, 0);
    const size_t estimated_candidate_count = color_solver_candidate_count(component_count, total_units);
    candidates.component_count = component_count;
    candidates.rgbs.reserve(estimated_candidate_count * 3);
    candidates.perceptual_coords.reserve(estimated_candidate_count * 3);
    candidates.weights.reserve(estimated_candidate_count * component_count);

    std::function<void(size_t, int)> recurse = [&](size_t idx, int remaining_units) {
        if (idx + 1 == component_count) {
            units[idx] = remaining_units;
            const std::array<float, 3> mixed = mix_color_solver_components(component_colors, units, mix_model);
            const std::array<float, 3> perceptual = oklab_from_srgb(mixed);
            candidates.rgbs.emplace_back(mixed[0]);
            candidates.rgbs.emplace_back(mixed[1]);
            candidates.rgbs.emplace_back(mixed[2]);
            candidates.perceptual_coords.emplace_back(perceptual[0]);
            candidates.perceptual_coords.emplace_back(perceptual[1]);
            candidates.perceptual_coords.emplace_back(perceptual[2]);
            for (size_t weight_idx = 0; weight_idx < component_count; ++weight_idx)
                candidates.weights.emplace_back(float(units[weight_idx]) / float(std::max(1, total_units)));
            return;
        }

        for (int unit = 0; unit <= remaining_units; ++unit) {
            units[idx] = unit;
            recurse(idx + 1, remaining_units - unit);
        }
    };
    recurse(0, total_units);
    build_color_solver_kd_trees(candidates);
    return candidates;
}

const ColorSolverCandidateSet &color_solver_candidates(ColorSolverCandidateCache                &cache,
                                                       const std::vector<std::array<float, 3>> &component_colors,
                                                       ColorSolverMixModel                       mix_model,
                                                       int                                       total_units)
{
    const std::string key = color_solver_candidate_cache_key(component_colors, mix_model, total_units);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;
    return cache.emplace(key, build_color_solver_candidates(component_colors, mix_model, total_units)).first->second;
}

std::vector<float> solve_color_solver_weights_for_target(const ColorSolverCandidateSet &candidates,
                                                         const std::array<float, 3>     &target_rgb,
                                                         ColorSolverLookupMode           lookup_mode,
                                                         ColorSolverMode                 solver_mode)
{
    if (candidates.empty())
        return {};

    const size_t candidate_count = candidates.rgbs.size() / 3;
    ColorSolverNearestResult nearest =
        solver_mode == ColorSolverMode::V2 ?
            nearest_color_solver_candidates_perceptual(candidates, target_rgb) :
            nearest_color_solver_candidates(candidates, target_rgb);
    if (nearest.best_idx >= candidate_count && solver_mode == ColorSolverMode::V2)
        nearest = nearest_color_solver_candidates(candidates, target_rgb);
    if (nearest.best_idx >= candidate_count)
        return {};

    std::vector<float> weights(candidates.component_count, 0.f);
    const size_t best_weight_idx = nearest.best_idx * candidates.component_count;
    if (lookup_mode == ColorSolverLookupMode::ClosestMix ||
        nearest.second_idx >= candidate_count ||
        nearest.best_error <= 1e-12f) {
        for (size_t idx = 0; idx < candidates.component_count; ++idx)
            weights[idx] = candidates.weights[best_weight_idx + idx];
        return weights;
    }

    const size_t second_weight_idx = nearest.second_idx * candidates.component_count;
    const float best_inv = 1.f / std::max(nearest.best_error, 1e-12f);
    const float second_inv = 1.f / std::max(nearest.second_error, 1e-12f);
    const float inv_sum = std::max(best_inv + second_inv, 1e-12f);
    for (size_t idx = 0; idx < candidates.component_count; ++idx)
        weights[idx] = clamp01((candidates.weights[best_weight_idx + idx] * best_inv +
                                candidates.weights[second_weight_idx + idx] * second_inv) / inv_sum);
    return weights;
}

std::string color_solver_ordered_stack_candidate_cache_key(const std::vector<std::array<float, 3>> &component_colors,
                                                           const std::vector<float>                &layer_opacities,
                                                           const std::array<float, 3>              &background_rgb,
                                                           ColorSolverMixModel                       mix_model,
                                                           int                                       stack_depth,
                                                           int                                       simulated_stack_depth,
                                                           size_t                                    candidate_limit,
                                                           size_t                                    stack_item_limit)
{
    std::ostringstream key;
    key << component_colors.size();
    key << "|mx" << mix_model_index(mix_model);
    key << "|sd" << stack_depth;
    key << "|vd" << simulated_stack_depth;
    key << "|lim" << candidate_limit;
    key << "|item" << stack_item_limit;
    key << "|bg"
        << int(std::lround(clamp01(background_rgb[0]) * 65535.f)) << ','
        << int(std::lround(clamp01(background_rgb[1]) * 65535.f)) << ','
        << int(std::lround(clamp01(background_rgb[2]) * 65535.f));
    for (const std::array<float, 3> &color : component_colors) {
        key << '|'
            << int(std::lround(clamp01(color[0]) * 65535.f)) << ','
            << int(std::lround(clamp01(color[1]) * 65535.f)) << ','
            << int(std::lround(clamp01(color[2]) * 65535.f));
    }
    key << "|op";
    for (size_t idx = 0; idx < component_colors.size(); ++idx) {
        const float opacity =
            idx < layer_opacities.size() && std::isfinite(layer_opacities[idx]) ?
                std::clamp(layer_opacities[idx], 1e-4f, 0.9999f) :
                0.5f;
        key << ',' << int(std::lround(opacity * 1000000.f));
    }
    return key.str();
}

ColorSolverOrderedStackCandidateSet build_color_solver_ordered_stack_candidates(
    const std::vector<std::array<float, 3>> &component_colors,
    const std::vector<float>                &layer_opacities,
    const std::array<float, 3>              &background_rgb,
    ColorSolverMixModel                       mix_model,
    int                                       stack_depth,
    int                                       simulated_stack_depth,
    size_t                                    candidate_limit,
    size_t                                    stack_item_limit)
{
    (void) mix_model;
    ColorSolverOrderedStackCandidateSet candidates;
    int simulated_depth = simulated_stack_depth > 0 ? simulated_stack_depth : stack_depth;
    if (component_colors.empty() || stack_depth <= 0 || simulated_depth <= 0 ||
        component_colors.size() > size_t(std::numeric_limits<uint16_t>::max()))
        return candidates;

    const size_t component_count = component_colors.size();
    stack_depth = std::min(stack_depth, simulated_depth);
    const size_t max_candidate_count = ordered_stack_candidate_storage_limit(stack_depth, candidate_limit, stack_item_limit);
    if (max_candidate_count == 0)
        return candidates;

    candidates.component_count = component_count;
    candidates.stack_depth = stack_depth;
    candidates.simulated_stack_depth = simulated_depth;

    std::vector<std::array<float, 3>> colors_with_background = component_colors;
    colors_with_background.emplace_back(background_rgb);
    std::vector<float> weights(colors_with_background.size(), 0.f);

    const size_t exact_candidate_count = ordered_stack_candidate_count(component_count, stack_depth, candidate_limit);
    if (exact_candidate_count > 0 && exact_candidate_count <= max_candidate_count) {
        candidates.rgbs.reserve(exact_candidate_count * 3);
        candidates.perceptual_coords.reserve(exact_candidate_count * 3);
        candidates.stacks.reserve(exact_candidate_count * size_t(stack_depth));

        std::vector<uint16_t> surface_to_deep(size_t(stack_depth), 0);
        std::function<void(int)> recurse = [&](int depth_idx) {
            if (depth_idx == stack_depth) {
                append_ordered_stack_candidate(candidates,
                                               colors_with_background,
                                               weights,
                                               surface_to_deep,
                                               layer_opacities,
                                               simulated_depth);
                return;
            }

            for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
                surface_to_deep[size_t(depth_idx)] = uint16_t(component_idx);
                recurse(depth_idx + 1);
            }
        };
        recurse(0);
        build_color_solver_kd_trees(candidates);
        return candidates;
    }
    if (candidate_limit == 0 && stack_item_limit == 0)
        return candidates;

    const size_t variant_budget = 8;
    const size_t max_control_candidates = std::max<size_t>(1, max_candidate_count / variant_budget);
    const int control_depth = std::min(stack_depth, ordered_stack_control_depth(component_count, stack_depth, max_control_candidates));
    if (control_depth <= 0)
        return candidates;

    const size_t control_candidate_count = ordered_stack_candidate_count(component_count, control_depth, 0);
    const size_t reserve_count = std::min(max_candidate_count, control_candidate_count * variant_budget);
    candidates.rgbs.reserve(reserve_count * 3);
    candidates.perceptual_coords.reserve(reserve_count * 3);
    candidates.stacks.reserve(reserve_count * size_t(stack_depth));

    bool limit_reached = false;
    std::vector<uint16_t> control(size_t(control_depth), 0);
    std::function<void(int)> recurse = [&](int depth_idx) {
        if (limit_reached)
            return;
        if (depth_idx == control_depth) {
            std::vector<std::vector<uint16_t>> variants =
                stretched_ordered_stack_variants(control, layer_opacities, stack_depth);
            for (const std::vector<uint16_t> &surface_to_deep : variants) {
                if (candidates.rgbs.size() / 3 >= max_candidate_count) {
                    limit_reached = true;
                    return;
                }
                append_ordered_stack_candidate(candidates,
                                               colors_with_background,
                                               weights,
                                               surface_to_deep,
                                               layer_opacities,
                                               simulated_depth);
            }
            return;
        }

        for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
            control[size_t(depth_idx)] = uint16_t(component_idx);
            recurse(depth_idx + 1);
            if (limit_reached)
                return;
        }
    };
    recurse(0);
    build_color_solver_kd_trees(candidates);
    return candidates;
}

const ColorSolverOrderedStackCandidateSet &color_solver_ordered_stack_candidates(
    ColorSolverOrderedStackCandidateCache        &cache,
    const std::vector<std::array<float, 3>>      &component_colors,
    const std::vector<float>                     &layer_opacities,
    const std::array<float, 3>                   &background_rgb,
    ColorSolverMixModel                            mix_model,
    int                                            stack_depth,
    int                                            simulated_stack_depth,
    size_t                                         candidate_limit,
    size_t                                         stack_item_limit)
{
    const std::string key =
        color_solver_ordered_stack_candidate_cache_key(component_colors,
                                                       layer_opacities,
                                                       background_rgb,
                                                       mix_model,
                                                       stack_depth,
                                                       simulated_stack_depth,
                                                       candidate_limit,
                                                       stack_item_limit);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;
    return cache.emplace(
        key,
        build_color_solver_ordered_stack_candidates(component_colors,
                                                   layer_opacities,
                                                   background_rgb,
                                                   mix_model,
                                                   stack_depth,
                                                   simulated_stack_depth,
                                                   candidate_limit,
                                                   stack_item_limit)).first->second;
}

std::vector<uint16_t> solve_color_solver_ordered_stack_for_target(
    const ColorSolverOrderedStackCandidateSet &candidates,
    const std::array<float, 3>                &target_rgb,
    ColorSolverMode                            solver_mode)
{
    if (candidates.empty())
        return {};

    const size_t candidate_count = candidates.rgbs.size() / 3;
    ColorSolverNearestResult nearest =
        solver_mode == ColorSolverMode::V2 ?
            nearest_color_solver_candidates_perceptual(candidates, target_rgb) :
            nearest_color_solver_candidates(candidates, target_rgb);
    if (nearest.best_idx >= candidate_count && solver_mode == ColorSolverMode::V2)
        nearest = nearest_color_solver_candidates(candidates, target_rgb);
    if (nearest.best_idx >= candidate_count)
        return {};

    const size_t stack_begin = nearest.best_idx * size_t(candidates.stack_depth);
    if (stack_begin + size_t(candidates.stack_depth) > candidates.stacks.size())
        return {};
    return std::vector<uint16_t>(candidates.stacks.begin() + stack_begin,
                                 candidates.stacks.begin() + stack_begin + candidates.stack_depth);
}

} // namespace Slic3r
