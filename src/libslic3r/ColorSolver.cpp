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
#include "prusa_fdm_mixer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <sstream>
#include <utility>

namespace Slic3r {
namespace {

constexpr float TD_EFFECTIVE_ALPHA_DENSITY_SCALE = 0.6761904762f;
constexpr float TD_EFFECTIVE_ALPHA_PASS_DENSITY = 0.05f;
constexpr float TD_EFFECTIVE_ALPHA_WHITE_DENSITY = 0.35f;
constexpr float OKLAB_SOFT_CAP4_DARK4_MIN_L_WEIGHT = 1.f;
constexpr float OKLAB_SOFT_CAP4_DARK4_MAX_AB_WEIGHT = 4.f;
constexpr float OKLAB_SOFT_CAP4_DARK4_DARK_PENALTY = 4.f;
constexpr float OKLAB_SOFT_CAP4_DARK4_DARK_TOLERANCE = 0.04f;

float clamp01(float value)
{
    if (!std::isfinite(value))
        return 0.f;
    return std::clamp(value, 0.f, 1.f);
}

int mix_model_index(ColorSolverMixModel mix_model)
{
    return std::clamp(int(mix_model), int(ColorSolverMixModel::PigmentPainter), int(ColorSolverMixModel::PrusaFdmMixer));
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

std::array<float, 3> srgb_to_linear_color(const std::array<float, 3> &rgb)
{
    return {
        srgb_to_linear_component(rgb[0]),
        srgb_to_linear_component(rgb[1]),
        srgb_to_linear_component(rgb[2])
    };
}

std::array<float, 3> linear_to_srgb_color(const std::array<float, 3> &rgb)
{
    return {
        linear_to_srgb_component(rgb[0]),
        linear_to_srgb_component(rgb[1]),
        linear_to_srgb_component(rgb[2])
    };
}

float linear_luminance(const std::array<float, 3> &rgb)
{
    return 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
}

std::string srgb_color_to_hex(const std::array<float, 3> &rgb)
{
    const int r = std::clamp(int(std::lround(clamp01(rgb[0]) * 255.f)), 0, 255);
    const int g = std::clamp(int(std::lround(clamp01(rgb[1]) * 255.f)), 0, 255);
    const int b = std::clamp(int(std::lround(clamp01(rgb[2]) * 255.f)), 0, 255);
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", unsigned(r), unsigned(g), unsigned(b));
    return std::string(buf);
}

std::array<float, 3> mix_prusa_fdm_components(const std::vector<std::array<float, 3>> &colors,
                                              const std::vector<float>                &weights)
{
    const size_t count = std::min(colors.size(), weights.size());
    if (count == 0)
        return { 0.f, 0.f, 0.f };

    float total_weight = 0.f;
    for (size_t idx = 0; idx < count; ++idx)
        if (std::isfinite(weights[idx]) && weights[idx] > 0.f)
            total_weight += weights[idx];
    if (total_weight <= 0.f)
        return {
            clamp01(colors.front()[0]),
            clamp01(colors.front()[1]),
            clamp01(colors.front()[2])
        };

    std::vector<prusa_fdm_mixer::Part> parts;
    parts.reserve(count);
    const double inv_total_weight = 1.0 / double(total_weight);
    for (size_t idx = 0; idx < count; ++idx) {
        const float weight = weights[idx];
        if (!std::isfinite(weight) || weight <= 0.f)
            continue;
        parts.push_back({ srgb_color_to_hex(colors[idx]), double(weight) * inv_total_weight });
    }
    if (parts.empty())
        return {
            clamp01(colors.front()[0]),
            clamp01(colors.front()[1]),
            clamp01(colors.front()[2])
        };

    const prusa_fdm_mixer::RGB rgb = prusa_fdm_mixer::mix_rgb(parts);
    return {
        float(rgb.r) / 255.f,
        float(rgb.g) / 255.f,
        float(rgb.b) / 255.f
    };
}

std::array<float, 3> mix_components_by_model(const std::vector<std::array<float, 3>> &colors,
                                             const std::vector<float>                &weights,
                                             ColorSolverMixModel                       mix_model)
{
    if (colors.empty() || colors.size() != weights.size())
        return { 0.f, 0.f, 0.f };
    return mix_model == ColorSolverMixModel::PrusaFdmMixer ?
        mix_prusa_fdm_components(colors, weights) :
        pigment_painter::mix_srgb(colors, weights);
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

std::array<float, 3> color_solver_oklab_axis_weights(const std::array<float, 3> &target_oklab)
{
    const float chroma = std::hypot(target_oklab[1], target_oklab[2]);
    const float chroma_factor = std::clamp((chroma - 0.015f) / 0.13f, 0.f, 1.f);
    return {
        1.f + (0.25f - 1.f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor,
        1.25f + (8.f - 1.25f) * chroma_factor
    };
}

float color_solver_oklab_chroma_factor(const std::array<float, 3> &target_oklab)
{
    const float chroma = std::hypot(target_oklab[1], target_oklab[2]);
    return std::clamp((chroma - 0.015f) / 0.13f, 0.f, 1.f);
}

std::array<float, 3> color_solver_perceptual_axis_weights(const std::array<float, 3> &target_oklab,
                                                          ColorSolverMode             solver_mode)
{
    std::array<float, 3> weights = color_solver_oklab_axis_weights(target_oklab);
    if (solver_mode == ColorSolverMode::OklabSoftCap4Dark4) {
        weights[0] = std::max(weights[0], OKLAB_SOFT_CAP4_DARK4_MIN_L_WEIGHT);
        weights[1] = std::min(weights[1], OKLAB_SOFT_CAP4_DARK4_MAX_AB_WEIGHT);
        weights[2] = std::min(weights[2], OKLAB_SOFT_CAP4_DARK4_MAX_AB_WEIGHT);
    }
    return weights;
}

bool color_solver_mode_is_perceptual(ColorSolverMode solver_mode)
{
    return solver_mode == ColorSolverMode::Oklab || solver_mode == ColorSolverMode::OklabSoftCap4Dark4;
}

uint8_t stack_role_index(ColorSolverStackComponentRole role)
{
    return uint8_t(std::clamp(int(role), int(ColorSolverStackComponentRole::Generic), int(ColorSolverStackComponentRole::White)));
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
                                              const std::array<float, 3>    &axis_weights,
                                              ColorSolverMode                solver_mode)
{
    const size_t coord_idx = candidate_idx * 3;
    const float dl = candidates.perceptual_coords[coord_idx + 0] - target_oklab[0];
    const float da = candidates.perceptual_coords[coord_idx + 1] - target_oklab[1];
    const float db = candidates.perceptual_coords[coord_idx + 2] - target_oklab[2];
    float error = axis_weights[0] * dl * dl + axis_weights[1] * da * da + axis_weights[2] * db * db;
    if (solver_mode == ColorSolverMode::OklabSoftCap4Dark4) {
        const float under_l = std::max(0.f, target_oklab[0] - candidates.perceptual_coords[coord_idx + 0] -
                                                OKLAB_SOFT_CAP4_DARK4_DARK_TOLERANCE);
        error += OKLAB_SOFT_CAP4_DARK4_DARK_PENALTY * color_solver_oklab_chroma_factor(target_oklab) * under_l * under_l;
    }
    return error;
}

template <class CandidateSet>
ColorSolverNearestResult nearest_color_solver_candidates_perceptual_linear(const CandidateSet             &candidates,
                                                                           const std::array<float, 3>    &target_oklab,
                                                                           const std::array<float, 3>    &axis_weights,
                                                                           ColorSolverMode                solver_mode)
{
    ColorSolverNearestResult result;
    const size_t candidate_count = candidates.perceptual_coords.size() / 3;
    for (size_t candidate_idx = 0; candidate_idx < candidate_count; ++candidate_idx)
        update_color_solver_nearest_result(
            result,
            candidate_idx,
            color_solver_candidate_perceptual_error(candidates, candidate_idx, target_oklab, axis_weights, solver_mode));
    return result;
}

template <class CandidateSet>
void query_color_solver_perceptual_kd_tree(const CandidateSet             &candidates,
                                           const std::array<float, 3>    &target_oklab,
                                           const std::array<float, 3>    &axis_weights,
                                           ColorSolverMode                solver_mode,
                                           int                            node_idx,
                                           ColorSolverNearestResult      &result)
{
    if (node_idx < 0 || size_t(node_idx) >= candidates.perceptual_kd_nodes.size())
        return;

    const size_t candidate_count = candidates.perceptual_coords.size() / 3;
    const ColorSolverCandidateSet::KdNode &node = candidates.perceptual_kd_nodes[size_t(node_idx)];
    if (size_t(node.candidate_idx) >= candidate_count) {
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, solver_mode, node.left, result);
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, solver_mode, node.right, result);
        return;
    }

    update_color_solver_nearest_result(
        result,
        size_t(node.candidate_idx),
        color_solver_candidate_perceptual_error(candidates, size_t(node.candidate_idx), target_oklab, axis_weights, solver_mode));

    const size_t coord_idx = size_t(node.candidate_idx) * 3;
    const size_t axis = std::min<size_t>(node.axis, 2);
    const float split_delta = target_oklab[axis] - candidates.perceptual_coords[coord_idx + axis];
    const int near_node = split_delta <= 0.f ? node.left : node.right;
    const int far_node = split_delta <= 0.f ? node.right : node.left;

    query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, solver_mode, near_node, result);
    if (axis_weights[axis] * split_delta * split_delta <= result.second_error)
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, solver_mode, far_node, result);
}

template <class CandidateSet>
ColorSolverNearestResult nearest_color_solver_candidates_perceptual(const CandidateSet             &candidates,
                                                                    const std::array<float, 3>    &target_rgb,
                                                                    ColorSolverMode                solver_mode)
{
    ColorSolverNearestResult result;
    const size_t candidate_count = candidates.perceptual_coords.size() / 3;
    if (candidate_count == 0 || candidates.perceptual_coords.size() != candidates.rgbs.size())
        return result;

    const std::array<float, 3> target_oklab = oklab_from_srgb(target_rgb);
    const std::array<float, 3> axis_weights = color_solver_perceptual_axis_weights(target_oklab, solver_mode);
    if (candidates.perceptual_kd_root >= 0 && !candidates.perceptual_kd_nodes.empty())
        query_color_solver_perceptual_kd_tree(candidates, target_oklab, axis_weights, solver_mode, candidates.perceptual_kd_root, result);
    if (result.best_idx >= candidate_count)
        result = nearest_color_solver_candidates_perceptual_linear(candidates, target_oklab, axis_weights, solver_mode);
    return result;
}

std::array<float, 3> transmission_from_density_bias(const std::array<float, 3> &density_bias, float scalar_transmission)
{
    const float transmission = std::clamp(std::isfinite(scalar_transmission) ? scalar_transmission : 0.5f, 1e-4f, 0.9999f);
    auto weighted_transmission = [&density_bias, transmission](float scale) {
        return 0.2126f * std::pow(transmission, density_bias[0] * scale) +
               0.7152f * std::pow(transmission, density_bias[1] * scale) +
               0.0722f * std::pow(transmission, density_bias[2] * scale);
    };

    float lo = 0.f;
    float hi = 8.f;
    while (weighted_transmission(hi) > transmission && hi < 256.f)
        hi *= 2.f;
    for (int step = 0; step < 24; ++step) {
        const float mid = 0.5f * (lo + hi);
        if (weighted_transmission(mid) > transmission)
            lo = mid;
        else
            hi = mid;
    }
    const float scale = 0.5f * (lo + hi);
    return {
        std::pow(transmission, density_bias[0] * scale),
        std::pow(transmission, density_bias[1] * scale),
        std::pow(transmission, density_bias[2] * scale)
    };
}

std::array<float, 3> rgb_density_bias(const std::array<float, 3> &srgb_color, float strength_scale)
{
    const std::array<float, 3> linear = srgb_to_linear_color(srgb_color);
    const float min_channel = std::min({ linear[0], linear[1], linear[2] });
    const float max_channel = std::max({ linear[0], linear[1], linear[2] });
    const float saturation = max_channel > 1e-4f ? std::clamp((max_channel - min_channel) / max_channel, 0.f, 1.f) : 0.f;
    const float strength = std::clamp(strength_scale, 0.f, 2.f) * saturation;
    if (strength <= 1e-4f)
        return { 1.f, 1.f, 1.f };

    const float luminance = std::clamp(linear_luminance(linear), 0.02f, 0.98f);
    std::array<float, 3> density_bias;
    for (size_t channel = 0; channel < 3; ++channel) {
        const float channel_level = std::max(linear[channel], 0.02f);
        density_bias[channel] = std::pow(std::clamp(luminance / channel_level, 0.25f, 4.f), strength);
    }
    return density_bias;
}

std::array<float, 3> beer_lambert_rgb_transmission(const std::array<float, 3> &srgb_color, float scalar_transmission)
{
    return transmission_from_density_bias(rgb_density_bias(srgb_color, 0.75f), scalar_transmission);
}

std::array<float, 3> td_effective_alpha_density_bias(const std::array<float, 3> &srgb_color,
                                                     ColorSolverStackComponentRole role)
{
    switch (role) {
    case ColorSolverStackComponentRole::Cyan:
        return { 1.f, TD_EFFECTIVE_ALPHA_PASS_DENSITY, TD_EFFECTIVE_ALPHA_PASS_DENSITY };
    case ColorSolverStackComponentRole::Magenta:
        return { TD_EFFECTIVE_ALPHA_PASS_DENSITY, 1.f, TD_EFFECTIVE_ALPHA_PASS_DENSITY };
    case ColorSolverStackComponentRole::Yellow:
        return { TD_EFFECTIVE_ALPHA_PASS_DENSITY, TD_EFFECTIVE_ALPHA_PASS_DENSITY, 1.f };
    case ColorSolverStackComponentRole::Black:
        return { 1.f, 1.f, 1.f };
    case ColorSolverStackComponentRole::White:
        return { TD_EFFECTIVE_ALPHA_WHITE_DENSITY, TD_EFFECTIVE_ALPHA_WHITE_DENSITY, TD_EFFECTIVE_ALPHA_WHITE_DENSITY };
    default:
        return rgb_density_bias(srgb_color, 0.85f);
    }
}

std::array<float, 3> td_effective_alpha_transmission(const std::array<float, 3> &srgb_color,
                                                     float scalar_transmission,
                                                     ColorSolverStackComponentRole role)
{
    return transmission_from_density_bias(td_effective_alpha_density_bias(srgb_color, role), scalar_transmission);
}

std::array<float, 3> mix_ordered_stack_beer_lambert_rgb(const std::vector<std::array<float, 3>> &colors_with_background,
                                                        const std::vector<uint16_t>             &surface_to_deep,
                                                        const std::vector<float>                &layer_opacities,
                                                        float                                    surface_scatter)
{
    if (colors_with_background.empty() || surface_to_deep.empty())
        return colors_with_background.empty() ? std::array<float, 3>{ { 0.f, 0.f, 0.f } } : colors_with_background.back();

    const size_t component_count = colors_with_background.size() - 1;
    std::array<float, 3> accumulated { 0.f, 0.f, 0.f };
    std::array<float, 3> remaining { 1.f, 1.f, 1.f };
    bool surface_layer = true;
    const float safe_surface_scatter =
        std::clamp(std::isfinite(surface_scatter) ? surface_scatter : 0.f, 0.f, 0.5f);
    for (uint16_t component_idx : surface_to_deep) {
        if (size_t(component_idx) >= component_count)
            continue;
        const float opacity =
            size_t(component_idx) < layer_opacities.size() && std::isfinite(layer_opacities[size_t(component_idx)]) ?
                std::clamp(layer_opacities[size_t(component_idx)], 1e-4f, 0.9999f) :
                0.5f;
        std::array<float, 3> transmission =
            beer_lambert_rgb_transmission(colors_with_background[size_t(component_idx)], 1.f - opacity);
        if (surface_layer) {
            for (float &channel_transmission : transmission) {
                const float channel_opacity = std::clamp(safe_surface_scatter + (1.f - safe_surface_scatter) * (1.f - channel_transmission),
                                                         1e-4f,
                                                         0.9999f);
                channel_transmission = 1.f - channel_opacity;
            }
            surface_layer = false;
        }
        const std::array<float, 3> layer_linear = srgb_to_linear_color(colors_with_background[size_t(component_idx)]);
        for (size_t channel = 0; channel < 3; ++channel) {
            const float channel_opacity = 1.f - transmission[channel];
            accumulated[channel] += remaining[channel] * channel_opacity * layer_linear[channel];
            remaining[channel] *= transmission[channel];
        }
        if (std::max({ remaining[0], remaining[1], remaining[2] }) <= 1e-5f)
            break;
    }

    const std::array<float, 3> background_linear = srgb_to_linear_color(colors_with_background.back());
    for (size_t channel = 0; channel < 3; ++channel)
        accumulated[channel] += remaining[channel] * background_linear[channel];
    return linear_to_srgb_color(accumulated);
}

std::array<float, 3> mix_ordered_stack_td_effective_alpha(
    const std::vector<std::array<float, 3>>       &colors_with_background,
    const std::vector<uint16_t>                   &surface_to_deep,
    const std::vector<float>                      &layer_opacities,
    float                                          surface_scatter,
    const std::vector<ColorSolverStackComponentRole> &component_roles)
{
    if (colors_with_background.empty() || surface_to_deep.empty())
        return colors_with_background.empty() ? std::array<float, 3>{ { 0.f, 0.f, 0.f } } : colors_with_background.back();

    const size_t component_count = colors_with_background.size() - 1;
    std::array<float, 3> accumulated { 0.f, 0.f, 0.f };
    std::array<float, 3> remaining { 1.f, 1.f, 1.f };
    bool surface_layer = true;
    const float safe_surface_scatter =
        std::clamp(std::isfinite(surface_scatter) ? surface_scatter : 0.f, 0.f, 0.5f);
    for (uint16_t component_idx : surface_to_deep) {
        if (size_t(component_idx) >= component_count)
            continue;
        const float opacity =
            size_t(component_idx) < layer_opacities.size() && std::isfinite(layer_opacities[size_t(component_idx)]) ?
                std::clamp(layer_opacities[size_t(component_idx)], 1e-4f, 0.9999f) :
                0.5f;
        const float scalar_transmission = std::pow(1.f - opacity, TD_EFFECTIVE_ALPHA_DENSITY_SCALE);
        const ColorSolverStackComponentRole role =
            size_t(component_idx) < component_roles.size() ?
                component_roles[size_t(component_idx)] :
                ColorSolverStackComponentRole::Generic;
        std::array<float, 3> transmission =
            td_effective_alpha_transmission(colors_with_background[size_t(component_idx)], scalar_transmission, role);
        if (surface_layer) {
            for (float &channel_transmission : transmission) {
                const float channel_opacity = std::clamp(safe_surface_scatter + (1.f - safe_surface_scatter) * (1.f - channel_transmission),
                                                         1e-4f,
                                                         0.9999f);
                channel_transmission = 1.f - channel_opacity;
            }
            surface_layer = false;
        }
        const std::array<float, 3> layer_linear = srgb_to_linear_color(colors_with_background[size_t(component_idx)]);
        for (size_t channel = 0; channel < 3; ++channel) {
            const float channel_opacity = 1.f - transmission[channel];
            accumulated[channel] += remaining[channel] * channel_opacity * layer_linear[channel];
            remaining[channel] *= transmission[channel];
        }
        if (std::max({ remaining[0], remaining[1], remaining[2] }) <= 1e-5f)
            break;
    }

    const std::array<float, 3> background_linear = srgb_to_linear_color(colors_with_background.back());
    for (size_t channel = 0; channel < 3; ++channel)
        accumulated[channel] += remaining[channel] * background_linear[channel];
    return linear_to_srgb_color(accumulated);
}

std::array<float, 3> mix_ordered_stack_with_buffers(const std::vector<std::array<float, 3>> &colors_with_background,
                                                    std::vector<float>                      &weights,
                                                    const std::vector<uint16_t>             &surface_to_deep,
                                                    const std::vector<float>                &layer_opacities,
                                                    ColorSolverMixModel                       mix_model,
                                                    float                                    surface_scatter,
                                                    bool                                     beer_lambert_rgb_correction,
                                                    bool                                     td_effective_alpha_correction,
                                                    const std::vector<ColorSolverStackComponentRole> &component_roles)
{
    if (colors_with_background.empty() || surface_to_deep.empty())
        return colors_with_background.empty() ? std::array<float, 3>{ { 0.f, 0.f, 0.f } } : colors_with_background.back();

    if (td_effective_alpha_correction)
        return mix_ordered_stack_td_effective_alpha(colors_with_background, surface_to_deep, layer_opacities, surface_scatter, component_roles);
    if (beer_lambert_rgb_correction)
        return mix_ordered_stack_beer_lambert_rgb(colors_with_background, surface_to_deep, layer_opacities, surface_scatter);

    const size_t component_count = colors_with_background.size() - 1;
    weights.assign(colors_with_background.size(), 0.f);
    float transmission = 1.f;
    bool surface_layer = true;
    const float safe_surface_scatter =
        std::clamp(std::isfinite(surface_scatter) ? surface_scatter : 0.f, 0.f, 0.5f);
    for (uint16_t component_idx : surface_to_deep) {
        if (size_t(component_idx) >= component_count)
            continue;
        float opacity =
            size_t(component_idx) < layer_opacities.size() && std::isfinite(layer_opacities[size_t(component_idx)]) ?
                std::clamp(layer_opacities[size_t(component_idx)], 1e-4f, 0.9999f) :
                0.5f;
        if (surface_layer) {
            opacity = std::clamp(safe_surface_scatter + (1.f - safe_surface_scatter) * opacity, 1e-4f, 0.9999f);
            surface_layer = false;
        }
        weights[size_t(component_idx)] += transmission * opacity;
        transmission *= 1.f - opacity;
        if (transmission <= 1e-5f)
            break;
    }
    weights.back() = std::max(0.f, transmission);
    return mix_components_by_model(colors_with_background, weights, mix_model);
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

struct OrderedStackExpansionBeamState {
    std::vector<uint16_t> stack;
    size_t next_control_idx { 0 };
    float score { 0.f };
};

int ordered_stack_run_length(const std::vector<uint16_t> &stack, uint16_t component)
{
    int run = 0;
    for (auto it = stack.rbegin(); it != stack.rend() && *it == component; ++it)
        ++run;
    return run;
}

float ordered_stack_beam_transition_score(const std::vector<uint16_t> &control,
                                          const std::vector<float>    &layer_opacities,
                                          const OrderedStackExpansionBeamState &state,
                                          uint16_t                     component,
                                          bool                         consume,
                                          int                          stack_depth)
{
    const int slot = int(state.stack.size());
    const float control_pos =
        stack_depth > 0 ?
            ((float(slot) + 0.5f) * float(control.size()) / float(stack_depth) - 0.5f) :
            float(state.next_control_idx);
    const float layout_penalty = std::abs(float(state.next_control_idx) - control_pos);
    const float depth_t =
        stack_depth > 1 ?
            std::clamp(float(slot) / float(stack_depth - 1), 0.f, 1.f) :
            0.f;
    const size_t component_idx = size_t(component);
    const float layer_opacity =
        component_idx < layer_opacities.size() && std::isfinite(layer_opacities[component_idx]) ?
            std::clamp(layer_opacities[component_idx], 1e-4f, 0.9999f) :
            0.5f;
    const float transmission = 1.f - layer_opacity;
    const int run = ordered_stack_run_length(state.stack, component);

    float score = -0.55f * layout_penalty;
    if (!consume)
        score += 0.20f * transmission + 0.12f * depth_t - 0.10f * (1.f - depth_t);
    else
        score += 0.02f;
    if (run > 0)
        score -= (0.12f + 0.30f * (1.f - depth_t)) * float(std::min(run, 5));
    return score;
}

void prune_ordered_stack_expansion_beam(std::vector<OrderedStackExpansionBeamState> &states, size_t beam_width)
{
    std::sort(states.begin(), states.end(), [](const OrderedStackExpansionBeamState &lhs,
                                               const OrderedStackExpansionBeamState &rhs) {
        return lhs.score > rhs.score;
    });
    std::vector<OrderedStackExpansionBeamState> pruned;
    pruned.reserve(std::min(states.size(), beam_width));
    for (OrderedStackExpansionBeamState &state : states) {
        const bool duplicate =
            std::any_of(pruned.begin(), pruned.end(), [&state](const OrderedStackExpansionBeamState &existing) {
                return existing.next_control_idx == state.next_control_idx && existing.stack == state.stack;
            });
        if (duplicate)
            continue;
        pruned.emplace_back(std::move(state));
        if (pruned.size() >= beam_width)
            break;
    }
    states = std::move(pruned);
}

std::vector<std::vector<uint16_t>> beam_stretched_ordered_stack_variants(const std::vector<uint16_t> &control,
                                                                         const std::vector<float>    &layer_opacities,
                                                                         int                          stack_depth,
                                                                         size_t                       variant_limit)
{
    std::vector<std::vector<uint16_t>> variants;
    if (control.empty() || stack_depth <= 0 || variant_limit == 0)
        return variants;
    if (int(control.size()) >= stack_depth) {
        std::vector<uint16_t> variant(control.begin(), control.begin() + std::min(control.size(), size_t(stack_depth)));
        append_unique_ordered_stack_variant(variants, std::move(variant));
        return variants;
    }

    const size_t beam_width = std::max<size_t>(variant_limit, std::min<size_t>(32, variant_limit * 2));
    std::vector<OrderedStackExpansionBeamState> beam(1);
    beam.front().stack.reserve(size_t(stack_depth));

    for (int slot = 0; slot < stack_depth; ++slot) {
        std::vector<OrderedStackExpansionBeamState> next;
        next.reserve(beam.size() * 2);
        for (const OrderedStackExpansionBeamState &state : beam) {
            if (state.next_control_idx >= control.size())
                continue;
            const uint16_t component = control[state.next_control_idx];
            const int remaining_slots = stack_depth - slot - 1;
            const int remaining_after_consume = int(control.size() - state.next_control_idx - 1);
            if (remaining_slots >= remaining_after_consume) {
                OrderedStackExpansionBeamState consumed = state;
                consumed.stack.emplace_back(component);
                consumed.score += ordered_stack_beam_transition_score(control, layer_opacities, state, component, true, stack_depth);
                ++consumed.next_control_idx;
                next.emplace_back(std::move(consumed));
            }

            const int remaining_after_duplicate = int(control.size() - state.next_control_idx);
            if (remaining_slots >= remaining_after_duplicate) {
                OrderedStackExpansionBeamState duplicated = state;
                duplicated.stack.emplace_back(component);
                duplicated.score += ordered_stack_beam_transition_score(control, layer_opacities, state, component, false, stack_depth);
                next.emplace_back(std::move(duplicated));
            }
        }
        if (next.empty())
            break;
        prune_ordered_stack_expansion_beam(next, beam_width);
        beam = std::move(next);
    }

    std::sort(beam.begin(), beam.end(), [](const OrderedStackExpansionBeamState &lhs,
                                           const OrderedStackExpansionBeamState &rhs) {
        return lhs.score > rhs.score;
    });
    for (OrderedStackExpansionBeamState &state : beam) {
        if (state.next_control_idx != control.size() || int(state.stack.size()) != stack_depth)
            continue;
        append_unique_ordered_stack_variant(variants, std::move(state.stack));
        if (variants.size() >= variant_limit)
            break;
    }

    if (variants.size() < variant_limit) {
        std::vector<std::vector<uint16_t>> fallback =
            stretched_ordered_stack_variants(control, layer_opacities, stack_depth);
        for (std::vector<uint16_t> &variant : fallback) {
            append_unique_ordered_stack_variant(variants, std::move(variant));
            if (variants.size() >= variant_limit)
                break;
        }
    }
    return variants;
}

void append_ordered_stack_candidate(ColorSolverOrderedStackCandidateSet       &candidates,
                                    const std::vector<std::array<float, 3>>  &colors_with_background,
                                    std::vector<float>                       &weights,
                                    const std::vector<uint16_t>              &surface_to_deep,
                                    const std::vector<float>                 &layer_opacities,
                                    ColorSolverMixModel                       mix_model,
                                    int                                       simulated_stack_depth,
                                    float                                     surface_scatter,
                                    bool                                      beer_lambert_rgb_correction,
                                    bool                                      td_effective_alpha_correction,
                                    const std::vector<ColorSolverStackComponentRole> &component_roles)
{
    std::vector<uint16_t> simulated_surface_to_deep;
    const std::vector<uint16_t> *mix_stack = &surface_to_deep;
    if (simulated_stack_depth > 0 && simulated_stack_depth != int(surface_to_deep.size())) {
        simulated_surface_to_deep = repeat_ordered_stack_to_depth(surface_to_deep, simulated_stack_depth);
        mix_stack = &simulated_surface_to_deep;
    }
    const std::array<float, 3> mixed =
        mix_ordered_stack_with_buffers(colors_with_background,
                                       weights,
                                       *mix_stack,
                                       layer_opacities,
                                       mix_model,
                                       surface_scatter,
                                       beer_lambert_rgb_correction,
                                       td_effective_alpha_correction,
                                       component_roles);
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
    return ColorSolverMixModel(std::clamp(model, int(ColorSolverMixModel::PigmentPainter), int(ColorSolverMixModel::PrusaFdmMixer)));
}

ColorSolverLookupMode color_solver_lookup_mode_from_index(int mode)
{
    return ColorSolverLookupMode(std::clamp(mode, int(ColorSolverLookupMode::ClosestMix), int(ColorSolverLookupMode::BlendClosestTwo)));
}

ColorSolverMode color_solver_mode_from_index(int mode)
{
    return ColorSolverMode(std::clamp(mode, int(ColorSolverMode::RGB), int(ColorSolverMode::OklabSoftCap4Dark4)));
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
    std::vector<float> float_weights;
    float_weights.reserve(weights.size());
    for (const int weight : weights)
        float_weights.emplace_back(float(std::max(0, weight)));
    return mix_components_by_model(component_colors, float_weights, mix_model);
}

std::array<float, 3> mix_color_solver_components(const std::vector<std::array<float, 3>> &component_colors,
                                                 const std::vector<float>                &weights,
                                                 ColorSolverMixModel                       mix_model)
{
    std::vector<float> safe_weights;
    safe_weights.reserve(weights.size());
    for (const float weight : weights)
        safe_weights.emplace_back(std::isfinite(weight) ? std::clamp(weight, 0.f, 1.f) : 0.f);
    return mix_components_by_model(component_colors, safe_weights, mix_model);
}

std::array<float, 3> mix_color_solver_ordered_stack(const std::vector<std::array<float, 3>> &component_colors,
                                                    const std::vector<uint16_t>             &surface_to_deep,
                                                    const std::vector<float>                &layer_opacities,
                                                    const std::array<float, 3>              &background_rgb,
                                                    ColorSolverMixModel                       mix_model,
                                                    float                                     surface_scatter,
                                                    bool                                      beer_lambert_rgb_correction,
                                                    bool                                      td_effective_alpha_correction,
                                                    const std::vector<ColorSolverStackComponentRole> &component_roles)
{
    if (component_colors.empty() || surface_to_deep.empty())
        return background_rgb;

    std::vector<std::array<float, 3>> colors = component_colors;
    colors.emplace_back(background_rgb);
    std::vector<float> weights;
    return mix_ordered_stack_with_buffers(colors,
                                          weights,
                                          surface_to_deep,
                                          layer_opacities,
                                          mix_model,
                                          surface_scatter,
                                          beer_lambert_rgb_correction,
                                          td_effective_alpha_correction,
                                          component_roles);
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
        color_solver_mode_is_perceptual(solver_mode) ?
            nearest_color_solver_candidates_perceptual(candidates, target_rgb, solver_mode) :
            nearest_color_solver_candidates(candidates, target_rgb);
    if (nearest.best_idx >= candidate_count && color_solver_mode_is_perceptual(solver_mode))
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
                                                           size_t                                    stack_item_limit,
                                                           float                                     surface_scatter,
                                                           bool                                      beer_lambert_rgb_correction,
                                                           bool                                      td_effective_alpha_correction,
                                                           const std::vector<ColorSolverStackComponentRole> &component_roles,
                                                           bool                                      beam_search_stack_expansion)
{
    std::ostringstream key;
    key << component_colors.size();
    key << "|mx" << mix_model_index(mix_model);
    key << "|sd" << stack_depth;
    key << "|vd" << simulated_stack_depth;
    key << "|lim" << candidate_limit;
    key << "|item" << stack_item_limit;
    const float safe_surface_scatter =
        std::clamp(std::isfinite(surface_scatter) ? surface_scatter : 0.f, 0.f, 0.5f);
    key << "|ss" << int(std::lround(safe_surface_scatter * 1000000.f));
    key << "|bl" << (beer_lambert_rgb_correction ? 1 : 0);
    key << "|ea" << (td_effective_alpha_correction ? 1 : 0);
    key << "|beam" << (beam_search_stack_expansion ? 1 : 0);
    if (td_effective_alpha_correction) {
        key << "|role";
        for (size_t idx = 0; idx < component_colors.size(); ++idx) {
            const ColorSolverStackComponentRole role =
                idx < component_roles.size() ? component_roles[idx] : ColorSolverStackComponentRole::Generic;
            key << ',' << int(stack_role_index(role));
        }
    }
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
    size_t                                    stack_item_limit,
    float                                     surface_scatter,
    bool                                      beer_lambert_rgb_correction,
    bool                                      td_effective_alpha_correction,
    const std::vector<ColorSolverStackComponentRole> &component_roles,
    bool                                      beam_search_stack_expansion)
{
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
                                               mix_model,
                                               simulated_depth,
                                               surface_scatter,
                                               beer_lambert_rgb_correction,
                                               td_effective_alpha_correction,
                                               component_roles);
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
                beam_search_stack_expansion ?
                    beam_stretched_ordered_stack_variants(control, layer_opacities, stack_depth, variant_budget) :
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
                                               mix_model,
                                               simulated_depth,
                                               surface_scatter,
                                               beer_lambert_rgb_correction,
                                               td_effective_alpha_correction,
                                               component_roles);
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
    size_t                                         stack_item_limit,
    float                                          surface_scatter,
    bool                                           beer_lambert_rgb_correction,
    bool                                           td_effective_alpha_correction,
    const std::vector<ColorSolverStackComponentRole> &component_roles,
    bool                                           beam_search_stack_expansion)
{
    const std::string key =
        color_solver_ordered_stack_candidate_cache_key(component_colors,
                                                       layer_opacities,
                                                       background_rgb,
                                                       mix_model,
                                                       stack_depth,
                                                       simulated_stack_depth,
                                                       candidate_limit,
                                                       stack_item_limit,
                                                       surface_scatter,
                                                       beer_lambert_rgb_correction,
                                                       td_effective_alpha_correction,
                                                       component_roles,
                                                       beam_search_stack_expansion);
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
                                                   stack_item_limit,
                                                   surface_scatter,
                                                   beer_lambert_rgb_correction,
                                                   td_effective_alpha_correction,
                                                   component_roles,
                                                   beam_search_stack_expansion)).first->second;
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
        color_solver_mode_is_perceptual(solver_mode) ?
            nearest_color_solver_candidates_perceptual(candidates, target_rgb, solver_mode) :
            nearest_color_solver_candidates(candidates, target_rgb);
    if (nearest.best_idx >= candidate_count && color_solver_mode_is_perceptual(solver_mode))
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
