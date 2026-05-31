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

#ifndef slic3r_ColorSolver_hpp_
#define slic3r_ColorSolver_hpp_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace Slic3r {

enum class ColorSolverMixModel : int
{
    PigmentPainter = 0,
    PrusaFdmMixer = 1
};

enum class ColorSolverLookupMode : int
{
    ClosestMix = 0,
    BlendClosestTwo = 1
};

enum class ColorSolverMode : int
{
    RGB = 0,
    Oklab = 1,
    OklabSoftCap4Dark4 = 2
};

enum class ColorSolverStackComponentRole : uint8_t
{
    Generic = 0,
    Cyan = 1,
    Magenta = 2,
    Yellow = 3,
    Black = 4,
    White = 5
};

struct ColorSolverCandidateSet {
    struct KdNode {
        uint32_t candidate_idx { 0 };
        int      left { -1 };
        int      right { -1 };
        uint8_t  axis { 0 };
    };

    size_t component_count { 0 };
    std::vector<float> rgbs;
    std::vector<float> perceptual_coords;
    std::vector<float> weights;
    std::vector<KdNode> kd_nodes;
    std::vector<KdNode> perceptual_kd_nodes;
    int kd_root { -1 };
    int perceptual_kd_root { -1 };

    bool empty() const
    {
        return component_count == 0 || rgbs.empty() || rgbs.size() % 3 != 0 ||
               weights.size() != (rgbs.size() / 3) * component_count;
    }
};

using ColorSolverCandidateCache = std::map<std::string, ColorSolverCandidateSet>;

struct ColorSolverOrderedStackCandidateSet {
    using KdNode = ColorSolverCandidateSet::KdNode;

    size_t component_count { 0 };
    int stack_depth { 0 };
    int simulated_stack_depth { 0 };
    std::vector<float> rgbs;
    std::vector<float> perceptual_coords;
    std::vector<uint16_t> stacks;
    std::vector<KdNode> kd_nodes;
    std::vector<KdNode> perceptual_kd_nodes;
    int kd_root { -1 };
    int perceptual_kd_root { -1 };

    bool empty() const
    {
        return component_count == 0 || stack_depth <= 0 || rgbs.empty() || rgbs.size() % 3 != 0 ||
               stacks.size() != (rgbs.size() / 3) * size_t(stack_depth);
    }
};

struct ColorSolverOrderedStackResult {
    std::vector<uint16_t> surface_to_deep;
    std::array<float, 3> rgb { { 0.f, 0.f, 0.f } };
    bool has_rgb { false };
};

using ColorSolverOrderedStackCandidateCache = std::map<std::string, ColorSolverOrderedStackCandidateSet>;

ColorSolverMixModel color_solver_mix_model_from_index(int model);
ColorSolverLookupMode color_solver_lookup_mode_from_index(int mode);
ColorSolverMode color_solver_mode_from_index(int mode);

int color_solver_total_units_for_component_count(size_t component_count);
size_t color_solver_candidate_count(size_t component_count, int total_units);

std::array<float, 3> mix_color_solver_components(const std::vector<std::array<float, 3>> &component_colors,
                                                 const std::vector<int>                  &weights,
                                                 ColorSolverMixModel                       mix_model);
std::array<float, 3> mix_color_solver_components(const std::vector<std::array<float, 3>> &component_colors,
                                                 const std::vector<float>                &weights,
                                                 ColorSolverMixModel                       mix_model);
std::array<float, 3> mix_color_solver_ordered_stack(const std::vector<std::array<float, 3>> &component_colors,
                                                    const std::vector<uint16_t>             &surface_to_deep,
                                                    const std::vector<float>                &layer_opacities,
                                                    const std::array<float, 3>              &background_rgb,
                                                    ColorSolverMixModel                       mix_model,
                                                    float                                     surface_scatter = 0.f,
                                                    bool                                      beer_lambert_rgb_correction = false,
                                                    bool                                      td_effective_alpha_correction = false,
                                                    const std::vector<ColorSolverStackComponentRole> &component_roles = {},
                                                    const std::vector<float>                &layer_opacities_by_depth = {});
std::array<float, 3> color_solver_oklab_from_srgb(const std::array<float, 3> &rgb);
std::array<float, 3> color_solver_srgb_from_oklab(const std::array<float, 3> &oklab);

std::string color_solver_candidate_cache_key(const std::vector<std::array<float, 3>> &component_colors,
                                             ColorSolverMixModel                       mix_model,
                                             int                                       total_units = 0);
ColorSolverCandidateSet build_color_solver_candidates(const std::vector<std::array<float, 3>> &component_colors,
                                                      ColorSolverMixModel                       mix_model,
                                                      int                                       total_units = 0);
const ColorSolverCandidateSet &color_solver_candidates(ColorSolverCandidateCache                &cache,
                                                       const std::vector<std::array<float, 3>> &component_colors,
                                                       ColorSolverMixModel                       mix_model,
                                                       int                                       total_units = 0);
std::vector<float> solve_color_solver_weights_for_target(const ColorSolverCandidateSet &candidates,
                                                         const std::array<float, 3>     &target_rgb,
                                                         ColorSolverLookupMode           lookup_mode,
                                                         ColorSolverMode                 solver_mode);
std::string color_solver_ordered_stack_candidate_cache_key(const std::vector<std::array<float, 3>> &component_colors,
                                                           const std::vector<float>                &layer_opacities,
                                                           const std::array<float, 3>              &background_rgb,
                                                           ColorSolverMixModel                       mix_model,
                                                           int                                       stack_depth,
                                                           int                                       simulated_stack_depth = 0,
                                                           size_t                                    candidate_limit = 0,
                                                           size_t                                    stack_item_limit = 0,
                                                           float                                     surface_scatter = 0.f,
                                                           bool                                      beer_lambert_rgb_correction = false,
                                                           bool                                      td_effective_alpha_correction = false,
                                                           const std::vector<ColorSolverStackComponentRole> &component_roles = {},
                                                           bool                                      beam_search_stack_expansion = false,
                                                           const std::vector<float>                  &layer_opacities_by_depth = {});
ColorSolverOrderedStackCandidateSet build_color_solver_ordered_stack_candidates(
    const std::vector<std::array<float, 3>> &component_colors,
    const std::vector<float>                &layer_opacities,
    const std::array<float, 3>              &background_rgb,
    ColorSolverMixModel                       mix_model,
    int                                       stack_depth,
    int                                       simulated_stack_depth = 0,
    size_t                                    candidate_limit = 0,
    size_t                                    stack_item_limit = 0,
    float                                     surface_scatter = 0.f,
    bool                                      beer_lambert_rgb_correction = false,
    bool                                      td_effective_alpha_correction = false,
    const std::vector<ColorSolverStackComponentRole> &component_roles = {},
    bool                                      beam_search_stack_expansion = false,
    const std::vector<float>                 &layer_opacities_by_depth = {});
const ColorSolverOrderedStackCandidateSet &color_solver_ordered_stack_candidates(
    ColorSolverOrderedStackCandidateCache        &cache,
    const std::vector<std::array<float, 3>>      &component_colors,
    const std::vector<float>                     &layer_opacities,
    const std::array<float, 3>                   &background_rgb,
    ColorSolverMixModel                            mix_model,
    int                                            stack_depth,
    int                                            simulated_stack_depth = 0,
    size_t                                         candidate_limit = 0,
    size_t                                         stack_item_limit = 0,
    float                                          surface_scatter = 0.f,
    bool                                           beer_lambert_rgb_correction = false,
    bool                                           td_effective_alpha_correction = false,
    const std::vector<ColorSolverStackComponentRole> &component_roles = {},
    bool                                           beam_search_stack_expansion = false,
    const std::vector<float>                      &layer_opacities_by_depth = {});
ColorSolverOrderedStackResult solve_color_solver_ordered_stack_result_for_target(
    const ColorSolverOrderedStackCandidateSet &candidates,
    const std::array<float, 3>                &target_rgb,
    ColorSolverMode                            solver_mode);
std::vector<uint16_t> solve_color_solver_ordered_stack_for_target(
    const ColorSolverOrderedStackCandidateSet &candidates,
    const std::array<float, 3>                &target_rgb,
    ColorSolverMode                            solver_mode);

} // namespace Slic3r

#endif
