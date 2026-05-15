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
    PigmentPainter = 0
};

enum class ColorSolverLookupMode : int
{
    ClosestMix = 0,
    BlendClosestTwo = 1
};

enum class ColorSolverMode : int
{
    Legacy = 0,
    V2 = 1
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
std::array<float, 3> color_solver_oklab_from_srgb(const std::array<float, 3> &rgb);

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

} // namespace Slic3r

#endif
