// original author: sentientstardust

#ifndef slic3r_TextureMappingContoning_hpp_
#define slic3r_TextureMappingContoning_hpp_

#include "ColorSolver.hpp"
#include "TextureMapping.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace Slic3r {

class PrintConfig;

struct TextureMappingContoningStack {
    std::vector<unsigned int> bottom_to_top;
};

class TextureMappingContoningSolver
{
public:
    TextureMappingContoningSolver() = default;
    TextureMappingContoningSolver(const TextureMappingZone &zone,
                                  const PrintConfig        &config,
                                  std::vector<unsigned int> component_ids,
                                  float                     layer_height_mm = 0.f);

    bool valid() const { return !m_component_ids.empty() && m_component_ids.size() == m_component_colors.size(); }
    const std::vector<unsigned int>& component_ids() const { return m_component_ids; }
    const std::vector<unsigned int>& components_bottom_to_top() const { return m_components_bottom_to_top; }

    TextureMappingContoningStack solve(const std::array<float, 3> &target_rgb,
                                       int                         stack_layers,
                                       bool                        lower_surface = false,
                                       int                         visible_stack_layers = 0) const;
    unsigned int component_for_depth(const std::array<float, 3> &target_rgb, int stack_layers, int depth_from_top, bool lower_surface = false) const;
    std::optional<std::array<float, 3>> stack_rgb(const std::vector<unsigned int> &bottom_to_top,
                                                 bool                             lower_surface = false,
                                                 int                              visible_stack_layers = 0) const;

private:
    struct Candidate {
        std::array<float, 3> rgb { { 0.f, 0.f, 0.f } };
        std::array<float, 3> oklab { { 0.f, 0.f, 0.f } };
        std::vector<int> counts;
        float dark_score { 0.f };
    };

    const std::vector<Candidate>& candidates_for_depth(int stack_layers) const;
    void arrange_stack_for_light_path(std::vector<unsigned int> &bottom_to_top,
                                      const std::array<float, 3> &target_rgb) const;
    std::optional<size_t> component_index(unsigned int component_id) const;

    std::vector<unsigned int> m_component_ids;
    std::vector<unsigned int> m_components_bottom_to_top;
    std::vector<std::array<float, 3>> m_component_colors;
    std::array<float, 3> m_background_rgb { { 0.f, 0.f, 0.f } };
    std::vector<float> m_component_luminance;
    std::vector<float> m_effective_transmission_distances_mm;
    std::vector<float> m_component_layer_opacity;
    float m_layer_height_mm { 0.2f };
    float m_surface_scatter { 0.f };
    ColorSolverMixModel m_mix_model { ColorSolverMixModel::PigmentPainter };
    bool m_td_adjustment_enabled { false };
    bool m_beer_lambert_rgb_correction_enabled { false };
    bool m_td_effective_alpha_correction_enabled { false };
    bool m_beam_search_stack_expansion_enabled { false };
    std::vector<ColorSolverStackComponentRole> m_component_roles;
    mutable std::map<int, std::vector<Candidate>> m_candidates_by_depth;
    mutable std::shared_ptr<ColorSolverOrderedStackCandidateCache> m_ordered_candidate_cache { std::make_shared<ColorSolverOrderedStackCandidateCache>() };
    mutable std::shared_ptr<std::mutex> m_ordered_candidate_cache_mutex { std::make_shared<std::mutex>() };
};

std::vector<unsigned int> texture_mapping_contoning_components_bottom_to_top(const TextureMappingZone      &zone,
                                                                             const PrintConfig            &config,
                                                                             std::vector<unsigned int>     component_ids);
float texture_mapping_contoning_min_feature_mm(const TextureMappingZone          &zone,
                                               const PrintConfig                &config,
                                               const std::vector<unsigned int>  &component_ids,
                                               float                             external_width_mm);
bool texture_mapping_contoning_normal_eligible(float normal_z, float threshold_deg);
std::optional<std::array<float, 3>> texture_mapping_contoning_component_colors(const PrintConfig               &config,
                                                                              const std::vector<unsigned int> &component_ids,
                                                                              std::vector<std::array<float, 3>> &out);
std::vector<ColorSolverStackComponentRole> texture_mapping_contoning_component_roles(const TextureMappingZone &zone,
                                                                                     size_t                    component_count);

} // namespace Slic3r

#endif
