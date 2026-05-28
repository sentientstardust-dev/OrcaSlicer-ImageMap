// original author: sentientstardust

#ifndef slic3r_TextureMappingContoning_hpp_
#define slic3r_TextureMappingContoning_hpp_

#include "TextureMapping.hpp"

#include <array>
#include <map>
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
                                  std::vector<unsigned int> component_ids);

    bool valid() const { return !m_component_ids.empty() && m_component_ids.size() == m_component_colors.size(); }
    const std::vector<unsigned int>& component_ids() const { return m_component_ids; }
    const std::vector<unsigned int>& components_bottom_to_top() const { return m_components_bottom_to_top; }

    TextureMappingContoningStack solve(const std::array<float, 3> &target_rgb, int stack_layers) const;
    unsigned int component_for_depth(const std::array<float, 3> &target_rgb, int stack_layers, int depth_from_top) const;

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

    std::vector<unsigned int> m_component_ids;
    std::vector<unsigned int> m_components_bottom_to_top;
    std::vector<std::array<float, 3>> m_component_colors;
    std::vector<float> m_component_luminance;
    std::vector<float> m_effective_transmission_distances_mm;
    mutable std::map<int, std::vector<Candidate>> m_candidates_by_depth;
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

} // namespace Slic3r

#endif
