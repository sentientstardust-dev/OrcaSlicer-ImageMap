// original author: sentientstardust

#ifndef slic3r_TextureMappingContoning_hpp_
#define slic3r_TextureMappingContoning_hpp_

#include "ColorSolver.hpp"
#include "TextureMapping.hpp"

#include <atomic>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace Slic3r {

class PrintConfig;

struct TextureMappingContoningStack {
    std::vector<unsigned int> bottom_to_top;
    std::optional<std::array<float, 3>> rgb;
};

struct TextureMappingContoningNearestMeasuredSampleFallbackIssue {
    bool lower_surface { false };
    bool stack_depth_mismatch { false };
    bool layer_height_mismatch { false };
    int shell_depth_from_surface { 0 };
    int physical_layer { -1 };
    int actual_stack_layers { 0 };
    int visible_stack_layers { 0 };
    int expected_stack_layers { 0 };
    float actual_layer_height_mm { 0.f };
    float expected_layer_height_mm { 0.f };
};

struct TextureMappingContoningNearestMeasuredSampleFallbackArea {
    double fallback_area_mm2 { 0. };
    double total_area_mm2 { 0. };
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
    std::string calibrated_stack_model_key() const;
    bool nearest_measured_sample_mode() const;
    int nearest_measured_sample_stack_depth() const;
    bool nearest_measured_sample_fallback_used() const;
    bool nearest_measured_sample_fallback_used(bool lower_surface) const;
    std::vector<TextureMappingContoningNearestMeasuredSampleFallbackIssue> nearest_measured_sample_fallback_issues(bool lower_surface) const;
    TextureMappingContoningNearestMeasuredSampleFallbackArea nearest_measured_sample_fallback_area(bool lower_surface) const;
    std::string nearest_measured_sample_fallback_name() const;
    bool nearest_measured_sample_stack_compatible(int                       stack_layers,
                                                  int                       visible_depth,
                                                  const std::vector<float> &surface_to_deep_layer_heights_mm,
                                                  const std::vector<int>   &surface_to_deep_layer_ids,
                                                  bool                      lower_surface) const;
    void record_nearest_measured_sample_fallback_area(bool lower_surface,
                                                      double fallback_area_mm2,
                                                      double total_area_mm2) const;

    TextureMappingContoningStack solve(const std::array<float, 3> &target_rgb,
                                       int                         stack_layers,
                                       bool                        lower_surface = false,
                                       int                         visible_stack_layers = 0,
                                       const std::vector<float>    &surface_to_deep_layer_heights_mm = {},
                                       const std::vector<int>      &surface_to_deep_layer_ids = {},
                                       bool                        record_nearest_measured_sample_fallback = true) const;
    TextureMappingContoningStack solve_without_beam_search_stack_expansion(
                                       const std::array<float, 3> &target_rgb,
                                       int                         stack_layers,
                                       bool                        lower_surface = false,
                                       int                         visible_stack_layers = 0,
                                       const std::vector<float>    &surface_to_deep_layer_heights_mm = {},
                                       const std::vector<int>      &surface_to_deep_layer_ids = {},
                                       bool                        record_nearest_measured_sample_fallback = true) const;
    unsigned int component_for_depth(const std::array<float, 3> &target_rgb, int stack_layers, int depth_from_top, bool lower_surface = false) const;
    std::optional<std::array<float, 3>> stack_rgb(const std::vector<unsigned int> &bottom_to_top,
                                                 bool                             lower_surface = false,
                                                 int                              visible_stack_layers = 0,
                                                 const std::vector<float>         &surface_to_deep_layer_heights_mm = {},
                                                 const std::vector<int>           &surface_to_deep_layer_ids = {},
                                                 bool                             record_nearest_measured_sample_fallback = true) const;

private:
    struct Candidate {
        std::array<float, 3> rgb { { 0.f, 0.f, 0.f } };
        std::array<float, 3> oklab { { 0.f, 0.f, 0.f } };
        std::vector<int> counts;
        float dark_score { 0.f };
    };
    struct PredictionOptions {
        const ColorSolverCalibratedStackModel *calibrated_stack_model { nullptr };
        bool beer_lambert_rgb_correction_enabled { false };
        bool td_effective_alpha_correction_enabled { false };
        bool adaptive_spectral_correction_enabled { false };
    };

    const std::vector<Candidate>& candidates_for_depth(int stack_layers) const;
    void arrange_stack_for_light_path(std::vector<unsigned int> &bottom_to_top,
                                      const std::array<float, 3> &target_rgb) const;
    std::optional<size_t> component_index(unsigned int component_id) const;
    std::vector<float> layer_opacities_by_depth(const std::vector<float> &surface_to_deep_layer_heights_mm,
                                                int                       visible_depth) const;
    bool nearest_measured_sample_compatible(int                       stack_layers,
                                            int                       visible_depth,
                                            const std::vector<float> &surface_to_deep_layer_heights_mm,
                                            const std::vector<int>   &surface_to_deep_layer_ids,
                                            bool                      lower_surface,
                                            bool                      record_fallback_issue) const;
    void record_nearest_measured_sample_fallback_issue(
        const TextureMappingContoningNearestMeasuredSampleFallbackIssue &issue) const;
    PredictionOptions prediction_options_for_layers(int                       stack_layers,
                                                    int                       visible_depth,
                                                    const std::vector<float> &surface_to_deep_layer_heights_mm,
                                                    const std::vector<int>   &surface_to_deep_layer_ids,
                                                    bool                      lower_surface,
                                                    bool                      record_fallback_issue) const;
    TextureMappingContoningStack solve_impl(const std::array<float, 3> &target_rgb,
                                            int                         stack_layers,
                                            bool                        lower_surface,
                                            int                         visible_stack_layers,
                                            const std::vector<float>    &surface_to_deep_layer_heights_mm,
                                            const std::vector<int>      &surface_to_deep_layer_ids,
                                            bool                        beam_search_stack_expansion_enabled,
                                            bool                        record_nearest_measured_sample_fallback) const;

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
    bool m_adaptive_spectral_correction_enabled { false };
    bool m_beer_lambert_rgb_correction_enabled { false };
    bool m_td_effective_alpha_correction_enabled { false };
    bool m_variable_layer_height_compensation_enabled { false };
    bool m_beam_search_stack_expansion_enabled { false };
    int m_effective_color_prediction_mode { TextureMappingZone::DefaultTopSurfaceContoningColorPredictionMode };
    ColorSolverCalibratedStackModel m_calibrated_stack_model;
    ColorSolverCalibratedStackModel m_nearest_measured_sample_fallback_model;
    int m_nearest_measured_sample_fallback_mode { TextureMappingZone::SlicerDefaultTopSurfaceContoningColorPredictionMode };
    std::vector<ColorSolverStackComponentRole> m_component_roles;
    mutable std::map<int, std::vector<Candidate>> m_candidates_by_depth;
    mutable std::shared_ptr<std::mutex> m_candidate_mutex { std::make_shared<std::mutex>() };
    mutable std::shared_ptr<ColorSolverOrderedStackCandidateCache> m_ordered_candidate_cache { std::make_shared<ColorSolverOrderedStackCandidateCache>() };
    mutable std::shared_ptr<std::mutex> m_ordered_candidate_cache_mutex { std::make_shared<std::mutex>() };
    mutable std::shared_ptr<std::atomic<bool>> m_nearest_measured_sample_upper_fallback_used { std::make_shared<std::atomic<bool>>(false) };
    mutable std::shared_ptr<std::atomic<bool>> m_nearest_measured_sample_lower_fallback_used { std::make_shared<std::atomic<bool>>(false) };
    mutable std::shared_ptr<std::mutex> m_nearest_measured_sample_fallback_issue_mutex { std::make_shared<std::mutex>() };
    mutable std::shared_ptr<std::vector<TextureMappingContoningNearestMeasuredSampleFallbackIssue>> m_nearest_measured_sample_fallback_issues { std::make_shared<std::vector<TextureMappingContoningNearestMeasuredSampleFallbackIssue>>() };
    mutable std::shared_ptr<std::mutex> m_nearest_measured_sample_fallback_area_mutex { std::make_shared<std::mutex>() };
    mutable std::shared_ptr<std::array<TextureMappingContoningNearestMeasuredSampleFallbackArea, 2>> m_nearest_measured_sample_fallback_areas { std::make_shared<std::array<TextureMappingContoningNearestMeasuredSampleFallbackArea, 2>>() };
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
