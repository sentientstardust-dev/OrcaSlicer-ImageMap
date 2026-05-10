#ifndef slic3r_MMUPaintedTexturePreview_hpp_
#define slic3r_MMUPaintedTexturePreview_hpp_

#include "GLModel.hpp"
#include "GLTexture.hpp"

#include "libslic3r/Color.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/TriangleSelector.hpp"

#include <array>
#include <utility>
#include <vector>

namespace Slic3r {

class TextureMappingManager;

bool build_mmu_texture_preview_models(
    const ModelVolume                                                    &model_volume,
    const std::vector<std::vector<TriangleSelector::FacetStateTriangle>> &triangles_per_type,
    const std::vector<ColorRGBA>                                         &state_colors,
    unsigned int                                                          base_filament_id,
    size_t                                                                num_physical,
    const TextureMappingManager                                          *texture_mgr,
    std::vector<GUI::GLModel>                                            &out_models,
    std::vector<ColorRGBA>                                               &out_colors,
    std::vector<unsigned int>                                            &out_filament_ids);

bool build_mmu_vertex_color_preview_models(
    const ModelVolume                                                    &model_volume,
    const std::vector<std::vector<TriangleSelector::FacetStateTriangle>> &triangles_per_type,
    const std::vector<ColorRGBA>                                         &state_colors,
    unsigned int                                                          base_filament_id,
    size_t                                                                num_physical,
    const TextureMappingManager                                          *texture_mgr,
    const Transform3d                                                    &world_matrix,
    std::vector<GUI::GLModel>                                            &out_models,
    std::vector<ColorRGBA>                                               &out_colors,
    std::vector<unsigned int>                                            &out_filament_ids,
    const ColorFacetsAnnotation                                          *texture_mapping_color_facets_override = nullptr,
    size_t                                                                preview_owner_key = 0);

bool build_mmu_vertex_color_preview_models(
    const ModelVolume                                                    &model_volume,
    const std::vector<std::vector<TriangleSelector::FacetStateTriangle>> &triangles_per_type,
    const std::vector<ColorRGBA>                                         &state_colors,
    unsigned int                                                          base_filament_id,
    size_t                                                                num_physical,
    const TextureMappingManager                                          *texture_mgr,
    std::vector<GUI::GLModel>                                            &out_models,
    std::vector<ColorRGBA>                                               &out_colors,
    std::vector<unsigned int>                                            &out_filament_ids,
    const ColorFacetsAnnotation                                          *texture_mapping_color_facets_override = nullptr,
    size_t                                                                preview_owner_key = 0);

size_t model_volume_texture_preview_signature(const ModelVolume &model_volume);
size_t model_volume_texture_mapping_color_preview_signature(const ModelVolume &model_volume);
bool model_volume_has_texture_preview_data(const ModelVolume &model_volume);
bool model_volume_has_complete_texture_preview_data(const ModelVolume &model_volume);
bool model_volume_has_vertex_color_preview_data(const ModelVolume &model_volume);
bool model_volume_has_texture_mapping_color_preview_data(const ModelVolume &model_volume);

bool ensure_model_volume_texture_preview(const ModelVolume &model_volume,
                                         GUI::GLTexture    &texture,
                                         size_t            &texture_signature);

bool texture_preview_simulation_is_pending();
bool texture_preview_simulation_has_temporary_pending();
void clear_texture_preview_simulation_cache();
bool texture_preview_simulation_enabled_for_filament(unsigned int                 filament_id,
                                                     size_t                       num_physical,
                                                     const TextureMappingManager *texture_mgr);
bool texture_preview_simulation_enabled_for_all_filaments(const std::vector<unsigned int> &filament_ids,
                                                          size_t                          num_physical,
                                                          const TextureMappingManager    *texture_mgr);
size_t texture_preview_simulation_generation_signature();

size_t texture_preview_settings_signature(size_t num_physical, const TextureMappingManager *texture_mgr);

void render_model_texture_preview_models(
    std::vector<GUI::GLModel>       &models,
    const std::vector<ColorRGBA>    &colors,
    const std::vector<unsigned int> &filament_ids,
    size_t                           num_physical,
    const TextureMappingManager     *texture_mgr,
    const ModelVolume               &model_volume,
    const GUI::GLTexture            &texture,
    const Transform3d               &model_matrix,
    const Transform3d               &view_matrix,
    const Transform3d               &projection_matrix,
    const std::array<float, 2>      &z_range,
    const std::array<float, 4>      &clipping_plane,
    int                              print_volume_type = -1,
    const std::array<float, 4>      &print_volume_xy = std::array<float, 4>{ 0.f, 0.f, 0.f, 0.f },
    const std::array<float, 2>      &print_volume_z = std::array<float, 2>{ 0.f, 0.f },
    bool                             opaque = false);

void render_model_texture_preview_model(
    GUI::GLModel                    &model,
    const ColorRGBA                 &color,
    unsigned int                     filament_id,
    size_t                           num_physical,
    const TextureMappingManager     *texture_mgr,
    const ModelVolume               &model_volume,
    const GUI::GLTexture            &texture,
    const Transform3d               &model_matrix,
    const Transform3d               &view_matrix,
    const Transform3d               &projection_matrix,
    const std::array<float, 2>      &z_range,
    const std::array<float, 4>      &clipping_plane,
    const std::pair<size_t, size_t> &render_range,
    int                              print_volume_type = -1,
    const std::array<float, 4>      &print_volume_xy = std::array<float, 4>{ 0.f, 0.f, 0.f, 0.f },
    const std::array<float, 2>      &print_volume_z = std::array<float, 2>{ 0.f, 0.f },
    bool                             opaque = false);

void render_model_vertex_color_preview_models(
    std::vector<GUI::GLModel>       &models,
    const std::vector<ColorRGBA>    &colors,
    const std::vector<unsigned int> &filament_ids,
    size_t                           num_physical,
    const TextureMappingManager     *texture_mgr,
    const Transform3d               &model_matrix,
    const Transform3d               &view_matrix,
    const Transform3d               &projection_matrix,
    const std::array<float, 2>      &z_range,
    const std::array<float, 4>      &clipping_plane,
    int                              print_volume_type = -1,
    const std::array<float, 4>      &print_volume_xy = std::array<float, 4>{ 0.f, 0.f, 0.f, 0.f },
    const std::array<float, 2>      &print_volume_z = std::array<float, 2>{ 0.f, 0.f },
    bool                             opaque = false);

} // namespace Slic3r

#endif // slic3r_MMUPaintedTexturePreview_hpp_
