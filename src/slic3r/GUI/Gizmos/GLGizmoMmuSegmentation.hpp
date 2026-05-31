#ifndef slic3r_GLGizmoMmuSegmentation_hpp_
#define slic3r_GLGizmoMmuSegmentation_hpp_

#include "GLGizmoPainterBase.hpp"

#include "libslic3r/ImageMapRawFilamentOffsetAtlas.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class wxWindow;

namespace Slic3r {
class ModelObject;
class ModelVolume;
}

namespace Slic3r::GUI {

class GLCanvas3D;
struct TextureMappingBackgroundConfigSnapshot;

void open_color_data_management_dialog(wxWindow *parent, GLCanvas3D &canvas, ModelObject *object, std::function<void()> on_object_changed = {});

enum class SlopeAutoPaintMode : int
{
    Top,
    Bottom,
    Side
};

struct SlopeAutoPaintSettings
{
    SlopeAutoPaintMode mode = SlopeAutoPaintMode::Top;
    float top_angle_deg = 10.f;
    float bottom_angle_deg = 10.f;
    unsigned int target_filament_id = 1;
    bool override_all = true;
    std::vector<unsigned int> override_filament_ids;
};

struct ColorAutoPaintSettings
{
    ColorRGBA target_color = ColorRGBA(1.f, 1.f, 1.f, 1.f);
    float tolerance_pct = 40.f;
    unsigned int target_filament_id = 1;
    bool override_all = true;
    bool show_result_area = true;
    std::vector<unsigned int> override_filament_ids;
};

class GLMmSegmentationGizmo3DScene
{
public:
    GLMmSegmentationGizmo3DScene() = delete;

    explicit GLMmSegmentationGizmo3DScene(size_t triangle_indices_buffers_count)
    {
    }

    virtual ~GLMmSegmentationGizmo3DScene() { release_geometry(); }

    [[nodiscard]] inline bool has_VBOs(size_t triangle_indices_idx) const
    {
        assert(triangle_indices_idx < this->triangle_patches.size());
        return this->triangle_indices_VBO_ids[triangle_indices_idx] != 0;
    }

    // Release the geometry data, release OpenGL VBOs.
    void release_geometry();
    // Finalize the initialization of the geometry, upload the geometry to OpenGL VBO objects
    // and possibly releasing it if it has been loaded into the VBOs.
    void finalize_vertices();
    // Finalize the initialization of the indices, upload the indices to OpenGL VBO objects
    // and possibly releasing it if it has been loaded into the VBOs.
    void finalize_triangle_indices();

    void clear()
    {
        this->vertices.clear();
        // BBS
        this->triangle_indices_VBO_ids.clear();
        this->triangle_indices_sizes.clear();

        for (TrianglePatch& patch : this->triangle_patches)
            patch.triangle_indices.clear();
        this->triangle_patches.clear();
    }

    void render(size_t triangle_indices_idx) const;

    std::vector<float>            vertices;
    //std::vector<std::vector<int>> triangle_indices;

    // BBS
    std::vector<TrianglePatch>    triangle_patches;

    // When the triangle indices are loaded into the graphics card as Vertex Buffer Objects,
    // the above mentioned std::vectors are cleared and the following variables keep their original length.
    std::vector<size_t> triangle_indices_sizes;

    // IDs of the Vertex Array Objects, into which the geometry has been loaded.
    // Zero if the VBOs are not sent to GPU yet.
    unsigned int              vertices_VBO_id{0};
    std::vector<unsigned int> triangle_indices_VBO_ids;
};

class GLGizmoMmuSegmentation : public GLGizmoPainterBase
{
public:
    GLGizmoMmuSegmentation(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id);
    ~GLGizmoMmuSegmentation() override;

    void render_painter_gizmo() override;
    bool gizmo_event(SLAGizmoEventType action, const Vec2d& mouse_position, bool shift_down, bool alt_down, bool control_down) override;

    void data_changed(bool is_serializing) override;

    // Keep this in sync with the shared triangle-selector state range.
    static const constexpr size_t EXTRUDERS_LIMIT = static_cast<size_t>(EnforcerBlockerType::ExtruderMax);

    const float get_cursor_radius_min() const override { return CursorRadiusMin; }

    // BBS
    bool on_number_key_down(int number);
    bool on_key_down_select_tool_type(int keyCode);

protected:
    // BBS
    ColorRGBA get_cursor_hover_color() const override;
    void on_set_state() override;

    EnforcerBlockerType get_left_button_state_type() const override
    {
        if (m_selected_extruder_idx < m_display_filament_ids.size())
            return EnforcerBlockerType(m_display_filament_ids[m_selected_extruder_idx]);
        return EnforcerBlockerType::Extruder1;
    }
    EnforcerBlockerType get_right_button_state_type() const override { return EnforcerBlockerType(-1); }

    void on_render_input_window(float x, float y, float bottom_limit) override;
    std::string on_get_name() const override;
    void show_tooltip_information(float caption_max, float x, float y);
    bool on_is_selectable() const override;
    bool on_is_activable() const override;

    wxString handle_snapshot_action_name(bool shift_down, Button button_down) const override;

    std::string get_gizmo_entering_text() const override { return "Entering color region painting"; }
    std::string get_gizmo_leaving_text() const override { return "Leaving color region painting"; }
    std::string get_action_snapshot_name() const override { return "Color region painting editing"; }

    // BBS
    size_t                            m_selected_extruder_idx = 0;
    std::vector<ColorRGBA>            m_extruders_colors;
    std::vector<unsigned int>         m_display_filament_ids;
    std::vector<int>                  m_volumes_extruder_idxs;

    // BBS
    wchar_t                           m_current_tool = 0;
    bool                              m_detect_geometry_edge = true;
    
    // Filament remap feature
    std::vector<unsigned int>         m_remap_source_filament_ids;
    std::vector<size_t>               m_extruder_remap;
    bool                              m_show_filament_remap_ui = false;

    static const constexpr float      CursorRadiusMin = 0.1f; // cannot be zero

private:
    bool on_init() override;

    // BBS. remove const.
    void update_model_object() override;
    //BBS: add logic to distinguish the first_time_update and later_update
    void update_from_model_object(bool first_update = false) override;
    void tool_changed(wchar_t old_tool, wchar_t new_tool);

    void on_opening() override;
    void on_shutdown() override;
    PainterGizmoType get_painter_type() const override;

    void init_model_triangle_selectors();

    // BBS
    void update_triangle_selectors_colors();
    void init_extruders_data();
    void init_extruders_data(const std::vector<ColorRGBA> &extruder_colors);
    void open_slope_auto_paint_overlay();
    void render_slope_auto_paint_overlay();
    void update_slope_auto_paint_preview(const SlopeAutoPaintSettings &settings);
    void clear_slope_auto_paint_preview();
    bool apply_slope_auto_paint(const SlopeAutoPaintSettings &settings, bool preview);
    void set_render_triangle_slope_uniforms(GLShaderProgram *shader, const ModelVolume *model_volume, const Matrix3f &normal_matrix) const override;
    bool should_render_triangle_texture_preview() const override;
    bool color_auto_paint_shader_preview_active() const;
    void update_color_auto_paint_shader_preview_settings();
    void open_color_auto_paint_overlay();
    void render_color_auto_paint_overlay();
    bool pick_color_auto_paint_source_from_model(const Vec2d &mouse_position);
    bool sample_color_auto_paint_source_from_model(const Vec2d &mouse_position, ColorRGBA &color) const;
    void set_color_auto_paint_target_color(const ColorRGBA &color);
    void set_color_auto_paint_picker_active(bool active);
    void refresh_color_auto_paint_common_colors();
    void update_color_auto_paint_preview(const ColorAutoPaintSettings &settings);
    void start_color_auto_paint_preview_worker();
    void request_color_auto_paint_preview_worker_cancel();
    void finish_color_auto_paint_preview_update(uint64_t generation,
                                                const ColorAutoPaintSettings &settings,
                                                bool active,
                                                std::vector<std::unique_ptr<TriangleSelectorPatch>> &&selectors);
    void cancel_color_auto_paint_preview_worker();
    void clear_color_auto_paint_preview();
    bool apply_color_auto_paint(const ColorAutoPaintSettings &settings, bool preview);
    bool apply_color_auto_paint_to_selector(const ModelObject &object,
                                            const ModelVolume &volume,
                                            TriangleSelector &selector,
                                            const Transform3d &trafo_no_translate,
                                            const ColorAutoPaintSettings &settings,
                                            bool clear_non_matching,
                                            const std::function<void(size_t, size_t)> &progress = {},
                                            const std::function<bool()> &cancel = {}) const;
    void render_extra_triangle_overlays(int mesh_id,
                                        const Transform3d &matrix,
                                        const Transform3d &view_matrix,
                                        const Transform3d &projection_matrix,
                                        const std::array<float, 2> &z_range,
                                        const std::array<float, 4> &clipping_plane) const override;
    
    // Filament remapping methods
    std::vector<unsigned int> selected_object_used_filament_ids() const;
    void refresh_filament_remap_sources(bool reset_targets = false);
    void remap_filament_assignments();
    void render_filament_remap_ui(float window_width, float max_tooltip_width);
    bool selected_object_has_imported_vertex_colors() const;
    bool selected_object_has_imported_texture_data() const;
    bool selected_object_has_bakeable_image_texture_data() const;
    bool selected_object_has_texture_mapping_color_data() const;
    bool selected_object_has_painted_regions() const;
    bool selected_object_has_color_auto_paint_source() const;
    void open_obj_vertex_color_mapping_dialog();
    void bake_selected_object_image_texture_to_vertex_colors();
    void convert_selected_object_vertex_colors_to_texture_mapping_colors();
    void convert_selected_object_image_texture_to_texture_mapping_colors();
    void convert_selected_regions_to_vertex_colors();
    void convert_selected_regions_to_image_texture();
    void convert_selected_regions_to_rgba_data();
    void finish_selected_regions_color_data_conversion(ModelObject &object);
    void clear_selected_object_image_texture_data();
    void clear_selected_object_texture_mapping_color_data();

    // This map holds all translated description texts, so they can be easily referenced during layout calculations
    // etc. When language changes, GUI is recreated and this class constructed again, so the change takes effect.
    std::map<std::string, wxString> m_desc;
    SlopeAutoPaintSettings m_slope_auto_paint_settings;
    SlopeAutoPaintSettings m_slope_auto_paint_preview_settings;
    bool m_show_slope_auto_paint_overlay = false;
    bool m_slope_auto_paint_overlay_positioned = false;
    bool m_slope_auto_paint_preview_active = false;
    ColorAutoPaintSettings m_color_auto_paint_settings;
    ColorAutoPaintSettings m_color_auto_paint_preview_settings;
    bool m_show_color_auto_paint_overlay = false;
    bool m_color_auto_paint_overlay_positioned = false;
    bool m_color_auto_paint_preview_active = false;
    bool m_color_auto_paint_picker_active = false;
    bool m_color_auto_paint_picker_restore_show_result_area = false;
    bool m_color_auto_paint_preview_update_pending = false;
    bool m_color_auto_paint_preview_worker_running = false;
    std::atomic<int> m_color_auto_paint_preview_progress_pct { 0 };
    std::atomic_bool m_color_auto_paint_preview_worker_cancel { false };
    uint64_t m_color_auto_paint_preview_generation = 0;
    ColorAutoPaintSettings m_color_auto_paint_requested_settings;
    std::thread m_color_auto_paint_preview_thread;
    std::vector<std::unique_ptr<TriangleSelectorPatch>> m_color_auto_paint_preview_selectors;
    std::vector<ColorRGBA> m_color_auto_paint_common_colors;
};

class GLGizmoTrueColorPainting : public GLGizmoPainterBase
{
public:
    GLGizmoTrueColorPainting(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id);
    ~GLGizmoTrueColorPainting() override;

    void render_painter_gizmo() override;
    bool gizmo_event(SLAGizmoEventType action, const Vec2d& mouse_position, bool shift_down, bool alt_down, bool control_down) override;

    const float get_cursor_radius_min() const override { return CursorRadiusMin; }

protected:
    ColorRGBA get_cursor_hover_color() const override;
    ColorRGBA get_cursor_sphere_left_button_color() const override;
    EnforcerBlockerType get_left_button_state_type() const override { return EnforcerBlockerType::ENFORCER; }
    EnforcerBlockerType get_right_button_state_type() const override { return EnforcerBlockerType(-1); }
    bool render_triangle_texture_preview_before_selector() const override;
    void render_extra_triangle_overlays(int mesh_id,
                                        const Transform3d &matrix,
                                        const Transform3d &view_matrix,
                                        const Transform3d &projection_matrix,
                                        const std::array<float, 2> &z_range,
                                        const std::array<float, 4> &clipping_plane) const override;

    void on_render_input_window(float x, float y, float bottom_limit) override;
    std::string on_get_name() const override;
    bool on_is_selectable() const override;
    bool on_is_activable() const override;

    wxString handle_snapshot_action_name(bool shift_down, Button button_down) const override;

    std::string get_gizmo_entering_text() const override { return "Entering RGB color painting"; }
    std::string get_gizmo_leaving_text() const override { return "Leaving RGB color painting"; }
    std::string get_action_snapshot_name() const override { return "RGB color painting editing"; }

private:
    enum class ColorInputMode : int
    {
        FilamentColors,
        RGB,
        CMY,
        CMYK,
        CMYW,
        RGBK,
        RGBW,
        BW,
        CMYKW,
        RGBKW
    };

    enum class PaintingStorageMode : int
    {
        ImageTexture,
        RgbaData
    };

    bool on_init() override;
    void update_model_object() override;
    void update_from_model_object(bool first_update = false) override;
    void on_opening() override;
    bool on_before_shutdown() override;
    void on_shutdown() override;
    void on_set_state() override;
    PainterGizmoType get_painter_type() const override;
    void on_brush_projected_mouse_positions(SLAGizmoEventType action,
                                            int mesh_idx,
                                            const std::vector<ProjectedMousePosition> &projected_mouse_positions) override;

    void init_model_triangle_selectors();
    ModelObject *selected_model_object() const;
    void update_selected_object_color_state();
    void open_color_data_management_dialog();
    PaintingStorageMode active_painting_storage_mode() const;
    bool selected_object_has_rgb_data() const;
    bool selected_object_has_image_texture_data() const;
    bool selected_object_has_imported_color_data() const;
    bool selected_object_has_raw_atlas_texture_data() const;
    void initialize_selected_object_rgb_data();
    void convert_selected_object_vertex_colors_to_rgb_data();
    void convert_selected_object_image_texture_to_rgb_data();
    void convert_selected_object_rgba_data_to_image_texture();
    void erase_selected_object_rgba_data_for_image_texture_painting();
    void refresh_selected_object_after_rgb_change(ModelObject *object);
    void refresh_selected_object_after_image_texture_change(ModelObject *object, bool rebuild_selectors = true);
    void remember_changed_rgb_data_object(ModelObject *object);
    void backup_changed_rgb_data_objects();
    void update_triangle_selectors_color();
    void update_rgb_data_preview_conversion();
    void start_rgb_data_preview_conversion(ModelObject &object);
    void cancel_rgb_data_preview_conversion();
    void start_image_texture_paint_worker();
    void cancel_image_texture_paint_worker();
    void process_completed_image_texture_paint_strokes();
    void clear_pending_image_texture_paint_previews();
    bool has_pending_image_texture_paint_strokes();
    bool rgb_data_preview_conversion_pending_for_selected_object() const;
    ColorFacetsAnnotation *preview_rgb_data_for_volume(const ModelVolume &volume) const;
    bool append_brush_stroke_point(int mesh_idx, const Vec3f &hit, const Transform3d &world_matrix);
    void clear_brush_stroke_points();
    void clear_rgb_preview_cache();
    void render_cached_rgb_data_preview(const ModelObject &object, const Selection &selection);
    bool pick_color_from_model(const Vec2d &mouse_position);
    bool sample_color_from_model(const Vec2d &mouse_position, ColorRGBA &color) const;
    void set_active_color_from_sample(const ColorRGBA &color);
    void sync_cmy_from_rgb();
    void sync_rgb_from_cmy();
    void sync_cmyk_from_rgb();
    void sync_rgb_from_cmyk();
    void sync_cmyw_from_rgb();
    void sync_rgb_from_cmyw();
    void sync_rgbk_from_rgb();
    void sync_rgb_from_rgbk();
    void sync_rgbw_from_rgb();
    void sync_rgb_from_rgbw();
    void sync_cmykw_from_rgb();
    void sync_rgb_from_cmykw();
    void sync_rgbkw_from_rgb();
    void sync_rgb_from_rgbkw();
    void sync_bw_from_rgb();
    void sync_rgb_from_bw();
    void ensure_filament_mix_colors();
    void sync_filament_mix_from_rgb();
    void sync_rgb_from_filament_mix();
    void sync_active_color_mode_from_rgb(bool update_filament_mix);
    bool render_color_input_mode_combo();
    bool render_brush_color_picker(const char *id);
    bool render_rgb_picker(float item_width);
    bool render_cmy_picker(float item_width);
    bool render_cmyk_picker(float item_width);
    bool render_cmyw_picker(float item_width);
    bool render_rgbk_picker(float item_width);
    bool render_rgbw_picker(float item_width);
    bool render_cmykw_picker(float item_width);
    bool render_rgbkw_picker(float item_width);
    bool render_bw_picker(float item_width);
    bool render_filament_colors_picker(float item_width);
    void render_background_color_picker(float max_tooltip_width);
    void clear_selected_object_background_color();
    void refresh_selected_object_after_background_color_change(ModelObject *object);

    std::array<float, 4> m_rgb_color { 0.f, 0.f, 0.f, 1.f };
    std::array<float, 3> m_cmy_color { 1.f, 1.f, 1.f };
    std::array<float, 4> m_cmyk_color { 0.f, 0.f, 0.f, 1.f };
    std::array<float, 4> m_cmyw_color { 0.f, 0.f, 0.f, 1.f };
    std::array<float, 4> m_rgbk_color { 0.f, 0.f, 0.f, 1.f };
    std::array<float, 4> m_rgbw_color { 0.f, 0.f, 0.f, 1.f };
    std::array<float, 2> m_bw_color { 1.f, 0.f };
    std::array<float, 5> m_cmykw_color { 0.f, 0.f, 0.f, 0.f, 1.f };
    std::array<float, 5> m_rgbkw_color { 0.f, 0.f, 0.f, 0.f, 1.f };
    std::vector<ColorRGBA> m_filament_mix_colors;
    std::vector<float>  m_filament_mix;
    float                m_brush_hardness = 1.f;
    float                m_opacity = 1.f;
    ColorInputMode       m_color_input_mode = ColorInputMode::FilamentColors;
    bool                 m_selected_has_rgb_data = false;
    bool                 m_selected_has_imported_color_data = false;
    bool                 m_selected_has_image_texture_data = false;
    bool                 m_selected_can_convert_vertex = false;
    bool                 m_selected_can_convert_image = false;
    bool                 m_selected_has_raw_atlas_texture_data = false;
    PaintingStorageMode  m_painting_storage_mode = PaintingStorageMode::ImageTexture;
    bool                 m_color_picker_active = false;
    bool                 m_brush_stroke_active = false;
    bool                 m_live_selector_preview_active = false;
    ObjectID             m_selected_color_state_object_id;
    std::vector<ObjectID> m_changed_rgb_data_object_ids;
    std::vector<std::vector<Vec3f>> m_brush_stroke_points_by_volume;
    struct RgbPreviewVolumeCache
    {
        ObjectID volume_id;
        bool valid = false;
        size_t signature = 0;
        std::vector<GLModel> models;
        std::vector<ColorRGBA> colors;
        std::vector<unsigned int> filament_ids;
    };
    std::vector<RgbPreviewVolumeCache> m_rgb_preview_cache;
    std::vector<ObjectID> m_preview_rgb_data_volume_ids;
    std::vector<std::unique_ptr<ColorFacetsAnnotation>> m_preview_rgb_data_by_volume;
    std::unique_ptr<TextureMappingBackgroundConfigSnapshot> m_background_color_edit_config_snapshot;
    struct RgbDataConversionState;
    std::shared_ptr<RgbDataConversionState> m_rgb_data_conversion_state;
    std::thread          m_rgb_data_conversion_thread;
    uint64_t             m_rgb_data_conversion_generation = 0;
    ObjectID             m_rgb_data_conversion_object_id;
    struct ImageTexturePaintTask;
    struct ImageTexturePaintResult;
    struct PendingImageTexturePaintPreview;
    static std::shared_ptr<ImageTexturePaintResult> compute_image_texture_paint_task(const ImageTexturePaintTask &task);
    static std::unique_ptr<PendingImageTexturePaintPreview> build_pending_image_texture_paint_preview(const ImageTexturePaintTask &task);
    static bool apply_image_texture_paint_result_to_volume(ModelVolume &volume, const ImageTexturePaintResult &result);
    std::mutex m_image_texture_paint_mutex;
    std::condition_variable m_image_texture_paint_cv;
    std::deque<std::shared_ptr<ImageTexturePaintTask>> m_image_texture_paint_tasks;
    std::deque<std::shared_ptr<ImageTexturePaintResult>> m_image_texture_paint_results;
    std::shared_ptr<std::atomic_bool> m_image_texture_paint_cancel;
    std::thread m_image_texture_paint_thread;
    bool m_image_texture_paint_stop = false;
    bool m_image_texture_paint_worker_running = false;
    bool m_image_texture_paint_task_active = false;
    uint64_t m_next_image_texture_paint_sequence = 1;
    uint64_t m_next_image_texture_paint_apply_sequence = 1;
    std::vector<std::unique_ptr<PendingImageTexturePaintPreview>> m_pending_image_texture_paint_previews;
    struct ColorPickerVolumeSourceCache
    {
        ObjectID volume_id;
        ObjectBase::Timestamp timestamp = 0;
        std::vector<ColorFacetTriangle>              rgb_facets;
        std::unordered_map<int, std::vector<size_t>> rgb_by_source_triangle;
    };
    const ColorPickerVolumeSourceCache &cached_volume_color_source(const ModelVolume &volume) const;
    mutable std::vector<ColorPickerVolumeSourceCache> m_color_picker_source_cache;

    static const constexpr float CursorRadiusMin = 0.1f;
};

class GLGizmoImageProjection : public GLGizmoBase
{
public:
    GLGizmoImageProjection(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id);
    ~GLGizmoImageProjection() override = default;

protected:
    bool on_init() override;
    void on_load(cereal::BinaryInputArchive& ar) override;
    void on_save(cereal::BinaryOutputArchive& ar) const override;
    void on_render() override;
    void on_render_input_window(float x, float y, float bottom_limit) override;
    std::string on_get_name() const override;
    void on_set_state() override;
    bool on_is_selectable() const override;
    bool on_is_activable() const override;
    CommonGizmosDataID on_get_requirements() const override;

private:
    enum class ProjectionMode : int
    {
        VertexColors,
        ImageTexture,
        RGBData
    };

    struct OverlayRect
    {
        float left = 0.f;
        float top = 0.f;
        float width = 0.f;
        float height = 0.f;
    };

    struct ProjectionInput;

    bool load_projection_image();
    void clear_projection_image();
    void ensure_text_font_names();
    void mark_text_projection_dirty();
    bool ensure_text_projection_image();
    bool update_text_raw_atlas(const std::vector<uint8_t> &glyph_mask);
    void render_text_projection_controls(float max_tooltip_width);
    bool render_projection_action_controls();
    void render_projection_overlay();
    void show_projection_overlay();
    bool ensure_overlay_texture();
    OverlayRect overlay_rect() const;
    const std::vector<uint8_t> &active_projection_rgba() const;
    uint32_t active_projection_width() const;
    uint32_t active_projection_height() const;
    bool active_projection_empty() const;
    bool active_raw_atlas_valid() const;
    const ImageMapRawFilamentOffsetAtlas &active_raw_atlas() const;
    bool effective_apply_transparency_as_background() const;
    ModelObject *selected_model_object() const;
    void open_color_data_management_dialog();
    void update_default_projection_mode();
    ProjectionMode default_projection_mode() const;
    bool projection_mode_allowed(ProjectionMode mode) const;
    bool selected_object_has_image_texture_data() const;
    bool selected_object_has_vertex_color_data() const;
    bool selected_object_has_rgb_data() const;
    bool selected_object_has_raw_atlas_texture_data() const;
    bool start_projection_job();
    bool project_image_to_selected_object();
    void fill_projection_input(ProjectionInput &input, const ModelObject *object) const;
    bool project_image_to_object(ModelObject *object, unsigned int texture_mapping_filament_id, const ProjectionInput &input, const std::function<void(int)> &progress_fn = {}, const std::function<void()> &check_cancel = {});
    bool project_to_vertex_colors(ModelObject *object, const ProjectionInput &input, const std::function<void(int)> &progress_fn = {}, const std::function<void()> &check_cancel = {});
    bool project_to_image_texture(ModelObject *object, const ProjectionInput &input, const std::function<void(int)> &progress_fn = {}, const std::function<void()> &check_cancel = {});
    bool project_to_rgb_data(ModelObject *object, const ProjectionInput &input, const std::function<void(int)> &progress_fn = {}, const std::function<void()> &check_cancel = {});
    void render_projection_progress();
    void request_projection_job_cancel();
    void refresh_projected_object(ModelObject *object);
    void remember_projected_object(ModelObject *object);
    void backup_projected_objects();

    ProjectionMode       m_projection_mode = ProjectionMode::RGBData;
    bool                 m_projection_mode_initialized = false;
    ObjectID             m_projection_mode_object_id;
    std::string          m_image_path;
    std::string          m_image_error;
    std::vector<uint8_t> m_image_rgba;
    uint32_t             m_image_width = 0;
    uint32_t             m_image_height = 0;
    bool                 m_text_mode = false;
    std::string          m_projection_text = "Text";
    std::vector<std::string> m_text_font_names;
    int                  m_text_font_idx = 0;
    std::string          m_text_font_name;
    float                m_text_font_size = 200.f;
    std::array<float, 3> m_text_color { 91.f / 255.f, 102.f / 255.f, 1.f };
    std::array<float, 3> m_text_background_color { 1.f, 1.f, 1.f };
    bool                 m_text_background_transparent = true;
    bool                 m_text_capitalize = true;
    bool                 m_text_dirty = true;
    std::string          m_text_error;
    std::vector<uint8_t> m_text_rgba;
    uint32_t             m_text_width = 0;
    uint32_t             m_text_height = 0;
    ImageMapRawFilamentOffsetAtlas m_text_raw_atlas;
    ObjectID             m_text_projection_object_id;
    bool                 m_text_projection_raw_mode = false;
    ImageMapRawFilamentOffsetAtlas m_raw_atlas;
    GLTexture            m_overlay_texture;
    bool                 m_overlay_texture_dirty = false;
    bool                 m_show_overlay = true;
    float                m_projection_opacity = 1.f;
    bool                 m_projection_opacity_reset_active = false;
    double               m_projection_opacity_last_click_time = -1.0;
    float                m_projection_rotation_deg = 0.f;
    bool                 m_projection_rotation_reset_active = false;
    double               m_projection_rotation_last_click_time = -1.0;
    bool                 m_projection_section_reset_active = false;
    double               m_projection_section_last_click_time = -1.0;
    bool                 m_projection_panel_expanded = true;
    bool                 m_apply_transparency_as_background = false;
    bool                 m_pass_through_model = false;
    bool                 m_improve_projection_accuracy = true;
    bool                 m_erase_region_painting = true;
    bool                 m_convert_existing_colors_to_raw_offsets = true;
    std::atomic_bool     m_projection_job_active { false };
    std::atomic_bool     m_projection_job_cancel_requested { false };
    std::atomic_int      m_projection_job_progress { 0 };
    std::vector<ObjectID> m_projected_object_ids;
};

} // namespace Slic3r


#endif // slic3r_GLGizmoMmuSegmentation_hpp_
