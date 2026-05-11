#ifndef slic3r_GLGizmoMmuSegmentation_hpp_
#define slic3r_GLGizmoMmuSegmentation_hpp_

#include "GLGizmoPainterBase.hpp"

#include "libslic3r/ImageMapRawFilamentOffsetAtlas.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class wxWindow;

namespace Slic3r {
class ModelObject;
}

namespace Slic3r::GUI {

class GLCanvas3D;

void open_color_data_management_dialog(wxWindow *parent, GLCanvas3D &canvas, ModelObject *object, std::function<void()> on_object_changed = {});

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
    ~GLGizmoMmuSegmentation() override = default;

    void render_painter_gizmo() override;

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
    std::vector<size_t>               m_extruder_remap;      // index → target extruder index
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
    
    // Filament remapping methods
    void remap_filament_assignments();
    void render_filament_remap_ui(float window_width, float max_tooltip_width);
    bool selected_object_has_imported_vertex_colors() const;
    bool selected_object_has_imported_texture_data() const;
    bool selected_object_has_bakeable_image_texture_data() const;
    bool selected_object_has_texture_mapping_color_data() const;
    bool selected_object_has_painted_regions() const;
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
};

class GLGizmoTrueColorPainting : public GLGizmoPainterBase
{
public:
    GLGizmoTrueColorPainting(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id);
    ~GLGizmoTrueColorPainting() override = default;

    void render_painter_gizmo() override;
    bool gizmo_event(SLAGizmoEventType action, const Vec2d& mouse_position, bool shift_down, bool alt_down, bool control_down) override;

    const float get_cursor_radius_min() const override { return CursorRadiusMin; }

protected:
    ColorRGBA get_cursor_hover_color() const override;
    ColorRGBA get_cursor_sphere_left_button_color() const override;
    EnforcerBlockerType get_left_button_state_type() const override { return EnforcerBlockerType::ENFORCER; }
    EnforcerBlockerType get_right_button_state_type() const override { return EnforcerBlockerType(-1); }

    void on_render_input_window(float x, float y, float bottom_limit) override;
    std::string on_get_name() const override;
    bool on_is_selectable() const override;
    bool on_is_activable() const override;

    wxString handle_snapshot_action_name(bool shift_down, Button button_down) const override;

    std::string get_gizmo_entering_text() const override { return "Entering true color painting"; }
    std::string get_gizmo_leaving_text() const override { return "Leaving true color painting"; }
    std::string get_action_snapshot_name() const override { return "True color painting editing"; }

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

    bool on_init() override;
    void update_model_object() override;
    void update_from_model_object(bool first_update = false) override;
    void on_opening() override;
    void on_shutdown() override;
    PainterGizmoType get_painter_type() const override;

    void init_model_triangle_selectors();
    ModelObject *selected_model_object() const;
    void update_selected_object_color_state();
    void open_color_data_management_dialog();
    bool selected_object_has_rgb_data() const;
    bool selected_object_has_imported_color_data() const;
    bool selected_object_has_raw_atlas_texture_data() const;
    void initialize_selected_object_rgb_data();
    void convert_selected_object_vertex_colors_to_rgb_data();
    void convert_selected_object_image_texture_to_rgb_data();
    void refresh_selected_object_after_rgb_change(ModelObject *object);
    void update_triangle_selectors_color();
    bool record_brush_stroke_point(const Vec2d &mouse_position);
    void clear_brush_stroke_points();
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
    bool                 m_selected_can_convert_vertex = false;
    bool                 m_selected_can_convert_image = false;
    bool                 m_selected_has_raw_atlas_texture_data = false;
    bool                 m_color_picker_active = false;
    bool                 m_brush_stroke_active = false;
    ObjectID             m_selected_color_state_object_id;
    std::vector<std::vector<Vec3f>> m_brush_stroke_points_by_volume;
    std::vector<std::unique_ptr<ColorFacetsAnnotation>> m_preview_rgb_data_by_volume;
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

    bool load_projection_image();
    void clear_projection_image();
    bool ensure_overlay_texture();
    OverlayRect overlay_rect() const;
    ModelObject *selected_model_object() const;
    void open_color_data_management_dialog();
    void update_default_projection_mode();
    ProjectionMode default_projection_mode() const;
    bool projection_mode_allowed(ProjectionMode mode) const;
    bool selected_object_has_image_texture_data() const;
    bool selected_object_has_vertex_color_data() const;
    bool selected_object_has_rgb_data() const;
    bool selected_object_has_raw_atlas_texture_data() const;
    bool project_image_to_selected_object();
    bool project_to_vertex_colors(ModelObject *object);
    bool project_to_image_texture(ModelObject *object);
    bool project_to_rgb_data(ModelObject *object);
    void refresh_projected_object(ModelObject *object);

    ProjectionMode       m_projection_mode = ProjectionMode::RGBData;
    bool                 m_projection_mode_initialized = false;
    ObjectID             m_projection_mode_object_id;
    std::string          m_image_path;
    std::string          m_image_error;
    std::vector<uint8_t> m_image_rgba;
    uint32_t             m_image_width = 0;
    uint32_t             m_image_height = 0;
    ImageMapRawFilamentOffsetAtlas m_raw_atlas;
    GLTexture            m_overlay_texture;
    bool                 m_overlay_texture_dirty = false;
    bool                 m_show_overlay = true;
    float                m_projection_opacity = 1.f;
    bool                 m_apply_transparency_as_background = false;
    bool                 m_pass_through_model = false;
    bool                 m_convert_existing_colors_to_raw_offsets = true;
};

} // namespace Slic3r


#endif // slic3r_GLGizmoMmuSegmentation_hpp_
