#include "GLGizmoMmuSegmentation.hpp"

#include "slic3r/GUI/GLCanvas3D.hpp"
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/ImGuiWrapper.hpp"
#include "slic3r/GUI/Camera.hpp"
#include "slic3r/GUI/Plater.hpp"
#include "slic3r/GUI/BitmapCache.hpp"
#include "slic3r/GUI/format.hpp"
#include "slic3r/GUI/GUI_ObjectList.hpp"
#include "slic3r/GUI/NotificationManager.hpp"
#include "slic3r/GUI/GUI.hpp"
#include "slic3r/GUI/MainFrame.hpp"
#include "slic3r/GUI/ObjColorDialog.hpp"
#include "slic3r/GUI/Tab.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/TextureMapping.hpp"
#include "slic3r/Utils/UndoRedo.hpp"
#include "GLGizmoUtils.hpp"


#include <glad/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace Slic3r::GUI {

static inline void show_notification_extruders_limit_exceeded()
{
    wxGetApp()
        .plater()
        ->get_notification_manager()
        ->push_notification(NotificationType::MmSegmentationExceededExtrudersLimit, NotificationManager::NotificationLevel::PrintInfoNotificationLevel,
                            GUI::format(_L("Filament count exceeds the maximum number that painting tool supports. Only the "
                                           "first %1% filaments will be available in painting tool."), GLGizmoMmuSegmentation::EXTRUDERS_LIMIT));
}

static unsigned int ensure_texture_mapping_zone()
{
    if (wxGetApp().preset_bundle == nullptr || wxGetApp().plater() == nullptr)
        return 0;

    TextureMappingManager &mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    const size_t num_physical = static_cast<size_t>(std::max(wxGetApp().filaments_cnt(), 0));
    std::vector<std::string> physical_colors = wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, false);
    physical_colors.resize(num_physical, "#26A69A");

    if (unsigned int existing_id = mgr.find_image_texture_zone_id(num_physical); existing_id != 0) {
        if (TextureMappingZone *zone = mgr.zone_from_id(existing_id);
            zone == nullptr || !TextureMappingManager::auto_adjust_texture_component_ids(*zone, num_physical, physical_colors)) {
            return existing_id;
        }
    } else if (num_physical < 2) {
        return 0;
    } else {
        mgr.ensure_image_texture_zone(num_physical, physical_colors);
    }

    const std::string texture_serialized = mgr.serialize_entries();
    DynamicPrintConfig *print_cfg = &wxGetApp().preset_bundle->prints.get_edited_preset().config;
    if (ConfigOptionString *opt = print_cfg->option<ConfigOptionString>("texture_mapping_definitions"))
        opt->value = texture_serialized;
    else
        print_cfg->set_key_value("texture_mapping_definitions", new ConfigOptionString(texture_serialized));

    if (ConfigOptionString *opt = wxGetApp().preset_bundle->project_config.option<ConfigOptionString>("texture_mapping_definitions"))
        opt->value = texture_serialized;
    else
        wxGetApp().preset_bundle->project_config.set_key_value("texture_mapping_definitions", new ConfigOptionString(texture_serialized));

    wxGetApp().sidebar().update_texture_mapping_panel(false);
    wxGetApp().sidebar().update_dynamic_filament_list();
    if (auto *print_tab = wxGetApp().get_tab(Preset::TYPE_PRINT))
        print_tab->update_dirty();
    if (wxGetApp().mainframe != nullptr)
        wxGetApp().mainframe->on_config_changed(print_cfg);

    return mgr.find_image_texture_zone_id(num_physical);
}

static bool model_volume_has_imported_image_texture_data(const ModelVolume *volume)
{
    return volume != nullptr &&
           !volume->imported_texture_rgba.empty() &&
           volume->imported_texture_width > 0 &&
           volume->imported_texture_height > 0;
}

static bool model_volume_has_bakeable_image_texture_data(const ModelVolume *volume)
{
    if (!model_volume_has_imported_image_texture_data(volume))
        return false;

    const indexed_triangle_set &its = volume->mesh().its;
    return !its.vertices.empty() &&
           !its.indices.empty() &&
           volume->imported_texture_uv_valid.size() == its.indices.size() &&
           volume->imported_texture_uvs_per_face.size() >= its.indices.size() * 6 &&
           volume->imported_texture_rgba.size() >= size_t(volume->imported_texture_width) * size_t(volume->imported_texture_height) * 4 &&
           std::any_of(volume->imported_texture_uv_valid.begin(), volume->imported_texture_uv_valid.end(), [](uint8_t valid) {
               return valid != 0;
           });
}

static float wrap_texture_uv_for_vertex_bake(float uv)
{
    if (!std::isfinite(uv))
        return 0.f;

    float wrapped = uv - std::floor(uv);
    if (wrapped < 0.f)
        wrapped += 1.f;
    return wrapped;
}

static ColorRGBA sample_texture_rgba_for_vertex_bake(const std::vector<uint8_t> &rgba,
                                                     uint32_t                    width,
                                                     uint32_t                    height,
                                                     const Vec2f                &uv)
{
    if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    const float u = wrap_texture_uv_for_vertex_bake(uv.x());
    const float v = wrap_texture_uv_for_vertex_bake(uv.y());
    const float x = u * float(width > 1 ? width - 1 : 0);
    const float y = v * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto sample_channel = [&rgba, width](size_t sx, size_t sy, size_t channel) {
        const size_t idx = (sy * size_t(width) + sx) * 4 + channel;
        return float(rgba[idx]) / 255.f;
    };
    auto blend_channel = [&](size_t channel) {
        const float c00 = sample_channel(x0, y0, channel);
        const float c10 = sample_channel(x1, y0, channel);
        const float c01 = sample_channel(x0, y1, channel);
        const float c11 = sample_channel(x1, y1, channel);
        const float cx0 = c00 + (c10 - c00) * tx;
        const float cx1 = c01 + (c11 - c01) * tx;
        return std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 1.f);
    };

    return ColorRGBA(blend_channel(0), blend_channel(1), blend_channel(2), 1.f);
}

static uint32_t pack_vertex_color_rgba(const ColorRGBA &color)
{
    auto to_u8 = [](float value) -> uint32_t {
        return uint32_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    const uint32_t r = to_u8(color.r());
    const uint32_t g = to_u8(color.g());
    const uint32_t b = to_u8(color.b());
    const uint32_t a = to_u8(color.a());
    return (r << 24) | (g << 16) | (b << 8) | a;
}

static bool barycentric_weights_for_region_vertex_colors(const Vec3f &point,
                                                         const Vec3f &p0,
                                                         const Vec3f &p1,
                                                         const Vec3f &p2,
                                                         Vec3f       &weights)
{
    const Vec3f edge_0 = p1 - p0;
    const Vec3f edge_1 = p2 - p0;
    const Vec3f delta = point - p0;
    const float d00 = edge_0.dot(edge_0);
    const float d01 = edge_0.dot(edge_1);
    const float d11 = edge_1.dot(edge_1);
    const float d20 = delta.dot(edge_0);
    const float d21 = delta.dot(edge_1);
    const float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) <= EPSILON)
        return false;

    weights.y() = (d11 * d20 - d01 * d21) / denom;
    weights.z() = (d00 * d21 - d01 * d20) / denom;
    weights.x() = 1.f - weights.y() - weights.z();
    return std::isfinite(weights.x()) && std::isfinite(weights.y()) && std::isfinite(weights.z());
}

void GLGizmoMmuSegmentation::on_opening()
{
    if (wxGetApp().filaments_cnt() > int(GLGizmoMmuSegmentation::EXTRUDERS_LIMIT))
        show_notification_extruders_limit_exceeded();
}

void GLGizmoMmuSegmentation::on_shutdown()
{
    m_parent.use_slope(false);
    m_parent.toggle_model_objects_visibility(true);
}

std::string GLGizmoMmuSegmentation::on_get_name() const
{
    return _u8L("Color Painting");
}

bool GLGizmoMmuSegmentation::on_is_selectable() const
{
    return (wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF
            && /*wxGetApp().get_mode() != comSimple && */wxGetApp().filaments_cnt() > 1);
}

bool GLGizmoMmuSegmentation::on_is_activable() const
{
    const Selection& selection = m_parent.get_selection();
    return !selection.is_empty() && (selection.is_single_full_instance() || selection.is_any_volume()) && wxGetApp().filaments_cnt() > 1;
}

static std::vector<int> get_extruder_id_for_volumes(const ModelObject &model_object)
{
    std::vector<int> extruders_idx;
    extruders_idx.reserve(model_object.volumes.size());
    for (const ModelVolume *model_volume : model_object.volumes) {
        if (!model_volume->is_model_part())
            continue;

        extruders_idx.emplace_back(model_volume->extruder_id());
    }

    return extruders_idx;
}

void GLGizmoMmuSegmentation::init_extruders_data()
{
    m_extruders_colors      = wxGetApp().plater()->get_extruders_colors();
    m_selected_extruder_idx = 0;

    // keep remap table consistent with current extruder count
    m_extruder_remap.resize(m_extruders_colors.size());
    for (size_t i = 0; i < m_extruder_remap.size(); ++i)
        m_extruder_remap[i] = i;
}

bool GLGizmoMmuSegmentation::on_init()
{
    // BBS
    m_shortcut_key = WXK_CONTROL_N;

    const wxString ctrl  = GUI::shortkey_ctrl_prefix();
    const wxString alt   = GUI::shortkey_alt_prefix();
    const wxString shift = GUI::shortkey_shift_prefix();

    m_desc["clipping_of_view"] = _L("Section view");
    m_desc["reset_direction"]  = _L("Reset direction");
    m_desc["cursor_size"]      = _L("Brush size");
    m_desc["cursor_type"]      = _L("Brush shape");
    m_desc["paint"]            = _L("Paint");
    m_desc["erase"]            = _L("Erase");
    m_desc["shortcut_key"]     = _L("Choose filament");
    m_desc["edge_detection"]   = _L("Edge detection");
    m_desc["gap_area"]         = _L("Gap area");
    m_desc["perform"]          = _L("Perform");
    m_desc["remove_all"]       = _L("Erase all painting");
    m_desc["circle"]           = _L("Circle");
    m_desc["sphere"]           = _L("Sphere");
    m_desc["pointer"]          = _L("Triangles");
    m_desc["filaments"]        = _L("Filaments");
    m_desc["tool_type"]        = _L("Tool type");
    m_desc["tool_brush"]       = _L("Brush");
    m_desc["tool_smart_fill"]  = _L("Smart fill");
    m_desc["tool_bucket_fill"] = _L("Bucket fill");
    m_desc["smart_fill_angle"] = _L("Smart fill angle");
    m_desc["height_range"]     = _L("Height range");
    m_desc["toggle_wireframe"] = _L("Toggle Wireframe");
    m_desc["perform_remap"]    = _L("Remap filaments");
    m_desc["remap"]            = _L("Remap");
    m_desc["cancel_remap"]     = _L("Cancel");

    std::pair<wxString, wxString> paint_shortcut            = {_L("Left mouse button"),         m_desc["paint"]};
    std::pair<wxString, wxString> erase_shortcut            = {shift + _L("Left mouse button"), m_desc["erase"]};
    std::pair<wxString, wxString> clipping_shortcut         = {alt + _L("Mouse wheel"),         m_desc["clipping_of_view"]};
    std::pair<wxString, wxString> toggle_wireframe_shortcut = {alt + shift + _L("Enter"),       m_desc["toggle_wireframe"]};

    m_shortcuts_brush = {
        paint_shortcut,
        erase_shortcut,
        {ctrl + _L("Mouse wheel"), m_desc["cursor_size"]},
        clipping_shortcut,
        toggle_wireframe_shortcut
    };

    m_shortcuts_bucket_fill = {
        paint_shortcut,
        erase_shortcut,
        {ctrl + _L("Mouse wheel"), m_desc["smart_fill_angle"]},
        clipping_shortcut,
        toggle_wireframe_shortcut
    };

    m_shortcuts_gap_fill = {
        {ctrl + _L("Mouse wheel"), m_desc["gap_area"]},
        toggle_wireframe_shortcut
    };

    init_extruders_data();

    return true;
}

GLGizmoMmuSegmentation::GLGizmoMmuSegmentation(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id)
    : GLGizmoPainterBase(parent, icon_filename, sprite_id), m_current_tool(ImGui::CircleButtonIcon)
{
}

void GLGizmoMmuSegmentation::render_painter_gizmo()
{
    const Selection& selection = m_parent.get_selection();

    glsafe(::glEnable(GL_BLEND));
    glsafe(::glEnable(GL_DEPTH_TEST));

    render_triangles(selection);

    m_c->object_clipper()->render_cut();
    m_c->instances_hider()->render_cut();
    render_cursor();

    glsafe(::glDisable(GL_BLEND));
}

void GLGizmoMmuSegmentation::data_changed(bool is_serializing)
{
    GLGizmoPainterBase::data_changed(is_serializing);
    if (m_state != On || wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() != ptFFF || wxGetApp().extruders_edited_cnt() <= 1)
        return;

    ModelObject* model_object = m_c->selection_info()->model_object();
    int prev_extruders_count = int(m_extruders_colors.size());
    if (prev_extruders_count != wxGetApp().filaments_cnt()) {
        if (wxGetApp().filaments_cnt() > int(GLGizmoMmuSegmentation::EXTRUDERS_LIMIT))
            show_notification_extruders_limit_exceeded();

        this->init_extruders_data();
        // Reinitialize triangle selectors because of change of extruder count need also change the size of GLIndexedVertexArray
        if (prev_extruders_count != wxGetApp().filaments_cnt())
            this->init_model_triangle_selectors();
    } else if (wxGetApp().plater()->get_extruders_colors() != m_extruders_colors) {
        this->init_extruders_data();
        this->update_triangle_selectors_colors();
    }
    else if (model_object != nullptr && get_extruder_id_for_volumes(*model_object) != m_volumes_extruder_idxs) {
        this->init_model_triangle_selectors();
    }
}

// BBS
bool GLGizmoMmuSegmentation::on_number_key_down(int number)
{
    int extruder_idx = number - 1;
    if (extruder_idx < m_extruders_colors.size() && extruder_idx >= 0)
        m_selected_extruder_idx = extruder_idx;

    return true;
}

bool GLGizmoMmuSegmentation::on_key_down_select_tool_type(int keyCode) {
    switch (keyCode)
    {
    case 'F':
        m_current_tool = ImGui::FillButtonIcon;
        break;
    case 'T':
        m_current_tool = ImGui::TriangleButtonIcon;
        break;
    case 'S':
        m_current_tool = ImGui::SphereButtonIcon;
        break;
    case 'C':
        m_current_tool = ImGui::CircleButtonIcon;
        break;
    case 'H':
        m_current_tool = ImGui::HeightRangeIcon;
        break;
    case 'G':
        m_current_tool = ImGui::GapFillIcon;
        break;
    default:
        return false;
        break;
    }
    return true;
}

static void render_extruders_combo(const std::string& label,
                                   const std::vector<std::string>& extruders,
                                   const std::vector<ColorRGBA>& extruders_colors,
                                   size_t& selection_idx)
{
    assert(!extruders_colors.empty());
    assert(extruders_colors.size() == extruders_colors.size());

    size_t selection_out = selection_idx;
    // It is necessary to use BeginGroup(). Otherwise, when using SameLine() is called, then other items will be drawn inside the combobox.
    ImGui::BeginGroup();
    ImVec2 combo_pos = ImGui::GetCursorScreenPos();
    if (ImGui::BeginCombo(label.c_str(), "")) {
        for (size_t extruder_idx = 0; extruder_idx < std::min(extruders.size(), GLGizmoMmuSegmentation::EXTRUDERS_LIMIT); ++extruder_idx) {
            ImGui::PushID(int(extruder_idx));
            ImVec2 start_position = ImGui::GetCursorScreenPos();

            if (ImGui::Selectable("", extruder_idx == selection_idx))
                selection_out = extruder_idx;

            ImGui::SameLine();
            ImGuiStyle &style  = ImGui::GetStyle();
            float       height = ImGui::GetTextLineHeight();
            ImGui::GetWindowDrawList()->AddRectFilled(start_position, ImVec2(start_position.x + height + height / 2, start_position.y + height), ImGuiWrapper::to_ImU32(extruders_colors[extruder_idx]));
            ImGui::GetWindowDrawList()->AddRect(start_position, ImVec2(start_position.x + height + height / 2, start_position.y + height), IM_COL32_BLACK);

            ImGui::SetCursorScreenPos(ImVec2(start_position.x + height + height / 2 + style.FramePadding.x, start_position.y));
            ImGui::Text("%s", extruders[extruder_idx].c_str());
            ImGui::PopID();
        }

        ImGui::EndCombo();
    }

    ImVec2      backup_pos = ImGui::GetCursorScreenPos();
    ImGuiStyle &style      = ImGui::GetStyle();

    ImGui::SetCursorScreenPos(ImVec2(combo_pos.x + style.FramePadding.x, combo_pos.y + style.FramePadding.y));
    ImVec2 p      = ImGui::GetCursorScreenPos();
    float  height = ImGui::GetTextLineHeight();

    ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + height + height / 2, p.y + height), ImGuiWrapper::to_ImU32(extruders_colors[selection_idx]));
    ImGui::GetWindowDrawList()->AddRect(p, ImVec2(p.x + height + height / 2, p.y + height), IM_COL32_BLACK);

    ImGui::SetCursorScreenPos(ImVec2(p.x + height + height / 2 + style.FramePadding.x, p.y));
    ImGui::Text("%s", extruders[selection_out].c_str());
    ImGui::SetCursorScreenPos(backup_pos);
    ImGui::EndGroup();

    selection_idx = selection_out;
}

void GLGizmoMmuSegmentation::render_tooltip_button(float x, float y)
{
    auto get_shortcuts = [this]() -> std::vector<std::pair<wxString, wxString>> {
        switch (m_tool_type) {
        case ToolType::BRUSH: return m_shortcuts_brush;

        case ToolType::BUCKET_FILL:
        case ToolType::SMART_FILL: return m_shortcuts_bucket_fill;

        case ToolType::GAP_FILL: return m_shortcuts_gap_fill;

        default: return {};
        }
    };

    GLGizmoUtils::render_tooltip_button(m_imgui, m_parent, get_shortcuts(), x, y);
}

void GLGizmoMmuSegmentation::on_render_input_window(float x, float y, float bottom_limit)
{
    if (!m_c->selection_info()->model_object()) return;

    const float approx_height = m_imgui->scaled(22.0f);
    y = std::min(y, bottom_limit - approx_height);
    GizmoImguiSetNextWIndowPos(x, y, ImGuiCond_Always);

    wchar_t old_tool = m_current_tool;

    // BBS
    ImGuiWrapper::push_toolbar_style(m_parent.get_scale());
    GizmoImguiBegin(get_name(), ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    // First calculate width of all the texts that are could possibly be shown. We will decide set the dialog width based on that:
    const float space_size = m_imgui->get_style_scaling() * 8;
    const float clipping_slider_left  = std::max(m_imgui->calc_text_size(m_desc.at("clipping_of_view")).x + m_imgui->scaled(1.5f),
        m_imgui->calc_text_size(m_desc.at("reset_direction")).x + m_imgui->scaled(1.5f) + ImGui::GetStyle().FramePadding.x * 2);
    const float cursor_slider_left = m_imgui->calc_text_size(m_desc.at("cursor_size")).x + m_imgui->scaled(1.5f);
    const float smart_fill_slider_left = m_imgui->calc_text_size(m_desc.at("smart_fill_angle")).x + m_imgui->scaled(1.5f);
    const float edge_detect_slider_left = m_imgui->calc_text_size(m_desc.at("edge_detection")).x + m_imgui->scaled(1.f);
    const float gap_area_slider_left = m_imgui->calc_text_size(m_desc.at("gap_area")).x + m_imgui->scaled(1.5f) + space_size;
    const float height_range_slider_left = m_imgui->calc_text_size(m_desc.at("height_range")).x + m_imgui->scaled(2.f);

    const float remove_btn_width = m_imgui->calc_text_size(m_desc.at("remove_all")).x + m_imgui->scaled(1.f);
    const float filter_btn_width = m_imgui->calc_text_size(m_desc.at("perform")).x + m_imgui->scaled(1.f);
    const float remap_btn_width = m_imgui->calc_text_size(m_desc.at("perform_remap")).x + m_imgui->scaled(1.f);
    const float buttons_width = remove_btn_width + filter_btn_width + remap_btn_width + m_imgui->scaled(2.f);
    const float minimal_slider_width = m_imgui->scaled(4.f);
    const float color_button_width = m_imgui->calc_text_size(std::string_view{""}).x + m_imgui->scaled(1.75f);

    float caption_max = 0.f;
    float total_text_max = 0.f;
    for (const auto &t : std::array<std::string, 6>{"paint", "erase", "cursor_size", "smart_fill_angle", "height_range", "clipping_of_view"}) {
        caption_max = std::max(caption_max, m_imgui->calc_text_size(m_desc[t + "_caption"]).x);
        total_text_max = std::max(total_text_max, m_imgui->calc_text_size(m_desc[t]).x);
    }
    total_text_max += caption_max + m_imgui->scaled(1.f);
    caption_max += m_imgui->scaled(1.f);

    const float circle_max_width = std::max(clipping_slider_left,cursor_slider_left);
    const float height_max_width = std::max(clipping_slider_left,height_range_slider_left);
    const float sliders_left_width = std::max(smart_fill_slider_left,
                                         std::max(cursor_slider_left, std::max(edge_detect_slider_left, std::max(gap_area_slider_left, std::max(height_range_slider_left,
                                                                                                                                              clipping_slider_left))))) + space_size;
    const float slider_icon_width = m_imgui->get_slider_icon_size().x;
    float window_width = minimal_slider_width + sliders_left_width + slider_icon_width;
    const int max_filament_items_per_line = 8;
    const float empty_button_width = m_imgui->calc_button_size("").x;
    const float filament_item_width = empty_button_width + m_imgui->scaled(1.5f);

    window_width = std::max(window_width, total_text_max);
    window_width = std::max(window_width, buttons_width);
    window_width = std::max(window_width, max_filament_items_per_line * filament_item_width + +m_imgui->scaled(0.5f));

    const float sliders_width = m_imgui->scaled(7.0f);
    const float drag_left_width = ImGui::GetStyle().WindowPadding.x + sliders_width - space_size;

    const float max_tooltip_width = ImGui::GetFontSize() * 20.0f;
    ImDrawList * draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    static float color_button_high  = 25.0;
    draw_list->AddRectFilled({pos.x - 10.0f, pos.y - 7.0f}, {pos.x + window_width + ImGui::GetFrameHeight(), pos.y + color_button_high}, ImGui::GetColorU32(ImGuiCol_FrameBgActive, 1.0f), 5.0f);

    float color_button = ImGui::GetCursorPos().y;

    m_imgui->text(m_desc.at("filaments"));

    float start_pos_x = ImGui::GetCursorPos().x;
    const ImVec2 max_label_size = ImGui::CalcTextSize("99", NULL, true);
    const float item_spacing = m_imgui->scaled(0.8f);
    size_t n_extruder_colors = std::min((size_t)EnforcerBlockerType::ExtruderMax, m_extruders_colors.size());
    for (int extruder_idx = 0; extruder_idx < n_extruder_colors; extruder_idx++) {
        const ColorRGBA &extruder_color = m_extruders_colors[extruder_idx];
        ImVec4           color_vec      = ImGuiWrapper::to_ImVec4(extruder_color);
        std::string color_label = std::string("##extruder color ") + std::to_string(extruder_idx);
        std::string item_text = std::to_string(extruder_idx + 1);
        const ImVec2 label_size = ImGui::CalcTextSize(item_text.c_str(), NULL, true);

        const ImVec2 button_size(max_label_size.x + m_imgui->scaled(0.5f),0.f);

        float button_offset = start_pos_x;
        if (extruder_idx % max_filament_items_per_line != 0) {
            button_offset += filament_item_width * (extruder_idx % max_filament_items_per_line);
            ImGui::SameLine(button_offset);
        }

        // draw filament background
        ImGuiColorEditFlags flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip;
        if (m_selected_extruder_idx != extruder_idx) flags |= ImGuiColorEditFlags_NoBorder;
        #ifdef __APPLE__
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA); // ORCA use orca color for selected filament border
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0);
            bool color_picked = ImGui::ColorButton(color_label.c_str(), color_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #else
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA); // ORCA use orca color for selected filament border
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0);
            bool color_picked = ImGui::ColorButton(color_label.c_str(), color_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #endif
        color_button_high = ImGui::GetCursorPos().y - color_button - 2.0;
        if (color_picked) { m_selected_extruder_idx = extruder_idx; }

        if (extruder_idx < 16 && ImGui::IsItemHovered()) m_imgui->tooltip(_L("Shortcut Key ") + std::to_string(extruder_idx + 1), max_tooltip_width);

        // draw filament id
        float gray = 0.299 * extruder_color.r() + 0.587 * extruder_color.g() + 0.114 * extruder_color.b();
        ImGui::SameLine(button_offset + (button_size.x - label_size.x) / 2.f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {10.0,15.0});
        if (gray * 255.f < 80.f)
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "%s", item_text.c_str());
        else
            ImGui::TextColored(ImVec4(0.0f, 0.0f, 0.0f, 1.0f), "%s", item_text.c_str());

        ImGui::PopStyleVar();
    }
    //ImGui::NewLine();
    ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.1));

    m_imgui->text(m_desc.at("tool_type"));

    std::array<wchar_t, 6> tool_ids;
    tool_ids = { ImGui::CircleButtonIcon, ImGui::SphereButtonIcon, ImGui::TriangleButtonIcon, ImGui::HeightRangeIcon, ImGui::FillButtonIcon, ImGui::GapFillIcon };
    std::array<wchar_t, 6> icons;
    if (m_is_dark_mode)
        icons = { ImGui::CircleButtonDarkIcon, ImGui::SphereButtonDarkIcon, ImGui::TriangleButtonDarkIcon, ImGui::HeightRangeDarkIcon, ImGui::FillButtonDarkIcon, ImGui::GapFillDarkIcon };
    else
        icons = { ImGui::CircleButtonIcon, ImGui::SphereButtonIcon, ImGui::TriangleButtonIcon, ImGui::HeightRangeIcon, ImGui::FillButtonIcon, ImGui::GapFillIcon };
    std::array<wxString, 6> tool_tips = { _L("Circle"), _L("Sphere"), _L("Triangle"), _L("Height Range"), _L("Fill"), _L("Gap Fill") };
    for (int i = 0; i < tool_ids.size(); i++) {
        std::string  str_label = std::string("");
        std::wstring btn_name  = icons[i] + boost::nowide::widen(str_label);

        if (i != 0) ImGui::SameLine((empty_button_width + m_imgui->scaled(1.75f)) * i + m_imgui->scaled(1.5f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));                     // ORCA Removes button background on dark mode
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.f));                       // ORCA Fixes icon rendered without colors while using Light theme
        if (m_current_tool == tool_ids[i]) {
            ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.f, 0.59f, 0.53f, 0.25f));  // ORCA use orca color for selected tool / brush
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.f, 0.59f, 0.53f, 0.25f));  // ORCA use orca color for selected tool / brush
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.f, 0.59f, 0.53f, 0.30f));  // ORCA use orca color for selected tool / brush
            ImGui::PushStyleColor(ImGuiCol_Border,          ImGuiWrapper::COL_ORCA);            // ORCA use orca color for border on selected tool / brush
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 1.0);
        }
        bool btn_clicked = ImGui::Button(into_u8(btn_name).c_str());
        if (m_current_tool == tool_ids[i])
        {
            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar(2);
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(1);

        if (btn_clicked && m_current_tool != tool_ids[i]) {
            m_current_tool = tool_ids[i];
            for (auto &triangle_selector : m_triangle_selectors) {
                triangle_selector->seed_fill_unselect_all_triangles();
                triangle_selector->request_update_render_data();
            }
        }

        if (ImGui::IsItemHovered()) {
            m_imgui->tooltip(tool_tips[i], max_tooltip_width);
        }
    }

    ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.1));

    if (m_current_tool != old_tool)
        this->tool_changed(old_tool, m_current_tool);

    if (m_current_tool == ImGui::CircleButtonIcon || m_current_tool == ImGui::SphereButtonIcon) {
        if (m_current_tool == ImGui::CircleButtonIcon)
            m_cursor_type = TriangleSelector::CursorType::CIRCLE;
        else
             m_cursor_type = TriangleSelector::CursorType::SPHERE;
        m_tool_type = ToolType::BRUSH;

        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc.at("cursor_size"));
        ImGui::SameLine(sliders_left_width);
        ImGui::PushItemWidth(sliders_width);
        m_imgui->bbl_slider_float_style("##cursor_radius", &m_cursor_radius, CursorRadiusMin, CursorRadiusMax, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + sliders_left_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##cursor_radius_input", &m_cursor_radius, 0.05f, 0.0f, 0.0f, "%.2f");

        if (m_imgui->bbl_checkbox(_L("Vertical"), m_vertical_only)) {
            if (m_vertical_only) {
                m_horizontal_only = false;
            }
        }
        if (m_imgui->bbl_checkbox(_L("Horizontal"), m_horizontal_only)) {
            if (m_horizontal_only) {
                m_vertical_only = false;
            }
        }
    } 
    else if (m_current_tool == ImGui::TriangleButtonIcon) {
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        m_tool_type   = ToolType::BRUSH;

        if (m_imgui->bbl_checkbox(_L("Vertical"), m_vertical_only)) {
            if (m_vertical_only) {
                m_horizontal_only = false;
            }
        }
        if (m_imgui->bbl_checkbox(_L("Horizontal"), m_horizontal_only)) {
            if (m_horizontal_only) {
                m_vertical_only = false;
            }
        }
    } 
    else if (m_current_tool == ImGui::FillButtonIcon) {
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        m_tool_type = ToolType::BUCKET_FILL;

        m_imgui->bbl_checkbox(m_desc["edge_detection"], m_detect_geometry_edge);

        if (m_detect_geometry_edge) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc["smart_fill_angle"]);
            std::string format_str = std::string("%.f") + I18N::translate_utf8("°", "Face angle threshold,"
                                                                                    "placed after the number with no whitespace in between.");
            ImGui::SameLine(sliders_left_width);
            ImGui::PushItemWidth(sliders_width);
            if (m_imgui->bbl_slider_float_style("##smart_fill_angle", &m_smart_fill_angle, SmartFillAngleMin, SmartFillAngleMax, format_str.data(), 1.0f, true))
                for (auto &triangle_selector : m_triangle_selectors) {
                    triangle_selector->seed_fill_unselect_all_triangles();
                    triangle_selector->request_update_render_data();
                }
            ImGui::SameLine(drag_left_width + sliders_left_width);
            ImGui::PushItemWidth(1.5 * slider_icon_width);
            ImGui::BBLDragFloat("##smart_fill_angle_input", &m_smart_fill_angle, 0.05f, 0.0f, 0.0f, "%.2f");
        } else {
            // set to negative value to disable edge detection
            m_smart_fill_angle = -1.f;
        }
    } 
    else if (m_current_tool == ImGui::HeightRangeIcon) {
        m_tool_type   = ToolType::BRUSH;
        m_cursor_type = TriangleSelector::CursorType::HEIGHT_RANGE;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc["height_range"] + ":");
        ImGui::SameLine(sliders_left_width);
        ImGui::PushItemWidth(sliders_width);
        std::string format_str = std::string("%.2f") + I18N::translate_utf8("mm", "Height range," "Facet in [cursor z, cursor z + height] will be selected.");
        m_imgui->bbl_slider_float_style("##cursor_height", &m_cursor_height, CursorHeightMin, CursorHeightMax, format_str.data(), 1.0f, true);
        ImGui::SameLine(drag_left_width + sliders_left_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##cursor_height_input", &m_cursor_height, 0.05f, 0.0f, 0.0f, "%.2f");
    }
    else if (m_current_tool == ImGui::GapFillIcon) {
        m_tool_type = ToolType::GAP_FILL;
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc["gap_area"] + ":");
        ImGui::SameLine(sliders_left_width);
        ImGui::PushItemWidth(sliders_width);
        std::string format_str = std::string("%.2f") + I18N::translate_utf8("", "Triangle patch area threshold,""triangle patch will be merged to neighbor if its area is less than threshold");
        m_imgui->bbl_slider_float_style("##gap_area", &TriangleSelectorPatch::gap_area, TriangleSelectorPatch::GapAreaMin, TriangleSelectorPatch::GapAreaMax, format_str.data(), 1.0f, true);
        ImGui::SameLine(drag_left_width + sliders_left_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##gap_area_input", &TriangleSelectorPatch::gap_area, 0.05f, 0.0f, 0.0f, "%.2f");
    }

    ImGui::Separator();
    if (m_c->object_clipper()->get_position() == 0.f) {
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc.at("clipping_of_view"));
    } else {
        if (m_imgui->button(m_desc.at("reset_direction"))) {
            wxGetApp().CallAfter([this]() { m_c->object_clipper()->set_position_by_ratio(-1., false); });
        }
    }

    auto clp_dist = float(m_c->object_clipper()->get_position());
    ImGui::SameLine(sliders_left_width);
    ImGui::PushItemWidth(sliders_width);
    bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
    ImGui::SameLine(drag_left_width + sliders_left_width);
    ImGui::PushItemWidth(1.5 * slider_icon_width);
    bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

    if (slider_clp_dist || b_clp_dist_input) {
        m_c->object_clipper()->set_position_by_ratio(clp_dist, true);
    }

    ImGui::Separator();

    const bool can_convert_regions_to_vertex_colors = selected_object_has_painted_regions();
    m_imgui->disabled_begin(!can_convert_regions_to_vertex_colors);
    if (m_imgui->button(_L("Convert regions to vertex colors")))
        convert_selected_regions_to_vertex_colors();
    if (ImGui::IsItemHovered()) {
        if (can_convert_regions_to_vertex_colors)
            m_imgui->tooltip(_L("Convert painted color regions into imported vertex color data, clear the regions, and assign a texture mapping zone."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have painted color regions."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    const bool can_bake_image_texture_data = selected_object_has_bakeable_image_texture_data();
    m_imgui->disabled_begin(!can_bake_image_texture_data);
    if (m_imgui->button(_L("Bake image texture to vertex colors")))
        bake_selected_object_image_texture_to_vertex_colors();
    if (ImGui::IsItemHovered()) {
        if (can_bake_image_texture_data)
            m_imgui->tooltip(_L("Sample imported image texture UVs into stored vertex colors, then discard the baked image texture data."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have imported image texture data with UVs."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    const bool can_clear_image_texture_data = selected_object_has_imported_texture_data();
    m_imgui->disabled_begin(!can_clear_image_texture_data);
    if (m_imgui->button(_L("Clear Image Texture Data")))
        clear_selected_object_image_texture_data();
    if (ImGui::IsItemHovered()) {
        if (can_clear_image_texture_data)
            m_imgui->tooltip(_L("Discard imported image texture data from the selected object."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have imported image texture data."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    ImGui::Separator();

    const bool can_apply_stored_vertex_colors = selected_object_has_imported_vertex_colors();
    m_imgui->disabled_begin(!can_apply_stored_vertex_colors);
    if (m_imgui->button(_L("Convert vertex colors to regions (will erase painting)")))
        open_obj_vertex_color_mapping_dialog();
    if (ImGui::IsItemHovered()) {
        if (can_apply_stored_vertex_colors)
            m_imgui->tooltip(_L("Open OBJ color mapping dialog using stored imported vertex colors."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have stored imported vertex colors."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    ImGui::Separator();

    // ORCA: Remap filaments section (Border only, Title in border). 
    // Styled as a panel for visual grouping.
    if (m_imgui->button(m_desc.at("perform_remap"))) {
        m_show_remap_panel = !m_show_remap_panel;
    }

    if (m_show_remap_panel)
    {
        ImGui::Spacing();

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        std::string title = into_u8(m_desc.at("perform_remap"));
        float available_width = ImGui::GetContentRegionAvail().x;

        // ORCA: Draw Background filled (consistent with Filaments section)
        // Use static to remember height from previous frame so we can draw it behind.
        static float remap_panel_high = 40.0f;
        ImVec2 p_bg_min = ImGui::GetCursorScreenPos();
        // Adjust background position: slight negative offset to align with padding, width fills available
        // height from static variable.
        draw_list->AddRectFilled({p_bg_min.x - 10.0f, p_bg_min.y - 7.0f}, {p_bg_min.x + available_width + ImGui::GetFrameHeight(), p_bg_min.y + remap_panel_high}, ImGui::GetColorU32(ImGuiCol_FrameBgActive, 1.0f), 5.0f);
        
        float start_y = ImGui::GetCursorPos().y;

        // ORCA: Title as simple text - Removed as per request (redundant with button)
        // m_imgui->text(title);

        ImGui::BeginGroup();
        // ORCA: Reduce vertical spacing within this group
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(m_imgui->scaled(0.4f), m_imgui->scaled(0.2f)));

        render_filament_remap_ui(window_width, max_tooltip_width);

        ImGui::PopStyleVar();
        ImGui::EndGroup();

        // ORCA: Update height for next frame fill
        remap_panel_high = ImGui::GetCursorPos().y - start_y;

        // ORCA: Add Remap and Cancel buttons (outside the panel)
        ImGui::Spacing();
        if (m_imgui->button(m_desc.at("remap"))) {
            this->remap_filament_assignments();
            // Reset mapping to identity after apply
            for (size_t i = 0; i < m_extruder_remap.size(); ++i) m_extruder_remap[i] = i;
        }
        ImGui::SameLine();
        if (m_imgui->button(m_desc.at("cancel_remap"))) {
            // Reset mapping to identity
            for (size_t i = 0; i < m_extruder_remap.size(); ++i) m_extruder_remap[i] = i;
        }
    }

    ImGui::Separator();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 10.0f));
    render_tooltip_button(x, y);

    float f_scale =m_parent.get_gizmos_manager().get_layout_scale();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 4.0f * f_scale));

    ImGui::SameLine();

    if (m_current_tool == ImGui::GapFillIcon) {
        if (m_imgui->button(m_desc.at("perform"))) {
            Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Gap fill", UndoRedo::SnapshotType::GizmoAction);

            for (int i = 0; i < m_triangle_selectors.size(); i++) {
                TriangleSelectorPatch* ts_mm = dynamic_cast<TriangleSelectorPatch*>(m_triangle_selectors[i].get());
                ts_mm->update_selector_triangles();
                ts_mm->request_update_render_data(true);
            }
            update_model_object();
            m_parent.set_as_dirty();
        }

        ImGui::SameLine();
    }

    if (m_imgui->button(m_desc.at("remove_all"))) {
        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Reset selection", UndoRedo::SnapshotType::GizmoAction);
        ModelObject *        mo  = m_c->selection_info()->model_object();
        int                  idx = -1;
        for (ModelVolume *mv : mo->volumes)
            if (mv->is_model_part()) {
                ++idx;
                m_triangle_selectors[idx]->reset();
                m_triangle_selectors[idx]->request_update_render_data(true);
            }

        update_model_object();
        m_parent.set_as_dirty();
    }
    ImGui::PopStyleVar(2);
    GizmoImguiEnd();

    // BBS
    ImGuiWrapper::pop_toolbar_style();
}


void GLGizmoMmuSegmentation::update_model_object()
{
    bool updated = false;
    ModelObject* mo = m_c->selection_info()->model_object();
    int idx = -1;
    for (ModelVolume* mv : mo->volumes) {
        if (! mv->is_model_part())
            continue;
        ++idx;
        updated |= mv->mmu_segmentation_facets.set(*m_triangle_selectors[idx].get());
    }

    if (updated) {
        const ModelObjectPtrs &mos = wxGetApp().model().objects;
        size_t obj_idx = std::find(mos.begin(), mos.end(), mo) - mos.begin();
        wxGetApp().obj_list()->update_info_items(obj_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(obj_idx, 0);
        m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));

        // ORCA: Refresh cache
        this->update_used_filaments();
    }
}

void GLGizmoMmuSegmentation::init_model_triangle_selectors()
{
    const ModelObject *mo = m_c->selection_info()->model_object();
    m_triangle_selectors.clear();
    m_volumes_extruder_idxs.clear();

    // Don't continue when extruders colors are not initialized
    if(m_extruders_colors.empty())
        return;

    // BBS: Don't continue when model object is null
    if (mo == nullptr)
        return;

    for (const ModelVolume *mv : mo->volumes) {
        if (!mv->is_model_part())
            continue;

        int extruder_idx = (mv->extruder_id() > 0) ? mv->extruder_id() - 1 : 0;
        std::vector<ColorRGBA> ebt_colors;
        ebt_colors.push_back(m_extruders_colors[size_t(extruder_idx)]);
        ebt_colors.insert(ebt_colors.end(), m_extruders_colors.begin(), m_extruders_colors.end());

        // This mesh does not account for the possible Z up SLA offset.
        const TriangleMesh* mesh = &mv->mesh();
        m_triangle_selectors.emplace_back(std::make_unique<TriangleSelectorPatch>(*mesh, mv, ebt_colors, 0.2));
        // Reset of TriangleSelector is done inside TriangleSelectorMmGUI's constructor, so we don't need it to perform it again in deserialize().
        EnforcerBlockerType max_ebt = (EnforcerBlockerType)std::min(m_extruders_colors.size(), (size_t)EnforcerBlockerType::ExtruderMax);
        m_triangle_selectors.back()->deserialize(mv->mmu_segmentation_facets.get_data(), false, max_ebt);
        m_triangle_selectors.back()->request_update_render_data();
        m_triangle_selectors.back()->set_wireframe_needed(true);
        m_volumes_extruder_idxs.push_back(mv->extruder_id());
    }
}

void GLGizmoMmuSegmentation::update_triangle_selectors_colors()
{
    for (int i = 0; i < m_triangle_selectors.size(); i++) {
        TriangleSelectorPatch* selector = dynamic_cast<TriangleSelectorPatch*>(m_triangle_selectors[i].get());
        int extruder_idx = m_volumes_extruder_idxs[i];
        int extruder_color_idx = std::max(0, extruder_idx - 1);
        std::vector<ColorRGBA> ebt_colors;
        ebt_colors.push_back(m_extruders_colors[extruder_color_idx]);
        ebt_colors.insert(ebt_colors.end(), m_extruders_colors.begin(), m_extruders_colors.end());
        selector->set_ebt_colors(ebt_colors);
    }
}

void GLGizmoMmuSegmentation::update_from_model_object(bool first_update)
{
    wxBusyCursor wait;

    // Extruder colors need to be reloaded before calling init_model_triangle_selectors to render painted triangles
    // using colors from loaded 3MF and not from printer profile in Slicer.
    if (int prev_extruders_count = int(m_extruders_colors.size());
        prev_extruders_count != wxGetApp().filaments_cnt() || wxGetApp().plater()->get_extruders_colors() != m_extruders_colors)
        this->init_extruders_data();

    this->init_model_triangle_selectors();

    // ORCA: Refresh cache when model changes
    this->update_used_filaments();
}

void GLGizmoMmuSegmentation::tool_changed(wchar_t old_tool, wchar_t new_tool)
{
    if ((old_tool == ImGui::GapFillIcon && new_tool == ImGui::GapFillIcon) ||
        (old_tool != ImGui::GapFillIcon && new_tool != ImGui::GapFillIcon))
        return;

    for (auto& selector_ptr : m_triangle_selectors) {
        TriangleSelectorPatch* tsp = dynamic_cast<TriangleSelectorPatch*>(selector_ptr.get());
        tsp->set_filter_state(new_tool == ImGui::GapFillIcon);
    }
}

bool GLGizmoMmuSegmentation::selected_object_has_imported_vertex_colors() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!volume->imported_vertex_colors_rgba.empty())
            return true;
    }
    return false;
}

bool GLGizmoMmuSegmentation::selected_object_has_imported_texture_data() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (model_volume_has_imported_image_texture_data(volume))
            return true;
    }
    return false;
}

bool GLGizmoMmuSegmentation::selected_object_has_bakeable_image_texture_data() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (model_volume_has_bakeable_image_texture_data(volume))
            return true;
    }
    return false;
}

bool GLGizmoMmuSegmentation::selected_object_has_painted_regions() const
{
    for (const auto &selector : m_triangle_selectors) {
        if (selector == nullptr)
            continue;
        const TriangleSelector::TriangleSplittingData data = selector->serialize();
        for (size_t state_idx = static_cast<size_t>(EnforcerBlockerType::Extruder1); state_idx < data.used_states.size(); ++state_idx)
            if (data.used_states[state_idx])
                return true;
    }

    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume != nullptr && volume->is_model_part() && !volume->mmu_segmentation_facets.empty())
            return true;
    }
    return false;
}

void GLGizmoMmuSegmentation::open_obj_vertex_color_mapping_dialog()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    ModelVolume *target_volume = nullptr;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!volume->imported_vertex_colors_rgba.empty()) {
            target_volume = volume;
            break;
        }
    }
    if (target_volume == nullptr)
        return;

    if (target_volume->mesh().its.vertices.size() != target_volume->imported_vertex_colors_rgba.size())
        return;

    ObjDialogInOut in_out;
    in_out.input_colors.reserve(target_volume->imported_vertex_colors_rgba.size());
    for (const uint32_t packed : target_volume->imported_vertex_colors_rgba) {
        const float r = float((packed >> 24) & 0xFF) / 255.f;
        const float g = float((packed >> 16) & 0xFF) / 255.f;
        const float b = float((packed >> 8) & 0xFF) / 255.f;
        const float a = float(packed & 0xFF) / 255.f;
        in_out.input_colors.emplace_back(RGBA{r, g, b, a});
    }

    if (in_out.input_colors.empty())
        return;

    in_out.is_single_color = true;
    const RGBA first_color = in_out.input_colors.front();
    for (const RGBA &color : in_out.input_colors) {
        if (color != first_color) {
            in_out.is_single_color = false;
            break;
        }
    }
    in_out.first_extruder_id = 1;
    in_out.deal_vertex_color = true;
    Model preview_model;
    preview_model.add_object(*object);
    in_out.model = &preview_model;

    const std::vector<std::string> extruder_colours = wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, false);
    ObjColorDialog color_dlg(nullptr, in_out, extruder_colours);
    if (color_dlg.ShowModal() != wxID_OK)
        return;
    if (in_out.filament_ids.empty())
        return;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert vertex colors to regions", UndoRedo::SnapshotType::GizmoAction);
    if (!Model::obj_import_vertex_color_deal_for_object(in_out.filament_ids, in_out.first_extruder_id, object))
        return;

    update_from_model_object();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::bake_selected_object_image_texture_to_vertex_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool baked = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Bake image texture to vertex colors", UndoRedo::SnapshotType::GizmoAction);

    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_bakeable_image_texture_data(volume))
            continue;

        const indexed_triangle_set &its = volume->mesh().its;

        struct VertexColorAccumulator
        {
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            double a = 0.0;
            double weight = 0.0;
        };

        std::vector<VertexColorAccumulator> accumulators(its.vertices.size());
        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            if (volume->imported_texture_uv_valid[tri_idx] == 0)
                continue;

            const auto &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const size_t uv_offset = tri_idx * 6;
            const std::array<Vec2f, 3> uvs = {
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5])
            };

            const Vec3f p0 = its.vertices[size_t(tri[0])].cast<float>();
            const Vec3f p1 = its.vertices[size_t(tri[1])].cast<float>();
            const Vec3f p2 = its.vertices[size_t(tri[2])].cast<float>();
            const float area = 0.5f * (p1 - p0).cross(p2 - p0).norm();
            const double weight = std::isfinite(area) && area > EPSILON ? double(area) : 1.0;
            const std::array<int, 3> vertex_indices = {tri[0], tri[1], tri[2]};

            for (size_t corner = 0; corner < 3; ++corner) {
                const ColorRGBA color = sample_texture_rgba_for_vertex_bake(volume->imported_texture_rgba,
                                                                            volume->imported_texture_width,
                                                                            volume->imported_texture_height,
                                                                            uvs[corner]);
                VertexColorAccumulator &acc = accumulators[size_t(vertex_indices[corner])];
                acc.r += double(color.r()) * weight;
                acc.g += double(color.g()) * weight;
                acc.b += double(color.b()) * weight;
                acc.a += double(color.a()) * weight;
                acc.weight += weight;
            }
        }

        std::vector<uint32_t> vertex_colors;
        vertex_colors.reserve(its.vertices.size());
        for (size_t vertex_idx = 0; vertex_idx < its.vertices.size(); ++vertex_idx) {
            const VertexColorAccumulator &acc = accumulators[vertex_idx];
            if (acc.weight > 0.0) {
                vertex_colors.emplace_back(pack_vertex_color_rgba(ColorRGBA(float(acc.r / acc.weight),
                                                                            float(acc.g / acc.weight),
                                                                            float(acc.b / acc.weight),
                                                                            float(acc.a / acc.weight))));
            } else if (vertex_idx < volume->imported_vertex_colors_rgba.size()) {
                vertex_colors.emplace_back(volume->imported_vertex_colors_rgba[vertex_idx]);
            } else {
                vertex_colors.emplace_back(pack_vertex_color_rgba(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
            }
        }

        if (vertex_colors.size() != its.vertices.size())
            continue;

        volume->imported_vertex_colors_rgba = std::move(vertex_colors);
        volume->imported_texture_uvs_per_face.clear();
        volume->imported_texture_uv_valid.clear();
        volume->imported_texture_rgba.clear();
        volume->imported_texture_width = 0;
        volume->imported_texture_height = 0;
        baked = true;
    }

    if (!baked)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

    m_parent.set_as_dirty();
    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::convert_selected_regions_to_vertex_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr || m_triangle_selectors.empty())
        return;

    std::vector<std::string> color_strings;
    if (wxGetApp().plater() != nullptr)
        color_strings = wxGetApp().plater()->get_extruder_colors_from_plater_config();

    std::vector<ColorRGBA> filament_colors;
    filament_colors.reserve(color_strings.size());
    for (const std::string &color_string : color_strings) {
        unsigned char rgba[4] = {38, 166, 154, 255};
        BitmapCache::parse_color4(color_string, rgba);
        filament_colors.emplace_back(float(rgba[0]) / 255.f,
                                     float(rgba[1]) / 255.f,
                                     float(rgba[2]) / 255.f,
                                     float(rgba[3]) / 255.f);
    }
    if (filament_colors.empty())
        filament_colors.emplace_back(0.15f, 0.65f, 0.6f, 1.f);

    auto color_for_filament_id = [&filament_colors](unsigned int filament_id) {
        if (filament_id >= 1 && filament_id <= filament_colors.size())
            return filament_colors[filament_id - 1];
        return filament_colors.front();
    };

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert regions to vertex colors", UndoRedo::SnapshotType::GizmoAction);

    bool converted = false;
    int selector_idx = -1;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        ++selector_idx;
        if (selector_idx < 0 || size_t(selector_idx) >= m_triangle_selectors.size() || m_triangle_selectors[size_t(selector_idx)] == nullptr)
            continue;

        const auto &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        std::vector<std::vector<TriangleSelector::FacetStateTriangle>> triangles_per_type;
        m_triangle_selectors[size_t(selector_idx)]->get_facet_triangles(triangles_per_type);
        if (triangles_per_type.empty())
            continue;

        struct VertexColorAccumulator
        {
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            double a = 0.0;
            double weight = 0.0;
        };

        std::vector<VertexColorAccumulator> accumulators(its.vertices.size());
        bool accumulated_any = false;
        const unsigned int base_filament_id = volume->extruder_id() > 0 ? unsigned(volume->extruder_id()) : 1u;

        for (size_t state_idx = 0; state_idx < triangles_per_type.size(); ++state_idx) {
            const unsigned int filament_id = state_idx == 0 ? base_filament_id : unsigned(state_idx);
            ColorRGBA state_color = color_for_filament_id(filament_id);
            state_color.a(1.f);

            for (const TriangleSelector::FacetStateTriangle &triangle : triangles_per_type[state_idx]) {
                if (triangle.source_triangle < 0)
                    continue;
                const size_t source_triangle = size_t(triangle.source_triangle);
                if (source_triangle >= its.indices.size())
                    continue;

                const auto &source_indices = its.indices[source_triangle];
                if (source_indices[0] < 0 || source_indices[1] < 0 || source_indices[2] < 0)
                    continue;
                if (size_t(source_indices[0]) >= its.vertices.size() ||
                    size_t(source_indices[1]) >= its.vertices.size() ||
                    size_t(source_indices[2]) >= its.vertices.size())
                    continue;

                const Vec3f source_p0 = its.vertices[size_t(source_indices[0])].cast<float>();
                const Vec3f source_p1 = its.vertices[size_t(source_indices[1])].cast<float>();
                const Vec3f source_p2 = its.vertices[size_t(source_indices[2])].cast<float>();
                const Vec3f centroid = (triangle.vertices[0] + triangle.vertices[1] + triangle.vertices[2]) / 3.f;
                Vec3f weights(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
                if (!barycentric_weights_for_region_vertex_colors(centroid, source_p0, source_p1, source_p2, weights))
                    weights = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);

                weights.x() = std::max(0.f, weights.x());
                weights.y() = std::max(0.f, weights.y());
                weights.z() = std::max(0.f, weights.z());
                const float weights_sum = weights.x() + weights.y() + weights.z();
                if (weights_sum > EPSILON)
                    weights /= weights_sum;
                else
                    weights = Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);

                const float area = 0.5f * (triangle.vertices[1] - triangle.vertices[0]).cross(triangle.vertices[2] - triangle.vertices[0]).norm();
                const double area_weight = std::max(double(area), 1e-6);
                const std::array<float, 3> bary = {weights.x(), weights.y(), weights.z()};

                for (size_t corner = 0; corner < 3; ++corner) {
                    VertexColorAccumulator &acc = accumulators[size_t(source_indices[corner])];
                    const double weight = area_weight * double(bary[corner]);
                    acc.r += double(state_color.r()) * weight;
                    acc.g += double(state_color.g()) * weight;
                    acc.b += double(state_color.b()) * weight;
                    acc.a += double(state_color.a()) * weight;
                    acc.weight += weight;
                }
                accumulated_any = true;
            }
        }

        if (!accumulated_any)
            continue;

        const ColorRGBA fallback_color = color_for_filament_id(base_filament_id);
        std::vector<uint32_t> vertex_colors;
        vertex_colors.reserve(its.vertices.size());
        for (const VertexColorAccumulator &acc : accumulators) {
            if (acc.weight > 0.0) {
                vertex_colors.emplace_back(pack_vertex_color_rgba(ColorRGBA(float(acc.r / acc.weight),
                                                                            float(acc.g / acc.weight),
                                                                            float(acc.b / acc.weight),
                                                                            float(acc.a / acc.weight))));
            } else {
                vertex_colors.emplace_back(pack_vertex_color_rgba(fallback_color));
            }
        }

        volume->imported_vertex_colors_rgba = std::move(vertex_colors);
        volume->mmu_segmentation_facets.reset();
        m_triangle_selectors[size_t(selector_idx)]->reset();
        m_triangle_selectors[size_t(selector_idx)]->request_update_render_data(true);
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object->config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object->volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
    }

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

void GLGizmoMmuSegmentation::clear_selected_object_image_texture_data()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool cleared = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Clear image texture data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_imported_image_texture_data(volume))
            continue;

        volume->imported_texture_uvs_per_face.clear();
        volume->imported_texture_uv_valid.clear();
        volume->imported_texture_rgba.clear();
        volume->imported_texture_width = 0;
        volume->imported_texture_height = 0;
        cleared = true;
    }

    if (!cleared)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

    m_parent.set_as_dirty();
    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

PainterGizmoType GLGizmoMmuSegmentation::get_painter_type() const
{
    return PainterGizmoType::MM_SEGMENTATION;
}

// BBS
ColorRGBA GLGizmoMmuSegmentation::get_cursor_hover_color() const
{
    if (m_selected_extruder_idx < m_extruders_colors.size())
        return m_extruders_colors[m_selected_extruder_idx];
    else
        return m_extruders_colors[0];
}

void GLGizmoMmuSegmentation::on_set_state()
{
    GLGizmoPainterBase::on_set_state();

    if (get_state() == Off) {
        ModelObject* mo = m_c->selection_info()->model_object();
        if (mo) Slic3r::save_object_mesh(*mo);
        m_parent.post_event(SimpleEvent(EVT_GLCANVAS_FORCE_UPDATE));
        if (m_current_tool == ImGui::GapFillIcon) {//exit gap fill
            m_current_tool = ImGui::CircleButtonIcon;
        }
    }
}

wxString GLGizmoMmuSegmentation::handle_snapshot_action_name(bool shift_down, GLGizmoPainterBase::Button button_down) const
{
    wxString action_name;
    if (shift_down)
        action_name = _L("Remove painted color");
    else {
        action_name        = GUI::format(_L("Painted using: Filament %1%"), m_selected_extruder_idx);
    }
    return action_name;
}

void GLMmSegmentationGizmo3DScene::release_geometry() {
    if (this->vertices_VBO_id) {
        glsafe(::glDeleteBuffers(1, &this->vertices_VBO_id));
        this->vertices_VBO_id = 0;
    }
    for(auto &triangle_indices_VBO_id : triangle_indices_VBO_ids) {
        glsafe(::glDeleteBuffers(1, &triangle_indices_VBO_id));
        triangle_indices_VBO_id = 0;
    }
#if !SLIC3R_OPENGL_ES
    if (OpenGLManager::get_gl_info().is_core_profile()) {
#endif // !SLIC3R_OPENGL_ES
        if (this->vertices_VAO_id > 0) {
            glsafe(::glDeleteVertexArrays(1, &this->vertices_VAO_id));
            this->vertices_VAO_id = 0;
        }
#if !SLIC3R_OPENGL_ES
    }
#endif // !SLIC3R_OPENGL_ES

    this->clear();
}

void GLMmSegmentationGizmo3DScene::render(size_t triangle_indices_idx) const
{
    assert(triangle_indices_idx < this->triangle_indices_VBO_ids.size());
    assert(this->triangle_patches.size() == this->triangle_indices_VBO_ids.size());
#if !SLIC3R_OPENGL_ES
    if (OpenGLManager::get_gl_info().is_core_profile()) {
#endif // !SLIC3R_OPENGL_ES
        assert(this->vertices_VAO_id != 0);
#if !SLIC3R_OPENGL_ES
    }
#endif // !SLIC3R_OPENGL_ES
    assert(this->vertices_VBO_id != 0);
    assert(this->triangle_indices_VBO_ids[triangle_indices_idx] != 0);

    GLShaderProgram* shader = wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

#if !SLIC3R_OPENGL_ES
    if (OpenGLManager::get_gl_info().is_core_profile()) {
#endif // !SLIC3R_OPENGL_ES
        glsafe(::glBindVertexArray(this->vertices_VAO_id));
#if !SLIC3R_OPENGL_ES
    }
#endif // !SLIC3R_OPENGL_ES
    // the following binding is needed to set the vertex attributes
    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_VBO_id));
    const GLint position_id = shader->get_attrib_location("v_position");
    if (position_id != -1) {
        glsafe(::glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (GLvoid*)0));
        glsafe(::glEnableVertexAttribArray(position_id));
    }

    // Render using the Vertex Buffer Objects.
    if (this->triangle_indices_VBO_ids[triangle_indices_idx] != 0 &&
        this->triangle_indices_sizes[triangle_indices_idx] > 0) {
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices_VBO_ids[triangle_indices_idx]));
        glsafe(::glDrawElements(GL_TRIANGLES, GLsizei(this->triangle_indices_sizes[triangle_indices_idx]), GL_UNSIGNED_INT, nullptr));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }

    if (position_id != -1)
        glsafe(::glDisableVertexAttribArray(position_id));

    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
#if !SLIC3R_OPENGL_ES
    if (OpenGLManager::get_gl_info().is_core_profile()) {
#endif // !SLIC3R_OPENGL_ES
        glsafe(::glBindVertexArray(0));
#if !SLIC3R_OPENGL_ES
    }
#endif // !SLIC3R_OPENGL_ES
}

void GLMmSegmentationGizmo3DScene::finalize_vertices()
{
#if !SLIC3R_OPENGL_ES
    if (OpenGLManager::get_gl_info().is_core_profile()) {
#endif // !SLIC3R_OPENGL_ES
        assert(this->vertices_VAO_id == 0);
#if !SLIC3R_OPENGL_ES
    }
#endif // !SLIC3R_OPENGL_ES
    assert(this->vertices_VBO_id == 0);
    if (!this->vertices.empty()) {
#if !SLIC3R_OPENGL_ES
        if (OpenGLManager::get_gl_info().is_core_profile()) {
#endif // !SLIC3R_OPENGL_ES
            glsafe(::glGenVertexArrays(1, &this->vertices_VAO_id));
            glsafe(::glBindVertexArray(this->vertices_VAO_id));
#if !SLIC3R_OPENGL_ES
        }
#endif // !SLIC3R_OPENGL_ES

        glsafe(::glGenBuffers(1, &this->vertices_VBO_id));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_VBO_id));
        glsafe(::glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(float), this->vertices.data(), GL_STATIC_DRAW));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
        this->vertices.clear();

#if !SLIC3R_OPENGL_ES
        if (OpenGLManager::get_gl_info().is_core_profile()) {
#endif // !SLIC3R_OPENGL_ES
            glsafe(::glBindVertexArray(0));
#if !SLIC3R_OPENGL_ES
        }
#endif // !SLIC3R_OPENGL_ES
    }
}

void GLMmSegmentationGizmo3DScene::finalize_triangle_indices()
{
    triangle_indices_VBO_ids.resize(this->triangle_patches.size());
    triangle_indices_sizes.resize(this->triangle_patches.size());
    assert(std::all_of(triangle_indices_VBO_ids.cbegin(), triangle_indices_VBO_ids.cend(), [](const auto &ti_VBO_id) { return ti_VBO_id == 0; }));

    for (size_t buffer_idx = 0; buffer_idx < this->triangle_patches.size(); ++buffer_idx) {
        std::vector<int>& triangle_indices = this->triangle_patches[buffer_idx].triangle_indices;
        triangle_indices_sizes[buffer_idx] = triangle_indices.size();
        if (!triangle_indices.empty()) {
            glsafe(::glGenBuffers(1, &this->triangle_indices_VBO_ids[buffer_idx]));
            glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices_VBO_ids[buffer_idx]));
            glsafe(::glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangle_indices.size() * sizeof(int), triangle_indices.data(), GL_STATIC_DRAW));
            glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
            triangle_indices.clear();
        }
    }
}

// ORCA: Update the cache of used filaments (both base volume extruders and painted triangles)
void GLGizmoMmuSegmentation::update_used_filaments()
{
    m_used_filaments.clear();

    // Add base extruder IDs from volumes (unpainted areas)
    for (int ext_id : m_volumes_extruder_idxs) {
        // ext_id is 1-based (1 = Extruder 1), 0 = Default (usually maps to first available or object default)
        // Here we assume 0 maps to index 0 (Extruder 1) for simplicity in display, 
        // or we should check logic in init_model_triangle_selectors where it does:
        // int extruder_idx = (mv->extruder_id() > 0) ? mv->extruder_id() - 1 : 0;
        int idx = (ext_id > 0) ? ext_id - 1 : 0;
        if (idx >= 0 && idx < m_extruders_colors.size())
             m_used_filaments.insert((size_t)idx);
    }

    // Add painted states
    for (const auto& selector : m_triangle_selectors) {
        if (!selector) continue;
        TriangleSelector::TriangleSplittingData data = selector->serialize();
        std::vector<EnforcerBlockerType> states = TriangleSelector::extract_used_facet_states(data);
        for (EnforcerBlockerType s : states) {
             int idx = (int)s - (int)EnforcerBlockerType::Extruder1;
             if (idx >= 0 && idx < m_extruders_colors.size())
                 m_used_filaments.insert((size_t)idx);
        }
    }
}

void GLGizmoMmuSegmentation::render_filament_remap_ui(float window_width, float max_tooltip_width)
{
    size_t n_extr = std::min((size_t)EnforcerBlockerType::ExtruderMax, m_extruders_colors.size());

    const ImVec2 max_label_size = ImGui::CalcTextSize("99", NULL, true);
    const ImVec2 button_size(max_label_size.x + m_imgui->scaled(0.5f), 0.f);

    int displayed_count = 0;
    const int max_per_line = 8;

    // ORCA: Use m_used_filaments to show only relevant source filaments
    for (size_t src : m_used_filaments) {
        if (src >= n_extr) continue;

        const ColorRGBA &src_col = m_extruders_colors[src];          // keep for text contrast
        const ColorRGBA &dst_col = m_extruders_colors[m_extruder_remap[src]];
        
        // ORCA: Button now shows the SOURCE color (per maintainer request)
        // This keeps the UI stable until "Remap" is clicked.
        ImVec4 col_vec = ImGuiWrapper::to_ImVec4(src_col);

        if (displayed_count > 0 && (displayed_count % max_per_line != 0))
            ImGui::SameLine();
        
        std::string btn_id = "##remap_src_" + std::to_string(src);
        std::string pop_id = "popup_" + std::to_string(src);
        
        ImGuiColorEditFlags flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_NoLabel  | ImGuiColorEditFlags_NoPicker |
                                    ImGuiColorEditFlags_NoTooltip;
        
        // ORCA: Show border ONLY if the popup is open (visual feedback for active selection)
        // Decoupled from m_selected_extruder_idx to prevent unwanted selection highlights.
        if (!ImGui::IsPopupOpen(pop_id.c_str()))
             flags |= ImGuiColorEditFlags_NoBorder;
        
        #ifdef __APPLE__
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0);
            bool clicked = ImGui::ColorButton(btn_id.c_str(), col_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #else
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0);
            bool clicked = ImGui::ColorButton(btn_id.c_str(), col_vec, flags, button_size);
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(1);
        #endif

        // overlay destination number with proper contrast calculation
        // ORCA: Text still shows DESTINATION index, but contrast is against SOURCE color now.
        std::string dst_txt = std::to_string(m_extruder_remap[src] + 1);
        float gray = 0.299f * src_col.r() + 0.587f * src_col.g() + 0.114f * src_col.b();
        ImVec2 txt_sz = ImGui::CalcTextSize(dst_txt.c_str());
        ImVec2 pos = ImGui::GetItemRectMin();
        ImVec2 size = ImGui::GetItemRectSize();
        
        if (gray * 255.f < 80.f)
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(pos.x + (size.x - txt_sz.x) * 0.5f, pos.y + (size.y - txt_sz.y) * 0.5f),
                IM_COL32(255,255,255,255), dst_txt.c_str());
        else
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(pos.x + (size.x - txt_sz.x) * 0.5f, pos.y + (size.y - txt_sz.y) * 0.5f),
                IM_COL32(0,0,0,255), dst_txt.c_str());

        // ORCA: Show NEW color as a small triangle in the corner if remapped
        if (src != m_extruder_remap[src]) {
            float s = m_imgui->scaled(0.55f);
            float offset = m_imgui->scaled(0.15f); // Inset to avoid rounded corner clipping
            ImVec2 p = ImVec2(pos.x + offset, pos.y + offset);
            
            // Contrast outline: White for dark backgrounds, Black for light backgrounds
            // Use dst_col (new color) for outline contrast check? Or src_col?
            // Usually outline is around the triangle (dst_col).
            float dst_gray = 0.299f * dst_col.r() + 0.587f * dst_col.g() + 0.114f * dst_col.b();
            ImU32 outline_col = (dst_gray * 255.f < 80.f) ? IM_COL32(255, 255, 255, 180) : IM_COL32(0, 0, 0, 180);

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            draw_list->AddTriangleFilled(
                p,
                ImVec2(p.x + s, p.y),
                ImVec2(p.x, p.y + s),
                ImGuiWrapper::to_ImU32(dst_col));
            
            // ORCA: Add a thin outline for better contrast when colors are similar
            draw_list->AddTriangle(
                p,
                ImVec2(p.x + s, p.y),
                ImVec2(p.x, p.y + s),
                outline_col,
                0.5f);
        }

        // popup with possible destinations
        if (clicked) {
            // Calculate popup position centered below the current button
            ImVec2 button_pos = ImGui::GetItemRectMin();
            ImVec2 button_size = ImGui::GetItemRectSize();
            ImVec2 popup_pos(button_pos.x + button_size.x * 0.5f, button_pos.y + button_size.y);
            
            // Set popup styling BEFORE opening popup
            ImGui::SetNextWindowPos(popup_pos, ImGuiCond_Appearing, ImVec2(0.5f, -0.1f));
            ImGui::SetNextWindowBgAlpha(1.0f); // Ensure full opacity
            ImGui::OpenPopup(pop_id.c_str());
        }
        
        // Apply popup styling before BeginPopup using standard Orca colors
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 1.0f);
        // ORCA: Use FrameBgActive for consistency and to ensure visibility of white filaments
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive));
        ImGui::PushStyleColor(ImGuiCol_Border, m_is_dark_mode ? ImVec4(0.5f, 0.5f, 0.5f, 1.0f) : ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        
        if (ImGui::BeginPopup(pop_id.c_str())) {
            
            m_imgui->text(_L("To:"));

            for (int dst = 0; dst < (int)n_extr; ++dst) {
                const ColorRGBA &dst_col_popup = m_extruders_colors[dst];
                ImVec4 dst_vec = ImGuiWrapper::to_ImVec4(dst_col_popup);
                if (dst > 0 && (dst % max_per_line != 0))
                     ImGui::SameLine();
                std::string dst_btn = "##dst_" + std::to_string(src) + "_" + std::to_string(dst);
                
                // Apply same styling to destination buttons
                ImGuiColorEditFlags dst_flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs |
                                               ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoPicker |
                                               ImGuiColorEditFlags_NoTooltip;
                // Show border for currently selected destination filament
                if (m_extruder_remap[src] != dst) dst_flags |= ImGuiColorEditFlags_NoBorder;
                
                #ifdef __APPLE__
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0);
                    bool dst_clicked = ImGui::ColorButton(dst_btn.c_str(), dst_vec, dst_flags, button_size);
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor(1);
                #else
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGuiWrapper::COL_ORCA);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0);
                    bool dst_clicked = ImGui::ColorButton(dst_btn.c_str(), dst_vec, dst_flags, button_size);
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor(1);
                #endif
                
                // overlay destination number on popup buttons
                std::string dst_num_txt = std::to_string(dst + 1);
                float dst_gray = 0.299f * dst_col_popup.r() + 0.587f * dst_col_popup.g() + 0.114f * dst_col_popup.b();
                ImVec2 dst_txt_sz = ImGui::CalcTextSize(dst_num_txt.c_str());
                ImVec2 dst_pos = ImGui::GetItemRectMin();
                ImVec2 dst_size = ImGui::GetItemRectSize();
                
                if (dst_gray * 255.f < 80.f)
                    ImGui::GetWindowDrawList()->AddText(
                        ImVec2(dst_pos.x + (dst_size.x - dst_txt_sz.x) * 0.5f, dst_pos.y + (dst_size.y - dst_txt_sz.y) * 0.5f),
                        IM_COL32(255,255,255,255), dst_num_txt.c_str());
                else
                    ImGui::GetWindowDrawList()->AddText(
                        ImVec2(dst_pos.x + (dst_size.x - dst_txt_sz.x) * 0.5f, dst_pos.y + (dst_size.y - dst_txt_sz.y) * 0.5f),
                        IM_COL32(0,0,0,255), dst_num_txt.c_str());
                
                if (dst_clicked)
                {
                    m_extruder_remap[src] = dst;
                    // update the source button color immediately
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::EndPopup();
        }
        
        // Clean up popup styling (always pop, whether popup was open or not)
        ImGui::PopStyleColor(2); // PopupBg and Border
        ImGui::PopStyleVar(2);   // PopupRounding and PopupBorderSize
        
        displayed_count++;
    }
}

void GLGizmoMmuSegmentation::remap_filament_assignments()
{
    if (m_extruder_remap.empty())
        return;

    constexpr size_t MAX_EBT = (size_t)EnforcerBlockerType::ExtruderMax;
    EnforcerBlockerStateMap state_map;

    // identity mapping by default
    for (size_t i = 0; i <= MAX_EBT; ++i)
        state_map[i] = static_cast<EnforcerBlockerType>(i);

    size_t n_extr = std::min(m_extruder_remap.size(), MAX_EBT);
    const int start_extruder = (int) EnforcerBlockerType::Extruder1;
    bool   any_change = false;
    for (size_t src = 0; src < n_extr; ++src) {
        size_t dst = m_extruder_remap[src];
        if (dst != src) {
            state_map[src+start_extruder] = static_cast<EnforcerBlockerType>(dst+start_extruder);
            any_change     = true;
        }
    }
    if (!any_change)
        return;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(),
                                  "Remap filament assignments",
                                  UndoRedo::SnapshotType::GizmoAction);

    bool updated = false;
    int idx = -1;
    ModelObject* mo = m_c->selection_info()->model_object();
    if (!mo) return;

    bool volume_extruder_changed = false;

    for (ModelVolume* mv : mo->volumes) {
        if (!mv->is_model_part()) continue;
        ++idx;
        TriangleSelectorGUI* ts = m_triangle_selectors[idx].get();
        if (!ts) continue;

        // Remap painted triangles
        ts->remap_triangle_state(state_map);
        ts->request_update_render_data(true);

        // ORCA: Remap base volume extruder as well if selected
        int current_ext_id = mv->extruder_id();
        int current_idx = (current_ext_id > 0) ? current_ext_id - 1 : 0;

        if (current_idx >= 0 && current_idx < m_extruder_remap.size()) {
            size_t dest_idx = m_extruder_remap[current_idx];
            if (dest_idx != current_idx) {
                // Check if volume has its own extruder config or uses object's fallback                                                                                                                                            
                const ConfigOption *vol_opt = mv->config.option("extruder");                                                                                                                                                        
                if (vol_opt != nullptr && vol_opt->getInt() != 0) {                                                                                                                                                                 
                    // Volume has its own extruder setting, update it                                                                                                                                                               
                    mv->config.set("extruder", (int)dest_idx + 1);                                                                                                                                                                  
                } else {                                                                                                                                                                                                            
                    // Volume uses object's extruder setting, update the object                                                                                                                                                     
                    mo->config.set("extruder", (int)dest_idx + 1);                                                                                                                                                                  
                }      
                if (idx < m_volumes_extruder_idxs.size())
                    m_volumes_extruder_idxs[idx] = (int)dest_idx + 1;
                volume_extruder_changed = true;
            }
        }

        updated = true;
    }

    if (updated) {
        // ORCA: Update renderer colors if base volume extruder changed
        if (volume_extruder_changed) {
            this->update_triangle_selectors_colors();
            // ORCA: Update GUI_ObjectList extruder column to reflect the new extruder value
            wxGetApp().obj_list()->update_objects_list_filament_column(wxGetApp().filaments_cnt());
        }

        // ORCA: Removed "Filament remapping finished" notification to reduce UI noise.
        update_model_object();
        m_parent.set_as_dirty();
        
        // ORCA: Refresh used filaments cache
        this->update_used_filaments();
    }
}

} // namespace Slic3r
