#include "GLGizmoMeshBoolean.hpp"
#include "slic3r/GUI/GLCanvas3D.hpp"
#include "slic3r/GUI/ImGuiWrapper.hpp"
#include "slic3r/GUI/GUI.hpp"
#include "slic3r/GUI/GUI_Utils.hpp"
#include "libslic3r/MeshBoolean.hpp"
#include "libslic3r/ModelTextureDataRemap.hpp"
#include "slic3r/GUI/GUI_ObjectList.hpp"
#include "slic3r/GUI/Plater.hpp"
#include "slic3r/GUI/Camera.hpp"
#include "slic3r/GUI/MsgDialog.hpp"
#include "slic3r/GUI/NotificationManager.hpp"
#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include <imgui/imgui_internal.h>
namespace Slic3r {
namespace GUI {

static const std::string warning_text = _u8L("Unable to perform boolean operation on selected parts");

static wxString mesh_boolean_color_transfer_warning_text()
{
    return _L("Color data could not be transferred because boolean face provenance was unavailable. The mesh boolean result was created without color data.");
}

static wxString mesh_boolean_color_remap_warning_text()
{
    return _L("Color data could not be transferred. The mesh boolean result was created without color data.");
}

static void show_mesh_boolean_color_transfer_warning_dialog()
{
    MessageDialog(wxGetApp().plater(), mesh_boolean_color_transfer_warning_text(), _L("Warning"), wxOK | wxICON_WARNING).ShowModal();
}

static void show_mesh_boolean_color_remap_warning_dialog()
{
    MessageDialog(wxGetApp().plater(), mesh_boolean_color_remap_warning_text(), _L("Warning"), wxOK | wxICON_WARNING).ShowModal();
}

static ColorRGBA mesh_boolean_volume_base_color(const ModelVolume &volume)
{
    std::vector<std::string> colors;
    if (wxGetApp().plater() != nullptr)
        colors = wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, true);
    const int extruder_id = std::max(volume.extruder_id(), 1);
    if (size_t(extruder_id - 1) < colors.size())
        return decode_color_to_float_array(colors[size_t(extruder_id - 1)]);
    return ColorRGBA(0.15f, 0.65f, 0.6f, 1.f);
}

static bool mesh_boolean_snapshot_has_transfer_data(const SimplifyTextureDataSnapshot &snapshot)
{
    return snapshot.source != SimplifyColorSource::None || snapshot.region_painting_present;
}

static bool mesh_boolean_texture_result_has_output(const SimplifyTextureDataResult &result)
{
    switch (result.source) {
    case SimplifyColorSource::RgbaData:
        return !result.remap_failed || (result.rgba_data != nullptr && !result.rgba_data->empty());
    case SimplifyColorSource::ImageTexture:
        return result.texture_width > 0 &&
               result.texture_height > 0 &&
               !result.texture_uv_valid.empty() &&
               (!result.texture_rgba.empty() || !result.texture_raw_filament_offsets.empty());
    case SimplifyColorSource::VertexColors:
        return !result.vertex_colors_rgba.empty();
    case SimplifyColorSource::None:
        break;
    }
    return result.region_painting_valid;
}

static void mesh_boolean_transform_snapshot(SimplifyTextureDataSnapshot &snapshot, const Transform3d &transform, bool fix_left_handed)
{
    transform_simplify_texture_data_snapshot(snapshot, transform);
    if (!fix_left_handed)
        return;
    if (transform.matrix().block(0, 0, 3, 3).determinant() >= 0.)
        return;
    if (!snapshot.source_mesh.indices.empty())
        its_flip_triangles(snapshot.source_mesh);
    for (size_t tri_idx = 0; tri_idx < snapshot.texture_uv_valid.size(); ++tri_idx) {
        const size_t uv_offset = tri_idx * 6;
        if (uv_offset + 5 >= snapshot.texture_uvs_per_face.size())
            break;
        std::swap(snapshot.texture_uvs_per_face[uv_offset + 2], snapshot.texture_uvs_per_face[uv_offset + 4]);
        std::swap(snapshot.texture_uvs_per_face[uv_offset + 3], snapshot.texture_uvs_per_face[uv_offset + 5]);
    }
}

static MultiSourceTextureDataSource mesh_boolean_gizmo_source(const ModelVolume &volume,
                                                              const Transform3d &transform,
                                                              const ColorRGBA &fallback_color,
                                                              bool &had_transfer_data)
{
    MultiSourceTextureDataSource source;
    source.snapshot = snapshot_simplify_texture_data(volume);
    mesh_boolean_transform_snapshot(source.snapshot, transform, true);
    source.fallback_color = fallback_color;
    had_transfer_data = true;
    had_transfer_data |= mesh_boolean_snapshot_has_transfer_data(source.snapshot);
    return source;
}

static MultiSourceTextureTriangleProvenance mesh_boolean_remap_provenance_entry(const MeshBoolean::mcut::MeshFaceProvenance &entry)
{
    MultiSourceTextureTriangleProvenance out;
    out.valid = entry.valid;
    out.source_index = entry.source_index;
    out.source_triangle = entry.source_triangle;
    out.source_barycentric = entry.source_barycentric;
    return out;
}

static std::vector<MultiSourceTextureTriangleProvenance> mesh_boolean_remap_provenance(const std::vector<MeshBoolean::mcut::MeshFaceProvenance> &provenance)
{
    std::vector<MultiSourceTextureTriangleProvenance> out;
    out.reserve(provenance.size());
    for (const MeshBoolean::mcut::MeshFaceProvenance &entry : provenance)
        out.emplace_back(mesh_boolean_remap_provenance_entry(entry));
    return out;
}

GLGizmoMeshBoolean::GLGizmoMeshBoolean(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id)
    : GLGizmoBase(parent, icon_filename, sprite_id)
{
}

GLGizmoMeshBoolean::~GLGizmoMeshBoolean() 
{
}

bool GLGizmoMeshBoolean::gizmo_event(SLAGizmoEventType action, const Vec2d& mouse_position, bool shift_down, bool alt_down, bool control_down) 
{
    if (action == SLAGizmoEventType::LeftDown) {
        const ModelObject* mo = m_c->selection_info()->model_object();
        if (mo == nullptr)
            return true;
        const ModelInstance* mi = mo->instances[m_parent.get_selection().get_instance_idx()];
        std::vector<Transform3d> trafo_matrices;
        for (const ModelVolume* mv : mo->volumes) {
            trafo_matrices.emplace_back(mi->get_transformation().get_matrix() * mv->get_matrix());
        }

        const Camera& camera = wxGetApp().plater()->get_camera();
        Vec3f  normal = Vec3f::Zero();
        Vec3f  hit = Vec3f::Zero();
        size_t facet = 0;
        Vec3f  closest_hit = Vec3f::Zero();
        Vec3f  closest_normal = Vec3f::Zero();
        double closest_hit_squared_distance = std::numeric_limits<double>::max();
        int    closest_hit_mesh_id = -1;

        // Cast a ray on all meshes, pick the closest hit and save it for the respective mesh
        for (int mesh_id = 0; mesh_id < int(trafo_matrices.size()); ++mesh_id) {
            if (m_c->raycaster()->raycasters()[mesh_id] ->unproject_on_mesh(mouse_position, trafo_matrices[mesh_id], camera, hit, normal,
                m_c->object_clipper()->get_clipping_plane(), &facet)) {
                // Is this hit the closest to the camera so far?
                double hit_squared_distance = (camera.get_position() - trafo_matrices[mesh_id] * hit.cast<double>()).squaredNorm();
                if (hit_squared_distance < closest_hit_squared_distance) {
                    closest_hit_squared_distance = hit_squared_distance;
                    closest_hit_mesh_id = mesh_id;
                    closest_hit = hit;
                    closest_normal = normal;
                }
            }
        }

        if (closest_hit == Vec3f::Zero() && closest_normal == Vec3f::Zero())
            return true;

        if (get_selecting_state() == MeshBooleanSelectingState::SelectTool) {
            m_tool.trafo = mo->volumes[closest_hit_mesh_id]->get_matrix();
            m_tool.volume_idx = closest_hit_mesh_id;
            set_tool_volume(mo->volumes[closest_hit_mesh_id]);
            return true;
        }
        if (get_selecting_state() == MeshBooleanSelectingState::SelectSource) {
            m_src.trafo = mo->volumes[closest_hit_mesh_id]->get_matrix();
            m_src.volume_idx = closest_hit_mesh_id;
            set_src_volume(mo->volumes[closest_hit_mesh_id]);
            m_selecting_state = MeshBooleanSelectingState::SelectTool;
            return true;
        }
    }
    return true;
}

bool GLGizmoMeshBoolean::on_mouse(const wxMouseEvent &mouse_event)
{
    // wxCoord == int --> wx/types.h
    Vec2i32 mouse_coord(mouse_event.GetX(), mouse_event.GetY());
    Vec2d mouse_pos = mouse_coord.cast<double>();

    // when control is down we allow scene pan and rotation even when clicking
    // over some object
    bool control_down           = mouse_event.CmdDown();
    bool grabber_contains_mouse = (get_hover_id() != -1);
    if (mouse_event.LeftDown()) {
        if ((!control_down || grabber_contains_mouse) &&            
            gizmo_event(SLAGizmoEventType::LeftDown, mouse_pos, mouse_event.ShiftDown(), mouse_event.AltDown(), false))
            // the gizmo got the event and took some action, there is no need
            // to do anything more
            return true;
    }

    return false;
}

bool GLGizmoMeshBoolean::on_init()
{
    m_shortcut_key = WXK_CONTROL_B;
    return true;
}

std::string GLGizmoMeshBoolean::on_get_name() const
{
    return _u8L("Mesh Boolean");
}

bool GLGizmoMeshBoolean::on_is_activable() const
{
    return m_parent.get_selection().is_single_full_instance() && m_parent.get_selection().get_volume_idxs().size() > 1;
}

void GLGizmoMeshBoolean::on_render()
{
    if (m_parent.get_selection().get_object_idx() < 0)
        return;
    static ModelObject* last_mo = nullptr;
    ModelObject* curr_mo = m_parent.get_selection().get_model()->objects[m_parent.get_selection().get_object_idx()];
    if (last_mo != curr_mo) {
        last_mo = curr_mo;
        m_src.reset();
        m_tool.reset();
        m_operation_mode = MeshBooleanOperation::Union;
        m_selecting_state = MeshBooleanSelectingState::SelectSource;
        return;
    }

    BoundingBoxf3 src_bb;
    BoundingBoxf3 tool_bb;
    const ModelObject* mo = m_c->selection_info()->model_object();
    const ModelInstance* mi = mo->instances[m_parent.get_selection().get_instance_idx()];
    const Selection& selection = m_parent.get_selection();
    const Selection::IndicesList& idxs = selection.get_volume_idxs();
    for (unsigned int i : idxs) {
        const GLVolume* volume = selection.get_volume(i);
        if(volume->volume_idx() == m_src.volume_idx) {
            src_bb = volume->transformed_convex_hull_bounding_box();
        }
        if (volume->volume_idx() == m_tool.volume_idx) {
            tool_bb = volume->transformed_convex_hull_bounding_box();
        }
    }

    ColorRGB src_color = { 1.0f, 1.0f, 1.0f };
    ColorRGB tool_color = {0.0f, 150.0f / 255.0f, 136.0f / 255.0f};
    m_parent.get_selection().render_bounding_box(src_bb, src_color, m_parent.get_scale());
    m_parent.get_selection().render_bounding_box(tool_bb, tool_color, m_parent.get_scale());
}

void GLGizmoMeshBoolean::on_set_state()
{
    if (m_state == EState::On) {
        m_src.reset();
        m_tool.reset();
        bool m_diff_delete_input = false;
        bool m_inter_delete_input = false;
        m_operation_mode = MeshBooleanOperation::Union;
        m_selecting_state = MeshBooleanSelectingState::SelectSource;
    }
    else if (m_state == EState::Off) {
        m_src.reset();
        m_tool.reset();
        bool m_diff_delete_input = false;
        bool m_inter_delete_input = false;
        m_operation_mode = MeshBooleanOperation::Undef;
        m_selecting_state = MeshBooleanSelectingState::Undef;
        wxGetApp().notification_manager()->close_plater_warning_notification(warning_text);
    }
}

CommonGizmosDataID GLGizmoMeshBoolean::on_get_requirements() const
{
    if (m_c && m_c->raycaster_ptr()) {
        m_c->raycaster_ptr()->set_only_support_model_part_flag(false);
    }
    return CommonGizmosDataID(
        int(CommonGizmosDataID::SelectionInfo)
        | int(CommonGizmosDataID::InstancesHider)
        | int(CommonGizmosDataID::Raycaster)
        | int(CommonGizmosDataID::ObjectClipper));
}

void GLGizmoMeshBoolean::on_render_input_window(float x, float y, float bottom_limit)
{
    y = std::min(y, bottom_limit - ImGui::GetWindowHeight());

    static float last_y = 0.0f;
    static float last_w = 0.0f;

    const float currt_scale = m_parent.get_scale();
    ImGuiWrapper::push_toolbar_style(currt_scale);
    GizmoImguiSetNextWIndowPos(x, y, ImGuiCond_Always, 0.0f, 0.0f);
    GizmoImguiBegin("MeshBoolean", ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    const int max_tab_length = 2 * ImGui::GetStyle().FramePadding.x + std::max(ImGui::CalcTextSize(_u8L("Union").c_str()).x,
        std::max(ImGui::CalcTextSize(_u8L("Difference").c_str()).x, ImGui::CalcTextSize(_u8L("Intersection").c_str()).x));
    const int max_cap_length = ImGui::GetStyle().WindowPadding.x + ImGui::GetStyle().ItemSpacing.x +
                               std::max({ImGui::CalcTextSize(_u8L("Source Volume").c_str()).x,
                                         ImGui::CalcTextSize(_u8L("Tool Volume").c_str()).x,
                                         ImGui::CalcTextSize(_u8L("Subtract from").c_str()).x,
                                         ImGui::CalcTextSize(_u8L("Subtract with").c_str()).x});

    const int select_btn_length = 2 * ImGui::GetStyle().FramePadding.x + std::max(ImGui::CalcTextSize(("1 " + _u8L("selected")).c_str()).x, ImGui::CalcTextSize(_u8L("Select").c_str()).x);

    auto selectable = [this](const std::string& label, bool selected, const ImVec2& size_arg) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0,0 });

        ImGuiWindow* window = ImGui::GetCurrentWindow();
        const ImVec2 label_size = ImGui::CalcTextSize(label.c_str(), NULL, true);
        ImVec2 pos = window->DC.CursorPos;
        ImVec2 size = ImGui::CalcItemSize(size_arg, label_size.x + ImGui::GetStyle().FramePadding.x * 2.0f, label_size.y + ImGui::GetStyle().FramePadding.y * 2.0f);
        bool hovered = ImGui::IsMouseHoveringRect(pos, pos + size);

        if (selected || hovered) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Button, { 0, 150.0f / 255.0f, 136.0f / 255.0f, 1.0f });
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, { 0, 150.0f / 255.0f, 136.0f / 255.0f, 1.0f });
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 0, 150.0f / 255.0f, 136.0f / 255.0f, 1.0f });
        }
        else {
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, { 0, 150.0f / 255.0f, 136.0f / 255.0f, 1.0f });
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 0, 150.0f / 255.0f, 136.0f / 255.0f, 1.0f });
        }

        bool res = ImGui::Button(label.c_str(), size_arg);

        if (selected || hovered) {
            ImGui::PopStyleColor(4);
        }
        else {
            ImGui::PopStyleColor(2);
        }

        ImGui::PopStyleVar(1);
        return res;
    };

    auto operate_button = [this](const wxString &label, bool enable) {
        if (!enable) {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            if (m_is_dark_mode) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(39.0f / 255.0f, 39.0f / 255.0f, 39.0f / 255.0f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(108.0f / 255.0f, 108.0f / 255.0f, 108.0f / 255.0f, 1.0f));
            }
            else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(230.0f / 255.0f, 230.0f / 255.0f, 230.0f / 255.0f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(163.0f / 255.0f, 163.0f / 255.0f, 163.0f / 255.0f, 1.0f));
            }
        }

        bool res = m_imgui->button(label.c_str());

        if (!enable) {
            ImGui::PopItemFlag();
            ImGui::PopStyleColor(2);
        }
        return res;
    };

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);
    if (selectable(_u8L("Union"), m_operation_mode == MeshBooleanOperation::Union, ImVec2(max_tab_length, 0.0f))) {
        m_operation_mode = MeshBooleanOperation::Union;
    }
    ImGui::SameLine(0, 0);
    if (selectable(_u8L("Difference"), m_operation_mode == MeshBooleanOperation::Difference, ImVec2(max_tab_length, 0.0f))) {
        m_operation_mode = MeshBooleanOperation::Difference;
    }
    ImGui::SameLine(0, 0);
    if (selectable(_u8L("Intersection"), m_operation_mode == MeshBooleanOperation::Intersection, ImVec2(max_tab_length, 0.0f))) {
        m_operation_mode = MeshBooleanOperation::Intersection;
    }
    ImGui::PopStyleVar();

    ImGui::AlignTextToFramePadding();
    std::string cap_str1 = m_operation_mode != MeshBooleanOperation::Difference ? _u8L("Part 1") : _u8L("Subtract from");
    m_imgui->text(cap_str1);
    ImGui::SameLine(max_cap_length);
    std::string select_src_str = m_src.mv ? "1 " + _u8L("selected") : _u8L("Select");
    select_src_str += "##select_source_volume";
    ImGui::PushItemWidth(select_btn_length);
    if (selectable(select_src_str, m_selecting_state == MeshBooleanSelectingState::SelectSource, ImVec2(select_btn_length, 0)))
        m_selecting_state = MeshBooleanSelectingState::SelectSource;
    ImGui::PopItemWidth();
    if (m_src.mv) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_src.mv->name);

        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, { 0, 0, 0, 0 });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_Button));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_Button));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_Button));
        ImGui::PushStyleColor(ImGuiCol_Border, { 0, 0, 0, 0 });
        if (ImGui::Button((into_u8(ImGui::TextSearchCloseIcon) + "##src").c_str(), {18, 18}))
        {
            m_src.reset();
        }
        ImGui::PopStyleColor(5);
    }

    ImGui::AlignTextToFramePadding();
    std::string cap_str2 = m_operation_mode != MeshBooleanOperation::Difference ? _u8L("Part 2") : _u8L("Subtract with");
    m_imgui->text(cap_str2);
    ImGui::SameLine(max_cap_length);
    std::string select_tool_str = m_tool.mv ? "1 " + _u8L("selected") : _u8L("Select");
    select_tool_str += "##select_tool_volume";
    ImGui::PushItemWidth(select_btn_length);
    if (selectable(select_tool_str, m_selecting_state == MeshBooleanSelectingState::SelectTool, ImVec2(select_btn_length, 0)))
        m_selecting_state = MeshBooleanSelectingState::SelectTool;
    ImGui::PopItemWidth();
    if (m_tool.mv) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_tool.mv->name);

        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, { 0, 0, 0, 0 });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_Button));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_Button));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_Button));
        ImGui::PushStyleColor(ImGuiCol_Border, { 0, 0, 0, 0 });
        if (ImGui::Button((into_u8(ImGui::TextSearchCloseIcon) + "tool").c_str(), {18, 18}))
        {
            m_tool.reset();
        }
        ImGui::PopStyleColor(5);
    }

    bool enable_button = m_src.mv && m_tool.mv;
    if (m_operation_mode == MeshBooleanOperation::Union)
    {
        if (operate_button(_L("Union") + "##btn", enable_button))
            run_boolean_operation("UNION", true);
    }
    else if (m_operation_mode == MeshBooleanOperation::Difference) {
        m_imgui->bbl_checkbox(_L("Delete input"), m_diff_delete_input);
        if (operate_button(_L("Difference") + "##btn", enable_button))
            run_boolean_operation("A_NOT_B", m_diff_delete_input);
    }
    else if (m_operation_mode == MeshBooleanOperation::Intersection){
        m_imgui->bbl_checkbox(_L("Delete input"), m_inter_delete_input);
        if (operate_button(_L("Intersection") + "##btn", enable_button))
            run_boolean_operation("INTERSECTION", m_inter_delete_input);
    }

    float win_w = ImGui::GetWindowWidth();
    if (last_w != win_w || last_y != y) {
        // ask canvas for another frame to render the window in the correct position
        m_imgui->set_requires_extra_frame();
        m_parent.set_as_dirty();
        m_parent.request_extra_frame();
        if (last_w != win_w)
            last_w = win_w;
        if (last_y != y)
            last_y = y;
    }

    GizmoImguiEnd();
    ImGuiWrapper::pop_toolbar_style();
}

void GLGizmoMeshBoolean::on_load(cereal::BinaryInputArchive &ar)
{
    ar(m_enable, m_operation_mode, m_selecting_state, m_diff_delete_input, m_inter_delete_input, m_src, m_tool);
    ModelObject *curr_model_object = m_c->selection_info() ? m_c->selection_info()->model_object() : nullptr;
    m_src.mv = curr_model_object == nullptr ? nullptr : m_src.volume_idx < 0 ? nullptr : curr_model_object->volumes[m_src.volume_idx];
    m_tool.mv = curr_model_object == nullptr ? nullptr : m_tool.volume_idx < 0 ? nullptr : curr_model_object->volumes[m_tool.volume_idx];
}

void GLGizmoMeshBoolean::on_save(cereal::BinaryOutputArchive &ar) const
{
    ar(m_enable, m_operation_mode, m_selecting_state, m_diff_delete_input, m_inter_delete_input, m_src, m_tool);
}

void GLGizmoMeshBoolean::run_boolean_operation(const std::string &boolean_opts, bool delete_input)
{
    TriangleMesh temp_src_mesh = m_src.mv->mesh();
    temp_src_mesh.transform(m_src.trafo, true);
    TriangleMesh temp_tool_mesh = m_tool.mv->mesh();
    temp_tool_mesh.transform(m_tool.trafo, true);

    const ColorRGBA source_base_color = mesh_boolean_volume_base_color(*m_src.mv);
    const ColorRGBA tool_base_color = mesh_boolean_volume_base_color(*m_tool.mv);
    bool had_transfer_data = false;
    std::vector<MultiSourceTextureDataSource> sources;
    sources.reserve(2);
    sources.emplace_back(mesh_boolean_gizmo_source(*m_src.mv, m_src.trafo, source_base_color, had_transfer_data));
    MultiSourceTextureDataSource tool_source = mesh_boolean_gizmo_source(*m_tool.mv,
                                                                         m_tool.trafo,
                                                                         tool_base_color,
                                                                         had_transfer_data);
    if (boolean_opts == "A_NOT_B" && tool_source.snapshot.source == SimplifyColorSource::None)
        tool_source.fallback_color = source_base_color;
    sources.emplace_back(std::move(tool_source));

    if (had_transfer_data) {
        std::vector<MeshBoolean::mcut::ProvenancedMesh> provenanced_results;
        const bool provenance_ok = MeshBoolean::mcut::make_boolean_with_provenance(temp_src_mesh, 0, temp_tool_mesh, 1, provenanced_results, boolean_opts);
        if (provenance_ok && !provenanced_results.empty()) {
            generate_new_volume(delete_input, provenanced_results.front().mesh, &sources, &provenanced_results.front().provenance);
            wxGetApp().notification_manager()->close_plater_warning_notification(warning_text);
            return;
        }
    }

    std::vector<TriangleMesh> temp_mesh_resuls;
    Slic3r::MeshBoolean::mcut::make_boolean(temp_src_mesh, temp_tool_mesh, temp_mesh_resuls, boolean_opts);
    if (temp_mesh_resuls.size() != 0) {
        if (had_transfer_data)
            show_mesh_boolean_color_transfer_warning_dialog();
        generate_new_volume(delete_input, *temp_mesh_resuls.begin());
        wxGetApp().notification_manager()->close_plater_warning_notification(warning_text);
    }
    else {
        wxGetApp().notification_manager()->push_plater_warning_notification(warning_text);
    }
}

void GLGizmoMeshBoolean::generate_new_volume(bool delete_input,
                                             const TriangleMesh& mesh_result,
                                             const std::vector<MultiSourceTextureDataSource>* sources,
                                             const std::vector<MeshBoolean::mcut::MeshFaceProvenance>* provenance) {

    wxGetApp().plater()->take_snapshot("Mesh Boolean");

    ModelObject* curr_model_object = m_c->selection_info()->model_object();

    // generate new volume
    ModelVolume* new_volume = curr_model_object->add_volume(std::move(mesh_result));

    // assign to new_volume from old_volume
    ModelVolume* old_volume = m_src.mv;
    std::string suffix;
    switch (m_operation_mode)
    {
    case MeshBooleanOperation::Union:
        suffix = "union";
        break;
    case MeshBooleanOperation::Difference:
        suffix = "difference";
        break;
    case MeshBooleanOperation::Intersection:
        suffix = "intersection";
        break;
    }
    new_volume->name = old_volume->name + " - " + suffix;
    new_volume->set_new_unique_id();
    new_volume->config.apply(old_volume->config);
    new_volume->set_type(old_volume->type());
    new_volume->set_material_id(old_volume->material_id());
    if (sources != nullptr && provenance != nullptr && new_volume->mesh().its.indices.size() == provenance->size()) {
        std::vector<MultiSourceTextureTriangleProvenance> remap_provenance = mesh_boolean_remap_provenance(*provenance);
        SimplifyTextureDataResult result = remap_multi_source_texture_data(*sources, new_volume->mesh().its, remap_provenance);
        if (mesh_boolean_texture_result_has_output(result) || result.region_painting_touched)
            apply_simplify_texture_data_result(*new_volume, std::move(result));
        else
            show_mesh_boolean_color_remap_warning_dialog();
    } else if (sources != nullptr && !sources->empty())
        show_mesh_boolean_color_transfer_warning_dialog();
    //Vec3d translate_z = { 0,0, (new_volume->source.mesh_offset - old_volume->source.mesh_offset).z() };
    //new_volume->translate(new_volume->get_transformation().get_matrix_no_offset() * translate_z);
    //new_volume->supported_facets.assign(old_volume->supported_facets);
    //new_volume->seam_facets.assign(old_volume->seam_facets);
    //new_volume->mmu_segmentation_facets.assign(old_volume->mmu_segmentation_facets);

    // delete old_volume
    std::swap(curr_model_object->volumes[m_src.volume_idx], curr_model_object->volumes.back());
    curr_model_object->delete_volume(curr_model_object->volumes.size() - 1);

    if (delete_input) {
        std::vector<ItemForDelete> items;
        auto obj_idx = m_parent.get_selection().get_object_idx();
        items.emplace_back(ItemType::itVolume, obj_idx, m_tool.volume_idx);
        wxGetApp().obj_list()->delete_from_model_and_list(items);
    }

    //bool sinking = curr_model_object->bounding_box().min.z() < SINKING_Z_THRESHOLD;
    //if (!sinking)
    //    curr_model_object->ensure_on_bed();
    //curr_model_object->sort_volumes(true);

    wxGetApp().plater()->update();
    wxGetApp().obj_list()->select_item([this, new_volume]() {
        wxDataViewItem sel_item;

        wxDataViewItemArray items = wxGetApp().obj_list()->reorder_volumes_and_get_selection(m_parent.get_selection().get_object_idx(), [new_volume](const ModelVolume* volume) { return volume == new_volume; });
        if (!items.IsEmpty())
            sel_item = items.front();

        return sel_item;
        });

    m_src.reset();
    m_tool.reset();
    m_selecting_state = MeshBooleanSelectingState::SelectSource;
}


}}
