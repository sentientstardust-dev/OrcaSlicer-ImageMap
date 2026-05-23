#include "GLGizmoTextureGradientPointPicker.hpp"

#include "libslic3r/Model.hpp"

#include "slic3r/GUI/3DScene.hpp"
#include "slic3r/GUI/Camera.hpp"
#include "slic3r/GUI/GLCanvas3D.hpp"
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/ImGuiWrapper.hpp"
#include "slic3r/GUI/Plater.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace Slic3r {
namespace GUI {

GLGizmoTextureGradientPointPicker::GLGizmoTextureGradientPointPicker(GLCanvas3D &parent,
                                                                     const std::string &icon_filename,
                                                                     unsigned int sprite_id)
    : GLGizmoBase(parent, icon_filename, sprite_id)
{
}

void GLGizmoTextureGradientPointPicker::set_pick_callback(PickCallback callback,
                                                          Target target,
                                                          HoverCallback hover_callback,
                                                          CancelCallback cancel_callback)
{
    m_pick_callback = std::move(callback);
    m_hover_callback = std::move(hover_callback);
    m_cancel_callback = std::move(cancel_callback);
    m_target = target;
    m_hover_preview_emitted = false;
    m_last_hover_preview_had_hit = false;
}

void GLGizmoTextureGradientPointPicker::cancel_hover_preview()
{
    if (m_cancel_callback) {
        CancelCallback cancel_callback = std::move(m_cancel_callback);
        m_cancel_callback = nullptr;
        cancel_callback();
    } else if (m_hover_callback && m_hover_preview_emitted) {
        m_hover_callback(nullptr);
    }
    m_hover_preview_emitted = false;
    m_last_hover_preview_had_hit = false;
    clear_hover();
    m_parent.set_as_dirty();
    m_parent.request_extra_frame();
}

bool GLGizmoTextureGradientPointPicker::on_init()
{
    m_shortcut_key = 0;
    return true;
}

std::string GLGizmoTextureGradientPointPicker::on_get_name() const
{
    return _u8L("Set simple gradient point");
}

void GLGizmoTextureGradientPointPicker::on_set_state()
{
    if (m_state != On)
        cancel_hover_preview();
    else
        clear_hover();
    if (m_state != On) {
        m_pick_callback = nullptr;
        m_hover_callback = nullptr;
        m_cancel_callback = nullptr;
        m_target = Target::None;
    }
    m_parent.enable_picking(m_state != On);
    m_parent.set_as_dirty();
    m_parent.request_extra_frame();
}

bool GLGizmoTextureGradientPointPicker::on_is_selectable() const
{
    return false;
}

void GLGizmoTextureGradientPointPicker::clear_hover()
{
    m_has_hover = false;
    m_hover_pick = {};
}

bool GLGizmoTextureGradientPointPicker::hover_preview_matches_last() const
{
    return m_hover_preview_emitted &&
           m_last_hover_preview_had_hit &&
           m_hover_pick.object_id == m_last_hover_preview_pick.object_id &&
           m_hover_pick.instance_id == m_last_hover_preview_pick.instance_id &&
           (m_hover_pick.global_point - m_last_hover_preview_pick.global_point).squaredNorm() <= 1e-6f;
}

void GLGizmoTextureGradientPointPicker::emit_hover_preview()
{
    if (!m_hover_callback)
        return;
    if (!m_has_hover) {
        if (!m_hover_preview_emitted || m_last_hover_preview_had_hit) {
            m_hover_callback(nullptr);
            m_hover_preview_emitted = true;
            m_last_hover_preview_had_hit = false;
        }
        return;
    }
    if (hover_preview_matches_last())
        return;
    m_hover_callback(&m_hover_pick);
    m_last_hover_preview_pick = m_hover_pick;
    m_hover_preview_emitted = true;
    m_last_hover_preview_had_hit = true;
}

bool GLGizmoTextureGradientPointPicker::update_hover(const wxMouseEvent &mouse_event)
{
    m_mouse_pos = Vec2d(double(mouse_event.GetX()), double(mouse_event.GetY()));
    clear_hover();

    const Model *model = m_parent.get_model();
    if (model == nullptr)
        return false;

    const std::vector<std::shared_ptr<SceneRaycasterItem>> *raycasters =
        m_parent.get_raycasters_for_picking(SceneRaycaster::EType::Volume);
    if (raycasters == nullptr || raycasters->empty())
        return false;

    const GLVolumePtrs &volumes = m_parent.get_volumes().volumes;
    const Camera &camera = wxGetApp().plater()->get_camera();
    const Vec3f camera_forward = camera.get_dir_forward().cast<float>();
    float closest_hit_squared_distance = std::numeric_limits<float>::max();
    Pick best_pick;

    for (const std::shared_ptr<SceneRaycasterItem> &item : *raycasters) {
        if (!item || !item->is_active() || item->get_raycaster() == nullptr)
            continue;

        Vec3f local_hit = Vec3f::Zero();
        Vec3f local_normal = Vec3f::Zero();
        const Transform3d &trafo = item->get_transform();
        if (!item->get_raycaster()->closest_hit(m_mouse_pos, trafo, camera, local_hit, local_normal, nullptr))
            continue;

        const Vec3f global_hit = (trafo * local_hit.cast<double>()).cast<float>();
        const Vec3f global_normal =
            (trafo.matrix().block(0, 0, 3, 3).inverse().transpose() * local_normal.cast<double>()).normalized().cast<float>();
        if (!item->use_back_faces() && global_normal.dot(camera_forward) >= 0.f)
            continue;

        const float distance = float((camera.get_position() - global_hit.cast<double>()).squaredNorm());
        if (!std::isfinite(distance) || distance >= closest_hit_squared_distance)
            continue;

        const int volume_id = SceneRaycaster::decode_id(SceneRaycaster::EType::Volume, item->get_id());
        if (volume_id < 0 || size_t(volume_id) >= volumes.size())
            continue;
        const GLVolume *volume = volumes[size_t(volume_id)];
        if (volume == nullptr || !volume->is_active || volume->disabled || volume->is_wipe_tower)
            continue;
        if (volume->object_idx() < 0 || volume->instance_idx() < 0)
            continue;
        if (size_t(volume->object_idx()) >= model->objects.size())
            continue;
        const ModelObject *model_object = model->objects[size_t(volume->object_idx())];
        if (model_object == nullptr || size_t(volume->instance_idx()) >= model_object->instances.size())
            continue;
        const ModelInstance *model_instance = model_object->instances[size_t(volume->instance_idx())];
        if (model_instance == nullptr)
            continue;

        best_pick.object_idx = volume->object_idx();
        best_pick.volume_idx = volume->volume_idx();
        best_pick.instance_idx = volume->instance_idx();
        best_pick.object_id = model_object->id().id;
        best_pick.instance_id = model_instance->id().id;
        best_pick.global_point = global_hit;
        best_pick.local_point = (model_instance->get_matrix().inverse() * global_hit.cast<double>()).cast<float>();
        closest_hit_squared_distance = distance;
    }

    if (!std::isfinite(closest_hit_squared_distance) || closest_hit_squared_distance == std::numeric_limits<float>::max())
        return false;

    m_hover_pick = best_pick;
    m_has_hover = true;
    return true;
}

bool GLGizmoTextureGradientPointPicker::on_mouse(const wxMouseEvent &mouse_event)
{
    if (m_state != On)
        return false;

    if (mouse_event.Leaving()) {
        clear_hover();
        emit_hover_preview();
        m_parent.set_as_dirty();
        return true;
    }

    const bool had_hover = m_has_hover;
    update_hover(mouse_event);
    emit_hover_preview();
    if (had_hover != m_has_hover || mouse_event.Moving() || mouse_event.Dragging()) {
        m_parent.set_as_dirty();
        m_parent.request_extra_frame();
    }

    if (mouse_event.LeftDown()) {
        if (m_has_hover && m_pick_callback) {
            Pick picked = m_hover_pick;
            CancelCallback cancel_callback = std::move(m_cancel_callback);
            m_cancel_callback = nullptr;
            HoverCallback hover_callback = std::move(m_hover_callback);
            m_hover_callback = nullptr;
            if (cancel_callback)
                cancel_callback();
            else if (hover_callback && m_hover_preview_emitted)
                hover_callback(nullptr);
            m_hover_preview_emitted = false;
            m_last_hover_preview_had_hit = false;
            m_pick_callback(picked);
        }
        if (m_parent.get_gizmos_manager().get_current_type() == GLGizmosManager::TextureGradientPointPicker)
            m_parent.get_gizmos_manager().open_gizmo(GLGizmosManager::TextureGradientPointPicker);
        return true;
    }

    return true;
}

void GLGizmoTextureGradientPointPicker::on_render()
{
}

void GLGizmoTextureGradientPointPicker::on_render_input_window(float, float, float)
{
    if (m_state != On)
        return;

    const Size canvas_size = m_parent.get_canvas_size();
    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(float(canvas_size.get_width()), float(canvas_size.get_height())), ImGuiCond_Always);
    const int flags = ImGuiWindowFlags_NoTitleBar |
                      ImGuiWindowFlags_NoResize |
                      ImGuiWindowFlags_NoMove |
                      ImGuiWindowFlags_NoScrollbar |
                      ImGuiWindowFlags_NoSavedSettings |
                      ImGuiWindowFlags_NoInputs |
                      ImGuiWindowFlags_NoBackground;
    if (ImGui::Begin("simple_gradient_point_picker_overlay", nullptr, flags)) {
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        const std::string message = m_target == Target::End ?
            _u8L("Click an object to set gradient end point") :
            _u8L("Click an object to set gradient start point");
        const ImVec2 text_size = ImGui::CalcTextSize(message.c_str());
        const ImVec2 text_pos(std::max(12.f, 0.5f * (float(canvas_size.get_width()) - text_size.x)), 24.f);
        draw_list->AddText(ImVec2(text_pos.x - 1.f, text_pos.y), IM_COL32(0, 0, 0, 240), message.c_str());
        draw_list->AddText(ImVec2(text_pos.x + 1.f, text_pos.y), IM_COL32(0, 0, 0, 240), message.c_str());
        draw_list->AddText(ImVec2(text_pos.x, text_pos.y - 1.f), IM_COL32(0, 0, 0, 240), message.c_str());
        draw_list->AddText(ImVec2(text_pos.x, text_pos.y + 1.f), IM_COL32(0, 0, 0, 240), message.c_str());
        draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 245), message.c_str());
        if (m_has_hover) {
            const ImVec2 center(float(m_mouse_pos.x()), float(m_mouse_pos.y()));
            draw_list->AddCircle(center, 16.f, IM_COL32(255, 255, 255, 235), 48, 3.f);
            draw_list->AddCircle(center, 19.f, IM_COL32(0, 0, 0, 180), 48, 1.5f);
        }
    }
    ImGui::End();
}

} // namespace GUI
} // namespace Slic3r
