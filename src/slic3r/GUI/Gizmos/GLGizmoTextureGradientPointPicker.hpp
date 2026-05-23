#ifndef slic3r_GLGizmoTextureGradientPointPicker_hpp_
#define slic3r_GLGizmoTextureGradientPointPicker_hpp_

#include "GLGizmoBase.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace Slic3r {
namespace GUI {

class GLGizmoTextureGradientPointPicker final : public GLGizmoBase
{
public:
    struct Pick {
        int object_idx = -1;
        int volume_idx = -1;
        int instance_idx = -1;
        size_t object_id = 0;
        size_t instance_id = 0;
        Vec3f local_point { Vec3f::Zero() };
        Vec3f global_point { Vec3f::Zero() };
    };

    enum class Target {
        None,
        Start,
        End
    };

    using PickCallback = std::function<void(const Pick&)>;
    using HoverCallback = std::function<void(const Pick*)>;
    using CancelCallback = std::function<void()>;

    GLGizmoTextureGradientPointPicker(GLCanvas3D &parent, const std::string &icon_filename, unsigned int sprite_id);

    void set_pick_callback(PickCallback callback,
                           Target target,
                           HoverCallback hover_callback = HoverCallback(),
                           CancelCallback cancel_callback = CancelCallback());
    void cancel_hover_preview();
    Target target() const { return m_target; }

private:
    bool on_init() override;
    std::string on_get_name() const override;
    void on_set_state() override;
    bool on_is_selectable() const override;
    bool on_mouse(const wxMouseEvent &mouse_event) override;
    void on_render() override;
    void on_render_input_window(float x, float y, float bottom_limit) override;

    bool update_hover(const wxMouseEvent &mouse_event);
    void clear_hover();
    void emit_hover_preview();
    bool hover_preview_matches_last() const;

    PickCallback m_pick_callback;
    HoverCallback m_hover_callback;
    CancelCallback m_cancel_callback;
    Target m_target { Target::None };
    Pick m_hover_pick;
    Pick m_last_hover_preview_pick;
    bool m_has_hover { false };
    bool m_hover_preview_emitted { false };
    bool m_last_hover_preview_had_hit { false };
    Vec2d m_mouse_pos { Vec2d::Zero() };
};

} // namespace GUI
} // namespace Slic3r

#endif
