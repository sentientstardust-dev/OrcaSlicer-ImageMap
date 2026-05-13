#ifndef slic3r_GUI_Utils_FixModelByCgal_hpp_
#define slic3r_GUI_Utils_FixModelByCgal_hpp_

#include <cstddef>
#include <string>
#include "../GUI/Widgets/ProgressDialog.hpp"

namespace Slic3r {

class Model;
class ModelObject;
class Print;

struct ModelRepairColorRemapStats
{
    bool   had_color_data    = false;
    bool   remap_requested   = false;
    bool   remap_skipped     = false;
    bool   remap_canceled    = false;
    bool   remap_failed      = false;
    bool   used_fallback_rgba = false;
    size_t volumes_remapped  = 0;
    size_t volumes_cleared   = 0;
};

struct ModelRepairOptions
{
    bool split_before_repair = false;
    bool weld_same_position_vertices = true;
};

enum class ModelRepairColorRemapChoice
{
    Remap,
    Skip,
    Cancel
};

struct ModelRepairPromptState
{
    bool repair_options_selected = false;
    bool repair_options_canceled = false;
    ModelRepairOptions repair_options;
    ModelRepairColorRemapChoice color_remap_choice = ModelRepairColorRemapChoice::Remap;
};

// Return false if fixing was canceled. fix_result is empty on success.
extern bool fix_model_with_cgal_gui(ModelObject                 &model_object,
                                    int                          volume_idx,
                                    GUI::ProgressDialog         &progress_dlg,
                                    const wxString              &msg_header,
                                    std::string                 &fix_result,
                                    ModelRepairColorRemapStats  *color_remap_stats = nullptr,
                                    ModelRepairPromptState      *prompt_state = nullptr);

} // namespace Slic3r

#endif /* slic3r_GUI_Utils_FixModelByCgal_hpp_ */
