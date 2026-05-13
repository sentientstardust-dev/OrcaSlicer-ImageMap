#include "FixModelByCgal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "libslic3r/MeshBoolean.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/ModelTextureDataRemap.hpp"
#include "libslic3r/format.hpp"
#include "../GUI/I18N.hpp"

#include <wx/checkbox.h>
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/stattext.h>

// Orca: This file provides utilities for repairing 3D model meshes using the CGAL library, handling mesh splitting, merging, and boolean operations.

namespace Slic3r {

namespace {

// Orca: Helper functions for analyzing mesh properties and transformations.

bool is_not_3dimensional_part(const TriangleMesh &mesh)
{
    // Orca: Determines if a mesh is degenerate or represents a non-3dimensional part by checking volume and bounding box dimensions.
    if (mesh.its.indices.empty())
        return true;

    indexed_triangle_set tmp = mesh.its;
    its_remove_degenerate_faces(tmp, true);
    if (tmp.indices.empty())
        return true;

    const BoundingBoxf3 bbox = mesh.bounding_box();
    const Vec3d size = bbox.size();
    const double min_dim = std::min(size.x(), std::min(size.y(), size.z()));
    const double max_dim = std::max(size.x(), std::max(size.y(), size.z()));
    if (min_dim <= EPSILON)
        return true;

    const double volume = std::abs(its_volume(mesh.its));
    const double bbox_volume = size.x() * size.y() * size.z();
    if (volume <= EPSILON)
        return true;

    const double min_relative_thickness = 1e-6;
    const double min_volume_ratio = 1e-6;
    if (min_dim / max_dim <= min_relative_thickness)
        return true;
    if (bbox_volume > 0.0 && volume / bbox_volume <= min_volume_ratio)
        return true;

    return false;
}

enum class RepairProgressStage
{
    Repair,
    Remap,
    Status
};

enum class ColorRemapPartState
{
    Unchanged,
    Pending,
    Preserved,
    Cleared
};

class ColorRemapCanceledException : public std::exception {
public:
    const char* what() const noexcept override { return "Color remap has been canceled"; }
};

using ColorSnapshotByVolume = std::unordered_map<ModelVolume*, SimplifyTextureDataSnapshot>;

bool color_snapshot_is_valid(const SimplifyTextureDataSnapshot &snapshot)
{
    return snapshot.source != SimplifyColorSource::None;
}

bool volume_may_change_during_repair(const ModelVolume &volume, const ModelRepairOptions &options)
{
    return options.weld_same_position_vertices ||
           (options.split_before_repair && volume.is_splittable()) ||
           its_num_open_edges(volume.mesh().its) != 0;
}

bool volume_has_remappable_color_data(const ModelVolume &volume)
{
    const indexed_triangle_set &its = volume.mesh().its;
    if (its.vertices.empty() || its.indices.empty())
        return false;

    if (!volume.texture_mapping_color_facets.empty())
        return true;

    const bool has_valid_uvs = volume.imported_texture_uv_valid.size() == its.indices.size() &&
                               volume.imported_texture_uvs_per_face.size() >= its.indices.size() * 6 &&
                               std::any_of(volume.imported_texture_uv_valid.begin(),
                                           volume.imported_texture_uv_valid.end(),
                                           [](uint8_t valid) { return valid != 0; });
    const bool has_rgba_texture = volume.imported_texture_width > 0 && volume.imported_texture_height > 0 &&
                                  volume.imported_texture_rgba.size() >=
                                      size_t(volume.imported_texture_width) * size_t(volume.imported_texture_height) * 4;
    const bool has_raw_texture = volume.imported_texture_width > 0 && volume.imported_texture_height > 0 &&
                                 volume.imported_texture_raw_channels > 0 &&
                                 volume.imported_texture_raw_filament_offsets.size() >=
                                     size_t(volume.imported_texture_width) * size_t(volume.imported_texture_height) *
                                         size_t(volume.imported_texture_raw_channels);
    if (has_valid_uvs && (has_rgba_texture || has_raw_texture))
        return true;

    return volume.imported_vertex_colors_rgba.size() == its.vertices.size();
}

bool target_has_remappable_color_data(const ModelObject &model_object, int volume_idx)
{
    const size_t start_volume = volume_idx == -1 ? 0 : size_t(volume_idx);
    const size_t end_volume =
        volume_idx == -1 ? model_object.volumes.size() : std::min(model_object.volumes.size(), size_t(volume_idx) + 1);
    for (size_t idx = start_volume; idx < end_volume; ++idx) {
        const ModelVolume *volume = model_object.volumes[idx];
        if (volume != nullptr && volume_has_remappable_color_data(*volume))
            return true;
    }
    return false;
}

ColorSnapshotByVolume collect_color_snapshots(ModelObject &model_object, int volume_idx, const ModelRepairOptions &options)
{
    ColorSnapshotByVolume snapshots;
    const size_t start_volume = volume_idx == -1 ? 0 : size_t(volume_idx);
    const size_t end_volume =
        volume_idx == -1 ? model_object.volumes.size() : std::min(model_object.volumes.size(), size_t(volume_idx) + 1);
    for (size_t idx = start_volume; idx < end_volume; ++idx) {
        ModelVolume *volume = model_object.volumes[idx];
        if (volume == nullptr || !volume_may_change_during_repair(*volume, options))
            continue;
        SimplifyTextureDataSnapshot snapshot = snapshot_simplify_texture_data(*volume);
        if (color_snapshot_is_valid(snapshot))
            snapshots.emplace(volume, std::move(snapshot));
    }
    return snapshots;
}

bool ask_repair_options(GUI::ProgressDialog           &progress_dialog,
                        ModelRepairOptions            &options,
                        ModelRepairColorRemapChoice   &color_remap_choice,
                        bool                           show_color_remap_option)
{
    wxDialog dialog(&progress_dialog, wxID_ANY, _L("Repair options"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);
    auto *top_sizer = new wxBoxSizer(wxVERTICAL);
    auto *message = new wxStaticText(&dialog, wxID_ANY, _L("Choose options to use when repairing model:"));
    message->Wrap(420);
    top_sizer->Add(message, 0, wxEXPAND | wxALL, 12);

    auto *weld_checkbox = new wxCheckBox(&dialog, wxID_ANY, _L("Weld close vertexes"));
    weld_checkbox->SetValue(options.weld_same_position_vertices);
    top_sizer->Add(weld_checkbox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 12);

    auto *split_checkbox = new wxCheckBox(&dialog, wxID_ANY, _L("Split parts"));
    split_checkbox->SetValue(options.split_before_repair);
    top_sizer->Add(split_checkbox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 12);

    wxCheckBox *remap_colors_checkbox = nullptr;
    if (show_color_remap_option) {
        remap_colors_checkbox = new wxCheckBox(&dialog, wxID_ANY, _L("Remap colors to repaired mesh"));
        remap_colors_checkbox->SetValue(color_remap_choice != ModelRepairColorRemapChoice::Skip);
        top_sizer->Add(remap_colors_checkbox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 12);
    }

    wxSizer *buttons = dialog.CreateSeparatedButtonSizer(wxOK | wxCANCEL);
    if (buttons != nullptr)
        top_sizer->Add(buttons, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 12);

    dialog.SetSizerAndFit(top_sizer);
    dialog.Layout();

    if (dialog.ShowModal() != wxID_OK)
        return false;

    options.weld_same_position_vertices = weld_checkbox->GetValue();
    options.split_before_repair = split_checkbox->GetValue();
    if (remap_colors_checkbox != nullptr)
        color_remap_choice = remap_colors_checkbox->GetValue() ? ModelRepairColorRemapChoice::Remap : ModelRepairColorRemapChoice::Skip;
    return true;
}

wxString trim_progress_header(wxString header)
{
    while (!header.empty() && (header.Last() == '\n' || header.Last() == '\r'))
        header.RemoveLast();
    return header;
}

wxString compose_progress_message(const wxString &msg_header, RepairProgressStage stage, const std::string &message)
{
    const wxString header = trim_progress_header(msg_header);
    const wxString translated_message = _(message);

    if (stage == RepairProgressStage::Repair) {
        wxString text = _L("Stage 1 - ") + header;
        if (!message.empty() && translated_message != _L("Repairing model object"))
            text += "\n" + translated_message;
        return text;
    }

    if (stage == RepairProgressStage::Remap)
        return _L("Stage 2 - Remapping color data");

    return header + (message.empty() ? wxString() : "\n" + translated_message);
}

int logarithmic_repair_stage_progress(std::chrono::steady_clock::duration elapsed)
{
    const double seconds = std::chrono::duration<double>(elapsed).count();
    if (seconds <= 0.0)
        return 0;

    const double log_elapsed = std::log1p(seconds / 60.0);
    const double reference = std::log1p(15.0);
    const double ratio = log_elapsed / (reference + log_elapsed * 0.2);
    return int(std::floor(std::clamp(50.0 * ratio, 0.0, 49.0)));
}

void merge_color_remap_stats(ModelRepairColorRemapStats &dst, const ModelRepairColorRemapStats &src)
{
    dst.had_color_data |= src.had_color_data;
    dst.remap_requested |= src.remap_requested;
    dst.remap_skipped |= src.remap_skipped;
    dst.remap_canceled |= src.remap_canceled;
    dst.remap_failed |= src.remap_failed;
    dst.used_fallback_rgba |= src.used_fallback_rgba;
    dst.volumes_remapped += src.volumes_remapped;
    dst.volumes_cleared += src.volumes_cleared;
}

void clear_remappable_color_data(ModelVolume &volume, ModelRepairColorRemapStats &stats)
{
    apply_simplify_texture_data_result(volume, SimplifyTextureDataResult());
    ++stats.volumes_cleared;
}

bool result_preserved_color(const SimplifyTextureDataSnapshot &snapshot, const SimplifyTextureDataResult &result)
{
    switch (snapshot.source) {
    case SimplifyColorSource::RgbaData:
        return result.source == SimplifyColorSource::RgbaData && result.rgba_data != nullptr;
    case SimplifyColorSource::ImageTexture:
        return result.source == SimplifyColorSource::ImageTexture ||
               (result.source == SimplifyColorSource::RgbaData && result.rgba_data != nullptr);
    case SimplifyColorSource::VertexColors:
        return result.source == SimplifyColorSource::VertexColors && !result.vertex_colors_rgba.empty();
    case SimplifyColorSource::None:
        break;
    }
    return false;
}

bool remap_or_clear_color_data(ModelVolume                         &volume,
                               const SimplifyTextureDataSnapshot   &snapshot,
                               bool                                 remap_requested,
                               std::atomic<bool>                   &cancel_requested,
                               const std::function<void(int)>      &status_fn,
                               ModelRepairColorRemapStats          &stats)
{
    if (!color_snapshot_is_valid(snapshot))
        return false;

    if (!remap_requested) {
        stats.remap_skipped = true;
        clear_remappable_color_data(volume, stats);
        return false;
    }

    try {
        SimplifyTextureDataResult result = remap_simplify_texture_data(
            snapshot,
            volume.mesh().its,
            [&cancel_requested]() {
                if (cancel_requested)
                    throw ColorRemapCanceledException();
            },
            status_fn);

        stats.used_fallback_rgba |= result.used_fallback_rgba;
        stats.remap_failed |= result.remap_failed && !result.used_fallback_rgba;
        const bool preserved = result_preserved_color(snapshot, result);
        apply_simplify_texture_data_result(volume, std::move(result));
        if (preserved) {
            ++stats.volumes_remapped;
            return true;
        } else {
            stats.remap_failed = true;
            ++stats.volumes_cleared;
        }
    } catch (ColorRemapCanceledException &) {
        stats.remap_canceled = true;
        throw;
    } catch (std::exception &) {
        stats.remap_failed = true;
        clear_remappable_color_data(volume, stats);
    }
    return false;
}

bool weld_same_position_vertices(TriangleMesh &mesh)
{
    RepairedMeshErrors repaired_errors = mesh.stats().repaired_errors;
    const int merged_vertices = its_merge_vertices(mesh.its, false);
    if (merged_vertices == 0)
        return false;

    const int removed_faces = its_remove_degenerate_faces(mesh.its, false);
    its_compactify_vertices(mesh.its, true);
    repaired_errors.edges_fixed += merged_vertices;
    repaired_errors.degenerate_facets += removed_faces;

    indexed_triangle_set welded_its = std::move(mesh.its);
    mesh = TriangleMesh(std::move(welded_its), repaired_errors);
    return true;
}

} // namespace

// Orca: Exception class for handling user-initiated cancellation of model repair operations.
class RepairCanceledException : public std::exception {
public:
    const char* what() const noexcept override { return "Model repair has been canceled"; }
};

// Orca: Main function to repair model objects using CGAL, with progress dialog and cancellation support.
// Returns false if fixing was canceled. fix_result contains error message if failed.
bool fix_model_with_cgal_gui(ModelObject                &model_object,
                             int                         volume_idx,
                             GUI::ProgressDialog        &progress_dialog,
                             const wxString             &msg_header,
                             std::string                &fix_result,
                             ModelRepairColorRemapStats *color_remap_stats,
                             ModelRepairPromptState     *prompt_state)
{
    ModelRepairColorRemapStats local_color_stats;
    ModelRepairPromptState local_prompt_state;
    ModelRepairPromptState &active_prompt_state = prompt_state != nullptr ? *prompt_state : local_prompt_state;

    if (active_prompt_state.repair_options_canceled)
        return false;

    if (!active_prompt_state.repair_options_selected) {
        const bool show_color_remap_option = target_has_remappable_color_data(model_object, volume_idx);
        if (!ask_repair_options(progress_dialog,
                                active_prompt_state.repair_options,
                                active_prompt_state.color_remap_choice,
                                show_color_remap_option)) {
            active_prompt_state.repair_options_canceled = true;
            return false;
        }
        active_prompt_state.repair_options_selected = true;
    }

    const ModelRepairOptions repair_options = active_prompt_state.repair_options;

    ColorSnapshotByVolume color_snapshots = collect_color_snapshots(model_object, volume_idx, repair_options);
    ModelRepairColorRemapChoice color_remap_choice = active_prompt_state.color_remap_choice;
    if (!color_snapshots.empty()) {
        local_color_stats.had_color_data = true;
        if (color_remap_choice == ModelRepairColorRemapChoice::Cancel) {
            if (color_remap_stats)
                merge_color_remap_stats(*color_remap_stats, local_color_stats);
            return false;
        }
        local_color_stats.remap_requested = color_remap_choice == ModelRepairColorRemapChoice::Remap;
        local_color_stats.remap_skipped = color_remap_choice == ModelRepairColorRemapChoice::Skip;
    }

    // Orca: Synchronization primitives for progress updates between worker thread and GUI.
    std::mutex mtx;
    std::condition_variable condition;
    struct Progress {
        std::string         message;
        RepairProgressStage stage    = RepairProgressStage::Repair;
        int                 percent  = 0;
        bool                updated  = false;
    } progress;

    std::atomic<bool> cancel_requested = false;
    std::atomic<bool> finished = false;
    std::atomic<bool> repair_canceled = false;

    bool   success = false;
    size_t ivolume = 0;
    const bool remap_requested = color_remap_choice == ModelRepairColorRemapChoice::Remap;

    // Orca: Lambda for updating progress from worker thread.
    auto on_progress = [&mtx, &condition, &ivolume, &model_object, &progress](RepairProgressStage stage, const char *msg, unsigned prcnt) {
        std::unique_lock<std::mutex> lock(mtx);
        progress.message = msg;
        progress.stage = stage;
        const size_t total = std::max<size_t>(1, model_object.volumes.size());
        const float stage_percent = (float(prcnt) + float(ivolume) * 100.f) / float(total);
        switch (stage) {
        case RepairProgressStage::Repair:
            progress.percent = int(std::ceil(stage_percent * 0.5f));
            break;
        case RepairProgressStage::Remap:
            progress.percent = 50 + int(std::ceil(stage_percent * 0.5f));
            break;
        case RepairProgressStage::Status:
            progress.percent = prcnt >= 100 ? 100 : int(std::ceil(stage_percent));
            break;
        }
        progress.percent = std::clamp(progress.percent, 0, 100);
        if (prcnt > 0 && progress.percent == 0)
            progress.percent = 1;
        progress.updated = true;
        condition.notify_all();
    };

    // Orca: Worker thread that performs the actual model repair operations.
    auto worker_thread = std::thread([&model_object,
                                      volume_idx,
                                      &ivolume,
                                      on_progress,
                                      &success,
                                      &cancel_requested,
                                      &finished,
                                      &repair_canceled,
                                      &fix_result,
                                      &local_color_stats,
                                      remap_requested,
                                      repair_options,
                                      color_snapshots = std::move(color_snapshots)]() {
        try {
            size_t start_volume = volume_idx == -1 ? 0 : size_t(volume_idx);
            size_t end_volume   = volume_idx == -1 ? std::numeric_limits<size_t>::max() : size_t(volume_idx);

            for (ivolume = start_volume; ivolume < model_object.volumes.size(); ++ivolume) {
                if (volume_idx != -1 && ivolume > end_volume)
                    break;
                if (cancel_requested)
                    throw RepairCanceledException();

                on_progress(RepairProgressStage::Repair, L("Repairing model object"), 15);

                ModelVolume *volume = model_object.volumes[ivolume];
                const auto color_snapshot_it = color_snapshots.find(volume);
                const SimplifyTextureDataSnapshot *color_snapshot =
                    color_snapshot_it == color_snapshots.end() ? nullptr : &color_snapshot_it->second;

                // Orca: Split splittable volumes into parts for individual processing.
                bool pre_split_weld_changed = false;
                if (repair_options.split_before_repair && repair_options.weld_same_position_vertices) {
                    TriangleMesh welded_mesh = volume->mesh();
                    pre_split_weld_changed = weld_same_position_vertices(welded_mesh);
                    if (pre_split_weld_changed)
                        volume->set_mesh(std::move(welded_mesh));
                }

                size_t parts_count = 1;
                if (repair_options.split_before_repair && volume->mesh().is_splittable()) {
                    parts_count = volume->split(1);
                    if (parts_count > 1) {
                        const std::string msg = Slic3r::format(L("Split into %1% parts"), parts_count);
                        on_progress(RepairProgressStage::Repair, msg.c_str(), 15);
                    }
                }

                const bool split_changed = parts_count > 1;
                size_t part_end = std::min(ivolume + parts_count - 1, model_object.volumes.size() - 1);
                if (volume_idx != -1)
                    end_volume = part_end;

                size_t removed_parts = 0;
                if (repair_options.split_before_repair) {
                    for (size_t idx = part_end + 1; idx > ivolume; --idx) {
                        const size_t part_idx = idx - 1;
                        const ModelVolume *part_volume = model_object.volumes[part_idx];
                        if (!is_not_3dimensional_part(part_volume->mesh()))
                            continue;

                        model_object.delete_volume(part_idx);
                        ++removed_parts;
                        if (part_end > 0)
                            --part_end;
                        else
                            part_end = 0;
                        if (volume_idx != -1)
                            end_volume = part_end;
                    }
                }

                if (removed_parts >= parts_count) {
                    ivolume = part_end;
                    on_progress(RepairProgressStage::Repair, L("Repair finished"), 100);
                    continue;
                }

                std::vector<ColorRemapPartState> color_part_states(part_end - ivolume + 1, ColorRemapPartState::Unchanged);
                if (split_changed && color_snapshot != nullptr)
                    std::fill(color_part_states.begin(), color_part_states.end(), ColorRemapPartState::Pending);
                else if (pre_split_weld_changed && color_snapshot != nullptr && !color_part_states.empty())
                    color_part_states.front() = ColorRemapPartState::Pending;

                auto clear_pending_color_parts = [&model_object, &local_color_stats, ivolume, part_end, &color_part_states]() {
                    for (size_t part_idx = ivolume; part_idx <= part_end && part_idx < model_object.volumes.size(); ++part_idx) {
                        ColorRemapPartState &state = color_part_states[part_idx - ivolume];
                        if (state != ColorRemapPartState::Pending)
                            continue;
                        clear_remappable_color_data(*model_object.volumes[part_idx], local_color_stats);
                        state = ColorRemapPartState::Cleared;
                    }
                };

                try {
                    for (size_t part_idx = ivolume; part_idx <= part_end && part_idx < model_object.volumes.size(); ++part_idx) {
                        ModelVolume *part_volume = model_object.volumes[part_idx];
                        TriangleMesh mesh = part_volume->mesh();
                        bool part_changed = split_changed || (part_idx == ivolume && pre_split_weld_changed);
                        bool mesh_changed = false;
                        if (repair_options.weld_same_position_vertices && !repair_options.split_before_repair)
                            mesh_changed = weld_same_position_vertices(mesh);
                        if (its_num_open_edges(mesh.its) != 0) {
                            std::string error;
                            if (!MeshBoolean::cgal::repair(mesh, nullptr, &error))
                                throw Slic3r::RuntimeError(error.empty() ? L("Repair failed") : error.c_str());

                            mesh_changed = true;
                        }

                        if (mesh_changed) {
                            part_volume->set_mesh(std::move(mesh));
                            part_changed = true;
                        }

                        if (part_changed && color_snapshot != nullptr) {
                            ColorRemapPartState &state = color_part_states[part_idx - ivolume];
                            if (state == ColorRemapPartState::Unchanged)
                                state = ColorRemapPartState::Pending;
                            const bool preserved = remap_or_clear_color_data(
                                *part_volume,
                                *color_snapshot,
                                remap_requested,
                                cancel_requested,
                                [on_progress](int percent) {
                                    const int clamped_percent = std::clamp(percent, 0, 100);
                                    on_progress(RepairProgressStage::Remap, L("Remapping color data"), unsigned(clamped_percent));
                                },
                                local_color_stats);
                            state = preserved ? ColorRemapPartState::Preserved : ColorRemapPartState::Cleared;
                        }

                        if (part_changed) {
                            part_volume->calculate_convex_hull();
                            part_volume->invalidate_convex_hull_2d();
                            part_volume->set_new_unique_id();
                        }
                    }
                } catch (ColorRemapCanceledException &) {
                    clear_pending_color_parts();
                    throw;
                } catch (std::exception &) {
                    clear_pending_color_parts();
                    throw;
                }

                ivolume = part_end;

                on_progress(RepairProgressStage::Repair, L("Repair finished"), 100);
            }

            model_object.invalidate_bounding_box();

            if (ivolume > 0)
                --ivolume;
            on_progress(RepairProgressStage::Status, L("Repair finished"), 100);
            success = true;
            finished = true;
        } catch (ColorRemapCanceledException &) {
            success = true;
            finished = true;
            on_progress(RepairProgressStage::Status, L("Color remap canceled"), 100);
        } catch (RepairCanceledException &) {
            repair_canceled = true;
            finished = true;
            on_progress(RepairProgressStage::Status, L("Repair canceled"), 100);
        } catch (std::exception &ex) {
            success = false;
            finished = true;
            fix_result = ex.what();
            on_progress(RepairProgressStage::Status, ex.what(), 100);
        }
    });

    auto repair_stage_started = std::chrono::steady_clock::now();
    RepairProgressStage last_stage = RepairProgressStage::Repair;
    int last_shown_percent = 0;

    // Orca: Main GUI loop to update progress dialog and handle cancellation.
    while (!finished) {
        std::string progress_message;
        RepairProgressStage progress_stage = RepairProgressStage::Repair;
        int progress_percent = 0;
        {
            std::unique_lock<std::mutex> lock(mtx);
            condition.wait_for(lock, std::chrono::milliseconds(250), [&progress]{ return progress.updated; });

            progress_message = progress.message;
            progress_stage = progress.stage;
            progress_percent = progress.percent;
            progress.updated = false;
        }

        if (progress_stage != last_stage) {
            last_stage = progress_stage;
            if (progress_stage == RepairProgressStage::Repair)
                repair_stage_started = std::chrono::steady_clock::now();
        }

        if (progress_stage == RepairProgressStage::Repair) {
            const int fallback_progress = logarithmic_repair_stage_progress(std::chrono::steady_clock::now() - repair_stage_started);
            progress_percent = std::max(progress_percent, fallback_progress);
            progress_percent = std::min(progress_percent, 50);
        } else if (progress_stage == RepairProgressStage::Remap)
            progress_percent = std::max(progress_percent, 50);

        int shown_percent = std::clamp(progress_percent, 0, 100);
        if (progress_stage != RepairProgressStage::Status)
            shown_percent = std::max(shown_percent, last_shown_percent);
        if (shown_percent >= 100)
            shown_percent = 99;
        last_shown_percent = shown_percent;

        if (!progress_dialog.Update(shown_percent, compose_progress_message(msg_header, progress_stage, progress_message)))
            cancel_requested = true;
        else
            progress_dialog.Fit();
    }

    if (worker_thread.joinable())
        worker_thread.join();

    if (local_color_stats.remap_canceled && !repair_canceled)
        progress_dialog.Resume();

    if (!repair_canceled && success)
        fix_result.clear();

    if (color_remap_stats)
        merge_color_remap_stats(*color_remap_stats, local_color_stats);

    return !repair_canceled;
}

} // namespace Slic3r
