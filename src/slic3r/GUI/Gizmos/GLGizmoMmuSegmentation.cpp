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
#include "slic3r/GUI/ObjColorDialog.hpp"
#include "slic3r/GUI/MainFrame.hpp"
#include "slic3r/GUI/Tab.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/TextureMapping.hpp"
#include "libslic3r/ColorSolver.hpp"
#include "libslic3r/Geometry.hpp"
#include "slic3r/Utils/UndoRedo.hpp"
#include "GLGizmosCommon.hpp"
#include "GLGizmoUtils.hpp"


#include <glad/gl.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <boost/log/trivial.hpp>
#include <nlohmann/json.hpp>
#include <wx/button.h>
#include <wx/colordlg.h>
#include <wx/dialog.h>
#include <wx/filedlg.h>
#include <wx/image.h>
#include <wx/menu.h>
#include <wx/panel.h>
#include <wx/sizer.h>
#include <wx/spinctrl.h>
#include <wx/statline.h>
#include <wx/stattext.h>

namespace Slic3r::GUI {

namespace {
constexpr size_t SlopePreviewOverrideIdCount = size_t(EnforcerBlockerType::ExtruderMax) + 1;
constexpr float SlopeAutoPaintMaxEdgeMm = 0.03f;
constexpr int SlopeAutoPaintMaxDepth = 13;
constexpr float SlopeAutoPaintMaxAngleDeg = 180.f;
}

struct ManagedRegionColorSource
{
    std::vector<std::vector<TriangleSelector::FacetStateTriangle>> triangles_per_type;
    std::vector<std::unordered_map<int, std::vector<size_t>>>      by_source_triangle;
    std::vector<ColorRGBA>                                         state_colors;
    std::vector<bool>                                              state_is_texture_mapping_zone;
    std::vector<std::vector<int>>                                  nearby_source_triangles;
    float                                                          nearby_painted_distance_sq = 0.f;
};

static std::optional<ColorRGBA> sample_managed_region_color_source(const ManagedRegionColorSource &source,
                                                                   int                             tri_idx,
                                                                   const Vec3f                    &point);
static ColorRGBA managed_color_data_state_color(const ManagedRegionColorSource &source, unsigned int state);
static ManagedRegionColorSource build_managed_region_color_source(const ModelVolume &volume);
static void render_extruders_combo(const std::string& label,
                                   const std::vector<std::string>& extruders,
                                   const std::vector<ColorRGBA>& extruders_colors,
                                   size_t& selection_idx);
static std::unique_ptr<ColorFacetsAnnotation> build_rgba_data_from_color_regions(const ModelVolume &volume, const ColorRGBA &fallback_color);
static std::unique_ptr<ColorFacetsAnnotation> build_rgba_data_from_color_region_data(
    const TriangleSelector::TriangleSplittingData &data,
    size_t                                         triangle_count,
    const ManagedRegionColorSource                &source,
    const ColorRGBA                               &fallback_color,
    const std::function<void()>                   &check_cancel = {});
static bool managed_color_data_replace_rgba(ColorFacetsAnnotation &annotation, uint32_t source_rgba, uint32_t target_rgba);
static ColorRGBA managed_color_data_background_color(const ModelObject *object);

static void call_after_if_true_color_painting_active(std::function<void()> fn, GUI_App *app = &wxGetApp())
{
    if (app == nullptr)
        return;
    app->CallAfter([fn = std::move(fn), app]() {
        const Plater *plater = app->plater();
        if (plater == nullptr)
            return;
        const GLCanvas3D *canvas = plater->canvas3D();
        if (canvas == nullptr)
            return;
        if (canvas->get_gizmos_manager().get_current_type() != GLGizmosManager::TrueColorPainting)
            return;
        fn();
    });
}

struct TrueColorRgbDataConversionVolumeSnapshot
{
    ObjectID              volume_id;
    indexed_triangle_set  its;
    std::vector<uint32_t> imported_vertex_colors_rgba;
    std::vector<float>    imported_texture_uvs_per_face;
    std::vector<uint8_t>  imported_texture_uv_valid;
    std::vector<uint8_t>  imported_texture_rgba;
    uint32_t              imported_texture_width = 0;
    uint32_t              imported_texture_height = 0;
    TriangleSelector::TriangleSplittingData mmu_segmentation_data;
    std::shared_ptr<ManagedRegionColorSource> region_source;
};

struct TrueColorRgbDataConversionResult
{
    ObjectID                                           object_id;
    std::vector<ObjectID>                              volume_ids;
    std::vector<std::unique_ptr<ColorFacetsAnnotation>> rgb_data;
    bool                                               canceled = false;
};

struct GLGizmoTrueColorPainting::RgbDataConversionState
{
    std::atomic_bool                  cancel { false };
    mutable std::mutex                mutex;
    bool                              finished = false;
    TrueColorRgbDataConversionResult result;
};

class TrueColorRgbDataConversionCanceledException : public std::exception
{
};

static inline void show_notification_extruders_limit_exceeded()
{
    wxGetApp()
        .plater()
        ->get_notification_manager()
        ->push_notification(NotificationType::MmSegmentationExceededExtrudersLimit, NotificationManager::NotificationLevel::PrintInfoNotificationLevel,
                            GUI::format(_L("Filament count exceeds the maximum number that painting tool supports. Only the "
                                           "first %1% filaments will be available in painting tool."), GLGizmoMmuSegmentation::EXTRUDERS_LIMIT));
}

static std::vector<ColorRGBA> get_extruders_colors()
{
    return wxGetApp().plater() != nullptr ? wxGetApp().plater()->get_extruders_colors() : std::vector<ColorRGBA>{};
}

static size_t extruder_color_index_for_filament_id(unsigned int filament_id, size_t color_count)
{
    if (color_count == 0)
        return 0;

    if (filament_id >= 1 && filament_id <= color_count)
        return size_t(filament_id - 1);

    if (wxGetApp().preset_bundle != nullptr) {
        const size_t       physical_count = size_t(std::max(wxGetApp().filaments_cnt(), 0));
        const unsigned int resolved_id =
            wxGetApp().preset_bundle->texture_mapping_zones.resolve_zone_component(filament_id, physical_count, 0);
        if (resolved_id >= 1 && resolved_id <= color_count)
            return size_t(resolved_id - 1);
    }

    return 0;
}

void GLGizmoMmuSegmentation::on_opening()
{
    if (get_extruders_colors().size() > GLGizmoMmuSegmentation::EXTRUDERS_LIMIT)
        show_notification_extruders_limit_exceeded();
}

void GLGizmoMmuSegmentation::on_shutdown()
{
    m_show_slope_auto_paint_overlay = false;
    m_slope_auto_paint_preview_active = false;
    m_parent.use_slope(false);
    m_parent.toggle_model_objects_visibility(true);
}

std::string GLGizmoMmuSegmentation::on_get_name() const
{
    return _u8L("Color Region Painting");
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

static std::vector<unsigned int> get_display_filament_ids(size_t total_filaments)
{
    std::vector<unsigned int> ordered_filament_ids;
    if (wxGetApp().plater() != nullptr)
        ordered_filament_ids = wxGetApp().plater()->sidebar().get_ui_ordered_filament_ids();

    std::vector<unsigned int> sanitized_filament_ids;
    sanitized_filament_ids.reserve(total_filaments);
    std::vector<bool> used_filament_ids(total_filaments + 1, false);
    const size_t physical_count = size_t(std::max(wxGetApp().filaments_cnt(), 0));
    auto real_filament_id = [physical_count](unsigned int filament_id) {
        if (filament_id >= 1 && filament_id <= physical_count)
            return true;
        return wxGetApp().preset_bundle != nullptr &&
               wxGetApp().preset_bundle->texture_mapping_zones.is_texture_mapping_zone_id(filament_id);
    };
    for (const unsigned int filament_id : ordered_filament_ids) {
        if (filament_id == 0 || filament_id > total_filaments || used_filament_ids[filament_id] || !real_filament_id(filament_id))
            continue;
        used_filament_ids[filament_id] = true;
        sanitized_filament_ids.emplace_back(filament_id);
    }

    for (unsigned int filament_id = 1; filament_id <= total_filaments; ++filament_id) {
        if (!used_filament_ids[filament_id] && real_filament_id(filament_id))
            sanitized_filament_ids.emplace_back(filament_id);
    }

    return sanitized_filament_ids;
}

static size_t display_filament_index_for_requested_id(const std::vector<unsigned int> &display_filament_ids,
                                                      size_t                          current_idx,
                                                      int                             requested_id)
{
    if (display_filament_ids.empty())
        return 0;

    const unsigned int requested = requested_id <= 0 ? 1u : unsigned(requested_id);
    auto exact_it = std::find(display_filament_ids.begin(), display_filament_ids.end(), requested);
    if (exact_it != display_filament_ids.end())
        return size_t(std::distance(display_filament_ids.begin(), exact_it));

    std::vector<unsigned int> sorted_ids = display_filament_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());

    const unsigned int current_id =
        current_idx < display_filament_ids.size() ? display_filament_ids[current_idx] : display_filament_ids.front();
    unsigned int selected_id = sorted_ids.front();
    if (requested > current_id) {
        auto it = std::lower_bound(sorted_ids.begin(), sorted_ids.end(), requested);
        selected_id = it == sorted_ids.end() ? sorted_ids.back() : *it;
    } else {
        auto it = std::upper_bound(sorted_ids.begin(), sorted_ids.end(), requested);
        if (it == sorted_ids.begin())
            selected_id = sorted_ids.front();
        else
            selected_id = *(--it);
    }

    auto selected_it = std::find(display_filament_ids.begin(), display_filament_ids.end(), selected_id);
    return selected_it != display_filament_ids.end() ? size_t(std::distance(display_filament_ids.begin(), selected_it)) : 0;
}

static bool is_texture_mapping_filament_id(unsigned int filament_id)
{
    return wxGetApp().preset_bundle != nullptr &&
           wxGetApp().preset_bundle->texture_mapping_zones.is_texture_mapping_zone_id(filament_id);
}

static wxString slope_auto_paint_filament_label(unsigned int filament_id)
{
    return is_texture_mapping_filament_id(filament_id) ?
        wxString::Format(_L("Filament %u (Texture Mapping)"), filament_id) :
        wxString::Format(_L("Filament %u"), filament_id);
}

static bool slope_auto_paint_matches_normal(const SlopeAutoPaintSettings &settings, float world_normal_z)
{
    constexpr float normal_epsilon = 0.0001f;
    const float z = std::clamp(world_normal_z, -1.f, 1.f);
    const float top_z = float(std::cos(Geometry::deg2rad(std::clamp(settings.top_angle_deg, 0.f, SlopeAutoPaintMaxAngleDeg))));
    const float bottom_z = -float(std::cos(Geometry::deg2rad(std::clamp(settings.bottom_angle_deg, 0.f, SlopeAutoPaintMaxAngleDeg))));
    switch (settings.mode) {
    case SlopeAutoPaintMode::Top:
        return z >= top_z - normal_epsilon;
    case SlopeAutoPaintMode::Bottom:
        return z <= bottom_z + normal_epsilon;
    case SlopeAutoPaintMode::Side:
        return z <= top_z + normal_epsilon && z >= bottom_z - normal_epsilon;
    }
    return false;
}

static void persist_texture_mapping_zone_definitions(TextureMappingManager &mgr)
{
    if (wxGetApp().preset_bundle == nullptr)
        return;

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
}

static unsigned int ensure_texture_mapping_zone(bool allow_raw_values = false, bool prefer_raw_values = false)
{
    if (wxGetApp().preset_bundle == nullptr || wxGetApp().plater() == nullptr)
        return 0;

    TextureMappingManager &mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    const size_t num_physical = static_cast<size_t>(std::max(wxGetApp().filaments_cnt(), 0));
    std::vector<std::string> physical_colors = wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, false);
    physical_colors.resize(num_physical, "#26A69A");

    if (unsigned int existing_id = mgr.find_image_texture_zone_id(num_physical, allow_raw_values, prefer_raw_values); existing_id != 0) {
        if (TextureMappingZone *zone = mgr.zone_from_id(existing_id);
            zone == nullptr || !TextureMappingManager::auto_adjust_texture_component_ids(*zone, num_physical, physical_colors)) {
            return existing_id;
        }
    } else if (num_physical < 2) {
        return 0;
    } else {
        mgr.ensure_image_texture_zone(num_physical, physical_colors, allow_raw_values, prefer_raw_values);
    }

    persist_texture_mapping_zone_definitions(mgr);

    return mgr.find_image_texture_zone_id(num_physical, allow_raw_values, prefer_raw_values);
}

static void enable_texture_mapping_zone_simulated_preview(unsigned int texture_mapping_filament_id)
{
    if (texture_mapping_filament_id == 0 || wxGetApp().preset_bundle == nullptr)
        return;

    TextureMappingManager &mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    TextureMappingZone *zone = mgr.zone_from_id(texture_mapping_filament_id);
    if (zone == nullptr || zone->preview_simulate_colors)
        return;

    zone->preview_simulate_colors = true;
    persist_texture_mapping_zone_definitions(mgr);
}

static constexpr size_t MaxImageProjectionRawOffsetChannels = 4;
static constexpr float  RawOffsetDataWarningWrapEm          = 24.f;

struct RawAtlasProjectionLayout
{
    std::vector<ImageMapRawFilament> filaments;
    std::vector<std::string>         channel_keys;
    std::vector<size_t>              atlas_to_target_channel;
};

static bool model_volume_has_raw_atlas_texture_data(const ModelVolume *volume)
{
    if (volume == nullptr ||
        volume->imported_texture_width == 0 ||
        volume->imported_texture_height == 0 ||
        volume->imported_texture_raw_channels == 0 ||
        volume->imported_texture_raw_filament_offsets.empty())
        return false;
    return volume->imported_texture_raw_filament_offsets.size() >=
           size_t(volume->imported_texture_width) *
               size_t(volume->imported_texture_height) *
               size_t(volume->imported_texture_raw_channels);
}

static wxString raw_offset_data_rgba_conversion_warning_text()
{
    return _L("Painting or projecting to RGBA data on a model with raw offset data will create RGBA color data from the raw offset atlas. "
              "(raw offset data will no longer be used)");
}

static wxString raw_offset_data_image_texture_projection_warning_text()
{
    return _L("Projecting a normal image in Image Texture mode will convert the raw offset atlas into a regular RGBA image before "
              "projecting, destroying the raw offset data.");
}

static void show_raw_offset_data_converted_to_rgba_message()
{
    show_info(wxGetApp().mainframe,
              _L("Raw offset data was converted to RGBA color data. The raw offset data will no longer be shown or used when slicing."),
              _L("Raw Offset Data Converted"));
}

static void show_raw_offset_data_converted_to_rgba_image_message()
{
    show_info(wxGetApp().mainframe,
              _L("Raw offset data was converted to a regular RGBA image. (Raw offset data no longer exists)"),
              _L("Raw Offset Data Converted"));
}

static std::string raw_atlas_color_mode_name_for_keys(const std::vector<std::string> &keys)
{
    if (keys.empty())
        return "Unknown";

    bool all_standard = true;
    std::string joined;
    for (const std::string &key : keys) {
        if (key.size() != 1 || !image_map_raw_filament_is_standard_color(key)) {
            all_standard = false;
            break;
        }
        joined += key;
    }
    auto has_key = [&keys](const std::string &key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    };
    if (all_standard && keys.size() == 3 && has_key("R") && has_key("G") && has_key("B"))
        return "RGB";
    if (all_standard && keys.size() == 3 && has_key("C") && has_key("M") && has_key("Y"))
        return "CMY";
    if (all_standard && keys.size() == 4 && has_key("C") && has_key("M") && has_key("Y") && has_key("K"))
        return "CMYK";
    if (all_standard && keys.size() == 4 && has_key("C") && has_key("M") && has_key("Y") && has_key("W"))
        return "CMYW";
    if (all_standard && keys.size() == 4 && has_key("R") && has_key("G") && has_key("B") && has_key("K"))
        return "RGBK";
    if (all_standard && keys.size() == 4 && has_key("R") && has_key("G") && has_key("B") && has_key("W"))
        return "RGBW";
    if (all_standard && keys.size() == 2 && has_key("K") && has_key("W"))
        return "BW";
    if (all_standard && keys.size() == 5 && has_key("C") && has_key("M") && has_key("Y") && has_key("K") && has_key("W"))
        return "CMYKW";
    if (all_standard && keys.size() == 5 && has_key("R") && has_key("G") && has_key("B") && has_key("K") && has_key("W"))
        return "RGBKW";
    if (all_standard && !joined.empty())
        return joined;

    return "Custom " + std::to_string(keys.size()) + "-channel";
}

static std::string raw_atlas_color_mode_name_for_volume(const ModelVolume &volume)
{
    const std::vector<ImageMapRawFilament> filaments =
        image_map_raw_filaments_from_metadata_json(volume.imported_texture_raw_metadata_json, volume.imported_texture_raw_channels);
    return raw_atlas_color_mode_name_for_keys(image_map_raw_filament_channel_keys(filaments));
}

static bool add_raw_layout_channel(RawAtlasProjectionLayout &layout,
                                   const std::string        &key,
                                   const ImageMapRawFilament &filament,
                                   std::string              *error)
{
    if (std::find(layout.channel_keys.begin(), layout.channel_keys.end(), key) != layout.channel_keys.end())
        return true;
    if (layout.channel_keys.size() >= MaxImageProjectionRawOffsetChannels) {
        if (error != nullptr) {
            *error = GUI::format("This raw filament offset atlas would require more than %1% raw offset channels on the selected object.",
                                 MaxImageProjectionRawOffsetChannels);
        }
        return false;
    }
    layout.channel_keys.emplace_back(key);
    layout.filaments.emplace_back(filament);
    return true;
}

static bool raw_channel_keys_are_unique(const std::vector<std::string> &keys)
{
    for (size_t idx = 0; idx < keys.size(); ++idx)
        for (size_t other = idx + 1; other < keys.size(); ++other)
            if (keys[idx] == keys[other])
                return false;
    return true;
}

static bool raw_atlas_projection_layout_for_object(const ModelObject &object,
                                                   const ImageMapRawFilamentOffsetAtlas &atlas,
                                                   RawAtlasProjectionLayout &layout,
                                                   std::string *error)
{
    layout = {};
    if (!atlas.valid()) {
        if (error != nullptr)
            *error = "The selected raw filament offset atlas is invalid.";
        return false;
    }

    for (const ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_raw_atlas_texture_data(volume))
            continue;
        const std::vector<ImageMapRawFilament> volume_filaments =
            image_map_raw_filaments_from_metadata_json(volume->imported_texture_raw_metadata_json, volume->imported_texture_raw_channels);
        const std::vector<std::string> volume_keys = image_map_raw_filament_channel_keys(volume_filaments);
        if (!raw_channel_keys_are_unique(volume_keys)) {
            if (error != nullptr)
                *error = "The selected object's existing raw filament offset metadata has duplicate channels.";
            return false;
        }
        for (size_t channel = 0; channel < volume_keys.size(); ++channel)
            if (!add_raw_layout_channel(layout, volume_keys[channel], volume_filaments[channel], error))
                return false;
    }

    const std::vector<ImageMapRawFilament> atlas_filaments =
        image_map_raw_filaments_for_channels(atlas.filaments, atlas.channels);
    const std::vector<std::string> atlas_keys = image_map_raw_filament_channel_keys(atlas_filaments);
    if (!raw_channel_keys_are_unique(atlas_keys)) {
        if (error != nullptr)
            *error = "The selected raw filament offset atlas has duplicate channels.";
        return false;
    }

    for (size_t channel = 0; channel < atlas_keys.size(); ++channel)
        if (!add_raw_layout_channel(layout, atlas_keys[channel], atlas_filaments[channel], error))
            return false;

    layout.atlas_to_target_channel.assign(atlas_keys.size(), size_t(-1));
    for (size_t atlas_channel = 0; atlas_channel < atlas_keys.size(); ++atlas_channel) {
        const auto it = std::find(layout.channel_keys.begin(), layout.channel_keys.end(), atlas_keys[atlas_channel]);
        if (it == layout.channel_keys.end()) {
            if (error != nullptr)
                *error = "The selected raw filament offset atlas is not compatible with the selected object.";
            return false;
        }
        layout.atlas_to_target_channel[atlas_channel] = size_t(std::distance(layout.channel_keys.begin(), it));
    }
    return true;
}

static std::string raw_layout_metadata_json(uint32_t width, uint32_t height, const RawAtlasProjectionLayout &layout)
{
    nlohmann::json root;
    root["format"] = "raw_filament_offset_atlas";
    root["image"] = {
        { "width", width },
        { "height", height },
        { "channels", layout.filaments.size() }
    };
    root["filaments"] = nlohmann::json::array();
    for (size_t idx = 0; idx < layout.filaments.size(); ++idx) {
        const ImageMapRawFilament &filament = layout.filaments[idx];
        nlohmann::json entry;
        entry["slot"] = filament.slot != 0 ? filament.slot : unsigned(idx + 1);
        entry["color"] = filament.color.empty() ? "custom" : filament.color;
        if (!filament.hex.empty())
            entry["hex"] = filament.hex;
        root["filaments"].push_back(std::move(entry));
    }
    return root.dump();
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
           volume->imported_texture_rgba.size() >=
               size_t(volume->imported_texture_width) * size_t(volume->imported_texture_height) * 4 &&
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

static Vec3f texture_barycentric_for_bleed_safe_sampling(const Vec3f &barycentric,
                                                         const Vec2f &uv0,
                                                         const Vec2f &uv1,
                                                         const Vec2f &uv2,
                                                         uint32_t     width,
                                                         uint32_t     height)
{
    auto normalized_nonnegative = [](Vec3f weights) {
        weights.x() = std::max(weights.x(), 0.f);
        weights.y() = std::max(weights.y(), 0.f);
        weights.z() = std::max(weights.z(), 0.f);
        const float sum = weights.x() + weights.y() + weights.z();
        if (sum <= EPSILON)
            return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
        weights /= sum;
        return weights;
    };

    Vec3f safe = normalized_nonnegative(barycentric);
    if (width == 0 || height == 0)
        return safe;

    auto pixel_edge_length = [width, height](const Vec2f &a, const Vec2f &b) {
        const Vec2f delta = b - a;
        return std::sqrt(Slic3r::sqr(delta.x() * float(width)) + Slic3r::sqr(delta.y() * float(height)));
    };
    const float max_edge = std::max({ pixel_edge_length(uv0, uv1),
                                      pixel_edge_length(uv1, uv2),
                                      pixel_edge_length(uv2, uv0) });
    if (!std::isfinite(max_edge) || max_edge <= EPSILON)
        return safe;

    const float min_barycentric = std::min(0.08f, 0.75f / max_edge);
    if (min_barycentric <= 0.f)
        return safe;

    safe.x() = std::max(safe.x(), min_barycentric);
    safe.y() = std::max(safe.y(), min_barycentric);
    safe.z() = std::max(safe.z(), min_barycentric);
    return normalized_nonnegative(safe);
}

static ColorRGBA sample_texture_rgba_for_face_bake(const std::vector<uint8_t> &rgba,
                                                   uint32_t                    width,
                                                   uint32_t                    height,
                                                   const Vec2f                &uv0,
                                                   const Vec2f                &uv1,
                                                   const Vec2f                &uv2,
                                                   const Vec3f                &barycentric)
{
    const Vec3f safe_barycentric = texture_barycentric_for_bleed_safe_sampling(barycentric, uv0, uv1, uv2, width, height);
    const Vec2f uv = uv0 * safe_barycentric.x() + uv1 * safe_barycentric.y() + uv2 * safe_barycentric.z();
    return sample_texture_rgba_for_vertex_bake(rgba, width, height, uv);
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

static ColorRGBA unpack_vertex_color_rgba_for_conversion(uint32_t packed)
{
    return ColorRGBA(float((packed >> 24) & 0xFFu) / 255.f,
                     float((packed >> 16) & 0xFFu) / 255.f,
                     float((packed >> 8) & 0xFFu) / 255.f,
                     float(packed & 0xFFu) / 255.f);
}

static float triangle_max_edge_length(const std::array<Vec3f, 3> &vertices)
{
    return std::max({ (vertices[1] - vertices[0]).norm(),
                      (vertices[2] - vertices[1]).norm(),
                      (vertices[0] - vertices[2]).norm() });
}

static float mesh_max_axis_span(const indexed_triangle_set &its)
{
    if (its.vertices.empty())
        return 1.f;

    Vec3f min_point = its.vertices.front().cast<float>();
    Vec3f max_point = min_point;
    for (const stl_vertex &vertex : its.vertices) {
        const Vec3f point = vertex.cast<float>();
        min_point = min_point.cwiseMin(point);
        max_point = max_point.cwiseMax(point);
    }

    const Vec3f span = max_point - min_point;
    return std::max({ span.x(), span.y(), span.z(), 1.f });
}

static int texture_mapping_depth_from_span(float span, float target_span, int max_depth)
{
    if (!std::isfinite(span) || !std::isfinite(target_span) || span <= target_span || target_span <= EPSILON)
        return 0;

    return std::clamp(int(std::ceil(std::log2(span / target_span))), 0, max_depth);
}

static int texture_mapping_depth_for_budget(size_t triangle_count, int requested_max_depth, size_t max_leaf_triangles)
{
    int depth = std::clamp(requested_max_depth, 0, 7);
    while (depth > 0) {
        double leaf_count = double(std::max<size_t>(triangle_count, 1));
        for (int idx = 0; idx < depth; ++idx)
            leaf_count *= 4.0;
        if (leaf_count <= double(max_leaf_triangles))
            break;
        --depth;
    }
    return depth;
}

static constexpr float TRUE_COLOR_BRUSH_SUBDIVISION_FRACTION = 1.f / 8.f;
static constexpr float TRUE_COLOR_BRUSH_MIN_SUBDIVISION_EDGE_MM = 0.1f;
static constexpr float IMAGE_PROJECTION_RGB_TARGET_TRIANGLE_IMAGE_FRACTION = 1.f / 128.f;
static constexpr float IMAGE_PROJECTION_RGB_MIN_TARGET_TRIANGLE_IMAGE_PX = 2.f;

static float true_color_brush_subdivision_target(float brush_radius)
{
    return std::max(std::max(brush_radius, 0.f) * TRUE_COLOR_BRUSH_SUBDIVISION_FRACTION,
                    TRUE_COLOR_BRUSH_MIN_SUBDIVISION_EDGE_MM);
}

static void normalize_color_mix_weights(std::vector<float> &weights)
{
    float sum = 0.f;
    for (float &weight : weights) {
        if (!std::isfinite(weight) || weight < 0.f)
            weight = 0.f;
        sum += weight;
    }

    if (sum <= EPSILON) {
        const float uniform = weights.empty() ? 0.f : 1.f / float(weights.size());
        for (float &weight : weights)
            weight = uniform;
        return;
    }

    const float inv_sum = 1.f / sum;
    for (float &weight : weights)
        weight *= inv_sum;
}

static ColorRGBA color_mix_from_weights(const std::vector<ColorRGBA> &colors,
                                        const std::vector<float>     &weights,
                                        const ColorRGBA              &fallback)
{
    if (colors.empty() || weights.empty())
        return fallback;

    float sum = 0.f;
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    for (size_t idx = 0; idx < colors.size() && idx < weights.size(); ++idx) {
        const float weight = std::max(weights[idx], 0.f);
        sum += weight;
        r += colors[idx].r() * weight;
        g += colors[idx].g() * weight;
        b += colors[idx].b() * weight;
    }

    if (sum <= EPSILON)
        return fallback;

    const float inv_sum = 1.f / sum;
    return ColorRGBA(r * inv_sum, g * inv_sum, b * inv_sum, fallback.a());
}

static float color_mix_error_squared(const std::vector<ColorRGBA> &colors, const std::vector<float> &weights, const ColorRGBA &target)
{
    const ColorRGBA mix = color_mix_from_weights(colors, weights, target);
    return Slic3r::sqr(mix.r() - target.r()) + Slic3r::sqr(mix.g() - target.g()) + Slic3r::sqr(mix.b() - target.b());
}

static std::vector<float> closest_color_mix_weights(const std::vector<ColorRGBA> &colors, const ColorRGBA &target)
{
    std::vector<float> weights(colors.size(), 0.f);
    if (colors.empty())
        return weights;

    auto improve = [&colors, &target](std::vector<float> candidate) {
        normalize_color_mix_weights(candidate);
        float step = 0.28f;
        for (int iter = 0; iter < 140; ++iter) {
            const ColorRGBA mix = color_mix_from_weights(colors, candidate, target);
            const float err_r = mix.r() - target.r();
            const float err_g = mix.g() - target.g();
            const float err_b = mix.b() - target.b();
            std::vector<float> next = candidate;
            for (size_t idx = 0; idx < colors.size(); ++idx) {
                const float grad = 2.f * (err_r * colors[idx].r() + err_g * colors[idx].g() + err_b * colors[idx].b());
                next[idx] -= step * grad;
            }
            normalize_color_mix_weights(next);
            candidate = std::move(next);
            step *= 0.985f;
        }
        return candidate;
    };

    std::vector<float> uniform(colors.size(), 1.f / float(colors.size()));
    weights = improve(uniform);
    float best_error = color_mix_error_squared(colors, weights, target);

    for (size_t idx = 0; idx < colors.size(); ++idx) {
        std::vector<float> single(colors.size(), 0.f);
        single[idx] = 1.f;
        single = improve(std::move(single));
        const float error = color_mix_error_squared(colors, single, target);
        if (error < best_error) {
            best_error = error;
            weights = std::move(single);
        }
    }

    return weights;
}

static float texture_triangle_uv_pixel_span_from_data(const std::vector<float>   &uvs_per_face,
                                                      const std::vector<uint8_t> &uv_valid,
                                                      uint32_t                    width_px,
                                                      uint32_t                    height_px,
                                                      size_t                      tri_idx)
{
    if (tri_idx >= uv_valid.size() || uv_valid[tri_idx] == 0)
        return 0.f;

    const size_t uv_offset = tri_idx * 6;
    if (uv_offset + 5 >= uvs_per_face.size())
        return 0.f;

    const Vec2f uv0(uvs_per_face[uv_offset + 0], uvs_per_face[uv_offset + 1]);
    const Vec2f uv1(uvs_per_face[uv_offset + 2], uvs_per_face[uv_offset + 3]);
    const Vec2f uv2(uvs_per_face[uv_offset + 4], uvs_per_face[uv_offset + 5]);
    const float width = float(std::max<uint32_t>(width_px, 1));
    const float height = float(std::max<uint32_t>(height_px, 1));
    auto pixel_edge_length = [width, height](const Vec2f &a, const Vec2f &b) {
        const Vec2f delta = b - a;
        return std::sqrt(Slic3r::sqr(delta.x() * width) + Slic3r::sqr(delta.y() * height));
    };

    return std::max({ pixel_edge_length(uv0, uv1),
                      pixel_edge_length(uv1, uv2),
                      pixel_edge_length(uv2, uv0) });
}

static float texture_triangle_uv_pixel_span(const ModelVolume *volume, size_t tri_idx)
{
    if (volume == nullptr)
        return 0.f;

    return texture_triangle_uv_pixel_span_from_data(volume->imported_texture_uvs_per_face,
                                                   volume->imported_texture_uv_valid,
                                                   volume->imported_texture_width,
                                                   volume->imported_texture_height,
                                                   tri_idx);
}

static float texture_triangle_uv_pixel_span(const TrueColorRgbDataConversionVolumeSnapshot &snapshot, size_t tri_idx)
{
    return texture_triangle_uv_pixel_span_from_data(snapshot.imported_texture_uvs_per_face,
                                                   snapshot.imported_texture_uv_valid,
                                                   snapshot.imported_texture_width,
                                                   snapshot.imported_texture_height,
                                                   tri_idx);
}

static std::optional<ColorRGBA> sample_image_texture_rgba_for_conversion(const std::vector<uint8_t>  &rgba,
                                                                         uint32_t                     width,
                                                                         uint32_t                     height,
                                                                         const std::vector<float>    &uvs_per_face,
                                                                         const std::vector<uint8_t>  &uv_valid,
                                                                         size_t                       tri_idx,
                                                                         const Vec3f                 &barycentric)
{
    if (tri_idx >= uv_valid.size() || uv_valid[tri_idx] == 0)
        return std::nullopt;

    const size_t uv_offset = tri_idx * 6;
    if (uv_offset + 5 >= uvs_per_face.size())
        return std::nullopt;

    const Vec2f uv0(uvs_per_face[uv_offset + 0], uvs_per_face[uv_offset + 1]);
    const Vec2f uv1(uvs_per_face[uv_offset + 2], uvs_per_face[uv_offset + 3]);
    const Vec2f uv2(uvs_per_face[uv_offset + 4], uvs_per_face[uv_offset + 5]);
    return sample_texture_rgba_for_face_bake(rgba, width, height, uv0, uv1, uv2, barycentric);
}

static std::optional<ColorRGBA> sample_vertex_colors_rgba_for_conversion(const indexed_triangle_set  &its,
                                                                         const std::vector<uint32_t> &vertex_colors,
                                                                         size_t                      tri_idx,
                                                                         const Vec3f                &barycentric)
{
    if (vertex_colors.size() != its.vertices.size() || tri_idx >= its.indices.size())
        return std::nullopt;

    const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
    if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0 ||
        size_t(tri[0]) >= vertex_colors.size() ||
        size_t(tri[1]) >= vertex_colors.size() ||
        size_t(tri[2]) >= vertex_colors.size())
        return std::nullopt;

    const ColorRGBA c0 = unpack_vertex_color_rgba_for_conversion(vertex_colors[size_t(tri[0])]);
    const ColorRGBA c1 = unpack_vertex_color_rgba_for_conversion(vertex_colors[size_t(tri[1])]);
    const ColorRGBA c2 = unpack_vertex_color_rgba_for_conversion(vertex_colors[size_t(tri[2])]);
    return ColorRGBA(c0.r() * barycentric.x() + c1.r() * barycentric.y() + c2.r() * barycentric.z(),
                     c0.g() * barycentric.x() + c1.g() * barycentric.y() + c2.g() * barycentric.z(),
                     c0.b() * barycentric.x() + c1.b() * barycentric.y() + c2.b() * barycentric.z(),
                     c0.a() * barycentric.x() + c1.a() * barycentric.y() + c2.a() * barycentric.z());
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

static std::string rgb_metadata_json(const ColorRGBA &background)
{
    const uint32_t packed = pack_vertex_color_rgba(background);
    char buffer[48];
    std::snprintf(buffer,
                  sizeof(buffer),
                  "{\"background_color\":\"#%02X%02X%02X%02X\"}",
                  unsigned((packed >> 24) & 0xFFu),
                  unsigned((packed >> 16) & 0xFFu),
                  unsigned((packed >> 8) & 0xFFu),
                  unsigned(packed & 0xFFu));
    return buffer;
}

static ColorRGBA rgba_data_conversion_fallback_color()
{
    return ColorRGBA(1.f, 1.f, 1.f, 0.f);
}

static ColorRGBA rgb_metadata_background_color(const ColorFacetsAnnotation &annotation)
{
    const std::string &metadata = annotation.metadata_json();
    const std::string key = "\"background_color\":\"#";
    const size_t start = metadata.find(key);
    if (start == std::string::npos || start + key.size() + 8 > metadata.size())
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    uint32_t packed = 0;
    for (size_t idx = 0; idx < 8; ++idx) {
        const char ch = metadata[start + key.size() + idx];
        const int value = ch >= '0' && ch <= '9' ? ch - '0' :
                          ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 :
                          ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
        if (value < 0)
            return ColorRGBA(1.f, 1.f, 1.f, 1.f);
        packed = (packed << 4) | uint32_t(value);
    }
    return unpack_vertex_color_rgba_for_conversion(packed);
}

static bool set_texture_mapping_background_config(ModelConfigObject &config, const ColorRGBA &color)
{
    const uint32_t packed = pack_vertex_color_rgba(color);
    char buffer[16];
    std::snprintf(buffer,
                  sizeof(buffer),
                  "#%02X%02X%02X%02X",
                  unsigned((packed >> 24) & 0xFFu),
                  unsigned((packed >> 16) & 0xFFu),
                  unsigned((packed >> 8) & 0xFFu),
                  unsigned(packed & 0xFFu));
    const std::string value(buffer);
    if (const ConfigOptionString *opt = dynamic_cast<const ConfigOptionString *>(config.option("texture_mapping_background_color"));
        opt != nullptr && opt->value == value)
        return false;
    config.set("texture_mapping_background_color", value);
    return true;
}

static bool texture_mapping_background_color_config_present(const ModelConfigObject &config)
{
    return config.has("texture_mapping_background_color");
}

static bool texture_mapping_background_color_metadata_present(const ColorFacetsAnnotation &annotation)
{
    const std::string &metadata = annotation.metadata_json();
    return metadata.find("\"background_color\":\"#") != std::string::npos;
}

static bool set_managed_color_data_background_color(ModelObject &object, const ColorRGBA &color)
{
    ColorRGBA background = color;
    background.a(1.f);
    ColorRGBA previous_background = managed_color_data_background_color(&object);
    previous_background.a(1.f);
    ColorRGBA transparent_background = background;
    transparent_background.a(0.f);
    const uint32_t previous_background_rgba = pack_vertex_color_rgba(previous_background);
    const uint32_t transparent_background_rgba = pack_vertex_color_rgba(transparent_background);
    bool changed = set_texture_mapping_background_config(object.config, background);
    const std::string metadata = rgb_metadata_json(background);
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        changed |= set_texture_mapping_background_config(volume->config, background);
        if (!volume->texture_mapping_color_facets.empty())
            changed |= managed_color_data_replace_rgba(volume->texture_mapping_color_facets,
                                                       previous_background_rgba,
                                                       transparent_background_rgba);
        if (!volume->texture_mapping_color_facets.empty() && volume->texture_mapping_color_facets.metadata_json() != metadata) {
            volume->texture_mapping_color_facets.set_metadata_json(metadata);
            changed = true;
        }
    }
    return changed;
}

static bool clear_texture_mapping_background_config(ModelConfigObject &config)
{
    return config.erase("texture_mapping_background_color");
}

static bool clear_managed_color_data_background_color(ModelObject &object)
{
    bool changed = clear_texture_mapping_background_config(object.config);
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        changed |= clear_texture_mapping_background_config(volume->config);
        if (texture_mapping_background_color_metadata_present(volume->texture_mapping_color_facets)) {
            volume->texture_mapping_color_facets.set_metadata_json(std::string());
            changed = true;
        }
    }
    return changed;
}

static wxColour wx_colour_from_color_rgba(const ColorRGBA &color)
{
    auto to_u8 = [](float value) {
        return static_cast<unsigned char>(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
    };
    return wxColour(to_u8(color.r()), to_u8(color.g()), to_u8(color.b()));
}

static ColorRGBA color_rgba_from_wx_colour(const wxColour &color)
{
    return ColorRGBA(float(color.Red()) / 255.f,
                     float(color.Green()) / 255.f,
                     float(color.Blue()) / 255.f,
                     1.f);
}

static bool show_background_color_dialog(wxWindow *parent,
                                         const wxColour &initial_color,
                                         wxColour &selected_color)
{
    wxColourData color_data;
    color_data.SetChooseFull(false);
    color_data.SetColour(initial_color);
    wxColourDialog dialog(parent, &color_data);
    dialog.SetTitle(_L("Background color"));
    const wxSize compact_size(parent != nullptr ? parent->FromDIP(420) : 420,
                              parent != nullptr ? parent->FromDIP(360) : 360);
    dialog.SetInitialSize(compact_size);
    dialog.SetSize(compact_size);
    dialog.CenterOnParent();

    if (dialog.ShowModal() != wxID_OK)
        return false;

    selected_color = dialog.GetColourData().GetColour();
    return selected_color.IsOk();
}

struct TextureMappingBackgroundConfigState
{
    bool        has = false;
    std::string value;
};

struct TextureMappingBackgroundConfigSnapshot
{
    TextureMappingBackgroundConfigState object;
    std::vector<std::pair<ModelVolume *, TextureMappingBackgroundConfigState>> volumes;
};

static TextureMappingBackgroundConfigState texture_mapping_background_config_state(const ModelConfigObject &config)
{
    TextureMappingBackgroundConfigState state;
    if (!config.has("texture_mapping_background_color"))
        return state;
    const ConfigOptionString *opt = dynamic_cast<const ConfigOptionString *>(config.option("texture_mapping_background_color"));
    if (opt == nullptr)
        return state;
    state.has = true;
    state.value = opt->value;
    return state;
}

static TextureMappingBackgroundConfigSnapshot snapshot_texture_mapping_background_config(ModelObject &object)
{
    TextureMappingBackgroundConfigSnapshot snapshot;
    snapshot.object = texture_mapping_background_config_state(object.config);
    snapshot.volumes.reserve(object.volumes.size());
    for (ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part())
            snapshot.volumes.emplace_back(volume, texture_mapping_background_config_state(volume->config));
    return snapshot;
}

static void restore_texture_mapping_background_config(ModelObject &object, const TextureMappingBackgroundConfigSnapshot &snapshot)
{
    if (snapshot.object.has)
        object.config.set("texture_mapping_background_color", snapshot.object.value);
    else
        clear_texture_mapping_background_config(object.config);

    for (const auto &volume_state : snapshot.volumes) {
        ModelVolume *volume = volume_state.first;
        if (volume == nullptr)
            continue;
        if (volume_state.second.has)
            volume->config.set("texture_mapping_background_color", volume_state.second.value);
        else
            clear_texture_mapping_background_config(volume->config);
    }
}

static void preview_texture_mapping_background_config(ModelObject &object, const ColorRGBA &color)
{
    set_texture_mapping_background_config(object.config, color);
    for (ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part())
            set_texture_mapping_background_config(volume->config, color);
}

static ColorRGBA managed_color_data_background_color(const ModelObject *object)
{
    if (object == nullptr)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    auto read_config_color = [](const ModelConfigObject &config) -> std::optional<ColorRGBA> {
        if (!config.has("texture_mapping_background_color"))
            return std::nullopt;
        const ConfigOptionString *opt = dynamic_cast<const ConfigOptionString *>(config.option("texture_mapping_background_color"));
        if (opt == nullptr)
            return std::nullopt;
        const std::string &text = opt->value;
        const size_t hash_pos = text.find('#');
        const size_t start = hash_pos == std::string::npos ? 0 : hash_pos + 1;
        if (start + 6 > text.size())
            return std::nullopt;
        unsigned int values[3] = { 255, 255, 255 };
        for (size_t channel = 0; channel < 3; ++channel) {
            int value = 0;
            for (size_t digit = 0; digit < 2; ++digit) {
                const char ch = text[start + channel * 2 + digit];
                const int hex = ch >= '0' && ch <= '9' ? ch - '0' :
                                ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 :
                                ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
                if (hex < 0)
                    return std::nullopt;
                value = (value << 4) | hex;
            }
            values[channel] = unsigned(value);
        }
        return ColorRGBA(float(values[0]) / 255.f, float(values[1]) / 255.f, float(values[2]) / 255.f, 1.f);
    };

    if (std::optional<ColorRGBA> color = read_config_color(object->config))
        return *color;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->texture_mapping_color_facets.empty())
            return rgb_metadata_background_color(volume->texture_mapping_color_facets);
    return ColorRGBA(1.f, 1.f, 1.f, 1.f);
}

static bool managed_color_data_has_background_color(const ModelObject *object)
{
    if (object == nullptr)
        return false;

    if (texture_mapping_background_color_config_present(object->config))
        return true;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (texture_mapping_background_color_config_present(volume->config))
            return true;
        if (texture_mapping_background_color_metadata_present(volume->texture_mapping_color_facets))
            return true;
    }

    return false;
}

static std::optional<ColorRGBA> configured_texture_mapping_background_color_for_volume(const ModelVolume &volume)
{
    auto read_config_color = [](const ModelConfigObject &config) -> std::optional<ColorRGBA> {
        if (!config.has("texture_mapping_background_color"))
            return std::nullopt;
        const ConfigOptionString *opt = dynamic_cast<const ConfigOptionString *>(config.option("texture_mapping_background_color"));
        if (opt == nullptr)
            return std::nullopt;
        const std::string &text = opt->value;
        const size_t hash_pos = text.find('#');
        const size_t start = hash_pos == std::string::npos ? 0 : hash_pos + 1;
        if (start + 6 > text.size())
            return std::nullopt;
        unsigned int values[3] = { 255, 255, 255 };
        for (size_t channel = 0; channel < 3; ++channel) {
            int value = 0;
            for (size_t digit = 0; digit < 2; ++digit) {
                const char ch = text[start + channel * 2 + digit];
                const int hex = ch >= '0' && ch <= '9' ? ch - '0' :
                                ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 :
                                ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
                if (hex < 0)
                    return std::nullopt;
                value = (value << 4) | hex;
            }
            values[channel] = unsigned(value);
        }
        return ColorRGBA(float(values[0]) / 255.f, float(values[1]) / 255.f, float(values[2]) / 255.f, 1.f);
    };

    if (std::optional<ColorRGBA> color = read_config_color(volume.config))
        return color;
    if (const ModelObject *object = volume.get_object()) {
        if (std::optional<ColorRGBA> color = read_config_color(object->config))
            return color;
    }
    if (!volume.texture_mapping_color_facets.metadata_json().empty())
        return rgb_metadata_background_color(volume.texture_mapping_color_facets);
    return std::nullopt;
}

static void refresh_imported_texture_storage(ModelVolume &volume)
{
    std::vector<uint8_t> refreshed(volume.imported_texture_rgba.begin(), volume.imported_texture_rgba.end());
    volume.imported_texture_rgba.swap(refreshed);
}

static void refresh_imported_texture_raw_storage(ModelVolume &volume)
{
    std::vector<uint8_t> refreshed(volume.imported_texture_raw_filament_offsets.begin(),
                                   volume.imported_texture_raw_filament_offsets.end());
    volume.imported_texture_raw_filament_offsets.swap(refreshed);
}

static void clear_imported_texture_raw_atlas(ModelVolume &volume)
{
    volume.imported_texture_raw_filament_offsets.clear();
    volume.imported_texture_raw_channels = 0;
    volume.imported_texture_raw_metadata_json.clear();
}

static ColorRGBA raw_filament_color_for_projection_preview(const ImageMapRawFilament &filament)
{
    const std::string key = image_map_raw_filament_channel_key(filament, 0);
    if (key == "C")
        return ColorRGBA(0.f, 0.75f, 0.75f, 1.f);
    if (key == "M")
        return ColorRGBA(0.9f, 0.f, 0.75f, 1.f);
    if (key == "Y")
        return ColorRGBA(0.95f, 0.85f, 0.f, 1.f);
    if (key == "K")
        return ColorRGBA(0.05f, 0.05f, 0.05f, 1.f);
    if (key == "W")
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    if (key == "R")
        return ColorRGBA(1.f, 0.f, 0.f, 1.f);
    if (key == "G")
        return ColorRGBA(0.f, 0.75f, 0.f, 1.f);
    if (key == "B")
        return ColorRGBA(0.f, 0.25f, 1.f, 1.f);

    unsigned char rgba[4] = { 255, 255, 255, 255 };
    if (!filament.hex.empty())
        GUI::BitmapCache::parse_color4(filament.hex, rgba);
    return ColorRGBA(float(rgba[0]) / 255.f, float(rgba[1]) / 255.f, float(rgba[2]) / 255.f, 1.f);
}

static std::vector<ColorRGBA> raw_filament_colors_for_projection_preview(const std::vector<ImageMapRawFilament> &filaments)
{
    std::vector<ColorRGBA> colors;
    colors.reserve(filaments.size());
    for (const ImageMapRawFilament &filament : filaments)
        colors.emplace_back(raw_filament_color_for_projection_preview(filament));
    return colors;
}

static std::vector<std::array<float, 3>> raw_projection_solver_component_colors(const std::vector<ColorRGBA> &filament_colors)
{
    std::vector<std::array<float, 3>> colors;
    colors.reserve(filament_colors.size());
    for (const ColorRGBA &color : filament_colors)
        colors.push_back({ color.r(), color.g(), color.b() });
    return colors;
}

struct RawOffsetProjectionPreviewSettings
{
    std::vector<std::array<float, 3>> component_colors;
    float raw_offset_base_visibility_factor = 1.f;
    float raw_offset_visibility_range_factor = 0.f;
    int generic_solver_mix_model = TextureMappingZone::DefaultGenericSolverMixModel;
};

static float raw_offset_projection_config_float(const char *key, float fallback)
{
    if (wxGetApp().preset_bundle == nullptr)
        return fallback;

    const DynamicPrintConfig &config = wxGetApp().preset_bundle->project_config;
    if (const ConfigOptionFloat *opt = config.option<ConfigOptionFloat>(key))
        return std::isfinite(opt->value) ? float(opt->value) : fallback;
    return fallback;
}

static std::array<float, 2> raw_offset_projection_visibility_factors()
{
    const float base_outer_width_mm =
        std::max(0.05f, raw_offset_projection_config_float("texture_mapping_outer_wall_gradient_max_line_width", 0.95f));
    const float min_outer_width_mm = std::clamp(raw_offset_projection_config_float("texture_mapping_outer_wall_gradient_min_line_width", 0.32f),
                                                0.05f,
                                                base_outer_width_mm);
    const float global_strength_factor =
        std::clamp(raw_offset_projection_config_float("texture_mapping_outer_wall_gradient_global_strength", 100.f) / 100.f, 0.f, 1.f);

    float width_range_mm =
        std::min((base_outer_width_mm - min_outer_width_mm) * global_strength_factor,
                 2.f * TextureMappingManager::max_component_surface_offset_mm());
    width_range_mm = std::clamp(width_range_mm, 0.f, base_outer_width_mm);

    return {
        std::clamp((base_outer_width_mm - width_range_mm) / base_outer_width_mm, 0.f, 1.f),
        std::clamp(width_range_mm / base_outer_width_mm, 0.f, 1.f)
    };
}

static RawOffsetProjectionPreviewSettings raw_offset_projection_preview_settings(const std::vector<ColorRGBA> &filament_colors,
                                                                                int                           generic_solver_mix_model)
{
    RawOffsetProjectionPreviewSettings settings;
    settings.component_colors = raw_projection_solver_component_colors(filament_colors);
    const std::array<float, 2> visibility_factors = raw_offset_projection_visibility_factors();
    settings.raw_offset_base_visibility_factor = visibility_factors[0];
    settings.raw_offset_visibility_range_factor = visibility_factors[1];
    settings.generic_solver_mix_model = generic_solver_mix_model;
    return settings;
}

static std::vector<float> raw_offset_projection_preview_weights(const RawOffsetProjectionPreviewSettings &settings,
                                                                const uint8_t                           *values,
                                                                size_t                                   value_count)
{
    const size_t component_count = settings.component_colors.size();
    if (values == nullptr || value_count == 0 || component_count == 0)
        return {};

    std::vector<float> width_factors(component_count, 0.f);
    for (size_t idx = 0; idx < component_count && idx < value_count; ++idx) {
        const float raw_visibility = std::clamp(float(values[idx]) / 255.f, 0.f, 1.f);
        width_factors[idx] = std::clamp(settings.raw_offset_base_visibility_factor +
                                        settings.raw_offset_visibility_range_factor * raw_visibility,
                                        0.f,
                                        1.f);
    }

    const auto min_width = std::min_element(width_factors.begin(), width_factors.end());
    if (min_width == width_factors.end())
        return {};

    std::vector<float> weights(component_count, 0.f);
    const float shared_width = *min_width / float(component_count);
    for (size_t idx = 0; idx < component_count; ++idx)
        weights[idx] = std::clamp(shared_width + std::max(0.f, width_factors[idx] - *min_width), 0.f, 1.f);
    return weights;
}

struct RawOffsetColorConversionSolver
{
    ColorSolverCandidateSet candidates;
    std::unordered_map<unsigned int, std::vector<uint8_t>> cache;
};

static ColorRGBA raw_projection_color_from_solver_rgb(const std::array<float, 3> &rgb, const ColorRGBA &fallback)
{
    return ColorRGBA(std::clamp(rgb[0], 0.f, 1.f),
                     std::clamp(rgb[1], 0.f, 1.f),
                     std::clamp(rgb[2], 0.f, 1.f),
                     fallback.a());
}

static ColorRGBA mix_raw_offset_projection_solver_color(const std::vector<std::array<float, 3>> &component_colors,
                                                        const std::vector<float>                &weights,
                                                        const ColorRGBA                         &fallback,
                                                        int                                      generic_solver_mix_model)
{
    if (component_colors.empty() || weights.empty())
        return fallback;

    const std::array<float, 3> mixed =
        mix_color_solver_components(component_colors, weights, color_solver_mix_model_from_index(generic_solver_mix_model));
    return raw_projection_color_from_solver_rgb(mixed, fallback);
}

static ColorRGBA mix_raw_offset_projection_color(const std::vector<ColorRGBA> &filament_colors,
                                                 const std::vector<float>     &weights,
                                                 const ColorRGBA              &fallback,
                                                 int                           generic_solver_mix_model)
{
    if (filament_colors.empty() || weights.empty())
        return fallback;

    return mix_raw_offset_projection_solver_color(raw_projection_solver_component_colors(filament_colors),
                                                 weights,
                                                 fallback,
                                                 generic_solver_mix_model);
}

static ColorRGBA raw_flat_blend_color_from_filaments(const std::vector<ColorRGBA> &filament_colors, int generic_solver_mix_model)
{
    const std::vector<float> weights(filament_colors.size(), 1.f);
    return mix_raw_offset_projection_color(filament_colors, weights, ColorRGBA(1.f, 1.f, 1.f, 1.f), generic_solver_mix_model);
}

static int raw_offset_color_conversion_total_units(size_t component_count)
{
    return component_count <= 4 ? 40 : (component_count == 5 ? 24 : 20);
}

static RawOffsetColorConversionSolver build_raw_offset_color_conversion_solver(const std::vector<ColorRGBA> &filament_colors,
                                                                               int                           generic_solver_mix_model)
{
    RawOffsetColorConversionSolver solver;
    solver.candidates =
        build_color_solver_candidates(raw_projection_solver_component_colors(filament_colors),
                                      color_solver_mix_model_from_index(generic_solver_mix_model),
                                      raw_offset_color_conversion_total_units(filament_colors.size()));
    solver.cache.reserve(4096);
    return solver;
}

static unsigned int raw_offset_color_conversion_cache_key(const ColorRGBA &color)
{
    auto to_u8 = [](float value) {
        return unsigned(std::clamp(int(std::lround(std::clamp(value, 0.f, 1.f) * 255.f)), 0, 255));
    };
    return (to_u8(color.r()) << 16) | (to_u8(color.g()) << 8) | to_u8(color.b());
}

static std::vector<uint8_t> raw_offset_values_from_color(const std::vector<ColorRGBA>   &filament_colors,
                                                         ColorRGBA                       color,
                                                         const std::optional<ColorRGBA> &background_color,
                                                         int                             generic_solver_mix_model,
                                                         RawOffsetColorConversionSolver *solver)
{
    std::vector<uint8_t> values(filament_colors.size(), 0);
    if (filament_colors.empty())
        return values;

    const float alpha = std::clamp(color.a(), 0.f, 1.f);
    if (alpha <= EPSILON && !background_color)
        return values;
    if (alpha < 1.f) {
        const ColorRGBA background =
            background_color ? *background_color : raw_flat_blend_color_from_filaments(filament_colors, generic_solver_mix_model);
        color = ColorRGBA(color.r() * alpha + background.r() * (1.f - alpha),
                          color.g() * alpha + background.g() * (1.f - alpha),
                          color.b() * alpha + background.b() * (1.f - alpha),
                          1.f);
    }
    color.a(1.f);
    if (solver != nullptr && !solver->candidates.empty()) {
        const unsigned int cache_key = raw_offset_color_conversion_cache_key(color);
        auto cached = solver->cache.find(cache_key);
        if (cached != solver->cache.end())
            return cached->second;

        const std::vector<float> weights =
            solve_color_solver_weights_for_target(solver->candidates,
                                                  { color.r(), color.g(), color.b() },
                                                  ColorSolverLookupMode::ClosestMix,
                                                  ColorSolverMode::Legacy);
        if (weights.size() == values.size()) {
            for (size_t idx = 0; idx < values.size(); ++idx)
                values[idx] = uint8_t(std::clamp(int(std::lround(std::clamp(weights[idx], 0.f, 1.f) * 255.f)), 0, 255));
            solver->cache.emplace(cache_key, values);
            return values;
        }
    }

    const std::vector<float> weights = closest_color_mix_weights(filament_colors, color);
    for (size_t idx = 0; idx < values.size() && idx < weights.size(); ++idx)
        values[idx] = uint8_t(std::clamp(int(std::lround(std::clamp(weights[idx], 0.f, 1.f) * 255.f)), 0, 255));
    return values;
}

static ColorRGBA simulated_preview_color_from_raw_offsets(const RawOffsetProjectionPreviewSettings &settings,
                                                          const uint8_t                            *values,
                                                          size_t                                   value_count,
                                                          uint8_t                                   alpha)
{
    if (values == nullptr || value_count == 0 || settings.component_colors.empty())
        return ColorRGBA(0.f, 0.f, 0.f, float(alpha) / 255.f);

    std::vector<float> weights = raw_offset_projection_preview_weights(settings, values, value_count);
    bool has_nonzero_weight = false;
    for (size_t idx = 0; idx < weights.size(); ++idx)
        has_nonzero_weight = has_nonzero_weight || weights[idx] > EPSILON;
    if (!has_nonzero_weight)
        std::fill(weights.begin(), weights.end(), 1.f);

    return mix_raw_offset_projection_solver_color(settings.component_colors,
                                                 weights,
                                                 ColorRGBA(0.f, 0.f, 0.f, float(alpha) / 255.f),
                                                 settings.generic_solver_mix_model);
}

static std::vector<uint8_t> image_projection_raw_atlas_simulated_preview_rgba(const ImageMapRawFilamentOffsetAtlas &atlas,
                                                                              int                                  generic_solver_mix_model)
{
    std::vector<uint8_t> preview;
    if (!atlas.valid())
        return preview;

    const std::vector<ImageMapRawFilament> filaments =
        image_map_raw_filaments_for_channels(atlas.filaments, atlas.channels);
    const std::vector<ColorRGBA> filament_colors = raw_filament_colors_for_projection_preview(filaments);
    const RawOffsetProjectionPreviewSettings preview_settings =
        raw_offset_projection_preview_settings(filament_colors, generic_solver_mix_model);

    const size_t pixel_count = size_t(atlas.width) * size_t(atlas.height);
    preview.assign(pixel_count * 4, 255);
    for (size_t pixel_idx = 0; pixel_idx < pixel_count; ++pixel_idx) {
        const size_t raw_idx = pixel_idx * size_t(atlas.channels);
        const uint8_t alpha = atlas.mask.size() > pixel_idx ? atlas.mask[pixel_idx] : 255;
        const ColorRGBA color =
            simulated_preview_color_from_raw_offsets(preview_settings,
                                                     raw_idx < atlas.offsets.size() ? atlas.offsets.data() + raw_idx : nullptr,
                                                     size_t(atlas.channels),
                                                     alpha);
        const size_t rgba_idx = pixel_idx * 4;
        preview[rgba_idx + 0] = uint8_t(std::clamp(color.r(), 0.f, 1.f) * 255.f + 0.5f);
        preview[rgba_idx + 1] = uint8_t(std::clamp(color.g(), 0.f, 1.f) * 255.f + 0.5f);
        preview[rgba_idx + 2] = uint8_t(std::clamp(color.b(), 0.f, 1.f) * 255.f + 0.5f);
        preview[rgba_idx + 3] = uint8_t(std::clamp(color.a(), 0.f, 1.f) * 255.f + 0.5f);
    }
    return preview;
}

static ColorRGBA preview_color_from_raw_offsets(const std::vector<uint8_t> &values, uint8_t alpha)
{
    if (values.empty())
        return ColorRGBA(0.f, 0.f, 0.f, float(alpha) / 255.f);
    if (values.size() == 1) {
        const float gray = float(values.front()) / 255.f;
        return ColorRGBA(gray, gray, gray, float(alpha) / 255.f);
    }
    return ColorRGBA(float(values[0]) / 255.f,
                     float(values.size() > 1 ? values[1] : 0) / 255.f,
                     float(values.size() > 2 ? values[2] : 0) / 255.f,
                     float(alpha) / 255.f);
}

static std::vector<uint8_t> raw_offset_pixel_values(const ModelVolume &volume, uint32_t x, uint32_t y)
{
    std::vector<uint8_t> values(size_t(volume.imported_texture_raw_channels), 0);
    if (volume.imported_texture_width == 0 || volume.imported_texture_raw_channels == 0)
        return values;
    const size_t idx =
        (size_t(y) * size_t(volume.imported_texture_width) + size_t(x)) *
        size_t(volume.imported_texture_raw_channels);
    if (idx + values.size() > volume.imported_texture_raw_filament_offsets.size())
        return values;
    std::copy(volume.imported_texture_raw_filament_offsets.begin() + idx,
              volume.imported_texture_raw_filament_offsets.begin() + idx + values.size(),
              values.begin());
    return values;
}

static bool refresh_imported_texture_preview_from_raw_offsets(ModelVolume                    &volume,
                                                              const std::vector<ColorRGBA> *filament_colors = nullptr,
                                                              int                            generic_solver_mix_model =
                                                                  TextureMappingZone::DefaultGenericSolverMixModel)
{
    if (!model_volume_has_raw_atlas_texture_data(&volume))
        return false;
    const size_t pixel_count = size_t(volume.imported_texture_width) * size_t(volume.imported_texture_height);
    bool changed = false;
    if (volume.imported_texture_rgba.size() != pixel_count * 4) {
        volume.imported_texture_rgba.assign(pixel_count * 4, 255);
        changed = true;
    }

    RawOffsetProjectionPreviewSettings preview_settings;
    if (filament_colors != nullptr && filament_colors->size() == size_t(volume.imported_texture_raw_channels))
        preview_settings = raw_offset_projection_preview_settings(*filament_colors, generic_solver_mix_model);

    std::vector<uint8_t> values(size_t(volume.imported_texture_raw_channels), 0);
    for (size_t pixel_idx = 0; pixel_idx < pixel_count; ++pixel_idx) {
        const size_t raw_idx = pixel_idx * size_t(volume.imported_texture_raw_channels);
        if (raw_idx + values.size() > volume.imported_texture_raw_filament_offsets.size())
            break;
        std::copy(volume.imported_texture_raw_filament_offsets.begin() + raw_idx,
                  volume.imported_texture_raw_filament_offsets.begin() + raw_idx + values.size(),
                  values.begin());
        const ColorRGBA preview =
            !preview_settings.component_colors.empty() ?
                simulated_preview_color_from_raw_offsets(preview_settings,
                                                         values.data(),
                                                         values.size(),
                                                         255) :
                preview_color_from_raw_offsets(values, 255);
        const uint8_t r = uint8_t(std::clamp(preview.r(), 0.f, 1.f) * 255.f + 0.5f);
        const uint8_t g = uint8_t(std::clamp(preview.g(), 0.f, 1.f) * 255.f + 0.5f);
        const uint8_t b = uint8_t(std::clamp(preview.b(), 0.f, 1.f) * 255.f + 0.5f);
        const size_t rgba_idx = pixel_idx * 4;
        if (volume.imported_texture_rgba[rgba_idx + 0] != r ||
            volume.imported_texture_rgba[rgba_idx + 1] != g ||
            volume.imported_texture_rgba[rgba_idx + 2] != b ||
            volume.imported_texture_rgba[rgba_idx + 3] != 255) {
            volume.imported_texture_rgba[rgba_idx + 0] = r;
            volume.imported_texture_rgba[rgba_idx + 1] = g;
            volume.imported_texture_rgba[rgba_idx + 2] = b;
            volume.imported_texture_rgba[rgba_idx + 3] = 255;
            changed = true;
        }
    }
    return changed;
}

static bool merge_imported_texture_raw_atlas(ModelVolume &volume, const RawAtlasProjectionLayout &layout)
{
    if (layout.filaments.empty() || volume.imported_texture_width == 0 || volume.imported_texture_height == 0)
        return false;
    const size_t expected_size =
        size_t(volume.imported_texture_width) *
        size_t(volume.imported_texture_height) *
        layout.filaments.size();
    std::vector<uint8_t> merged(expected_size, 0);
    if (model_volume_has_raw_atlas_texture_data(&volume)) {
        const std::vector<ImageMapRawFilament> old_filaments =
            image_map_raw_filaments_from_metadata_json(volume.imported_texture_raw_metadata_json, volume.imported_texture_raw_channels);
        const std::vector<std::string> old_keys = image_map_raw_filament_channel_keys(old_filaments);
        std::vector<size_t> old_to_new(size_t(volume.imported_texture_raw_channels), size_t(-1));
        for (size_t old_channel = 0; old_channel < old_keys.size() && old_channel < old_to_new.size(); ++old_channel) {
            const auto found = std::find(layout.channel_keys.begin(), layout.channel_keys.end(), old_keys[old_channel]);
            if (found != layout.channel_keys.end())
                old_to_new[old_channel] = size_t(std::distance(layout.channel_keys.begin(), found));
        }

        const size_t pixel_count = size_t(volume.imported_texture_width) * size_t(volume.imported_texture_height);
        for (size_t pixel_idx = 0; pixel_idx < pixel_count; ++pixel_idx) {
            const size_t old_base = pixel_idx * size_t(volume.imported_texture_raw_channels);
            const size_t new_base = pixel_idx * layout.filaments.size();
            for (size_t old_channel = 0; old_channel < old_to_new.size(); ++old_channel) {
                const size_t new_channel = old_to_new[old_channel];
                if (new_channel != size_t(-1) &&
                    old_base + old_channel < volume.imported_texture_raw_filament_offsets.size() &&
                    new_base + new_channel < merged.size())
                    merged[new_base + new_channel] = volume.imported_texture_raw_filament_offsets[old_base + old_channel];
            }
        }
    }

    const std::string metadata =
        raw_layout_metadata_json(volume.imported_texture_width, volume.imported_texture_height, layout);
    const uint32_t merged_channels = uint32_t(layout.filaments.size());
    const bool changed =
        volume.imported_texture_raw_channels != merged_channels ||
        volume.imported_texture_raw_filament_offsets.size() != expected_size ||
        volume.imported_texture_raw_metadata_json != metadata ||
        !std::equal(volume.imported_texture_raw_filament_offsets.begin(),
                    volume.imported_texture_raw_filament_offsets.end(),
                    merged.begin(),
                    merged.end());
    volume.imported_texture_raw_channels = merged_channels;
    volume.imported_texture_raw_metadata_json = metadata;
    volume.imported_texture_raw_filament_offsets = std::move(merged);
    return changed;
}

static std::optional<ColorRGBA> sample_rgb_color_facets(const std::vector<ColorFacetTriangle>                 &facets,
                                                        const std::unordered_map<int, std::vector<size_t>>    &facets_by_source_triangle,
                                                        int                                                    source_triangle,
                                                        const Vec3f                                           &point)
{
    auto found = facets_by_source_triangle.find(source_triangle);
    if (found == facets_by_source_triangle.end())
        return std::nullopt;

    const float tolerance = -1e-4f;
    for (const size_t facet_idx : found->second) {
        if (facet_idx >= facets.size())
            continue;

        const ColorFacetTriangle &facet = facets[facet_idx];
        Vec3f weights = Vec3f::Zero();
        if (!barycentric_weights_for_region_vertex_colors(point, facet.vertices[0], facet.vertices[1], facet.vertices[2], weights))
            continue;
        if (weights.x() >= tolerance && weights.y() >= tolerance && weights.z() >= tolerance)
            return unpack_vertex_color_rgba_for_conversion(facet.rgba);
    }

    if (found->second.empty() || found->second.front() >= facets.size())
        return std::nullopt;
    return unpack_vertex_color_rgba_for_conversion(facets[found->second.front()].rgba);
}

struct RGBStrokeVertexKey
{
    long long x = 0;
    long long y = 0;
    long long z = 0;
};

struct RGBStrokeEdgeKey
{
    RGBStrokeVertexKey a;
    RGBStrokeVertexKey b;
};

struct RGBStrokeBoundaryEdge
{
    int   source_triangle = -1;
    Vec3f a = Vec3f::Zero();
    Vec3f b = Vec3f::Zero();
};

struct RGBStrokeEdgeData
{
    int                   count = 0;
    RGBStrokeBoundaryEdge edge;
};

struct RGBStrokeEdgeKeyHash
{
    size_t operator()(const RGBStrokeEdgeKey &key) const
    {
        size_t hash = 1469598103934665603ull;
        auto mix = [&hash](long long value) {
            hash ^= std::hash<long long>{}(value) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
        };
        mix(key.a.x);
        mix(key.a.y);
        mix(key.a.z);
        mix(key.b.x);
        mix(key.b.y);
        mix(key.b.z);
        return hash;
    }
};

static bool operator==(const RGBStrokeVertexKey &lhs, const RGBStrokeVertexKey &rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

static bool operator==(const RGBStrokeEdgeKey &lhs, const RGBStrokeEdgeKey &rhs)
{
    return lhs.a == rhs.a && lhs.b == rhs.b;
}

static bool rgb_stroke_vertex_key_less(const RGBStrokeVertexKey &lhs, const RGBStrokeVertexKey &rhs)
{
    if (lhs.x != rhs.x)
        return lhs.x < rhs.x;
    if (lhs.y != rhs.y)
        return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

static RGBStrokeVertexKey rgb_stroke_vertex_key(const Vec3f &point)
{
    auto key = [](float value) {
        return std::isfinite(value) ? static_cast<long long>(std::llround(double(value) * 100000.0)) : 0ll;
    };
    return { key(point.x()), key(point.y()), key(point.z()) };
}

struct RGBStrokeBoundaryEdges
{
    std::unordered_map<int, std::vector<RGBStrokeBoundaryEdge>> by_source_triangle;
    std::vector<RGBStrokeBoundaryEdge> all;
};

static RGBStrokeBoundaryEdges build_rgb_stroke_boundary_edges(
    const std::vector<TriangleSelector::FacetStateTriangle> &stroke_facets)
{
    std::unordered_map<RGBStrokeEdgeKey, RGBStrokeEdgeData, RGBStrokeEdgeKeyHash> edges;
    edges.reserve(stroke_facets.size() * 3);

    auto add_edge = [&edges](const TriangleSelector::FacetStateTriangle &facet, const Vec3f &a, const Vec3f &b) {
        RGBStrokeEdgeKey key { rgb_stroke_vertex_key(a), rgb_stroke_vertex_key(b) };
        if (rgb_stroke_vertex_key_less(key.b, key.a))
            std::swap(key.a, key.b);

        RGBStrokeEdgeData &data = edges[key];
        ++data.count;
        if (data.count == 1)
            data.edge = { facet.source_triangle, a, b };
    };

    for (const TriangleSelector::FacetStateTriangle &facet : stroke_facets) {
        add_edge(facet, facet.vertices[0], facet.vertices[1]);
        add_edge(facet, facet.vertices[1], facet.vertices[2]);
        add_edge(facet, facet.vertices[2], facet.vertices[0]);
    }

    RGBStrokeBoundaryEdges boundary_edges;
    boundary_edges.all.reserve(edges.size());
    for (const auto &edge : edges) {
        if (edge.second.count != 1)
            continue;
        boundary_edges.by_source_triangle[edge.second.edge.source_triangle].emplace_back(edge.second.edge);
        boundary_edges.all.emplace_back(edge.second.edge);
    }
    return boundary_edges;
}

static float distance_to_segment(const Vec3f &point, const Vec3f &a, const Vec3f &b)
{
    const Vec3f ab = b - a;
    const float len2 = ab.squaredNorm();
    if (len2 <= EPSILON)
        return (point - a).norm();
    const float t = std::clamp((point - a).dot(ab) / len2, 0.f, 1.f);
    return (point - (a + ab * t)).norm();
}

static float distance_between_segments(const Vec3f &p1, const Vec3f &q1, const Vec3f &p2, const Vec3f &q2)
{
    const Vec3f d1 = q1 - p1;
    const Vec3f d2 = q2 - p2;
    const Vec3f r = p1 - p2;
    const float a = d1.dot(d1);
    const float e = d2.dot(d2);
    const float f = d2.dot(r);

    float s = 0.f;
    float t = 0.f;
    if (a <= EPSILON && e <= EPSILON)
        return (p1 - p2).norm();
    if (a <= EPSILON) {
        t = std::clamp(f / e, 0.f, 1.f);
    } else {
        const float c = d1.dot(r);
        if (e <= EPSILON) {
            s = std::clamp(-c / a, 0.f, 1.f);
        } else {
            const float b = d1.dot(d2);
            const float denom = a * e - b * b;
            if (denom > EPSILON)
                s = std::clamp((b * f - c * e) / denom, 0.f, 1.f);
            const float tnom = b * s + f;
            if (tnom < 0.f) {
                t = 0.f;
                s = std::clamp(-c / a, 0.f, 1.f);
            } else if (tnom > e) {
                t = 1.f;
                s = std::clamp((b - c) / a, 0.f, 1.f);
            } else {
                t = tnom / e;
            }
        }
    }
    return (p1 + d1 * s - (p2 + d2 * t)).norm();
}

static bool rgb_point_in_triangle(const Vec3f &point, const std::array<Vec3f, 3> &triangle)
{
    Vec3f weights = Vec3f::Zero();
    const float tolerance = -1e-4f;
    return barycentric_weights_for_region_vertex_colors(point, triangle[0], triangle[1], triangle[2], weights) &&
           weights.x() >= tolerance &&
           weights.y() >= tolerance &&
           weights.z() >= tolerance;
}

static bool rgb_triangles_overlap(const std::array<Vec3f, 3> &lhs, const std::array<Vec3f, 3> &rhs)
{
    for (const Vec3f &point : lhs)
        if (rgb_point_in_triangle(point, rhs))
            return true;
    for (const Vec3f &point : rhs)
        if (rgb_point_in_triangle(point, lhs))
            return true;

    const float edge_tolerance = 1e-4f;
    for (size_t lhs_idx = 0; lhs_idx < 3; ++lhs_idx)
        for (size_t rhs_idx = 0; rhs_idx < 3; ++rhs_idx)
            if (distance_between_segments(lhs[lhs_idx],
                                          lhs[(lhs_idx + 1) % 3],
                                          rhs[rhs_idx],
                                          rhs[(rhs_idx + 1) % 3]) <= edge_tolerance)
                return true;
    return false;
}

static Vec3f transform_point(const Transform3d &matrix, const Vec3f &point)
{
    return (matrix * point.cast<double>()).cast<float>();
}

static std::array<Vec3f, 3> transform_triangle(const Transform3d &matrix, const std::array<Vec3f, 3> &vertices)
{
    return {
        transform_point(matrix, vertices[0]),
        transform_point(matrix, vertices[1]),
        transform_point(matrix, vertices[2])
    };
}

static Vec3f closest_point_on_triangle(const Vec3f &point, const Vec3f &a, const Vec3f &b, const Vec3f &c)
{
    const Vec3f ab = b - a;
    const Vec3f ac = c - a;
    const Vec3f ap = point - a;
    const float d1 = ab.dot(ap);
    const float d2 = ac.dot(ap);
    if (d1 <= 0.f && d2 <= 0.f)
        return a;

    const Vec3f bp = point - b;
    const float d3 = ab.dot(bp);
    const float d4 = ac.dot(bp);
    if (d3 >= 0.f && d4 <= d3)
        return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
        return a + ab * (d1 / (d1 - d3));

    const Vec3f cp = point - c;
    const float d5 = ab.dot(cp);
    const float d6 = ac.dot(cp);
    if (d6 >= 0.f && d5 <= d6)
        return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
        return a + ac * (d2 / (d2 - d6));

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && d4 - d3 >= 0.f && d5 - d6 >= 0.f)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));

    const float denom_sum = va + vb + vc;
    if (std::abs(denom_sum) <= EPSILON)
        return a;
    const float denom = 1.f / denom_sum;
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

static float distance_to_triangle(const Vec3f &point, const std::array<Vec3f, 3> &vertices)
{
    return (point - closest_point_on_triangle(point, vertices[0], vertices[1], vertices[2])).norm();
}

static bool aabb_overlap(const Vec3f &min_a, const Vec3f &max_a, const Vec3f &min_b, const Vec3f &max_b)
{
    return min_a.x() <= max_b.x() && max_a.x() >= min_b.x() &&
           min_a.y() <= max_b.y() && max_a.y() >= min_b.y() &&
           min_a.z() <= max_b.z() && max_a.z() >= min_b.z();
}

static bool triangle_intersects_brush_segment(const std::array<Vec3f, 3> &vertices, const Vec3f &a, const Vec3f &b, float radius)
{
    if (distance_to_triangle(a, vertices) <= radius || distance_to_triangle(b, vertices) <= radius)
        return true;
    for (const Vec3f &vertex : vertices)
        if (distance_to_segment(vertex, a, b) <= radius)
            return true;
    for (size_t edge_idx = 0; edge_idx < 3; ++edge_idx)
        if (distance_between_segments(vertices[edge_idx], vertices[(edge_idx + 1) % 3], a, b) <= radius)
            return true;
    return false;
}

static float distance_to_brush_path(const std::vector<Vec3f> &stroke_points, const Vec3f &point)
{
    if (stroke_points.empty())
        return std::numeric_limits<float>::max();
    if (stroke_points.size() == 1)
        return (point - stroke_points.front()).norm();

    float distance = std::numeric_limits<float>::max();
    for (size_t idx = 1; idx < stroke_points.size(); ++idx)
        distance = std::min(distance, distance_to_segment(point, stroke_points[idx - 1], stroke_points[idx]));
    return distance;
}

static float sample_rgb_brush_path_alpha(const std::vector<Vec3f> &stroke_points,
                                         const Vec3f              &point,
                                         float                     hardness,
                                         float                     opacity,
                                         float                     brush_radius)
{
    opacity = std::clamp(opacity, 0.f, 1.f);
    if (opacity <= 0.f || brush_radius <= EPSILON)
        return 0.f;

    const float distance = distance_to_brush_path(stroke_points, point);
    if (!std::isfinite(distance) || distance > brush_radius)
        return 0.f;

    hardness = std::clamp(hardness, 0.f, 1.f);
    const float solid_radius = brush_radius * hardness;
    if (distance <= solid_radius)
        return opacity;

    const float fade_width = brush_radius - solid_radius;
    if (fade_width <= EPSILON)
        return opacity;

    const float t = std::clamp((brush_radius - distance) / fade_width, 0.f, 1.f);
    const float soft_alpha = t * t * (3.f - 2.f * t);
    return opacity * soft_alpha;
}

static float sample_rgb_stroke_alpha(const std::vector<TriangleSelector::FacetStateTriangle>      &stroke_facets,
                                     const std::unordered_map<int, std::vector<size_t>>           &stroke_by_source_triangle,
                                     const RGBStrokeBoundaryEdges                                 &stroke_boundary_edges,
                                     int                                                           source_triangle,
                                     const Vec3f                                                  &point,
                                     float                                                         hardness,
                                     float                                                         opacity,
                                     float                                                         brush_radius)
{
    if (opacity <= 0.f)
        return 0.f;

    auto found = stroke_by_source_triangle.find(source_triangle);
    if (found == stroke_by_source_triangle.end())
        return 0.f;

    const float tolerance = -1e-4f;
    bool inside_stroke = false;
    for (const size_t facet_idx : found->second) {
        if (facet_idx >= stroke_facets.size())
            continue;

        const TriangleSelector::FacetStateTriangle &facet = stroke_facets[facet_idx];
        Vec3f weights = Vec3f::Zero();
        if (!barycentric_weights_for_region_vertex_colors(point, facet.vertices[0], facet.vertices[1], facet.vertices[2], weights))
            continue;
        if (weights.x() < tolerance || weights.y() < tolerance || weights.z() < tolerance)
            continue;
        inside_stroke = true;
        break;
    }

    if (!inside_stroke)
        return 0.f;

    hardness = std::clamp(hardness, 0.f, 1.f);
    opacity = std::clamp(opacity, 0.f, 1.f);
    const float fade_width = brush_radius * (1.f - hardness);
    if (fade_width <= EPSILON)
        return opacity;

    if (stroke_boundary_edges.all.empty())
        return opacity;

    float boundary_distance = std::numeric_limits<float>::max();
    auto boundary_found = stroke_boundary_edges.by_source_triangle.find(source_triangle);
    if (boundary_found != stroke_boundary_edges.by_source_triangle.end())
        for (const RGBStrokeBoundaryEdge &edge : boundary_found->second)
            boundary_distance = std::min(boundary_distance, distance_to_segment(point, edge.a, edge.b));

    if (!std::isfinite(boundary_distance) || boundary_distance > fade_width) {
        for (const RGBStrokeBoundaryEdge &edge : stroke_boundary_edges.all)
            boundary_distance = std::min(boundary_distance, distance_to_segment(point, edge.a, edge.b));
    }

    if (!std::isfinite(boundary_distance))
        return opacity;

    const float t = std::clamp(boundary_distance / fade_width, 0.f, 1.f);
    const float soft_alpha = t * t * (3.f - 2.f * t);
    return opacity * soft_alpha;
}

static std::vector<bool> rgb_brush_candidate_source_triangles(
    const ModelVolume                                     &volume,
    const std::vector<Vec3f>                              &stroke_points,
    float                                                  brush_radius,
    const std::unordered_map<int, std::vector<size_t>>    &stroke_by_source_triangle,
    const Transform3d                                     &world_matrix)
{
    const indexed_triangle_set &its = volume.mesh().its;
    std::vector<bool> candidates(its.indices.size(), false);
    for (const auto &entry : stroke_by_source_triangle)
        if (entry.first >= 0 && size_t(entry.first) < candidates.size())
            candidates[size_t(entry.first)] = true;

    if (stroke_points.empty() || brush_radius <= EPSILON)
        return candidates;

    std::vector<Vec3f> stroke_points_world;
    stroke_points_world.reserve(stroke_points.size());
    for (const Vec3f &point : stroke_points)
        stroke_points_world.emplace_back(transform_point(world_matrix, point));

    Vec3f path_min = stroke_points_world.front();
    Vec3f path_max = stroke_points_world.front();
    for (const Vec3f &point : stroke_points_world) {
        path_min = path_min.cwiseMin(point);
        path_max = path_max.cwiseMax(point);
    }
    path_min -= Vec3f::Constant(brush_radius);
    path_max += Vec3f::Constant(brush_radius);

    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        if (candidates[tri_idx])
            continue;

        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
            continue;
        if (size_t(tri[0]) >= its.vertices.size() ||
            size_t(tri[1]) >= its.vertices.size() ||
            size_t(tri[2]) >= its.vertices.size())
            continue;

        const std::array<Vec3f, 3> local_vertices = {
            its.vertices[size_t(tri[0])].cast<float>(),
            its.vertices[size_t(tri[1])].cast<float>(),
            its.vertices[size_t(tri[2])].cast<float>()
        };
        const std::array<Vec3f, 3> vertices = transform_triangle(world_matrix, local_vertices);
        Vec3f tri_min = vertices[0].cwiseMin(vertices[1]).cwiseMin(vertices[2]);
        Vec3f tri_max = vertices[0].cwiseMax(vertices[1]).cwiseMax(vertices[2]);
        if (!aabb_overlap(tri_min, tri_max, path_min, path_max))
            continue;

        for (size_t point_idx = 0; point_idx < stroke_points_world.size(); ++point_idx) {
            const Vec3f segment_a = stroke_points_world[point_idx];
            const Vec3f segment_b = point_idx + 1 < stroke_points_world.size() ?
                stroke_points_world[point_idx + 1] :
                stroke_points_world[point_idx];
            Vec3f segment_min = segment_a.cwiseMin(segment_b) - Vec3f::Constant(brush_radius);
            Vec3f segment_max = segment_a.cwiseMax(segment_b) + Vec3f::Constant(brush_radius);
            if (aabb_overlap(tri_min, tri_max, segment_min, segment_max) &&
                triangle_intersects_brush_segment(vertices, segment_a, segment_b, brush_radius)) {
                candidates[tri_idx] = true;
                break;
            }
        }
    }

    return candidates;
}

static int rgb_color_tree_max_depth(const TriangleColorSplittingData &data, int bitstream_end, int &bit_idx, int depth)
{
    if (bit_idx + 3 >= bitstream_end)
        return std::clamp(depth, 0, 7);

    int code = 0;
    for (int bit = 0; bit < 4; ++bit)
        code |= int(data.bitstream[size_t(bit_idx++)]) << bit;

    const int split_sides = code & 0b11;
    if (split_sides == 0)
        return std::clamp(depth, 0, 7);

    int max_depth = std::clamp(depth + 1, 0, 7);
    for (int child_idx = split_sides; child_idx >= 0; --child_idx)
        max_depth = std::max(max_depth, rgb_color_tree_max_depth(data, bitstream_end, bit_idx, depth + 1));
    return std::clamp(max_depth, 0, 7);
}

static std::vector<int> rgb_existing_source_triangle_depths(const TriangleColorSplittingData &data, size_t triangle_count)
{
    std::vector<int> depths(triangle_count, 0);
    for (auto mapping_it = data.triangles_to_split.begin(); mapping_it != data.triangles_to_split.end(); ++mapping_it) {
        if (mapping_it->triangle_idx < 0 || size_t(mapping_it->triangle_idx) >= depths.size())
            continue;

        const auto next_it = std::next(mapping_it);
        const int bitstream_start = mapping_it->bitstream_start_idx;
        const int bitstream_end = next_it == data.triangles_to_split.end() ?
            int(data.bitstream.size()) :
            next_it->bitstream_start_idx;
        if (bitstream_start < 0 || bitstream_start >= bitstream_end || size_t(bitstream_end) > data.bitstream.size())
            continue;

        int bit_idx = bitstream_start;
        depths[size_t(mapping_it->triangle_idx)] = rgb_color_tree_max_depth(data, bitstream_end, bit_idx, 0);
    }
    return depths;
}

static bool apply_rgb_stroke_to_volume(ModelVolume                                           &volume,
                                       const std::vector<TriangleSelector::FacetStateTriangle> &stroke_facets,
                                       const ColorRGBA                                      &brush_color,
                                       float                                                hardness,
                                       float                                                opacity,
                                       float                                                brush_radius,
                                       const std::vector<Vec3f>                            &brush_stroke_points,
                                       const Transform3d                                   &world_matrix)
{
    if (stroke_facets.empty())
        return false;

    std::vector<ColorFacetTriangle> existing_facets;
    volume.texture_mapping_color_facets.get_facet_triangles(volume, existing_facets);
    std::unordered_map<int, std::vector<size_t>> existing_by_source_triangle;
    existing_by_source_triangle.reserve(existing_facets.size());
    for (size_t idx = 0; idx < existing_facets.size(); ++idx)
        existing_by_source_triangle[existing_facets[idx].source_triangle].emplace_back(idx);

    std::unordered_map<int, std::vector<size_t>> stroke_by_source_triangle;
    stroke_by_source_triangle.reserve(stroke_facets.size());
    for (size_t idx = 0; idx < stroke_facets.size(); ++idx)
        stroke_by_source_triangle[stroke_facets[idx].source_triangle].emplace_back(idx);
    RGBStrokeBoundaryEdges stroke_boundary_edges = build_rgb_stroke_boundary_edges(stroke_facets);
    const std::vector<bool> brush_candidate_triangles =
        rgb_brush_candidate_source_triangles(volume, brush_stroke_points, brush_radius, stroke_by_source_triangle, world_matrix);
    const bool use_brush_path = !brush_stroke_points.empty();
    std::vector<Vec3f> brush_stroke_points_world;
    if (use_brush_path) {
        brush_stroke_points_world.reserve(brush_stroke_points.size());
        for (const Vec3f &point : brush_stroke_points)
            brush_stroke_points_world.emplace_back(transform_point(world_matrix, point));
    }

    const ColorRGBA background = rgb_metadata_background_color(volume.texture_mapping_color_facets);
    const uint32_t brush_packed = pack_vertex_color_rgba(brush_color);
    TextureMappingColorSampler sampler = [&existing_facets,
                                          &existing_by_source_triangle,
                                          &stroke_facets,
                                          &stroke_by_source_triangle,
                                          &stroke_boundary_edges,
                                          background,
                                          brush_color,
                                          brush_packed,
                                          hardness,
                                          opacity,
                                          brush_radius,
                                          use_brush_path,
                                          &brush_candidate_triangles,
                                          &brush_stroke_points_world,
                                          &world_matrix](size_t tri_idx, const Vec3f &point, const Vec3f &) {
        ColorRGBA source_color = background;
        if (std::optional<ColorRGBA> sampled =
                sample_rgb_color_facets(existing_facets, existing_by_source_triangle, int(tri_idx), point)) {
            source_color = *sampled;
        }

        if (use_brush_path && (tri_idx >= brush_candidate_triangles.size() || !brush_candidate_triangles[tri_idx]))
            return pack_vertex_color_rgba(source_color);

        const float alpha = use_brush_path ?
            sample_rgb_brush_path_alpha(brush_stroke_points_world,
                                        transform_point(world_matrix, point),
                                        hardness,
                                        opacity,
                                        brush_radius) :
            sample_rgb_stroke_alpha(stroke_facets,
                                    stroke_by_source_triangle,
                                    stroke_boundary_edges,
                                    int(tri_idx),
                                    point,
                                    hardness,
                                    opacity,
                                    brush_radius);
        if (alpha <= 0.f)
            return pack_vertex_color_rgba(source_color);
        if (alpha >= 1.f)
            return brush_packed;

        return pack_vertex_color_rgba(ColorRGBA(source_color.r() * (1.f - alpha) + brush_color.r() * alpha,
                                                source_color.g() * (1.f - alpha) + brush_color.g() * alpha,
                                                source_color.b() * (1.f - alpha) + brush_color.b() * alpha,
                                                source_color.a() * (1.f - alpha) + brush_color.a() * alpha));
    };

    const float mesh_span = mesh_max_axis_span(volume.mesh().its);
    const int safe_max_depth = texture_mapping_depth_for_budget(volume.mesh().its.indices.size(), 7, 1800000);
    const float brush_subdivision_target = true_color_brush_subdivision_target(brush_radius);
    const std::vector<int> existing_source_triangle_depths =
        rgb_existing_source_triangle_depths(volume.texture_mapping_color_facets.get_data(), volume.mesh().its.indices.size());
    TextureMappingColorSubdivisionDepths subdivision_depths =
        [mesh_span,
         safe_max_depth,
         brush_subdivision_target,
         &brush_candidate_triangles,
         &existing_source_triangle_depths,
         &world_matrix](size_t tri_idx, const std::array<Vec3f, 3> &vertices) {
        const int base_depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices),
                                                               std::max(mesh_span / 220.f, 0.18f),
                                                               std::min(6, safe_max_depth));
        const int preserved_depth = tri_idx < existing_source_triangle_depths.size() ?
            existing_source_triangle_depths[tri_idx] :
            0;
        int min_depth = std::max(base_depth, preserved_depth);
        int max_depth = std::max(safe_max_depth, preserved_depth);
        if (tri_idx < brush_candidate_triangles.size() && brush_candidate_triangles[tri_idx]) {
            const std::array<Vec3f, 3> world_vertices = transform_triangle(world_matrix, vertices);
            const int brush_depth = texture_mapping_depth_from_span(triangle_max_edge_length(world_vertices),
                                                                    brush_subdivision_target,
                                                                    7);
            min_depth = std::max(min_depth, brush_depth);
            max_depth = std::max(max_depth, min_depth);
        }
        return std::make_pair(min_depth, max_depth);
    };

    TextureMappingColorLeafResamplePredicate resample_leaf =
        [opacity,
         brush_radius,
         use_brush_path,
         &brush_stroke_points_world,
         &world_matrix,
         &stroke_by_source_triangle,
         &stroke_facets](size_t tri_idx, const std::array<Vec3f, 3> &vertices, const std::array<Vec3f, 3> &, uint32_t) {
        if (opacity <= 0.f || brush_radius <= EPSILON)
            return false;

        if (use_brush_path) {
            const std::array<Vec3f, 3> world_vertices = transform_triangle(world_matrix, vertices);
            for (size_t point_idx = 0; point_idx < brush_stroke_points_world.size(); ++point_idx) {
                const Vec3f segment_a = brush_stroke_points_world[point_idx];
                const Vec3f segment_b = point_idx + 1 < brush_stroke_points_world.size() ?
                    brush_stroke_points_world[point_idx + 1] :
                    brush_stroke_points_world[point_idx];
                if (triangle_intersects_brush_segment(world_vertices, segment_a, segment_b, brush_radius))
                    return true;
            }
            return false;
        }

        auto found = stroke_by_source_triangle.find(int(tri_idx));
        if (found == stroke_by_source_triangle.end())
            return false;
        for (const size_t facet_idx : found->second) {
            if (facet_idx >= stroke_facets.size())
                continue;
            if (rgb_triangles_overlap(vertices, stroke_facets[facet_idx].vertices))
                return true;
        }
        return false;
    };

    return volume.texture_mapping_color_facets.set_from_triangle_sampler(volume,
                                                                         sampler,
                                                                         safe_max_depth,
                                                                         0.012f,
                                                                         subdivision_depths,
                                                                         &brush_candidate_triangles,
                                                                         resample_leaf);
}

static bool build_volume_rgb_data(const indexed_triangle_set &its,
                                  const ColorRGBA           &background,
                                  ColorFacetsAnnotation     &out,
                                  const std::function<void()> &check_cancel = {})
{
    if (its.indices.empty() || its.vertices.empty())
        return false;

    out.reset();
    const uint32_t packed = pack_vertex_color_rgba(background);
    TextureMappingColorSampler sampler = [packed, &check_cancel](size_t, const Vec3f &, const Vec3f &) {
        if (check_cancel)
            check_cancel();
        return packed;
    };
    TextureMappingColorProgressFn progress_fn;
    if (check_cancel)
        progress_fn = [&check_cancel](size_t, size_t) { check_cancel(); };
    const bool changed = out.set_from_triangle_sampler(its, sampler, 0, 0.f, {}, nullptr, {}, progress_fn);
    out.set_metadata_json(rgb_metadata_json(background));
    return changed || !out.empty();
}

static bool build_volume_rgb_data(const ModelVolume &volume, const ColorRGBA &background, ColorFacetsAnnotation &out)
{
    return build_volume_rgb_data(volume.mesh().its, background, out);
}

static bool build_rgba_data_from_color_sampler(const indexed_triangle_set              &its,
                                               const TextureMappingColorSampler        &sampler,
                                               bool                                    has_image_texture_source,
                                               bool                                    has_geometry_color_source,
                                               const std::function<float(size_t)>      &texture_span_fn,
                                               const ColorRGBA                         &fallback_color,
                                               ColorFacetsAnnotation                   &out,
                                               const std::function<void()>             &check_cancel = {})
{
    if (its.indices.empty() || its.vertices.empty())
        return false;

    TextureMappingColorProgressFn progress_fn;
    if (check_cancel)
        progress_fn = [&check_cancel](size_t, size_t) { check_cancel(); };

    bool sampled = false;
    if (has_image_texture_source) {
        const int safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 3200000);
        TextureMappingColorSubdivisionDepths subdivision_depths =
            [safe_max_depth, &texture_span_fn, &check_cancel](size_t tri_idx, const std::array<Vec3f, 3> &) {
                if (check_cancel)
                    check_cancel();
                const float span = texture_span_fn ? texture_span_fn(tri_idx) : 0.f;
                const int depth = texture_mapping_depth_from_span(span, 8.f, safe_max_depth);
                return std::make_pair(depth, depth);
            };
        sampled = out.set_from_triangle_sampler(its, sampler, safe_max_depth, 0.015f, subdivision_depths, nullptr, {}, progress_fn);
    } else if (has_geometry_color_source) {
        const float target_edge = std::max(mesh_max_axis_span(its) / 160.f, 0.25f);
        TextureMappingColorSubdivisionDepths subdivision_depths =
            [target_edge, &check_cancel](size_t, const std::array<Vec3f, 3> &vertices) {
                if (check_cancel)
                    check_cancel();
                const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_edge, 5);
                return std::make_pair(depth, depth);
            };
        sampled = out.set_from_triangle_sampler(its, sampler, 5, 0.025f, subdivision_depths, nullptr, {}, progress_fn);
    } else {
        sampled = build_volume_rgb_data(its, fallback_color, out, check_cancel);
    }

    if (sampled && out.metadata_json().empty())
        out.set_metadata_json(rgb_metadata_json(fallback_color));
    return sampled || !out.empty();
}

static bool initialize_volume_rgb_data(ModelVolume &volume, const ColorRGBA &background)
{
    std::unique_ptr<ColorFacetsAnnotation> rgb_data = ColorFacetsAnnotation::make_temporary();
    if (!rgb_data || !build_volume_rgb_data(volume, background, *rgb_data))
        return false;
    if (volume.texture_mapping_color_facets.equals(*rgb_data))
        return false;
    volume.texture_mapping_color_facets.assign(*rgb_data);
    return true;
}

struct ProjectionContext
{
    Matrix4d                     view_projection = Matrix4d::Identity();
    Vec3d                        camera_forward = Vec3d::Zero();
    Vec3d                        camera_position = Vec3d::Zero();
    int                          canvas_width = 1;
    int                          canvas_height = 1;
    float                        overlay_left = 0.f;
    float                        overlay_top = 0.f;
    float                        overlay_width = 0.f;
    float                        overlay_height = 0.f;
    const std::vector<uint8_t>  *image_rgba = nullptr;
    uint32_t                     image_width = 0;
    uint32_t                     image_height = 0;
    float                        image_opacity = 1.f;
    bool                         apply_transparency_as_background = false;
    ClippingPlane                section_clipping_plane;
};

static bool projection_section_view_active(const ProjectionContext &context)
{
    return context.section_clipping_plane.is_active();
}

static bool projection_world_point_visible_in_section(const ProjectionContext &context, const Vec3d &world_point)
{
    return !projection_section_view_active(context) || context.section_clipping_plane.distance(world_point) >= -1e-7;
}

static bool projection_sample_allowed_by_camera_facing(const ProjectionContext &context,
                                                       const Vec3d             &world_point,
                                                       const Vec3d             &world_normal)
{
    static constexpr double back_face_rejection_dot = 0.25;

    if (!projection_section_view_active(context))
        return true;
    if (world_normal.squaredNorm() <= EPSILON)
        return true;

    Vec3d view_direction = world_point - context.camera_position;
    if (view_direction.squaredNorm() <= EPSILON)
        view_direction = context.camera_forward;
    if (view_direction.squaredNorm() <= EPSILON)
        return true;

    return world_normal.normalized().dot(view_direction.normalized()) <= back_face_rejection_dot;
}

static std::vector<Vec3d> projection_visible_world_polygon(const ProjectionContext     &context,
                                                           const Transform3d          &world_matrix,
                                                           const std::array<Vec3f, 3> &vertices)
{
    std::vector<Vec3d> polygon;
    polygon.reserve(4);
    for (const Vec3f &vertex : vertices)
        polygon.emplace_back(world_matrix * vertex.cast<double>());

    if (!projection_section_view_active(context))
        return polygon;

    std::vector<Vec3d> clipped;
    clipped.reserve(4);

    Vec3d previous = polygon.back();
    double previous_distance = context.section_clipping_plane.distance(previous);
    bool previous_inside = previous_distance >= -1e-7;
    for (const Vec3d &current : polygon) {
        const double current_distance = context.section_clipping_plane.distance(current);
        const bool current_inside = current_distance >= -1e-7;
        if (current_inside != previous_inside) {
            const double denom = previous_distance - current_distance;
            if (std::abs(denom) > 1e-12) {
                const double t = std::clamp(previous_distance / denom, 0.0, 1.0);
                clipped.emplace_back(previous + (current - previous) * t);
            }
        }
        if (current_inside)
            clipped.emplace_back(current);
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }

    return clipped;
}

static void apply_projection_section_view(ProjectionContext &context, const CommonGizmosDataObjects::ObjectClipper *clipper)
{
    if (clipper == nullptr || clipper->get_position() == 0.)
        return;

    const ClippingPlane *plane = clipper->get_clipping_plane();
    if (plane != nullptr && plane->is_active())
        context.section_clipping_plane = *plane;
}

struct VolumeColorSource
{
    std::vector<ColorFacetTriangle>              rgb_facets;
    std::unordered_map<int, std::vector<size_t>> rgb_by_source_triangle;
    std::optional<ColorRGBA>                     rgb_background_color;
};

static ColorRGBA sample_managed_volume_color_source(const ModelVolume              &volume,
                                                    const VolumeColorSource        &rgba_source,
                                                    const ManagedRegionColorSource &region_source,
                                                    size_t                          tri_idx,
                                                    const Vec3f                    &point,
                                                    const Vec3f                    &barycentric,
                                                    bool                            use_rgba,
                                                    bool                            use_image_texture,
                                                    bool                            use_vertex_colors,
                                                    bool                            use_color_regions,
                                                    const ColorRGBA                &fallback_color);

static ColorRGBA sample_rgba_bilinear_clamped(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height, float u, float v)
{
    if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);

    u = std::clamp(u, 0.f, 1.f);
    v = std::clamp(v, 0.f, 1.f);
    const float x = u * float(width > 1 ? width - 1 : 0);
    const float y = v * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto channel = [&rgba, width](size_t sx, size_t sy, size_t ch) {
        return float(rgba[(sy * size_t(width) + sx) * 4 + ch]) / 255.f;
    };
    auto blend_channel = [&](size_t ch) {
        const float c00 = channel(x0, y0, ch);
        const float c10 = channel(x1, y0, ch);
        const float c01 = channel(x0, y1, ch);
        const float c11 = channel(x1, y1, ch);
        return std::clamp((c00 + (c10 - c00) * tx) + ((c01 + (c11 - c01) * tx) - (c00 + (c10 - c00) * tx)) * ty, 0.f, 1.f);
    };
    auto blend_premultiplied_channel = [&](size_t ch) {
        const float c00 = channel(x0, y0, ch) * channel(x0, y0, 3);
        const float c10 = channel(x1, y0, ch) * channel(x1, y0, 3);
        const float c01 = channel(x0, y1, ch) * channel(x0, y1, 3);
        const float c11 = channel(x1, y1, ch) * channel(x1, y1, 3);
        return std::clamp((c00 + (c10 - c00) * tx) + ((c01 + (c11 - c01) * tx) - (c00 + (c10 - c00) * tx)) * ty, 0.f, 1.f);
    };

    const float a = blend_channel(3);
    if (a <= 0.f)
        return ColorRGBA(blend_channel(0), blend_channel(1), blend_channel(2), 0.f);
    return ColorRGBA(std::clamp(blend_premultiplied_channel(0) / a, 0.f, 1.f),
                     std::clamp(blend_premultiplied_channel(1) / a, 0.f, 1.f),
                     std::clamp(blend_premultiplied_channel(2) / a, 0.f, 1.f),
                     a);
}

static ColorRGBA sample_rgba_bilinear_wrapped(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height, float u, float v)
{
    return sample_rgba_bilinear_clamped(rgba, width, height, wrap_texture_uv_for_vertex_bake(u), wrap_texture_uv_for_vertex_bake(v));
}

static std::vector<uint8_t> sample_raw_offsets_bilinear_clamped(const ImageMapRawFilamentOffsetAtlas &atlas, float u, float v)
{
    std::vector<uint8_t> values;
    if (!atlas.valid())
        return values;

    values.assign(atlas.channels, 0);
    u = std::clamp(u, 0.f, 1.f);
    v = std::clamp(v, 0.f, 1.f);
    const float x = u * float(atlas.width > 1 ? atlas.width - 1 : 0);
    const float y = v * float(atlas.height > 1 ? atlas.height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(atlas.width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(atlas.height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(atlas.width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(atlas.height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto channel = [&atlas](size_t sx, size_t sy, size_t ch) {
        return float(atlas.offsets[(sy * size_t(atlas.width) + sx) * size_t(atlas.channels) + ch]);
    };
    for (size_t ch = 0; ch < values.size(); ++ch) {
        const float c00 = channel(x0, y0, ch);
        const float c10 = channel(x1, y0, ch);
        const float c01 = channel(x0, y1, ch);
        const float c11 = channel(x1, y1, ch);
        const float value = (c00 + (c10 - c00) * tx) + ((c01 + (c11 - c01) * tx) - (c00 + (c10 - c00) * tx)) * ty;
        values[ch] = uint8_t(std::clamp(int(std::lround(value)), 0, 255));
    }
    return values;
}

static std::vector<uint8_t> sample_raw_offsets_bilinear_wrapped(const std::vector<uint8_t> &offsets,
                                                                uint32_t                    width,
                                                                uint32_t                    height,
                                                                uint32_t                    channels,
                                                                float                       u,
                                                                float                       v)
{
    std::vector<uint8_t> values(size_t(channels), 0);
    if (width == 0 || height == 0 || channels == 0 ||
        offsets.size() < size_t(width) * size_t(height) * size_t(channels))
        return values;

    u = wrap_texture_uv_for_vertex_bake(u);
    v = wrap_texture_uv_for_vertex_bake(v);
    const float x = u * float(width > 1 ? width - 1 : 0);
    const float y = v * float(height > 1 ? height - 1 : 0);
    const size_t x0 = std::min<size_t>(size_t(std::floor(x)), size_t(width - 1));
    const size_t y0 = std::min<size_t>(size_t(std::floor(y)), size_t(height - 1));
    const size_t x1 = std::min<size_t>(x0 + 1, size_t(width - 1));
    const size_t y1 = std::min<size_t>(y0 + 1, size_t(height - 1));
    const float tx = x - float(x0);
    const float ty = y - float(y0);

    auto channel = [&offsets, width, channels](size_t sx, size_t sy, size_t ch) {
        return float(offsets[(sy * size_t(width) + sx) * size_t(channels) + ch]);
    };
    for (size_t ch = 0; ch < values.size(); ++ch) {
        const float c00 = channel(x0, y0, ch);
        const float c10 = channel(x1, y0, ch);
        const float c01 = channel(x0, y1, ch);
        const float c11 = channel(x1, y1, ch);
        const float value = (c00 + (c10 - c00) * tx) + ((c01 + (c11 - c01) * tx) - (c00 + (c10 - c00) * tx)) * ty;
        values[ch] = uint8_t(std::clamp(int(std::lround(value)), 0, 255));
    }
    return values;
}

static bool checked_texture_buffer_size(uint32_t width, uint32_t height, uint32_t channels, size_t& size)
{
    size = 0;
    if (width == 0 || height == 0 || channels == 0)
        return false;
    if (size_t(width) > std::numeric_limits<size_t>::max() / size_t(height))
        return false;
    const size_t pixels = size_t(width) * size_t(height);
    if (pixels > std::numeric_limits<size_t>::max() / size_t(channels))
        return false;
    size = pixels * size_t(channels);
    return true;
}

static std::vector<uint8_t> resize_rgba_texture_bilinear(
    const std::vector<uint8_t>& rgba, uint32_t width, uint32_t height, uint32_t resized_width, uint32_t resized_height)
{
    size_t resized_size = 0;
    if (!checked_texture_buffer_size(resized_width, resized_height, 4, resized_size))
        return {};

    std::vector<uint8_t> resized(resized_size, 255);
    size_t               source_size = 0;
    if (!checked_texture_buffer_size(width, height, 4, source_size) || rgba.size() < source_size)
        return resized;

    auto channel = [&rgba, width](size_t sx, size_t sy, size_t ch) { return float(rgba[(sy * size_t(width) + sx) * 4 + ch]) / 255.f; };
    for (uint32_t y = 0; y < resized_height; ++y) {
        const float  source_y = std::clamp((float(y) + 0.5f) * float(height) / float(resized_height) - 0.5f, 0.f, float(height - 1));
        const size_t y0       = std::min<size_t>(size_t(std::floor(source_y)), size_t(height - 1));
        const size_t y1       = std::min<size_t>(y0 + 1, size_t(height - 1));
        const float  ty       = source_y - float(y0);
        for (uint32_t x = 0; x < resized_width; ++x) {
            const float  source_x      = std::clamp((float(x) + 0.5f) * float(width) / float(resized_width) - 0.5f, 0.f, float(width - 1));
            const size_t x0            = std::min<size_t>(size_t(std::floor(source_x)), size_t(width - 1));
            const size_t x1            = std::min<size_t>(x0 + 1, size_t(width - 1));
            const float  tx            = source_x - float(x0);
            auto         blend_channel = [&](size_t ch) {
                const float c00 = channel(x0, y0, ch);
                const float c10 = channel(x1, y0, ch);
                const float c01 = channel(x0, y1, ch);
                const float c11 = channel(x1, y1, ch);
                const float cx0 = c00 + (c10 - c00) * tx;
                const float cx1 = c01 + (c11 - c01) * tx;
                return std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 1.f);
            };
            auto blend_premultiplied_channel = [&](size_t ch) {
                const float c00 = channel(x0, y0, ch) * channel(x0, y0, 3);
                const float c10 = channel(x1, y0, ch) * channel(x1, y0, 3);
                const float c01 = channel(x0, y1, ch) * channel(x0, y1, 3);
                const float c11 = channel(x1, y1, ch) * channel(x1, y1, 3);
                const float cx0 = c00 + (c10 - c00) * tx;
                const float cx1 = c01 + (c11 - c01) * tx;
                return std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 1.f);
            };
            const float  a   = blend_channel(3);
            const size_t idx = (size_t(y) * size_t(resized_width) + size_t(x)) * 4;
            resized[idx + 0] =
                uint8_t(std::clamp(a > 0.f ? blend_premultiplied_channel(0) / a : blend_channel(0), 0.f, 1.f) * 255.f + 0.5f);
            resized[idx + 1] =
                uint8_t(std::clamp(a > 0.f ? blend_premultiplied_channel(1) / a : blend_channel(1), 0.f, 1.f) * 255.f + 0.5f);
            resized[idx + 2] =
                uint8_t(std::clamp(a > 0.f ? blend_premultiplied_channel(2) / a : blend_channel(2), 0.f, 1.f) * 255.f + 0.5f);
            resized[idx + 3] = uint8_t(a * 255.f + 0.5f);
        }
    }
    return resized;
}

static std::vector<uint8_t> resize_raw_offsets_bilinear(const std::vector<uint8_t>& offsets,
                                                        uint32_t                    width,
                                                        uint32_t                    height,
                                                        uint32_t                    channels,
                                                        uint32_t                    resized_width,
                                                        uint32_t                    resized_height)
{
    size_t resized_size = 0;
    if (!checked_texture_buffer_size(resized_width, resized_height, channels, resized_size))
        return {};

    std::vector<uint8_t> resized(resized_size, 0);
    size_t               source_size = 0;
    if (!checked_texture_buffer_size(width, height, channels, source_size) || offsets.size() < source_size)
        return resized;

    auto channel = [&offsets, width, channels](size_t sx, size_t sy, size_t ch) {
        return float(offsets[(sy * size_t(width) + sx) * size_t(channels) + ch]);
    };
    for (uint32_t y = 0; y < resized_height; ++y) {
        const float  source_y = std::clamp((float(y) + 0.5f) * float(height) / float(resized_height) - 0.5f, 0.f, float(height - 1));
        const size_t y0       = std::min<size_t>(size_t(std::floor(source_y)), size_t(height - 1));
        const size_t y1       = std::min<size_t>(y0 + 1, size_t(height - 1));
        const float  ty       = source_y - float(y0);
        for (uint32_t x = 0; x < resized_width; ++x) {
            const float  source_x = std::clamp((float(x) + 0.5f) * float(width) / float(resized_width) - 0.5f, 0.f, float(width - 1));
            const size_t x0       = std::min<size_t>(size_t(std::floor(source_x)), size_t(width - 1));
            const size_t x1       = std::min<size_t>(x0 + 1, size_t(width - 1));
            const float  tx       = source_x - float(x0);
            const size_t idx      = (size_t(y) * size_t(resized_width) + size_t(x)) * size_t(channels);
            for (size_t ch = 0; ch < size_t(channels); ++ch) {
                const float c00   = channel(x0, y0, ch);
                const float c10   = channel(x1, y0, ch);
                const float c01   = channel(x0, y1, ch);
                const float c11   = channel(x1, y1, ch);
                const float cx0   = c00 + (c10 - c00) * tx;
                const float cx1   = c01 + (c11 - c01) * tx;
                resized[idx + ch] = uint8_t(std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 255.f) + 0.5f);
            }
        }
    }
    return resized;
}

static std::string resized_raw_texture_metadata_json(const std::string& metadata_json, uint32_t width, uint32_t height, uint32_t channels)
{
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(metadata_json);
        if (!root.is_object())
            root = nlohmann::json::object();
    } catch (...) {
        root = nlohmann::json::object();
    }
    root["format"] = "raw_filament_offset_atlas";
    root["image"]  = {{"width", width}, {"height", height}, {"channels", channels}};
    root.erase("regions");
    return root.dump();
}

static bool resize_volume_image_texture(ModelVolume& volume, uint32_t resized_width, uint32_t resized_height)
{
    if (!model_volume_has_imported_image_texture_data(&volume) || resized_width == 0 || resized_height == 0)
        return false;

    const uint32_t old_width  = volume.imported_texture_width;
    const uint32_t old_height = volume.imported_texture_height;
    if (old_width == resized_width && old_height == resized_height)
        return false;

    const bool has_raw_atlas = model_volume_has_raw_atlas_texture_data(&volume);
    if (has_raw_atlas) {
        std::vector<uint8_t> resized_offsets = resize_raw_offsets_bilinear(volume.imported_texture_raw_filament_offsets, old_width,
                                                                           old_height, volume.imported_texture_raw_channels, resized_width,
                                                                           resized_height);
        if (resized_offsets.empty())
            return false;
        volume.imported_texture_raw_filament_offsets = std::move(resized_offsets);
        volume.imported_texture_width                = resized_width;
        volume.imported_texture_height               = resized_height;
        volume.imported_texture_raw_metadata_json    = resized_raw_texture_metadata_json(volume.imported_texture_raw_metadata_json,
                                                                                         resized_width, resized_height,
                                                                                         volume.imported_texture_raw_channels);
        refresh_imported_texture_preview_from_raw_offsets(volume);
        volume.imported_texture_raw_filament_offsets.set_new_unique_id();
        volume.imported_texture_rgba.set_new_unique_id();
        return true;
    }

    size_t source_rgba_size = 0;
    if (!checked_texture_buffer_size(old_width, old_height, 4, source_rgba_size) || volume.imported_texture_rgba.size() < source_rgba_size)
        return false;

    std::vector<uint8_t> resized_rgba = resize_rgba_texture_bilinear(volume.imported_texture_rgba, old_width, old_height, resized_width,
                                                                     resized_height);
    if (resized_rgba.empty())
        return false;

    volume.imported_texture_rgba = std::move(resized_rgba);
    const bool cleared_raw_data  = !volume.imported_texture_raw_filament_offsets.empty() || volume.imported_texture_raw_channels != 0 ||
                                  !volume.imported_texture_raw_metadata_json.empty();
    if (cleared_raw_data)
        clear_imported_texture_raw_atlas(volume);
    volume.imported_texture_width  = resized_width;
    volume.imported_texture_height = resized_height;
    volume.imported_texture_rgba.set_new_unique_id();
    if (cleared_raw_data)
        volume.imported_texture_raw_filament_offsets.set_new_unique_id();
    return true;
}

static bool resize_object_image_textures(ModelObject& object, double scale)
{
    if (!std::isfinite(scale) || scale <= 0.0)
        return false;

    bool changed = false;
    for (ModelVolume* volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_imported_image_texture_data(volume))
            continue;
        const uint32_t resized_width  = std::max<uint32_t>(1, uint32_t(std::llround(double(volume->imported_texture_width) * scale)));
        const uint32_t resized_height = std::max<uint32_t>(1, uint32_t(std::llround(double(volume->imported_texture_height) * scale)));
        changed |= resize_volume_image_texture(*volume, resized_width, resized_height);
    }
    return changed;
}

static ColorRGBA blend_projection_color(const ColorRGBA &base, const ColorRGBA &overlay, float opacity)
{
    const float alpha = std::clamp(overlay.a(), 0.f, 1.f) * std::clamp(opacity, 0.f, 1.f);
    if (alpha <= 0.f)
        return base;
    const float out_alpha = std::clamp(alpha + base.a() * (1.f - alpha), 0.f, 1.f);
    if (out_alpha <= EPSILON)
        return ColorRGBA(overlay.r(), overlay.g(), overlay.b(), 0.f);
    return ColorRGBA(base.r() * (1.f - alpha) + overlay.r() * alpha,
                     base.g() * (1.f - alpha) + overlay.g() * alpha,
                     base.b() * (1.f - alpha) + overlay.b() * alpha,
                     out_alpha);
}

static float projection_overlay_alpha(const ColorRGBA &overlay, const ProjectionContext &context)
{
    return std::clamp(overlay.a(), 0.f, 1.f) * std::clamp(context.image_opacity, 0.f, 1.f);
}

static bool projection_overlay_has_paintable_alpha(const ColorRGBA &overlay, const ProjectionContext &context)
{
    return projection_overlay_alpha(overlay, context) > 0.5f / 255.f;
}

static ColorRGBA apply_projection_color(const ColorRGBA &base, const ColorRGBA &overlay, const ProjectionContext &context, bool image_texture_target)
{
    const float opacity = std::clamp(context.image_opacity, 0.f, 1.f);
    if (!context.apply_transparency_as_background)
        return blend_projection_color(base, overlay, opacity);

    const float alpha = std::clamp(overlay.a(), 0.f, 1.f) * opacity;
    if (image_texture_target)
        return ColorRGBA(overlay.r() * alpha, overlay.g() * alpha, overlay.b() * alpha, 1.f);
    return ColorRGBA(overlay.r(), overlay.g(), overlay.b(), alpha);
}

static bool wx_image_to_rgba(const wxImage &image, std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height)
{
    if (!image.IsOk() || image.GetWidth() <= 0 || image.GetHeight() <= 0)
        return false;

    width = uint32_t(image.GetWidth());
    height = uint32_t(image.GetHeight());
    rgba.assign(size_t(width) * size_t(height) * 4, 255);

    const unsigned char *rgb = image.GetData();
    const unsigned char *alpha = image.HasAlpha() ? image.GetAlpha() : nullptr;
    const bool has_mask = image.HasMask();
    const int mask_r = has_mask ? image.GetMaskRed() : -1;
    const int mask_g = has_mask ? image.GetMaskGreen() : -1;
    const int mask_b = has_mask ? image.GetMaskBlue() : -1;
    if (rgb == nullptr)
        return false;

    for (size_t idx = 0; idx < size_t(width) * size_t(height); ++idx) {
        rgba[idx * 4 + 0] = rgb[idx * 3 + 0];
        rgba[idx * 4 + 1] = rgb[idx * 3 + 1];
        rgba[idx * 4 + 2] = rgb[idx * 3 + 2];
        rgba[idx * 4 + 3] =
            has_mask &&
            int(rgb[idx * 3 + 0]) == mask_r &&
            int(rgb[idx * 3 + 1]) == mask_g &&
            int(rgb[idx * 3 + 2]) == mask_b ?
                0 :
                (alpha == nullptr ? 255 : alpha[idx]);
    }
    return true;
}

static bool project_point_to_screen(const ProjectionContext &context, const Vec3d &world_point, Vec2f &screen, float *ndc_z = nullptr)
{
    const Vec4d clip = context.view_projection * Vec4d(world_point.x(), world_point.y(), world_point.z(), 1.0);
    if (clip.w() <= 0.0)
        return false;

    const Vec3d ndc = clip.head<3>() / clip.w();
    if (ndc.x() < -1.0 || ndc.x() > 1.0 || ndc.y() < -1.0 || ndc.y() > 1.0 || ndc.z() < -1.0 || ndc.z() > 1.0)
        return false;

    screen.x() = float((ndc.x() * 0.5 + 0.5) * double(context.canvas_width));
    screen.y() = float((1.0 - (ndc.y() * 0.5 + 0.5)) * double(context.canvas_height));
    if (ndc_z != nullptr)
        *ndc_z = float(ndc.z());
    return true;
}

static bool project_point_to_depth_clipped_screen(const ProjectionContext &context,
                                                  const Vec3d             &world_point,
                                                  Vec2f                   &screen,
                                                  float                   *ndc_z = nullptr)
{
    const Vec4d clip = context.view_projection * Vec4d(world_point.x(), world_point.y(), world_point.z(), 1.0);
    if (clip.w() <= 0.0)
        return false;

    const Vec3d ndc = clip.head<3>() / clip.w();
    if (ndc.z() < -1.0 || ndc.z() > 1.0)
        return false;

    screen.x() = float((ndc.x() * 0.5 + 0.5) * double(context.canvas_width));
    screen.y() = float((1.0 - (ndc.y() * 0.5 + 0.5)) * double(context.canvas_height));
    if (ndc_z != nullptr)
        *ndc_z = float(ndc.z());
    return std::isfinite(screen.x()) && std::isfinite(screen.y());
}

static std::optional<ColorRGBA> projected_image_color_at_point(const ProjectionContext &context,
                                                               const Transform3d      &world_matrix,
                                                               const Vec3f            &point)
{
    if (context.image_rgba == nullptr || context.overlay_width <= 0.f || context.overlay_height <= 0.f)
        return std::nullopt;

    const Vec3d world_point = world_matrix * point.cast<double>();
    if (!projection_world_point_visible_in_section(context, world_point))
        return std::nullopt;

    Vec2f screen = Vec2f::Zero();
    if (!project_point_to_screen(context, world_point, screen))
        return std::nullopt;
    if (screen.x() < context.overlay_left ||
        screen.y() < context.overlay_top ||
        screen.x() > context.overlay_left + context.overlay_width ||
        screen.y() > context.overlay_top + context.overlay_height)
        return std::nullopt;

    const float u = (screen.x() - context.overlay_left) / context.overlay_width;
    const float v = (screen.y() - context.overlay_top) / context.overlay_height;
    return sample_rgba_bilinear_clamped(*context.image_rgba, context.image_width, context.image_height, u, v);
}

static std::vector<uint8_t> projected_raw_offsets_at_point(const ProjectionContext &context,
                                                           const ImageMapRawFilamentOffsetAtlas &atlas,
                                                           const Transform3d &world_matrix,
                                                           const Vec3f &point)
{
    if (!atlas.valid() || context.overlay_width <= 0.f || context.overlay_height <= 0.f)
        return {};

    const Vec3d world_point = world_matrix * point.cast<double>();
    if (!projection_world_point_visible_in_section(context, world_point))
        return {};

    Vec2f screen = Vec2f::Zero();
    if (!project_point_to_screen(context, world_point, screen))
        return {};
    if (screen.x() < context.overlay_left ||
        screen.y() < context.overlay_top ||
        screen.x() > context.overlay_left + context.overlay_width ||
        screen.y() > context.overlay_top + context.overlay_height)
        return {};

    const float u = (screen.x() - context.overlay_left) / context.overlay_width;
    const float v = (screen.y() - context.overlay_top) / context.overlay_height;
    return sample_raw_offsets_bilinear_clamped(atlas, u, v);
}

static bool projection_triangle_intersects_overlay(const ProjectionContext      &context,
                                                   const Transform3d           &world_matrix,
                                                   const std::array<Vec3f, 3>  &vertices)
{
    const std::vector<Vec3d> polygon = projection_visible_world_polygon(context, world_matrix, vertices);
    if (polygon.empty())
        return false;

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    bool any_projected = false;

    for (const Vec3d &vertex : polygon) {
        Vec2f screen = Vec2f::Zero();
        if (!project_point_to_depth_clipped_screen(context, vertex, screen))
            continue;
        min_x = std::min(min_x, screen.x());
        min_y = std::min(min_y, screen.y());
        max_x = std::max(max_x, screen.x());
        max_y = std::max(max_y, screen.y());
        any_projected = true;
    }

    if (!any_projected)
        return false;
    return max_x >= context.overlay_left &&
           min_x <= context.overlay_left + context.overlay_width &&
           max_y >= context.overlay_top &&
           min_y <= context.overlay_top + context.overlay_height;
}

static bool project_point_to_image_pixel(const ProjectionContext &context,
                                         const Vec3d             &world_point,
                                         Vec2f                   &image_pixel)
{
    if (context.overlay_width <= EPSILON ||
        context.overlay_height <= EPSILON ||
        context.image_width == 0 ||
        context.image_height == 0)
        return false;

    Vec2f screen = Vec2f::Zero();
    if (!project_point_to_screen(context, world_point, screen))
        return false;

    image_pixel.x() = ((screen.x() - context.overlay_left) / context.overlay_width) * float(context.image_width);
    image_pixel.y() = ((screen.y() - context.overlay_top) / context.overlay_height) * float(context.image_height);
    return std::isfinite(image_pixel.x()) && std::isfinite(image_pixel.y());
}

static bool project_depth_clipped_point_to_image_pixel(const ProjectionContext &context,
                                                       const Vec3d             &world_point,
                                                       Vec2f                   &image_pixel)
{
    if (context.overlay_width <= EPSILON ||
        context.overlay_height <= EPSILON ||
        context.image_width == 0 ||
        context.image_height == 0)
        return false;

    Vec2f screen = Vec2f::Zero();
    if (!project_point_to_depth_clipped_screen(context, world_point, screen))
        return false;

    image_pixel.x() = ((screen.x() - context.overlay_left) / context.overlay_width) * float(context.image_width);
    image_pixel.y() = ((screen.y() - context.overlay_top) / context.overlay_height) * float(context.image_height);
    return std::isfinite(image_pixel.x()) && std::isfinite(image_pixel.y());
}

static float projection_triangle_image_pixel_span(const ProjectionContext     &context,
                                                  const Transform3d          &world_matrix,
                                                  const std::array<Vec3f, 3> &vertices)
{
    const std::vector<Vec3d> polygon = projection_visible_world_polygon(context, world_matrix, vertices);
    if (polygon.empty())
        return 0.f;

    std::vector<Vec2f> image_pixels;
    image_pixels.reserve(polygon.size());
    size_t projected_count = 0;
    for (const Vec3d &vertex : polygon) {
        Vec2f image_pixel = Vec2f::Zero();
        if (project_point_to_image_pixel(context, vertex, image_pixel)) {
            image_pixels.emplace_back(image_pixel);
            ++projected_count;
        }
    }

    if (projected_count < 2)
        return 0.f;

    float span = 0.f;
    for (size_t i = 0; i + 1 < projected_count; ++i)
        for (size_t j = i + 1; j < projected_count; ++j)
            span = std::max(span, (image_pixels[i] - image_pixels[j]).norm());
    return span;
}

static void projection_clip_screen_polygon(std::vector<Vec2f> &polygon, int axis, float bound, bool keep_greater)
{
    if (polygon.empty())
        return;

    std::vector<Vec2f> clipped;
    clipped.reserve(polygon.size() + 1);

    auto component = [axis](const Vec2f &point) {
        return axis == 0 ? point.x() : point.y();
    };
    auto inside = [&component, bound, keep_greater](const Vec2f &point) {
        return keep_greater ? component(point) >= bound : component(point) <= bound;
    };

    Vec2f previous = polygon.back();
    bool previous_inside = inside(previous);
    for (const Vec2f &current : polygon) {
        const bool current_inside = inside(current);
        if (current_inside != previous_inside) {
            const float denom = component(current) - component(previous);
            if (std::abs(denom) > EPSILON) {
                const float t = std::clamp((bound - component(previous)) / denom, 0.f, 1.f);
                clipped.emplace_back(previous + (current - previous) * t);
            }
        }
        if (current_inside)
            clipped.emplace_back(current);
        previous = current;
        previous_inside = current_inside;
    }

    polygon = std::move(clipped);
}

static float projection_triangle_overlay_image_pixel_span(const ProjectionContext     &context,
                                                          const Transform3d          &world_matrix,
                                                          const std::array<Vec3f, 3> &vertices)
{
    if (context.overlay_width <= EPSILON ||
        context.overlay_height <= EPSILON ||
        context.image_width == 0 ||
        context.image_height == 0)
        return 0.f;

    const std::vector<Vec3d> world_polygon = projection_visible_world_polygon(context, world_matrix, vertices);
    if (world_polygon.empty())
        return 0.f;

    std::vector<Vec2f> polygon;
    polygon.reserve(world_polygon.size());
    for (const Vec3d &vertex : world_polygon) {
        Vec2f screen = Vec2f::Zero();
        if (project_point_to_depth_clipped_screen(context, vertex, screen))
            polygon.emplace_back(screen);
    }

    if (polygon.size() < 2)
        return projection_triangle_image_pixel_span(context, world_matrix, vertices);

    projection_clip_screen_polygon(polygon, 0, context.overlay_left, true);
    projection_clip_screen_polygon(polygon, 0, context.overlay_left + context.overlay_width, false);
    projection_clip_screen_polygon(polygon, 1, context.overlay_top, true);
    projection_clip_screen_polygon(polygon, 1, context.overlay_top + context.overlay_height, false);
    if (polygon.size() < 2)
        return projection_triangle_image_pixel_span(context, world_matrix, vertices);

    float span = 0.f;
    for (size_t i = 0; i + 1 < polygon.size(); ++i) {
        const Vec2f pixel_i(((polygon[i].x() - context.overlay_left) / context.overlay_width) * float(context.image_width),
                            ((polygon[i].y() - context.overlay_top) / context.overlay_height) * float(context.image_height));
        for (size_t j = i + 1; j < polygon.size(); ++j) {
            const Vec2f pixel_j(((polygon[j].x() - context.overlay_left) / context.overlay_width) * float(context.image_width),
                                ((polygon[j].y() - context.overlay_top) / context.overlay_height) * float(context.image_height));
            span = std::max(span, (pixel_i - pixel_j).norm());
        }
    }
    return span;
}

static float image_projection_rgb_target_triangle_pixel_span(const ProjectionContext &context)
{
    const float image_span = float(std::max(context.image_width, context.image_height));
    return std::max(image_span * IMAGE_PROJECTION_RGB_TARGET_TRIANGLE_IMAGE_FRACTION,
                    IMAGE_PROJECTION_RGB_MIN_TARGET_TRIANGLE_IMAGE_PX);
}

static bool barycentric_weights_2d(const Vec2f &point, const Vec2f &a, const Vec2f &b, const Vec2f &c, Vec3f &weights)
{
    const Vec2f v0 = b - a;
    const Vec2f v1 = c - a;
    const Vec2f v2 = point - a;
    const float d00 = v0.dot(v0);
    const float d01 = v0.dot(v1);
    const float d11 = v1.dot(v1);
    const float d20 = v2.dot(v0);
    const float d21 = v2.dot(v1);
    const float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) <= EPSILON)
        return false;

    weights.y() = (d11 * d20 - d01 * d21) / denom;
    weights.z() = (d00 * d21 - d01 * d20) / denom;
    weights.x() = 1.f - weights.y() - weights.z();
    return std::isfinite(weights.x()) && std::isfinite(weights.y()) && std::isfinite(weights.z());
}

static Vec3f normalized_nonnegative_barycentric(Vec3f weights)
{
    weights.x() = std::max(weights.x(), 0.f);
    weights.y() = std::max(weights.y(), 0.f);
    weights.z() = std::max(weights.z(), 0.f);
    const float sum = weights.x() + weights.y() + weights.z();
    if (sum <= EPSILON)
        return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
    weights /= sum;
    return weights;
}

static float distance_to_segment_2d(const Vec2f &point, const Vec2f &a, const Vec2f &b)
{
    const Vec2f ab = b - a;
    const float len2 = ab.squaredNorm();
    if (len2 <= EPSILON)
        return (point - a).norm();
    const float t = std::clamp((point - a).dot(ab) / len2, 0.f, 1.f);
    return (point - (a + ab * t)).norm();
}

static bool conservative_barycentric_weights_2d(const Vec2f &point,
                                                const Vec2f &a,
                                                const Vec2f &b,
                                                const Vec2f &c,
                                                float         tolerance,
                                                Vec3f        &weights)
{
    if (!barycentric_weights_2d(point, a, b, c, weights))
        return false;
    if (weights.x() >= -1e-4f && weights.y() >= -1e-4f && weights.z() >= -1e-4f)
        return true;

    const float distance = std::min({ distance_to_segment_2d(point, a, b),
                                      distance_to_segment_2d(point, b, c),
                                      distance_to_segment_2d(point, c, a) });
    if (distance > tolerance)
        return false;

    weights = normalized_nonnegative_barycentric(weights);
    return true;
}

static std::array<Vec2f, 3> unwrap_projection_uvs(std::array<Vec2f, 3> uvs)
{
    auto unwrap_axis = [&uvs](bool use_u_axis) mutable {
        std::array<float, 3> values = {
            use_u_axis ? uvs[0].x() : uvs[0].y(),
            use_u_axis ? uvs[1].x() : uvs[1].y(),
            use_u_axis ? uvs[2].x() : uvs[2].y()
        };

        if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
            return;

        auto span = [](const std::array<float, 3> &v) {
            return std::max({ v[0], v[1], v[2] }) - std::min({ v[0], v[1], v[2] });
        };

        const bool has_repeat_evidence = std::any_of(values.begin(), values.end(), [](float value) {
            constexpr float eps = 1e-6f;
            return value < -eps || value > 1.f + eps;
        });
        const float original_span = span(values);
        if (!has_repeat_evidence || original_span <= 0.5f)
            return;

        std::array<float, 3> best = values;
        float best_span = original_span;
        for (size_t anchor = 0; anchor < values.size(); ++anchor) {
            std::array<float, 3> candidate = values;
            for (size_t i = 0; i < candidate.size(); ++i) {
                const float delta = values[i] - values[anchor];
                candidate[i] = values[anchor] + delta - std::round(delta);
            }
            const float candidate_span = span(candidate);
            if (candidate_span + 1e-6f < best_span) {
                best = candidate;
                best_span = candidate_span;
            }
        }
        if (best_span >= original_span - 1e-6f)
            return;

        if (use_u_axis) {
            uvs[0].x() = best[0];
            uvs[1].x() = best[1];
            uvs[2].x() = best[2];
        } else {
            uvs[0].y() = best[0];
            uvs[1].y() = best[1];
            uvs[2].y() = best[2];
        }
    };

    unwrap_axis(true);
    unwrap_axis(false);

    const float min_u = std::min({ uvs[0].x(), uvs[1].x(), uvs[2].x() });
    const float min_v = std::min({ uvs[0].y(), uvs[1].y(), uvs[2].y() });
    const Vec2f offset(std::floor(min_u), std::floor(min_v));
    for (Vec2f &uv : uvs)
        uv -= offset;

    return uvs;
}

static uint32_t wrapped_texture_pixel(int value, uint32_t size)
{
    if (size == 0)
        return 0;
    int wrapped = value % int(size);
    if (wrapped < 0)
        wrapped += int(size);
    return uint32_t(wrapped);
}

static VolumeColorSource build_volume_color_source(const ModelVolume &volume, const ColorFacetsAnnotation *annotation_override = nullptr)
{
    VolumeColorSource source;
    const ColorFacetsAnnotation *annotation = annotation_override != nullptr ?
        annotation_override :
        (!volume.texture_mapping_color_facets.empty() ? &volume.texture_mapping_color_facets : nullptr);
    if (annotation != nullptr && !annotation->empty()) {
        annotation->get_facet_triangles(volume, source.rgb_facets);
        source.rgb_background_color = rgb_metadata_background_color(*annotation);
        source.rgb_by_source_triangle.reserve(source.rgb_facets.size());
        for (size_t idx = 0; idx < source.rgb_facets.size(); ++idx)
            source.rgb_by_source_triangle[source.rgb_facets[idx].source_triangle].emplace_back(idx);
    }
    return source;
}

static ColorRGBA sample_volume_color_source(const ModelVolume       &volume,
                                            const VolumeColorSource &source,
                                            size_t                   tri_idx,
                                            const Vec3f             &point,
                                            const Vec3f             &barycentric,
                                            bool                     use_image_texture = true,
                                            const ColorRGBA         *fallback_color = nullptr)
{
    if (source.rgb_background_color) {
        if (std::optional<ColorRGBA> color = sample_rgb_color_facets(source.rgb_facets,
                                                                     source.rgb_by_source_triangle,
                                                                     int(tri_idx),
                                                                     point))
            return *color;
        return *source.rgb_background_color;
    }

    const indexed_triangle_set &its = volume.mesh().its;
    if (use_image_texture && model_volume_has_bakeable_image_texture_data(&volume) && tri_idx < volume.imported_texture_uv_valid.size()) {
        const size_t uv_offset = tri_idx * 6;
        if (volume.imported_texture_uv_valid[tri_idx] != 0 && uv_offset + 5 < volume.imported_texture_uvs_per_face.size()) {
            const Vec2f uv0(volume.imported_texture_uvs_per_face[uv_offset + 0], volume.imported_texture_uvs_per_face[uv_offset + 1]);
            const Vec2f uv1(volume.imported_texture_uvs_per_face[uv_offset + 2], volume.imported_texture_uvs_per_face[uv_offset + 3]);
            const Vec2f uv2(volume.imported_texture_uvs_per_face[uv_offset + 4], volume.imported_texture_uvs_per_face[uv_offset + 5]);
            return sample_texture_rgba_for_face_bake(volume.imported_texture_rgba,
                                                     volume.imported_texture_width,
                                                     volume.imported_texture_height,
                                                     uv0,
                                                     uv1,
                                                     uv2,
                                                     barycentric);
        }
    }

    if (volume.imported_vertex_colors_rgba.size() == its.vertices.size() && tri_idx < its.indices.size()) {
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        if (tri[0] >= 0 && tri[1] >= 0 && tri[2] >= 0 &&
            size_t(tri[0]) < volume.imported_vertex_colors_rgba.size() &&
            size_t(tri[1]) < volume.imported_vertex_colors_rgba.size() &&
            size_t(tri[2]) < volume.imported_vertex_colors_rgba.size()) {
            const ColorRGBA c0 = unpack_vertex_color_rgba_for_conversion(volume.imported_vertex_colors_rgba[size_t(tri[0])]);
            const ColorRGBA c1 = unpack_vertex_color_rgba_for_conversion(volume.imported_vertex_colors_rgba[size_t(tri[1])]);
            const ColorRGBA c2 = unpack_vertex_color_rgba_for_conversion(volume.imported_vertex_colors_rgba[size_t(tri[2])]);
            return ColorRGBA(c0.r() * barycentric.x() + c1.r() * barycentric.y() + c2.r() * barycentric.z(),
                             c0.g() * barycentric.x() + c1.g() * barycentric.y() + c2.g() * barycentric.z(),
                             c0.b() * barycentric.x() + c1.b() * barycentric.y() + c2.b() * barycentric.z(),
                             c0.a() * barycentric.x() + c1.a() * barycentric.y() + c2.a() * barycentric.z());
        }
    }

    return fallback_color != nullptr ? *fallback_color : ColorRGBA(1.f, 1.f, 1.f, 1.f);
}

static bool snapshot_has_bakeable_image_texture_data(const TrueColorRgbDataConversionVolumeSnapshot &snapshot)
{
    return !snapshot.its.vertices.empty() &&
           !snapshot.its.indices.empty() &&
           snapshot.imported_texture_width > 0 &&
           snapshot.imported_texture_height > 0 &&
           snapshot.imported_texture_uv_valid.size() == snapshot.its.indices.size() &&
           snapshot.imported_texture_uvs_per_face.size() >= snapshot.its.indices.size() * 6 &&
           snapshot.imported_texture_rgba.size() >=
               size_t(snapshot.imported_texture_width) * size_t(snapshot.imported_texture_height) * 4 &&
           std::any_of(snapshot.imported_texture_uv_valid.begin(), snapshot.imported_texture_uv_valid.end(), [](uint8_t valid) {
               return valid != 0;
           });
}

static ColorRGBA sample_snapshot_color_source(const TrueColorRgbDataConversionVolumeSnapshot &snapshot,
                                              size_t                                          tri_idx,
                                              const Vec3f                                    &point,
                                              const Vec3f                                    &barycentric,
                                              bool                                            use_image_texture,
                                              bool                                            use_vertex_colors,
                                              bool                                            use_color_regions,
                                              const ColorRGBA                                &fallback_color)
{
    if (use_image_texture && tri_idx < snapshot.imported_texture_uv_valid.size()) {
        if (std::optional<ColorRGBA> color =
                sample_image_texture_rgba_for_conversion(snapshot.imported_texture_rgba,
                                                         snapshot.imported_texture_width,
                                                         snapshot.imported_texture_height,
                                                         snapshot.imported_texture_uvs_per_face,
                                                         snapshot.imported_texture_uv_valid,
                                                         tri_idx,
                                                         barycentric))
            return *color;
    }

    if (use_vertex_colors) {
        if (std::optional<ColorRGBA> color =
                sample_vertex_colors_rgba_for_conversion(snapshot.its, snapshot.imported_vertex_colors_rgba, tri_idx, barycentric))
            return *color;
    }

    if (use_color_regions && snapshot.region_source) {
        if (std::optional<ColorRGBA> color = sample_managed_region_color_source(*snapshot.region_source, int(tri_idx), point))
            return *color;
    }

    return fallback_color;
}

static bool build_volume_rgb_data_from_snapshot(const TrueColorRgbDataConversionVolumeSnapshot &snapshot,
                                                const ColorRGBA                                &fallback_color,
                                                ColorFacetsAnnotation                          &out,
                                                const std::function<void()>                    &check_cancel)
{
    const indexed_triangle_set &its = snapshot.its;
    if (its.indices.empty() || its.vertices.empty())
        return false;

    const bool has_image_texture = snapshot_has_bakeable_image_texture_data(snapshot);
    const bool has_vertex_colors = snapshot.imported_vertex_colors_rgba.size() == its.vertices.size();
    const bool has_color_regions = snapshot.region_source && !snapshot.mmu_segmentation_data.triangles_to_split.empty();
    const bool use_image_texture = has_image_texture;
    const bool use_vertex_colors = !use_image_texture && has_vertex_colors;
    const bool use_color_regions = !use_image_texture && !use_vertex_colors && has_color_regions;

    out.reset();
    if (use_color_regions) {
        if (std::unique_ptr<ColorFacetsAnnotation> rgb_data =
                build_rgba_data_from_color_region_data(snapshot.mmu_segmentation_data,
                                                       its.indices.size(),
                                                       *snapshot.region_source,
                                                       fallback_color,
                                                       check_cancel)) {
            out.assign(std::move(*rgb_data));
            return !out.empty();
        }
    }

    TextureMappingColorSampler sampler = [&snapshot,
                                          fallback_color,
                                          use_image_texture,
                                          use_vertex_colors,
                                          use_color_regions,
                                          &check_cancel](size_t tri_idx, const Vec3f &point, const Vec3f &barycentric) {
        if (check_cancel)
            check_cancel();
        return pack_vertex_color_rgba(sample_snapshot_color_source(snapshot,
                                                                   tri_idx,
                                                                   point,
                                                                   barycentric,
                                                                   use_image_texture,
                                                                   use_vertex_colors,
                                                                   use_color_regions,
                                                                   fallback_color));
    };

    return build_rgba_data_from_color_sampler(its,
                                              sampler,
                                              use_image_texture,
                                              use_vertex_colors || use_color_regions,
                                              [&snapshot](size_t tri_idx) { return texture_triangle_uv_pixel_span(snapshot, tri_idx); },
                                              fallback_color,
                                              out,
                                              check_cancel);
}

static bool build_volume_rgb_data_from_current_surface_color(const ModelVolume &volume,
                                                             const ColorRGBA   &fallback_color,
                                                             ColorFacetsAnnotation &out)
{
    const indexed_triangle_set &its = volume.mesh().its;
    if (its.indices.empty() || its.vertices.empty())
        return false;

    const bool has_image_texture = model_volume_has_bakeable_image_texture_data(&volume);
    const bool has_vertex_colors = volume.imported_vertex_colors_rgba.size() == its.vertices.size();
    const bool has_color_regions = !volume.mmu_segmentation_facets.empty();
    const bool use_image_texture = has_image_texture;
    const bool use_vertex_colors = !use_image_texture && has_vertex_colors;
    const bool use_color_regions = !use_image_texture && !use_vertex_colors && has_color_regions;
    if (!use_image_texture && !use_vertex_colors && !use_color_regions)
        return build_volume_rgb_data(volume, fallback_color, out);

    out.reset();
    if (use_color_regions) {
        if (std::unique_ptr<ColorFacetsAnnotation> rgb_data = build_rgba_data_from_color_regions(volume, fallback_color)) {
            out.assign(std::move(*rgb_data));
            return !out.empty();
        }
    }

    const VolumeColorSource source;
    const ManagedRegionColorSource region_source = use_color_regions ? build_managed_region_color_source(volume) :
                                                                      ManagedRegionColorSource();
    TextureMappingColorSampler sampler = [&volume,
                                          source,
                                          region_source,
                                          fallback_color,
                                          use_image_texture,
                                          use_vertex_colors,
                                          use_color_regions](size_t tri_idx, const Vec3f &point, const Vec3f &barycentric) {
        return pack_vertex_color_rgba(sample_managed_volume_color_source(volume,
                                                                         source,
                                                                         region_source,
                                                                         tri_idx,
                                                                         point,
                                                                         barycentric,
                                                                         false,
                                                                         use_image_texture,
                                                                         use_vertex_colors,
                                                                         use_color_regions,
                                                                         fallback_color));
    };

    return build_rgba_data_from_color_sampler(its,
                                              sampler,
                                              use_image_texture,
                                              use_vertex_colors || use_color_regions,
                                              [&volume](size_t tri_idx) { return texture_triangle_uv_pixel_span(&volume, tri_idx); },
                                              fallback_color,
                                              out);
}

static bool initialize_volume_rgb_data_from_current_surface_color(ModelVolume &volume, const ColorRGBA &fallback_color)
{
    std::unique_ptr<ColorFacetsAnnotation> rgb_data = ColorFacetsAnnotation::make_temporary();
    if (!rgb_data || !build_volume_rgb_data_from_current_surface_color(volume, fallback_color, *rgb_data))
        return false;
    if (volume.texture_mapping_color_facets.equals(*rgb_data))
        return false;
    volume.texture_mapping_color_facets.assign(*rgb_data);
    return true;
}

static ColorRGBA projection_base_color_for_volume(const ModelVolume &volume)
{
    std::vector<ColorRGBA> colors = get_extruders_colors();
    if (!colors.empty()) {
        int extruder_idx = volume.extruder_id() > 0 ? volume.extruder_id() - 1 : 0;
        extruder_idx = std::clamp(extruder_idx, 0, int(colors.size() - 1));
        ColorRGBA color = colors[size_t(extruder_idx)];
        color.a(1.f);
        return color;
    }
    return ColorRGBA(0.15f, 0.65f, 0.6f, 1.f);
}

static constexpr uint32_t GENERATED_IMAGE_TEXTURE_SIZE = 4096;
static constexpr int GENERATED_IMAGE_TEXTURE_UV_MAP_VERSION = 1;

struct GeneratedImageTextureIsland
{
    size_t tri_idx = 0;
    std::array<Vec2f, 3> local_uvs;
    float width = 0.f;
    float height = 0.f;
    int rect_width = 0;
    int rect_height = 0;
    int x = 0;
    int y = 0;
};

struct GeneratedImageTextureAtlas
{
    int padding_px = 0;
    float scale = 0.f;
    std::vector<GeneratedImageTextureIsland> islands;
    std::vector<int> island_by_triangle;
};

static bool make_generated_image_texture_island(const indexed_triangle_set &its,
                                                size_t                      tri_idx,
                                                const Transform3d          *metric_matrix,
                                                GeneratedImageTextureIsland &island)
{
    if (tri_idx >= its.indices.size())
        return false;

    const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
    if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
        return false;
    if (size_t(tri[0]) >= its.vertices.size() ||
        size_t(tri[1]) >= its.vertices.size() ||
        size_t(tri[2]) >= its.vertices.size())
        return false;

    auto transformed_vertex = [&its, &tri, metric_matrix](int corner) -> Vec3f {
        const Vec3f local = its.vertices[size_t(tri[corner])].cast<float>();
        if (metric_matrix == nullptr)
            return local;
        return ((*metric_matrix) * local.cast<double>()).cast<float>();
    };
    const std::array<Vec3f, 3> vertices = { transformed_vertex(0), transformed_vertex(1), transformed_vertex(2) };
    const float edge_01 = (vertices[1] - vertices[0]).norm();
    const float edge_12 = (vertices[2] - vertices[1]).norm();
    const float edge_20 = (vertices[0] - vertices[2]).norm();

    int a = 0;
    int b = 1;
    int c = 2;
    float base_length = edge_01;
    if (edge_12 > base_length) {
        a = 1;
        b = 2;
        c = 0;
        base_length = edge_12;
    }
    if (edge_20 > base_length) {
        a = 2;
        b = 0;
        c = 1;
        base_length = edge_20;
    }
    if (!std::isfinite(base_length) || base_length <= EPSILON)
        return false;

    const Vec3f base = vertices[b] - vertices[a];
    const Vec3f side = vertices[c] - vertices[a];
    const float projected = side.dot(base) / base_length;
    const float height_sq = std::max(0.f, side.squaredNorm() - projected * projected);
    const float height = std::sqrt(height_sq);

    std::array<Vec2f, 3> local_uvs;
    local_uvs[size_t(a)] = Vec2f(0.f, 0.f);
    local_uvs[size_t(b)] = Vec2f(base_length, 0.f);
    local_uvs[size_t(c)] = Vec2f(projected, height);

    const float min_x = std::min({ local_uvs[0].x(), local_uvs[1].x(), local_uvs[2].x() });
    const float min_y = std::min({ local_uvs[0].y(), local_uvs[1].y(), local_uvs[2].y() });
    const float max_x = std::max({ local_uvs[0].x(), local_uvs[1].x(), local_uvs[2].x() });
    const float max_y = std::max({ local_uvs[0].y(), local_uvs[1].y(), local_uvs[2].y() });
    for (Vec2f &uv : local_uvs)
        uv -= Vec2f(min_x, min_y);

    island.tri_idx = tri_idx;
    island.local_uvs = local_uvs;
    island.width = std::max(max_x - min_x, 1e-4f);
    island.height = std::max(max_y - min_y, 1e-4f);
    return std::isfinite(island.width) && std::isfinite(island.height);
}

static bool pack_generated_image_texture_islands(std::vector<GeneratedImageTextureIsland> &islands,
                                                 uint32_t                                  texture_size,
                                                 int                                       padding_px,
                                                 float                                     scale)
{
    if (islands.empty() || texture_size == 0 || !std::isfinite(scale) || scale <= 0.f)
        return false;

    const int atlas_size = int(texture_size);
    for (GeneratedImageTextureIsland &island : islands) {
        const int content_width = std::max(1, int(std::ceil(island.width * scale))) + 1;
        const int content_height = std::max(1, int(std::ceil(island.height * scale))) + 1;
        island.rect_width = content_width + padding_px * 2;
        island.rect_height = content_height + padding_px * 2;
        if (island.rect_width > atlas_size || island.rect_height > atlas_size)
            return false;
    }

    std::vector<size_t> order(islands.size());
    for (size_t idx = 0; idx < order.size(); ++idx)
        order[idx] = idx;
    std::sort(order.begin(), order.end(), [&islands](size_t lhs, size_t rhs) {
        const GeneratedImageTextureIsland &a = islands[lhs];
        const GeneratedImageTextureIsland &b = islands[rhs];
        if (a.rect_height != b.rect_height)
            return a.rect_height > b.rect_height;
        if (a.rect_width != b.rect_width)
            return a.rect_width > b.rect_width;
        return a.tri_idx < b.tri_idx;
    });

    int x = 0;
    int y = 0;
    int row_height = 0;
    for (const size_t island_idx : order) {
        GeneratedImageTextureIsland &island = islands[island_idx];
        if (x + island.rect_width > atlas_size) {
            y += row_height;
            x = 0;
            row_height = 0;
        }
        if (y + island.rect_height > atlas_size)
            return false;
        island.x = x;
        island.y = y;
        x += island.rect_width;
        row_height = std::max(row_height, island.rect_height);
    }

    return true;
}

static bool pack_generated_image_texture_atlas(GeneratedImageTextureAtlas &atlas, uint32_t texture_size)
{
    if (atlas.islands.empty())
        return false;

    float max_dimension = 0.f;
    for (const GeneratedImageTextureIsland &island : atlas.islands)
        max_dimension = std::max({ max_dimension, island.width, island.height });
    if (!std::isfinite(max_dimension) || max_dimension <= EPSILON)
        return false;

    const std::array<int, 3> padding_options = { 2, 1, 0 };
    for (const int padding_px : padding_options) {
        const float available = float(int(texture_size) - padding_px * 2 - 1);
        if (available <= 0.f)
            continue;

        float high = available / max_dimension;
        if (!std::isfinite(high) || high <= 0.f)
            continue;

        bool found = false;
        std::vector<GeneratedImageTextureIsland> best;
        float best_scale = 0.f;
        float upper = high;
        for (int attempt = 0; attempt < 12; ++attempt) {
            std::vector<GeneratedImageTextureIsland> candidate = atlas.islands;
            if (pack_generated_image_texture_islands(candidate, texture_size, padding_px, high)) {
                found = true;
                best = std::move(candidate);
                best_scale = high;
                break;
            }
            upper = high;
            high *= 0.5f;
        }
        if (!found)
            continue;

        float low = best_scale;
        high = upper;
        for (int iter = 0; iter < 24; ++iter) {
            const float mid = (low + high) * 0.5f;
            std::vector<GeneratedImageTextureIsland> candidate = atlas.islands;
            if (pack_generated_image_texture_islands(candidate, texture_size, padding_px, mid)) {
                low = mid;
                best = std::move(candidate);
                best_scale = mid;
            } else {
                high = mid;
            }
        }

        atlas.padding_px = padding_px;
        atlas.scale = best_scale;
        atlas.islands = std::move(best);
        atlas.island_by_triangle.assign(atlas.island_by_triangle.size(), -1);
        for (size_t island_idx = 0; island_idx < atlas.islands.size(); ++island_idx)
            if (atlas.islands[island_idx].tri_idx < atlas.island_by_triangle.size())
                atlas.island_by_triangle[atlas.islands[island_idx].tri_idx] = int(island_idx);
        return true;
    }

    return false;
}

static bool initialize_generated_image_texture(ModelVolume                &volume,
                                               const ColorRGBA            &background,
                                               GeneratedImageTextureAtlas *atlas_out,
                                               const Transform3d          *metric_matrix = nullptr)
{
    const indexed_triangle_set &its = volume.mesh().its;
    if (its.vertices.empty() || its.indices.empty())
        return false;

    GeneratedImageTextureAtlas atlas;
    atlas.island_by_triangle.assign(its.indices.size(), -1);
    atlas.islands.reserve(its.indices.size());
    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        GeneratedImageTextureIsland island;
        if (make_generated_image_texture_island(its, tri_idx, metric_matrix, island))
            atlas.islands.emplace_back(island);
    }
    if (!pack_generated_image_texture_atlas(atlas, GENERATED_IMAGE_TEXTURE_SIZE))
        return false;

    const uint8_t r = uint8_t(std::clamp(background.r(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t g = uint8_t(std::clamp(background.g(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t b = uint8_t(std::clamp(background.b(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t a = uint8_t(std::clamp(background.a(), 0.f, 1.f) * 255.f + 0.5f);

    volume.imported_texture_width = GENERATED_IMAGE_TEXTURE_SIZE;
    volume.imported_texture_height = GENERATED_IMAGE_TEXTURE_SIZE;
    volume.imported_texture_rgba.assign(size_t(GENERATED_IMAGE_TEXTURE_SIZE) * size_t(GENERATED_IMAGE_TEXTURE_SIZE) * 4, 0);
    volume.uv_map_generator_version = GENERATED_IMAGE_TEXTURE_UV_MAP_VERSION;
    clear_imported_texture_raw_atlas(volume);
    for (size_t idx = 0; idx < size_t(GENERATED_IMAGE_TEXTURE_SIZE) * size_t(GENERATED_IMAGE_TEXTURE_SIZE); ++idx) {
        volume.imported_texture_rgba[idx * 4 + 0] = r;
        volume.imported_texture_rgba[idx * 4 + 1] = g;
        volume.imported_texture_rgba[idx * 4 + 2] = b;
        volume.imported_texture_rgba[idx * 4 + 3] = a;
    }
    volume.imported_texture_uv_valid.assign(its.indices.size(), 0);
    volume.imported_texture_uvs_per_face.assign(its.indices.size() * 6, 0.f);

    const float texture_size = float(GENERATED_IMAGE_TEXTURE_SIZE);
    for (const GeneratedImageTextureIsland &island : atlas.islands) {
        if (island.tri_idx >= its.indices.size())
            continue;
        volume.imported_texture_uv_valid[island.tri_idx] = 1;
        const size_t uv_offset = island.tri_idx * 6;
        for (size_t corner = 0; corner < 3; ++corner) {
            const Vec2f pixel(float(island.x + atlas.padding_px) + 0.5f + island.local_uvs[corner].x() * atlas.scale,
                              float(island.y + atlas.padding_px) + 0.5f + island.local_uvs[corner].y() * atlas.scale);
            volume.imported_texture_uvs_per_face[uv_offset + corner * 2 + 0] = std::clamp(pixel.x() / texture_size, 0.f, 1.f);
            volume.imported_texture_uvs_per_face[uv_offset + corner * 2 + 1] = std::clamp(pixel.y() / texture_size, 0.f, 1.f);
        }
    }

    if (atlas_out != nullptr)
        *atlas_out = std::move(atlas);
    return true;
}

static bool write_rgba_pixel(std::vector<uint8_t> &rgba, uint32_t width, uint32_t x, uint32_t y, const ColorRGBA &color)
{
    if (width == 0)
        return false;
    const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4;
    if (idx + 3 >= rgba.size())
        return false;
    const uint8_t r = uint8_t(std::clamp(color.r(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t g = uint8_t(std::clamp(color.g(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t b = uint8_t(std::clamp(color.b(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t a = uint8_t(std::clamp(color.a(), 0.f, 1.f) * 255.f + 0.5f);
    if (rgba[idx + 0] == r && rgba[idx + 1] == g && rgba[idx + 2] == b && rgba[idx + 3] == a)
        return false;
    rgba[idx + 0] = r;
    rgba[idx + 1] = g;
    rgba[idx + 2] = b;
    rgba[idx + 3] = a;
    return true;
}

static bool fill_rgba_pixels(std::vector<uint8_t> &rgba, const ColorRGBA &color)
{
    const uint8_t r = uint8_t(std::clamp(color.r(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t g = uint8_t(std::clamp(color.g(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t b = uint8_t(std::clamp(color.b(), 0.f, 1.f) * 255.f + 0.5f);
    const uint8_t a = uint8_t(std::clamp(color.a(), 0.f, 1.f) * 255.f + 0.5f);
    bool changed = false;
    for (size_t idx = 0; idx + 3 < rgba.size(); idx += 4) {
        changed = changed ||
                  rgba[idx + 0] != r ||
                  rgba[idx + 1] != g ||
                  rgba[idx + 2] != b ||
                  rgba[idx + 3] != a;
        rgba[idx + 0] = r;
        rgba[idx + 1] = g;
        rgba[idx + 2] = b;
        rgba[idx + 3] = a;
    }
    return changed;
}

static bool write_raw_offset_pixel(std::vector<uint8_t> &offsets,
                                   uint32_t width,
                                   uint32_t channels,
                                   uint32_t x,
                                   uint32_t y,
                                   const std::vector<uint8_t> &values)
{
    if (width == 0 || channels == 0 || values.empty())
        return false;
    const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * size_t(channels);
    if (idx + size_t(channels) > offsets.size())
        return false;

    bool changed = false;
    for (size_t channel = 0; channel < size_t(channels); ++channel) {
        const uint8_t value = channel < values.size() ? values[channel] : 0;
        changed = changed || offsets[idx + channel] != value;
        offsets[idx + channel] = value;
    }
    return changed;
}

static ColorRGBA read_rgba_pixel(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t x, uint32_t y)
{
    if (width == 0)
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4;
    if (idx + 3 >= rgba.size())
        return ColorRGBA(1.f, 1.f, 1.f, 1.f);
    return ColorRGBA(float(rgba[idx + 0]) / 255.f,
                     float(rgba[idx + 1]) / 255.f,
                     float(rgba[idx + 2]) / 255.f,
                     float(rgba[idx + 3]) / 255.f);
}

static Transform3d projection_world_matrix_for_volume(const GLCanvas3D &parent,
                                                      const ModelObject *object,
                                                      const ModelVolume *volume,
                                                      int                instance_idx)
{
    if (object == nullptr || volume == nullptr || object->instances.empty())
        return Transform3d::Identity();

    instance_idx = std::clamp(instance_idx, 0, int(object->instances.size() - 1));
    const ModelInstance *instance = object->instances[size_t(instance_idx)];
    if (parent.get_canvas_type() == GLCanvas3D::CanvasAssembleView)
        return instance->get_assemble_transformation().get_matrix() * volume->get_matrix();
    return instance->get_transformation().get_matrix() * volume->get_matrix();
}

static std::vector<Vec3d> projection_smoothed_vertex_normals(const indexed_triangle_set &its)
{
    std::vector<Vec3d> normals(its.vertices.size(), Vec3d::Zero());
    for (const stl_triangle_vertex_indices &tri : its.indices) {
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
            continue;
        if (size_t(tri[0]) >= its.vertices.size() ||
            size_t(tri[1]) >= its.vertices.size() ||
            size_t(tri[2]) >= its.vertices.size())
            continue;

        const Vec3d v0 = its.vertices[size_t(tri[0])].cast<double>();
        const Vec3d v1 = its.vertices[size_t(tri[1])].cast<double>();
        const Vec3d v2 = its.vertices[size_t(tri[2])].cast<double>();
        const Vec3d normal = (v1 - v0).cross(v2 - v0);
        if (normal.squaredNorm() <= EPSILON)
            continue;

        normals[size_t(tri[0])] += normal;
        normals[size_t(tri[1])] += normal;
        normals[size_t(tri[2])] += normal;
    }

    for (Vec3d &normal : normals)
        if (normal.squaredNorm() > EPSILON)
            normal.normalize();
    return normals;
}

static Vec3d projection_interpolated_local_normal(const std::vector<Vec3d>         &vertex_normals,
                                                  const stl_triangle_vertex_indices &tri,
                                                  const Vec3f                      &barycentric)
{
    if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
        return Vec3d::Zero();
    if (size_t(tri[0]) >= vertex_normals.size() ||
        size_t(tri[1]) >= vertex_normals.size() ||
        size_t(tri[2]) >= vertex_normals.size())
        return Vec3d::Zero();

    Vec3d normal = vertex_normals[size_t(tri[0])] * double(barycentric.x()) +
                   vertex_normals[size_t(tri[1])] * double(barycentric.y()) +
                   vertex_normals[size_t(tri[2])] * double(barycentric.z());
    if (normal.squaredNorm() > EPSILON)
        normal.normalize();
    return normal;
}

static bool projection_point_allowed_by_camera_facing(const ProjectionContext            &context,
                                                      const Transform3d                 &world_matrix,
                                                      const Matrix3d                    &world_normal_matrix,
                                                      const std::vector<Vec3d>          &vertex_normals,
                                                      const stl_triangle_vertex_indices &tri,
                                                      const Vec3f                       &point,
                                                      const Vec3f                       &barycentric)
{
    if (!projection_section_view_active(context))
        return true;

    const Vec3d world_point = world_matrix * point.cast<double>();
    const Vec3d local_normal = projection_interpolated_local_normal(vertex_normals, tri, barycentric);
    Vec3d world_normal = world_normal_matrix * local_normal;
    if (world_normal.squaredNorm() > EPSILON)
        world_normal.normalize();
    return projection_sample_allowed_by_camera_facing(context, world_point, world_normal);
}

struct ProjectionVisibility
{
    int                width = 0;
    int                height = 0;
    float              left = 0.f;
    float              top = 0.f;
    float              scale = 1.f;
    std::vector<float> depth;
    std::vector<float> local_depth_tolerance;
    std::vector<uint64_t> triangle_keys;
};

static constexpr float PROJECTION_VISIBILITY_DEPTH_TOLERANCE = 2e-4f;
static constexpr float PROJECTION_VISIBILITY_SAME_TRIANGLE_DEPTH_TOLERANCE = 2e-3f;
static constexpr float PROJECTION_VISIBILITY_PROJECTED_TRIANGLE_DEPTH_TOLERANCE = 5e-3f;
static constexpr float PROJECTION_VISIBILITY_MAX_LOCAL_DEPTH_TOLERANCE = 2e-2f;
static constexpr uint64_t PROJECTION_VISIBILITY_INVALID_TRIANGLE_KEY = std::numeric_limits<uint64_t>::max();

static uint64_t projection_visibility_triangle_key(size_t volume_idx, size_t tri_idx)
{
    return (uint64_t(volume_idx) << 32) | uint64_t(tri_idx);
}

static bool projection_visibility_valid(const ProjectionVisibility &visibility)
{
    return visibility.width > 0 &&
           visibility.height > 0 &&
           visibility.depth.size() == size_t(visibility.width) * size_t(visibility.height) &&
           visibility.local_depth_tolerance.size() == visibility.depth.size() &&
           visibility.triangle_keys.size() == visibility.depth.size();
}

static void projection_visibility_prepare_local_depth_tolerances(ProjectionVisibility &visibility)
{
    visibility.local_depth_tolerance.assign(visibility.depth.size(), PROJECTION_VISIBILITY_DEPTH_TOLERANCE);
    if (visibility.width <= 0 || visibility.height <= 0 || visibility.depth.size() != size_t(visibility.width) * size_t(visibility.height))
        return;

    for (int y = 0; y < visibility.height; ++y) {
        for (int x = 0; x < visibility.width; ++x) {
            const size_t center_idx = size_t(y) * size_t(visibility.width) + size_t(x);
            const float center_nearest = visibility.depth[center_idx];
            if (!std::isfinite(center_nearest))
                continue;

            float tolerance = PROJECTION_VISIBILITY_DEPTH_TOLERANCE;
            const int min_x = std::max(0, x - 1);
            const int max_x = std::min(visibility.width - 1, x + 1);
            const int min_y = std::max(0, y - 1);
            const int max_y = std::min(visibility.height - 1, y + 1);
            for (int sample_y = min_y; sample_y <= max_y; ++sample_y) {
                for (int sample_x = min_x; sample_x <= max_x; ++sample_x) {
                    const size_t idx = size_t(sample_y) * size_t(visibility.width) + size_t(sample_x);
                    const float nearest = visibility.depth[idx];
                    if (std::isfinite(nearest))
                        tolerance = std::max(tolerance, std::abs(center_nearest - nearest) + PROJECTION_VISIBILITY_DEPTH_TOLERANCE);
                }
            }
            visibility.local_depth_tolerance[center_idx] = std::min(tolerance, PROJECTION_VISIBILITY_MAX_LOCAL_DEPTH_TOLERANCE);
        }
    }
}

static float projection_visibility_local_depth_tolerance(const ProjectionVisibility &visibility, int x, int y)
{
    if (x < 0 || y < 0 || x >= visibility.width || y >= visibility.height)
        return PROJECTION_VISIBILITY_DEPTH_TOLERANCE;

    const size_t idx = size_t(y) * size_t(visibility.width) + size_t(x);
    if (idx >= visibility.local_depth_tolerance.size())
        return PROJECTION_VISIBILITY_DEPTH_TOLERANCE;
    return visibility.local_depth_tolerance[idx];
}

static bool projection_visibility_depth_matches_sample(const ProjectionVisibility &visibility,
                                                       int                         x,
                                                       int                         y,
                                                       float                       depth,
                                                       uint64_t                    triangle_key)
{
    if (x < 0 || y < 0 || x >= visibility.width || y >= visibility.height)
        return false;

    const size_t center_idx = size_t(y) * size_t(visibility.width) + size_t(x);
    const float center_nearest = visibility.depth[center_idx];

    const bool has_triangle_key = triangle_key != PROJECTION_VISIBILITY_INVALID_TRIANGLE_KEY;
    const bool center_same_triangle =
        has_triangle_key && visibility.triangle_keys[center_idx] == triangle_key;
    const float local_tolerance = projection_visibility_local_depth_tolerance(visibility, x, y);
    float center_tolerance = PROJECTION_VISIBILITY_DEPTH_TOLERANCE;
    if (center_same_triangle) {
        center_tolerance = std::max(PROJECTION_VISIBILITY_SAME_TRIANGLE_DEPTH_TOLERANCE, local_tolerance);
    } else if (has_triangle_key) {
        center_tolerance = std::max(PROJECTION_VISIBILITY_PROJECTED_TRIANGLE_DEPTH_TOLERANCE, local_tolerance);
    }
    if (std::isfinite(center_nearest) && depth <= center_nearest + center_tolerance)
        return true;

    if (!has_triangle_key)
        return false;

    const int search_radius = std::isfinite(center_nearest) ? 1 : 2;
    const int min_x = std::max(0, x - search_radius);
    const int max_x = std::min(visibility.width - 1, x + search_radius);
    const int min_y = std::max(0, y - search_radius);
    const int max_y = std::min(visibility.height - 1, y + search_radius);
    for (int sample_y = min_y; sample_y <= max_y; ++sample_y) {
        for (int sample_x = min_x; sample_x <= max_x; ++sample_x) {
            if (sample_x == x && sample_y == y)
                continue;

            const size_t idx = size_t(sample_y) * size_t(visibility.width) + size_t(sample_x);
            if (visibility.triangle_keys[idx] != triangle_key)
                continue;

            const float nearest = visibility.depth[idx];
            const float nearby_tolerance =
                std::max(PROJECTION_VISIBILITY_SAME_TRIANGLE_DEPTH_TOLERANCE,
                         projection_visibility_local_depth_tolerance(visibility, sample_x, sample_y));
            if (std::isfinite(nearest) && depth <= nearest + nearby_tolerance)
                return true;
        }
    }

    for (int sample_y = min_y; sample_y <= max_y; ++sample_y) {
        for (int sample_x = min_x; sample_x <= max_x; ++sample_x) {
            if (sample_x == x && sample_y == y)
                continue;

            const size_t idx = size_t(sample_y) * size_t(visibility.width) + size_t(sample_x);
            const float nearest = visibility.depth[idx];
            const float nearby_tolerance =
                std::max(PROJECTION_VISIBILITY_PROJECTED_TRIANGLE_DEPTH_TOLERANCE,
                         projection_visibility_local_depth_tolerance(visibility, sample_x, sample_y));
            if (std::isfinite(nearest) && depth <= nearest + nearby_tolerance)
                return true;
        }
    }

    return false;
}

static ProjectionVisibility build_projection_visibility(const ProjectionContext &context,
                                                        const GLCanvas3D       &parent,
                                                        const ModelObject      *object,
                                                        int                     instance_idx)
{
    ProjectionVisibility visibility;
    if (object == nullptr || context.overlay_width <= 0.f || context.overlay_height <= 0.f)
        return visibility;

    const float max_dim = std::max(context.overlay_width, context.overlay_height);
    visibility.scale = max_dim > 2048.f ? 2048.f / max_dim : 1.f;
    visibility.width = std::max(1, int(std::ceil(context.overlay_width * visibility.scale)));
    visibility.height = std::max(1, int(std::ceil(context.overlay_height * visibility.scale)));
    visibility.left = context.overlay_left;
    visibility.top = context.overlay_top;
    visibility.depth.assign(size_t(visibility.width) * size_t(visibility.height), std::numeric_limits<float>::max());
    visibility.triangle_keys.assign(visibility.depth.size(), PROJECTION_VISIBILITY_INVALID_TRIANGLE_KEY);

    auto rasterize_triangle = [&visibility](const std::array<Vec2f, 3> &screen, const std::array<float, 3> &depths, uint64_t triangle_key) {
        const float min_screen_x = std::min({ screen[0].x(), screen[1].x(), screen[2].x() });
        const float max_screen_x = std::max({ screen[0].x(), screen[1].x(), screen[2].x() });
        const float min_screen_y = std::min({ screen[0].y(), screen[1].y(), screen[2].y() });
        const float max_screen_y = std::max({ screen[0].y(), screen[1].y(), screen[2].y() });
        const int min_x = std::clamp(int(std::floor((min_screen_x - visibility.left) * visibility.scale)) - 1, 0, visibility.width - 1);
        const int max_x = std::clamp(int(std::ceil((max_screen_x - visibility.left) * visibility.scale)) + 1, 0, visibility.width - 1);
        const int min_y = std::clamp(int(std::floor((min_screen_y - visibility.top) * visibility.scale)) - 1, 0, visibility.height - 1);
        const int max_y = std::clamp(int(std::ceil((max_screen_y - visibility.top) * visibility.scale)) + 1, 0, visibility.height - 1);

        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                const Vec2f pixel(visibility.left + (float(x) + 0.5f) / visibility.scale,
                                  visibility.top + (float(y) + 0.5f) / visibility.scale);
                Vec3f weights = Vec3f::Zero();
                if (!conservative_barycentric_weights_2d(pixel,
                                                          screen[0],
                                                          screen[1],
                                                          screen[2],
                                                          0.7072f / visibility.scale,
                                                          weights))
                    continue;

                const float depth = depths[0] * weights.x() + depths[1] * weights.y() + depths[2] * weights.z();
                const size_t idx = size_t(y) * size_t(visibility.width) + size_t(x);
                if (depth < visibility.depth[idx]) {
                    visibility.depth[idx] = depth;
                    visibility.triangle_keys[idx] = triangle_key;
                }
            }
        }
    };

    for (size_t volume_idx = 0; volume_idx < object->volumes.size(); ++volume_idx) {
        const ModelVolume *volume = object->volumes[volume_idx];
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const Transform3d world_matrix = projection_world_matrix_for_volume(parent, object, volume, instance_idx);
        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            const std::vector<Vec3d> polygon = projection_visible_world_polygon(context, world_matrix, vertices);
            if (polygon.size() < 3 || !projection_triangle_intersects_overlay(context, world_matrix, vertices))
                continue;

            for (size_t polygon_idx = 1; polygon_idx + 1 < polygon.size(); ++polygon_idx) {
                const std::array<Vec3d, 3> fan = { polygon[0], polygon[polygon_idx], polygon[polygon_idx + 1] };
                std::array<Vec2f, 3> screen;
                std::array<float, 3> depths;
                bool projected = true;
                for (size_t idx = 0; idx < fan.size(); ++idx) {
                    if (!project_point_to_depth_clipped_screen(context, fan[idx], screen[idx], &depths[idx])) {
                        projected = false;
                        break;
                    }
                }
                if (projected)
                    rasterize_triangle(screen, depths, projection_visibility_triangle_key(volume_idx, tri_idx));
            }
        }
    }

    projection_visibility_prepare_local_depth_tolerances(visibility);
    return visibility;
}

static bool projection_point_is_visible(const ProjectionVisibility &visibility,
                                        const ProjectionContext    &context,
                                        const Transform3d         &world_matrix,
                                        const Vec3f               &point,
                                        uint64_t                   triangle_key = PROJECTION_VISIBILITY_INVALID_TRIANGLE_KEY)
{
    const Vec3d world_point = world_matrix * point.cast<double>();
    if (!projection_world_point_visible_in_section(context, world_point))
        return false;

    if (!projection_visibility_valid(visibility))
        return true;

    Vec2f screen = Vec2f::Zero();
    float depth = 0.f;
    if (!project_point_to_screen(context, world_point, screen, &depth))
        return false;

    const int x = int(std::floor((screen.x() - visibility.left) * visibility.scale));
    const int y = int(std::floor((screen.y() - visibility.top) * visibility.scale));
    if (x < 0 || y < 0 || x >= visibility.width || y >= visibility.height)
        return false;

    return projection_visibility_depth_matches_sample(visibility, x, y, depth, triangle_key);
}

static bool projection_screen_triangle_has_visible_sample(const ProjectionVisibility   &visibility,
                                                          const std::array<Vec2f, 3>  &screen,
                                                          const std::array<float, 3>  &depths,
                                                          uint64_t                     triangle_key)
{
    if (!projection_visibility_valid(visibility))
        return true;

    const float min_screen_x = std::min({ screen[0].x(), screen[1].x(), screen[2].x() });
    const float max_screen_x = std::max({ screen[0].x(), screen[1].x(), screen[2].x() });
    const float min_screen_y = std::min({ screen[0].y(), screen[1].y(), screen[2].y() });
    const float max_screen_y = std::max({ screen[0].y(), screen[1].y(), screen[2].y() });
    const int min_x = std::clamp(int(std::floor((min_screen_x - visibility.left) * visibility.scale)) - 1, 0, visibility.width - 1);
    const int max_x = std::clamp(int(std::ceil((max_screen_x - visibility.left) * visibility.scale)) + 1, 0, visibility.width - 1);
    const int min_y = std::clamp(int(std::floor((min_screen_y - visibility.top) * visibility.scale)) - 1, 0, visibility.height - 1);
    const int max_y = std::clamp(int(std::ceil((max_screen_y - visibility.top) * visibility.scale)) + 1, 0, visibility.height - 1);

    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const Vec2f pixel(visibility.left + (float(x) + 0.5f) / visibility.scale,
                              visibility.top + (float(y) + 0.5f) / visibility.scale);
            Vec3f weights = Vec3f::Zero();
            if (!conservative_barycentric_weights_2d(pixel,
                                                      screen[0],
                                                      screen[1],
                                                      screen[2],
                                                      0.7072f / visibility.scale,
                                                      weights))
                continue;

            const float depth = depths[0] * weights.x() + depths[1] * weights.y() + depths[2] * weights.z();
            if (projection_visibility_depth_matches_sample(visibility, x, y, depth, triangle_key))
                return true;
        }
    }

    return false;
}

struct ProjectionPaintableImageMask
{
    uint32_t              width = 0;
    uint32_t              height = 0;
    bool                  all_paintable = false;
    std::vector<uint32_t> prefix;
};

static ProjectionPaintableImageMask build_projection_paintable_image_mask(const ProjectionContext &context,
                                                                          bool                     transparent_bg_paintable = false)
{
    ProjectionPaintableImageMask mask;
    mask.width = context.image_width;
    mask.height = context.image_height;
    if (context.image_rgba == nullptr ||
        context.image_width == 0 ||
        context.image_height == 0 ||
        context.image_rgba->size() < size_t(context.image_width) * size_t(context.image_height) * 4)
        return mask;

    if (transparent_bg_paintable && context.apply_transparency_as_background) {
        mask.all_paintable = true;
        return mask;
    }

    const size_t stride = size_t(mask.width) + 1;
    mask.prefix.assign((size_t(mask.height) + 1) * stride, 0);
    const float opacity = std::clamp(context.image_opacity, 0.f, 1.f);
    for (uint32_t y = 0; y < mask.height; ++y) {
        uint32_t row_sum = 0;
        for (uint32_t x = 0; x < mask.width; ++x) {
            const size_t pixel_idx = (size_t(y) * size_t(mask.width) + size_t(x)) * 4 + 3;
            const bool paintable = float((*context.image_rgba)[pixel_idx]) * opacity > 0.5f;
            row_sum += paintable ? 1u : 0u;
            mask.prefix[(size_t(y) + 1) * stride + size_t(x) + 1] =
                mask.prefix[size_t(y) * stride + size_t(x) + 1] + row_sum;
        }
    }
    return mask;
}

static bool projection_image_rect_has_paintable_alpha(const ProjectionPaintableImageMask &mask,
                                                      int                                 min_x,
                                                      int                                 min_y,
                                                      int                                 max_x,
                                                      int                                 max_y)
{
    if (mask.width == 0 || mask.height == 0)
        return false;
    if (mask.all_paintable)
        return true;
    if (mask.prefix.empty())
        return false;
    if (max_x < 0 || max_y < 0 || min_x >= int(mask.width) || min_y >= int(mask.height))
        return false;

    min_x = std::clamp(min_x, 0, int(mask.width) - 1);
    max_x = std::clamp(max_x, 0, int(mask.width) - 1);
    min_y = std::clamp(min_y, 0, int(mask.height) - 1);
    max_y = std::clamp(max_y, 0, int(mask.height) - 1);
    if (max_x < min_x || max_y < min_y)
        return false;

    const size_t stride = size_t(mask.width) + 1;
    const size_t x0 = size_t(min_x);
    const size_t y0 = size_t(min_y);
    const size_t x1 = size_t(max_x) + 1;
    const size_t y1 = size_t(max_y) + 1;
    const uint32_t count = mask.prefix[y1 * stride + x1] -
                           mask.prefix[y0 * stride + x1] -
                           mask.prefix[y1 * stride + x0] +
                           mask.prefix[y0 * stride + x0];
    return count > 0;
}

static bool projection_triangle_image_pixel_bounds(const ProjectionContext     &context,
                                                   const Transform3d          &world_matrix,
                                                   const std::array<Vec3f, 3> &vertices,
                                                   float                      &min_x,
                                                   float                      &min_y,
                                                   float                      &max_x,
                                                   float                      &max_y)
{
    min_x = std::numeric_limits<float>::max();
    min_y = std::numeric_limits<float>::max();
    max_x = std::numeric_limits<float>::lowest();
    max_y = std::numeric_limits<float>::lowest();
    bool any_projected = false;

    const std::vector<Vec3d> polygon = projection_visible_world_polygon(context, world_matrix, vertices);
    if (polygon.empty())
        return false;

    for (const Vec3d &vertex : polygon) {
        Vec2f image_pixel = Vec2f::Zero();
        if (!project_depth_clipped_point_to_image_pixel(context, vertex, image_pixel))
            continue;
        min_x = std::min(min_x, image_pixel.x());
        min_y = std::min(min_y, image_pixel.y());
        max_x = std::max(max_x, image_pixel.x());
        max_y = std::max(max_y, image_pixel.y());
        any_projected = true;
    }

    return any_projected;
}

static bool projection_triangle_intersects_paintable_image(const ProjectionContext            &context,
                                                           const ProjectionPaintableImageMask &mask,
                                                           const Transform3d                 &world_matrix,
                                                           const std::array<Vec3f, 3>        &vertices)
{
    if (mask.width == 0 || mask.height == 0)
        return false;
    if (mask.all_paintable)
        return true;

    float min_x = 0.f;
    float min_y = 0.f;
    float max_x = 0.f;
    float max_y = 0.f;
    if (!projection_triangle_image_pixel_bounds(context, world_matrix, vertices, min_x, min_y, max_x, max_y))
        return false;

    return projection_image_rect_has_paintable_alpha(mask,
                                                     int(std::floor(min_x)) - 1,
                                                     int(std::floor(min_y)) - 1,
                                                     int(std::ceil(max_x)) + 1,
                                                     int(std::ceil(max_y)) + 1);
}

static bool projection_triangle_has_visible_sample(const ProjectionVisibility &visibility,
                                                   const ProjectionContext    &context,
                                                   const Transform3d         &world_matrix,
                                                   const std::array<Vec3f, 3> &vertices,
                                                   uint64_t                   triangle_key)
{
    const std::vector<Vec3d> polygon = projection_visible_world_polygon(context, world_matrix, vertices);
    if (polygon.size() < 3)
        return false;

    if (!projection_visibility_valid(visibility))
        return true;

    if (projection_point_is_visible(visibility, context, world_matrix, (vertices[0] + vertices[1] + vertices[2]) / 3.f, triangle_key))
        return true;

    for (const Vec3f &vertex : vertices)
        if (projection_point_is_visible(visibility, context, world_matrix, vertex, triangle_key))
            return true;

    for (size_t idx = 0; idx < 3; ++idx)
        if (projection_point_is_visible(visibility, context, world_matrix, (vertices[idx] + vertices[(idx + 1) % 3]) * 0.5f, triangle_key))
            return true;

    for (size_t polygon_idx = 1; polygon_idx + 1 < polygon.size(); ++polygon_idx) {
        const std::array<Vec3d, 3> fan = { polygon[0], polygon[polygon_idx], polygon[polygon_idx + 1] };
        std::array<Vec2f, 3> screen;
        std::array<float, 3> depths;
        bool projected = true;
        for (size_t idx = 0; idx < fan.size(); ++idx) {
            if (!project_point_to_depth_clipped_screen(context, fan[idx], screen[idx], &depths[idx])) {
                projected = false;
                break;
            }
        }
        if (projected && projection_screen_triangle_has_visible_sample(visibility, screen, depths, triangle_key))
            return true;
    }

    return false;
}

static bool projection_triangle_should_project(const ProjectionContext            &context,
                                               const ProjectionVisibility         &visibility,
                                               const ProjectionPaintableImageMask &paintable_mask,
                                               const Transform3d                 &world_matrix,
                                               const std::array<Vec3f, 3>        &vertices,
                                               uint64_t                           triangle_key)
{
    return projection_triangle_intersects_overlay(context, world_matrix, vertices) &&
           projection_triangle_intersects_paintable_image(context, paintable_mask, world_matrix, vertices) &&
           projection_triangle_has_visible_sample(visibility, context, world_matrix, vertices, triangle_key);
}

struct ProjectionRegionStateSource
{
    std::vector<std::vector<TriangleSelector::FacetStateTriangle>> triangles_per_type;
    std::vector<std::unordered_map<int, std::vector<size_t>>>      by_source_triangle;
    std::vector<std::vector<int>>                                  nearby_source_triangles;
    float                                                          nearby_painted_distance_sq = 0.f;
};

static ProjectionRegionStateSource build_projection_region_state_source(const ModelVolume &volume)
{
    ProjectionRegionStateSource source;
    if (volume.mmu_segmentation_facets.empty())
        return source;

    volume.mmu_segmentation_facets.get_facet_triangles(volume, source.triangles_per_type);
    source.by_source_triangle.resize(source.triangles_per_type.size());
    for (size_t state_idx = 0; state_idx < source.triangles_per_type.size(); ++state_idx) {
        std::unordered_map<int, std::vector<size_t>> &by_source = source.by_source_triangle[state_idx];
        by_source.reserve(source.triangles_per_type[state_idx].size());
        for (size_t idx = 0; idx < source.triangles_per_type[state_idx].size(); ++idx)
            by_source[source.triangles_per_type[state_idx][idx].source_triangle].emplace_back(idx);
    }

    const indexed_triangle_set &its = volume.mesh().its;
    const float mesh_span = mesh_max_axis_span(its);
    const float nearby_distance = std::max(mesh_span / 5000.f, 0.002f);
    source.nearby_painted_distance_sq = nearby_distance * nearby_distance;

    std::vector<std::vector<int>> triangles_by_vertex(its.vertices.size());
    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        for (int corner = 0; corner < 3; ++corner)
            if (tri[corner] >= 0 && size_t(tri[corner]) < triangles_by_vertex.size())
                triangles_by_vertex[size_t(tri[corner])].emplace_back(int(tri_idx));
    }

    source.nearby_source_triangles.resize(its.indices.size());
    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        std::vector<int> &nearby = source.nearby_source_triangles[tri_idx];
        nearby.emplace_back(int(tri_idx));
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        for (int corner = 0; corner < 3; ++corner) {
            if (tri[corner] < 0 || size_t(tri[corner]) >= triangles_by_vertex.size())
                continue;
            nearby.insert(nearby.end(), triangles_by_vertex[size_t(tri[corner])].begin(), triangles_by_vertex[size_t(tri[corner])].end());
        }
        std::sort(nearby.begin(), nearby.end());
        nearby.erase(std::unique(nearby.begin(), nearby.end()), nearby.end());
    }

    return source;
}

struct ProjectionRegionStateCandidate
{
    std::optional<unsigned int> inside_state;
    float                       inside_score = -std::numeric_limits<float>::max();
    std::optional<unsigned int> nearest_state;
    float                       nearest_distance_sq = std::numeric_limits<float>::max();
    std::optional<unsigned int> nearest_painted_state;
    float                       nearest_painted_distance_sq = std::numeric_limits<float>::max();
};

static void consider_projection_region_source_triangle(const ProjectionRegionStateSource &source,
                                                       int                                source_triangle,
                                                       const Vec3f                       &point,
                                                       ProjectionRegionStateCandidate    &candidate)
{
    for (size_t state_idx = 0; state_idx < source.triangles_per_type.size(); ++state_idx) {
        if (state_idx >= source.by_source_triangle.size())
            continue;

        const auto found = source.by_source_triangle[state_idx].find(source_triangle);
        if (found == source.by_source_triangle[state_idx].end())
            continue;

        const float tolerance = -1e-4f;
        const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles = source.triangles_per_type[state_idx];
        for (const size_t facet_idx : found->second) {
            if (facet_idx >= state_triangles.size())
                continue;

            const TriangleSelector::FacetStateTriangle &facet = state_triangles[facet_idx];
            Vec3f weights = Vec3f::Zero();
            if (!barycentric_weights_for_region_vertex_colors(point, facet.vertices[0], facet.vertices[1], facet.vertices[2], weights))
                continue;
            const unsigned int state = unsigned(state_idx);
            if (weights.x() >= tolerance && weights.y() >= tolerance && weights.z() >= tolerance) {
                const float score = std::min({ weights.x(), weights.y(), weights.z() });
                if (!candidate.inside_state ||
                    score > candidate.inside_score + 1e-6f ||
                    (std::abs(score - candidate.inside_score) <= 1e-6f && state > *candidate.inside_state)) {
                    candidate.inside_state = state;
                    candidate.inside_score = score;
                }
            }

            const Vec3f closest = closest_point_on_triangle(point, facet.vertices[0], facet.vertices[1], facet.vertices[2]);
            const float distance_sq = (point - closest).squaredNorm();
            if (!candidate.nearest_state ||
                distance_sq < candidate.nearest_distance_sq - 1e-8f ||
                (std::abs(distance_sq - candidate.nearest_distance_sq) <= 1e-8f && state > *candidate.nearest_state)) {
                candidate.nearest_state = state;
                candidate.nearest_distance_sq = distance_sq;
            }
            if (state > 0 &&
                (!candidate.nearest_painted_state ||
                 distance_sq < candidate.nearest_painted_distance_sq - 1e-8f ||
                 (std::abs(distance_sq - candidate.nearest_painted_distance_sq) <= 1e-8f && state > *candidate.nearest_painted_state))) {
                candidate.nearest_painted_state = state;
                candidate.nearest_painted_distance_sq = distance_sq;
            }
        }
    }
}

static unsigned int sample_projection_region_state(const ProjectionRegionStateSource &source,
                                                   int                                source_triangle,
                                                   const Vec3f                       &point)
{
    ProjectionRegionStateCandidate same_triangle;
    consider_projection_region_source_triangle(source, source_triangle, point, same_triangle);

    if (same_triangle.inside_state)
        return *same_triangle.inside_state;

    ProjectionRegionStateCandidate nearby = same_triangle;
    if (source_triangle >= 0 && size_t(source_triangle) < source.nearby_source_triangles.size()) {
        for (const int nearby_triangle : source.nearby_source_triangles[size_t(source_triangle)]) {
            if (nearby_triangle == source_triangle)
                continue;
            consider_projection_region_source_triangle(source, nearby_triangle, point, nearby);
        }
    }

    if (nearby.nearest_painted_state && nearby.nearest_painted_distance_sq <= source.nearby_painted_distance_sq)
        return *nearby.nearest_painted_state;

    if (same_triangle.nearest_state)
        return *same_triangle.nearest_state;
    if (nearby.nearest_state)
        return *nearby.nearest_state;
    return 0;
}

static bool projection_region_triangle_bitstream_range(const TriangleSelector::TriangleSplittingData &data,
                                                       size_t                                         tri_idx,
                                                       int                                           &bitstream_start,
                                                       int                                           &bitstream_end)
{
    auto mapping_it = std::lower_bound(data.triangles_to_split.begin(),
                                       data.triangles_to_split.end(),
                                       int(tri_idx),
                                       [](const TriangleSelector::TriangleBitStreamMapping &lhs, int rhs) {
                                           return lhs.triangle_idx < rhs;
                                       });
    if (mapping_it == data.triangles_to_split.end() || mapping_it->triangle_idx != int(tri_idx))
        return false;

    const auto next_it = std::next(mapping_it);
    bitstream_start = mapping_it->bitstream_start_idx;
    bitstream_end = next_it == data.triangles_to_split.end() ?
        int(data.bitstream.size()) :
        next_it->bitstream_start_idx;
    return bitstream_start >= 0 &&
           bitstream_start < bitstream_end &&
           size_t(bitstream_end) <= data.bitstream.size();
}

static int projection_region_read_nibble(const TriangleSelector::TriangleSplittingData &data,
                                         int                                            bitstream_end,
                                         int                                           &bit_idx)
{
    if (bit_idx + 3 >= bitstream_end || size_t(bit_idx + 3) >= data.bitstream.size())
        return -1;

    int code = 0;
    for (int bit = 0; bit < 4; ++bit)
        code |= int(data.bitstream[size_t(bit_idx++)]) << bit;
    return code;
}

static int projection_region_tree_max_depth(const TriangleSelector::TriangleSplittingData &data,
                                            int                                            bitstream_end,
                                            int                                           &bit_idx,
                                            int                                            depth)
{
    const int code = projection_region_read_nibble(data, bitstream_end, bit_idx);
    if (code < 0)
        return std::clamp(depth, 0, 7);

    const int split_sides = code & 0b11;
    if (split_sides == 0) {
        if ((code & 0b1100) == 0b1100) {
            int next_code = projection_region_read_nibble(data, bitstream_end, bit_idx);
            while (next_code == 0b1111)
                next_code = projection_region_read_nibble(data, bitstream_end, bit_idx);
        }
        return std::clamp(depth, 0, 7);
    }

    int max_depth = std::clamp(depth + 1, 0, 7);
    for (int child_idx = split_sides; child_idx >= 0; --child_idx)
        max_depth = std::max(max_depth, projection_region_tree_max_depth(data, bitstream_end, bit_idx, depth + 1));
    return std::clamp(max_depth, 0, 7);
}

static std::vector<int> projection_region_existing_source_triangle_depths(const TriangleSelector::TriangleSplittingData &data,
                                                                          size_t                                         triangle_count)
{
    std::vector<int> depths(triangle_count, 0);
    for (auto mapping_it = data.triangles_to_split.begin(); mapping_it != data.triangles_to_split.end(); ++mapping_it) {
        if (mapping_it->triangle_idx < 0 || size_t(mapping_it->triangle_idx) >= depths.size())
            continue;

        int bitstream_start = 0;
        int bitstream_end = 0;
        if (!projection_region_triangle_bitstream_range(data, size_t(mapping_it->triangle_idx), bitstream_start, bitstream_end))
            continue;

        int bit_idx = bitstream_start;
        depths[size_t(mapping_it->triangle_idx)] = projection_region_tree_max_depth(data, bitstream_end, bit_idx, 0);
    }
    return depths;
}

static void projection_region_append_nibble(std::vector<bool> &bitstream, unsigned int code)
{
    for (size_t bit_idx = 0; bit_idx < 4; ++bit_idx)
        bitstream.push_back((code & (1u << bit_idx)) != 0);
}

static void projection_region_append_leaf(TriangleSelector::TriangleSplittingData &data, unsigned int state)
{
    state = std::min<unsigned int>(state, static_cast<unsigned int>(EnforcerBlockerType::ExtruderMax));
    if (state <= 2) {
        projection_region_append_nibble(data.bitstream, state << 2);
        return;
    }

    projection_region_append_nibble(data.bitstream, 0b1100u);
    state -= 3;
    while (state >= 15) {
        projection_region_append_nibble(data.bitstream, 0b1111u);
        state -= 15;
    }
    projection_region_append_nibble(data.bitstream, state);
}

static std::array<std::array<Vec3f, 3>, 4> projection_region_split_triangle(const std::array<Vec3f, 3> &vertices)
{
    const Vec3f &a = vertices[0];
    const Vec3f &b = vertices[1];
    const Vec3f &c = vertices[2];
    const Vec3f ab = 0.5f * (a + b);
    const Vec3f bc = 0.5f * (b + c);
    const Vec3f ca = 0.5f * (c + a);
    return {
        std::array<Vec3f, 3>{ a, ab, ca },
        std::array<Vec3f, 3>{ ab, b, bc },
        std::array<Vec3f, 3>{ bc, c, ca },
        std::array<Vec3f, 3>{ ab, bc, ca }
    };
}

using ProjectionRegionStateSampler = std::function<unsigned int(size_t, const Vec3f &, const Vec3f &)>;

static bool projection_region_append_sampled_triangle(TriangleSelector::TriangleSplittingData &data,
                                                      const ProjectionRegionStateSampler      &sampler,
                                                      size_t                                   source_triangle,
                                                      const std::array<Vec3f, 3>              &vertices,
                                                      const std::array<Vec3f, 3>              &barycentrics,
                                                      int                                      depth,
                                                      int                                      target_depth)
{
    if (depth < target_depth) {
        projection_region_append_nibble(data.bitstream, 3u);
        const std::array<std::array<Vec3f, 3>, 4> child_vertices = projection_region_split_triangle(vertices);
        const std::array<std::array<Vec3f, 3>, 4> child_barycentrics = projection_region_split_triangle(barycentrics);
        bool has_painted_state = false;
        for (int child_idx = 3; child_idx >= 0; --child_idx)
            has_painted_state |= projection_region_append_sampled_triangle(data,
                                                                           sampler,
                                                                           source_triangle,
                                                                           child_vertices[size_t(child_idx)],
                                                                           child_barycentrics[size_t(child_idx)],
                                                                           depth + 1,
                                                                           target_depth);
        return has_painted_state;
    }

    const Vec3f centroid = (vertices[0] + vertices[1] + vertices[2]) / 3.f;
    const Vec3f centroid_bary = (barycentrics[0] + barycentrics[1] + barycentrics[2]) / 3.f;
    const unsigned int state = sampler(source_triangle, centroid, centroid_bary);
    projection_region_append_leaf(data, state);
    return state != 0;
}

static bool projection_region_append_preserved_triangle(const TriangleSelector::TriangleSplittingData &old_data,
                                                        TriangleSelector::TriangleSplittingData       &new_data,
                                                        size_t                                         tri_idx)
{
    int bitstream_start = 0;
    int bitstream_end = 0;
    if (!projection_region_triangle_bitstream_range(old_data, tri_idx, bitstream_start, bitstream_end))
        return false;

    new_data.triangles_to_split.emplace_back(int(tri_idx), int(new_data.bitstream.size()));
    new_data.bitstream.insert(new_data.bitstream.end(),
                              old_data.bitstream.begin() + bitstream_start,
                              old_data.bitstream.begin() + bitstream_end);
    return true;
}

static bool object_is_whole_image_texture_mapped_without_regions(const ModelObject &object)
{
    if (wxGetApp().preset_bundle == nullptr)
        return false;

    const TextureMappingManager &texture_mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    bool has_model_part = false;
    for (const ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        has_model_part = true;
        if (!volume->mmu_segmentation_facets.empty())
            return false;

        const int extruder_id = volume->extruder_id();
        if (extruder_id <= 0)
            return false;

        const TextureMappingZone *zone = texture_mgr.zone_from_id(unsigned(extruder_id));
        if (zone == nullptr || !zone->enabled || zone->deleted || !zone->is_image_texture())
            return false;
    }
    return has_model_part;
}

static int generic_solver_mix_model_for_projection_object(const ModelObject *object)
{
    if (wxGetApp().preset_bundle == nullptr || object == nullptr)
        return TextureMappingZone::DefaultGenericSolverMixModel;

    const TextureMappingManager &texture_mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const int extruder_id = volume->extruder_id();
        if (extruder_id <= 0)
            continue;

        const TextureMappingZone *zone = texture_mgr.zone_from_id(unsigned(extruder_id));
        if (zone != nullptr && zone->enabled && !zone->deleted && zone->is_image_texture())
            return TextureMappingZone::DefaultGenericSolverMixModel;
    }

    return TextureMappingZone::DefaultGenericSolverMixModel;
}

static bool project_texture_mapping_zone_to_regions(ModelObject             &object,
                                                    const GLCanvas3D       &parent,
                                                    const ProjectionContext &context,
                                                    int                      instance_idx,
                                                    bool                     pass_through_model,
                                                    unsigned int             texture_mapping_filament_id)
{
    if (texture_mapping_filament_id == 0)
        return false;

    const ProjectionVisibility visibility = pass_through_model ?
        ProjectionVisibility() :
        build_projection_visibility(context, parent, &object, instance_idx);
    const ProjectionPaintableImageMask paintable_mask = build_projection_paintable_image_mask(context);
    const float projection_target_span = image_projection_rgb_target_triangle_pixel_span(context);
    bool changed = false;

    const std::array<Vec3f, 3> root_barycentrics = {
        Vec3f(1.f, 0.f, 0.f),
        Vec3f(0.f, 1.f, 0.f),
        Vec3f(0.f, 0.f, 1.f)
    };

    for (size_t volume_idx = 0; volume_idx < object.volumes.size(); ++volume_idx) {
        ModelVolume *volume = object.volumes[volume_idx];
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const Transform3d world_matrix = projection_world_matrix_for_volume(parent, &object, volume, instance_idx);
        const Matrix3d world_normal_matrix = world_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();
        const std::vector<Vec3d> vertex_normals = projection_smoothed_vertex_normals(its);
        std::vector<bool> projected_triangles(its.indices.size(), false);
        std::vector<int> projected_triangle_depths(its.indices.size(), 0);
        size_t projected_triangle_count = 0;

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            if (!projection_triangle_should_project(context,
                                                    visibility,
                                                    paintable_mask,
                                                    world_matrix,
                                                    vertices,
                                                    projection_visibility_triangle_key(volume_idx, tri_idx)))
                continue;

            projected_triangles[tri_idx] = true;
            projected_triangle_depths[tri_idx] =
                texture_mapping_depth_from_span(projection_triangle_overlay_image_pixel_span(context, world_matrix, vertices),
                                                projection_target_span,
                                                7);
            ++projected_triangle_count;
        }

        if (projected_triangle_count == 0)
            continue;

        const TriangleSelector::TriangleSplittingData &old_data = volume->mmu_segmentation_facets.get_data();
        const std::vector<int> existing_depths = projection_region_existing_source_triangle_depths(old_data, its.indices.size());
        const std::vector<int> rgb_depths =
            rgb_existing_source_triangle_depths(volume->texture_mapping_color_facets.get_data(), its.indices.size());
        const ProjectionRegionStateSource state_source = build_projection_region_state_source(*volume);
        const int projected_safe_max_depth = texture_mapping_depth_for_budget(projected_triangle_count, 7, 2200000);

        TriangleSelector::TriangleSplittingData new_data;
        new_data.triangles_to_split.reserve(std::max(old_data.triangles_to_split.size(), projected_triangle_count));
        new_data.bitstream.reserve(old_data.bitstream.size() + projected_triangle_count * 4);

        ProjectionRegionStateSampler sampler =
            [context,
             world_matrix,
             world_normal_matrix,
             pass_through_model,
             texture_mapping_filament_id,
             volume,
             volume_idx,
             &visibility,
             &vertex_normals,
             &projected_triangles,
             &state_source](size_t tri_idx, const Vec3f &point, const Vec3f &barycentric) {
            unsigned int state = sample_projection_region_state(state_source, int(tri_idx), point);
            if (tri_idx < projected_triangles.size() && projected_triangles[tri_idx]) {
                const stl_triangle_vertex_indices &tri = volume->mesh().its.indices[tri_idx];
                if (pass_through_model ||
                    (projection_point_allowed_by_camera_facing(context,
                                                               world_matrix,
                                                               world_normal_matrix,
                                                               vertex_normals,
                                                               tri,
                                                               point,
                                                               barycentric) &&
                     projection_point_is_visible(visibility,
                                                 context,
                                                 world_matrix,
                                                 point,
                                                 projection_visibility_triangle_key(volume_idx, tri_idx)))) {
                    if (std::optional<ColorRGBA> projected = projected_image_color_at_point(context, world_matrix, point);
                        projected && projection_overlay_has_paintable_alpha(*projected, context)) {
                        state = texture_mapping_filament_id;
                    }
                }
            }
            return state;
        };

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            if (tri_idx >= projected_triangles.size() || !projected_triangles[tri_idx]) {
                projection_region_append_preserved_triangle(old_data, new_data, tri_idx);
                continue;
            }

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };

            int target_depth = 0;
            if (model_volume_has_bakeable_image_texture_data(volume))
                target_depth = texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(volume, tri_idx),
                                                               8.f,
                                                               projected_safe_max_depth);
            target_depth = std::max(target_depth, std::min(projected_triangle_depths[tri_idx], projected_safe_max_depth));
            if (tri_idx < existing_depths.size())
                target_depth = std::max(target_depth, existing_depths[tri_idx]);
            if (tri_idx < rgb_depths.size())
                target_depth = std::max(target_depth, rgb_depths[tri_idx]);
            target_depth = std::clamp(target_depth, 0, 7);

            const size_t bitstream_start = new_data.bitstream.size();
            new_data.triangles_to_split.emplace_back(int(tri_idx), int(bitstream_start));
            if (!projection_region_append_sampled_triangle(new_data,
                                                           sampler,
                                                           tri_idx,
                                                           vertices,
                                                           root_barycentrics,
                                                           0,
                                                           target_depth)) {
                new_data.triangles_to_split.pop_back();
                new_data.bitstream.resize(bitstream_start);
            }
        }

        new_data.triangles_to_split.shrink_to_fit();
        new_data.bitstream.shrink_to_fit();

        TriangleSelector selector(volume->mesh());
        selector.deserialize(new_data, false);
        changed |= volume->mmu_segmentation_facets.set(selector);
    }

    return changed;
}

enum class ManagedColorDataType
{
    ColorRegions,
    VertexColors,
    ImageTexture,
    RgbaData
};

enum class ManagedColorDataPreviewKind
{
    None,
    ColorRegions,
    VertexColors,
    ImageTexture,
    RawOffsetAtlas,
    RgbaData
};

struct ManagedColorDataCreateSource
{
    std::optional<ManagedColorDataType> type;
    bool                                erase_color_regions = false;
};

struct ManagedColorDataSummary
{
    bool     has_color_regions = false;
    bool     has_vertex_colors = false;
    bool     has_image_texture = false;
    bool     has_raw_offset_image_texture = false;
    bool     has_rgba_data = false;
    size_t   color_region_triangle_count = 0;
    size_t   vertex_color_count = 0;
    size_t   image_texture_count = 0;
    size_t   raw_offset_image_texture_count = 0;
    uint32_t max_texture_width = 0;
    uint32_t max_texture_height = 0;
    size_t   rgba_data_triangle_count = 0;
    std::vector<std::string> raw_offset_color_modes;
};

static bool object_has_color_regions(const ModelObject &object)
{
    for (const ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->mmu_segmentation_facets.empty())
            return true;
    return false;
}

static bool object_has_vertex_color_data(const ModelObject &object)
{
    for (const ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->imported_vertex_colors_rgba.empty())
            return true;
    return false;
}

static bool object_has_image_texture_data(const ModelObject &object)
{
    for (const ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part() && model_volume_has_imported_image_texture_data(volume))
            return true;
    return false;
}

static bool object_has_rgba_data(const ModelObject &object)
{
    for (const ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->texture_mapping_color_facets.empty())
            return true;
    return false;
}

static bool model_volume_has_non_region_color_data(const ModelVolume *volume)
{
    return volume != nullptr &&
           volume->is_model_part() &&
           (!volume->imported_vertex_colors_rgba.empty() ||
            model_volume_has_imported_image_texture_data(volume) ||
            !volume->texture_mapping_color_facets.empty());
}

static bool object_has_non_region_color_data(const ModelObject &object)
{
    for (const ModelVolume *volume : object.volumes)
        if (model_volume_has_non_region_color_data(volume))
            return true;
    return false;
}

static bool object_has_managed_color_data(const ModelObject &object, ManagedColorDataType type)
{
    switch (type) {
    case ManagedColorDataType::ColorRegions:
        return object_has_color_regions(object);
    case ManagedColorDataType::VertexColors:
        return object_has_vertex_color_data(object);
    case ManagedColorDataType::ImageTexture:
        return object_has_image_texture_data(object);
    case ManagedColorDataType::RgbaData:
        return object_has_rgba_data(object);
    }
    return false;
}

static bool managed_color_data_summary_has_type(const ManagedColorDataSummary &summary, ManagedColorDataType type)
{
    switch (type) {
    case ManagedColorDataType::ColorRegions:
        return summary.has_color_regions;
    case ManagedColorDataType::VertexColors:
        return summary.has_vertex_colors;
    case ManagedColorDataType::ImageTexture:
        return summary.has_image_texture;
    case ManagedColorDataType::RgbaData:
        return summary.has_rgba_data;
    }
    return false;
}

static wxString managed_color_data_type_label(ManagedColorDataType type)
{
    switch (type) {
    case ManagedColorDataType::ColorRegions:
        return _L("3mf color regions");
    case ManagedColorDataType::VertexColors:
        return _L("Vertex Colors");
    case ManagedColorDataType::ImageTexture:
        return _L("Image Texture");
    case ManagedColorDataType::RgbaData:
        return _L("RGBA data");
    }
    return wxString();
}

static ManagedColorDataSummary summarize_managed_color_data(const ModelObject *object)
{
    ManagedColorDataSummary summary;
    if (object == nullptr)
        return summary;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        if (!volume->mmu_segmentation_facets.empty()) {
            summary.has_color_regions = true;
            summary.color_region_triangle_count += volume->mmu_segmentation_facets.get_data().triangles_to_split.size();
        }

        if (!volume->imported_vertex_colors_rgba.empty()) {
            summary.has_vertex_colors = true;
            summary.vertex_color_count += volume->imported_vertex_colors_rgba.size();
        }

        if (model_volume_has_imported_image_texture_data(volume)) {
            summary.has_image_texture = true;
            ++summary.image_texture_count;
            summary.max_texture_width = std::max(summary.max_texture_width, volume->imported_texture_width);
            summary.max_texture_height = std::max(summary.max_texture_height, volume->imported_texture_height);
        }

        if (model_volume_has_raw_atlas_texture_data(volume)) {
            summary.has_raw_offset_image_texture = true;
            ++summary.raw_offset_image_texture_count;
            const std::string mode = raw_atlas_color_mode_name_for_volume(*volume);
            if (std::find(summary.raw_offset_color_modes.begin(), summary.raw_offset_color_modes.end(), mode) ==
                summary.raw_offset_color_modes.end())
                summary.raw_offset_color_modes.emplace_back(mode);
        }

        if (!volume->texture_mapping_color_facets.empty()) {
            summary.has_rgba_data = true;
            summary.rgba_data_triangle_count += volume->texture_mapping_color_facets.get_data().colors_rgba.size();
        }
    }
    return summary;
}

static wxString managed_color_data_size_text(const ManagedColorDataSummary &summary, ManagedColorDataType type)
{
    switch (type) {
    case ManagedColorDataType::ColorRegions:
        return wxString::Format(_L("%llu triangles"), static_cast<unsigned long long>(summary.color_region_triangle_count));
    case ManagedColorDataType::VertexColors:
        return wxString::Format(_L("%llu vertices"), static_cast<unsigned long long>(summary.vertex_color_count));
    case ManagedColorDataType::ImageTexture:
        if (!summary.has_image_texture)
            return _L("0 x 0");
        if (summary.image_texture_count <= 1)
            return wxString::Format(_L("%u x %u"),
                                    static_cast<unsigned>(summary.max_texture_width),
                                    static_cast<unsigned>(summary.max_texture_height));
        return wxString::Format(_L("%llu textures, max %u x %u"),
                                static_cast<unsigned long long>(summary.image_texture_count),
                                static_cast<unsigned>(summary.max_texture_width),
                                static_cast<unsigned>(summary.max_texture_height));
    case ManagedColorDataType::RgbaData:
        return wxString::Format(_L("%llu triangles"), static_cast<unsigned long long>(summary.rgba_data_triangle_count));
    }
    return wxString();
}

static wxString managed_color_data_raw_offset_mode_text(const ManagedColorDataSummary &summary)
{
    if (!summary.has_raw_offset_image_texture)
        return wxString();

    wxString text;
    for (size_t idx = 0; idx < summary.raw_offset_color_modes.size(); ++idx) {
        if (idx > 0)
            text += ", ";
        text += from_u8(summary.raw_offset_color_modes[idx]);
    }
    if (text.empty())
        text = _L("Unknown");
    if (summary.raw_offset_image_texture_count > 1)
        text += wxString::Format(_L(" (%llu textures)"),
                                 static_cast<unsigned long long>(summary.raw_offset_image_texture_count));
    return text;
}

static ManagedColorDataPreviewKind managed_color_data_preview_kind(const ManagedColorDataSummary &summary)
{
    if (summary.has_rgba_data)
        return ManagedColorDataPreviewKind::RgbaData;
    if (summary.has_raw_offset_image_texture)
        return ManagedColorDataPreviewKind::RawOffsetAtlas;
    if (summary.has_image_texture)
        return ManagedColorDataPreviewKind::ImageTexture;
    if (summary.has_vertex_colors)
        return ManagedColorDataPreviewKind::VertexColors;
    if (summary.has_color_regions)
        return ManagedColorDataPreviewKind::ColorRegions;
    return ManagedColorDataPreviewKind::None;
}

static bool managed_color_data_preview_kind_matches_type(ManagedColorDataPreviewKind kind, ManagedColorDataType type)
{
    switch (type) {
    case ManagedColorDataType::ColorRegions:
        return kind == ManagedColorDataPreviewKind::ColorRegions;
    case ManagedColorDataType::VertexColors:
        return kind == ManagedColorDataPreviewKind::VertexColors;
    case ManagedColorDataType::ImageTexture:
        return kind == ManagedColorDataPreviewKind::ImageTexture;
    case ManagedColorDataType::RgbaData:
        return kind == ManagedColorDataPreviewKind::RgbaData;
    }
    return false;
}

static wxString managed_color_data_preview_marker(bool shown)
{
    return shown ? _L("*") : wxString();
}

static bool clear_object_managed_color_data(ModelObject &object, ManagedColorDataType type)
{
    bool changed = false;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        switch (type) {
        case ManagedColorDataType::ColorRegions:
            if (!volume->mmu_segmentation_facets.empty()) {
                volume->mmu_segmentation_facets.reset();
                changed = true;
            }
            break;
        case ManagedColorDataType::VertexColors:
            if (!volume->imported_vertex_colors_rgba.empty()) {
                volume->imported_vertex_colors_rgba.clear();
                changed = true;
            }
            break;
        case ManagedColorDataType::ImageTexture:
            if (!volume->imported_texture_rgba.empty() ||
                !volume->imported_texture_uvs_per_face.empty() ||
                !volume->imported_texture_uv_valid.empty() ||
                volume->imported_texture_width != 0 ||
                volume->imported_texture_height != 0) {
                volume->imported_texture_uvs_per_face.clear();
                volume->imported_texture_uv_valid.clear();
                volume->imported_texture_rgba.clear();
                clear_imported_texture_raw_atlas(*volume);
                volume->imported_texture_width = 0;
                volume->imported_texture_height = 0;
                volume->uv_map_generator_version = 0;
                changed = true;
            }
            break;
        case ManagedColorDataType::RgbaData:
            if (!volume->texture_mapping_color_facets.empty()) {
                volume->texture_mapping_color_facets.reset();
                changed = true;
            }
            break;
        }
    }
    return changed;
}

static void assign_texture_mapping_zone_preserving_painted_regions(ModelObject &object, unsigned int texture_mapping_filament_id)
{
    if (texture_mapping_filament_id == 0)
        return;

    bool has_painted_regions = false;
    for (const ModelVolume *volume : object.volumes) {
        if (volume != nullptr && volume->is_model_part() && !volume->mmu_segmentation_facets.empty()) {
            has_painted_regions = true;
            break;
        }
    }

    if (!has_painted_regions)
        object.config.set("extruder", int(texture_mapping_filament_id));

    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!has_painted_regions || volume->mmu_segmentation_facets.empty())
            volume->config.set("extruder", int(texture_mapping_filament_id));
    }
}

static bool image_texture_zone_is_usable(const TextureMappingManager &texture_mgr, unsigned int zone_id)
{
    const TextureMappingZone *zone = texture_mgr.zone_from_id(zone_id);
    return zone != nullptr && zone->enabled && !zone->deleted && zone->is_image_texture();
}

static bool non_region_color_data_is_assigned_to_image_texture_zone(const ModelObject &object)
{
    if (wxGetApp().preset_bundle == nullptr)
        return false;

    const TextureMappingManager &texture_mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    bool has_color_data = false;
    for (const ModelVolume *volume : object.volumes) {
        if (!model_volume_has_non_region_color_data(volume))
            continue;

        has_color_data = true;
        const int extruder_id = volume->extruder_id();
        if (extruder_id <= 0 || !image_texture_zone_is_usable(texture_mgr, unsigned(extruder_id)))
            return false;
    }
    return has_color_data;
}

static bool assign_non_region_color_data_to_image_texture_zone(ModelObject &object)
{
    if (!object_has_non_region_color_data(object) ||
        non_region_color_data_is_assigned_to_image_texture_zone(object))
        return false;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id == 0)
        return false;

    object.config.set("extruder", int(texture_mapping_filament_id));
    for (ModelVolume *volume : object.volumes)
        if (model_volume_has_non_region_color_data(volume))
            volume->config.set("extruder", int(texture_mapping_filament_id));
    return true;
}

static bool erase_color_regions_and_assign_texture_mapping_zone(ModelObject &object)
{
    bool changed = false;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!volume->mmu_segmentation_facets.empty()) {
            volume->mmu_segmentation_facets.reset();
            changed = true;
        }
    }

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    if (texture_mapping_filament_id != 0) {
        object.config.set("extruder", int(texture_mapping_filament_id));
        for (ModelVolume *volume : object.volumes)
            if (volume != nullptr && volume->is_model_part())
                volume->config.set("extruder", int(texture_mapping_filament_id));
        changed = true;
    }
    return changed;
}

static unsigned int painted_image_texture_zone_id_for_projection(const ModelObject &object)
{
    if (wxGetApp().preset_bundle == nullptr)
        return 0;

    const TextureMappingManager &texture_mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    std::vector<unsigned int> painted_ids;
    for (const ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part() || volume->mmu_segmentation_facets.empty())
            continue;

        const auto &used_states = volume->mmu_segmentation_facets.get_data().used_states;
        for (size_t state_idx = static_cast<size_t>(EnforcerBlockerType::Extruder1); state_idx < used_states.size(); ++state_idx) {
            if (used_states[state_idx] && image_texture_zone_is_usable(texture_mgr, unsigned(state_idx)))
                painted_ids.emplace_back(unsigned(state_idx));
        }
    }

    for (const TextureMappingZone &zone : texture_mgr.zones())
        if (image_texture_zone_is_usable(texture_mgr, zone.zone_id) &&
            std::find(painted_ids.begin(), painted_ids.end(), zone.zone_id) != painted_ids.end())
            return zone.zone_id;
    return 0;
}

static unsigned int base_image_texture_zone_id_for_projection(const ModelObject &object)
{
    if (wxGetApp().preset_bundle == nullptr)
        return 0;

    const TextureMappingManager &texture_mgr = wxGetApp().preset_bundle->texture_mapping_zones;
    std::vector<unsigned int> base_ids;
    if (const ConfigOption *opt = object.config.option("extruder"); opt != nullptr) {
        const int extruder_id = opt->getInt();
        if (extruder_id > 0 && image_texture_zone_is_usable(texture_mgr, unsigned(extruder_id)))
            base_ids.emplace_back(unsigned(extruder_id));
    }

    for (const ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        const int extruder_id = volume->extruder_id();
        if (extruder_id > 0 && image_texture_zone_is_usable(texture_mgr, unsigned(extruder_id)))
            base_ids.emplace_back(unsigned(extruder_id));
    }

    for (const TextureMappingZone &zone : texture_mgr.zones())
        if (image_texture_zone_is_usable(texture_mgr, zone.zone_id) &&
            std::find(base_ids.begin(), base_ids.end(), zone.zone_id) != base_ids.end())
            return zone.zone_id;
    return 0;
}

static unsigned int texture_mapping_zone_id_for_image_projection(ModelObject &object)
{
    if (const unsigned int painted_id = painted_image_texture_zone_id_for_projection(object); painted_id != 0)
        return painted_id;
    if (const unsigned int base_id = base_image_texture_zone_id_for_projection(object); base_id != 0)
        return base_id;
    return ensure_texture_mapping_zone();
}

struct ManagedColorSourceFlags
{
    bool use_rgba = false;
    bool use_image_texture = false;
    bool use_vertex_colors = false;
    bool use_color_regions = false;
};

static ManagedColorSourceFlags managed_color_source_flags(const ManagedColorDataCreateSource &source)
{
    ManagedColorSourceFlags flags;
    if (!source.type)
        return flags;

    switch (*source.type) {
    case ManagedColorDataType::ColorRegions:
        flags.use_color_regions = true;
        break;
    case ManagedColorDataType::VertexColors:
        flags.use_vertex_colors = true;
        break;
    case ManagedColorDataType::ImageTexture:
        flags.use_image_texture = true;
        break;
    case ManagedColorDataType::RgbaData:
        flags.use_rgba = true;
        break;
    }
    return flags;
}

static bool managed_color_source_flags_empty(const ManagedColorSourceFlags &flags)
{
    return !flags.use_rgba && !flags.use_image_texture && !flags.use_vertex_colors && !flags.use_color_regions;
}

static bool managed_color_source_has_volume_data(const ModelVolume &volume, ManagedColorDataType type)
{
    switch (type) {
    case ManagedColorDataType::ColorRegions:
        return !volume.mmu_segmentation_facets.empty();
    case ManagedColorDataType::VertexColors:
        return volume.imported_vertex_colors_rgba.size() == volume.mesh().its.vertices.size();
    case ManagedColorDataType::ImageTexture:
        return model_volume_has_bakeable_image_texture_data(&volume);
    case ManagedColorDataType::RgbaData:
        return !volume.texture_mapping_color_facets.empty();
    }
    return false;
}

static ManagedColorSourceFlags managed_color_source_flags_for_type(ManagedColorDataType type)
{
    return managed_color_source_flags(ManagedColorDataCreateSource{ std::optional<ManagedColorDataType>(type) });
}

static ManagedColorSourceFlags managed_target_or_highest_existing_color_source_flags(const ModelVolume &volume, ManagedColorDataType target)
{
    if (managed_color_source_has_volume_data(volume, target))
        return managed_color_source_flags_for_type(target);
    if (managed_color_source_has_volume_data(volume, ManagedColorDataType::RgbaData))
        return managed_color_source_flags_for_type(ManagedColorDataType::RgbaData);
    if (managed_color_source_has_volume_data(volume, ManagedColorDataType::ImageTexture))
        return managed_color_source_flags_for_type(ManagedColorDataType::ImageTexture);
    if (managed_color_source_has_volume_data(volume, ManagedColorDataType::VertexColors))
        return managed_color_source_flags_for_type(ManagedColorDataType::VertexColors);
    return ManagedColorSourceFlags();
}

static ColorRGBA blank_color_for_managed_target(ManagedColorDataType target)
{
    return target == ManagedColorDataType::RgbaData ? rgba_data_conversion_fallback_color() :
                                                      ColorRGBA(1.f, 1.f, 1.f, 1.f);
}

static std::vector<ColorRGBA> parse_managed_color_strings(const std::vector<std::string> &color_strings)
{
    std::vector<ColorRGBA> colors;
    colors.reserve(color_strings.size());
    for (const std::string &color_string : color_strings) {
        unsigned char rgba[4] = { 38, 166, 154, 255 };
        BitmapCache::parse_color4(color_string, rgba);
        colors.emplace_back(float(rgba[0]) / 255.f,
                            float(rgba[1]) / 255.f,
                            float(rgba[2]) / 255.f,
                            float(rgba[3]) / 255.f);
    }
    return colors;
}

static ColorRGBA managed_filament_color(unsigned int                      filament_id,
                                        unsigned int                      base_filament_id,
                                        const std::vector<ColorRGBA>     &physical_colors,
                                        const std::vector<ColorRGBA>     &display_colors)
{
    const size_t physical_count = physical_colors.size();
    const ColorRGBA fallback = physical_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : physical_colors.front();

    auto physical_or_fallback = [&physical_colors, fallback](unsigned int id) {
        if (id >= 1 && id <= physical_colors.size())
            return physical_colors[size_t(id - 1)];
        return fallback;
    };

    const bool texture_mapping_zone =
        wxGetApp().preset_bundle != nullptr &&
        wxGetApp().preset_bundle->texture_mapping_zones.is_texture_mapping_zone_id(filament_id);
    if (texture_mapping_zone) {
        if (base_filament_id != 0 && base_filament_id != filament_id)
            return managed_filament_color(base_filament_id, 0, physical_colors, display_colors);
        return fallback;
    }

    if (filament_id >= 1 && filament_id <= physical_count)
        return physical_or_fallback(filament_id);

    if (filament_id >= 1 && filament_id <= display_colors.size())
        return display_colors[size_t(filament_id - 1)];

    return fallback;
}

static bool managed_filament_is_texture_mapping_zone(unsigned int filament_id)
{
    if (wxGetApp().preset_bundle == nullptr)
        return false;

    const TextureMappingZone *zone = wxGetApp().preset_bundle->texture_mapping_zones.zone_from_id(filament_id);
    return zone != nullptr && zone->enabled && !zone->deleted;
}

static std::vector<ColorRGBA> managed_region_color_source_state_colors(const ModelVolume &volume, size_t state_count)
{
    const std::vector<std::string> physical_color_strings =
        wxGetApp().plater() != nullptr ? wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, false) :
        std::vector<std::string>();
    const std::vector<std::string> display_color_strings =
        wxGetApp().plater() != nullptr ? wxGetApp().plater()->get_extruder_colors_from_plater_config() :
        physical_color_strings;
    const std::vector<ColorRGBA> physical_colors = parse_managed_color_strings(physical_color_strings);
    const std::vector<ColorRGBA> display_colors = parse_managed_color_strings(display_color_strings);
    const unsigned int base_filament_id = volume.extruder_id() > 0 ? unsigned(volume.extruder_id()) : 1u;

    std::vector<ColorRGBA> state_colors;
    state_colors.reserve(state_count);
    for (size_t state_idx = 0; state_idx < state_count; ++state_idx) {
        const unsigned int filament_id = state_idx == 0 ? base_filament_id : unsigned(state_idx);
        ColorRGBA color = managed_filament_color(filament_id, base_filament_id, physical_colors, display_colors);
        color.a(1.f);
        state_colors.emplace_back(color);
    }
    return state_colors;
}

static std::vector<bool> managed_region_color_source_texture_zone_states(const ModelVolume &volume, size_t state_count)
{
    const unsigned int base_filament_id = volume.extruder_id() > 0 ? unsigned(volume.extruder_id()) : 1u;

    std::vector<bool> states;
    states.reserve(state_count);
    for (size_t state_idx = 0; state_idx < state_count; ++state_idx) {
        const unsigned int filament_id = state_idx == 0 ? base_filament_id : unsigned(state_idx);
        states.emplace_back(managed_filament_is_texture_mapping_zone(filament_id));
    }
    return states;
}

static ManagedRegionColorSource build_managed_region_color_state_source(const ModelVolume &volume, size_t state_count)
{
    ManagedRegionColorSource source;
    source.state_colors = managed_region_color_source_state_colors(volume, state_count);
    source.state_is_texture_mapping_zone = managed_region_color_source_texture_zone_states(volume, state_count);
    return source;
}

static ManagedRegionColorSource build_managed_region_color_source(const ModelVolume &volume)
{
    ManagedRegionColorSource source;
    if (volume.mmu_segmentation_facets.empty())
        return source;

    volume.mmu_segmentation_facets.get_facet_triangles(volume, source.triangles_per_type);
    source.by_source_triangle.resize(source.triangles_per_type.size());

    const indexed_triangle_set &its = volume.mesh().its;
    const float mesh_span = mesh_max_axis_span(its);
    const float nearby_distance = std::max(mesh_span / 5000.f, 0.002f);
    source.nearby_painted_distance_sq = nearby_distance * nearby_distance;

    source.state_colors = managed_region_color_source_state_colors(volume, source.triangles_per_type.size());
    source.state_is_texture_mapping_zone = managed_region_color_source_texture_zone_states(volume, source.triangles_per_type.size());
    for (size_t state_idx = 0; state_idx < source.triangles_per_type.size(); ++state_idx) {
        std::unordered_map<int, std::vector<size_t>> &by_source = source.by_source_triangle[state_idx];
        by_source.reserve(source.triangles_per_type[state_idx].size());
        for (size_t idx = 0; idx < source.triangles_per_type[state_idx].size(); ++idx)
            by_source[source.triangles_per_type[state_idx][idx].source_triangle].emplace_back(idx);
    }

    std::vector<std::vector<int>> triangles_by_vertex(its.vertices.size());
    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        for (int corner = 0; corner < 3; ++corner)
            if (tri[corner] >= 0 && size_t(tri[corner]) < triangles_by_vertex.size())
                triangles_by_vertex[size_t(tri[corner])].emplace_back(int(tri_idx));
    }

    source.nearby_source_triangles.resize(its.indices.size());
    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        std::vector<int> &nearby = source.nearby_source_triangles[tri_idx];
        nearby.emplace_back(int(tri_idx));
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        for (int corner = 0; corner < 3; ++corner) {
            if (tri[corner] < 0 || size_t(tri[corner]) >= triangles_by_vertex.size())
                continue;
            nearby.insert(nearby.end(), triangles_by_vertex[size_t(tri[corner])].begin(), triangles_by_vertex[size_t(tri[corner])].end());
        }
        std::sort(nearby.begin(), nearby.end());
        nearby.erase(std::unique(nearby.begin(), nearby.end()), nearby.end());
    }

    return source;
}

static ManagedRegionColorSource build_managed_region_color_overlay_source(const ModelVolume &volume)
{
    ManagedRegionColorSource source = build_managed_region_color_source(volume);
    if (source.state_colors.empty())
        source = build_managed_region_color_state_source(volume, 1);
    return source;
}

struct ManagedRegionColorCandidate
{
    std::optional<ColorRGBA> inside_color;
    size_t                   inside_state = 0;
    float                    inside_score = -std::numeric_limits<float>::max();
    std::optional<ColorRGBA> nearest_color;
    size_t                   nearest_state = 0;
    float                    nearest_distance_sq = std::numeric_limits<float>::max();
    std::optional<ColorRGBA> nearest_painted_color;
    size_t                   nearest_painted_state = 0;
    float                    nearest_painted_distance_sq = std::numeric_limits<float>::max();
};

static void consider_managed_region_source_triangle(const ManagedRegionColorSource &source,
                                                    int                             source_triangle,
                                                    const Vec3f                    &point,
                                                    ManagedRegionColorCandidate    &candidate)
{
    for (size_t state_idx = 0; state_idx < source.triangles_per_type.size(); ++state_idx) {
        if (state_idx >= source.by_source_triangle.size() || state_idx >= source.state_colors.size())
            continue;

        const auto found = source.by_source_triangle[state_idx].find(source_triangle);
        if (found == source.by_source_triangle[state_idx].end())
            continue;

        const float tolerance = -1e-4f;
        const std::vector<TriangleSelector::FacetStateTriangle> &state_triangles = source.triangles_per_type[state_idx];
        for (const size_t facet_idx : found->second) {
            if (facet_idx >= state_triangles.size())
                continue;

            const TriangleSelector::FacetStateTriangle &facet = state_triangles[facet_idx];
            Vec3f weights = Vec3f::Zero();
            if (!barycentric_weights_for_region_vertex_colors(point, facet.vertices[0], facet.vertices[1], facet.vertices[2], weights))
                continue;
            if (weights.x() >= tolerance && weights.y() >= tolerance && weights.z() >= tolerance) {
                const float score = std::min({ weights.x(), weights.y(), weights.z() });
                if (!candidate.inside_color ||
                    score > candidate.inside_score + 1e-6f ||
                    (std::abs(score - candidate.inside_score) <= 1e-6f && state_idx > candidate.inside_state)) {
                    candidate.inside_color = source.state_colors[state_idx];
                    candidate.inside_score = score;
                    candidate.inside_state = state_idx;
                }
            }

            const Vec3f closest = closest_point_on_triangle(point, facet.vertices[0], facet.vertices[1], facet.vertices[2]);
            const float distance_sq = (point - closest).squaredNorm();
            if (!candidate.nearest_color ||
                distance_sq < candidate.nearest_distance_sq - 1e-8f ||
                (std::abs(distance_sq - candidate.nearest_distance_sq) <= 1e-8f && state_idx > candidate.nearest_state)) {
                candidate.nearest_color = source.state_colors[state_idx];
                candidate.nearest_distance_sq = distance_sq;
                candidate.nearest_state = state_idx;
            }
            if (state_idx > 0 &&
                (!candidate.nearest_painted_color ||
                 distance_sq < candidate.nearest_painted_distance_sq - 1e-8f ||
                 (std::abs(distance_sq - candidate.nearest_painted_distance_sq) <= 1e-8f && state_idx > candidate.nearest_painted_state))) {
                candidate.nearest_painted_color = source.state_colors[state_idx];
                candidate.nearest_painted_distance_sq = distance_sq;
                candidate.nearest_painted_state = state_idx;
            }
        }
    }
}

static std::optional<ColorRGBA> sample_managed_region_color_source(const ManagedRegionColorSource &source,
                                                                   int                             source_triangle,
                                                                   const Vec3f                    &point)
{
    ManagedRegionColorCandidate same_triangle;
    consider_managed_region_source_triangle(source, source_triangle, point, same_triangle);

    if (same_triangle.inside_color && same_triangle.inside_state > 0)
        return same_triangle.inside_color;

    ManagedRegionColorCandidate nearby = same_triangle;
    if (source_triangle >= 0 && size_t(source_triangle) < source.nearby_source_triangles.size()) {
        for (const int nearby_triangle : source.nearby_source_triangles[size_t(source_triangle)]) {
            if (nearby_triangle == source_triangle)
                continue;
            consider_managed_region_source_triangle(source, nearby_triangle, point, nearby);
        }
    }

    if (nearby.nearest_painted_color && nearby.nearest_painted_distance_sq <= source.nearby_painted_distance_sq)
        return nearby.nearest_painted_color;

    if (same_triangle.inside_color)
        return same_triangle.inside_color;
    if (same_triangle.nearest_color)
        return same_triangle.nearest_color;
    if (nearby.nearest_color)
        return nearby.nearest_color;
    return std::nullopt;
}

static bool managed_region_state_is_texture_mapping_zone(const ManagedRegionColorSource &source, unsigned int state)
{
    return size_t(state) < source.state_is_texture_mapping_zone.size() &&
           source.state_is_texture_mapping_zone[size_t(state)];
}

static std::optional<unsigned int> sample_managed_region_state_source(const ManagedRegionColorSource &source,
                                                                      int                             source_triangle,
                                                                      const Vec3f                    &point)
{
    ManagedRegionColorCandidate same_triangle;
    consider_managed_region_source_triangle(source, source_triangle, point, same_triangle);

    if (same_triangle.inside_color && same_triangle.inside_state > 0)
        return unsigned(same_triangle.inside_state);

    ManagedRegionColorCandidate nearby = same_triangle;
    if (source_triangle >= 0 && size_t(source_triangle) < source.nearby_source_triangles.size()) {
        for (const int nearby_triangle : source.nearby_source_triangles[size_t(source_triangle)]) {
            if (nearby_triangle == source_triangle)
                continue;
            consider_managed_region_source_triangle(source, nearby_triangle, point, nearby);
        }
    }

    if (nearby.nearest_painted_color && nearby.nearest_painted_distance_sq <= source.nearby_painted_distance_sq)
        return unsigned(nearby.nearest_painted_state);
    if (same_triangle.inside_color)
        return unsigned(same_triangle.inside_state);
    if (same_triangle.nearest_color)
        return unsigned(same_triangle.nearest_state);
    if (nearby.nearest_color)
        return unsigned(nearby.nearest_state);
    return std::nullopt;
}

static std::optional<ColorRGBA> sample_managed_region_overlay_color_source(const ManagedRegionColorSource &source,
                                                                          int                             source_triangle,
                                                                          const Vec3f                    &point)
{
    const std::optional<unsigned int> state = sample_managed_region_state_source(source, source_triangle, point);
    if (state && managed_region_state_is_texture_mapping_zone(source, *state))
        return std::nullopt;
    if (state)
        return managed_color_data_state_color(source, *state);
    if (managed_region_state_is_texture_mapping_zone(source, 0))
        return std::nullopt;
    return managed_color_data_state_color(source, 0);
}

static std::optional<ColorRGBA> sample_managed_region_rgba_overlay_color_source(const ManagedRegionColorSource &source,
                                                                               int                             source_triangle,
                                                                               const Vec3f                    &point,
                                                                               const ColorRGBA                &unpainted_color)
{
    const std::optional<unsigned int> state = sample_managed_region_state_source(source, source_triangle, point);
    if (state && managed_region_state_is_texture_mapping_zone(source, *state))
        return std::nullopt;
    if (state && *state == 0)
        return unpainted_color;
    if (state)
        return managed_color_data_state_color(source, *state);
    if (managed_region_state_is_texture_mapping_zone(source, 0))
        return std::nullopt;
    return unpainted_color;
}

static std::vector<bool> managed_region_overlay_source_triangles(const ManagedRegionColorSource &source, size_t triangle_count)
{
    std::vector<bool> triangles(triangle_count, false);
    if (source.triangles_per_type.empty()) {
        if (!managed_region_state_is_texture_mapping_zone(source, 0))
            std::fill(triangles.begin(), triangles.end(), true);
        return triangles;
    }

    for (size_t state_idx = 0; state_idx < source.triangles_per_type.size(); ++state_idx) {
        if (managed_region_state_is_texture_mapping_zone(source, unsigned(state_idx)))
            continue;

        for (const TriangleSelector::FacetStateTriangle &triangle : source.triangles_per_type[state_idx]) {
            if (triangle.source_triangle < 0 || size_t(triangle.source_triangle) >= triangle_count)
                continue;
            triangles[size_t(triangle.source_triangle)] = true;
            if (size_t(triangle.source_triangle) < source.nearby_source_triangles.size())
                for (const int nearby_triangle : source.nearby_source_triangles[size_t(triangle.source_triangle)])
                    if (nearby_triangle >= 0 && size_t(nearby_triangle) < triangle_count)
                        triangles[size_t(nearby_triangle)] = true;
        }
    }
    return triangles;
}

static bool managed_region_source_uses_texture_mapping_zone(const ManagedRegionColorSource &source)
{
    if (managed_region_state_is_texture_mapping_zone(source, 0))
        return true;

    for (size_t state_idx = 1; state_idx < source.triangles_per_type.size(); ++state_idx)
        if (managed_region_state_is_texture_mapping_zone(source, unsigned(state_idx)) &&
            !source.triangles_per_type[state_idx].empty())
            return true;

    return false;
}

static char managed_color_data_hex_digit(unsigned int value)
{
    value &= 0x0Fu;
    return value < 10u ? char('0' + value) : char('A' + value - 10u);
}

static void managed_color_data_append_nibble(std::vector<bool> &bits, unsigned int code)
{
    for (int bit = 0; bit < 4; ++bit)
        bits.emplace_back((code & (1u << bit)) != 0u);
}

static int managed_color_data_read_nibble(const TriangleSelector::TriangleSplittingData &data, int bitstream_end, int &bit_idx)
{
    if (bit_idx < 0 || bit_idx + 4 > bitstream_end || size_t(bit_idx + 3) >= data.bitstream.size())
        return -1;

    int code = 0;
    for (int bit = 0; bit < 4; ++bit)
        code |= int(data.bitstream[size_t(bit_idx++)]) << bit;
    return code;
}

static void managed_color_data_bits_to_hex(const std::vector<bool> &bits, std::string &out)
{
    int offset = 0;
    const int end = int(bits.size());
    while (offset < end) {
        int next_code = 0;
        for (int bit = 3; bit >= 0; --bit) {
            next_code <<= 1;
            next_code |= int(bits[size_t(offset + bit)]);
        }
        offset += 4;
        out.insert(out.begin(), managed_color_data_hex_digit(unsigned(next_code)));
    }
}

static void managed_color_data_append_rgba_hex(std::string &out, uint32_t rgba)
{
    for (int shift = 28; shift >= 0; shift -= 4)
        out.push_back(managed_color_data_hex_digit((rgba >> shift) & 0x0Fu));
}

static int managed_color_data_hex_value(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    return -1;
}

static bool managed_color_data_parse_rgba_hex(const std::string &text, size_t offset, uint32_t &rgba)
{
    if (offset + 8 > text.size())
        return false;

    rgba = 0;
    for (size_t idx = 0; idx < 8; ++idx) {
        const int value = managed_color_data_hex_value(text[offset + idx]);
        if (value < 0)
            return false;
        rgba = (rgba << 4) | uint32_t(value);
    }
    return true;
}

static bool managed_color_data_replace_rgba(ColorFacetsAnnotation &annotation, uint32_t source_rgba, uint32_t target_rgba)
{
    if (annotation.empty() || source_rgba == target_rgba)
        return false;

    const TriangleColorSplittingData &data = annotation.get_data();
    if (std::find(data.colors_rgba.begin(), data.colors_rgba.end(), source_rgba) == data.colors_rgba.end())
        return false;

    std::unique_ptr<ColorFacetsAnnotation> rewritten = ColorFacetsAnnotation::make_temporary();
    if (!rewritten)
        return false;

    std::string target_hex;
    managed_color_data_append_rgba_hex(target_hex, target_rgba);
    bool changed = false;
    for (const ColorTriangleBitStreamMapping &mapping : data.triangles_to_split) {
        std::string encoded = annotation.get_triangle_as_string(mapping.triangle_idx);
        const size_t separator = encoded.find('|');
        if (separator != std::string::npos) {
            for (size_t offset = separator + 1; offset + 8 <= encoded.size(); offset += 8) {
                uint32_t rgba = 0;
                if (managed_color_data_parse_rgba_hex(encoded, offset, rgba) && rgba == source_rgba) {
                    encoded.replace(offset, 8, target_hex);
                    changed = true;
                }
            }
        }
        if (!encoded.empty())
            rewritten->set_triangle_from_string(mapping.triangle_idx, encoded);
    }

    if (!changed)
        return false;

    rewritten->set_metadata_json(annotation.metadata_json());
    annotation.assign(*rewritten);
    return true;
}

struct ManagedColorDataRgbaAccumulator
{
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    double a = 0.0;
    double area_weight = 0.0;
    double count_r = 0.0;
    double count_g = 0.0;
    double count_b = 0.0;
    double count_a = 0.0;
    size_t count = 0;
};

static double managed_color_data_triangle_area(const std::array<Vec3f, 3> &vertices)
{
    const Vec3f edge0 = vertices[1] - vertices[0];
    const Vec3f edge1 = vertices[2] - vertices[0];
    const double area = 0.5 * double(edge0.cross(edge1).norm());
    return std::isfinite(area) && area > double(EPSILON) ? area : 0.0;
}

static double managed_color_data_triangle_area(const ColorFacetTriangle &facet)
{
    return managed_color_data_triangle_area(facet.vertices);
}

static void managed_color_data_accumulate_rgba(ManagedColorDataRgbaAccumulator &accumulator, uint32_t rgba, double area)
{
    const ColorRGBA color = unpack_vertex_color_rgba_for_conversion(rgba);
    if (area > 0.0) {
        accumulator.r += double(color.r()) * area;
        accumulator.g += double(color.g()) * area;
        accumulator.b += double(color.b()) * area;
        accumulator.a += double(color.a()) * area;
        accumulator.area_weight += area;
    }
    accumulator.count_r += double(color.r());
    accumulator.count_g += double(color.g());
    accumulator.count_b += double(color.b());
    accumulator.count_a += double(color.a());
    ++accumulator.count;
}

static uint32_t managed_color_data_averaged_rgba(const ManagedColorDataRgbaAccumulator &accumulator)
{
    if (accumulator.area_weight > 0.0)
        return pack_vertex_color_rgba(ColorRGBA(float(accumulator.r / accumulator.area_weight),
                                               float(accumulator.g / accumulator.area_weight),
                                               float(accumulator.b / accumulator.area_weight),
                                               float(accumulator.a / accumulator.area_weight)));

    const double count = std::max<double>(1.0, double(accumulator.count));
    return pack_vertex_color_rgba(ColorRGBA(float(accumulator.count_r / count),
                                           float(accumulator.count_g / count),
                                           float(accumulator.count_b / count),
                                           float(accumulator.count_a / count)));
}

static void managed_color_data_merge_rgba_accumulator(ManagedColorDataRgbaAccumulator       &target,
                                                      const ManagedColorDataRgbaAccumulator &source)
{
    target.r += source.r;
    target.g += source.g;
    target.b += source.b;
    target.a += source.a;
    target.area_weight += source.area_weight;
    target.count_r += source.count_r;
    target.count_g += source.count_g;
    target.count_b += source.count_b;
    target.count_a += source.count_a;
    target.count += source.count;
}

static int managed_color_data_read_rgba_nibble(const TriangleColorSplittingData &data, int bitstream_end, int &bit_idx)
{
    if (bit_idx < 0 || bit_idx + 4 > bitstream_end || size_t(bit_idx + 3) >= data.bitstream.size())
        return -1;

    int code = 0;
    for (int bit = 0; bit < 4; ++bit)
        code |= int(data.bitstream[size_t(bit_idx++)]) << bit;
    return code;
}

static std::array<std::array<Vec3f, 3>, 4> managed_color_data_split_rgba_triangle(const std::array<Vec3f, 3> &vertices,
                                                                                  int                         split_sides,
                                                                                  int                         special_side)
{
    std::array<std::array<Vec3f, 3>, 4> children{};
    special_side = std::clamp(special_side, 0, 2);
    const int j = (special_side + 1) % 3;
    const int k = (special_side + 2) % 3;
    const Vec3f a = vertices[size_t(special_side)];
    const Vec3f b = vertices[size_t(j)];
    const Vec3f c = vertices[size_t(k)];
    const Vec3f ab = 0.5f * (a + b);
    const Vec3f bc = 0.5f * (b + c);
    const Vec3f ca = 0.5f * (c + a);

    if (split_sides == 1) {
        children[0] = { a, b, bc };
        children[1] = { bc, c, a };
    } else if (split_sides == 2) {
        children[0] = { a, ab, ca };
        children[1] = { ab, b, ca };
        children[2] = { b, c, ca };
    } else {
        children[0] = { a, ab, ca };
        children[1] = { ab, b, bc };
        children[2] = { bc, c, ca };
        children[3] = { ab, bc, ca };
    }

    return children;
}

struct ManagedColorDataUnsplitResult
{
    bool valid = false;
    bool input_leaf = false;
    bool changed = false;
    ManagedColorDataRgbaAccumulator accumulator;
    std::vector<bool> bits;
    std::vector<uint32_t> colors;
};

static ManagedColorDataUnsplitResult managed_color_data_unsplit_rgba_node_once(const TriangleColorSplittingData &data,
                                                                               int                               bitstream_end,
                                                                               size_t                            color_end,
                                                                               int                              &bit_idx,
                                                                               size_t                           &color_idx,
                                                                               const std::array<Vec3f, 3>       &vertices)
{
    ManagedColorDataUnsplitResult result;
    const int code = managed_color_data_read_rgba_nibble(data, bitstream_end, bit_idx);
    if (code < 0)
        return result;

    const int split_sides = code & 0b11;
    if (split_sides == 0) {
        if (color_idx >= color_end || color_idx >= data.colors_rgba.size())
            return result;

        const uint32_t rgba = data.colors_rgba[color_idx++];
        managed_color_data_append_nibble(result.bits, 0u);
        result.colors.emplace_back(rgba);
        managed_color_data_accumulate_rgba(result.accumulator, rgba, managed_color_data_triangle_area(vertices));
        result.valid = true;
        result.input_leaf = true;
        return result;
    }

    const int special_side = (code >> 2) & 0b11;
    const std::array<std::array<Vec3f, 3>, 4> child_vertices =
        managed_color_data_split_rgba_triangle(vertices, split_sides, special_side);
    std::array<ManagedColorDataUnsplitResult, 4> child_results;
    bool all_children_are_leaves = true;
    bool any_child_changed = false;
    for (int child_idx = split_sides; child_idx >= 0; --child_idx) {
        child_results[size_t(child_idx)] = managed_color_data_unsplit_rgba_node_once(data,
                                                                                     bitstream_end,
                                                                                     color_end,
                                                                                     bit_idx,
                                                                                     color_idx,
                                                                                     child_vertices[size_t(child_idx)]);
        const ManagedColorDataUnsplitResult &child = child_results[size_t(child_idx)];
        if (!child.valid)
            return ManagedColorDataUnsplitResult();
        all_children_are_leaves &= child.input_leaf;
        any_child_changed |= child.changed;
        managed_color_data_merge_rgba_accumulator(result.accumulator, child.accumulator);
    }

    result.valid = true;
    if (all_children_are_leaves) {
        managed_color_data_append_nibble(result.bits, 0u);
        result.colors.emplace_back(managed_color_data_averaged_rgba(result.accumulator));
        result.changed = true;
        return result;
    }

    managed_color_data_append_nibble(result.bits, unsigned(code));
    for (int child_idx = split_sides; child_idx >= 0; --child_idx) {
        const ManagedColorDataUnsplitResult &child = child_results[size_t(child_idx)];
        result.bits.insert(result.bits.end(), child.bits.begin(), child.bits.end());
        result.colors.insert(result.colors.end(), child.colors.begin(), child.colors.end());
    }
    result.changed = any_child_changed;
    return result;
}

static bool rgba_data_has_split_triangles(const ColorFacetsAnnotation &annotation)
{
    const std::vector<bool> &bitstream = annotation.get_data().bitstream;
    for (size_t bit_idx = 0; bit_idx + 3 < bitstream.size(); bit_idx += 4) {
        int code = 0;
        for (int bit = 0; bit < 4; ++bit)
            code |= int(bitstream[bit_idx + size_t(bit)]) << bit;
        if ((code & 0b11) != 0)
            return true;
    }
    return false;
}

static bool object_has_splittable_rgba_data(const ModelObject &object)
{
    for (const ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part() && rgba_data_has_split_triangles(volume->texture_mapping_color_facets))
            return true;
    return false;
}

static bool unsplit_volume_rgba_data_once(ModelVolume &volume)
{
    const TriangleColorSplittingData &data = volume.texture_mapping_color_facets.get_data();
    if (data.triangles_to_split.empty() || !rgba_data_has_split_triangles(volume.texture_mapping_color_facets))
        return false;

    const indexed_triangle_set &its = volume.mesh().its;
    if (its.indices.empty())
        return false;

    std::unique_ptr<ColorFacetsAnnotation> unsplit = ColorFacetsAnnotation::make_temporary();
    if (!unsplit)
        return false;

    unsplit->set_metadata_json(volume.texture_mapping_color_facets.metadata_json());
    unsplit->reserve(int(data.triangles_to_split.size()));
    bool any = false;
    for (auto mapping_it = data.triangles_to_split.begin(); mapping_it != data.triangles_to_split.end(); ++mapping_it) {
        if (mapping_it->triangle_idx < 0 || size_t(mapping_it->triangle_idx) >= its.indices.size())
            continue;

        const auto &tri = its.indices[size_t(mapping_it->triangle_idx)];
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
            continue;
        if (size_t(tri[0]) >= its.vertices.size() ||
            size_t(tri[1]) >= its.vertices.size() ||
            size_t(tri[2]) >= its.vertices.size())
            continue;

        const auto next_it = std::next(mapping_it);
        const int bitstream_end = next_it == data.triangles_to_split.end() ?
            int(data.bitstream.size()) :
            next_it->bitstream_start_idx;
        const size_t color_end = next_it == data.triangles_to_split.end() ?
            data.colors_rgba.size() :
            size_t(next_it->color_start_idx);
        if (mapping_it->bitstream_start_idx < 0 ||
            mapping_it->bitstream_start_idx >= bitstream_end ||
            size_t(bitstream_end) > data.bitstream.size() ||
            mapping_it->color_start_idx < 0 ||
            size_t(mapping_it->color_start_idx) >= color_end ||
            color_end > data.colors_rgba.size())
            continue;

        const std::array<Vec3f, 3> vertices = {
            its.vertices[size_t(tri[0])].cast<float>(),
            its.vertices[size_t(tri[1])].cast<float>(),
            its.vertices[size_t(tri[2])].cast<float>()
        };
        int bit_idx = mapping_it->bitstream_start_idx;
        size_t color_idx = size_t(mapping_it->color_start_idx);
        ManagedColorDataUnsplitResult result =
            managed_color_data_unsplit_rgba_node_once(data, bitstream_end, color_end, bit_idx, color_idx, vertices);
        if (!result.valid || result.bits.empty() || result.colors.empty())
            continue;

        std::string encoded;
        managed_color_data_bits_to_hex(result.bits, encoded);
        encoded.push_back('|');
        for (uint32_t rgba : result.colors)
            managed_color_data_append_rgba_hex(encoded, rgba);
        unsplit->set_triangle_from_string(mapping_it->triangle_idx, encoded);
        any = true;
    }
    if (!any)
        return false;

    unsplit->shrink_to_fit();
    if (volume.texture_mapping_color_facets.equals(*unsplit))
        return false;

    volume.texture_mapping_color_facets.assign(*unsplit);
    return true;
}

static bool unsplit_object_rgba_data_once(ModelObject &object)
{
    bool changed = false;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        changed |= unsplit_volume_rgba_data_once(*volume);
    }
    return changed;
}

static bool simplify_volume_rgba_data_to_one_color_per_triangle(ModelVolume &volume)
{
    if (volume.texture_mapping_color_facets.empty())
        return false;

    const indexed_triangle_set &its = volume.mesh().its;
    if (its.indices.empty())
        return false;

    std::vector<ColorFacetTriangle> facets;
    volume.texture_mapping_color_facets.get_facet_triangles(volume, facets);
    if (facets.empty())
        return false;

    std::vector<ManagedColorDataRgbaAccumulator> accumulators(its.indices.size());
    for (const ColorFacetTriangle &facet : facets) {
        if (facet.source_triangle < 0 || size_t(facet.source_triangle) >= accumulators.size())
            continue;
        managed_color_data_accumulate_rgba(accumulators[size_t(facet.source_triangle)], facet.rgba, managed_color_data_triangle_area(facet));
    }

    std::unique_ptr<ColorFacetsAnnotation> simplified = ColorFacetsAnnotation::make_temporary();
    if (!simplified)
        return false;

    simplified->set_metadata_json(volume.texture_mapping_color_facets.metadata_json());
    simplified->reserve(int(its.indices.size()));
    bool any = false;
    for (size_t triangle_idx = 0; triangle_idx < accumulators.size(); ++triangle_idx) {
        if (accumulators[triangle_idx].count == 0)
            continue;

        std::string encoded = "0|";
        managed_color_data_append_rgba_hex(encoded, managed_color_data_averaged_rgba(accumulators[triangle_idx]));
        simplified->set_triangle_from_string(int(triangle_idx), encoded);
        any = true;
    }
    if (!any)
        return false;

    simplified->shrink_to_fit();
    if (volume.texture_mapping_color_facets.equals(*simplified))
        return false;

    volume.texture_mapping_color_facets.assign(*simplified);
    return true;
}

static bool simplify_object_rgba_data_to_one_color_per_triangle(ModelObject &object)
{
    bool changed = false;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        changed |= simplify_volume_rgba_data_to_one_color_per_triangle(*volume);
    }
    return changed;
}

static ColorRGBA managed_color_data_state_color(const ManagedRegionColorSource &source, unsigned int state)
{
    if (state < source.state_colors.size())
        return source.state_colors[size_t(state)];
    return source.state_colors.empty() ? ColorRGBA(1.f, 1.f, 1.f, 1.f) : source.state_colors.front();
}

static ColorRGBA managed_color_data_region_background_color(const ManagedRegionColorSource &source, const ColorRGBA &fallback_color)
{
    ColorRGBA color = source.state_colors.empty() ? fallback_color : managed_color_data_state_color(source, 0);
    color.a(1.f);
    return color;
}

static ColorRGBA managed_color_data_region_unpainted_color(const ManagedRegionColorSource &source, const ColorRGBA &fallback_color)
{
    ColorRGBA color = managed_color_data_region_background_color(source, fallback_color);
    color.a(0.f);
    return color;
}

static ColorRGBA managed_color_data_region_rgba_color(const ManagedRegionColorSource &source,
                                                      unsigned int                    state,
                                                      const ColorRGBA                &unpainted_color)
{
    return state == 0 || managed_region_state_is_texture_mapping_zone(source, state) ?
        unpainted_color :
        managed_color_data_state_color(source, state);
}

static bool managed_color_data_append_region_tree_as_rgba_tree(const TriangleSelector::TriangleSplittingData &data,
                                                               int                                             bitstream_end,
                                                               int                                            &bit_idx,
                                                               const ManagedRegionColorSource                &source,
                                                               const ColorRGBA                               &unpainted_color,
                                                               std::vector<bool>                             &out_bits,
                                                               std::vector<uint32_t>                         &out_colors)
{
    const int code = managed_color_data_read_nibble(data, bitstream_end, bit_idx);
    if (code < 0)
        return false;

    const int split_sides = code & 0b11;
    if (split_sides != 0) {
        const int special_side = code >> 2;
        if (special_side < 0 || special_side >= 3 || (split_sides != 1 && split_sides != 2 && special_side != 0))
            return false;
        managed_color_data_append_nibble(out_bits, unsigned(code));
        for (int child_idx = split_sides; child_idx >= 0; --child_idx)
            if (!managed_color_data_append_region_tree_as_rgba_tree(data,
                                                                    bitstream_end,
                                                                    bit_idx,
                                                                    source,
                                                                    unpainted_color,
                                                                    out_bits,
                                                                    out_colors))
                return false;
        return true;
    }

    unsigned int state = 0;
    if ((code & 0b1100) == 0b1100) {
        int next_code = managed_color_data_read_nibble(data, bitstream_end, bit_idx);
        if (next_code < 0)
            return false;
        unsigned int extension_count = 0;
        while (next_code == 0b1111) {
            ++extension_count;
            next_code = managed_color_data_read_nibble(data, bitstream_end, bit_idx);
            if (next_code < 0)
                return false;
        }
        state = unsigned(next_code) + 15u * extension_count + 3u;
    } else {
        state = unsigned(code >> 2);
    }

    managed_color_data_append_nibble(out_bits, 0u);
    out_colors.emplace_back(pack_vertex_color_rgba(managed_color_data_region_rgba_color(source, state, unpainted_color)));
    return true;
}

static bool append_direct_color_region_rgba_triangle(const TriangleSelector::TriangleSplittingData &data,
                                                     size_t                                         mapping_idx,
                                                     const ManagedRegionColorSource                &source,
                                                     const ColorRGBA                               &unpainted_color,
                                                     ColorFacetsAnnotation                         &out)
{
    if (mapping_idx >= data.triangles_to_split.size())
        return false;

    const TriangleSelector::TriangleBitStreamMapping &mapping = data.triangles_to_split[mapping_idx];
    const int bitstream_start = mapping.bitstream_start_idx;
    const int bitstream_end = mapping_idx + 1 == data.triangles_to_split.size() ?
        int(data.bitstream.size()) :
        data.triangles_to_split[mapping_idx + 1].bitstream_start_idx;
    if (mapping.triangle_idx < 0 ||
        bitstream_start < 0 ||
        bitstream_start >= bitstream_end ||
        size_t(bitstream_end) > data.bitstream.size())
        return false;

    int bit_idx = bitstream_start;
    std::vector<bool> out_bits;
    std::vector<uint32_t> out_colors;
    if (!managed_color_data_append_region_tree_as_rgba_tree(data, bitstream_end, bit_idx, source, unpainted_color, out_bits, out_colors) ||
        out_bits.empty() ||
        out_colors.empty())
        return false;

    std::string encoded;
    managed_color_data_bits_to_hex(out_bits, encoded);
    encoded.push_back('|');
    for (uint32_t rgba : out_colors)
        managed_color_data_append_rgba_hex(encoded, rgba);
    out.set_triangle_from_string(mapping.triangle_idx, encoded);
    return true;
}

static std::unique_ptr<ColorFacetsAnnotation> build_rgba_data_from_color_regions(const ModelVolume &volume, const ColorRGBA &fallback_color)
{
    if (volume.mmu_segmentation_facets.empty())
        return nullptr;

    const TriangleSelector::TriangleSplittingData &data = volume.mmu_segmentation_facets.get_data();
    if (data.triangles_to_split.empty())
        return nullptr;

    const ManagedRegionColorSource source = build_managed_region_color_source(volume);
    return build_rgba_data_from_color_region_data(data, volume.mesh().its.indices.size(), source, fallback_color);
}

static std::unique_ptr<ColorFacetsAnnotation> build_rgba_data_from_color_region_data(
    const TriangleSelector::TriangleSplittingData &data,
    size_t                                         triangle_count,
    const ManagedRegionColorSource                &source,
    const ColorRGBA                               &fallback_color,
    const std::function<void()>                   &check_cancel)
{
    if (data.triangles_to_split.empty())
        return nullptr;

    std::unique_ptr<ColorFacetsAnnotation> out = ColorFacetsAnnotation::make_temporary();
    if (!out)
        return nullptr;

    out->reserve(int(data.triangles_to_split.size()));
    const ColorRGBA background_color = managed_color_data_region_background_color(source, fallback_color);
    const ColorRGBA unpainted_color = managed_color_data_region_unpainted_color(source, fallback_color);
    bool any = false;
    for (size_t mapping_idx = 0; mapping_idx < data.triangles_to_split.size(); ++mapping_idx) {
        if (check_cancel)
            check_cancel();
        const int triangle_idx = data.triangles_to_split[mapping_idx].triangle_idx;
        if (triangle_idx < 0 || size_t(triangle_idx) >= triangle_count)
            continue;
        any |= append_direct_color_region_rgba_triangle(data, mapping_idx, source, unpainted_color, *out);
    }
    if (!any)
        return nullptr;

    out->set_metadata_json(rgb_metadata_json(background_color));
    out->shrink_to_fit();
    return out;
}

static ColorRGBA sample_managed_volume_color_source(const ModelVolume                &volume,
                                                    const VolumeColorSource          &rgba_source,
                                                    const ManagedRegionColorSource   &region_source,
                                                    size_t                            tri_idx,
                                                    const Vec3f                      &point,
                                                    const Vec3f                      &barycentric,
                                                    bool                              use_rgba,
                                                    bool                              use_image_texture,
                                                    bool                              use_vertex_colors,
                                                    bool                              use_color_regions,
                                                    const ColorRGBA                  &fallback_color)
{
    if (use_rgba && rgba_source.rgb_background_color) {
        if (std::optional<ColorRGBA> color = sample_rgb_color_facets(rgba_source.rgb_facets,
                                                                     rgba_source.rgb_by_source_triangle,
                                                                     int(tri_idx),
                                                                     point))
            return *color;
        return *rgba_source.rgb_background_color;
    }

    if (use_image_texture && model_volume_has_bakeable_image_texture_data(&volume) && tri_idx < volume.imported_texture_uv_valid.size()) {
        if (std::optional<ColorRGBA> color =
                sample_image_texture_rgba_for_conversion(volume.imported_texture_rgba,
                                                         volume.imported_texture_width,
                                                         volume.imported_texture_height,
                                                         volume.imported_texture_uvs_per_face,
                                                         volume.imported_texture_uv_valid,
                                                         tri_idx,
                                                         barycentric))
            return *color;
    }

    if (use_vertex_colors)
        if (std::optional<ColorRGBA> color =
                sample_vertex_colors_rgba_for_conversion(volume.mesh().its, volume.imported_vertex_colors_rgba, tri_idx, barycentric))
            return *color;

    if (use_color_regions) {
        if (std::optional<ColorRGBA> color = sample_managed_region_color_source(region_source, int(tri_idx), point))
            return *color;
    }

    return fallback_color;
}

static ColorRGBA sample_managed_volume_color_source_with_region_overlay(const ModelVolume              &volume,
                                                                        const VolumeColorSource        &rgba_source,
                                                                        const ManagedRegionColorSource &region_source,
                                                                        size_t                          tri_idx,
                                                                        const Vec3f                    &point,
                                                                        const Vec3f                    &barycentric,
                                                                        const ManagedColorSourceFlags  &base_source_flags,
                                                                        bool                            use_region_overlay,
                                                                        const ColorRGBA                &fallback_color)
{
    if (use_region_overlay) {
        if (std::optional<ColorRGBA> color = sample_managed_region_overlay_color_source(region_source, int(tri_idx), point))
            return *color;
    }

    return sample_managed_volume_color_source(volume,
                                              rgba_source,
                                              region_source,
                                              tri_idx,
                                              point,
                                              barycentric,
                                              base_source_flags.use_rgba,
                                              base_source_flags.use_image_texture,
                                              base_source_flags.use_vertex_colors,
                                              false,
                                              fallback_color);
}

static bool convert_object_to_vertex_colors(ModelObject &object, const ManagedColorDataCreateSource &source, bool replace_existing = false)
{
    if (!replace_existing && object_has_vertex_color_data(object))
        return false;

    bool changed = false;
    const ManagedColorSourceFlags source_flags = managed_color_source_flags(source);
    const ColorRGBA fallback_color = blank_color_for_managed_target(ManagedColorDataType::VertexColors);
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty())
            continue;

        const bool use_region_overlay = source_flags.use_color_regions;
        const ManagedColorSourceFlags base_source_flags = use_region_overlay ?
            managed_target_or_highest_existing_color_source_flags(*volume, ManagedColorDataType::VertexColors) :
            source_flags;
        const VolumeColorSource rgba_source = build_volume_color_source(*volume);
        const ManagedRegionColorSource region_source = use_region_overlay ? build_managed_region_color_overlay_source(*volume) :
                                                                            build_managed_region_color_source(*volume);
        const ColorRGBA sample_fallback_color = use_region_overlay ?
            managed_color_data_region_background_color(region_source, fallback_color) :
            fallback_color;
        std::vector<std::array<float, 4>> accumulators(its.vertices.size(), { 0.f, 0.f, 0.f, 0.f });
        std::vector<unsigned int> counts(its.vertices.size(), 0);

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            for (int corner = 0; corner < 3; ++corner) {
                if (tri[corner] < 0 || size_t(tri[corner]) >= its.vertices.size())
                    continue;
                Vec3f barycentric = Vec3f::Zero();
                barycentric[corner] = 1.f;
                const Vec3f point = its.vertices[size_t(tri[corner])].cast<float>();
                const ColorRGBA color = use_region_overlay ?
                    sample_managed_volume_color_source_with_region_overlay(*volume,
                                                                          rgba_source,
                                                                          region_source,
                                                                          tri_idx,
                                                                          point,
                                                                          barycentric,
                                                                          base_source_flags,
                                                                          true,
                                                                          sample_fallback_color) :
                    sample_managed_volume_color_source(*volume,
                                                       rgba_source,
                                                       region_source,
                                                       tri_idx,
                                                       point,
                                                       barycentric,
                                                       source_flags.use_rgba,
                                                       source_flags.use_image_texture,
                                                       source_flags.use_vertex_colors,
                                                       source_flags.use_color_regions,
                                                       sample_fallback_color);
                std::array<float, 4> &accumulator = accumulators[size_t(tri[corner])];
                accumulator[0] += color.r();
                accumulator[1] += color.g();
                accumulator[2] += color.b();
                accumulator[3] += color.a();
                ++counts[size_t(tri[corner])];
            }
        }

        std::vector<uint32_t> vertex_colors;
        vertex_colors.reserve(its.vertices.size());
        for (size_t idx = 0; idx < its.vertices.size(); ++idx) {
            ColorRGBA color = fallback_color;
            if (counts[idx] > 0) {
                const float inv = 1.f / float(counts[idx]);
                color = ColorRGBA(accumulators[idx][0] * inv,
                                  accumulators[idx][1] * inv,
                                  accumulators[idx][2] * inv,
                                  accumulators[idx][3] * inv);
            }
            vertex_colors.emplace_back(pack_vertex_color_rgba(color));
        }

        volume->imported_vertex_colors_rgba = std::move(vertex_colors);
        changed = true;
    }
    return changed;
}

static bool convert_object_to_image_texture(ModelObject &object, const ManagedColorDataCreateSource &source, bool replace_existing = false)
{
    if (!replace_existing && object_has_image_texture_data(object))
        return false;

    bool changed = false;
    const ManagedColorSourceFlags source_flags = managed_color_source_flags(source);
    const ColorRGBA fallback_color = blank_color_for_managed_target(ManagedColorDataType::ImageTexture);
    std::optional<ColorRGBA> color_region_background_color;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const bool use_region_overlay = source_flags.use_color_regions;
        const ManagedRegionColorSource region_source = use_region_overlay ? build_managed_region_color_overlay_source(*volume) :
                                                                            ManagedRegionColorSource();
        const ColorRGBA sample_fallback_color = use_region_overlay ?
            managed_color_data_region_background_color(region_source, fallback_color) :
            fallback_color;
        const ManagedColorSourceFlags base_source_flags = use_region_overlay ?
            managed_target_or_highest_existing_color_source_flags(*volume, ManagedColorDataType::ImageTexture) :
            source_flags;
        const bool preserve_existing_image_target = use_region_overlay && base_source_flags.use_image_texture;
        const bool preserve_existing_image_pixels =
            preserve_existing_image_target && managed_region_source_uses_texture_mapping_zone(region_source);
        GeneratedImageTextureAtlas generated_atlas;
        if (!preserve_existing_image_target) {
            Transform3d metric_matrix = volume->get_matrix();
            if (!object.instances.empty() && object.instances.front() != nullptr)
                metric_matrix = object.instances.front()->get_transformation().get_matrix() * metric_matrix;
            if (!initialize_generated_image_texture(*volume, sample_fallback_color, &generated_atlas, &metric_matrix))
                continue;
        }

        if (use_region_overlay) {
            const ColorRGBA background_color = managed_color_data_region_background_color(region_source, fallback_color);
            if (!color_region_background_color)
                color_region_background_color = background_color;
            changed |= set_texture_mapping_background_config(volume->config, background_color);
        }

        const std::vector<uint8_t> preserved_texture_rgba = preserve_existing_image_pixels ?
            std::vector<uint8_t>(volume->imported_texture_rgba.begin(), volume->imported_texture_rgba.end()) :
            std::vector<uint8_t>();
        if (preserve_existing_image_target)
            changed |= fill_rgba_pixels(volume->imported_texture_rgba, sample_fallback_color);

        const VolumeColorSource rgba_source = build_volume_color_source(*volume);
        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            if (tri_idx >= volume->imported_texture_uv_valid.size() ||
                volume->imported_texture_uv_valid[tri_idx] == 0)
                continue;
            const size_t uv_offset = tri_idx * 6;
            if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
                continue;

            const std::array<Vec2f, 3> uvs = unwrap_projection_uvs(std::array<Vec2f, 3>{
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5])
            });
            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            const float texture_width = float(volume->imported_texture_width);
            const float texture_height = float(volume->imported_texture_height);
            const std::array<Vec2f, 3> pixel_uvs = {
                Vec2f(uvs[0].x() * texture_width, uvs[0].y() * texture_height),
                Vec2f(uvs[1].x() * texture_width, uvs[1].y() * texture_height),
                Vec2f(uvs[2].x() * texture_width, uvs[2].y() * texture_height)
            };
            const int padding_px = preserve_existing_image_target ? 2 : std::max(generated_atlas.padding_px, 0);
            int min_x = int(std::floor(std::min({ uvs[0].x(), uvs[1].x(), uvs[2].x() }) * texture_width)) - padding_px;
            int max_x = int(std::ceil(std::max({ uvs[0].x(), uvs[1].x(), uvs[2].x() }) * texture_width)) + padding_px;
            int min_y = int(std::floor(std::min({ uvs[0].y(), uvs[1].y(), uvs[2].y() }) * texture_height)) - padding_px;
            int max_y = int(std::ceil(std::max({ uvs[0].y(), uvs[1].y(), uvs[2].y() }) * texture_height)) + padding_px;
            if (!preserve_existing_image_target && tri_idx < generated_atlas.island_by_triangle.size()) {
                const int island_idx = generated_atlas.island_by_triangle[tri_idx];
                if (island_idx >= 0 && size_t(island_idx) < generated_atlas.islands.size()) {
                    const GeneratedImageTextureIsland &island = generated_atlas.islands[size_t(island_idx)];
                    min_x = std::clamp(min_x, island.x, island.x + island.rect_width - 1);
                    max_x = std::clamp(max_x, island.x, island.x + island.rect_width - 1);
                    min_y = std::clamp(min_y, island.y, island.y + island.rect_height - 1);
                    max_y = std::clamp(max_y, island.y, island.y + island.rect_height - 1);
                }
            }
            if (preserve_existing_image_target &&
                (max_x - min_x > int(volume->imported_texture_width) * 2 ||
                 max_y - min_y > int(volume->imported_texture_height) * 2)) {
                min_x = 0;
                max_x = int(volume->imported_texture_width) - 1;
                min_y = 0;
                max_y = int(volume->imported_texture_height) - 1;
            }
            for (int y_px = min_y; y_px <= max_y; ++y_px) {
                for (int x_px = min_x; x_px <= max_x; ++x_px) {
                    Vec3f barycentric = Vec3f::Zero();
                    const Vec2f pixel(float(x_px) + 0.5f, float(y_px) + 0.5f);
                    if (!conservative_barycentric_weights_2d(pixel,
                                                              pixel_uvs[0],
                                                              pixel_uvs[1],
                                                              pixel_uvs[2],
                                                              float(padding_px) + 0.7072f,
                                                              barycentric))
                        continue;

                    const Vec3f point = vertices[0] * barycentric.x() +
                                        vertices[1] * barycentric.y() +
                                        vertices[2] * barycentric.z();
                    std::optional<ColorRGBA> overlay_color;
                    if (use_region_overlay)
                        overlay_color = sample_managed_region_overlay_color_source(region_source, int(tri_idx), point);

                    const uint32_t write_x = preserve_existing_image_target ?
                        wrapped_texture_pixel(x_px, volume->imported_texture_width) :
                        uint32_t(x_px);
                    const uint32_t write_y = preserve_existing_image_target ?
                        wrapped_texture_pixel(y_px, volume->imported_texture_height) :
                        uint32_t(y_px);
                    const ColorRGBA color = overlay_color ?
                        *overlay_color :
                        (!preserved_texture_rgba.empty() ?
                             read_rgba_pixel(preserved_texture_rgba, volume->imported_texture_width, write_x, write_y) :
                             sample_managed_volume_color_source(*volume,
                                                                rgba_source,
                                                                region_source,
                                                                tri_idx,
                                                                point,
                                                                barycentric,
                                                                base_source_flags.use_rgba,
                                                                base_source_flags.use_image_texture,
                                                                base_source_flags.use_vertex_colors,
                                                                false,
                                                                sample_fallback_color));
                    write_rgba_pixel(volume->imported_texture_rgba,
                                     volume->imported_texture_width,
                                     write_x,
                                     write_y,
                                     color);
                }
            }
        }

        refresh_imported_texture_storage(*volume);
        changed = true;
    }
    if (color_region_background_color)
        changed |= set_texture_mapping_background_config(object.config, *color_region_background_color);
    return changed;
}

static bool object_has_regenerable_image_texture_uvs(const ModelObject &object)
{
    for (const ModelVolume *volume : object.volumes)
        if (volume != nullptr && volume->is_model_part() && model_volume_has_bakeable_image_texture_data(volume))
            return true;
    return false;
}

static wxString managed_image_texture_uv_map_info_text(const ModelObject *object)
{
    if (object == nullptr)
        return _L("UV map: none");

    bool has_texture = false;
    bool has_imported = false;
    bool has_generated = false;
    int generated_version = -1;
    bool mixed_generated_versions = false;
    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_imported_image_texture_data(volume))
            continue;

        has_texture = true;
        if (volume->uv_map_generator_version > 0) {
            has_generated = true;
            if (generated_version < 0)
                generated_version = volume->uv_map_generator_version;
            else if (generated_version != volume->uv_map_generator_version)
                mixed_generated_versions = true;
        } else {
            has_imported = true;
        }
    }

    if (!has_texture)
        return _L("UV map: none");
    if (has_imported && has_generated)
        return _L("UV map: mixed");
    if (has_generated) {
        if (mixed_generated_versions)
            return _L("UV map: generated (mixed)");
        return wxString::Format(_L("UV map: generated (v%d)"), generated_version);
    }
    return _L("UV map: imported");
}

static bool regenerate_volume_image_texture_uv_map(ModelObject &object, ModelVolume &volume)
{
    if (!model_volume_has_bakeable_image_texture_data(&volume))
        return false;

    const indexed_triangle_set &its = volume.mesh().its;
    const uint32_t source_width = volume.imported_texture_width;
    const uint32_t source_height = volume.imported_texture_height;
    const std::vector<uint8_t> source_rgba(volume.imported_texture_rgba.begin(), volume.imported_texture_rgba.end());
    const std::vector<float> source_uvs(volume.imported_texture_uvs_per_face.begin(), volume.imported_texture_uvs_per_face.end());
    const std::vector<uint8_t> source_uv_valid(volume.imported_texture_uv_valid.begin(), volume.imported_texture_uv_valid.end());
    const bool source_has_raw_atlas = model_volume_has_raw_atlas_texture_data(&volume);
    const uint32_t source_raw_channels = volume.imported_texture_raw_channels;
    const std::vector<uint8_t> source_raw_offsets(volume.imported_texture_raw_filament_offsets.begin(),
                                                  volume.imported_texture_raw_filament_offsets.end());
    const std::string source_raw_metadata = volume.imported_texture_raw_metadata_json;

    GeneratedImageTextureAtlas generated_atlas;
    Transform3d metric_matrix = volume.get_matrix();
    if (!object.instances.empty() && object.instances.front() != nullptr)
        metric_matrix = object.instances.front()->get_transformation().get_matrix() * metric_matrix;
    if (!initialize_generated_image_texture(volume,
                                            blank_color_for_managed_target(ManagedColorDataType::ImageTexture),
                                            &generated_atlas,
                                            &metric_matrix))
        return false;

    if (source_has_raw_atlas) {
        volume.imported_texture_raw_channels = source_raw_channels;
        volume.imported_texture_raw_metadata_json = source_raw_metadata;
        volume.imported_texture_raw_filament_offsets.assign(size_t(volume.imported_texture_width) *
                                                                size_t(volume.imported_texture_height) *
                                                                size_t(source_raw_channels),
                                                            0);
    }

    bool changed = true;
    const float target_width = float(volume.imported_texture_width);
    const float target_height = float(volume.imported_texture_height);
    for (const GeneratedImageTextureIsland &island : generated_atlas.islands) {
        const size_t tri_idx = island.tri_idx;
        if (tri_idx >= its.indices.size() ||
            tri_idx >= source_uv_valid.size() ||
            source_uv_valid[tri_idx] == 0)
            continue;

        const size_t source_uv_offset = tri_idx * 6;
        const size_t target_uv_offset = tri_idx * 6;
        if (source_uv_offset + 5 >= source_uvs.size() ||
            target_uv_offset + 5 >= volume.imported_texture_uvs_per_face.size())
            continue;

        const std::array<Vec2f, 3> source_face_uvs = unwrap_projection_uvs(std::array<Vec2f, 3>{
            Vec2f(source_uvs[source_uv_offset + 0], source_uvs[source_uv_offset + 1]),
            Vec2f(source_uvs[source_uv_offset + 2], source_uvs[source_uv_offset + 3]),
            Vec2f(source_uvs[source_uv_offset + 4], source_uvs[source_uv_offset + 5])
        });
        const std::array<Vec2f, 3> target_pixel_uvs = {
            Vec2f(volume.imported_texture_uvs_per_face[target_uv_offset + 0] * target_width,
                  volume.imported_texture_uvs_per_face[target_uv_offset + 1] * target_height),
            Vec2f(volume.imported_texture_uvs_per_face[target_uv_offset + 2] * target_width,
                  volume.imported_texture_uvs_per_face[target_uv_offset + 3] * target_height),
            Vec2f(volume.imported_texture_uvs_per_face[target_uv_offset + 4] * target_width,
                  volume.imported_texture_uvs_per_face[target_uv_offset + 5] * target_height)
        };

        const int min_x = std::clamp(island.x, 0, int(volume.imported_texture_width) - 1);
        const int max_x = std::clamp(island.x + island.rect_width - 1, 0, int(volume.imported_texture_width) - 1);
        const int min_y = std::clamp(island.y, 0, int(volume.imported_texture_height) - 1);
        const int max_y = std::clamp(island.y + island.rect_height - 1, 0, int(volume.imported_texture_height) - 1);
        for (int y_px = min_y; y_px <= max_y; ++y_px) {
            for (int x_px = min_x; x_px <= max_x; ++x_px) {
                Vec3f barycentric = Vec3f::Zero();
                const Vec2f pixel(float(x_px) + 0.5f, float(y_px) + 0.5f);
                if (!barycentric_weights_2d(pixel, target_pixel_uvs[0], target_pixel_uvs[1], target_pixel_uvs[2], barycentric))
                    continue;
                if (barycentric.x() < -1e-4f || barycentric.y() < -1e-4f || barycentric.z() < -1e-4f)
                    barycentric = normalized_nonnegative_barycentric(barycentric);

                const Vec3f safe_barycentric =
                    texture_barycentric_for_bleed_safe_sampling(barycentric,
                                                                source_face_uvs[0],
                                                                source_face_uvs[1],
                                                                source_face_uvs[2],
                                                                source_width,
                                                                source_height);
                const Vec2f source_uv = source_face_uvs[0] * safe_barycentric.x() +
                                        source_face_uvs[1] * safe_barycentric.y() +
                                        source_face_uvs[2] * safe_barycentric.z();
                if (source_has_raw_atlas) {
                    const std::vector<uint8_t> values =
                        sample_raw_offsets_bilinear_wrapped(source_raw_offsets,
                                                            source_width,
                                                            source_height,
                                                            source_raw_channels,
                                                            source_uv.x(),
                                                            source_uv.y());
                    write_raw_offset_pixel(volume.imported_texture_raw_filament_offsets,
                                           volume.imported_texture_width,
                                           source_raw_channels,
                                           uint32_t(x_px),
                                           uint32_t(y_px),
                                           values);
                } else {
                    const ColorRGBA color = sample_rgba_bilinear_wrapped(source_rgba,
                                                                         source_width,
                                                                         source_height,
                                                                         source_uv.x(),
                                                                         source_uv.y());
                    write_rgba_pixel(volume.imported_texture_rgba,
                                     volume.imported_texture_width,
                                     uint32_t(x_px),
                                     uint32_t(y_px),
                                     color);
                }
            }
        }
    }

    if (source_has_raw_atlas) {
        refresh_imported_texture_preview_from_raw_offsets(volume);
        refresh_imported_texture_raw_storage(volume);
    }
    refresh_imported_texture_storage(volume);
    return changed;
}

static bool regenerate_object_image_texture_uv_maps(ModelObject &object)
{
    bool changed = false;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        changed |= regenerate_volume_image_texture_uv_map(object, *volume);
    }
    return changed;
}

static bool convert_object_to_rgba_data(ModelObject &object, const ManagedColorDataCreateSource &source, bool replace_existing = false)
{
    if (!replace_existing && object_has_rgba_data(object))
        return false;

    bool changed = false;
    const ManagedColorSourceFlags source_flags = managed_color_source_flags(source);
    const ColorRGBA fallback_color = blank_color_for_managed_target(ManagedColorDataType::RgbaData);
    std::optional<ColorRGBA> color_region_background_color;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.indices.empty() || its.vertices.empty())
            continue;

        const bool use_region_overlay = source_flags.use_color_regions;
        const ManagedColorSourceFlags base_source_flags = use_region_overlay ?
            managed_target_or_highest_existing_color_source_flags(*volume, ManagedColorDataType::RgbaData) :
            source_flags;
        const ManagedRegionColorSource region_source = use_region_overlay ? build_managed_region_color_overlay_source(*volume) :
                                                                            ManagedRegionColorSource();
        const ColorRGBA sample_fallback_color = use_region_overlay ?
            managed_color_data_region_background_color(region_source, fallback_color) :
            fallback_color;
        const ColorRGBA region_unpainted_color = use_region_overlay ?
            managed_color_data_region_unpainted_color(region_source, fallback_color) :
            fallback_color;
        std::unique_ptr<ColorFacetsAnnotation> rgb_data;
        bool sampled = false;
        if (use_region_overlay &&
            managed_color_source_flags_empty(base_source_flags) &&
            !volume->mmu_segmentation_facets.empty()) {
            rgb_data = build_rgba_data_from_color_regions(*volume, fallback_color);
            sampled = rgb_data != nullptr && !rgb_data->empty();
        }

        if (!sampled) {
            rgb_data = ColorFacetsAnnotation::make_temporary();
            if (!rgb_data)
                continue;

            const VolumeColorSource rgba_source = build_volume_color_source(*volume);
            TextureMappingColorSampler sampler = [volume,
                                                  rgba_source,
                                                  region_source,
                                                  sample_fallback_color,
                                                  region_unpainted_color,
                                                  source_flags,
                                                  base_source_flags,
                                                  use_region_overlay](
                                                     size_t tri_idx,
                                                     const Vec3f &point,
                                                     const Vec3f &barycentric) {
                ColorRGBA color = sample_fallback_color;
                if (use_region_overlay) {
                    if (std::optional<ColorRGBA> overlay_color =
                            sample_managed_region_rgba_overlay_color_source(region_source, int(tri_idx), point, region_unpainted_color)) {
                        color = *overlay_color;
                    } else {
                        color = sample_managed_volume_color_source(*volume,
                                                                   rgba_source,
                                                                   region_source,
                                                                   tri_idx,
                                                                   point,
                                                                   barycentric,
                                                                   base_source_flags.use_rgba,
                                                                   base_source_flags.use_image_texture,
                                                                   base_source_flags.use_vertex_colors,
                                                                   false,
                                                                   sample_fallback_color);
                    }
                } else {
                    color = sample_managed_volume_color_source(*volume,
                                                               rgba_source,
                                                               region_source,
                                                               tri_idx,
                                                               point,
                                                               barycentric,
                                                               source_flags.use_rgba,
                                                               source_flags.use_image_texture,
                                                               source_flags.use_vertex_colors,
                                                               source_flags.use_color_regions,
                                                               sample_fallback_color);
                }
                return pack_vertex_color_rgba(color);
            };

            const bool has_image_texture_source = base_source_flags.use_image_texture && model_volume_has_bakeable_image_texture_data(volume);
            const bool has_geometry_color_source =
                (base_source_flags.use_vertex_colors && !volume->imported_vertex_colors_rgba.empty()) ||
                (use_region_overlay && !volume->mmu_segmentation_facets.empty());
            if (use_region_overlay && base_source_flags.use_rgba && !volume->texture_mapping_color_facets.empty()) {
                rgb_data->assign(volume->texture_mapping_color_facets);
                std::vector<bool> resample_triangles = managed_region_overlay_source_triangles(region_source, its.indices.size());
                const bool has_resample_triangle = std::find(resample_triangles.begin(), resample_triangles.end(), true) != resample_triangles.end();
                if (has_resample_triangle) {
                    const float target_edge = std::max(mesh_max_axis_span(its) / 160.f, 0.25f);
                    TextureMappingColorSubdivisionDepths subdivision_depths =
                        [target_edge](size_t, const std::array<Vec3f, 3> &vertices) {
                            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_edge, 5);
                            return std::make_pair(depth, depth);
                        };
                    sampled = rgb_data->set_from_triangle_sampler(its, sampler, 5, 0.025f, subdivision_depths, &resample_triangles);
                } else {
                    sampled = !rgb_data->empty();
                }
            } else {
                sampled = build_rgba_data_from_color_sampler(its,
                                                             sampler,
                                                             has_image_texture_source,
                                                             has_geometry_color_source,
                                                             [volume](size_t tri_idx) {
                                                                 return texture_triangle_uv_pixel_span(volume, tri_idx);
                                                             },
                                                             sample_fallback_color,
                                                             *rgb_data);
            }
        }

        if (!sampled && rgb_data->empty())
            continue;
        if (rgb_data->metadata_json().empty())
            rgb_data->set_metadata_json(rgb_metadata_json(sample_fallback_color));
        if (!replace_existing && volume->texture_mapping_color_facets.equals(*rgb_data))
            continue;
        if (source_flags.use_color_regions) {
            const ColorRGBA background_color = rgb_metadata_background_color(*rgb_data);
            if (!color_region_background_color)
                color_region_background_color = background_color;
            changed |= set_texture_mapping_background_config(volume->config, background_color);
        }
        volume->texture_mapping_color_facets.assign(*rgb_data);
        changed = true;
    }

    if (changed) {
        if (color_region_background_color)
            changed |= set_texture_mapping_background_config(object.config, *color_region_background_color);
        const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
        assign_texture_mapping_zone_preserving_painted_regions(object, texture_mapping_filament_id);
    }
    return changed;
}

static bool append_dialog_vertex_colors_for_volume(const ModelVolume                    &volume,
                                                   std::vector<RGBA>                   &input_colors,
                                                   const ManagedColorDataCreateSource  &source)
{
    const indexed_triangle_set &its = volume.mesh().its;
    if (its.vertices.empty())
        return false;

    const ManagedColorSourceFlags source_flags = managed_color_source_flags(source);
    if (source_flags.use_vertex_colors && volume.imported_vertex_colors_rgba.size() == its.vertices.size()) {
        input_colors.reserve(input_colors.size() + volume.imported_vertex_colors_rgba.size());
        for (const uint32_t packed : volume.imported_vertex_colors_rgba) {
            const ColorRGBA color = unpack_vertex_color_rgba_for_conversion(packed);
            input_colors.emplace_back(RGBA{ color.r(), color.g(), color.b(), color.a() });
        }
        return true;
    }

    const ColorRGBA fallback_color = blank_color_for_managed_target(ManagedColorDataType::ColorRegions);
    const VolumeColorSource rgba_source = build_volume_color_source(volume);
    const ManagedRegionColorSource empty_region_source;
    std::vector<std::array<float, 4>> accumulators(its.vertices.size(), { 0.f, 0.f, 0.f, 0.f });
    std::vector<unsigned int> counts(its.vertices.size(), 0);

    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        for (int corner = 0; corner < 3; ++corner) {
            if (tri[corner] < 0 || size_t(tri[corner]) >= its.vertices.size())
                continue;
            Vec3f barycentric = Vec3f::Zero();
            barycentric[corner] = 1.f;
            const ColorRGBA color = sample_managed_volume_color_source(volume,
                                                                       rgba_source,
                                                                       empty_region_source,
                                                                       tri_idx,
                                                                       its.vertices[size_t(tri[corner])].cast<float>(),
                                                                       barycentric,
                                                                       source_flags.use_rgba,
                                                                       source_flags.use_image_texture,
                                                                       false,
                                                                       false,
                                                                       fallback_color);
            std::array<float, 4> &accumulator = accumulators[size_t(tri[corner])];
            accumulator[0] += color.r();
            accumulator[1] += color.g();
            accumulator[2] += color.b();
            accumulator[3] += color.a();
            ++counts[size_t(tri[corner])];
        }
    }

    input_colors.reserve(input_colors.size() + its.vertices.size());
    for (size_t idx = 0; idx < its.vertices.size(); ++idx) {
        ColorRGBA color = fallback_color;
        if (counts[idx] > 0) {
            const float inv = 1.f / float(counts[idx]);
            color = ColorRGBA(accumulators[idx][0] * inv,
                              accumulators[idx][1] * inv,
                              accumulators[idx][2] * inv,
                              accumulators[idx][3] * inv);
        }
        input_colors.emplace_back(RGBA{ color.r(), color.g(), color.b(), color.a() });
    }
    return true;
}

static std::string encode_managed_region_state_to_hex(unsigned int state)
{
    std::vector<int> nibbles;
    if (state < 3U) {
        nibbles.emplace_back(int(state) << 2);
    } else {
        nibbles.emplace_back(0x0C);
        unsigned int remainder = state - 3U;
        while (remainder >= 15U) {
            nibbles.emplace_back(0x0F);
            remainder -= 15U;
        }
        nibbles.emplace_back(int(remainder));
    }

    std::string encoded;
    encoded.reserve(nibbles.size());
    for (auto it = nibbles.rbegin(); it != nibbles.rend(); ++it) {
        const int nibble = *it;
        encoded.push_back(char(nibble < 10 ? ('0' + nibble) : ('A' + (nibble - 10))));
    }
    return encoded;
}

static unsigned char normalized_region_filament_id(unsigned char filament_id, unsigned char first_extruder_id)
{
    if (filament_id == 0)
        return first_extruder_id == 0 ? 1 : first_extruder_id;
    return filament_id;
}

static bool set_volume_regions_from_vertex_filament_ids(ModelVolume                      &volume,
                                                        const std::vector<unsigned char> &vertex_filament_ids,
                                                        size_t                            offset,
                                                        unsigned char                     first_extruder_id)
{
    const indexed_triangle_set &its = volume.mesh().its;
    if (offset + its.vertices.size() > vertex_filament_ids.size())
        return false;

    first_extruder_id = first_extruder_id == 0 ? 1 : first_extruder_id;
    volume.config.set("extruder", int(first_extruder_id));
    volume.mmu_segmentation_facets.reset();
    volume.mmu_segmentation_facets.reserve(int(its.indices.size()));

    auto filament_id = [&vertex_filament_ids, offset, first_extruder_id](int vertex_idx) {
        return normalized_region_filament_id(vertex_filament_ids[offset + size_t(vertex_idx)], first_extruder_id);
    };
    auto encoded = [](unsigned char id) {
        return encode_managed_region_state_to_hex(unsigned(id));
    };
    auto safe_angle = [](const Vec3f &a, const Vec3f &b) {
        if (a.squaredNorm() <= EPSILON || b.squaredNorm() <= EPSILON)
            return 0.f;
        return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.f, 1.f));
    };

    for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
        const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
            continue;
        if (size_t(tri[0]) >= its.vertices.size() ||
            size_t(tri[1]) >= its.vertices.size() ||
            size_t(tri[2]) >= its.vertices.size())
            continue;

        const unsigned char id0 = filament_id(tri[0]);
        const unsigned char id1 = filament_id(tri[1]);
        const unsigned char id2 = filament_id(tri[2]);
        if (id0 == first_extruder_id && id1 == first_extruder_id && id2 == first_extruder_id)
            continue;

        if (id0 == id1 && id1 == id2) {
            volume.mmu_segmentation_facets.set_triangle_from_string(int(tri_idx), encoded(id0));
            continue;
        }

        const std::string result0 = encoded(id0);
        const std::string result1 = encoded(id1);
        const std::string result2 = encoded(id2);
        if (id0 != id1 && id1 != id2 && id0 != id2) {
            const Vec3f v0 = its.vertices[size_t(tri[0])].cast<float>();
            const Vec3f v1 = its.vertices[size_t(tri[1])].cast<float>();
            const Vec3f v2 = its.vertices[size_t(tri[2])].cast<float>();
            const float angle0 = safe_angle(v1 - v0, v2 - v0);
            const float angle1 = safe_angle(v0 - v1, v2 - v1);
            const float angle2 = PI - angle0 - angle1;
            std::array<float, 3> angles = { angle0, angle1, angle2 };
            int max_angle_vertex_index = 0;
            for (size_t idx = 1; idx < angles.size(); ++idx)
                if (angles[idx] > angles[size_t(max_angle_vertex_index)])
                    max_angle_vertex_index = int(idx);

            if (max_angle_vertex_index == 0)
                volume.mmu_segmentation_facets.set_triangle_from_string(int(tri_idx),
                                                                        result0 + result1 + result2 + (result1 + result2 + "5") + "3");
            else if (max_angle_vertex_index == 1)
                volume.mmu_segmentation_facets.set_triangle_from_string(int(tri_idx),
                                                                        result0 + result1 + result2 + (result0 + result2 + "9") + "3");
            else
                volume.mmu_segmentation_facets.set_triangle_from_string(int(tri_idx),
                                                                        result0 + result1 + result2 + (result1 + result0 + "1") + "3");
            continue;
        }

        if (id0 == id1)
            volume.mmu_segmentation_facets.set_triangle_from_string(int(tri_idx), result2 + result0 + result0 + "A");
        else if (id1 == id2)
            volume.mmu_segmentation_facets.set_triangle_from_string(int(tri_idx), result0 + result1 + result2 + "2");
        else if (id0 == id2)
            volume.mmu_segmentation_facets.set_triangle_from_string(int(tri_idx), result1 + result0 + result0 + "6");
    }

    return true;
}

static bool apply_dialog_vertex_filaments_to_color_regions(ModelObject                      &object,
                                                           const std::vector<unsigned char> &vertex_filament_ids,
                                                           unsigned char                     first_extruder_id)
{
    size_t offset = 0;
    bool changed = false;
    first_extruder_id = first_extruder_id == 0 ? 1 : first_extruder_id;
    object.config.set("extruder", int(first_extruder_id));
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        const size_t vertex_count = volume->mesh().its.vertices.size();
        if (!set_volume_regions_from_vertex_filament_ids(*volume, vertex_filament_ids, offset, first_extruder_id))
            return false;
        offset += vertex_count;
        changed = true;
    }
    return changed && offset == vertex_filament_ids.size();
}

static std::unique_ptr<Model> build_color_region_dialog_preview_model(const ModelObject &object, size_t vertex_count)
{
    TriangleMesh mesh = object.raw_mesh();
    if (mesh.empty() || mesh.its.vertices.size() != vertex_count)
        return nullptr;

    std::unique_ptr<Model> preview_model = std::make_unique<Model>();
    preview_model->add_object(object.name.c_str(), object.input_file.c_str(), std::move(mesh));
    return preview_model;
}

static bool convert_object_to_color_regions(ModelObject &object, const ManagedColorDataCreateSource &source, wxWindow *parent)
{
    if (object_has_color_regions(object))
        return false;

    std::vector<RGBA> input_colors;
    for (const ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        append_dialog_vertex_colors_for_volume(*volume, input_colors, source);
    }
    if (input_colors.empty())
        return false;

    bool is_single_color = true;
    const RGBA first_color = input_colors.front();
    for (const RGBA &color : input_colors) {
        if (color != first_color) {
            is_single_color = false;
            break;
        }
    }

    std::vector<unsigned char> filament_ids;
    unsigned char first_extruder_id = 1;
    std::unique_ptr<Model> preview_model = build_color_region_dialog_preview_model(object, input_colors.size());
    const std::vector<std::string> extruder_colours = wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, false);
    ObjDialogInOut in_out;
    in_out.input_colors = input_colors;
    in_out.is_single_color = is_single_color;
    in_out.filament_ids = filament_ids;
    in_out.first_extruder_id = first_extruder_id;
    in_out.deal_vertex_color = true;
    in_out.model = preview_model.get();
    ObjColorDialog color_dlg(parent, in_out, extruder_colours);
    if (color_dlg.ShowModal() != wxID_OK)
        return false;
    filament_ids = in_out.filament_ids;
    first_extruder_id = in_out.first_extruder_id;
    if (filament_ids.size() != input_colors.size())
        return false;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Create 3mf color regions", UndoRedo::SnapshotType::GizmoAction);
    return apply_dialog_vertex_filaments_to_color_regions(object, filament_ids, first_extruder_id);
}

static bool convert_object_managed_color_data(ModelObject                           &object,
                                              ManagedColorDataType                   type,
                                              const ManagedColorDataCreateSource    &source,
                                              wxWindow                              *parent = nullptr)
{
    switch (type) {
    case ManagedColorDataType::ColorRegions:
        return convert_object_to_color_regions(object, source, parent);
    case ManagedColorDataType::VertexColors:
        return convert_object_to_vertex_colors(object, source);
    case ManagedColorDataType::ImageTexture:
        return convert_object_to_image_texture(object, source);
    case ManagedColorDataType::RgbaData:
        return convert_object_to_rgba_data(object, source);
    }
    return false;
}

static void refresh_managed_color_data_object(GLCanvas3D &parent, ModelObject *object)
{
    if (object == nullptr)
        return;

    parent.update_volumes_colors_by_extruder();
    parent.set_as_dirty();
    parent.request_extra_frame();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        parent.invalidate_texture_mapping_preview_for_object(object_idx);
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
}

class ResizeImageTextureDialog : public wxDialog
{
public:
    ResizeImageTextureDialog(wxWindow* parent, uint32_t width, uint32_t height)
        : wxDialog(parent, wxID_ANY, _L("Resize Texture"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE)
        , m_original_width(width)
        , m_original_height(height)
    {
        wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
        main_sizer->Add(new wxStaticText(this, wxID_ANY,
                                         wxString::Format(_L("Original size: %u x %u"), static_cast<unsigned>(m_original_width),
                                                          static_cast<unsigned>(m_original_height))),
                        0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, FromDIP(16));

        wxFlexGridSizer* grid = new wxFlexGridSizer(2, FromDIP(8), FromDIP(8));
        grid->AddGrowableCol(1, 1);
        const uint32_t max_axis          = std::min<uint32_t>(std::max({uint32_t(8192), m_original_width, m_original_height}),
                                                              uint32_t(std::numeric_limits<int>::max()));
        const uint32_t original_max_axis = std::max(m_original_width, m_original_height);
        const double   max_scale         = original_max_axis > 0 ? double(max_axis) / double(original_max_axis) : 1.0;
        auto scaled_max = [max_scale](uint32_t value) {
            return std::min<uint32_t>(uint32_t(std::numeric_limits<int>::max()),
                                      std::max<uint32_t>(std::max<uint32_t>(1, value),
                                                         uint32_t(std::floor(double(value) * max_scale))));
        };
        m_max_width  = scaled_max(m_original_width);
        m_max_height = scaled_max(m_original_height);
        m_width_spin  = new wxSpinCtrl(this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize(FromDIP(96), -1), wxSP_ARROW_KEYS, 1,
                                       int(m_max_width), int(std::min(m_original_width, m_max_width)));
        m_height_spin = new wxSpinCtrl(this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize(FromDIP(96), -1), wxSP_ARROW_KEYS, 1,
                                       int(m_max_height), int(std::min(m_original_height, m_max_height)));
        grid->Add(new wxStaticText(this, wxID_ANY, _L("Width")), 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(m_width_spin, 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(new wxStaticText(this, wxID_ANY, _L("Height")), 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(m_height_spin, 0, wxALIGN_CENTER_VERTICAL);
        main_sizer->Add(grid, 0, wxEXPAND | wxALL, FromDIP(16));

        m_scale_text = new wxStaticText(this, wxID_ANY, wxString());
        main_sizer->Add(m_scale_text, 0, wxEXPAND | wxLEFT | wxRIGHT, FromDIP(16));
        main_sizer->Add(new wxStaticLine(this), 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, FromDIP(16));

        wxStdDialogButtonSizer* buttons = new wxStdDialogButtonSizer();
        m_ok_button                     = new wxButton(this, wxID_OK, _L("OK"));
        wxButton* cancel_button         = new wxButton(this, wxID_CANCEL, _L("Cancel"));
        buttons->AddButton(m_ok_button);
        buttons->AddButton(cancel_button);
        buttons->Realize();
        main_sizer->Add(buttons, 0, wxEXPAND | wxALL, FromDIP(16));

        SetSizer(main_sizer);
        update_summary();
        Fit();
        SetMinSize(GetSize());
        CenterOnParent();

        m_width_spin->Bind(wxEVT_SPINCTRL, [this](wxSpinEvent&) { update_from_width(); });
        m_width_spin->Bind(wxEVT_TEXT, [this](wxCommandEvent&) { update_from_width(); });
        m_height_spin->Bind(wxEVT_SPINCTRL, [this](wxSpinEvent&) { update_from_height(); });
        m_height_spin->Bind(wxEVT_TEXT, [this](wxCommandEvent&) { update_from_height(); });
    }

    double scale() const { return m_scale; }

private:
    static uint32_t rounded_dimension(double value) { return std::max<uint32_t>(1, uint32_t(std::llround(value))); }

    void set_spin_value(wxSpinCtrl* spin, uint32_t value, uint32_t max_value)
    {
        value = std::clamp<uint32_t>(value, 1, max_value);
        if (spin != nullptr && spin->GetValue() != int(value))
            spin->SetValue(int(value));
    }

    void update_from_width()
    {
        if (m_updating || m_width_spin == nullptr || m_height_spin == nullptr || m_original_width == 0)
            return;

        m_updating           = true;
        const uint32_t width = std::clamp<uint32_t>(uint32_t(std::max(1, m_width_spin->GetValue())), 1, m_max_width);
        set_spin_value(m_width_spin, width, m_max_width);
        m_scale = double(width) / double(m_original_width);
        set_spin_value(m_height_spin, rounded_dimension(double(m_original_height) * m_scale), m_max_height);
        m_updating = false;
        update_summary();
    }

    void update_from_height()
    {
        if (m_updating || m_width_spin == nullptr || m_height_spin == nullptr || m_original_height == 0)
            return;

        m_updating            = true;
        const uint32_t height = std::clamp<uint32_t>(uint32_t(std::max(1, m_height_spin->GetValue())), 1, m_max_height);
        set_spin_value(m_height_spin, height, m_max_height);
        m_scale = double(height) / double(m_original_height);
        set_spin_value(m_width_spin, rounded_dimension(double(m_original_width) * m_scale), m_max_width);
        m_updating = false;
        update_summary();
    }

    void update_summary()
    {
        if (m_width_spin == nullptr || m_height_spin == nullptr)
            return;

        const uint32_t width           = std::clamp<uint32_t>(uint32_t(std::max(1, m_width_spin->GetValue())), 1, m_max_width);
        const uint32_t height          = std::clamp<uint32_t>(uint32_t(std::max(1, m_height_spin->GetValue())), 1, m_max_height);
        const double   dimension_pct   = m_scale * 100.0;
        const double   original_pixels = double(m_original_width) * double(m_original_height);
        const double   resized_pixels  = double(width) * double(height);
        const double   pixel_pct       = original_pixels > 0.0 ? resized_pixels * 100.0 / original_pixels : 100.0;
        if (m_scale_text != nullptr)
            m_scale_text->SetLabel(wxString::Format(_L("New size: %u x %u (%.1f%% dimensions, %.1f%% pixels)"),
                                                    static_cast<unsigned>(width), static_cast<unsigned>(height), dimension_pct, pixel_pct));
        if (m_ok_button != nullptr)
            m_ok_button->Enable(width != m_original_width || height != m_original_height);
        Layout();
        Fit();
    }

    uint32_t      m_original_width  = 0;
    uint32_t      m_original_height = 0;
    uint32_t      m_max_width       = 1;
    uint32_t      m_max_height      = 1;
    wxSpinCtrl*   m_width_spin      = nullptr;
    wxSpinCtrl*   m_height_spin     = nullptr;
    wxStaticText* m_scale_text      = nullptr;
    wxButton*     m_ok_button       = nullptr;
    double        m_scale           = 1.0;
    bool          m_updating        = false;
};

class ColorDataManagementDialog : public wxDialog
{
public:
    ColorDataManagementDialog(wxWindow *parent, GLCanvas3D &canvas, ModelObject *object, std::function<void()> on_object_changed = {})
        : wxDialog(parent,
                   wxID_ANY,
                   _L("Manage Color Data for this object"),
                   wxDefaultPosition,
                   wxDefaultSize,
                   wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER)
        , m_canvas(canvas)
        , m_object(object)
        , m_on_object_changed(std::move(on_object_changed))
    {
        wxBoxSizer *main_sizer = new wxBoxSizer(wxVERTICAL);
        wxBoxSizer *background_sizer = new wxBoxSizer(wxHORIZONTAL);
        background_sizer->Add(new wxStaticText(this, wxID_ANY, _L("Background color")), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, FromDIP(8));
        m_background_picker = new wxPanel(this,
                                          wxID_ANY,
                                          wxDefaultPosition,
                                          wxSize(FromDIP(38), FromDIP(24)),
                                          wxBORDER_SIMPLE);
        m_background_picker->SetMinSize(wxSize(FromDIP(38), FromDIP(24)));
        m_background_picker->SetToolTip(_L("Background color"));
        background_sizer->Add(m_background_picker, 0, wxALIGN_CENTER_VERTICAL);
        m_background_clear = new wxButton(this, wxID_ANY, _L("Clear"));
        background_sizer->Add(m_background_clear, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, FromDIP(8));
        main_sizer->Add(background_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, FromDIP(16));

        wxFlexGridSizer *grid = new wxFlexGridSizer(6, 8, 14);
        grid->AddGrowableCol(3, 1);

        grid->Add(new wxStaticText(this, wxID_ANY, _L("Type")), 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(new wxStaticText(this, wxID_ANY, _L("Status")), 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(new wxStaticText(this, wxID_ANY, _L("Shown")), 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(new wxStaticText(this, wxID_ANY, _L("Size")), 0, wxALIGN_CENTER_VERTICAL);
        grid->AddSpacer(1);
        grid->AddSpacer(1);

        add_row(grid, ManagedColorDataType::ColorRegions, managed_color_data_type_label(ManagedColorDataType::ColorRegions));
        add_row(grid, ManagedColorDataType::VertexColors, managed_color_data_type_label(ManagedColorDataType::VertexColors));
        add_row(grid, ManagedColorDataType::ImageTexture, managed_color_data_type_label(ManagedColorDataType::ImageTexture));
        add_raw_image_texture_info_row(grid);
        add_row(grid, ManagedColorDataType::RgbaData, managed_color_data_type_label(ManagedColorDataType::RgbaData));

        main_sizer->Add(grid, 1, wxEXPAND | wxALL, 16);
        main_sizer->Add(new wxStaticLine(this), 0, wxEXPAND | wxLEFT | wxRIGHT, 16);

        wxStdDialogButtonSizer *buttons = new wxStdDialogButtonSizer();
        wxButton *close_button = new wxButton(this, wxID_CLOSE, _L("Close"));
        buttons->AddButton(close_button);
        buttons->Realize();
        main_sizer->Add(buttons, 0, wxEXPAND | wxALL, 16);

        SetSizer(main_sizer);
        refresh_rows();
        Fit();
        SetMinSize(GetSize());
        CenterOnParent();

        Bind(wxEVT_BUTTON, [this](wxCommandEvent &) { EndModal(wxID_CLOSE); }, wxID_CLOSE);
        m_background_picker->Bind(wxEVT_LEFT_DOWN, [this](wxMouseEvent &) { pick_background_color(); });
        m_background_clear->Bind(wxEVT_BUTTON, [this](wxCommandEvent &) { clear_background_color(); });
    }

private:
    struct Row
    {
        ManagedColorDataType type;
        wxStaticText        *status = nullptr;
        wxStaticText        *preview = nullptr;
        wxStaticText        *size = nullptr;
        wxButton            *clear = nullptr;
        wxButton            *create = nullptr;
        wxButton            *actions = nullptr;
    };

    void add_raw_image_texture_info_row(wxFlexGridSizer *grid)
    {
        wxStaticText *label = new wxStaticText(this, wxID_ANY, _L("(Raw Offset Atlas)"));
        wxStaticText *status = new wxStaticText(this, wxID_ANY, wxString());
        wxStaticText *preview = new wxStaticText(this, wxID_ANY, wxString());
        wxStaticText *mode = new wxStaticText(this, wxID_ANY, wxString());

        grid->Add(label, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, FromDIP(12));
        grid->Add(status, 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(preview, 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(mode, 1, wxEXPAND | wxALIGN_CENTER_VERTICAL);
        grid->AddSpacer(1);
        grid->AddSpacer(1);

        m_raw_image_texture_info_windows = { label, status, preview, mode };
        m_raw_image_texture_status = status;
        m_raw_image_texture_preview = preview;
        m_raw_image_texture_mode = mode;
    }

    void add_row(wxFlexGridSizer *grid, ManagedColorDataType type, const wxString &label)
    {
        wxStaticText *status = new wxStaticText(this, wxID_ANY, wxString());
        wxStaticText *preview = new wxStaticText(this, wxID_ANY, wxString());
        wxStaticText *size = new wxStaticText(this, wxID_ANY, wxString());
        wxButton *clear = new wxButton(this, wxID_ANY, _L("Clear"));
        wxButton *create = new wxButton(this, wxID_ANY, _L("Create From..."));
        wxButton *actions = nullptr;
        wxBoxSizer *size_sizer = new wxBoxSizer(wxHORIZONTAL);
        size_sizer->Add(size, 1, wxALIGN_CENTER_VERTICAL);
        if (type == ManagedColorDataType::ImageTexture || type == ManagedColorDataType::RgbaData) {
            actions = new wxButton(this, wxID_ANY, wxString::FromUTF8("\xE2\x96\xBE"), wxDefaultPosition, wxDefaultSize, wxBU_EXACTFIT);
            actions->SetToolTip(type == ManagedColorDataType::ImageTexture ? _L("Image texture actions") : _L("RGBA data actions"));
            size_sizer->Add(actions, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, FromDIP(6));
        }

        grid->Add(new wxStaticText(this, wxID_ANY, label), 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(status, 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(preview, 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(size_sizer, 1, wxEXPAND | wxALIGN_CENTER_VERTICAL);
        grid->Add(clear, 0, wxALIGN_CENTER_VERTICAL);
        grid->Add(create, 0, wxALIGN_CENTER_VERTICAL);

        clear->Bind(wxEVT_BUTTON, [this, type](wxCommandEvent &) { clear_data(type); });
        create->Bind(wxEVT_BUTTON, [this, type, create](wxCommandEvent &) { show_create_menu(type, create); });
        if (actions != nullptr) {
            actions->Bind(wxEVT_BUTTON, [this, type, actions](wxCommandEvent &) {
                if (type == ManagedColorDataType::ImageTexture)
                    show_image_texture_menu(actions);
                else if (type == ManagedColorDataType::RgbaData)
                    show_rgba_data_menu(actions);
            });
        }
        m_rows.push_back({ type, status, preview, size, clear, create, actions });
    }

    void refresh_rows()
    {
        refresh_background_controls();
        const ManagedColorDataSummary summary = summarize_managed_color_data(m_object);
        const ManagedColorDataPreviewKind preview_kind = managed_color_data_preview_kind(summary);
        for (Row &row : m_rows) {
            const bool has_data = managed_color_data_summary_has_type(summary, row.type);
            row.status->SetLabel(has_data ? _L("Present") : _L("None"));
            row.preview->SetLabel(managed_color_data_preview_marker(managed_color_data_preview_kind_matches_type(preview_kind, row.type)));
            row.size->SetLabel(managed_color_data_size_text(summary, row.type));
            row.clear->Enable(has_data);
            row.create->Enable(m_object != nullptr && !has_data);
            if (row.actions != nullptr)
                row.actions->Enable(m_object != nullptr);
        }
        const bool show_raw_info = summary.has_raw_offset_image_texture;
        if (m_raw_image_texture_status != nullptr)
            m_raw_image_texture_status->SetLabel(show_raw_info ? _L("Present") : wxString());
        if (m_raw_image_texture_preview != nullptr)
            m_raw_image_texture_preview->SetLabel(
                show_raw_info ?
                    managed_color_data_preview_marker(preview_kind == ManagedColorDataPreviewKind::RawOffsetAtlas) :
                    wxString());
        if (m_raw_image_texture_mode != nullptr)
            m_raw_image_texture_mode->SetLabel(show_raw_info ? managed_color_data_raw_offset_mode_text(summary) : wxString());
        for (wxWindow *window : m_raw_image_texture_info_windows)
            if (window != nullptr)
                window->Show(show_raw_info);
        Layout();
        Fit();
    }

    void refresh_background_controls()
    {
        if (m_background_picker != nullptr) {
            m_background_picker->SetBackgroundColour(wx_colour_from_color_rgba(managed_color_data_background_color(m_object)));
            m_background_picker->Refresh();
        }
        if (m_background_clear != nullptr)
            m_background_clear->Enable(managed_color_data_has_background_color(m_object));
    }

    void clear_data(ManagedColorDataType type)
    {
        if (m_object == nullptr || !object_has_managed_color_data(*m_object, type))
            return;

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), clear_snapshot_name(type), UndoRedo::SnapshotType::GizmoAction);
        if (!clear_object_managed_color_data(*m_object, type))
            return;
        if (type == ManagedColorDataType::ColorRegions)
            assign_non_region_color_data_to_image_texture_zone(*m_object);

        refresh_object_after_change();
        refresh_rows();
    }

    void show_create_menu(ManagedColorDataType type, wxButton *button)
    {
        if (m_object == nullptr || object_has_managed_color_data(*m_object, type))
            return;

        wxMenu menu;
        std::vector<std::pair<int, ManagedColorDataCreateSource>> sources;
        const ManagedColorDataSummary summary = summarize_managed_color_data(m_object);
        auto add_item = [&menu, &sources](const wxString &label, const ManagedColorDataCreateSource &source) {
            const int id = wxWindow::NewControlId();
            menu.Append(id, label);
            sources.emplace_back(id, source);
        };

        const std::array<ManagedColorDataType, 4> types = {
            ManagedColorDataType::ColorRegions,
            ManagedColorDataType::VertexColors,
            ManagedColorDataType::ImageTexture,
            ManagedColorDataType::RgbaData
        };
        bool added_data_source = false;
        for (const ManagedColorDataType source_type : types) {
            if (source_type == type || !managed_color_data_summary_has_type(summary, source_type))
                continue;
            add_item(managed_color_data_type_label(source_type),
                     ManagedColorDataCreateSource{ std::optional<ManagedColorDataType>(source_type) });
            if (source_type == ManagedColorDataType::ColorRegions && type != ManagedColorDataType::ColorRegions)
                add_item(_L("3mf color regions (erase regions)"),
                         ManagedColorDataCreateSource{ std::optional<ManagedColorDataType>(source_type), true });
            added_data_source = true;
        }
        if (added_data_source)
            menu.AppendSeparator();
        add_item(_L("Blank Canvas"), ManagedColorDataCreateSource{});

        menu.Bind(wxEVT_COMMAND_MENU_SELECTED, [this, type, sources](wxCommandEvent &event) {
            for (const auto &source : sources) {
                if (source.first == event.GetId()) {
                    create_data(type, source.second);
                    break;
                }
            }
        });
        button->PopupMenu(&menu, wxPoint(0, button->GetSize().GetHeight()));
    }

    void show_image_texture_menu(wxButton *button)
    {
        if (button == nullptr)
            return;

        wxMenu menu;
        wxMenuItem *info = menu.Append(wxWindow::NewControlId(), managed_image_texture_uv_map_info_text(m_object));
        info->Enable(false);
        menu.AppendSeparator();

        const int   resize_id = wxWindow::NewControlId();
        wxMenuItem* resize    = menu.Append(resize_id, _L("Resize Texture"));
        resize->Enable(m_object != nullptr && object_has_image_texture_data(*m_object));

        const int   generate_id = wxWindow::NewControlId();
        wxMenuItem* generate    = menu.Append(generate_id, _L("Generate new UV Map"));
        generate->Enable(m_object != nullptr && object_has_regenerable_image_texture_uvs(*m_object));
        menu.Bind(wxEVT_COMMAND_MENU_SELECTED, [this, resize_id, generate_id](wxCommandEvent& event) {
            if (event.GetId() == resize_id)
                resize_image_texture();
            if (event.GetId() == generate_id)
                generate_new_uv_map();
        });
        button->PopupMenu(&menu, wxPoint(0, button->GetSize().GetHeight()));
    }

    void show_rgba_data_menu(wxButton *button)
    {
        if (button == nullptr)
            return;

        wxMenu menu;
        const int unsplit_id = wxWindow::NewControlId();
        const int simplify_id = wxWindow::NewControlId();
        wxMenuItem *unsplit = menu.Append(unsplit_id, _L("Unsplit once"));
        wxMenuItem *simplify = menu.Append(simplify_id, _L("Simplify to 1 color per triangle"));
        unsplit->Enable(m_object != nullptr && object_has_splittable_rgba_data(*m_object));
        simplify->Enable(m_object != nullptr && object_has_rgba_data(*m_object));
        menu.Bind(wxEVT_COMMAND_MENU_SELECTED, [this, unsplit_id, simplify_id](wxCommandEvent &event) {
            if (event.GetId() == unsplit_id)
                unsplit_rgba_data_once();
            else if (event.GetId() == simplify_id)
                simplify_rgba_data_to_one_color_per_triangle();
        });
        button->PopupMenu(&menu, wxPoint(0, button->GetSize().GetHeight()));
    }

    void resize_image_texture()
    {
        if (m_object == nullptr)
            return;

        const ManagedColorDataSummary summary = summarize_managed_color_data(m_object);
        if (!summary.has_image_texture || summary.max_texture_width == 0 || summary.max_texture_height == 0)
            return;

        ResizeImageTextureDialog dialog(this, summary.max_texture_width, summary.max_texture_height);
        if (dialog.ShowModal() != wxID_OK)
            return;

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Resize image texture", UndoRedo::SnapshotType::GizmoAction);
        if (!resize_object_image_textures(*m_object, dialog.scale()))
            return;

        refresh_object_after_change();
        refresh_rows();
    }

    void generate_new_uv_map()
    {
        if (m_object == nullptr || !object_has_regenerable_image_texture_uvs(*m_object))
            return;

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Generate new image texture UV map", UndoRedo::SnapshotType::GizmoAction);
        if (!regenerate_object_image_texture_uv_maps(*m_object))
            return;

        refresh_object_after_change();
        refresh_rows();
    }

    void unsplit_rgba_data_once()
    {
        if (m_object == nullptr || !object_has_splittable_rgba_data(*m_object))
            return;

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Unsplit RGBA data", UndoRedo::SnapshotType::GizmoAction);
        if (!unsplit_object_rgba_data_once(*m_object))
            return;

        refresh_object_after_change();
        refresh_rows();
    }

    void simplify_rgba_data_to_one_color_per_triangle()
    {
        if (m_object == nullptr || !object_has_rgba_data(*m_object))
            return;

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Simplify RGBA data", UndoRedo::SnapshotType::GizmoAction);
        if (!simplify_object_rgba_data_to_one_color_per_triangle(*m_object))
            return;

        refresh_object_after_change();
        refresh_rows();
    }

    void create_data(ManagedColorDataType type, const ManagedColorDataCreateSource &source)
    {
        if (m_object == nullptr || object_has_managed_color_data(*m_object, type) || (source.type && *source.type == type))
            return;

        if (type == ManagedColorDataType::ColorRegions) {
            if (!convert_object_managed_color_data(*m_object, type, source, this))
                return;

            refresh_object_after_change();
            refresh_rows();
            return;
        }

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), create_snapshot_name(type), UndoRedo::SnapshotType::GizmoAction);
        if (!convert_object_managed_color_data(*m_object, type, source))
            return;
        if (source.erase_color_regions)
            erase_color_regions_and_assign_texture_mapping_zone(*m_object);

        refresh_object_after_change();
        refresh_rows();
    }

    void pick_background_color()
    {
        if (m_object == nullptr)
            return;

        wxColour color;
        if (!show_background_color_dialog(this, wx_colour_from_color_rgba(managed_color_data_background_color(m_object)), color))
            return;

        set_background_color(color);
    }

    void set_background_color(const wxColour &color)
    {
        if (m_object == nullptr || !color.IsOk())
            return;

        const ColorRGBA background = color_rgba_from_wx_colour(color);
        if (managed_color_data_background_color(m_object) == background)
            return;

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Set color data background", UndoRedo::SnapshotType::GizmoAction);
        if (!set_managed_color_data_background_color(*m_object, background))
            return;

        refresh_background_controls();
        refresh_object_after_change();
    }

    void clear_background_color()
    {
        if (m_object == nullptr || !managed_color_data_has_background_color(m_object))
            return;

        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Clear color data background", UndoRedo::SnapshotType::GizmoAction);
        if (!clear_managed_color_data_background_color(*m_object))
            return;

        refresh_background_controls();
        refresh_object_after_change();
    }

    void notify_object_changed()
    {
        if (m_on_object_changed)
            m_on_object_changed();
    }

    void refresh_object_after_change()
    {
        refresh_managed_color_data_object(m_canvas, m_object);
        notify_object_changed();
        m_canvas.set_as_dirty();
        m_canvas.request_extra_frame();
        m_canvas.render();
    }

    const char *clear_snapshot_name(ManagedColorDataType type) const
    {
        switch (type) {
        case ManagedColorDataType::ColorRegions:
            return "Clear 3mf color regions";
        case ManagedColorDataType::VertexColors:
            return "Clear vertex colors";
        case ManagedColorDataType::ImageTexture:
            return "Clear image texture data";
        case ManagedColorDataType::RgbaData:
            return "Clear RGBA data";
        }
        return "Clear color data";
    }

    const char *create_snapshot_name(ManagedColorDataType type) const
    {
        switch (type) {
        case ManagedColorDataType::ColorRegions:
            return "Create 3mf color regions";
        case ManagedColorDataType::VertexColors:
            return "Create vertex colors";
        case ManagedColorDataType::ImageTexture:
            return "Create image texture";
        case ManagedColorDataType::RgbaData:
            return "Create RGBA data";
        }
        return "Create color data";
    }

    GLCanvas3D        &m_canvas;
    ModelObject       *m_object = nullptr;
    std::function<void()> m_on_object_changed;
    wxPanel          *m_background_picker = nullptr;
    wxButton          *m_background_clear = nullptr;
    std::vector<Row>   m_rows;
    std::vector<wxWindow *> m_raw_image_texture_info_windows;
    wxStaticText       *m_raw_image_texture_status = nullptr;
    wxStaticText       *m_raw_image_texture_preview = nullptr;
    wxStaticText       *m_raw_image_texture_mode = nullptr;
};

void open_color_data_management_dialog(wxWindow *parent, GLCanvas3D &canvas, ModelObject *object, std::function<void()> on_object_changed)
{
    if (object == nullptr)
        return;

    ColorDataManagementDialog dialog(parent, canvas, object, std::move(on_object_changed));
    dialog.ShowModal();
}

void GLGizmoMmuSegmentation::init_extruders_data(const std::vector<ColorRGBA> &extruder_colors)
{
    const unsigned int old_selected_filament_id =
        m_selected_extruder_idx < m_display_filament_ids.size() ? m_display_filament_ids[m_selected_extruder_idx] :
        (m_selected_extruder_idx < m_extruders_colors.size() ? unsigned(m_selected_extruder_idx + 1) : 0);

    m_extruders_colors = extruder_colors;
    m_display_filament_ids = get_display_filament_ids(m_extruders_colors.size());

    m_selected_extruder_idx = 0;
    if (!m_display_filament_ids.empty()) {
        auto selected_it = std::find(m_display_filament_ids.begin(), m_display_filament_ids.end(), old_selected_filament_id);
        if (selected_it != m_display_filament_ids.end())
            m_selected_extruder_idx = size_t(std::distance(m_display_filament_ids.begin(), selected_it));
    }

    // keep remap table consistent with current extruder count
    m_extruder_remap.resize(m_display_filament_ids.size());
    for (size_t i = 0; i < m_extruder_remap.size(); ++i)
        m_extruder_remap[i] = i;
}

void GLGizmoMmuSegmentation::init_extruders_data()
{
    init_extruders_data(get_extruders_colors());
}

bool GLGizmoMmuSegmentation::on_init()
{
    // BBS
    m_shortcut_key = WXK_CONTROL_N;

    // FIXME: maybe should be using GUI::shortkey_ctrl_prefix() or equivalent?
    const wxString ctrl  = _L("Ctrl+");
    // FIXME: maybe should be using GUI::shortkey_alt_prefix() or equivalent?
    const wxString alt   = _L("Alt+");
    const wxString shift = _L("Shift+");

    m_desc["clipping_of_view_caption"] = alt + _L("Mouse wheel");
    m_desc["clipping_of_view"]     = _L("Section view");
    m_desc["reset_direction"]     = _L("Reset direction");
    m_desc["cursor_size_caption"]  = ctrl + _L("Mouse wheel");
    m_desc["cursor_size"]          = _L("Pen size");
    m_desc["cursor_type"]          = _L("Pen shape");

    m_desc["paint_caption"]        = _L("Left mouse button");
    m_desc["paint"]                = _L("Paint");
    m_desc["erase_caption"]        = shift + _L("Left mouse button");
    m_desc["erase"]                = _L("Erase");
    m_desc["shortcut_key_caption"] = _L("Key 1~9");
    m_desc["shortcut_key"]         = _L("Choose filament");
    m_desc["edge_detection"]       = _L("Edge detection");
    m_desc["gap_area_caption"]     = ctrl + _L("Mouse wheel");
    m_desc["gap_area"]             = _L("Gap area");
    m_desc["perform"]              = _L("Perform");

    m_desc["remove_all"]           = _L("Erase all painting");
    m_desc["circle"]               = _L("Circle");
    m_desc["sphere"]               = _L("Sphere");
    m_desc["pointer"]              = _L("Triangles");

    m_desc["filaments"]            = _L("Filaments");
    m_desc["tool_type"]            = _L("Tool type");
    m_desc["tool_brush"]           = _L("Brush");
    m_desc["tool_smart_fill"]      = _L("Smart fill");
    m_desc["tool_bucket_fill"]     = _L("Bucket fill");

    m_desc["smart_fill_angle_caption"] = ctrl + _L("Mouse wheel");
    m_desc["smart_fill_angle"]     = _L("Smart fill angle");

    m_desc["height_range_caption"] = ctrl + _L("Mouse wheel");
    m_desc["height_range"]         = _L("Height range");

    //add toggle wire frame hint
    m_desc["toggle_wireframe_caption"]        = alt + shift + _L("Enter");
    m_desc["toggle_wireframe"]                = _L("Toggle Wireframe");

    // Filament remapping descriptions
    m_desc["perform_remap"]                   = _L("Remap filaments");
    m_desc["remap"]                           = _L("Remap");
    m_desc["cancel_remap"]                    = _L("Cancel");

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

void GLGizmoMmuSegmentation::set_render_triangle_slope_uniforms(GLShaderProgram *shader,
                                                                const ModelVolume *model_volume,
                                                                const Matrix3f &normal_matrix) const
{
    if (!m_slope_auto_paint_preview_active || shader == nullptr) {
        GLGizmoPainterBase::set_render_triangle_slope_uniforms(shader, model_volume, normal_matrix);
        return;
    }

    const SlopeAutoPaintSettings &settings = m_slope_auto_paint_preview_settings;
    const int preview_mode = settings.mode == SlopeAutoPaintMode::Bottom ? 2 :
                             settings.mode == SlopeAutoPaintMode::Side   ? 3 :
                                                                           1;
    const float top_z = float(std::cos(Geometry::deg2rad(std::clamp(settings.top_angle_deg, 0.f, SlopeAutoPaintMaxAngleDeg))));
    const float bottom_z = -float(std::cos(Geometry::deg2rad(std::clamp(settings.bottom_angle_deg, 0.f, SlopeAutoPaintMaxAngleDeg))));
    const unsigned int base_filament_id =
        model_volume != nullptr && model_volume->extruder_id() > 0 ? unsigned(model_volume->extruder_id()) : 1u;
    const size_t target_color_idx = extruder_color_index_for_filament_id(settings.target_filament_id, m_extruders_colors.size());
    ColorRGBA highlight_color = m_extruders_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : m_extruders_colors[target_color_idx];
    highlight_color.a(1.f);

    std::array<std::array<unsigned int, 4>, 4> override_mask_bits{};
    if (!settings.override_all) {
        for (const unsigned int filament_id : settings.override_filament_ids) {
            if (filament_id >= SlopePreviewOverrideIdCount)
                continue;
            const unsigned int slot = filament_id / 16u;
            override_mask_bits[size_t(slot / 4u)][size_t(slot % 4u)] |= 1u << (filament_id % 16u);
        }
    }
    std::array<std::array<float, 4>, 4> override_masks{};
    for (size_t group_idx = 0; group_idx < override_masks.size(); ++group_idx)
        for (size_t component_idx = 0; component_idx < override_masks[group_idx].size(); ++component_idx)
            override_masks[group_idx][component_idx] = float(override_mask_bits[group_idx][component_idx]);

    shader->set_uniform("slope.actived", true);
    shader->set_uniform("slope.volume_world_normal_matrix", normal_matrix);
    shader->set_uniform("slope.normal_z", 0.f);
    shader->set_uniform("slope.preview_mode", preview_mode);
    shader->set_uniform("slope.top_z", top_z);
    shader->set_uniform("slope.bottom_z", bottom_z);
    shader->set_uniform("slope.highlight_color", highlight_color);
    shader->set_uniform("slope.override_all", settings.override_all);
    shader->set_uniform("slope.current_state", 0);
    shader->set_uniform("slope.base_state", int(base_filament_id));
    shader->set_uniform("slope.override_mask0", override_masks[0]);
    shader->set_uniform("slope.override_mask1", override_masks[1]);
    shader->set_uniform("slope.override_mask2", override_masks[2]);
    shader->set_uniform("slope.override_mask3", override_masks[3]);
}

bool GLGizmoMmuSegmentation::should_render_triangle_texture_preview() const
{
    return !m_slope_auto_paint_preview_active;
}

void GLGizmoMmuSegmentation::render_slope_auto_paint_overlay()
{
    if (!m_show_slope_auto_paint_overlay)
        return;

    std::vector<unsigned int> filament_ids = m_display_filament_ids;
    if (filament_ids.empty())
        filament_ids.emplace_back(1u);

    std::vector<std::string> filament_labels;
    std::vector<ColorRGBA> filament_colors;
    filament_labels.reserve(filament_ids.size());
    filament_colors.reserve(filament_ids.size());
    for (const unsigned int filament_id : filament_ids) {
        filament_labels.emplace_back(into_u8(slope_auto_paint_filament_label(filament_id)));
        const size_t color_idx = extruder_color_index_for_filament_id(filament_id, m_extruders_colors.size());
        filament_colors.emplace_back(m_extruders_colors.empty() ? ColorRGBA(0.15f, 0.65f, 0.6f, 1.f) : m_extruders_colors[color_idx]);
    }

    if (std::find(filament_ids.begin(), filament_ids.end(), m_slope_auto_paint_settings.target_filament_id) == filament_ids.end())
        m_slope_auto_paint_settings.target_filament_id = filament_ids.front();

    if (!m_slope_auto_paint_overlay_positioned) {
        const Size canvas_size = m_parent.get_canvas_size();
        const float window_width = m_imgui->scaled(22.f);
        ImGui::SetNextWindowPos(ImVec2(std::max(m_imgui->scaled(8.f), float(canvas_size.get_width()) - window_width - m_imgui->scaled(24.f)),
                                       m_parent.get_main_toolbar_height() + m_imgui->scaled(16.f)),
                                ImGuiCond_Always);
        m_slope_auto_paint_overlay_positioned = true;
    }

    m_imgui->push_common_window_style(m_parent.get_scale());
    ImGui::SetNextWindowSize(ImVec2(m_imgui->scaled(22.f), 0.f), ImGuiCond_FirstUseEver);
    bool open = true;
    const int flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
    m_imgui->begin(_L("Paint by slope"), &open, flags);

    bool changed = false;
    const float label_width = m_imgui->scaled(7.5f);
    const float item_width = m_imgui->scaled(11.f);

    std::vector<std::string> modes = {
        into_u8(_L("Upper surfaces")),
        into_u8(_L("Lower surfaces")),
        into_u8(_L("Sides"))
    };
    int mode_idx = m_slope_auto_paint_settings.mode == SlopeAutoPaintMode::Bottom ? 1 :
                   m_slope_auto_paint_settings.mode == SlopeAutoPaintMode::Side   ? 2 :
                                                                                     0;
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered));
    if (m_imgui->combo(_L("Mode"), modes, mode_idx, 0, label_width, item_width)) {
        m_slope_auto_paint_settings.mode = mode_idx == 1 ? SlopeAutoPaintMode::Bottom :
                                           mode_idx == 2 ? SlopeAutoPaintMode::Side :
                                                           SlopeAutoPaintMode::Top;
        changed = true;
    }
    ImGui::PopStyleColor();

    size_t target_idx = 0;
    for (size_t idx = 0; idx < filament_ids.size(); ++idx) {
        if (filament_ids[idx] == m_slope_auto_paint_settings.target_filament_id) {
            target_idx = idx;
            break;
        }
    }
    const size_t old_target_idx = target_idx;
    ImGui::AlignTextToFramePadding();
    m_imgui->text(_L("Paint with"));
    ImGui::SameLine(label_width);
    ImGui::PushItemWidth(item_width);
    render_extruders_combo("##slope_auto_paint_target", filament_labels, filament_colors, target_idx);
    ImGui::PopItemWidth();
    if (target_idx != old_target_idx && target_idx < filament_ids.size()) {
        m_slope_auto_paint_settings.target_filament_id = filament_ids[target_idx];
        changed = true;
    }

    auto render_angle = [this, label_width, item_width](const wxString &label, const char *id, float &angle) {
        bool angle_changed = false;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(label);
        ImGui::SameLine(label_width);
        ImGui::PushItemWidth(item_width);
        angle_changed |= ImGui::SliderFloat(id, &angle, 0.f, SlopeAutoPaintMaxAngleDeg, "%.1f");
        ImGui::PopItemWidth();
        angle = std::clamp(angle, 0.f, SlopeAutoPaintMaxAngleDeg);
        return angle_changed;
    };
    if (m_slope_auto_paint_settings.mode == SlopeAutoPaintMode::Top ||
        m_slope_auto_paint_settings.mode == SlopeAutoPaintMode::Side)
        changed |= render_angle(_L("Angle from top"), "##slope_top_angle", m_slope_auto_paint_settings.top_angle_deg);
    if (m_slope_auto_paint_settings.mode == SlopeAutoPaintMode::Bottom ||
        m_slope_auto_paint_settings.mode == SlopeAutoPaintMode::Side)
        changed |= render_angle(_L("Angle from bottom"), "##slope_bottom_angle", m_slope_auto_paint_settings.bottom_angle_deg);

    ImGui::Separator();
    bool override_all = m_slope_auto_paint_settings.override_all;
    if (m_imgui->bbl_checkbox(_L("All"), override_all)) {
        if (override_all) {
            m_slope_auto_paint_settings.override_all = true;
            m_slope_auto_paint_settings.override_filament_ids.clear();
            changed = true;
        } else {
            override_all = true;
        }
    }

    const float row_height = ImGui::GetTextLineHeightWithSpacing();
    const float override_height = std::min(row_height * (float(filament_ids.size()) + 0.5f), row_height * 6.5f);
    ImGui::BeginChild("##slope_auto_paint_override_scroll", ImVec2(0.f, override_height), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
    for (size_t idx = 0; idx < filament_ids.size(); ++idx) {
        const unsigned int filament_id = filament_ids[idx];
        const bool currently_selected = !m_slope_auto_paint_settings.override_all &&
            std::find(m_slope_auto_paint_settings.override_filament_ids.begin(),
                      m_slope_auto_paint_settings.override_filament_ids.end(),
                      filament_id) != m_slope_auto_paint_settings.override_filament_ids.end();
        bool selected = currently_selected;
        ImGui::PushID(int(filament_id));
        if (ImGui::Checkbox("##override", &selected)) {
            if (selected) {
                if (m_slope_auto_paint_settings.override_all) {
                    m_slope_auto_paint_settings.override_all = false;
                    m_slope_auto_paint_settings.override_filament_ids.clear();
                }
                if (std::find(m_slope_auto_paint_settings.override_filament_ids.begin(),
                              m_slope_auto_paint_settings.override_filament_ids.end(),
                              filament_id) == m_slope_auto_paint_settings.override_filament_ids.end())
                    m_slope_auto_paint_settings.override_filament_ids.emplace_back(filament_id);
            } else {
                auto &override_ids = m_slope_auto_paint_settings.override_filament_ids;
                override_ids.erase(std::remove(override_ids.begin(), override_ids.end(), filament_id), override_ids.end());
                if (override_ids.empty())
                    m_slope_auto_paint_settings.override_all = true;
            }
            changed = true;
        }
        ImGui::SameLine();
        ImGuiColorEditFlags swatch_flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel |
                                           ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip;
        ImGui::ColorButton("##swatch", ImGuiWrapper::to_ImVec4(filament_colors[idx]), swatch_flags,
                           ImVec2(ImGui::GetTextLineHeight() * 1.5f, ImGui::GetTextLineHeight()));
        ImGui::SameLine();
        ImGui::Text("%s", filament_labels[idx].c_str());
        ImGui::PopID();
    }
    ImGui::EndChild();

    if (changed)
        update_slope_auto_paint_preview(m_slope_auto_paint_settings);
    else if (!m_slope_auto_paint_preview_active)
        update_slope_auto_paint_preview(m_slope_auto_paint_settings);

    ImGui::Separator();
    bool close_overlay = false;
    m_imgui->push_confirm_button_style();
    if (m_imgui->bbl_button(_L("Apply"))) {
        Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Paint by slope", UndoRedo::SnapshotType::GizmoAction);
        if (apply_slope_auto_paint(m_slope_auto_paint_settings, false)) {
            update_model_object();
            m_parent.set_as_dirty();
        }
        close_overlay = true;
    }
    m_imgui->pop_confirm_button_style();
    ImGui::SameLine();
    m_imgui->push_cancel_button_style();
    if (m_imgui->bbl_button(_L("Cancel")))
        close_overlay = true;
    m_imgui->pop_cancel_button_style();

    m_imgui->end();
    m_imgui->pop_common_window_style();

    if (!open || close_overlay) {
        m_show_slope_auto_paint_overlay = false;
        m_slope_auto_paint_overlay_positioned = false;
        clear_slope_auto_paint_preview();
    }
}

void GLGizmoMmuSegmentation::open_slope_auto_paint_overlay()
{
    if (m_c->selection_info()->model_object() == nullptr)
        return;

    m_slope_auto_paint_settings = SlopeAutoPaintSettings();
    m_slope_auto_paint_settings.target_filament_id = m_selected_extruder_idx < m_display_filament_ids.size() ?
        m_display_filament_ids[m_selected_extruder_idx] :
        1u;
    if (!m_display_filament_ids.empty() &&
        std::find(m_display_filament_ids.begin(), m_display_filament_ids.end(), m_slope_auto_paint_settings.target_filament_id) ==
            m_display_filament_ids.end())
        m_slope_auto_paint_settings.target_filament_id = m_display_filament_ids.front();

    m_show_slope_auto_paint_overlay = true;
    m_slope_auto_paint_overlay_positioned = false;
    update_slope_auto_paint_preview(m_slope_auto_paint_settings);
}

void GLGizmoMmuSegmentation::update_slope_auto_paint_preview(const SlopeAutoPaintSettings &settings)
{
    ModelObject *mo = m_c->selection_info()->model_object();
    const Selection &selection = m_parent.get_selection();
    m_slope_auto_paint_preview_active = mo != nullptr &&
                                        !m_triangle_selectors.empty() &&
                                        selection.get_instance_idx() >= 0 &&
                                        selection.get_instance_idx() < int(mo->instances.size());
    m_slope_auto_paint_preview_settings = settings;
    m_parent.set_as_dirty();
    m_parent.schedule_extra_frame(0);
}

void GLGizmoMmuSegmentation::clear_slope_auto_paint_preview()
{
    m_slope_auto_paint_preview_active = false;
    m_parent.set_as_dirty();
    m_parent.schedule_extra_frame(0);
}

bool GLGizmoMmuSegmentation::apply_slope_auto_paint(const SlopeAutoPaintSettings &settings, bool preview)
{
    if (preview) {
        update_slope_auto_paint_preview(settings);
        return false;
    }

    ModelObject *mo = m_c->selection_info()->model_object();
    if (mo == nullptr || m_triangle_selectors.empty())
        return false;

    const Selection &selection = m_parent.get_selection();
    if (selection.get_instance_idx() < 0 || selection.get_instance_idx() >= int(mo->instances.size()))
        return false;

    const Transform3d instance_trafo_not_translate = m_parent.get_canvas_type() == GLCanvas3D::CanvasAssembleView ?
        mo->instances[selection.get_instance_idx()]->get_assemble_transformation().get_matrix_no_offset() :
        mo->instances[selection.get_instance_idx()]->get_transformation().get_matrix_no_offset();

    const std::unordered_set<unsigned int> override_ids(settings.override_filament_ids.begin(), settings.override_filament_ids.end());
    bool changed = false;
    int selector_idx = -1;

    for (const ModelVolume *mv : mo->volumes) {
        if (!mv->is_model_part())
            continue;

        ++selector_idx;
        if (selector_idx < 0 || size_t(selector_idx) >= m_triangle_selectors.size() ||
            m_triangle_selectors[size_t(selector_idx)] == nullptr)
            continue;

        const unsigned int base_filament_id = mv->extruder_id() > 0 ? static_cast<unsigned int>(mv->extruder_id()) : 1u;
        TriangleSelectorGUI *selector_gui = m_triangle_selectors[size_t(selector_idx)].get();
        TriangleSelector *selector = selector_gui;

        const Transform3d trafo_matrix_not_translate = instance_trafo_not_translate * mv->get_matrix_no_offset();
        const bool selector_changed =
            selector->apply_state_by_smooth_world_normal(trafo_matrix_not_translate,
                                                         static_cast<EnforcerBlockerType>(settings.target_filament_id),
                                                         [&settings, &override_ids, base_filament_id](
                                                             EnforcerBlockerType current_state, float world_normal_z) {
                                                             if (!slope_auto_paint_matches_normal(settings, world_normal_z))
                                                                 return false;
                                                             const unsigned int current_filament_id = current_state == EnforcerBlockerType::NONE ?
                                                                 base_filament_id :
                                                                 static_cast<unsigned int>(current_state);
                                                             return settings.override_all || override_ids.count(current_filament_id) != 0;
                                                         },
                                                         SlopeAutoPaintMaxEdgeMm,
                                                         SlopeAutoPaintMaxDepth,
                                                         preview);
        selector_gui->request_update_render_data(true);
        changed |= selector_changed;
    }

    return changed;
}

void GLGizmoMmuSegmentation::data_changed(bool is_serializing)
{
    GLGizmoPainterBase::data_changed(is_serializing);
    if (m_state != On || wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() != ptFFF || wxGetApp().extruders_edited_cnt() <= 1)
        return;

    ModelObject* model_object = m_c->selection_info()->model_object();
    const std::vector<ColorRGBA> current_extruder_colors = get_extruders_colors();
    const int prev_extruders_count = int(m_extruders_colors.size());
    const int current_extruders_count = int(current_extruder_colors.size());
    const std::vector<unsigned int> current_display_filament_ids = get_display_filament_ids(current_extruder_colors.size());
    if (prev_extruders_count != current_extruders_count) {
        if (current_extruder_colors.size() > GLGizmoMmuSegmentation::EXTRUDERS_LIMIT)
            show_notification_extruders_limit_exceeded();

        this->init_extruders_data(current_extruder_colors);
        // Reinitialize triangle selectors because of change of extruder count need also change the size of GLIndexedVertexArray
        if (prev_extruders_count != current_extruders_count)
            this->init_model_triangle_selectors();
    }
    else if (current_extruder_colors != m_extruders_colors) {
        this->init_extruders_data(current_extruder_colors);
        this->update_triangle_selectors_colors();
    }
    else if (current_display_filament_ids != m_display_filament_ids) {
        this->init_extruders_data(current_extruder_colors);
    }
    else if (model_object != nullptr && get_extruder_id_for_volumes(*model_object) != m_volumes_extruder_idxs) {
        this->init_model_triangle_selectors();
    }
}

// BBS
bool GLGizmoMmuSegmentation::on_number_key_down(int number)
{
    int extruder_idx = number - 1;
    if (extruder_idx >= 0 && size_t(extruder_idx) < m_display_filament_ids.size())
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
    assert(extruders.size() == extruders_colors.size());

    size_t selection_out = selection_idx;
    // It is necessary to use BeginGroup(). Otherwise, when using SameLine() is called, then other items will be drawn inside the combobox.
    ImGui::BeginGroup();
    ImVec2 combo_pos = ImGui::GetCursorScreenPos();
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered));
    const bool combo_open = ImGui::BeginCombo(label.c_str(), "");
    ImGui::PopStyleColor();
    if (combo_open) {
        for (size_t extruder_idx = 0; extruder_idx < extruders.size(); ++extruder_idx) {
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

void GLGizmoMmuSegmentation::show_tooltip_information(float caption_max, float x, float y)
{
    ImTextureID normal_id = m_parent.get_gizmos_manager().get_icon_texture_id(GLGizmosManager::MENU_ICON_NAME::IC_TOOLBAR_TOOLTIP);
    ImTextureID hover_id  = m_parent.get_gizmos_manager().get_icon_texture_id(GLGizmosManager::MENU_ICON_NAME::IC_TOOLBAR_TOOLTIP_HOVER);

    caption_max += m_imgui->calc_text_size(std::string_view{": "}).x + 15.f;

    float  scale       = m_parent.get_scale();
    ImVec2 button_size = ImVec2(25 * scale, 25 * scale); // ORCA: Use exact resolution will prevent blur on icon
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0}); // ORCA: Dont add padding
    ImGui::ImageButton3(normal_id, hover_id, button_size);

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip2(ImVec2(x, y));
        auto draw_text_with_caption = [this, &caption_max](const wxString &caption, const wxString &text) {
            m_imgui->text_colored(ImGuiWrapper::COL_ACTIVE, caption);
            ImGui::SameLine(caption_max);
            m_imgui->text_colored(ImGuiWrapper::COL_WINDOW_BG, text);
        };

        std::vector<std::string> tip_items;
        switch (m_tool_type) {
            case ToolType::BRUSH:
                tip_items = {"paint", "erase", "cursor_size", "clipping_of_view", "toggle_wireframe"};
                break;
            case ToolType::BUCKET_FILL:
                tip_items = {"paint", "erase", "smart_fill_angle", "clipping_of_view", "toggle_wireframe"};
                break;
            case ToolType::SMART_FILL:
                // TODO:
                break;
            case ToolType::GAP_FILL:
                tip_items = {"gap_area", "toggle_wireframe"};
                break;
            default:
                break;
        }
        for (const auto &t : tip_items) draw_text_with_caption(m_desc.at(t + "_caption") + ": ", m_desc.at(t));
        ImGui::EndTooltip();
    }
    ImGui::PopStyleVar(2);
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
    const size_t total_filament_count = m_extruders_colors.size();
    const unsigned int max_display_filament_id = m_display_filament_ids.empty() ?
        unsigned(std::max<size_t>(total_filament_count, 1)) :
        *std::max_element(m_display_filament_ids.begin(), m_display_filament_ids.end());
    const std::string max_filament_label = std::to_string(std::max<unsigned int>(max_display_filament_id, 1u));
    const ImVec2 max_filament_label_size = ImGui::CalcTextSize(max_filament_label.c_str(), NULL, true);

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
    const float filament_item_width = std::max(empty_button_width, max_filament_label_size.x + m_imgui->scaled(1.4f)) + m_imgui->scaled(1.5f);

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
    size_t n_extruder_colors = std::min(GLGizmoMmuSegmentation::EXTRUDERS_LIMIT, m_display_filament_ids.size());
    for (size_t extruder_idx = 0; extruder_idx < n_extruder_colors; ++extruder_idx) {
        const unsigned int actual_filament_id = m_display_filament_ids[extruder_idx];
        if (actual_filament_id == 0 || actual_filament_id > m_extruders_colors.size())
            continue;
        const ColorRGBA &extruder_color = m_extruders_colors[actual_filament_id - 1];
        ImVec4           color_vec      = ImGuiWrapper::to_ImVec4(extruder_color);
        std::string color_label = std::string("##extruder color ") + std::to_string(extruder_idx);
        std::string item_text = std::to_string(actual_filament_id);
        const ImVec2 label_size = ImGui::CalcTextSize(item_text.c_str(), NULL, true);

        const ImVec2 button_size(max_filament_label_size.x + m_imgui->scaled(0.5f), 0.f);

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

        if (ImGui::IsItemHovered()) {
            if (extruder_idx < 9)
                m_imgui->tooltip(wxString::Format(_L("Shortcut Key %d: Filament %u"),
                                                   int(extruder_idx + 1),
                                                   actual_filament_id),
                                  max_tooltip_width);
            else
                m_imgui->tooltip(wxString::Format(_L("Filament %u"), actual_filament_id), max_tooltip_width);
        }

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

    if (n_extruder_colors > 0) {
        int selected_filament = m_selected_extruder_idx < m_display_filament_ids.size() ?
            int(m_display_filament_ids[m_selected_extruder_idx]) :
            int(m_selected_extruder_idx) + 1;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(_L("Selected filament"));
        ImGui::SameLine();
        ImGui::PushItemWidth(m_imgui->scaled(4.5f));
        if (ImGui::InputInt("##selected_filament", &selected_filament, 1, 10, ImGuiInputTextFlags_CharsDecimal)) {
            m_selected_extruder_idx =
                display_filament_index_for_requested_id(m_display_filament_ids, m_selected_extruder_idx, selected_filament);
        }
        ImGui::SameLine();
        m_imgui->text(wxString::Format(_L("/ %u"), max_display_filament_id));
        ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.1));
    }

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
        ImGui::SameLine(circle_max_width);
        ImGui::PushItemWidth(sliders_width);
        m_imgui->bbl_slider_float_style("##cursor_radius", &m_cursor_radius, CursorRadiusMin, CursorRadiusMax, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + circle_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##cursor_radius_input", &m_cursor_radius, 0.05f, 0.0f, 0.0f, "%.2f");

        ImGui::Separator();
        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(circle_max_width);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + circle_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true); }

    } else if (m_current_tool == ImGui::TriangleButtonIcon) {
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        m_tool_type   = ToolType::BRUSH;

        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(clipping_slider_left);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + clipping_slider_left);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true); }

    } else if (m_current_tool == ImGui::FillButtonIcon) {
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        m_imgui->bbl_checkbox(m_desc["edge_detection"], m_detect_geometry_edge);
        m_tool_type = ToolType::BUCKET_FILL;

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
        ImGui::Separator();
        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(sliders_left_width);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + sliders_left_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true);}

    } else if (m_current_tool == ImGui::HeightRangeIcon) {
        m_tool_type   = ToolType::BRUSH;
        m_cursor_type = TriangleSelector::CursorType::HEIGHT_RANGE;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc["height_range"] + ":");
        ImGui::SameLine(height_max_width);
        ImGui::PushItemWidth(sliders_width);
        std::string format_str = std::string("%.2f") + I18N::translate_utf8("mm", "Heigh range," "Facet in [cursor z, cursor z + height] will be selected.");
        m_imgui->bbl_slider_float_style("##cursor_height", &m_cursor_height, CursorHeightMin, CursorHeightMax, format_str.data(), 1.0f, true);
        ImGui::SameLine(drag_left_width + height_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##cursor_height_input", &m_cursor_height, 0.05f, 0.0f, 0.0f, "%.2f");

        ImGui::Separator();
        if (m_c->object_clipper()->get_position() == 0.f) {
            ImGui::AlignTextToFramePadding();
            m_imgui->text(m_desc.at("clipping_of_view"));
        }
        else {
            if (m_imgui->button(m_desc.at("reset_direction"))) {
                wxGetApp().CallAfter([this]() {
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
                    });
            }
        }

        auto clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::SameLine(height_max_width);
        ImGui::PushItemWidth(sliders_width);
        bool slider_clp_dist = m_imgui->bbl_slider_float_style("##clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.0f, true);
        ImGui::SameLine(drag_left_width + height_max_width);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        bool b_clp_dist_input = ImGui::BBLDragFloat("##clp_dist_input", &clp_dist, 0.05f, 0.0f, 0.0f, "%.2f");

        if (slider_clp_dist || b_clp_dist_input) { m_c->object_clipper()->set_position_by_ratio(clp_dist, true); }
    }
    else if (m_current_tool == ImGui::GapFillIcon) {
        m_tool_type = ToolType::GAP_FILL;
        m_cursor_type = TriangleSelector::CursorType::POINTER;
        ImGui::AlignTextToFramePadding();
        m_imgui->text(m_desc["gap_area"] + ":");
        ImGui::SameLine(gap_area_slider_left);
        ImGui::PushItemWidth(sliders_width);
        std::string format_str = std::string("%.2f") + I18N::translate_utf8("", "Triangle patch area threshold,""triangle patch will be merged to neighbor if its area is less than threshold");
        m_imgui->bbl_slider_float_style("##gap_area", &TriangleSelectorPatch::gap_area, TriangleSelectorPatch::GapAreaMin, TriangleSelectorPatch::GapAreaMax, format_str.data(), 1.0f, true);
        ImGui::SameLine(drag_left_width + gap_area_slider_left);
        ImGui::PushItemWidth(1.5 * slider_icon_width);
        ImGui::BBLDragFloat("##gap_area_input", &TriangleSelectorPatch::gap_area, 0.05f, 0.0f, 0.0f, "%.2f");
    }

    ImGui::Separator();
    if(m_imgui->bbl_checkbox(_L("Vertical"), m_vertical_only)){
        if(m_vertical_only){
            m_horizontal_only = false;
        }
    }
    if(m_imgui->bbl_checkbox(_L("Horizontal"), m_horizontal_only)){
        if(m_horizontal_only){
            m_vertical_only = false;
        }
    }

    ImGui::Separator();

    if (m_imgui->button(_L("Paint by slope"))) {
        if (m_show_slope_auto_paint_overlay) {
            m_show_slope_auto_paint_overlay = false;
            m_slope_auto_paint_overlay_positioned = false;
            clear_slope_auto_paint_preview();
        } else {
            open_slope_auto_paint_overlay();
        }
    }
    if (ImGui::IsItemHovered())
        m_imgui->tooltip(_L("Paint color regions by surface slope with a live preview."), max_tooltip_width);

    ImGui::Separator();

    const bool has_painted_regions = selected_object_has_painted_regions();
    const bool has_image_texture_color_data = selected_object_has_imported_texture_data();
    const bool has_rgba_color_data = selected_object_has_texture_mapping_color_data();
    const bool can_convert_regions_to_vertex_colors =
        has_painted_regions && !has_image_texture_color_data && !has_rgba_color_data;
    const bool can_convert_regions_to_image_texture = has_painted_regions && !has_rgba_color_data;
    const bool can_convert_regions_to_rgba_data = has_painted_regions;

    m_imgui->disabled_begin(!can_convert_regions_to_vertex_colors);
    if (m_imgui->button(_L("Convert regions to vertex colors")))
        convert_selected_regions_to_vertex_colors();
    if (ImGui::IsItemHovered()) {
        if (can_convert_regions_to_vertex_colors)
            m_imgui->tooltip(_L("Convert painted color regions into imported vertex color data, clear the regions, and assign a texture mapping zone."), max_tooltip_width);
        else if (!has_painted_regions)
            m_imgui->tooltip(_L("This object does not have painted color regions."), max_tooltip_width);
        else if (has_rgba_color_data)
            m_imgui->tooltip(_L("This object already has RGBA data, which is shown before vertex colors."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object already has image texture data, which is shown before vertex colors."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    m_imgui->disabled_begin(!can_convert_regions_to_image_texture);
    if (m_imgui->button(_L("Convert regions to Image Texture")))
        convert_selected_regions_to_image_texture();
    if (ImGui::IsItemHovered()) {
        if (can_convert_regions_to_image_texture)
            m_imgui->tooltip(_L("Convert painted color regions into image texture data, clear the regions, and assign a texture mapping zone."),
                             max_tooltip_width);
        else if (!has_painted_regions)
            m_imgui->tooltip(_L("This object does not have painted color regions."), max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object already has RGBA data, which is shown before image texture data."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    m_imgui->disabled_begin(!can_convert_regions_to_rgba_data);
    if (m_imgui->button(_L("Convert regions to RGBA data")))
        convert_selected_regions_to_rgba_data();
    if (ImGui::IsItemHovered()) {
        if (can_convert_regions_to_rgba_data)
            m_imgui->tooltip(_L("Convert painted color regions into RGBA data, clear the regions, and assign a texture mapping zone."),
                             max_tooltip_width);
        else
            m_imgui->tooltip(_L("This object does not have painted color regions."), max_tooltip_width);
    }
    m_imgui->disabled_end();

    // "Convert vertex colors to regions (will erase painting)" button
#if 0
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
#endif


    if (m_imgui->button(m_desc.at("perform_remap"))) {
        m_show_filament_remap_ui = !m_show_filament_remap_ui;
        if (m_show_filament_remap_ui) {
            // reset remap to identity on opening
            m_extruder_remap.resize(m_display_filament_ids.size());
            for (size_t i = 0; i < m_extruder_remap.size(); ++i)
                m_extruder_remap[i] = i;
        }
    }

    // Render filament swap UI if enabled
    if (m_show_filament_remap_ui) {
        ImGui::Separator();
        render_filament_remap_ui(window_width, max_tooltip_width);
    }
    ImGui::Separator();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 10.0f));
    float get_cur_y = ImGui::GetContentRegionMax().y + ImGui::GetFrameHeight() + y;
    show_tooltip_information(caption_max, x, get_cur_y);

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

    render_slope_auto_paint_overlay();
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
        const size_t num_physical = static_cast<size_t>(std::max(wxGetApp().filaments_cnt(), 0));
        size_t       num_total    = num_physical;
        if (wxGetApp().preset_bundle != nullptr)
            num_total = wxGetApp().preset_bundle->texture_mapping_zones.total_filaments(num_physical);

        size_t max_used_state = 0;
        for (const ModelVolume *mv : mo->volumes) {
            if (!mv->is_model_part())
                continue;
            const auto &used_states = mv->mmu_segmentation_facets.get_data().used_states;
            for (size_t state_idx = static_cast<size_t>(EnforcerBlockerType::Extruder1); state_idx < used_states.size(); ++state_idx) {
                if (used_states[state_idx])
                    max_used_state = std::max(max_used_state, state_idx);
            }
        }

        if (max_used_state > num_physical) {
            BOOST_LOG_TRIVIAL(warning) << "GLGizmoMmuSegmentation::update_model_object painted virtual extruder state detected"
                                       << " max_used_state=" << max_used_state
                                       << " physical_filaments=" << num_physical
                                       << " total_filaments=" << num_total;
        }

        const ModelObjectPtrs &mos = wxGetApp().model().objects;
        size_t obj_idx = std::find(mos.begin(), mos.end(), mo) - mos.begin();
        wxGetApp().obj_list()->update_info_items(obj_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(obj_idx, 0);
        m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
    }
}

void GLGizmoMmuSegmentation::init_model_triangle_selectors()
{
    const ModelObject *mo = m_c->selection_info()->model_object();
    m_triangle_selectors.clear();
    m_slope_auto_paint_preview_active = false;
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

        const unsigned int extruder_id       = mv->extruder_id() > 0 ? unsigned(mv->extruder_id()) : 1u;
        const size_t       extruder_color_id = extruder_color_index_for_filament_id(extruder_id, m_extruders_colors.size());
        std::vector<ColorRGBA> ebt_colors;
        ebt_colors.push_back(m_extruders_colors[extruder_color_id]);
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
    if (m_extruders_colors.empty())
        return;

    for (int i = 0; i < m_triangle_selectors.size(); i++) {
        TriangleSelectorPatch* selector = dynamic_cast<TriangleSelectorPatch*>(m_triangle_selectors[i].get());
        int extruder_idx = i < int(m_volumes_extruder_idxs.size()) ? m_volumes_extruder_idxs[i] : 1;
        const unsigned int extruder_id        = extruder_idx > 0 ? unsigned(extruder_idx) : 1u;
        const size_t       extruder_color_idx = extruder_color_index_for_filament_id(extruder_id, m_extruders_colors.size());
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
    const std::vector<ColorRGBA> current_extruder_colors = get_extruders_colors();
    if (int prev_extruders_count = int(m_extruders_colors.size());
        prev_extruders_count != int(current_extruder_colors.size()) || current_extruder_colors != m_extruders_colors)
        this->init_extruders_data(current_extruder_colors);

    this->init_model_triangle_selectors();
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

bool GLGizmoMmuSegmentation::selected_object_has_texture_mapping_color_data() const
{
    const ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return false;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        if (!volume->texture_mapping_color_facets.empty())
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

    std::vector<RGBA> input_colors;
    input_colors.reserve(target_volume->imported_vertex_colors_rgba.size());
    for (const uint32_t packed : target_volume->imported_vertex_colors_rgba) {
        const float r = float((packed >> 24) & 0xFF) / 255.f;
        const float g = float((packed >> 16) & 0xFF) / 255.f;
        const float b = float((packed >> 8) & 0xFF) / 255.f;
        const float a = float(packed & 0xFF) / 255.f;
        input_colors.emplace_back(RGBA{r, g, b, a});
    }

    if (input_colors.empty())
        return;

    bool is_single_color = true;
    const RGBA first_color = input_colors.front();
    for (const RGBA &color : input_colors) {
        if (color != first_color) {
            is_single_color = false;
            break;
        }
    }

    std::vector<unsigned char> filament_ids;
    unsigned char first_extruder_id = 1;
    const std::vector<std::string> extruder_colours = wxGetApp().plater()->get_extruder_colors_from_plater_config(nullptr, false);
    ObjDialogInOut in_out;
    in_out.input_colors = input_colors;
    in_out.is_single_color = is_single_color;
    in_out.filament_ids = filament_ids;
    in_out.first_extruder_id = first_extruder_id;
    in_out.deal_vertex_color = true;
    ObjColorDialog color_dlg(nullptr, in_out, extruder_colours);
    if (color_dlg.ShowModal() != wxID_OK)
        return;
    filament_ids = in_out.filament_ids;
    first_extruder_id = in_out.first_extruder_id;
    if (filament_ids.empty())
        return;

    if (!Model::obj_import_vertex_color_deal_for_object(filament_ids, first_extruder_id, object))
        return;

    update_from_model_object();
    m_parent.set_as_dirty();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        m_parent.invalidate_texture_mapping_preview_for_object(object_idx);
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
            const std::array<int, 3> vertex_indices = { tri[0], tri[1], tri[2] };

            for (size_t corner = 0; corner < 3; ++corner) {
                Vec3f barycentric = Vec3f::Zero();
                barycentric[int(corner)] = 1.f;
                const ColorRGBA color = sample_texture_rgba_for_face_bake(volume->imported_texture_rgba,
                                                                          volume->imported_texture_width,
                                                                          volume->imported_texture_height,
                                                                          uvs[0],
                                                                          uvs[1],
                                                                          uvs[2],
                                                                          barycentric);
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
        clear_imported_texture_raw_atlas(*volume);
        volume->imported_texture_width = 0;
        volume->imported_texture_height = 0;
        volume->uv_map_generator_version = 0;
        baked = true;
    }

    if (!baked)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

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

void GLGizmoMmuSegmentation::convert_selected_object_vertex_colors_to_texture_mapping_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert vertex colors to RGB data", UndoRedo::SnapshotType::GizmoAction);

    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() ||
            its.indices.empty() ||
            volume->imported_vertex_colors_rgba.size() != its.vertices.size())
            continue;

        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size())
                return 0xFFFFFFFFu;

            const auto &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                return 0xFFFFFFFFu;
            if (size_t(tri[0]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[1]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[2]) >= volume->imported_vertex_colors_rgba.size())
                return 0xFFFFFFFFu;

            const ColorRGBA c0 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[0])]);
            const ColorRGBA c1 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[1])]);
            const ColorRGBA c2 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[2])]);
            return pack_vertex_color_rgba(ColorRGBA(c0.r() * barycentric.x() + c1.r() * barycentric.y() + c2.r() * barycentric.z(),
                                                    c0.g() * barycentric.x() + c1.g() * barycentric.y() + c2.g() * barycentric.z(),
                                                    c0.b() * barycentric.x() + c1.b() * barycentric.y() + c2.b() * barycentric.z(),
                                                    c0.a() * barycentric.x() + c1.a() * barycentric.y() + c2.a() * barycentric.z()));
        };

        const float target_edge = std::max(mesh_max_axis_span(its) / 160.f, 0.25f);
        TextureMappingColorSubdivisionDepths subdivision_depths = [target_edge](size_t, const std::array<Vec3f, 3> &vertices) {
            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_edge, 5);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, 5, 0.025f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    assign_texture_mapping_zone_preserving_painted_regions(*object, texture_mapping_filament_id);

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

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

void GLGizmoMmuSegmentation::convert_selected_object_image_texture_to_texture_mapping_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert image texture to RGB data", UndoRedo::SnapshotType::GizmoAction);

    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_bakeable_image_texture_data(volume))
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size() ||
                tri_idx >= volume->imported_texture_uv_valid.size() ||
                volume->imported_texture_uv_valid[tri_idx] == 0)
                return 0xFFFFFFFFu;

            const size_t uv_offset = tri_idx * 6;
            if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
                return 0xFFFFFFFFu;

            const Vec2f uv0(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]);
            const Vec2f uv1(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]);
            const Vec2f uv2(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5]);
            return pack_vertex_color_rgba(sample_texture_rgba_for_face_bake(volume->imported_texture_rgba,
                                                                            volume->imported_texture_width,
                                                                            volume->imported_texture_height,
                                                                            uv0,
                                                                            uv1,
                                                                            uv2,
                                                                            barycentric));
        };

        const int safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 3200000);
        TextureMappingColorSubdivisionDepths subdivision_depths = [volume, safe_max_depth](size_t tri_idx, const std::array<Vec3f, 3> &) {
            const int depth = texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(volume, tri_idx), 8.f, safe_max_depth);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, safe_max_depth, 0.015f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    assign_texture_mapping_zone_preserving_painted_regions(*object, texture_mapping_filament_id);

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

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

void GLGizmoMmuSegmentation::convert_selected_regions_to_vertex_colors()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr || m_triangle_selectors.empty() || !selected_object_has_painted_regions())
        return;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert regions to vertex colors", UndoRedo::SnapshotType::GizmoAction);
    update_model_object();

    ManagedColorDataCreateSource source{ std::optional<ManagedColorDataType>(ManagedColorDataType::ColorRegions) };
    if (!convert_object_to_vertex_colors(*object, source, true))
        return;

    finish_selected_regions_color_data_conversion(*object);
}

void GLGizmoMmuSegmentation::convert_selected_regions_to_image_texture()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr || m_triangle_selectors.empty() || !selected_object_has_painted_regions())
        return;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert regions to image texture", UndoRedo::SnapshotType::GizmoAction);
    update_model_object();

    ManagedColorDataCreateSource source{ std::optional<ManagedColorDataType>(ManagedColorDataType::ColorRegions) };
    if (!convert_object_to_image_texture(*object, source, true))
        return;

    finish_selected_regions_color_data_conversion(*object);
}

void GLGizmoMmuSegmentation::convert_selected_regions_to_rgba_data()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr || m_triangle_selectors.empty() || !selected_object_has_painted_regions())
        return;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert regions to RGBA data", UndoRedo::SnapshotType::GizmoAction);
    update_model_object();

    ManagedColorDataCreateSource source{ std::optional<ManagedColorDataType>(ManagedColorDataType::ColorRegions) };
    if (!convert_object_to_rgba_data(*object, source, true))
        return;

    finish_selected_regions_color_data_conversion(*object);
}

void GLGizmoMmuSegmentation::finish_selected_regions_color_data_conversion(ModelObject &object)
{
    int selector_idx = -1;
    for (ModelVolume *volume : object.volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        ++selector_idx;
        if (selector_idx >= 0 &&
            size_t(selector_idx) < m_triangle_selectors.size() &&
            m_triangle_selectors[size_t(selector_idx)] != nullptr) {
            m_triangle_selectors[size_t(selector_idx)]->reset();
            m_triangle_selectors[size_t(selector_idx)]->request_update_render_data(true);
        }
    }
    erase_color_regions_and_assign_texture_mapping_zone(object);

    update_from_model_object();
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();
    m_parent.request_extra_frame();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), &object) - objects.begin());
    if (object_idx < objects.size()) {
        m_parent.invalidate_texture_mapping_preview_for_object(object_idx);
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
    m_parent.render();
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
        clear_imported_texture_raw_atlas(*volume);
        volume->imported_texture_width = 0;
        volume->imported_texture_height = 0;
        volume->uv_map_generator_version = 0;
        cleared = true;
    }

    if (!cleared)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

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

void GLGizmoMmuSegmentation::clear_selected_object_texture_mapping_color_data()
{
    ModelObject *object = m_c->selection_info()->model_object();
    if (object == nullptr)
        return;

    bool cleared = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Clear RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || volume->texture_mapping_color_facets.empty())
            continue;

        volume->texture_mapping_color_facets.reset();
        cleared = true;
    }

    if (!cleared)
        return;

    for (auto &selector : m_triangle_selectors)
        if (selector != nullptr)
            selector->request_update_render_data(true);

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

PainterGizmoType GLGizmoMmuSegmentation::get_painter_type() const
{
    return PainterGizmoType::MM_SEGMENTATION;
}

// BBS
ColorRGBA GLGizmoMmuSegmentation::get_cursor_hover_color() const
{
    if (m_selected_extruder_idx < m_display_filament_ids.size()) {
        const unsigned int actual_filament_id = m_display_filament_ids[m_selected_extruder_idx];
        if (actual_filament_id >= 1 && actual_filament_id <= m_extruders_colors.size())
            return m_extruders_colors[actual_filament_id - 1];
    }
    return m_extruders_colors.empty() ? ColorRGBA() : m_extruders_colors[0];
}

void GLGizmoMmuSegmentation::on_set_state()
{
    GLGizmoPainterBase::on_set_state();

    if (get_state() == Off) {
        ModelObject* mo = m_c->selection_info()->model_object();
        if (mo) Slic3r::save_object_mesh(*mo);
        m_parent.post_event(SimpleEvent(EVT_GLCANVAS_FORCE_UPDATE));
    }
}

wxString GLGizmoMmuSegmentation::handle_snapshot_action_name(bool shift_down, GLGizmoPainterBase::Button button_down) const
{
    wxString action_name;
    if (shift_down)
        action_name = _L("Remove painted color");
    else {
        const unsigned int filament_id =
            m_selected_extruder_idx < m_display_filament_ids.size() ?
                m_display_filament_ids[m_selected_extruder_idx] :
                unsigned(m_selected_extruder_idx + 1);
        action_name        = GUI::format(_L("Painted using: Filament %1%"), filament_id);
    }
    return action_name;
}

GLGizmoTrueColorPainting::GLGizmoTrueColorPainting(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id)
    : GLGizmoPainterBase(parent, icon_filename, sprite_id)
{
}

GLGizmoTrueColorPainting::~GLGizmoTrueColorPainting()
{
    cancel_rgb_data_preview_conversion();
}

bool GLGizmoTrueColorPainting::on_init()
{
    m_cursor_type = TriangleSelector::CursorType::SPHERE;
    m_tool_type = ToolType::BRUSH;
    m_triangle_splitting_enabled = true;
    m_cursor_radius = 1.f;
    sync_active_color_mode_from_rgb(true);
    return true;
}

void GLGizmoTrueColorPainting::on_opening()
{
    update_selected_object_color_state();
    if (m_color_input_mode == ColorInputMode::FilamentColors)
        sync_active_color_mode_from_rgb(true);
}

void GLGizmoTrueColorPainting::on_shutdown()
{
    cancel_rgb_data_preview_conversion();
    m_color_picker_active = false;
    clear_brush_stroke_points();
    m_preview_rgb_data_volume_ids.clear();
    m_preview_rgb_data_by_volume.clear();
    m_color_picker_source_cache.clear();
    m_background_color_edit_config_snapshot.reset();
    m_parent.use_slope(false);
    m_parent.toggle_model_objects_visibility(true);
}

PainterGizmoType GLGizmoTrueColorPainting::get_painter_type() const
{
    return PainterGizmoType::TRUE_COLOR;
}

std::string GLGizmoTrueColorPainting::on_get_name() const
{
    return _u8L("True Color Painting");
}

bool GLGizmoTrueColorPainting::on_is_selectable() const
{
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF;
}

bool GLGizmoTrueColorPainting::on_is_activable() const
{
    const Selection& selection = m_parent.get_selection();
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF &&
           !selection.is_empty() &&
           (selection.is_single_full_instance() || selection.is_any_volume());
}

ColorRGBA GLGizmoTrueColorPainting::get_cursor_hover_color() const
{
    if (m_color_picker_active) {
        ColorRGBA color;
        if (sample_color_from_model(m_parent.get_local_mouse_position(), color))
            return ColorRGBA(std::clamp(color.r(), 0.f, 1.f),
                             std::clamp(color.g(), 0.f, 1.f),
                             std::clamp(color.b(), 0.f, 1.f),
                             1.f);
    }
    return ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f);
}

ColorRGBA GLGizmoTrueColorPainting::get_cursor_sphere_left_button_color() const
{
    return ColorRGBA(m_rgb_color[0],
                     m_rgb_color[1],
                     m_rgb_color[2],
                     0.15f + 0.35f * std::clamp(m_opacity, 0.f, 1.f));
}

void GLGizmoTrueColorPainting::render_painter_gizmo()
{
    update_rgb_data_preview_conversion();
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;
    if (object->id() != m_selected_color_state_object_id)
        update_selected_object_color_state();

    const Selection& selection = m_parent.get_selection();
    glsafe(::glEnable(GL_BLEND));
    glsafe(::glEnable(GL_DEPTH_TEST));

    render_triangles(selection);
    m_c->object_clipper()->render_cut();
    m_c->instances_hider()->render_cut();
    render_cursor();

    glsafe(::glDisable(GL_BLEND));
}

bool GLGizmoTrueColorPainting::gizmo_event(SLAGizmoEventType action,
                                           const Vec2d& mouse_position,
                                           bool shift_down,
                                           bool alt_down,
                                           bool control_down)
{
    const bool painting_event =
        action == SLAGizmoEventType::LeftDown ||
        action == SLAGizmoEventType::RightDown ||
        action == SLAGizmoEventType::Dragging ||
        action == SLAGizmoEventType::LeftUp ||
        action == SLAGizmoEventType::RightUp ||
        action == SLAGizmoEventType::Moving;
    const ModelObject *object = selected_model_object();
    if (object == nullptr || object->id() != m_selected_color_state_object_id)
        update_selected_object_color_state();
    update_rgb_data_preview_conversion();

    if (m_color_picker_active) {
        if (action == SLAGizmoEventType::LeftDown) {
            pick_color_from_model(mouse_position);
            m_parent.set_as_dirty();
            return true;
        }
        if (action == SLAGizmoEventType::RightDown) {
            m_parent.set_as_dirty();
            return true;
        }
        if (painting_event)
            return true;
    }

    if (rgb_data_preview_conversion_pending_for_selected_object() &&
        action == SLAGizmoEventType::LeftDown) {
        int mesh_id = -1;
        Vec3f hit = Vec3f::Zero();
        size_t facet = 0;
        if (raycast_to_selected_mesh(mouse_position, mesh_id, hit, facet) && mesh_id >= 0) {
            if (alt_down)
                return false;
            show_info(wxGetApp().mainframe,
                      _L("Source model does not have RGBA data. Please wait for colors to be auto-converted into this format before painting"),
                      _L("RGBA Color Conversion"));
            clear_brush_stroke_points();
            m_parent.set_as_dirty();
            return true;
        }
    }

    if (action == SLAGizmoEventType::LeftDown) {
        clear_brush_stroke_points();
        m_brush_stroke_active = !shift_down && !control_down && record_brush_stroke_point(mouse_position);
    } else if (action == SLAGizmoEventType::Dragging && m_brush_stroke_active && !shift_down && !control_down) {
        record_brush_stroke_point(mouse_position);
    } else if (action == SLAGizmoEventType::RightDown || (action == SLAGizmoEventType::Dragging && shift_down)) {
        clear_brush_stroke_points();
    }

    const bool handled = GLGizmoPainterBase::gizmo_event(action, mouse_position, shift_down, alt_down, control_down);
    if (action == SLAGizmoEventType::LeftUp || action == SLAGizmoEventType::RightUp) {
        clear_brush_stroke_points();
        m_brush_stroke_active = false;
    }
    return handled;
}

void GLGizmoTrueColorPainting::init_model_triangle_selectors()
{
    ModelObject *object = selected_model_object();
    m_triangle_selectors.clear();
    if (object == nullptr)
        return;

    bool needs_async_rgb_preview = false;
    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const ColorFacetsAnnotation *preview_rgb_data = nullptr;
        bool render_normal_surface = false;
        if (volume->texture_mapping_color_facets.empty()) {
            preview_rgb_data = preview_rgb_data_for_volume(*volume);
            render_normal_surface = preview_rgb_data == nullptr;
            needs_async_rgb_preview |= preview_rgb_data == nullptr;
        } else {
            preview_rgb_data = &volume->texture_mapping_color_facets;
        }

        const std::vector<ColorRGBA> colors = {
            render_normal_surface ? projection_base_color_for_volume(*volume) : ColorRGBA(1.f, 1.f, 1.f, 0.f),
            ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f)
        };
        m_triangle_selectors.emplace_back(std::make_unique<TriangleSelectorPatch>(volume->mesh(), volume, colors, 0.2f));
        if (TriangleSelectorPatch *patch = dynamic_cast<TriangleSelectorPatch *>(m_triangle_selectors.back().get())) {
            patch->set_none_state_rendered(render_normal_surface);
            patch->set_texture_mapping_color_preview(preview_rgb_data);
            patch->set_texture_preview_needed(preview_rgb_data != nullptr);
            patch->set_texture_preview_opaque(true);
        }
        m_triangle_selectors.back()->set_wireframe_needed(true);
        m_triangle_selectors.back()->request_update_render_data(true);
    }
    if (needs_async_rgb_preview)
        start_rgb_data_preview_conversion(*object);
}

ColorFacetsAnnotation *GLGizmoTrueColorPainting::preview_rgb_data_for_volume(const ModelVolume &volume) const
{
    for (size_t idx = 0; idx < m_preview_rgb_data_volume_ids.size() && idx < m_preview_rgb_data_by_volume.size(); ++idx)
        if (m_preview_rgb_data_volume_ids[idx] == volume.id())
            return m_preview_rgb_data_by_volume[idx].get();
    return nullptr;
}

bool GLGizmoTrueColorPainting::rgb_data_preview_conversion_pending_for_selected_object() const
{
    std::shared_ptr<RgbDataConversionState> state = m_rgb_data_conversion_state;
    if (!state || m_rgb_data_conversion_object_id != m_selected_color_state_object_id)
        return false;

    std::lock_guard<std::mutex> lock(state->mutex);
    return !state->finished && !state->cancel.load(std::memory_order_relaxed);
}

void GLGizmoTrueColorPainting::cancel_rgb_data_preview_conversion()
{
    ++m_rgb_data_conversion_generation;
    std::shared_ptr<RgbDataConversionState> state = m_rgb_data_conversion_state;
    if (state)
        state->cancel.store(true, std::memory_order_relaxed);
    if (m_rgb_data_conversion_thread.joinable())
        m_rgb_data_conversion_thread.join();
    m_rgb_data_conversion_state.reset();
    m_rgb_data_conversion_object_id = ObjectID();
}

void GLGizmoTrueColorPainting::update_rgb_data_preview_conversion()
{
    std::shared_ptr<RgbDataConversionState> state = m_rgb_data_conversion_state;
    if (!state)
        return;

    bool finished = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        finished = state->finished;
    }
    if (!finished)
        return;

    if (m_rgb_data_conversion_thread.joinable())
        m_rgb_data_conversion_thread.join();

    TrueColorRgbDataConversionResult result;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        result = std::move(state->result);
    }
    m_rgb_data_conversion_state.reset();
    m_rgb_data_conversion_object_id = ObjectID();

    ModelObject *object = selected_model_object();
    if (result.canceled || object == nullptr || object->id() != result.object_id)
        return;

    bool applied = false;
    for (size_t idx = 0; idx < result.volume_ids.size() && idx < result.rgb_data.size(); ++idx) {
        if (!result.rgb_data[idx])
            continue;

        bool volume_exists = false;
        bool volume_has_rgb_data = false;
        for (const ModelVolume *volume : object->volumes) {
            if (volume == nullptr || !volume->is_model_part() || volume->id() != result.volume_ids[idx])
                continue;
            volume_exists = true;
            volume_has_rgb_data = !volume->texture_mapping_color_facets.empty();
            break;
        }
        if (!volume_exists || volume_has_rgb_data)
            continue;

        auto found = std::find(m_preview_rgb_data_volume_ids.begin(), m_preview_rgb_data_volume_ids.end(), result.volume_ids[idx]);
        if (found == m_preview_rgb_data_volume_ids.end()) {
            m_preview_rgb_data_volume_ids.emplace_back(result.volume_ids[idx]);
            m_preview_rgb_data_by_volume.emplace_back(std::move(result.rgb_data[idx]));
        } else {
            const size_t preview_idx = size_t(std::distance(m_preview_rgb_data_volume_ids.begin(), found));
            if (preview_idx < m_preview_rgb_data_by_volume.size())
                m_preview_rgb_data_by_volume[preview_idx] = std::move(result.rgb_data[idx]);
        }
        applied = true;
    }

    if (applied) {
        init_model_triangle_selectors();
        m_parent.set_as_dirty();
        m_parent.request_extra_frame();
    }
}

void GLGizmoTrueColorPainting::start_rgb_data_preview_conversion(ModelObject &object)
{
    update_rgb_data_preview_conversion();
    if (rgb_data_preview_conversion_pending_for_selected_object())
        return;

    if (m_rgb_data_conversion_state || m_rgb_data_conversion_thread.joinable())
        cancel_rgb_data_preview_conversion();

    std::vector<TrueColorRgbDataConversionVolumeSnapshot> snapshots;
    snapshots.reserve(object.volumes.size());
    for (const ModelVolume *volume : object.volumes) {
        if (volume == nullptr ||
            !volume->is_model_part() ||
            !volume->texture_mapping_color_facets.empty() ||
            preview_rgb_data_for_volume(*volume) != nullptr)
            continue;

        TrueColorRgbDataConversionVolumeSnapshot snapshot;
        snapshot.volume_id = volume->id();
        snapshot.its = volume->mesh().its;
        snapshot.imported_vertex_colors_rgba.assign(volume->imported_vertex_colors_rgba.begin(), volume->imported_vertex_colors_rgba.end());
        snapshot.imported_texture_uvs_per_face.assign(volume->imported_texture_uvs_per_face.begin(),
                                                      volume->imported_texture_uvs_per_face.end());
        snapshot.imported_texture_uv_valid.assign(volume->imported_texture_uv_valid.begin(), volume->imported_texture_uv_valid.end());
        snapshot.imported_texture_rgba.assign(volume->imported_texture_rgba.begin(), volume->imported_texture_rgba.end());
        snapshot.imported_texture_width = volume->imported_texture_width;
        snapshot.imported_texture_height = volume->imported_texture_height;
        if (!volume->mmu_segmentation_facets.empty()) {
            snapshot.mmu_segmentation_data = volume->mmu_segmentation_facets.get_data();
            snapshot.region_source = std::make_shared<ManagedRegionColorSource>(
                build_managed_region_color_state_source(*volume, snapshot.mmu_segmentation_data.used_states.size()));
        }
        snapshots.emplace_back(std::move(snapshot));
    }
    if (snapshots.empty())
        return;

    std::shared_ptr<RgbDataConversionState> state = std::make_shared<RgbDataConversionState>();
    const ObjectID object_id = object.id();
    const uint64_t generation = ++m_rgb_data_conversion_generation;
    m_rgb_data_conversion_state = state;
    m_rgb_data_conversion_object_id = object_id;
    m_rgb_data_conversion_thread = std::thread([this, state, object_id, generation, snapshots = std::move(snapshots)]() mutable {
        TrueColorRgbDataConversionResult result;
        result.object_id = object_id;
        auto check_cancel = [&state]() {
            if (state->cancel.load(std::memory_order_relaxed))
                throw TrueColorRgbDataConversionCanceledException();
        };

        try {
            for (const TrueColorRgbDataConversionVolumeSnapshot &snapshot : snapshots) {
                check_cancel();
                std::unique_ptr<ColorFacetsAnnotation> rgb_data = ColorFacetsAnnotation::make_temporary();
                if (rgb_data &&
                    build_volume_rgb_data_from_snapshot(snapshot, rgba_data_conversion_fallback_color(), *rgb_data, check_cancel) &&
                    !rgb_data->empty()) {
                    result.volume_ids.emplace_back(snapshot.volume_id);
                    result.rgb_data.emplace_back(std::move(rgb_data));
                }
            }
        } catch (const TrueColorRgbDataConversionCanceledException &) {
            result.canceled = true;
            result.volume_ids.clear();
            result.rgb_data.clear();
        } catch (...) {
            result.canceled = true;
            result.volume_ids.clear();
            result.rgb_data.clear();
        }

        if (state->cancel.load(std::memory_order_relaxed)) {
            result.canceled = true;
            result.volume_ids.clear();
            result.rgb_data.clear();
        }

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->result = std::move(result);
            state->finished = true;
        }

        call_after_if_true_color_painting_active([this, generation]() {
            if (m_rgb_data_conversion_generation != generation)
                return;
            update_rgb_data_preview_conversion();
            m_parent.set_as_dirty();
            m_parent.request_extra_frame();
        });
    });

    m_parent.set_as_dirty();
    m_parent.request_extra_frame();
}

bool GLGizmoTrueColorPainting::record_brush_stroke_point(const Vec2d &mouse_position)
{
    int mesh_id = -1;
    Vec3f hit = Vec3f::Zero();
    size_t facet = 0;
    if (!raycast_to_selected_mesh(mouse_position, mesh_id, hit, facet) || mesh_id < 0)
        return false;

    if (m_brush_stroke_points_by_volume.size() <= size_t(mesh_id))
        m_brush_stroke_points_by_volume.resize(size_t(mesh_id) + 1);

    const ModelObject *object = selected_model_object();
    const ModelVolume *hit_volume = nullptr;
    if (object != nullptr) {
        int model_part_idx = -1;
        for (const ModelVolume *volume : object->volumes) {
            if (volume == nullptr || !volume->is_model_part())
                continue;
            ++model_part_idx;
            if (model_part_idx == mesh_id) {
                hit_volume = volume;
                break;
            }
        }
    }

    Transform3d world_matrix = Transform3d::Identity();
    if (object != nullptr && hit_volume != nullptr) {
        const Selection &selection = m_parent.get_selection();
        world_matrix = projection_world_matrix_for_volume(m_parent, object, hit_volume, selection.get_instance_idx());
    }

    std::vector<Vec3f> &points = m_brush_stroke_points_by_volume[size_t(mesh_id)];
    const float min_spacing = true_color_brush_subdivision_target(m_cursor_radius);
    if (points.empty() ||
        (transform_point(world_matrix, points.back()) - transform_point(world_matrix, hit)).norm() >= min_spacing)
        points.emplace_back(hit);
    return true;
}

void GLGizmoTrueColorPainting::clear_brush_stroke_points()
{
    m_brush_stroke_points_by_volume.clear();
    m_brush_stroke_active = false;
}

void GLGizmoTrueColorPainting::update_triangle_selectors_color()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 1.f, 1.f, 0.f),
        ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f)
    };
    for (std::unique_ptr<TriangleSelectorGUI> &selector : m_triangle_selectors) {
        TriangleSelectorPatch *patch = dynamic_cast<TriangleSelectorPatch *>(selector.get());
        if (patch == nullptr)
            continue;
        patch->set_ebt_colors(colors);
    }
    m_parent.set_as_dirty();
}

void GLGizmoTrueColorPainting::update_from_model_object(bool first_update)
{
    (void)first_update;
    update_rgb_data_preview_conversion();
    update_selected_object_color_state();
    init_model_triangle_selectors();
}

void GLGizmoTrueColorPainting::update_model_object()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    const bool converting_raw_to_rgba = selected_object_has_raw_atlas_texture_data() && !selected_object_has_rgb_data();
    bool updated = false;
    int selector_idx = -1;
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        ++selector_idx;
        if (selector_idx < 0 ||
            size_t(selector_idx) >= m_triangle_selectors.size() ||
            m_triangle_selectors[size_t(selector_idx)] == nullptr)
            continue;

        std::vector<std::vector<TriangleSelector::FacetStateTriangle>> triangles_per_type;
        m_triangle_selectors[size_t(selector_idx)]->get_facet_triangles(triangles_per_type);
        const size_t paint_state = size_t(EnforcerBlockerType::ENFORCER);
        if (triangles_per_type.size() <= paint_state || triangles_per_type[paint_state].empty())
            continue;

        bool initialized_rgb_data = false;
        if (volume->texture_mapping_color_facets.empty()) {
            bool initialized = false;
            if (ColorFacetsAnnotation *preview = preview_rgb_data_for_volume(*volume);
                preview != nullptr && !preview->empty()) {
                volume->texture_mapping_color_facets.assign(*preview);
                initialized = true;
                initialized_rgb_data = true;
            }
            if (!initialized && rgb_data_preview_conversion_pending_for_selected_object()) {
                show_info(wxGetApp().mainframe,
                          _L("Source model does not have RGBA data. Please wait for colors to be auto-converted into this format before painting"),
                          _L("RGBA Color Conversion"));
                clear_brush_stroke_points();
                return;
            }
            if (!initialized)
                initialized_rgb_data = initialize_volume_rgb_data_from_current_surface_color(*volume, rgba_data_conversion_fallback_color());
        }

        const ColorRGBA brush_color(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f);
        const std::vector<Vec3f> empty_brush_stroke_points;
        const std::vector<Vec3f> &brush_stroke_points =
            selector_idx < int(m_brush_stroke_points_by_volume.size()) ?
                m_brush_stroke_points_by_volume[size_t(selector_idx)] :
                empty_brush_stroke_points;
        const Selection &selection = m_parent.get_selection();
        const Transform3d world_matrix =
            projection_world_matrix_for_volume(m_parent, object, volume, selection.get_instance_idx());
        const bool stroke_changed = apply_rgb_stroke_to_volume(*volume,
                                                               triangles_per_type[paint_state],
                                                               brush_color,
                                                               m_brush_hardness,
                                                               m_opacity,
                                                               m_cursor_radius,
                                                               brush_stroke_points,
                                                               world_matrix);
        updated |= initialized_rgb_data || stroke_changed;
        m_triangle_selectors[size_t(selector_idx)]->reset();
        m_triangle_selectors[size_t(selector_idx)]->request_update_render_data(true);
    }

    if (!updated)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    assign_texture_mapping_zone_preserving_painted_regions(*object, texture_mapping_filament_id);

    refresh_selected_object_after_rgb_change(object);
    if (converting_raw_to_rgba && selected_object_has_rgb_data())
        show_raw_offset_data_converted_to_rgba_message();
}

ModelObject *GLGizmoTrueColorPainting::selected_model_object() const
{
    if (m_c == nullptr)
        return nullptr;
    const auto *selection_info = m_c->selection_info();
    return selection_info != nullptr ? selection_info->model_object() : nullptr;
}

void GLGizmoTrueColorPainting::open_color_data_management_dialog()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    cancel_rgb_data_preview_conversion();
    m_preview_rgb_data_volume_ids.clear();
    m_preview_rgb_data_by_volume.clear();
    Slic3r::GUI::open_color_data_management_dialog(wxGetApp().mainframe, m_parent, object, [this]() {
        update_selected_object_color_state();
        init_model_triangle_selectors();
        m_parent.set_as_dirty();
        m_parent.request_extra_frame();
    });
    update_selected_object_color_state();
    init_model_triangle_selectors();
    m_parent.set_as_dirty();
    m_parent.request_extra_frame();
}

void GLGizmoTrueColorPainting::update_selected_object_color_state()
{
    m_selected_has_rgb_data = false;
    m_selected_has_imported_color_data = false;
    m_selected_can_convert_vertex = false;
    m_selected_can_convert_image = false;
    m_selected_has_raw_atlas_texture_data = false;

    const ModelObject *object = selected_model_object();
    const ObjectID previous_object_id = m_selected_color_state_object_id;
    m_selected_color_state_object_id = object != nullptr ? object->id() : ObjectID();
    if (m_selected_color_state_object_id != previous_object_id) {
        cancel_rgb_data_preview_conversion();
        m_preview_rgb_data_volume_ids.clear();
        m_preview_rgb_data_by_volume.clear();
        m_color_picker_source_cache.clear();
        m_background_color_edit_config_snapshot.reset();
    }
    if (object == nullptr)
        return;

    for (const ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        m_selected_has_rgb_data |= !volume->texture_mapping_color_facets.empty();
        m_selected_can_convert_vertex |= !volume->imported_vertex_colors_rgba.empty();
        m_selected_can_convert_image |= model_volume_has_bakeable_image_texture_data(volume);
        m_selected_has_raw_atlas_texture_data |= model_volume_has_raw_atlas_texture_data(volume);
    }
    m_selected_has_imported_color_data = m_selected_can_convert_vertex || m_selected_can_convert_image;
}

bool GLGizmoTrueColorPainting::selected_object_has_rgb_data() const
{
    return m_selected_has_rgb_data;
}

bool GLGizmoTrueColorPainting::selected_object_has_imported_color_data() const
{
    return m_selected_has_imported_color_data;
}

bool GLGizmoTrueColorPainting::selected_object_has_raw_atlas_texture_data() const
{
    return m_selected_has_raw_atlas_texture_data;
}

void GLGizmoTrueColorPainting::initialize_selected_object_rgb_data()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    bool initialized = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Create blank RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;
        initialized |= initialize_volume_rgb_data(*volume, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    }

    if (!initialized)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    assign_texture_mapping_zone_preserving_painted_regions(*object, texture_mapping_filament_id);

    refresh_selected_object_after_rgb_change(object);
}

void GLGizmoTrueColorPainting::convert_selected_object_vertex_colors_to_rgb_data()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert vertex colors to RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() ||
            its.indices.empty() ||
            volume->imported_vertex_colors_rgba.size() != its.vertices.size())
            continue;

        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size())
                return 0xFFFFFFFFu;

            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                return 0xFFFFFFFFu;
            if (size_t(tri[0]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[1]) >= volume->imported_vertex_colors_rgba.size() ||
                size_t(tri[2]) >= volume->imported_vertex_colors_rgba.size())
                return 0xFFFFFFFFu;

            const ColorRGBA c0 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[0])]);
            const ColorRGBA c1 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[1])]);
            const ColorRGBA c2 = unpack_vertex_color_rgba_for_conversion(volume->imported_vertex_colors_rgba[size_t(tri[2])]);
            return pack_vertex_color_rgba(ColorRGBA(c0.r() * barycentric.x() + c1.r() * barycentric.y() + c2.r() * barycentric.z(),
                                                    c0.g() * barycentric.x() + c1.g() * barycentric.y() + c2.g() * barycentric.z(),
                                                    c0.b() * barycentric.x() + c1.b() * barycentric.y() + c2.b() * barycentric.z(),
                                                    c0.a() * barycentric.x() + c1.a() * barycentric.y() + c2.a() * barycentric.z()));
        };

        const float target_edge = std::max(mesh_max_axis_span(its) / 160.f, 0.25f);
        TextureMappingColorSubdivisionDepths subdivision_depths = [target_edge](size_t, const std::array<Vec3f, 3> &vertices) {
            const int depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices), target_edge, 5);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, 5, 0.025f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    assign_texture_mapping_zone_preserving_painted_regions(*object, texture_mapping_filament_id);

    refresh_selected_object_after_rgb_change(object);
}

void GLGizmoTrueColorPainting::convert_selected_object_image_texture_to_rgb_data()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    bool converted = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Convert image texture to RGB data", UndoRedo::SnapshotType::GizmoAction);
    for (ModelVolume *volume : object->volumes) {
        if (volume == nullptr || !volume->is_model_part() || !model_volume_has_bakeable_image_texture_data(volume))
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        TextureMappingColorSampler sampler = [volume, &its](size_t tri_idx, const Vec3f &, const Vec3f &barycentric) {
            if (tri_idx >= its.indices.size() ||
                tri_idx >= volume->imported_texture_uv_valid.size() ||
                volume->imported_texture_uv_valid[tri_idx] == 0)
                return 0xFFFFFFFFu;

            const size_t uv_offset = tri_idx * 6;
            if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
                return 0xFFFFFFFFu;

            const Vec2f uv0(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]);
            const Vec2f uv1(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]);
            const Vec2f uv2(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5]);
            return pack_vertex_color_rgba(sample_texture_rgba_for_face_bake(volume->imported_texture_rgba,
                                                                            volume->imported_texture_width,
                                                                            volume->imported_texture_height,
                                                                            uv0,
                                                                            uv1,
                                                                            uv2,
                                                                            barycentric));
        };

        const int safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 3200000);
        TextureMappingColorSubdivisionDepths subdivision_depths = [volume, safe_max_depth](size_t tri_idx, const std::array<Vec3f, 3> &) {
            const int depth = texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(volume, tri_idx), 8.f, safe_max_depth);
            return std::make_pair(depth, depth);
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume, sampler, safe_max_depth, 0.015f, subdivision_depths);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        converted = true;
    }

    if (!converted)
        return;

    const unsigned int texture_mapping_filament_id = ensure_texture_mapping_zone();
    assign_texture_mapping_zone_preserving_painted_regions(*object, texture_mapping_filament_id);

    refresh_selected_object_after_rgb_change(object);
}

void GLGizmoTrueColorPainting::refresh_selected_object_after_rgb_change(ModelObject *object)
{
    update_selected_object_color_state();
    init_model_triangle_selectors();
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

bool GLGizmoTrueColorPainting::pick_color_from_model(const Vec2d &mouse_position)
{
    ColorRGBA color;
    if (!sample_color_from_model(mouse_position, color))
        return false;

    set_active_color_from_sample(color);
    return true;
}

bool GLGizmoTrueColorPainting::sample_color_from_model(const Vec2d &mouse_position, ColorRGBA &color) const
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;

    int mesh_idx = -1;
    Vec3f hit = Vec3f::Zero();
    size_t tri_idx = 0;
    if (!raycast_to_selected_mesh(mouse_position, mesh_idx, hit, tri_idx) || mesh_idx < 0)
        return false;

    ModelVolume *volume = nullptr;
    int part_idx = -1;
    for (ModelVolume *candidate : object->volumes) {
        if (candidate == nullptr || !candidate->is_model_part())
            continue;
        ++part_idx;
        if (part_idx == mesh_idx) {
            volume = candidate;
            break;
        }
    }
    if (volume == nullptr)
        return false;

    const indexed_triangle_set &its = volume->mesh().its;
    if (tri_idx >= its.indices.size())
        return false;

    const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
    if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
        return false;
    if (size_t(tri[0]) >= its.vertices.size() ||
        size_t(tri[1]) >= its.vertices.size() ||
        size_t(tri[2]) >= its.vertices.size())
        return false;

    Vec3f barycentric = Vec3f::Zero();
    if (!barycentric_weights_for_region_vertex_colors(hit,
                                                      its.vertices[size_t(tri[0])].cast<float>(),
                                                      its.vertices[size_t(tri[1])].cast<float>(),
                                                      its.vertices[size_t(tri[2])].cast<float>(),
                                                      barycentric))
        return false;

    if (!volume->texture_mapping_color_facets.empty()) {
        const ColorPickerVolumeSourceCache &source = cached_volume_color_source(*volume);
        if (std::optional<ColorRGBA> sampled = sample_rgb_color_facets(source.rgb_facets,
                                                                       source.rgb_by_source_triangle,
                                                                       int(tri_idx),
                                                                       hit)) {
            color = sampled->a() <= EPSILON ? managed_color_data_background_color(object) : *sampled;
        } else {
            color = managed_color_data_background_color(object);
        }
        return true;
    }

    const VolumeColorSource source;
    color = sample_volume_color_source(*volume, source, tri_idx, hit, barycentric);
    return true;
}

const GLGizmoTrueColorPainting::ColorPickerVolumeSourceCache &
GLGizmoTrueColorPainting::cached_volume_color_source(const ModelVolume &volume) const
{
    const ObjectID volume_id = volume.id();
    const ObjectBase::Timestamp timestamp = volume.texture_mapping_color_facets.timestamp();
    auto cache_it = std::find_if(m_color_picker_source_cache.begin(),
                                 m_color_picker_source_cache.end(),
                                 [volume_id](const ColorPickerVolumeSourceCache &cache) {
                                     return cache.volume_id == volume_id;
                                 });
    if (cache_it == m_color_picker_source_cache.end()) {
        m_color_picker_source_cache.emplace_back();
        cache_it = m_color_picker_source_cache.end() - 1;
        cache_it->volume_id = volume_id;
    }
    if (cache_it->timestamp != timestamp) {
        cache_it->timestamp = timestamp;
        cache_it->rgb_facets.clear();
        cache_it->rgb_by_source_triangle.clear();
        volume.texture_mapping_color_facets.get_facet_triangles(volume, cache_it->rgb_facets);
        cache_it->rgb_by_source_triangle.reserve(cache_it->rgb_facets.size());
        for (size_t idx = 0; idx < cache_it->rgb_facets.size(); ++idx)
            cache_it->rgb_by_source_triangle[cache_it->rgb_facets[idx].source_triangle].emplace_back(idx);
    }
    return *cache_it;
}

void GLGizmoTrueColorPainting::set_active_color_from_sample(const ColorRGBA &color)
{
    m_rgb_color[0] = std::clamp(color.r(), 0.f, 1.f);
    m_rgb_color[1] = std::clamp(color.g(), 0.f, 1.f);
    m_rgb_color[2] = std::clamp(color.b(), 0.f, 1.f);
    m_rgb_color[3] = 1.f;
    sync_active_color_mode_from_rgb(true);
    update_triangle_selectors_color();
}

void GLGizmoTrueColorPainting::sync_cmy_from_rgb()
{
    m_cmy_color[0] = 1.f - std::clamp(m_rgb_color[0], 0.f, 1.f);
    m_cmy_color[1] = 1.f - std::clamp(m_rgb_color[1], 0.f, 1.f);
    m_cmy_color[2] = 1.f - std::clamp(m_rgb_color[2], 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_rgb_from_cmy()
{
    m_rgb_color[0] = 1.f - std::clamp(m_cmy_color[0], 0.f, 1.f);
    m_rgb_color[1] = 1.f - std::clamp(m_cmy_color[1], 0.f, 1.f);
    m_rgb_color[2] = 1.f - std::clamp(m_cmy_color[2], 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_cmyk_from_rgb()
{
    const float r = std::clamp(m_rgb_color[0], 0.f, 1.f);
    const float g = std::clamp(m_rgb_color[1], 0.f, 1.f);
    const float b = std::clamp(m_rgb_color[2], 0.f, 1.f);
    const float k = 1.f - std::max({ r, g, b });
    m_cmyk_color[3] = std::clamp(k, 0.f, 1.f);
    if (k >= 1.f - EPSILON) {
        m_cmyk_color[0] = 0.f;
        m_cmyk_color[1] = 0.f;
        m_cmyk_color[2] = 0.f;
        return;
    }

    const float denom = 1.f - k;
    m_cmyk_color[0] = std::clamp((1.f - r - k) / denom, 0.f, 1.f);
    m_cmyk_color[1] = std::clamp((1.f - g - k) / denom, 0.f, 1.f);
    m_cmyk_color[2] = std::clamp((1.f - b - k) / denom, 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_rgb_from_cmyk()
{
    const float c = std::clamp(m_cmyk_color[0], 0.f, 1.f);
    const float m = std::clamp(m_cmyk_color[1], 0.f, 1.f);
    const float y = std::clamp(m_cmyk_color[2], 0.f, 1.f);
    const float k = std::clamp(m_cmyk_color[3], 0.f, 1.f);
    m_rgb_color[0] = std::clamp((1.f - c) * (1.f - k), 0.f, 1.f);
    m_rgb_color[1] = std::clamp((1.f - m) * (1.f - k), 0.f, 1.f);
    m_rgb_color[2] = std::clamp((1.f - y) * (1.f - k), 0.f, 1.f);
}

void GLGizmoTrueColorPainting::sync_cmyw_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(0.f, 1.f, 1.f, 1.f),
        ColorRGBA(1.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_cmyw_color.size() && idx < weights.size(); ++idx)
        m_cmyw_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_cmyw()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(0.f, 1.f, 1.f, 1.f),
        ColorRGBA(1.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights(m_cmyw_color.begin(), m_cmyw_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_rgbk_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_rgbk_color.size() && idx < weights.size(); ++idx)
        m_rgbk_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_rgbk()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f)
    };
    const std::vector<float> weights(m_rgbk_color.begin(), m_rgbk_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(0.f, 0.f, 0.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_rgbw_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_rgbw_color.size() && idx < weights.size(); ++idx)
        m_rgbw_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_rgbw()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights(m_rgbw_color.begin(), m_rgbw_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_cmykw_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(0.f, 1.f, 1.f, 1.f),
        ColorRGBA(1.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_cmykw_color.size() && idx < weights.size(); ++idx)
        m_cmykw_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_cmykw()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(0.f, 1.f, 1.f, 1.f),
        ColorRGBA(1.f, 0.f, 1.f, 1.f),
        ColorRGBA(1.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights(m_cmykw_color.begin(), m_cmykw_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_rgbkw_from_rgb()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights = closest_color_mix_weights(colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    for (size_t idx = 0; idx < m_rgbkw_color.size() && idx < weights.size(); ++idx)
        m_rgbkw_color[idx] = weights[idx];
}

void GLGizmoTrueColorPainting::sync_rgb_from_rgbkw()
{
    const std::vector<ColorRGBA> colors = {
        ColorRGBA(1.f, 0.f, 0.f, 1.f),
        ColorRGBA(0.f, 1.f, 0.f, 1.f),
        ColorRGBA(0.f, 0.f, 1.f, 1.f),
        ColorRGBA(0.f, 0.f, 0.f, 1.f),
        ColorRGBA(1.f, 1.f, 1.f, 1.f)
    };
    const std::vector<float> weights(m_rgbkw_color.begin(), m_rgbkw_color.end());
    const ColorRGBA mixed = color_mix_from_weights(colors, weights, ColorRGBA(1.f, 1.f, 1.f, 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_bw_from_rgb()
{
    const float r = std::clamp(m_rgb_color[0], 0.f, 1.f);
    const float g = std::clamp(m_rgb_color[1], 0.f, 1.f);
    const float b = std::clamp(m_rgb_color[2], 0.f, 1.f);
    const float luminance = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.f, 1.f);
    m_bw_color[0] = 1.f - luminance;
    m_bw_color[1] = luminance;
}

void GLGizmoTrueColorPainting::sync_rgb_from_bw()
{
    const float black = std::clamp(m_bw_color[0], 0.f, 1.f);
    const float white = std::clamp(m_bw_color[1], 0.f, 1.f);
    const float sum = black + white;
    const float value = sum <= EPSILON ? 1.f : white / sum;
    m_rgb_color[0] = value;
    m_rgb_color[1] = value;
    m_rgb_color[2] = value;
}

void GLGizmoTrueColorPainting::ensure_filament_mix_colors()
{
    std::vector<ColorRGBA> colors = get_extruders_colors();
    const size_t physical_count = size_t(std::max(wxGetApp().filaments_cnt(), 0));
    if (physical_count > 0 && colors.size() > physical_count)
        colors.resize(physical_count);

    bool changed = colors.size() != m_filament_mix_colors.size();
    if (!changed) {
        for (size_t idx = 0; idx < colors.size(); ++idx) {
            if (colors[idx] != m_filament_mix_colors[idx]) {
                changed = true;
                break;
            }
        }
    }

    if (changed) {
        m_filament_mix_colors = std::move(colors);
        m_filament_mix.clear();
    }

    if (m_filament_mix.size() != m_filament_mix_colors.size())
        sync_filament_mix_from_rgb();
}

void GLGizmoTrueColorPainting::sync_filament_mix_from_rgb()
{
    m_filament_mix = closest_color_mix_weights(m_filament_mix_colors, ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
}

void GLGizmoTrueColorPainting::sync_rgb_from_filament_mix()
{
    const ColorRGBA mixed = color_mix_from_weights(m_filament_mix_colors,
                                                  m_filament_mix,
                                                  ColorRGBA(m_rgb_color[0], m_rgb_color[1], m_rgb_color[2], 1.f));
    m_rgb_color[0] = mixed.r();
    m_rgb_color[1] = mixed.g();
    m_rgb_color[2] = mixed.b();
}

void GLGizmoTrueColorPainting::sync_active_color_mode_from_rgb(bool update_filament_mix)
{
    switch (m_color_input_mode) {
    case ColorInputMode::FilamentColors:
        ensure_filament_mix_colors();
        if (update_filament_mix)
            sync_filament_mix_from_rgb();
        break;
    case ColorInputMode::RGB:
        break;
    case ColorInputMode::CMY:
        sync_cmy_from_rgb();
        break;
    case ColorInputMode::CMYK:
        sync_cmyk_from_rgb();
        break;
    case ColorInputMode::CMYW:
        sync_cmyw_from_rgb();
        break;
    case ColorInputMode::RGBK:
        sync_rgbk_from_rgb();
        break;
    case ColorInputMode::RGBW:
        sync_rgbw_from_rgb();
        break;
    case ColorInputMode::BW:
        sync_bw_from_rgb();
        break;
    case ColorInputMode::CMYKW:
        sync_cmykw_from_rgb();
        break;
    case ColorInputMode::RGBKW:
        sync_rgbkw_from_rgb();
        break;
    }
}

bool GLGizmoTrueColorPainting::render_brush_color_picker(const char *id)
{
    m_imgui->text(_L("Brush color:"));
    ImGui::SameLine();
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    return ImGui::ColorEdit3(id, m_rgb_color.data(), flags);
}

bool GLGizmoTrueColorPainting::render_color_input_mode_combo()
{
    const char *mode_labels[] = {
        "Filament colors",
        "RGB",
        "CMY",
        "CMYK",
        "CMYW",
        "RGBK",
        "RGBW",
        "BW",
        "CMYKW",
        "RGBKW"
    };
    const int mode_count = int(sizeof(mode_labels) / sizeof(mode_labels[0]));
    int mode = std::clamp(int(m_color_input_mode), 0, mode_count - 1);
    bool changed = false;
    if (ImGui::BeginCombo("##true_color_mode", mode_labels[mode])) {
        for (int idx = 0; idx < mode_count; ++idx) {
            const bool selected = idx == mode;
            if (ImGui::Selectable(mode_labels[idx], selected)) {
                mode = idx;
                m_color_input_mode = ColorInputMode(mode);
                sync_active_color_mode_from_rgb(true);
                update_triangle_selectors_color();
                changed = true;
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_rgb_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    changed |= render_brush_color_picker("##true_color_rgb_visual");
    render_color_input_mode_combo();
    changed |= ImGui::SliderFloat("Red", &m_rgb_color[0], 0.f, 1.f, "%.2f");
    changed |= ImGui::SliderFloat("Green", &m_rgb_color[1], 0.f, 1.f, "%.2f");
    changed |= ImGui::SliderFloat("Blue", &m_rgb_color[2], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();
    return changed;
}

bool GLGizmoTrueColorPainting::render_cmy_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_cmy_visual")) {
        sync_cmy_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    bool cmy_changed = false;
    cmy_changed |= ImGui::SliderFloat("Cyan", &m_cmy_color[0], 0.f, 1.f, "%.2f");
    cmy_changed |= ImGui::SliderFloat("Magenta", &m_cmy_color[1], 0.f, 1.f, "%.2f");
    cmy_changed |= ImGui::SliderFloat("Yellow", &m_cmy_color[2], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (cmy_changed) {
        sync_rgb_from_cmy();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_cmyk_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_cmyk_visual")) {
        sync_cmyk_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    bool cmyk_changed = false;
    cmyk_changed |= ImGui::SliderFloat("Cyan", &m_cmyk_color[0], 0.f, 1.f, "%.2f");
    cmyk_changed |= ImGui::SliderFloat("Magenta", &m_cmyk_color[1], 0.f, 1.f, "%.2f");
    cmyk_changed |= ImGui::SliderFloat("Yellow", &m_cmyk_color[2], 0.f, 1.f, "%.2f");
    cmyk_changed |= ImGui::SliderFloat("Key", &m_cmyk_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (cmyk_changed) {
        sync_rgb_from_cmyk();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_cmyw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_cmyw_visual")) {
        sync_cmyw_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    bool cmyw_changed = false;
    cmyw_changed |= ImGui::SliderFloat("Cyan", &m_cmyw_color[0], 0.f, 1.f, "%.2f");
    cmyw_changed |= ImGui::SliderFloat("Magenta", &m_cmyw_color[1], 0.f, 1.f, "%.2f");
    cmyw_changed |= ImGui::SliderFloat("Yellow", &m_cmyw_color[2], 0.f, 1.f, "%.2f");
    cmyw_changed |= ImGui::SliderFloat("White", &m_cmyw_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (cmyw_changed) {
        sync_rgb_from_cmyw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_rgbk_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_rgbk_visual")) {
        sync_rgbk_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    bool rgbk_changed = false;
    rgbk_changed |= ImGui::SliderFloat("Red", &m_rgbk_color[0], 0.f, 1.f, "%.2f");
    rgbk_changed |= ImGui::SliderFloat("Green", &m_rgbk_color[1], 0.f, 1.f, "%.2f");
    rgbk_changed |= ImGui::SliderFloat("Blue", &m_rgbk_color[2], 0.f, 1.f, "%.2f");
    rgbk_changed |= ImGui::SliderFloat("Black", &m_rgbk_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (rgbk_changed) {
        sync_rgb_from_rgbk();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_rgbw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_rgbw_visual")) {
        sync_rgbw_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    bool rgbw_changed = false;
    rgbw_changed |= ImGui::SliderFloat("Red", &m_rgbw_color[0], 0.f, 1.f, "%.2f");
    rgbw_changed |= ImGui::SliderFloat("Green", &m_rgbw_color[1], 0.f, 1.f, "%.2f");
    rgbw_changed |= ImGui::SliderFloat("Blue", &m_rgbw_color[2], 0.f, 1.f, "%.2f");
    rgbw_changed |= ImGui::SliderFloat("White", &m_rgbw_color[3], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (rgbw_changed) {
        sync_rgb_from_rgbw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_cmykw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_cmykw_visual")) {
        sync_cmykw_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    bool cmykw_changed = false;
    cmykw_changed |= ImGui::SliderFloat("Cyan", &m_cmykw_color[0], 0.f, 1.f, "%.2f");
    cmykw_changed |= ImGui::SliderFloat("Magenta", &m_cmykw_color[1], 0.f, 1.f, "%.2f");
    cmykw_changed |= ImGui::SliderFloat("Yellow", &m_cmykw_color[2], 0.f, 1.f, "%.2f");
    cmykw_changed |= ImGui::SliderFloat("Black", &m_cmykw_color[3], 0.f, 1.f, "%.2f");
    cmykw_changed |= ImGui::SliderFloat("White", &m_cmykw_color[4], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (cmykw_changed) {
        sync_rgb_from_cmykw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_rgbkw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_rgbkw_visual")) {
        sync_rgbkw_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    bool rgbkw_changed = false;
    rgbkw_changed |= ImGui::SliderFloat("Red", &m_rgbkw_color[0], 0.f, 1.f, "%.2f");
    rgbkw_changed |= ImGui::SliderFloat("Green", &m_rgbkw_color[1], 0.f, 1.f, "%.2f");
    rgbkw_changed |= ImGui::SliderFloat("Blue", &m_rgbkw_color[2], 0.f, 1.f, "%.2f");
    rgbkw_changed |= ImGui::SliderFloat("Black", &m_rgbkw_color[3], 0.f, 1.f, "%.2f");
    rgbkw_changed |= ImGui::SliderFloat("White", &m_rgbkw_color[4], 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (rgbkw_changed) {
        sync_rgb_from_rgbkw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_bw_picker(float item_width)
{
    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_bw_visual")) {
        sync_bw_from_rgb();
        changed = true;
    }
    render_color_input_mode_combo();

    float value = std::clamp(m_bw_color[1], 0.f, 1.f);
    const bool slider_changed = ImGui::SliderFloat("Black / White", &value, 0.f, 1.f, "%.2f");
    ImGui::PopItemWidth();

    if (slider_changed) {
        m_bw_color[0] = 1.f - value;
        m_bw_color[1] = value;
        sync_rgb_from_bw();
        changed = true;
    }
    return changed;
}

bool GLGizmoTrueColorPainting::render_filament_colors_picker(float item_width)
{
    ensure_filament_mix_colors();
    if (m_filament_mix_colors.empty()) {
        m_imgui->text(_L("No real filaments are available."));
        return false;
    }

    bool changed = false;
    ImGui::PushItemWidth(item_width);
    if (render_brush_color_picker("##true_color_filament_visual"))
        changed = true;
    if (ImGui::IsItemDeactivatedAfterEdit())
        sync_filament_mix_from_rgb();
    render_color_input_mode_combo();

    bool mix_changed = false;
    const float swatch_size = ImGui::GetFrameHeight();
    const ImGuiColorEditFlags swatch_flags = ImGuiColorEditFlags_NoAlpha |
                                             ImGuiColorEditFlags_NoInputs |
                                             ImGuiColorEditFlags_NoLabel |
                                             ImGuiColorEditFlags_NoPicker |
                                             ImGuiColorEditFlags_NoTooltip;
    const auto render_filament_slider = [this, swatch_size, swatch_flags](size_t idx) {
        const std::string label = GUI::format(_u8L("Filament %1%"), idx + 1);
        const bool slider_changed = ImGui::SliderFloat(label.c_str(), &m_filament_mix[idx], 0.f, 1.f, "%.2f");
        ImGui::SameLine();
        const std::string swatch_id = "##true_color_filament_swatch_" + std::to_string(idx);
        const ImVec4 swatch_color = ImGuiWrapper::to_ImVec4(m_filament_mix_colors[idx]);
        ImGui::ColorButton(swatch_id.c_str(), swatch_color, swatch_flags, ImVec2(swatch_size, swatch_size));
        return slider_changed;
    };

    const size_t visible_filament_rows = 6;
    const bool scrollable = m_filament_mix.size() > visible_filament_rows;
    const std::string max_label = GUI::format(_u8L("Filament %1%"), m_filament_mix.size());
    const float list_width = item_width +
                             ImGui::GetStyle().ItemInnerSpacing.x +
                             ImGui::CalcTextSize(max_label.c_str(), nullptr, true).x +
                             ImGui::GetStyle().ItemSpacing.x +
                             swatch_size +
                             (scrollable ? ImGui::GetStyle().ScrollbarSize : 0.f);
    if (scrollable)
        ImGui::BeginChild("##true_color_filament_mix_scroll",
                          ImVec2(list_width, ImGui::GetFrameHeightWithSpacing() * float(visible_filament_rows)),
                          false,
                          ImGuiWindowFlags_AlwaysVerticalScrollbar);
    for (size_t idx = 0; idx < m_filament_mix.size(); ++idx) {
        ImGui::PushItemWidth(item_width);
        mix_changed |= render_filament_slider(idx);
        ImGui::PopItemWidth();
    }
    if (scrollable)
        ImGui::EndChild();
    ImGui::PopItemWidth();

    if (mix_changed) {
        sync_rgb_from_filament_mix();
        changed = true;
    }
    return changed;
}

void GLGizmoTrueColorPainting::render_background_color_picker(float max_tooltip_width)
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    m_imgui->text(_L("Background color"));
    ImGui::SameLine();
    const ColorRGBA background = managed_color_data_background_color(object);
    float background_vec[3] = {
        std::clamp(background.r(), 0.f, 1.f),
        std::clamp(background.g(), 0.f, 1.f),
        std::clamp(background.b(), 0.f, 1.f)
    };
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_DisplayRGB |
                                ImGuiColorEditFlags_InputRGB |
                                ImGuiColorEditFlags_NoInputs;
    ImGui::PushItemWidth(m_imgui->scaled(8.f));
    const bool changed = ImGui::ColorEdit3("##true_color_background_color", background_vec, flags);
    ImGui::PopItemWidth();
    if (changed) {
        if (!m_background_color_edit_config_snapshot) {
            Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Set color data background", UndoRedo::SnapshotType::GizmoAction);
            m_background_color_edit_config_snapshot =
                std::make_unique<TextureMappingBackgroundConfigSnapshot>(snapshot_texture_mapping_background_config(*object));
        }
        preview_texture_mapping_background_config(*object, ColorRGBA(background_vec[0], background_vec[1], background_vec[2], 1.f));
        refresh_managed_color_data_object(m_parent, object);
        m_parent.render();
    }
    if (ImGui::IsItemHovered())
        m_imgui->tooltip(_L("Background color"), max_tooltip_width);

    if (ImGui::IsItemDeactivatedAfterEdit() && m_background_color_edit_config_snapshot) {
        const ColorRGBA selected_background(background_vec[0], background_vec[1], background_vec[2], 1.f);
        restore_texture_mapping_background_config(*object, *m_background_color_edit_config_snapshot);
        m_background_color_edit_config_snapshot.reset();
        set_managed_color_data_background_color(*object, selected_background);
        refresh_selected_object_after_background_color_change(object);
    }

    ImGui::SameLine();
    const bool has_background_color = managed_color_data_has_background_color(object);
    m_imgui->disabled_begin(!has_background_color);
    if (m_imgui->button(_L("Clear")))
        clear_selected_object_background_color();
    m_imgui->disabled_end();
}

void GLGizmoTrueColorPainting::clear_selected_object_background_color()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr || !managed_color_data_has_background_color(object))
        return;

    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Clear color data background", UndoRedo::SnapshotType::GizmoAction);
    if (!clear_managed_color_data_background_color(*object))
        return;

    m_background_color_edit_config_snapshot.reset();
    refresh_selected_object_after_background_color_change(object);
}

void GLGizmoTrueColorPainting::refresh_selected_object_after_background_color_change(ModelObject *object)
{
    if (object == nullptr)
        return;

    cancel_rgb_data_preview_conversion();
    m_preview_rgb_data_volume_ids.clear();
    m_preview_rgb_data_by_volume.clear();
    m_color_picker_source_cache.clear();
    update_selected_object_color_state();
    init_model_triangle_selectors();
    refresh_managed_color_data_object(m_parent, object);
    m_parent.render();
}

void GLGizmoTrueColorPainting::on_render_input_window(float x, float y, float bottom_limit)
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;
    if (object->id() != m_selected_color_state_object_id)
        update_selected_object_color_state();
    update_rgb_data_preview_conversion();

    const float approx_height = m_imgui->scaled(22.0f);
    y = std::min(y, bottom_limit - approx_height);
    GizmoImguiSetNextWIndowPos(x, y, ImGuiCond_Always);

    ImGuiWrapper::push_toolbar_style(m_parent.get_scale());
    GizmoImguiBegin(get_name(), ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                                    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    const float slider_width = m_imgui->scaled(8.f);
    const float max_tooltip_width = ImGui::GetFontSize() * 20.f;
    if (rgb_data_preview_conversion_pending_for_selected_object())
        m_imgui->warning_text(_L("Generating RGBA color conversion... Please wait"), m_imgui->scaled(RawOffsetDataWarningWrapEm));

    if (m_imgui->button(_L("Manage Color Data for this object")))
        open_color_data_management_dialog();

    if (selected_object_has_raw_atlas_texture_data() && !selected_object_has_rgb_data())
        m_imgui->warning_text(raw_offset_data_rgba_conversion_warning_text(), m_imgui->scaled(RawOffsetDataWarningWrapEm));

    bool color_changed = false;
    switch (m_color_input_mode) {
    case ColorInputMode::FilamentColors:
        color_changed = render_filament_colors_picker(slider_width);
        break;
    case ColorInputMode::RGB:
        color_changed = render_rgb_picker(slider_width);
        break;
    case ColorInputMode::CMY:
        color_changed = render_cmy_picker(slider_width);
        break;
    case ColorInputMode::CMYK:
        color_changed = render_cmyk_picker(slider_width);
        break;
    case ColorInputMode::CMYW:
        color_changed = render_cmyw_picker(slider_width);
        break;
    case ColorInputMode::RGBK:
        color_changed = render_rgbk_picker(slider_width);
        break;
    case ColorInputMode::RGBW:
        color_changed = render_rgbw_picker(slider_width);
        break;
    case ColorInputMode::BW:
        color_changed = render_bw_picker(slider_width);
        break;
    case ColorInputMode::CMYKW:
        color_changed = render_cmykw_picker(slider_width);
        break;
    case ColorInputMode::RGBKW:
        color_changed = render_rgbkw_picker(slider_width);
        break;
    }
    if (color_changed)
        update_triangle_selectors_color();

    const wxString picker_label = m_color_picker_active ? _L("Stop picking color from model") : _L("Pick color from model");
    if (m_imgui->button(picker_label)) {
        m_color_picker_active = !m_color_picker_active;
        m_parent.set_as_dirty();
    }

    ImGui::Separator();
    m_imgui->text(_L("Pen size"));
    ImGui::PushItemWidth(slider_width);
    m_imgui->bbl_slider_float_style("##true_color_cursor_radius", &m_cursor_radius, CursorRadiusMin, CursorRadiusMax, "%.2f", 1.f, true);
    ImGui::PopItemWidth();

    float softness_pct = (1.f - m_brush_hardness) * 100.f;
    m_imgui->text(_L("Brush softness"));
    ImGui::PushItemWidth(slider_width);
    if (m_imgui->bbl_slider_float_style("##true_color_softness", &softness_pct, 0.f, 100.f, "%.0f%%", 1.f, true))
        m_brush_hardness = 1.f - std::clamp(softness_pct / 100.f, 0.f, 1.f);
    ImGui::PopItemWidth();

    float opacity_pct = m_opacity * 100.f;
    m_imgui->text(_L("Opacity"));
    ImGui::PushItemWidth(slider_width);
    if (m_imgui->bbl_slider_float_style("##true_color_opacity", &opacity_pct, 0.f, 100.f, "%.0f%%", 1.f, true)) {
        m_opacity = std::clamp(opacity_pct / 100.f, 0.f, 1.f);
        m_parent.set_as_dirty();
    }
    ImGui::PopItemWidth();

    ImGui::Separator();
    if (m_c->object_clipper()->get_position() == 0.f) {
        m_imgui->text(_L("Section view"));
    } else if (m_imgui->button(_L("Reset direction"))) {
        wxGetApp().CallAfter([this]() {
            m_c->object_clipper()->set_position_by_ratio(-1., false);
        });
    }

    float clp_dist = float(m_c->object_clipper()->get_position());
    ImGui::PushItemWidth(slider_width);
    if (m_imgui->bbl_slider_float_style("##true_color_clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.f, true))
        m_c->object_clipper()->set_position_by_ratio(clp_dist, true);
    ImGui::PopItemWidth();

    if (ImGui::IsItemHovered())
        m_imgui->tooltip(_L("Section view"), max_tooltip_width);

    ImGui::Separator();
    render_background_color_picker(max_tooltip_width);

    GizmoImguiEnd();
    ImGuiWrapper::pop_toolbar_style();
}

wxString GLGizmoTrueColorPainting::handle_snapshot_action_name(bool shift_down, GLGizmoPainterBase::Button button_down) const
{
    (void)shift_down;
    (void)button_down;
    return _L("Paint RGB color");
}

GLGizmoImageProjection::GLGizmoImageProjection(GLCanvas3D& parent, const std::string& icon_filename, unsigned int sprite_id)
    : GLGizmoBase(parent, icon_filename, sprite_id)
{
}

bool GLGizmoImageProjection::on_init()
{
    return true;
}

void GLGizmoImageProjection::on_render()
{
    if (m_c != nullptr && m_c->object_clipper() != nullptr)
        m_c->object_clipper()->render_cut();
    if (m_c != nullptr && m_c->instances_hider() != nullptr)
        m_c->instances_hider()->render_cut();
}

std::string GLGizmoImageProjection::on_get_name() const
{
    return _u8L("Project image to model surface");
}

void GLGizmoImageProjection::on_set_state()
{
    if (get_state() == On) {
        m_parent.enable_picking(false);
        m_show_overlay = true;
        m_projection_mode_initialized = false;
    } else if (get_state() == Off) {
        m_parent.enable_picking(true);
        m_parent.toggle_model_objects_visibility(true);
    }
}

bool GLGizmoImageProjection::on_is_selectable() const
{
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF;
}

bool GLGizmoImageProjection::on_is_activable() const
{
    const Selection& selection = m_parent.get_selection();
    return wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptFFF &&
           !selection.is_empty() &&
           (selection.is_single_full_instance() || selection.is_any_volume());
}

CommonGizmosDataID GLGizmoImageProjection::on_get_requirements() const
{
    return CommonGizmosDataID(int(CommonGizmosDataID::SelectionInfo) |
                              int(CommonGizmosDataID::InstancesHider) |
                              int(CommonGizmosDataID::ObjectClipper));
}

bool GLGizmoImageProjection::load_projection_image()
{
    m_image_error.clear();
    m_raw_atlas = {};
    wxFileDialog dialog(wxGetApp().mainframe,
                        _L("Load projection image"),
                        "",
                        "",
                        _L("Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp|All files (*.*)|*.*"),
                        wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    if (dialog.ShowModal() != wxID_OK)
        return false;

    wxImage image(dialog.GetPath(), wxBITMAP_TYPE_ANY);
    std::vector<uint8_t> rgba;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!wx_image_to_rgba(image, rgba, width, height)) {
        m_image_error = _u8L("Unable to load the selected image.");
        return false;
    }

    ImageMapRawFilamentOffsetAtlas raw_atlas;
    std::string raw_atlas_error;
    const bool loaded_raw_atlas =
        decode_image_map_raw_filament_offset_atlas(rgba, width, height, raw_atlas, &raw_atlas_error);
    if (loaded_raw_atlas) {
        ModelObject *object = selected_model_object();
        RawAtlasProjectionLayout raw_layout;
        if (object == nullptr || !raw_atlas_projection_layout_for_object(*object, raw_atlas, raw_layout, &raw_atlas_error)) {
            m_image_path.clear();
            m_image_rgba.clear();
            m_image_width = 0;
            m_image_height = 0;
            m_raw_atlas = {};
            m_overlay_texture.reset();
            m_overlay_texture_dirty = false;
            m_show_overlay = false;
            m_image_error = raw_atlas_error.empty() ?
                _u8L("The selected raw filament offset atlas is not compatible with the selected object.") :
                raw_atlas_error;
            m_parent.set_as_dirty();
            return false;
        }

        std::vector<uint8_t> preview =
            image_projection_raw_atlas_simulated_preview_rgba(raw_atlas,
                                                              generic_solver_mix_model_for_projection_object(selected_model_object()));
        if (preview.empty()) {
            m_image_error = _u8L("Unable to preview the selected raw filament offset atlas.");
            return false;
        }
        rgba = std::move(preview);
        width = raw_atlas.width;
        height = raw_atlas.height;
        m_raw_atlas = std::move(raw_atlas);
        m_convert_existing_colors_to_raw_offsets = true;
        if (!projection_mode_allowed(m_projection_mode))
            m_projection_mode_initialized = false;
    }

    m_image_path = into_u8(dialog.GetPath());
    m_image_rgba = std::move(rgba);
    m_image_width = width;
    m_image_height = height;
    m_overlay_texture_dirty = true;
    m_show_overlay = true;
    m_parent.set_as_dirty();
    return true;
}

void GLGizmoImageProjection::clear_projection_image()
{
    m_image_path.clear();
    m_image_error.clear();
    m_image_rgba.clear();
    m_image_width = 0;
    m_image_height = 0;
    m_raw_atlas = {};
    m_convert_existing_colors_to_raw_offsets = true;
    m_overlay_texture.reset();
    m_overlay_texture_dirty = false;
    m_show_overlay = false;
    m_parent.set_as_dirty();
}

bool GLGizmoImageProjection::ensure_overlay_texture()
{
    if (m_overlay_texture.get_id() != 0 && !m_overlay_texture_dirty)
        return true;
    if (m_image_rgba.empty() || m_image_width == 0 || m_image_height == 0)
        return false;

    std::vector<unsigned char> raw(m_image_rgba.begin(), m_image_rgba.end());
    if (!m_overlay_texture.load_from_raw_data(std::move(raw), m_image_width, m_image_height)) {
        m_image_error = _u8L("Unable to display the selected image.");
        return false;
    }
    glsafe(::glBindTexture(GL_TEXTURE_2D, m_overlay_texture.get_id()));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0));
    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
    m_overlay_texture_dirty = false;
    return true;
}

GLGizmoImageProjection::OverlayRect GLGizmoImageProjection::overlay_rect() const
{
    OverlayRect rect;
    if (m_image_width == 0 || m_image_height == 0)
        return rect;

    const Size canvas_size = m_parent.get_canvas_size();
    const float canvas_w = float(std::max(1, canvas_size.get_width()));
    const float canvas_h = float(std::max(1, canvas_size.get_height()));
    const float max_w = canvas_w * 0.48f;
    const float max_h = canvas_h * 0.48f;
    float scale = std::min(max_w / float(m_image_width), max_h / float(m_image_height));
    if (!std::isfinite(scale) || scale <= 0.f)
        scale = 1.f;

    rect.width = float(m_image_width) * scale;
    rect.height = float(m_image_height) * scale;
    rect.left = (canvas_w - rect.width) * 0.5f;
    rect.top = (canvas_h - rect.height) * 0.5f;
    return rect;
}

ModelObject *GLGizmoImageProjection::selected_model_object() const
{
    if (m_c == nullptr)
        return nullptr;
    const auto *selection_info = m_c->selection_info();
    return selection_info != nullptr ? selection_info->model_object() : nullptr;
}

void GLGizmoImageProjection::open_color_data_management_dialog()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    Slic3r::GUI::open_color_data_management_dialog(wxGetApp().mainframe, m_parent, object, [this]() {
        m_projection_mode_initialized = false;
        update_default_projection_mode();
        m_parent.set_as_dirty();
        m_parent.request_extra_frame();
    });
    m_projection_mode_initialized = false;
    update_default_projection_mode();
    m_parent.set_as_dirty();
    m_parent.request_extra_frame();
}

void GLGizmoImageProjection::update_default_projection_mode()
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return;

    if (m_projection_mode_initialized &&
        object->id() == m_projection_mode_object_id &&
        projection_mode_allowed(m_projection_mode))
        return;

    m_projection_mode = default_projection_mode();
    m_projection_mode_initialized = true;
    m_projection_mode_object_id = object->id();
}

GLGizmoImageProjection::ProjectionMode GLGizmoImageProjection::default_projection_mode() const
{
    if (m_raw_atlas.valid())
        return ProjectionMode::ImageTexture;
    if (selected_object_has_rgb_data())
        return ProjectionMode::RGBData;
    return ProjectionMode::ImageTexture;
}

bool GLGizmoImageProjection::projection_mode_allowed(ProjectionMode mode) const
{
    const bool has_rgba_data = selected_object_has_rgb_data();
    if (m_raw_atlas.valid())
        return mode == ProjectionMode::ImageTexture && !has_rgba_data;
    if (mode == ProjectionMode::ImageTexture)
        return !has_rgba_data;
    if (has_rgba_data)
        return mode == ProjectionMode::RGBData;
    if (selected_object_has_image_texture_data())
        return mode == ProjectionMode::ImageTexture || mode == ProjectionMode::RGBData;
    return true;
}

bool GLGizmoImageProjection::selected_object_has_image_texture_data() const
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && model_volume_has_bakeable_image_texture_data(volume))
            return true;
    return false;
}

bool GLGizmoImageProjection::selected_object_has_vertex_color_data() const
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->imported_vertex_colors_rgba.empty())
            return true;
    return false;
}

bool GLGizmoImageProjection::selected_object_has_rgb_data() const
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && !volume->texture_mapping_color_facets.empty())
            return true;
    return false;
}

bool GLGizmoImageProjection::selected_object_has_raw_atlas_texture_data() const
{
    const ModelObject *object = selected_model_object();
    if (object == nullptr)
        return false;
    for (const ModelVolume *volume : object->volumes)
        if (volume != nullptr && volume->is_model_part() && model_volume_has_raw_atlas_texture_data(volume))
            return true;
    return false;
}

void GLGizmoImageProjection::on_render_input_window(float x, float y, float bottom_limit)
{
    update_default_projection_mode();

    if (m_show_overlay && ensure_overlay_texture()) {
        const OverlayRect rect = overlay_rect();
        if (rect.width > 0.f && rect.height > 0.f) {
            ImGui::GetBackgroundDrawList()->AddImage((void *)(intptr_t)m_overlay_texture.get_id(),
                                                     ImVec2(rect.left, rect.top),
                                                     ImVec2(rect.left + rect.width, rect.top + rect.height),
                                                     ImVec2(0.f, 0.f),
                                                     ImVec2(1.f, 1.f),
                                                     ImGui::GetColorU32(
                                                         ImVec4(1.f, 1.f, 1.f, 0.72f * std::clamp(m_projection_opacity, 0.f, 1.f))));
        }
    }

    const float approx_height = m_imgui->scaled(12.0f);
    y = std::min(y, bottom_limit - approx_height);
    GizmoImguiSetNextWIndowPos(x, y, ImGuiCond_Always);

    ImGuiWrapper::push_toolbar_style(m_parent.get_scale());
    GizmoImguiBegin(get_name(), ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);
    const float max_tooltip_width = ImGui::GetFontSize() * 20.f;

    if (m_imgui->button(_L("Load image")))
        load_projection_image();

    ImGui::SameLine();
    m_imgui->disabled_begin(m_image_rgba.empty());
    if (m_imgui->button(_L("Clear image")))
        clear_projection_image();
    m_imgui->disabled_end();

    if (!m_image_path.empty()) {
        const size_t slash = m_image_path.find_last_of("/\\");
        ImGui::SameLine();
        m_imgui->text(from_u8(slash == std::string::npos ? m_image_path : m_image_path.substr(slash + 1)));
    }

    if (!m_image_error.empty())
        m_imgui->warning_text(from_u8(m_image_error));

    const bool has_rgba_data = selected_object_has_rgb_data();
    if (!m_image_rgba.empty() && !m_raw_atlas.valid() && selected_object_has_raw_atlas_texture_data() && !has_rgba_data) {
        if (m_projection_mode == ProjectionMode::ImageTexture)
            m_imgui->warning_text(raw_offset_data_image_texture_projection_warning_text(), m_imgui->scaled(RawOffsetDataWarningWrapEm));
        else if (m_projection_mode == ProjectionMode::RGBData)
            m_imgui->warning_text(raw_offset_data_rgba_conversion_warning_text(), m_imgui->scaled(RawOffsetDataWarningWrapEm));
    }

    if (!m_image_rgba.empty() && ImGui::Checkbox("Show image overlay", &m_show_overlay)) {
        m_parent.set_as_dirty();
        m_parent.request_extra_frame();
    }

    if (m_imgui->button(_L("Manage Color Data for this object")))
        open_color_data_management_dialog();

    // if (has_rgba_data)
    //     m_imgui->warning_text(_L("Note: Image Texture mode is disabled as this object has RGBA data (must use RGBA data mode)"));

    m_imgui->text(_L("Apply to:"));
    ImGui::SameLine();
    const char *mode_labels[] = { "Vertex colors", "Image Texture", "RGBA data" };
    int mode = std::clamp(int(m_projection_mode), 0, 2);
    if (ImGui::BeginCombo("##projection_mode", mode_labels[mode])) {
        for (int idx = 0; idx < 3; ++idx) {
            const ProjectionMode candidate = ProjectionMode(idx);
            const bool allowed = projection_mode_allowed(candidate);
            const bool selected = m_projection_mode == candidate;
            const ImGuiSelectableFlags flags = allowed ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled;
            if (ImGui::Selectable(mode_labels[idx], selected, flags) && allowed) {
                mode = idx;
                m_projection_mode = candidate;
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    float opacity_pct = m_projection_opacity * 100.f;
    m_imgui->text(_L("Opacity"));
    ImGui::PushItemWidth(m_imgui->scaled(8.f));
    if (m_imgui->bbl_slider_float_style("##image_projection_opacity", &opacity_pct, 0.f, 100.f, "%.0f%%", 1.f, true)) {
        m_projection_opacity = std::clamp(opacity_pct / 100.f, 0.f, 1.f);
        m_parent.set_as_dirty();
    }
    ImGui::PopItemWidth();

    if (m_c != nullptr && m_c->object_clipper() != nullptr) {
        ImGui::Separator();
        if (m_c->object_clipper()->get_position() == 0.f) {
            m_imgui->text(_L("Section view"));
        } else if (m_imgui->button(_L("Reset section view direction"))) {
            wxGetApp().CallAfter([this]() {
                if (m_c != nullptr && m_c->object_clipper() != nullptr)
                    m_c->object_clipper()->set_position_by_ratio(-1., false);
            });
        }

        float clp_dist = float(m_c->object_clipper()->get_position());
        ImGui::PushItemWidth(m_imgui->scaled(8.f));
        if (m_imgui->bbl_slider_float_style("##image_projection_clp_dist", &clp_dist, 0.f, 1.f, "%.2f", 1.f, true))
            m_c->object_clipper()->set_position_by_ratio(clp_dist, true);
        ImGui::PopItemWidth();

        if (ImGui::IsItemHovered())
            m_imgui->tooltip(_L("Section view"), max_tooltip_width);
    }

    ImGui::Checkbox("Apply transparent regions as background color", &m_apply_transparency_as_background);
    ImGui::Checkbox("Pass through model", &m_pass_through_model);

    const bool show_raw_conversion_option =
        m_raw_atlas.valid() &&
        selected_model_object() != nullptr &&
        !selected_object_has_raw_atlas_texture_data();
    if (show_raw_conversion_option)
        ImGui::Checkbox("Convert existing colors to raw offset data (slow)", &m_convert_existing_colors_to_raw_offsets);

    m_imgui->disabled_begin(m_image_rgba.empty() || !projection_mode_allowed(m_projection_mode));
    if (m_imgui->button(_L("Project image onto model")) && project_image_to_selected_object()) {
        m_show_overlay = false;
        m_parent.set_as_dirty();
        m_parent.request_extra_frame();
    }
    m_imgui->disabled_end();

    GizmoImguiEnd();
    ImGuiWrapper::pop_toolbar_style();
}

bool GLGizmoImageProjection::project_image_to_selected_object()
{
    ModelObject *object = selected_model_object();
    if (object == nullptr || m_image_rgba.empty())
        return false;

    RawAtlasProjectionLayout raw_layout;
    if (m_raw_atlas.valid()) {
        std::string raw_atlas_error;
        if (!raw_atlas_projection_layout_for_object(*object, m_raw_atlas, raw_layout, &raw_atlas_error)) {
            m_image_error = raw_atlas_error.empty() ?
                _u8L("The selected raw filament offset atlas is not compatible with the selected object.") :
                raw_atlas_error;
            m_image_path.clear();
            m_image_rgba.clear();
            m_image_width = 0;
            m_image_height = 0;
            m_raw_atlas = {};
            m_overlay_texture.reset();
            m_overlay_texture_dirty = false;
            m_show_overlay = false;
            m_parent.set_as_dirty();
            return false;
        }
    }

    update_default_projection_mode();
    if (!projection_mode_allowed(m_projection_mode))
        return false;

    const bool converting_raw_to_rgba_image =
        !m_raw_atlas.valid() &&
        m_projection_mode == ProjectionMode::ImageTexture &&
        selected_object_has_raw_atlas_texture_data() &&
        !selected_object_has_rgb_data();
    const bool creating_rgba_data_from_raw =
        !m_raw_atlas.valid() &&
        m_projection_mode == ProjectionMode::RGBData &&
        selected_object_has_raw_atlas_texture_data() &&
        !selected_object_has_rgb_data();

    bool changed = false;
    Plater::TakeSnapshot snapshot(wxGetApp().plater(), "Project image onto model", UndoRedo::SnapshotType::GizmoAction);
    switch (m_projection_mode) {
    case ProjectionMode::VertexColors:
        changed = project_to_vertex_colors(object);
        break;
    case ProjectionMode::ImageTexture:
        changed = project_to_image_texture(object);
        break;
    case ProjectionMode::RGBData:
        changed = project_to_rgb_data(object);
        break;
    }

    if (!changed)
        return false;

    const bool whole_image_texture_mapped_without_regions = object_is_whole_image_texture_mapped_without_regions(*object);
    const unsigned int texture_mapping_filament_id =
        (!whole_image_texture_mapped_without_regions || m_raw_atlas.valid()) ? texture_mapping_zone_id_for_image_projection(*object) : 0;
    if (m_raw_atlas.valid() && texture_mapping_filament_id != 0)
        enable_texture_mapping_zone_simulated_preview(texture_mapping_filament_id);

    if (!whole_image_texture_mapped_without_regions) {
        if (texture_mapping_filament_id != 0) {
            const Selection &selection = m_parent.get_selection();
            const int instance_idx = selection.get_instance_idx();
            const Camera &camera = wxGetApp().plater()->get_camera();
            const std::array<int, 4> &viewport = camera.get_viewport();
            const OverlayRect rect = overlay_rect();

            ProjectionContext context;
            context.view_projection = camera.get_projection_matrix().matrix() * camera.get_view_matrix().matrix();
            context.camera_forward = camera.get_dir_forward();
            context.camera_position = camera.get_position();
            context.canvas_width = std::max(1, viewport[2]);
            context.canvas_height = std::max(1, viewport[3]);
            context.overlay_left = rect.left;
            context.overlay_top = rect.top;
            context.overlay_width = rect.width;
            context.overlay_height = rect.height;
            context.image_rgba = &m_image_rgba;
            context.image_width = m_image_width;
            context.image_height = m_image_height;
            context.image_opacity = m_projection_opacity;
            context.apply_transparency_as_background = m_apply_transparency_as_background;
            apply_projection_section_view(context, m_c != nullptr ? m_c->object_clipper() : nullptr);

            project_texture_mapping_zone_to_regions(*object,
                                                    m_parent,
                                                    context,
                                                    instance_idx,
                                                    m_pass_through_model,
                                                    texture_mapping_filament_id);
        }
    }

    refresh_projected_object(object);
    m_projection_mode_initialized = true;
    m_projection_mode_object_id = object->id();
    if (converting_raw_to_rgba_image && !selected_object_has_raw_atlas_texture_data())
        show_raw_offset_data_converted_to_rgba_image_message();
    if (creating_rgba_data_from_raw && object_has_rgba_data(*object))
        show_raw_offset_data_converted_to_rgba_message();
    return true;
}

bool GLGizmoImageProjection::project_to_vertex_colors(ModelObject *object)
{
    const Selection &selection = m_parent.get_selection();
    const int instance_idx = selection.get_instance_idx();
    const Camera &camera = wxGetApp().plater()->get_camera();
    const std::array<int, 4> &viewport = camera.get_viewport();
    const OverlayRect rect = overlay_rect();

    ProjectionContext context;
    context.view_projection = camera.get_projection_matrix().matrix() * camera.get_view_matrix().matrix();
    context.camera_forward = camera.get_dir_forward();
    context.camera_position = camera.get_position();
    context.canvas_width = std::max(1, viewport[2]);
    context.canvas_height = std::max(1, viewport[3]);
    context.overlay_left = rect.left;
    context.overlay_top = rect.top;
    context.overlay_width = rect.width;
    context.overlay_height = rect.height;
    context.image_rgba = &m_image_rgba;
    context.image_width = m_image_width;
    context.image_height = m_image_height;
    context.image_opacity = m_projection_opacity;
    context.apply_transparency_as_background = m_apply_transparency_as_background;
    apply_projection_section_view(context, m_c != nullptr ? m_c->object_clipper() : nullptr);

    const ProjectionVisibility visibility = m_pass_through_model ?
        ProjectionVisibility() :
        build_projection_visibility(context, m_parent, object, instance_idx);

    bool changed = false;
    for (size_t volume_idx = 0; volume_idx < object->volumes.size(); ++volume_idx) {
        ModelVolume *volume = object->volumes[volume_idx];
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const VolumeColorSource source = build_volume_color_source(*volume);
        const ColorRGBA fallback_color = projection_base_color_for_volume(*volume);
        std::vector<std::array<float, 4>> base_accum(its.vertices.size(), { 0.f, 0.f, 0.f, 0.f });
        std::vector<unsigned int> base_counts(its.vertices.size(), 0);

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            for (int corner = 0; corner < 3; ++corner) {
                if (tri[corner] < 0 || size_t(tri[corner]) >= its.vertices.size())
                    continue;
                Vec3f barycentric = Vec3f::Zero();
                barycentric[corner] = 1.f;
                const ColorRGBA color = sample_volume_color_source(*volume,
                                                                   source,
                                                                   tri_idx,
                                                                   its.vertices[size_t(tri[corner])].cast<float>(),
                                                                   barycentric,
                                                                   true,
                                                                   &fallback_color);
                std::array<float, 4> &accum = base_accum[size_t(tri[corner])];
                accum[0] += color.r();
                accum[1] += color.g();
                accum[2] += color.b();
                accum[3] += color.a();
                ++base_counts[size_t(tri[corner])];
            }
        }

        std::vector<ColorRGBA> base_colors(its.vertices.size(), ColorRGBA(1.f, 1.f, 1.f, 1.f));
        for (size_t idx = 0; idx < base_colors.size(); ++idx) {
            if (base_counts[idx] == 0)
                continue;
            const float inv = 1.f / float(base_counts[idx]);
            base_colors[idx] = ColorRGBA(base_accum[idx][0] * inv,
                                         base_accum[idx][1] * inv,
                                         base_accum[idx][2] * inv,
                                         base_accum[idx][3] * inv);
        }

        std::vector<std::array<float, 4>> projected_accum(its.vertices.size(), { 0.f, 0.f, 0.f, 0.f });
        std::vector<unsigned int> projected_counts(its.vertices.size(), 0);
        const Transform3d world_matrix = projection_world_matrix_for_volume(m_parent, object, volume, instance_idx);
        const Matrix3d world_normal_matrix = world_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();
        const std::vector<Vec3d> vertex_normals = projection_smoothed_vertex_normals(its);

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };

            for (int corner = 0; corner < 3; ++corner) {
                const size_t vertex_idx = size_t(tri[corner]);
                Vec3f barycentric = Vec3f::Zero();
                barycentric[corner] = 1.f;
                if (!m_pass_through_model &&
                    !projection_point_allowed_by_camera_facing(context,
                                                               world_matrix,
                                                               world_normal_matrix,
                                                               vertex_normals,
                                                               tri,
                                                               vertices[size_t(corner)],
                                                               barycentric))
                    continue;
                if (!m_pass_through_model &&
                    !projection_point_is_visible(visibility,
                                                 context,
                                                 world_matrix,
                                                 vertices[size_t(corner)],
                                                 projection_visibility_triangle_key(volume_idx, tri_idx)))
                    continue;
                if (std::optional<ColorRGBA> projected = projected_image_color_at_point(context, world_matrix, vertices[size_t(corner)])) {
                    if (!context.apply_transparency_as_background && !projection_overlay_has_paintable_alpha(*projected, context))
                        continue;
                    const ColorRGBA color = apply_projection_color(base_colors[vertex_idx], *projected, context, false);
                    std::array<float, 4> &accum = projected_accum[vertex_idx];
                    accum[0] += color.r();
                    accum[1] += color.g();
                    accum[2] += color.b();
                    accum[3] += color.a();
                    ++projected_counts[vertex_idx];
                }
            }
        }

        volume->imported_vertex_colors_rgba.assign(its.vertices.size(), 0xFFFFFFFFu);
        for (size_t idx = 0; idx < its.vertices.size(); ++idx) {
            ColorRGBA color = base_colors[idx];
            if (projected_counts[idx] > 0) {
                const float inv = 1.f / float(projected_counts[idx]);
                color = ColorRGBA(projected_accum[idx][0] * inv,
                                  projected_accum[idx][1] * inv,
                                  projected_accum[idx][2] * inv,
                                  projected_accum[idx][3] * inv);
            }
            volume->imported_vertex_colors_rgba[idx] = pack_vertex_color_rgba(color);
        }
        changed = true;
    }
    return changed;
}

bool GLGizmoImageProjection::project_to_image_texture(ModelObject *object)
{
    const Selection &selection = m_parent.get_selection();
    const int instance_idx = selection.get_instance_idx();
    const Camera &camera = wxGetApp().plater()->get_camera();
    const std::array<int, 4> &viewport = camera.get_viewport();
    const OverlayRect rect = overlay_rect();

    ProjectionContext context;
    context.view_projection = camera.get_projection_matrix().matrix() * camera.get_view_matrix().matrix();
    context.camera_forward = camera.get_dir_forward();
    context.camera_position = camera.get_position();
    context.canvas_width = std::max(1, viewport[2]);
    context.canvas_height = std::max(1, viewport[3]);
    context.overlay_left = rect.left;
    context.overlay_top = rect.top;
    context.overlay_width = rect.width;
    context.overlay_height = rect.height;
    context.image_rgba = &m_image_rgba;
    context.image_width = m_image_width;
    context.image_height = m_image_height;
    context.image_opacity = m_projection_opacity;
    context.apply_transparency_as_background = m_apply_transparency_as_background;
    apply_projection_section_view(context, m_c != nullptr ? m_c->object_clipper() : nullptr);

    const ProjectionVisibility visibility = m_pass_through_model ?
        ProjectionVisibility() :
        build_projection_visibility(context, m_parent, object, instance_idx);
    const bool raw_atlas_projection = m_raw_atlas.valid();
    RawAtlasProjectionLayout raw_layout;
    if (raw_atlas_projection) {
        std::string raw_atlas_error;
        if (!raw_atlas_projection_layout_for_object(*object, m_raw_atlas, raw_layout, &raw_atlas_error)) {
            m_image_error = raw_atlas_error.empty() ?
                _u8L("The selected raw filament offset atlas is not compatible with the selected object.") :
                raw_atlas_error;
            return false;
        }
    }
    const std::vector<ColorRGBA> raw_filament_colors =
        raw_atlas_projection ? raw_filament_colors_for_projection_preview(raw_layout.filaments) : std::vector<ColorRGBA>();
    const int raw_projection_mix_model =
        raw_atlas_projection ? generic_solver_mix_model_for_projection_object(object) : TextureMappingZone::DefaultGenericSolverMixModel;
    const RawOffsetProjectionPreviewSettings raw_projection_preview_settings =
        raw_atlas_projection ? raw_offset_projection_preview_settings(raw_filament_colors, raw_projection_mix_model) : RawOffsetProjectionPreviewSettings();
    bool object_had_raw_atlas_texture = false;
    if (raw_atlas_projection) {
        for (const ModelVolume *volume : object->volumes) {
            if (volume != nullptr && volume->is_model_part() && model_volume_has_raw_atlas_texture_data(volume)) {
                object_had_raw_atlas_texture = true;
                break;
            }
        }
    }
    const bool convert_existing_colors_for_raw_projection =
        m_convert_existing_colors_to_raw_offsets || object_had_raw_atlas_texture;
    RawOffsetColorConversionSolver raw_conversion_solver =
        raw_atlas_projection && convert_existing_colors_for_raw_projection ?
            build_raw_offset_color_conversion_solver(raw_filament_colors, raw_projection_mix_model) :
            RawOffsetColorConversionSolver();

    bool changed = false;
    for (size_t volume_idx = 0; volume_idx < object->volumes.size(); ++volume_idx) {
        ModelVolume *volume = object->volumes[volume_idx];
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const ColorRGBA fallback_color = projection_base_color_for_volume(*volume);
        const Transform3d world_matrix = projection_world_matrix_for_volume(m_parent, object, volume, instance_idx);
        const Matrix3d world_normal_matrix = world_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();
        const std::vector<Vec3d> vertex_normals = projection_smoothed_vertex_normals(its);
        const bool had_raw_atlas_texture = model_volume_has_raw_atlas_texture_data(volume);
        const bool discard_existing_texture_for_raw_atlas =
            raw_atlas_projection &&
            !had_raw_atlas_texture &&
            !convert_existing_colors_for_raw_projection;
        const bool generated_texture =
            !model_volume_has_bakeable_image_texture_data(volume) ||
            discard_existing_texture_for_raw_atlas;
        GeneratedImageTextureAtlas generated_atlas;
        if (generated_texture) {
            if (!initialize_generated_image_texture(*volume, fallback_color, &generated_atlas, &world_matrix))
                continue;
        }

        const VolumeColorSource source = build_volume_color_source(*volume);
        const bool rewrite_texture_base = generated_texture || !volume->texture_mapping_color_facets.empty();
        const std::vector<uint8_t> source_texture_rgba(volume->imported_texture_rgba.begin(), volume->imported_texture_rgba.end());
        const bool seed_raw_from_existing_colors =
            raw_atlas_projection &&
            convert_existing_colors_for_raw_projection &&
            !had_raw_atlas_texture &&
            (!generated_texture ||
             !volume->texture_mapping_color_facets.empty() ||
             volume->imported_vertex_colors_rgba.size() == its.vertices.size());
        const std::optional<ColorRGBA> raw_conversion_background_color =
            seed_raw_from_existing_colors ? configured_texture_mapping_background_color_for_volume(*volume) : std::optional<ColorRGBA>();
        std::vector<uint8_t> raw_seeded_pixels;
        if (seed_raw_from_existing_colors)
            raw_seeded_pixels.assign(size_t(volume->imported_texture_width) * size_t(volume->imported_texture_height), 0);

        bool volume_changed = generated_texture;
        if (raw_atlas_projection) {
            volume_changed |= merge_imported_texture_raw_atlas(*volume, raw_layout);
        } else if (model_volume_has_raw_atlas_texture_data(volume)) {
            clear_imported_texture_raw_atlas(*volume);
            volume_changed = true;
        }

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;
            if (tri_idx >= volume->imported_texture_uv_valid.size() ||
                volume->imported_texture_uv_valid[tri_idx] == 0)
                continue;

            const size_t uv_offset = tri_idx * 6;
            if (uv_offset + 5 >= volume->imported_texture_uvs_per_face.size())
                continue;

            const std::array<Vec2f, 3> uvs = unwrap_projection_uvs(std::array<Vec2f, 3>{
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 0], volume->imported_texture_uvs_per_face[uv_offset + 1]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 2], volume->imported_texture_uvs_per_face[uv_offset + 3]),
                Vec2f(volume->imported_texture_uvs_per_face[uv_offset + 4], volume->imported_texture_uvs_per_face[uv_offset + 5])
            });
            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            const float texture_width = float(volume->imported_texture_width);
            const float texture_height = float(volume->imported_texture_height);
            if (texture_width <= 0.f || texture_height <= 0.f)
                continue;
            const std::array<Vec2f, 3> pixel_uvs = {
                Vec2f(uvs[0].x() * texture_width, uvs[0].y() * texture_height),
                Vec2f(uvs[1].x() * texture_width, uvs[1].y() * texture_height),
                Vec2f(uvs[2].x() * texture_width, uvs[2].y() * texture_height)
            };
            const float min_u = std::min({ uvs[0].x(), uvs[1].x(), uvs[2].x() });
            const float max_u = std::max({ uvs[0].x(), uvs[1].x(), uvs[2].x() });
            const float min_v = std::min({ uvs[0].y(), uvs[1].y(), uvs[2].y() });
            const float max_v = std::max({ uvs[0].y(), uvs[1].y(), uvs[2].y() });
            const int padding_px = generated_texture ? std::max(generated_atlas.padding_px, 0) : 1;
            int min_x = int(std::floor(min_u * texture_width)) - padding_px;
            int max_x = int(std::ceil(max_u * texture_width)) + padding_px;
            int min_y = int(std::floor(min_v * texture_height)) - padding_px;
            int max_y = int(std::ceil(max_v * texture_height)) + padding_px;
            if (generated_texture && tri_idx < generated_atlas.island_by_triangle.size()) {
                const int island_idx = generated_atlas.island_by_triangle[tri_idx];
                if (island_idx >= 0 && size_t(island_idx) < generated_atlas.islands.size()) {
                    const GeneratedImageTextureIsland &island = generated_atlas.islands[size_t(island_idx)];
                    min_x = std::clamp(min_x, island.x, island.x + island.rect_width - 1);
                    max_x = std::clamp(max_x, island.x, island.x + island.rect_width - 1);
                    min_y = std::clamp(min_y, island.y, island.y + island.rect_height - 1);
                    max_y = std::clamp(max_y, island.y, island.y + island.rect_height - 1);
                }
            }
            const bool uv_raster_too_large =
                max_x - min_x > int(volume->imported_texture_width) * 2 ||
                max_y - min_y > int(volume->imported_texture_height) * 2;

            if (!uv_raster_too_large) {
                for (int y_px = min_y; y_px <= max_y; ++y_px) {
                    for (int x_px = min_x; x_px <= max_x; ++x_px) {
                        const Vec2f pixel(float(x_px) + 0.5f, float(y_px) + 0.5f);
                        Vec3f barycentric = Vec3f::Zero();
                        if (generated_texture) {
                            if (!barycentric_weights_2d(pixel, pixel_uvs[0], pixel_uvs[1], pixel_uvs[2], barycentric))
                                continue;
                            if (barycentric.x() < -1e-4f || barycentric.y() < -1e-4f || barycentric.z() < -1e-4f)
                                barycentric = normalized_nonnegative_barycentric(barycentric);
                        } else {
                            if (!conservative_barycentric_weights_2d(pixel,
                                                                      pixel_uvs[0],
                                                                      pixel_uvs[1],
                                                                      pixel_uvs[2],
                                                                      float(padding_px) + 0.7072f,
                                                                      barycentric))
                                continue;
                        }

                        const Vec3f point = vertices[0] * barycentric.x() +
                                            vertices[1] * barycentric.y() +
                                            vertices[2] * barycentric.z();
                        ColorRGBA color = rewrite_texture_base ?
                            sample_volume_color_source(*volume, source, tri_idx, point, barycentric, false, &fallback_color) :
                            read_rgba_pixel(source_texture_rgba,
                                            volume->imported_texture_width,
                                            wrapped_texture_pixel(x_px, volume->imported_texture_width),
                                            wrapped_texture_pixel(y_px, volume->imported_texture_height));
                        const uint32_t wrapped_x = wrapped_texture_pixel(x_px, volume->imported_texture_width);
                        const uint32_t wrapped_y = wrapped_texture_pixel(y_px, volume->imported_texture_height);
                        const size_t raw_seed_idx = size_t(wrapped_y) * size_t(volume->imported_texture_width) + size_t(wrapped_x);
                        if (seed_raw_from_existing_colors &&
                            raw_seed_idx < raw_seeded_pixels.size() &&
                            raw_seeded_pixels[raw_seed_idx] == 0) {
                            const std::vector<uint8_t> raw_values =
                                raw_offset_values_from_color(raw_filament_colors,
                                                             color,
                                                             raw_conversion_background_color,
                                                             raw_projection_mix_model,
                                                             &raw_conversion_solver);
                            volume_changed |= write_raw_offset_pixel(volume->imported_texture_raw_filament_offsets,
                                                                     volume->imported_texture_width,
                                                                     volume->imported_texture_raw_channels,
                                                                     wrapped_x,
                                                                     wrapped_y,
                                                                     raw_values);
                            raw_seeded_pixels[raw_seed_idx] = 1;
                        }
                        if (m_pass_through_model ||
                            (projection_point_allowed_by_camera_facing(context,
                                                                       world_matrix,
                                                                       world_normal_matrix,
                                                                       vertex_normals,
                                                                       tri,
                                                                       point,
                                                                       barycentric) &&
                             projection_point_is_visible(visibility,
                                                         context,
                                                         world_matrix,
                                                         point,
                                                         projection_visibility_triangle_key(volume_idx, tri_idx)))) {
                            if (std::optional<ColorRGBA> projected = projected_image_color_at_point(context, world_matrix, point)) {
                                const bool transparent_sample =
                                    !context.apply_transparency_as_background &&
                                    !projection_overlay_has_paintable_alpha(*projected, context);
                                if (transparent_sample) {
                                    if (!rewrite_texture_base) {
                                        continue;
                                    }
                                } else {
                                    if (raw_atlas_projection) {
                                        std::vector<uint8_t> atlas_raw_values =
                                            projected_raw_offsets_at_point(context, m_raw_atlas, world_matrix, point);
                                        if (atlas_raw_values.empty())
                                            continue;
                                        std::vector<uint8_t> raw_values = raw_offset_pixel_values(*volume, wrapped_x, wrapped_y);
                                        if (raw_values.size() != size_t(volume->imported_texture_raw_channels))
                                            raw_values.assign(size_t(volume->imported_texture_raw_channels), 0);
                                        const float alpha = projection_overlay_alpha(*projected, context);
                                        for (size_t atlas_channel = 0;
                                             atlas_channel < atlas_raw_values.size() &&
                                             atlas_channel < raw_layout.atlas_to_target_channel.size();
                                             ++atlas_channel) {
                                            const size_t target_channel = raw_layout.atlas_to_target_channel[atlas_channel];
                                            if (target_channel >= raw_values.size())
                                                continue;
                                            const float base = float(raw_values[target_channel]);
                                            const float projected_value = float(atlas_raw_values[atlas_channel]);
                                            raw_values[target_channel] = uint8_t(std::clamp(
                                                int(std::lround(base * (1.f - alpha) + projected_value * alpha)), 0, 255));
                                        }
                                        volume_changed |= write_raw_offset_pixel(volume->imported_texture_raw_filament_offsets,
                                                                                volume->imported_texture_width,
                                                                                volume->imported_texture_raw_channels,
                                                                                wrapped_x,
                                                                                wrapped_y,
                                                                                raw_values);
                                        color = simulated_preview_color_from_raw_offsets(raw_projection_preview_settings,
                                                                                        raw_values.data(),
                                                                                        raw_values.size(),
                                                                                        255);
                                    } else {
                                        color = apply_projection_color(color, *projected, context, true);
                                    }
                                }
                            } else if (!rewrite_texture_base) {
                                continue;
                            }
                        }
                        volume_changed |= write_rgba_pixel(volume->imported_texture_rgba,
                                                           volume->imported_texture_width,
                                                           wrapped_x,
                                                           wrapped_y,
                                                           color);
                    }
                }
            }
        }
        if (raw_atlas_projection)
            volume_changed |= refresh_imported_texture_preview_from_raw_offsets(*volume, &raw_filament_colors, raw_projection_mix_model);
        if (volume_changed) {
            refresh_imported_texture_storage(*volume);
            if (raw_atlas_projection)
                refresh_imported_texture_raw_storage(*volume);
            changed = true;
        }
    }
    return changed;
}

bool GLGizmoImageProjection::project_to_rgb_data(ModelObject *object)
{
    const Selection &selection = m_parent.get_selection();
    const int instance_idx = selection.get_instance_idx();
    const Camera &camera = wxGetApp().plater()->get_camera();
    const std::array<int, 4> &viewport = camera.get_viewport();
    const OverlayRect rect = overlay_rect();

    ProjectionContext context;
    context.view_projection = camera.get_projection_matrix().matrix() * camera.get_view_matrix().matrix();
    context.camera_forward = camera.get_dir_forward();
    context.camera_position = camera.get_position();
    context.canvas_width = std::max(1, viewport[2]);
    context.canvas_height = std::max(1, viewport[3]);
    context.overlay_left = rect.left;
    context.overlay_top = rect.top;
    context.overlay_width = rect.width;
    context.overlay_height = rect.height;
    context.image_rgba = &m_image_rgba;
    context.image_width = m_image_width;
    context.image_height = m_image_height;
    context.image_opacity = m_projection_opacity;
    context.apply_transparency_as_background = m_apply_transparency_as_background;
    apply_projection_section_view(context, m_c != nullptr ? m_c->object_clipper() : nullptr);

    const ProjectionVisibility visibility = m_pass_through_model ?
        ProjectionVisibility() :
        build_projection_visibility(context, m_parent, object, instance_idx);
    const ProjectionPaintableImageMask paintable_mask = build_projection_paintable_image_mask(context, true);

    bool changed = false;
    for (size_t volume_idx = 0; volume_idx < object->volumes.size(); ++volume_idx) {
        ModelVolume *volume = object->volumes[volume_idx];
        if (volume == nullptr || !volume->is_model_part())
            continue;

        const indexed_triangle_set &its = volume->mesh().its;
        if (its.vertices.empty() || its.indices.empty())
            continue;

        const VolumeColorSource source = build_volume_color_source(*volume);
        const ColorRGBA fallback_color = projection_base_color_for_volume(*volume);
        const Transform3d world_matrix = projection_world_matrix_for_volume(m_parent, object, volume, instance_idx);
        const Matrix3d world_normal_matrix = world_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();
        const std::vector<Vec3d> vertex_normals = projection_smoothed_vertex_normals(its);
        std::vector<bool> projected_triangles(its.indices.size(), false);
        std::vector<int> projected_triangle_depths(its.indices.size(), 0);
        const float projection_target_span = image_projection_rgb_target_triangle_pixel_span(context);
        size_t projected_triangle_count = 0;

        for (size_t tri_idx = 0; tri_idx < its.indices.size(); ++tri_idx) {
            const stl_triangle_vertex_indices &tri = its.indices[tri_idx];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0)
                continue;
            if (size_t(tri[0]) >= its.vertices.size() ||
                size_t(tri[1]) >= its.vertices.size() ||
                size_t(tri[2]) >= its.vertices.size())
                continue;

            const std::array<Vec3f, 3> vertices = {
                its.vertices[size_t(tri[0])].cast<float>(),
                its.vertices[size_t(tri[1])].cast<float>(),
                its.vertices[size_t(tri[2])].cast<float>()
            };
            const bool projected_triangle = projection_triangle_should_project(context,
                                                                               visibility,
                                                                               paintable_mask,
                                                                               world_matrix,
                                                                               vertices,
                                                                               projection_visibility_triangle_key(volume_idx, tri_idx));
            projected_triangles[tri_idx] = projected_triangle;
            if (projected_triangle) {
                projected_triangle_depths[tri_idx] =
                    std::max(1,
                             texture_mapping_depth_from_span(projection_triangle_image_pixel_span(context, world_matrix, vertices),
                                                             projection_target_span,
                                                             7));
                ++projected_triangle_count;
            }
        }

        if (projected_triangle_count == 0)
            continue;

        TextureMappingColorSampler sampler = [this,
                                              volume,
                                              source,
                                              context,
                                              world_matrix,
                                              world_normal_matrix,
                                              fallback_color,
                                              volume_idx,
                                              &vertex_normals,
                                              &projected_triangles,
                                              &visibility](size_t tri_idx,
                                                           const Vec3f &point,
                                                           const Vec3f &barycentric) {
            ColorRGBA color = sample_volume_color_source(*volume, source, tri_idx, point, barycentric, true, &fallback_color);
            if (tri_idx < projected_triangles.size() && projected_triangles[tri_idx]) {
                const stl_triangle_vertex_indices &tri = volume->mesh().its.indices[tri_idx];
                if (m_pass_through_model ||
                    (projection_point_allowed_by_camera_facing(context,
                                                               world_matrix,
                                                               world_normal_matrix,
                                                               vertex_normals,
                                                               tri,
                                                               point,
                                                               barycentric) &&
                     projection_point_is_visible(visibility,
                                                 context,
                                                 world_matrix,
                                                 point,
                                                 projection_visibility_triangle_key(volume_idx, tri_idx)))) {
                    if (std::optional<ColorRGBA> projected = projected_image_color_at_point(context, world_matrix, point)) {
                        if (!context.apply_transparency_as_background && !projection_overlay_has_paintable_alpha(*projected, context))
                            return pack_vertex_color_rgba(color);
                        color = apply_projection_color(color, *projected, context, false);
                    }
                }
            }
            return pack_vertex_color_rgba(color);
        };

        const float mesh_span = mesh_max_axis_span(its);
        const int background_safe_max_depth = texture_mapping_depth_for_budget(its.indices.size(), 7, 2200000);
        const int projected_safe_max_depth = projected_triangle_count == 0 ?
            background_safe_max_depth :
            texture_mapping_depth_for_budget(projected_triangle_count, 7, 2200000);
        const int safe_max_depth = std::max(background_safe_max_depth, projected_safe_max_depth);
        const float split_threshold = safe_max_depth < 5 ? 0.018f : 0.012f;
        TextureMappingColorSubdivisionDepths subdivision_depths =
            [volume,
             mesh_span,
             background_safe_max_depth,
             projected_safe_max_depth,
             &projected_triangles,
             &projected_triangle_depths](size_t tri_idx, const std::array<Vec3f, 3> &vertices) {
            const bool projected_triangle = tri_idx < projected_triangles.size() && projected_triangles[tri_idx];
            const int triangle_max_depth = projected_triangle ? projected_safe_max_depth : background_safe_max_depth;
            int depth = 0;
            if (model_volume_has_bakeable_image_texture_data(volume)) {
                depth = texture_mapping_depth_from_span(texture_triangle_uv_pixel_span(volume, tri_idx), 8.f, triangle_max_depth);
            } else if (!projected_triangle) {
                depth = texture_mapping_depth_from_span(triangle_max_edge_length(vertices),
                                                        std::max(mesh_span / 180.f, 0.18f),
                                                        std::min(6, triangle_max_depth));
            }
            if (projected_triangle && tri_idx < projected_triangle_depths.size())
                depth = std::max(depth, std::min(projected_triangle_depths[tri_idx], triangle_max_depth));
            depth = std::clamp(depth, 0, triangle_max_depth);
            return std::make_pair(depth, depth);
        };

        TextureMappingColorLeafResamplePredicate resample_leaf =
            [this, context, world_matrix, volume_idx, &visibility, &projected_triangles](size_t tri_idx,
                                                                                         const std::array<Vec3f, 3> &vertices,
                                                                                         const std::array<Vec3f, 3> &,
                                                                                         uint32_t) {
            if (tri_idx >= projected_triangles.size() || !projected_triangles[tri_idx])
                return false;
            if (!projection_triangle_intersects_overlay(context, world_matrix, vertices))
                return false;
            if (m_pass_through_model)
                return true;

            return projection_triangle_has_visible_sample(visibility,
                                                         context,
                                                         world_matrix,
                                                         vertices,
                                                         projection_visibility_triangle_key(volume_idx, tri_idx));
        };

        volume->texture_mapping_color_facets.set_from_triangle_sampler(*volume,
                                                                       sampler,
                                                                       safe_max_depth,
                                                                       split_threshold,
                                                                       subdivision_depths,
                                                                       &projected_triangles,
                                                                       resample_leaf);
        if (volume->texture_mapping_color_facets.metadata_json().empty())
            volume->texture_mapping_color_facets.set_metadata_json(rgb_metadata_json(ColorRGBA(1.f, 1.f, 1.f, 1.f)));
        changed = true;
    }
    return changed;
}

void GLGizmoImageProjection::refresh_projected_object(ModelObject *object)
{
    m_parent.update_volumes_colors_by_extruder();
    m_parent.set_as_dirty();
    m_parent.request_extra_frame();

    const ModelObjectPtrs &objects = wxGetApp().model().objects;
    const size_t object_idx = size_t(std::find(objects.begin(), objects.end(), object) - objects.begin());
    if (object_idx < objects.size()) {
        m_parent.invalidate_texture_mapping_preview_for_object(object_idx);
        wxGetApp().obj_list()->update_info_items(object_idx);
        wxGetApp().plater()->get_partplate_list().notify_instance_update(object_idx, 0);
    }
    m_parent.post_event(SimpleEvent(EVT_GLCANVAS_SCHEDULE_BACKGROUND_PROCESS));
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

    this->clear();
}

void GLMmSegmentationGizmo3DScene::render(size_t triangle_indices_idx) const
{
    assert(triangle_indices_idx < this->triangle_indices_VBO_ids.size());
    assert(this->triangle_patches.size() == this->triangle_indices_VBO_ids.size());
    assert(this->vertices_VBO_id != 0);
    assert(this->triangle_indices_VBO_ids[triangle_indices_idx] != 0);

    GLShaderProgram* shader = wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

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
}

void GLMmSegmentationGizmo3DScene::finalize_vertices()
{
    assert(this->vertices_VBO_id == 0);
    if (!this->vertices.empty()) {
        glsafe(::glGenBuffers(1, &this->vertices_VBO_id));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_VBO_id));
        glsafe(::glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(float), this->vertices.data(), GL_STATIC_DRAW));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
        this->vertices.clear();
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

void GLGizmoMmuSegmentation::render_filament_remap_ui(float window_width, float max_tooltip_width)
{
    size_t n_extr = std::min({(size_t)EnforcerBlockerType::ExtruderMax, m_display_filament_ids.size(), m_extruder_remap.size()});

    unsigned int max_label_id = 1;
    for (size_t idx = 0; idx < n_extr; ++idx)
        max_label_id = std::max(max_label_id, m_display_filament_ids[idx]);
    const std::string max_label = std::to_string(max_label_id);
    const ImVec2 max_label_size = ImGui::CalcTextSize(max_label.c_str(), NULL, true);
    const ImVec2 button_size(max_label_size.x + m_imgui->scaled(0.5f), 0.f);
    const int max_items_per_line = 8;
    const float item_width = button_size.x + m_imgui->scaled(1.5f);
    const float start_pos_x = ImGui::GetCursorPosX();

    for (int src = 0; src < (int)n_extr; ++src) {
        const unsigned int dst_filament_id =
            m_extruder_remap[src] < m_display_filament_ids.size() ? m_display_filament_ids[m_extruder_remap[src]] : 0;
        if (dst_filament_id == 0 || dst_filament_id > m_extruders_colors.size())
            continue;
        const ColorRGBA &dst_col = m_extruders_colors[dst_filament_id - 1];
        ImVec4 col_vec = ImGuiWrapper::to_ImVec4(dst_col);

        if (src % max_items_per_line != 0) {
            ImGui::SameLine(start_pos_x + item_width * (src % max_items_per_line));
        }
        std::string btn_id = "##remap_src_" + std::to_string(src);
        
        ImGuiColorEditFlags flags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_NoLabel  | ImGuiColorEditFlags_NoPicker |
                                    ImGuiColorEditFlags_NoTooltip;
        if (m_selected_extruder_idx != src) flags |= ImGuiColorEditFlags_NoBorder;
        
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
        std::string dst_txt = std::to_string(dst_filament_id);
        float gray = 0.299f * dst_col.r() + 0.587f * dst_col.g() + 0.114f * dst_col.b();
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

        // popup with possible destinations
        std::string pop_id = "popup_" + std::to_string(src);
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
        ImGui::PushStyleColor(ImGuiCol_PopupBg, m_is_dark_mode ? ImGuiWrapper::COL_WINDOW_BG_DARK : ImGuiWrapper::COL_WINDOW_BG);
        ImGui::PushStyleColor(ImGuiCol_Border, m_is_dark_mode ? ImVec4(0.5f, 0.5f, 0.5f, 1.0f) : ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        
        if (ImGui::BeginPopup(pop_id.c_str())) {
            const float popup_start_pos_x = ImGui::GetCursorPosX();
            
            for (int dst = 0; dst < (int)n_extr; ++dst) {
                const unsigned int popup_filament_id = m_display_filament_ids[dst];
                if (popup_filament_id == 0 || popup_filament_id > m_extruders_colors.size())
                    continue;
                const ColorRGBA &dst_col_popup = m_extruders_colors[popup_filament_id - 1];
                ImVec4 dst_vec = ImGuiWrapper::to_ImVec4(dst_col_popup);
                if (dst % max_items_per_line != 0)
                    ImGui::SameLine(popup_start_pos_x + item_width * (dst % max_items_per_line));
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
                std::string dst_num_txt = std::to_string(popup_filament_id);
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
    }

    ImGui::Dummy(ImVec2(0.0f, ImGui::GetFontSize() * 0.3f));

    if (m_imgui->button(m_desc.at("remap"))) {
        remap_filament_assignments();
        m_show_filament_remap_ui = false;
    }

    ImGui::SameLine();
    if (m_imgui->button(m_desc.at("cancel_remap")))
        m_show_filament_remap_ui = false;
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

    size_t n_extr = std::min({m_extruder_remap.size(), m_display_filament_ids.size(), MAX_EBT});
    bool   any_change = false;
    for (size_t src = 0; src < n_extr; ++src) {
        const size_t dst = m_extruder_remap[src];
        if (dst >= m_display_filament_ids.size())
            continue;

        const unsigned int src_state = m_display_filament_ids[src];
        const unsigned int dst_state = m_display_filament_ids[dst];
        if (src_state == 0 || dst_state == 0 || src_state == dst_state)
            continue;

        state_map[src_state] = static_cast<EnforcerBlockerType>(dst_state);
        if (src_state == 1)
            state_map[0] = static_cast<EnforcerBlockerType>(dst_state);

        any_change = true;
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

    for (ModelVolume* mv : mo->volumes) {
        if (!mv->is_model_part()) continue;
        ++idx;
        TriangleSelectorGUI* ts = m_triangle_selectors[idx].get();
        if (!ts) continue;
        ts->remap_triangle_state(state_map);
        ts->request_update_render_data(true);
        updated = true;
    }

    if (updated) {
        wxGetApp().plater()->get_notification_manager()->push_notification(
            _L("Filament remapping finished.").ToStdString());
        update_model_object();
        m_parent.set_as_dirty();
    }
}

} // namespace Slic3r
