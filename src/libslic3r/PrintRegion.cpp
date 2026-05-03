#include "Exception.hpp"
#include "Print.hpp"
#include "TextureMapping.hpp"

#include <algorithm>

namespace Slic3r {

static bool filament_id_uses_texture_mapping(const Print &print, unsigned int filament_id)
{
    if (filament_id == 0)
        return false;

    const size_t num_physical = print.config().filament_diameter.size();
    if (num_physical == 0)
        return false;

    const TextureMappingZone *zone = print.texture_mapping_manager().zone_from_id(filament_id);
    return zone != nullptr && zone->enabled && !zone->deleted && zone->is_image_texture();
}

static void append_used_physical_extruders_for_filament_id(const Print                  &print,
                                                           int                           filament_id,
                                                           std::vector<unsigned int>    &object_extruders)
{
    if (filament_id <= 0)
        return;

    const size_t num_physical = print.config().filament_colour.size();
    if (num_physical == 0)
        return;

    auto append_physical = [num_physical, &object_extruders](unsigned int physical_id) {
        if (physical_id >= 1 && physical_id <= num_physical)
            object_extruders.emplace_back(physical_id - 1);
    };

    const TextureMappingZone *zone = print.texture_mapping_manager().zone_from_id(unsigned(filament_id));
    if (zone != nullptr) {
        if (!zone->enabled || zone->deleted)
            return;

        std::vector<std::string> colors = print.config().filament_colour.values;
        colors.resize(num_physical, "#FFFFFF");
        std::vector<unsigned int> component_ids = zone->is_image_texture() ?
            TextureMappingManager::effective_texture_component_ids(*zone, num_physical, colors) :
            TextureMappingManager::selected_component_ids(*zone, num_physical);

        component_ids.erase(std::remove_if(component_ids.begin(),
                                           component_ids.end(),
                                           [num_physical](unsigned int id) { return id == 0 || id > num_physical; }),
                            component_ids.end());
        std::sort(component_ids.begin(), component_ids.end());
        component_ids.erase(std::unique(component_ids.begin(), component_ids.end()), component_ids.end());

        if (component_ids.empty()) {
            const unsigned int resolved = print.texture_mapping_manager().resolve_zone_component(unsigned(filament_id), num_physical, 0);
            append_physical(resolved);
        } else {
            for (unsigned int component_id : component_ids)
                append_physical(component_id);
        }
        return;
    }

    append_physical(unsigned(filament_id));
}

// 1-based extruder identifier for this region and role.
unsigned int PrintRegion::extruder(FlowRole role) const
{
    size_t extruder = 0;
    if (role == frPerimeter || role == frExternalPerimeter)
        extruder = m_config.wall_filament;
    else if (role == frInfill)
        extruder = m_config.sparse_infill_filament;
    else if (role == frSolidInfill || role == frTopSolidInfill)
        extruder = m_config.solid_infill_filament;
    else
        throw Slic3r::InvalidArgument("Unknown role");
    return extruder;
}

Flow PrintRegion::flow(const PrintObject &object, FlowRole role, double layer_height, bool first_layer) const
{
    const PrintConfig          &print_config = object.print()->config();
    ConfigOptionFloatOrPercent config_width;
    // Get extrusion width from configuration.
    // (might be an absolute value, or a percent value, or zero for auto)
    if (role == frExternalPerimeter &&
        filament_id_uses_texture_mapping(*object.print(), unsigned(std::max(0, m_config.wall_filament.value)))) {
        config_width = ConfigOptionFloatOrPercent(
            std::max(0.05, print_config.texture_mapping_outer_wall_gradient_max_line_width.value),
            false);
    } else if (first_layer && print_config.initial_layer_line_width.value > 0) {
        config_width = print_config.initial_layer_line_width;
    } else if (role == frExternalPerimeter) {
        config_width = m_config.outer_wall_line_width;
    } else if (role == frPerimeter) {
        config_width = m_config.inner_wall_line_width;
    } else if (role == frInfill) {
        config_width = m_config.sparse_infill_line_width;
    } else if (role == frSolidInfill) {
        config_width = m_config.internal_solid_infill_line_width;
    } else if (role == frTopSolidInfill) {
        config_width = m_config.top_surface_line_width;
    } else {
        throw Slic3r::InvalidArgument("Unknown role");
    }

    if (config_width.value == 0)
        config_width = object.config().line_width;
    
    // Get the configured nozzle_diameter for the extruder associated to the flow role requested.
    // Here this->extruder(role) - 1 may underflow to MAX_INT, but then the get_at() will follback to zero'th element, so everything is all right.
    auto nozzle_diameter = float(print_config.nozzle_diameter.get_at(this->extruder(role) - 1));
    return Flow::new_from_config_width(role, config_width, nozzle_diameter, float(layer_height));
}

coordf_t PrintRegion::nozzle_dmr_avg(const PrintConfig &print_config) const
{
    return (print_config.nozzle_diameter.get_at(m_config.wall_filament.value    - 1) + 
            print_config.nozzle_diameter.get_at(m_config.sparse_infill_filament.value       - 1) + 
            print_config.nozzle_diameter.get_at(m_config.solid_infill_filament.value - 1)) / 3.;
}

coordf_t PrintRegion::bridging_height_avg(const PrintConfig &print_config) const
{
    return this->nozzle_dmr_avg(print_config) * sqrt(m_config.bridge_flow.value);
}

void PrintRegion::collect_object_printing_extruders(const PrintConfig &print_config, const PrintRegionConfig &region_config, const bool has_brim, std::vector<unsigned int> &object_extruders)
{
    // These checks reflect the same logic used in the GUI for enabling/disabling extruder selection fields.
    // BBS
    auto num_extruders = (int)print_config.filament_diameter.size();
    auto emplace_extruder = [num_extruders, &object_extruders](int extruder_id) {
    	int i = std::max(0, extruder_id - 1);
        object_extruders.emplace_back((i >= num_extruders) ? 0 : i);
    };
    if (region_config.wall_loops.value > 0 || has_brim)
    	emplace_extruder(region_config.wall_filament);
    if (region_config.sparse_infill_density.value > 0)
    	emplace_extruder(region_config.sparse_infill_filament);
    if (region_config.top_shell_layers.value > 0 || region_config.bottom_shell_layers.value > 0)
    	emplace_extruder(region_config.solid_infill_filament);
}

void PrintRegion::collect_object_printing_extruders(const Print &print, std::vector<unsigned int> &object_extruders) const
{
    // PrintRegion, if used by some PrintObject, shall have all the extruders set to an existing printer extruder.
    // If not, then there must be something wrong with the Print::apply() function.
#ifndef NDEBUG
    // BBS
    auto num_extruders = int(print.config().filament_diameter.size());
    assert(this->config().wall_filament <= num_extruders || print.texture_mapping_manager().is_texture_mapping_zone_id(this->config().wall_filament));
    assert(this->config().sparse_infill_filament <= num_extruders || print.texture_mapping_manager().is_texture_mapping_zone_id(this->config().sparse_infill_filament));
    assert(this->config().solid_infill_filament <= num_extruders || print.texture_mapping_manager().is_texture_mapping_zone_id(this->config().solid_infill_filament));
#endif
    if (this->config().wall_loops.value > 0 || print.has_brim())
        append_used_physical_extruders_for_filament_id(print, this->config().wall_filament.value, object_extruders);
    if (this->config().sparse_infill_density.value > 0)
        append_used_physical_extruders_for_filament_id(print, this->config().sparse_infill_filament.value, object_extruders);
    if (this->config().top_shell_layers.value > 0 || this->config().bottom_shell_layers.value > 0)
        append_used_physical_extruders_for_filament_id(print, this->config().solid_infill_filament.value, object_extruders);
}

}
