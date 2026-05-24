#include <assert.h>
#include <stdio.h>
#include <memory>

#include "../ClipperUtils.hpp"
#include "../Color.hpp"
#include "../ColorSolver.hpp"
#include "../Geometry.hpp"
#include "../Layer.hpp"
#include "../Print.hpp"
#include "../PrintConfig.hpp"
#include "../Surface.hpp"
#include "../TextureMapping.hpp"
#include "../TextureMappingOffset.hpp"

#include "AABBTreeLines.hpp"
#include "ExtrusionEntity.hpp"
#include "FillBase.hpp"
#include "FillRectilinear.hpp"
#include "FillLightning.hpp"
#include "FillConcentricInternal.hpp"
#include "FillTpmsD.hpp"
#include "FillTpmsFK.hpp"
#include "FillConcentric.hpp"
#include "libslic3r.h"

#include <cmath>
#include <limits>
#include <optional>

namespace Slic3r {

// Calculate infill rotation angle (in radians) for a given layer from a rotation template.
// Grammar subset handled (rotation only):
//   [±]α[*Z or !][joint][-][N|B|T][length][* or !]
//   [±]α*                    sets an initial angle only (no layer processed)
// Where:
// - α: angle in degrees. Without a sign it's absolute; with +/− it's relative. α% means a percentage of 360°.
// - Runtime: *Z repeats the instruction Z times; bare * is a no-op used for initialization; ! runs once globally and then stops.
// - Solid signs (D,S,O,M,R) are not processed here; if present they are treated as invalid/non-rotation characters.
// - Joint signs (shape of the turn across a range):
//     / linear;
//     N,n vertical sinus (n = lazy/half amplitude);
//     Z,z horizontal sinus (z = lazy/half amplitude);
//     $ arcsin; L quarter circle H→V; l quarter circle V→H;
//     U,u squared; Q,q cubic; ~ random; ^ pseudorandom; | middle step; # vertical step at end.
// - Counting / range length:
//     After the joint (or after α) a count determines duration of the turn:
//       N = layer count, B = bottom_shell_layers, T = top_shell_layers.
//     Prefix '-' flips the joint (swap initial/final orientation).
// - Length modifiers convert the count to a Z range instead of a pure layer count:
//     mm, cm, m, ' (feet), " (inches), # (standard height of N layers), % (percent of model height).
//
// Behavior:
// - The template string is tokenized by commas/whitespace and evaluated cyclically with one or more "ranges" per token.
// - Absolute α resets the accumulated angle at the start of its range; relative α accumulates.
// - *Z and ! control repetition and one-time execution of tokens across layers.
// - If the template contains no metalanguage symbols, it is treated as a simple comma-separated list of angles repeated by modulo.
// - Returns angle in radians for the requested layer_id. 0° aligns with +X; fillers may internally rotate as needed.
double calculate_infill_rotation_angle(const PrintObject* object,
                                       size_t             layer_id,
                                       const double&      fixed_infill_angle,
                                       const std::string& template_string)
{
    if (template_string.empty()) {
        return Geometry::deg2rad(fixed_infill_angle);
    }
    double             angle = 0.0;
    ConfigOptionFloats rotate_angles;
    const std::string  search_string = "/NnZz$LlUuQq~^|#";
    if (regex_search(template_string, std::regex("[+\\-%*@\'\"cm" + search_string + "]"))) { // template metalanguage of rotating infill
        std::regex                 del("[\\s,]+");
        std::sregex_token_iterator it(template_string.begin(), template_string.end(), del, -1);
        std::vector<std::string>   tk;
        std::sregex_token_iterator end;
        while (it != end) {
            tk.push_back(*it++);
        }
        int    t            = 0;
        int    repeats      = 0;
        double angle_add    = 0;
        double angle_steps  = 1;
        double angle_start  = 0;
        double limit_fill_z = object->get_layer(0)->bottom_z();
        double start_fill_z = limit_fill_z;
        bool   _noop        = false;
        auto              fill_form = std::string::npos;
        bool              _absolute = false;
        bool              _negative = false;
        std::vector<bool> stop(tk.size(), false);

        for (int i = 0; i <= layer_id; i++) {
            double fill_z = object->get_layer(i)->bottom_z();

            if (limit_fill_z < object->get_layer(i)->slice_z) {
                if (repeats) { // if repeats >0 then restore parameters for new iteration
                    limit_fill_z += limit_fill_z - start_fill_z;
                    start_fill_z = fill_z;
                    repeats--;
                } else {
                    start_fill_z = fill_z;
                    limit_fill_z = object->get_layer(i)->print_z;
                    // Solid handling removed: this function only computes rotation.
                    fill_form    = std::string::npos;
                    do {
                        if (!stop[t]) {
                            _noop     = false;
                            _absolute = false;
                            _negative = false;
                            angle_start += angle_add;
                            angle_add   = 0;
                            angle_steps = 1;
                            repeats     = 1;
                            if (tk[t].find('!') != std::string::npos) // this is an one-time instruction
                                stop[t] = true;

                            char* cs = &tk[t][0];

                            if ((cs[0] >= '0' && cs[0] <= '9') && !(cs[0] == '+' || cs[0] == '-')) // absolute/relative
                                _absolute = true;

                            angle_add = strtod(cs, &cs); // read angle parameter

                            if (cs[0] == '%') { // percentage of angles
                                angle_add *= 3.6;
                                cs = &cs[1];
                            }

                            int tit = tk[t].find('*');
                            if (tit != std::string::npos) // overall angle_cycles
                                repeats = strtol(&tk[t][tit + 1], &cs, 0);

                            if (repeats) {                                // run if overall cycles greater than 0
                                // Solid signs (D,S,O,M,R) are not handled here; if present they behave as invalid characters.

                                if (cs[0] == 'B') {
                                    angle_steps = object->print()->default_region_config().bottom_shell_layers.value;
                                } else if (cs[0] == 'T') {
                                    angle_steps = object->print()->default_region_config().top_shell_layers.value;
                                } else {
                                    fill_form = search_string.find(cs[0]);
                                    if (fill_form != std::string::npos)
                                        cs = &cs[1];

                                    _negative   = (cs[0] == '-'); // negative parameter
                                    angle_steps = abs(strtod(cs, &cs));

                                    if (angle_steps && cs[0] != '\0' && cs[0] != '!') {
                                        if (cs[0] == '%') // value in the percents of fill_z
                                            limit_fill_z = angle_steps * object->height() * 1e-8;
                                        else if (cs[0] == '#') // value in the feet
                                            limit_fill_z = angle_steps * object->config().layer_height;
                                        else if (cs[0] == '\'') // value in the feet
                                            limit_fill_z = angle_steps * 12 * 25.4;
                                        else if (cs[0] == '\"') // value in the inches
                                            limit_fill_z = angle_steps * 25.4;
                                        else if (cs[0] == 'c') // value in centimeters
                                            limit_fill_z = angle_steps * 10.;
                                        else if (cs[0] == 'm') {
                                            if (cs[1] == 'm') { // value in the millimeters
                                                limit_fill_z = angle_steps * 1.;
                                            } else{
                                                limit_fill_z = angle_steps * 1000.;
                                            }
                                        }
                                        limit_fill_z += fill_z;
                                        angle_steps = 0; // limit_fill_z has already count
                                    }
                                }
                                if (angle_steps) { // if limit_fill_z does not setting by lenght method. Get count the layer id above model height
                                    if (fill_form == std::string::npos && !_absolute)
                                        angle_add *= (int) angle_steps;
                                    int idx      = i + std::max(angle_steps - 1, 0.);
                                    int sdx      = std::max(0, idx - (int) object->layers().size());
                                    idx          = std::min(idx, (int) object->layers().size() - 1);
                                    limit_fill_z = object->get_layer(idx)->print_z + sdx * object->config().layer_height;
                                }
                                repeats = std::max(repeats - 1, 0);
                            } else
                                _noop = true; // set the dumb cycle
                            if (_absolute) {  // is absolute
                                angle_start = angle_add;
                                angle_add   = 0;
                            }
                        }
                        if (++t >= tk.size())
                            t = 0;
                    } while (std::all_of(stop.begin(), stop.end(), [](bool v) { return v; }) ?
                                 false :
                                 (t ? _noop : false) || stop[t]); // if this is a dumb instruction which never reaprated twice
                }
            }
            double top_z    = object->get_layer(i)->print_z;
            double negvalue = (_negative ? limit_fill_z - top_z : top_z - start_fill_z) / (limit_fill_z - start_fill_z);

            switch (fill_form) {
            case 0: break;                                                  // /-joint, linear
            case 1: negvalue -= sin(negvalue * PI * 2.) / (PI * 2.); break; // N-joint, sinus, vertical start
            case 2: negvalue -= sin(negvalue * PI * 2.) / (PI * 4.); break; // n-joint, sinus, vertical start, lazy
            case 3: negvalue += sin(negvalue * PI * 2.) / (PI * 2.); break; // Z-joint, sinus, horizontal start
            case 4: negvalue += sin(negvalue * PI * 2.) / (PI * 4.); break; // z-joint, sinus, horizontal start, lazy
            case 5: negvalue = asin(negvalue * 2. - 1.) / PI + 0.5; break;  // $-joint, arcsin
            case 6: negvalue = sin(negvalue * PI / 2.); break;              // L-joint, quarter of circle, horizontal start
            case 7: negvalue = 1. - cos(negvalue * PI / 2.); break;         // l-joint, quarter of circle, vertical start
            case 8: negvalue = 1. - pow(1. - negvalue, 2); break;           // U-joint, squared, x2
            case 9: negvalue = pow(1 - negvalue, 2); break;                 // u-joint, squared, x2 inverse
            case 10: negvalue = 1. - pow(1. - negvalue, 3); break;          // Q-joint, cubic, x3
            case 11: negvalue = pow(1. - negvalue, 3); break;               // q-joint, cubic, x3 inverse
            case 12: negvalue = (double) rand() / RAND_MAX; break;          // ~-joint, random, fill the whole angle
            case 13: negvalue += (double) rand() / RAND_MAX - 0.5; break;   // ^-joint, pseudorandom, disperse at middle line
            case 14: negvalue = 0.5; break;                                 // |-joint, like #-joint but placed at middle angle
            case 15: negvalue = _negative ? 0. : 1.; break;                 // #-joint, vertical at the end angle
            }
            angle = Geometry::deg2rad(angle_start + angle_add * negvalue);
        }
    } else {
        rotate_angles.deserialize(template_string);
        auto rotate_angle_idx = layer_id % rotate_angles.size();
        angle                 = Geometry::deg2rad(rotate_angles.values[rotate_angle_idx]);
    }
    return angle;
}

struct SurfaceFillParams
{
	// Zero based extruder ID.
    unsigned int 	extruder = 0;
	// Infill pattern, adjusted for the density etc.
    InfillPattern  	pattern = InfillPattern(0);

    // FillBase
    // in unscaled coordinates
    coordf_t    	spacing = 0.;
    // infill / perimeter overlap, in unscaled coordinates
    coordf_t    	overlap = 0.;
    // Angle as provided by the region config, in radians.
    float       	angle = 0.f;
    // Orca: fixed_angle
    bool        fixed_angle = false;
    // Is bridging used for this fill? Bridging parameters may be used even if this->flow.bridge() is not set.
    bool 			bridge;
    // Non-negative for a bridge.
    float 			bridge_angle = 0.f;

    // FillParams
    float       	density = 0.f;
    // Infill line multiplier count.
    int   multiline = 1;
    // Don't adjust spacing to fill the space evenly.
//    bool        	dont_adjust = false;
    // Length of the infill anchor along the perimeter line.
    // 1000mm is roughly the maximum length line that fits into a 32bit coord_t.
    float 			anchor_length     = 1000.f;
    float 			anchor_length_max = 1000.f;

    // width, height of extrusion, nozzle diameter, is bridge
    // For the output, for fill generator.
    Flow 			flow;

	// For the output
    ExtrusionRole	extrusion_role = ExtrusionRole(0);

	// Various print settings?

	// Index of this entry in a linear vector.
    size_t 			idx = 0;
	// infill speed settings
	float			sparse_infill_speed = 0;
	float			top_surface_speed = 0;
	float			solid_infill_speed = 0;

    // Params for lattice infill angles
    float lateral_lattice_angle_1 = 0.f;
    float lateral_lattice_angle_2 = 0.f;
    float infill_lock_depth          = 0;
    float skin_infill_depth          = 0;
    bool symmetric_infill_y_axis = false;

    // Params for Lateral honeycomb
    float infill_overhang_angle = 60.f;

    bool texture_mapping_top_surface_image = false;
    unsigned int texture_mapping_top_surface_zone_id = 0;
    unsigned int texture_mapping_top_surface_component_id = 0;
    int texture_mapping_top_surface_stack_depth = -1;
    bool texture_mapping_top_surface_fixed_coloring = true;
    float texture_mapping_top_surface_min_width_mm = TextureMappingZone::DefaultTopSurfaceImageMinLineWidthMm;
    float texture_mapping_top_surface_max_width_mm = TextureMappingZone::DefaultTopSurfaceImageMaxLineWidthMm;
    bool texture_mapping_top_surface_same_layer_partition = false;
    int texture_mapping_top_surface_component_index = 0;
    int texture_mapping_top_surface_component_count = 0;

	bool operator<(const SurfaceFillParams &rhs) const {
#define RETURN_COMPARE_NON_EQUAL(KEY) if (this->KEY < rhs.KEY) return true; if (this->KEY > rhs.KEY) return false;
#define RETURN_COMPARE_NON_EQUAL_TYPED(TYPE, KEY) if (TYPE(this->KEY) < TYPE(rhs.KEY)) return true; if (TYPE(this->KEY) > TYPE(rhs.KEY)) return false;

		// Sort first by decreasing bridging angle, so that the bridges are processed with priority when trimming one layer by the other.
		if (this->bridge_angle > rhs.bridge_angle) return true;
		if (this->bridge_angle < rhs.bridge_angle) return false;

		RETURN_COMPARE_NON_EQUAL(extruder);
		RETURN_COMPARE_NON_EQUAL_TYPED(unsigned, pattern);
		RETURN_COMPARE_NON_EQUAL(spacing);
		RETURN_COMPARE_NON_EQUAL(overlap);
		RETURN_COMPARE_NON_EQUAL(angle);
		RETURN_COMPARE_NON_EQUAL(fixed_angle);
		RETURN_COMPARE_NON_EQUAL(density);
		RETURN_COMPARE_NON_EQUAL(multiline);
//		RETURN_COMPARE_NON_EQUAL_TYPED(unsigned, dont_adjust);
		RETURN_COMPARE_NON_EQUAL(anchor_length);
		RETURN_COMPARE_NON_EQUAL(anchor_length_max);
		RETURN_COMPARE_NON_EQUAL(flow.width());
		RETURN_COMPARE_NON_EQUAL(flow.height());
		RETURN_COMPARE_NON_EQUAL(flow.nozzle_diameter());
		RETURN_COMPARE_NON_EQUAL_TYPED(unsigned, bridge);
		RETURN_COMPARE_NON_EQUAL_TYPED(unsigned, extrusion_role);
		RETURN_COMPARE_NON_EQUAL(sparse_infill_speed);
		RETURN_COMPARE_NON_EQUAL(top_surface_speed);
		RETURN_COMPARE_NON_EQUAL(solid_infill_speed);
        RETURN_COMPARE_NON_EQUAL(lateral_lattice_angle_1);
		RETURN_COMPARE_NON_EQUAL(lateral_lattice_angle_2);
		RETURN_COMPARE_NON_EQUAL(symmetric_infill_y_axis);
		RETURN_COMPARE_NON_EQUAL(infill_lock_depth);
		RETURN_COMPARE_NON_EQUAL(skin_infill_depth);		RETURN_COMPARE_NON_EQUAL(infill_overhang_angle);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_image);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_zone_id);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_component_id);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_stack_depth);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_fixed_coloring);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_min_width_mm);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_max_width_mm);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_same_layer_partition);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_component_index);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_component_count);

		return false;
	}

	bool operator==(const SurfaceFillParams &rhs) const {
		return  this->extruder 			== rhs.extruder 		&&
				this->pattern 			== rhs.pattern 			&&
				this->spacing 			== rhs.spacing 			&&
				this->overlap 			== rhs.overlap 			&&
				this->angle   			== rhs.angle   			&&
				this->fixed_angle == rhs.fixed_angle &&
				this->bridge   			== rhs.bridge   		&&
				this->bridge_angle 		== rhs.bridge_angle		&&
				this->density   		== rhs.density   		&&
				this->multiline             == rhs.multiline    &&
//				this->dont_adjust   	== rhs.dont_adjust 		&&
				this->anchor_length  	== rhs.anchor_length    &&
				this->anchor_length_max == rhs.anchor_length_max &&
				this->flow 				== rhs.flow 			&&
				this->extrusion_role	== rhs.extrusion_role	&&
				this->sparse_infill_speed	== rhs.sparse_infill_speed &&
				this->top_surface_speed		== rhs.top_surface_speed &&
				this->solid_infill_speed	== rhs.solid_infill_speed &&
                this->lateral_lattice_angle_1		== rhs.lateral_lattice_angle_1 &&
				this->lateral_lattice_angle_2	    == rhs.lateral_lattice_angle_2 &&
				this->infill_lock_depth      ==  rhs.infill_lock_depth &&
				this->skin_infill_depth      ==  rhs.skin_infill_depth &&
                this->infill_overhang_angle == rhs.infill_overhang_angle &&
                this->texture_mapping_top_surface_image == rhs.texture_mapping_top_surface_image &&
                this->texture_mapping_top_surface_zone_id == rhs.texture_mapping_top_surface_zone_id &&
                this->texture_mapping_top_surface_component_id == rhs.texture_mapping_top_surface_component_id &&
                this->texture_mapping_top_surface_stack_depth == rhs.texture_mapping_top_surface_stack_depth &&
                this->texture_mapping_top_surface_fixed_coloring == rhs.texture_mapping_top_surface_fixed_coloring &&
                this->texture_mapping_top_surface_min_width_mm == rhs.texture_mapping_top_surface_min_width_mm &&
                this->texture_mapping_top_surface_max_width_mm == rhs.texture_mapping_top_surface_max_width_mm &&
                this->texture_mapping_top_surface_same_layer_partition == rhs.texture_mapping_top_surface_same_layer_partition &&
                this->texture_mapping_top_surface_component_index == rhs.texture_mapping_top_surface_component_index &&
                this->texture_mapping_top_surface_component_count == rhs.texture_mapping_top_surface_component_count;
	}
};

struct SurfaceFill {
	SurfaceFill(const SurfaceFillParams& params) : region_id(size_t(-1)), surface(stCount, ExPolygon()), params(params) {}

	size_t 				region_id;
	Surface 			surface;
	ExPolygons       	expolygons;
	SurfaceFillParams	params;
    // BBS
    std::vector<size_t> region_id_group;
    ExPolygons          no_overlap_expolygons;
};

struct TopSurfaceImageStackSlice {
    unsigned int component_id = 0;
    int depth = 0;
    size_t component_index = 0;
    size_t component_count = 0;
    bool same_layer_partition = false;
    float angle_rad = float(PI / 4.0);
    ExPolygons area;
};

struct TopSurfaceImageRegionPlan {
    const TextureMappingZone *zone = nullptr;
    unsigned int zone_id = 0;
    std::vector<unsigned int> components_bottom_to_top;
    std::vector<TopSurfaceImageStackSlice> slices;
    float min_width_mm = TextureMappingZone::DefaultTopSurfaceImageMinLineWidthMm;
    float max_width_mm = TextureMappingZone::DefaultTopSurfaceImageMaxLineWidthMm;
    bool fixed_coloring = true;
    bool same_layer_partition = false;
    int colored_top_layers = TextureMappingZone::DefaultTopSurfaceImageColoredTopLayers;
};

static float top_surface_image_filament_luminance(const PrintConfig &config, unsigned int component_id)
{
    ColorRGB color;
    if (component_id == 0 || component_id > config.filament_colour.values.size() ||
        !decode_color(config.filament_colour.get_at(size_t(component_id - 1)), color))
        return std::numeric_limits<float>::max();
    return 0.2126f * color.r() + 0.7152f * color.g() + 0.0722f * color.b();
}

static std::vector<unsigned int> top_surface_image_components_bottom_to_top(const TextureMappingZone &zone,
                                                                            const PrintConfig &config,
                                                                            std::vector<unsigned int> components)
{
    components.erase(std::remove_if(components.begin(), components.end(), [](unsigned int id) { return id == 0; }), components.end());
    components.erase(std::unique(components.begin(), components.end()), components.end());
    if (components.empty())
        return components;

    bool has_complete_td = true;
    for (unsigned int component_id : components) {
        const size_t idx = size_t(component_id - 1);
        const float td = idx < zone.filament_transmission_distances_mm.size() ?
            zone.filament_transmission_distances_mm[idx] :
            0.f;
        if (!std::isfinite(td) || td <= 0.f) {
            has_complete_td = false;
            break;
        }
    }
    if (has_complete_td) {
        std::stable_sort(components.begin(), components.end(), [&zone](unsigned int lhs, unsigned int rhs) {
            return zone.filament_transmission_distances_mm[size_t(lhs - 1)] <
                   zone.filament_transmission_distances_mm[size_t(rhs - 1)];
        });
        return components;
    }

    auto black_it = std::min_element(components.begin(), components.end(), [&config](unsigned int lhs, unsigned int rhs) {
        return top_surface_image_filament_luminance(config, lhs) < top_surface_image_filament_luminance(config, rhs);
    });
    if (black_it != components.end() && black_it != components.begin())
        std::rotate(components.begin(), black_it, std::next(black_it));
    return components;
}

static std::optional<std::array<float, 4>> top_surface_image_equal_blend_background(const PrintConfig &config,
                                                                                   const std::vector<unsigned int> &component_ids)
{
    std::vector<std::array<float, 3>> colors;
    colors.reserve(component_ids.size());
    for (unsigned int component_id : component_ids) {
        if (component_id == 0 || component_id > config.filament_colour.values.size())
            return std::nullopt;
        ColorRGB color;
        if (!decode_color(config.filament_colour.get_at(size_t(component_id - 1)), color))
            return std::nullopt;
        colors.push_back({ color.r(), color.g(), color.b() });
    }
    if (colors.empty())
        return std::nullopt;
    std::vector<float> weights(colors.size(), 1.f / float(colors.size()));
    const std::array<float, 3> mixed = mix_color_solver_components(colors, weights, ColorSolverMixModel::PigmentPainter);
    return std::array<float, 4> { mixed[0], mixed[1], mixed[2], 1.f };
}

static ExPolygons top_surface_image_visible_top_mask(const Layer &layer, unsigned int zone_id)
{
    ExPolygons mask;
    for (const LayerRegion *layerm : layer.regions()) {
        if (layerm == nullptr || unsigned(std::max(0, layerm->region().config().solid_infill_filament.value)) != zone_id)
            continue;
        for (const Surface &surface : layerm->fill_surfaces.surfaces)
            if (surface.is_top())
                mask.emplace_back(surface.expolygon);
    }
    return mask.size() > 1 ? union_ex(mask) : mask;
}

static std::vector<TopSurfaceImageRegionPlan> top_surface_image_region_plans(const Layer &layer)
{
    std::vector<TopSurfaceImageRegionPlan> plans(layer.regions().size());
    const PrintObject *object = layer.object();
    if (object == nullptr || object->print() == nullptr)
        return plans;

    const Print *print = object->print();
    const PrintConfig &print_config = print->config();
    const TextureMappingManager &texture_mgr = print->texture_mapping_manager();
    const size_t num_physical = print_config.filament_colour.values.size();
    if (num_physical == 0)
        return plans;

    Polygons current_layer_perimeters;
    for (const LayerRegion *layerm : layer.regions())
        if (layerm != nullptr)
            layerm->perimeters.polygons_covered_by_width(current_layer_perimeters, 0.f);

    for (size_t region_id = 0; region_id < layer.regions().size(); ++region_id) {
        const LayerRegion *layerm = layer.regions()[region_id];
        if (layerm == nullptr)
            continue;
        const int raw_zone_id = layerm->region().config().solid_infill_filament.value;
        if (raw_zone_id <= 0)
            continue;
        const unsigned int zone_id = unsigned(raw_zone_id);
        const TextureMappingZone *zone = texture_mgr.zone_from_id(zone_id);
        if (zone == nullptr || !zone->enabled || zone->deleted || !zone->top_surface_image_printing_active())
            continue;

        std::vector<std::string> filament_colours = print_config.filament_colour.values;
        filament_colours.resize(num_physical, "#FFFFFF");
        std::vector<unsigned int> components =
            TextureMappingManager::effective_texture_component_ids(*zone, num_physical, filament_colours);
        components.erase(std::remove_if(components.begin(), components.end(), [num_physical](unsigned int id) {
            return id == 0 || id > num_physical;
        }), components.end());
        components.erase(std::unique(components.begin(), components.end()), components.end());
        if (components.empty())
            continue;

        TopSurfaceImageRegionPlan plan;
        plan.zone = zone;
        plan.zone_id = zone_id;
        plan.same_layer_partition =
            zone->top_surface_image_printing_method == int(TextureMappingZone::TopSurfaceImageSameLayer45Partition);
        plan.components_bottom_to_top = plan.same_layer_partition ?
            components :
            top_surface_image_components_bottom_to_top(*zone, print_config, components);
        if (plan.components_bottom_to_top.empty())
            continue;
        plan.max_width_mm = std::clamp(zone->top_surface_image_max_line_width_mm,
                                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
        plan.min_width_mm = std::clamp(zone->top_surface_image_min_line_width_mm,
                                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                                       plan.max_width_mm);
        plan.fixed_coloring = zone->top_surface_image_fixed_coloring_filaments;
        plan.colored_top_layers = std::clamp(zone->top_surface_image_colored_top_layers,
                                             TextureMappingZone::MinTopSurfaceImageColoredTopLayers,
                                             TextureMappingZone::MaxTopSurfaceImageColoredTopLayers);

        const int stack_depth = plan.same_layer_partition ?
            plan.colored_top_layers :
            int(plan.components_bottom_to_top.size());
        for (int depth = 0; depth < stack_depth; ++depth) {
            const Layer *top_layer = &layer;
            for (int i = 0; i < depth && top_layer != nullptr; ++i)
                top_layer = top_layer->upper_layer;
            if (top_layer == nullptr)
                break;

            ExPolygons area = top_surface_image_visible_top_mask(*top_layer, zone_id);
            if (area.empty())
                continue;
            if (!current_layer_perimeters.empty())
                area = diff_ex(area, current_layer_perimeters, ApplySafetyOffset::Yes);
            if (area.empty())
                continue;

            if (plan.same_layer_partition) {
                const ExPolygons depth_area = area;
                for (size_t component_idx = 0; component_idx < plan.components_bottom_to_top.size(); ++component_idx) {
                    TopSurfaceImageStackSlice slice;
                    slice.component_id = plan.components_bottom_to_top[component_idx];
                    slice.depth = depth;
                    slice.component_index = component_idx;
                    slice.component_count = plan.components_bottom_to_top.size();
                    slice.same_layer_partition = true;
                    slice.angle_rad = (depth & 1) ? float(-PI / 4.0) : float(PI / 4.0);
                    slice.area = depth_area;
                    plan.slices.emplace_back(std::move(slice));
                }
            } else {
                TopSurfaceImageStackSlice slice;
                slice.component_id = plan.components_bottom_to_top[size_t(stack_depth - 1 - depth)];
                slice.depth = depth;
                slice.component_index = size_t(stack_depth - 1 - depth);
                slice.component_count = plan.components_bottom_to_top.size();
                slice.area = std::move(area);
                plan.slices.emplace_back(std::move(slice));
            }
        }

        if (!plan.slices.empty())
            plans[region_id] = std::move(plan);
    }

    return plans;
}

static SurfaceFill& surface_fill_for_params(std::vector<SurfaceFill> &surface_fills, const SurfaceFillParams &params)
{
    for (SurfaceFill &fill : surface_fills)
        if (fill.params == params)
            return fill;
    surface_fills.emplace_back(params);
    return surface_fills.back();
}

static void append_surface_fill_expolygons(SurfaceFill &fill,
                                           size_t region_id,
                                           const Surface &surface,
                                           ExPolygons &&expolygons,
                                           const LayerRegion &layerm)
{
    if (expolygons.empty())
        return;
    if (fill.region_id == size_t(-1)) {
        fill.region_id = region_id;
        fill.surface = surface;
        fill.surface.expolygon = ExPolygon();
        fill.expolygons = std::move(expolygons);
        fill.region_id_group.push_back(region_id);
        fill.no_overlap_expolygons = layerm.fill_no_overlap_expolygons;
    } else {
        append(fill.expolygons, std::move(expolygons));
        auto t = find(fill.region_id_group.begin(), fill.region_id_group.end(), region_id);
        if (t == fill.region_id_group.end()) {
            fill.region_id_group.push_back(region_id);
            fill.no_overlap_expolygons = union_ex(fill.no_overlap_expolygons, layerm.fill_no_overlap_expolygons);
        }
    }
}

static SurfaceFillParams top_surface_image_params_for_slice(const Layer &layer,
                                                            const Surface &surface,
                                                            const SurfaceFillParams &base_params,
                                                            const TopSurfaceImageRegionPlan &plan,
                                                            const TopSurfaceImageStackSlice &slice)
{
    SurfaceFillParams params = base_params;
    params.extruder = slice.component_id;
    params.pattern = ipRectilinear;
    params.density = 100.f;
    params.angle = slice.angle_rad;
    params.fixed_angle = true;
    params.bridge = false;
    params.bridge_angle = 0.f;
    params.multiline = 1;
    params.anchor_length = 1000.f;
    params.anchor_length_max = 1000.f;
    params.extrusion_role = surface.is_top() ? erTopSolidInfill : erSolidInfill;
    const PrintConfig &print_config = layer.object()->print()->config();
    const float nozzle = slice.component_id > 0 && size_t(slice.component_id - 1) < print_config.nozzle_diameter.values.size() ?
        float(print_config.nozzle_diameter.get_at(size_t(slice.component_id - 1))) :
        float(print_config.nozzle_diameter.values.empty() ? 0.4 : print_config.nozzle_diameter.values.front());
    const float height = float((surface.thickness == -1) ? layer.height : surface.thickness);
    params.flow = Flow(plan.max_width_mm, height, nozzle);
    params.spacing = slice.same_layer_partition && slice.component_count > 0 ?
        (plan.min_width_mm * float(slice.component_count) + (plan.max_width_mm - plan.min_width_mm)) :
        params.flow.spacing();
    params.texture_mapping_top_surface_image = true;
    params.texture_mapping_top_surface_zone_id = plan.zone_id;
    params.texture_mapping_top_surface_component_id = slice.component_id;
    params.texture_mapping_top_surface_stack_depth = slice.depth;
    params.texture_mapping_top_surface_fixed_coloring = plan.fixed_coloring;
    params.texture_mapping_top_surface_min_width_mm = plan.min_width_mm;
    params.texture_mapping_top_surface_max_width_mm = plan.max_width_mm;
    params.texture_mapping_top_surface_same_layer_partition = slice.same_layer_partition;
    params.texture_mapping_top_surface_component_index = int(slice.component_index);
    params.texture_mapping_top_surface_component_count = int(slice.component_count);
    return params;
}

static ExtrusionPaths top_surface_image_split_path(const ExtrusionPath &path,
                                                   const TextureMappingOffsetContext &context,
                                                   float min_width_mm,
                                                   float max_width_mm,
                                                   float nozzle_diameter)
{
    ExtrusionPaths out;
    if (path.polyline.points.size() < 2)
        return out;
    const float height = path.height > 0.f ? path.height : context.layer_height_mm;
    const float sample_step_mm = std::clamp(max_width_mm * 0.5f, 0.16f, 0.40f);
    for (size_t point_idx = 1; point_idx < path.polyline.points.size(); ++point_idx) {
        const Point p0 = path.polyline.points[point_idx - 1];
        const Point p1 = path.polyline.points[point_idx];
        const double len_mm = unscale<double>(p0.distance_to(p1));
        if (!std::isfinite(len_mm) || len_mm <= EPSILON)
            continue;
        const int steps = std::max(1, int(std::ceil(len_mm / sample_step_mm)));
        for (int step = 0; step < steps; ++step) {
            const double t0 = double(step) / double(steps);
            const double t1 = double(step + 1) / double(steps);
            const Point q0 = lerp(p0, p1, t0);
            const Point q1 = lerp(p0, p1, t1);
            if (q0 == q1)
                continue;
            const Point qm = lerp(p0, p1, 0.5 * (t0 + t1));
            const std::vector<float> weights =
                sample_weight_field_components(context.weight_field,
                                               unscale<float>(qm.x()),
                                               unscale<float>(qm.y()),
                                               context.high_resolution_texture_sampling);
            const float coverage = context.active_component_idx < weights.size() ?
                std::clamp(weights[context.active_component_idx], 0.f, 1.f) :
                0.f;
            const float width = std::clamp(min_width_mm + coverage * (max_width_mm - min_width_mm),
                                           min_width_mm,
                                           max_width_mm);
            Polyline polyline;
            polyline.points.emplace_back(q0);
            polyline.points.emplace_back(q1);
            ExtrusionPath segment(std::move(polyline), path);
            segment.width = width;
            segment.height = height;
            segment.mm3_per_mm = Flow(width, height, nozzle_diameter).mm3_per_mm();
            out.emplace_back(std::move(segment));
        }
    }
    return out;
}

static ExtrusionEntity* top_surface_image_modulated_entity(const ExtrusionEntity &entity,
                                                           const TextureMappingOffsetContext &context,
                                                           float min_width_mm,
                                                           float max_width_mm,
                                                           float nozzle_diameter)
{
    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity)) {
        ExtrusionPaths paths = top_surface_image_split_path(*path, context, min_width_mm, max_width_mm, nozzle_diameter);
        if (paths.empty())
            return entity.clone();
        return new ExtrusionMultiPath(std::move(paths));
    }
    if (const ExtrusionMultiPath *multi_path = dynamic_cast<const ExtrusionMultiPath *>(&entity)) {
        ExtrusionPaths paths;
        for (const ExtrusionPath &path : multi_path->paths) {
            ExtrusionPaths split = top_surface_image_split_path(path, context, min_width_mm, max_width_mm, nozzle_diameter);
            append(paths, std::move(split));
        }
        if (paths.empty())
            return entity.clone();
        ExtrusionMultiPath *out = new ExtrusionMultiPath(std::move(paths));
        if (!multi_path->can_reverse())
            out->set_reverse();
        return out;
    }
    if (const ExtrusionLoop *loop = dynamic_cast<const ExtrusionLoop *>(&entity)) {
        ExtrusionPaths paths;
        for (const ExtrusionPath &path : loop->paths) {
            ExtrusionPaths split = top_surface_image_split_path(path, context, min_width_mm, max_width_mm, nozzle_diameter);
            append(paths, std::move(split));
        }
        if (paths.empty())
            return entity.clone();
        return new ExtrusionLoop(std::move(paths));
    }
    if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity)) {
        ExtrusionEntityCollection *out = new ExtrusionEntityCollection(*collection);
        for (ExtrusionEntity *&child : out->entities) {
            ExtrusionEntity *replacement =
                top_surface_image_modulated_entity(*child, context, min_width_mm, max_width_mm, nozzle_diameter);
            delete child;
            child = replacement;
        }
        return out;
    }
    return entity.clone();
}

static std::vector<float> top_surface_image_same_layer_fractions(const std::optional<TextureMappingOffsetContext> &context,
                                                                 int                                                   component_count,
                                                                 float                                                 x_mm,
                                                                 float                                                 y_mm)
{
    component_count = std::max(1, component_count);
    std::vector<float> fractions(size_t(component_count), 1.f / float(component_count));
    if (!context)
        return fractions;
    std::vector<float> weights =
        sample_weight_field_components(context->weight_field,
                                       x_mm,
                                       y_mm,
                                       context->high_resolution_texture_sampling);
    if (weights.size() != size_t(component_count))
        return fractions;
    fractions.assign(size_t(component_count), 0.f);
    float sum = 0.f;
    for (size_t i = 0; i < weights.size(); ++i) {
        fractions[i] = std::max(0.f, weights[i]);
        sum += fractions[i];
    }
    if (sum <= EPSILON || !std::isfinite(sum)) {
        std::fill(fractions.begin(), fractions.end(), 1.f / float(component_count));
        return fractions;
    }
    for (float &fraction : fractions)
        fraction /= sum;
    return fractions;
}

static void top_surface_image_same_layer_partition_fill(ExtrusionEntityCollection                       &collection,
                                                        const Surface                                  &surface,
                                                        const SurfaceFillParams                        &params,
                                                        const std::optional<TextureMappingOffsetContext> &context)
{
    int component_count = std::max(1, params.texture_mapping_top_surface_component_count);
    int component_index = std::clamp(params.texture_mapping_top_surface_component_index, 0, component_count - 1);
    if (context && int(context->component_ids.size()) == component_count) {
        auto component_it = std::find(context->component_ids.begin(),
                                      context->component_ids.end(),
                                      params.texture_mapping_top_surface_component_id);
        if (component_it != context->component_ids.end())
            component_index = int(std::distance(context->component_ids.begin(), component_it));
        component_index = std::clamp(component_index, 0, component_count - 1);
    }
    const float min_width = std::clamp(params.texture_mapping_top_surface_min_width_mm,
                                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
    const float max_width = std::clamp(params.texture_mapping_top_surface_max_width_mm,
                                       min_width,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
    const float pitch = min_width * float(component_count) + (max_width - min_width);
    if (pitch <= EPSILON)
        return;
    const BoundingBox bbox = get_extents(surface.expolygon);
    if (!bbox.defined)
        return;
    const double theta = params.angle;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    double min_u = std::numeric_limits<double>::max();
    double min_v = std::numeric_limits<double>::max();
    double max_u = -std::numeric_limits<double>::max();
    double max_v = -std::numeric_limits<double>::max();
    for (size_t i = 0; i < 4; ++i) {
        const Point corner = bbox[i];
        const double x = unscale<double>(corner.x());
        const double y = unscale<double>(corner.y());
        const double u = x * cos_t + y * sin_t;
        const double v = -x * sin_t + y * cos_t;
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    const double sample_step = std::clamp(double(max_width), 0.32, 0.80);
    const double u_start = std::floor((min_u - pitch) / sample_step) * sample_step;
    const double u_end = std::ceil((max_u + pitch) / sample_step) * sample_step;
    const int band_start = int(std::floor((min_v - pitch) / double(pitch)));
    const int band_end = int(std::ceil((max_v + pitch) / double(pitch)));
    const ExPolygons clip { surface.expolygon };
    for (int band = band_start; band <= band_end; ++band) {
        const double band_v = double(band) * double(pitch);
        const double sample_v = band_v + double(pitch) * 0.5;
        for (double u0 = u_start; u0 < u_end - EPSILON; u0 += sample_step) {
            const double u1 = std::min(u0 + sample_step, u_end);
            const double um = 0.5 * (u0 + u1);
            const double sample_x = um * cos_t - sample_v * sin_t;
            const double sample_y = um * sin_t + sample_v * cos_t;
            const std::vector<float> fractions =
                top_surface_image_same_layer_fractions(context,
                                                       component_count,
                                                       float(sample_x),
                                                       float(sample_y));
            const float available_width = std::max(0.f, pitch - min_width * float(component_count));
            double lane_offset = 0.0;
            float lane_width = min_width;
            for (int i = 0; i < component_count; ++i) {
                const float width = min_width + available_width * fractions[size_t(i)];
                if (i == component_index) {
                    lane_width = width;
                    break;
                }
                lane_offset += width;
            }
            lane_width = std::clamp(lane_width, min_width, max_width);
            const double v_center = band_v + lane_offset + 0.5 * double(lane_width);
            const double x0 = u0 * cos_t - v_center * sin_t;
            const double y0 = u0 * sin_t + v_center * cos_t;
            const double x1 = u1 * cos_t - v_center * sin_t;
            const double y1 = u1 * sin_t + v_center * cos_t;
            Polyline polyline;
            polyline.points.emplace_back(Point::new_scale(x0, y0));
            polyline.points.emplace_back(Point::new_scale(x1, y1));
            if (polyline.points.front() == polyline.points.back())
                continue;
            ExtrusionPath path(params.extrusion_role,
                               Flow(lane_width, params.flow.height(), params.flow.nozzle_diameter()).mm3_per_mm(),
                               lane_width,
                               params.flow.height());
            path.polyline = std::move(polyline);
            path.intersect_expolygons(clip, &collection);
        }
    }
}

static void apply_top_surface_image_collection_metadata(ExtrusionEntityCollection &collection,
                                                        const SurfaceFillParams &params,
                                                        const std::optional<TextureMappingOffsetContext> &context)
{
    collection.texture_mapping_top_surface_image = true;
    collection.texture_mapping_top_surface_zone_id = params.texture_mapping_top_surface_zone_id;
    collection.texture_mapping_top_surface_desired_component_id = params.texture_mapping_top_surface_component_id;
    collection.texture_mapping_top_surface_stack_depth = params.texture_mapping_top_surface_stack_depth;
    collection.texture_mapping_top_surface_fixed_coloring = params.texture_mapping_top_surface_fixed_coloring;
    if (params.texture_mapping_top_surface_fixed_coloring &&
        params.texture_mapping_top_surface_component_id > 0)
        collection.texture_mapping_extruder_override = int(params.texture_mapping_top_surface_component_id - 1);
    if (!context)
        return;
    ExtrusionEntitiesPtr replacement;
    replacement.reserve(collection.entities.size());
    for (const ExtrusionEntity *entity : collection.entities)
        replacement.emplace_back(top_surface_image_modulated_entity(*entity,
                                                                    *context,
                                                                    params.texture_mapping_top_surface_min_width_mm,
                                                                    params.texture_mapping_top_surface_max_width_mm,
                                                                    params.flow.nozzle_diameter()));
    collection.clear();
    collection.entities = std::move(replacement);
}


// Detect narrow infill regions
// Based on the anti-vibration algorithm from PrusaSlicer:
// https://github.com/prusa3d/PrusaSlicer/blob/5dc04b4e8f14f65bbcc5377d62cad3e86c2aea36/src/libslic3r/Fill/FillEnsuring.cpp#L37-L273

static coord_t _MAX_LINE_LENGTH_TO_FILTER() // 4 mm.
{
    return scaled<coord_t>(4.);
}
const constexpr size_t  MAX_SKIPS_ALLOWED           = 2; // Skip means propagation through long line.
const constexpr size_t  MIN_DEPTH_FOR_LINE_REMOVING = 5;

struct LineNode
{
    struct State
    {
        // The total number of long lines visited before this node was reached.
        // We just need the minimum number of all possible paths to decide whether we can remove the line or not.
        int min_skips_taken             = 0;
        // The total number of short lines visited before this node was reached.
        int total_short_lines           = 0;
        // Some initial line is touching some long line. This information is propagated to neighbors.
        bool initial_touches_long_lines = false;
        bool initialized                = false;

        void reset() {
            this->min_skips_taken            = 0;
            this->total_short_lines          = 0;
            this->initial_touches_long_lines = false;
            this->initialized                = false;
        }
    };

    explicit LineNode(const Line &line) : line(line) {}

    Line                   line;
    // Pointers to line nodes in the previous and the next section that overlap with this line.
    std::vector<LineNode*> next_section_overlapping_lines;
    std::vector<LineNode*> prev_section_overlapping_lines;

    bool                   is_removed = false;

    State                  state;

    // Return true if some initial line is touching some long line and this information was propagated into the current line.
    bool is_initial_line_touching_long_lines() const {
        if (prev_section_overlapping_lines.empty())
            return false;

        for (LineNode *line_node : prev_section_overlapping_lines) {
            if (line_node->state.initial_touches_long_lines)
                return true;
        }

        return false;
    }

    // Return true if the current line overlaps with some long line in the previous section.
    bool is_touching_long_lines_in_previous_layer() const {
        if (prev_section_overlapping_lines.empty())
            return false;

        const auto MAX_LINE_LENGTH_TO_FILTER = _MAX_LINE_LENGTH_TO_FILTER();
        for (LineNode *line_node : prev_section_overlapping_lines) {
            if (!line_node->is_removed && line_node->line.length() >= MAX_LINE_LENGTH_TO_FILTER)
                return true;
        }

        return false;
    }

    // Return true if the current line overlaps with some line in the next section.
    bool has_next_layer_neighbours() const {
        if (next_section_overlapping_lines.empty())
            return false;

        for (LineNode *line_node : next_section_overlapping_lines) {
            if (!line_node->is_removed)
                return true;
        }

        return false;
    }
};

using LineNodes = std::vector<LineNode>;

inline bool are_lines_overlapping_in_y_axes(const Line &first_line, const Line &second_line) {
    return (second_line.a.y() <= first_line.a.y() && first_line.a.y() <= second_line.b.y())
        || (second_line.a.y() <= first_line.b.y() && first_line.b.y() <= second_line.b.y())
        || (first_line.a.y() <= second_line.a.y() && second_line.a.y() <= first_line.b.y())
        || (first_line.a.y() <= second_line.b.y() && second_line.b.y() <= first_line.b.y());
}

bool can_line_note_be_removed(const LineNode &line_node) {
    const auto MAX_LINE_LENGTH_TO_FILTER = _MAX_LINE_LENGTH_TO_FILTER();
    return (line_node.line.length() < MAX_LINE_LENGTH_TO_FILTER)
        && (line_node.state.total_short_lines > int(MIN_DEPTH_FOR_LINE_REMOVING)
            || (!line_node.is_initial_line_touching_long_lines() && !line_node.has_next_layer_neighbours()));
}

// Remove the node and propagate its removal to the previous sections.
void propagate_line_node_remove(const LineNode &line_node) {
    std::queue<LineNode *> line_node_queue;
    for (LineNode *prev_line : line_node.prev_section_overlapping_lines) {
        if (prev_line->is_removed)
            continue;

        line_node_queue.emplace(prev_line);
    }

    for (; !line_node_queue.empty(); line_node_queue.pop()) {
        LineNode &line_to_check = *line_node_queue.front();

        if (can_line_note_be_removed(line_to_check)) {
            line_to_check.is_removed = true;

            for (LineNode *prev_line : line_to_check.prev_section_overlapping_lines) {
                if (prev_line->is_removed)
                    continue;

                line_node_queue.emplace(prev_line);
            }
        }
    }
}

// Filter out short extrusions that could create vibrations.
static std::vector<Lines> filter_vibrating_extrusions(const std::vector<Lines> &lines_sections) {
    // Initialize all line nodes.
    std::vector<LineNodes> line_nodes_sections(lines_sections.size());
    for (const Lines &lines_section : lines_sections) {
        const size_t section_idx = &lines_section - lines_sections.data();

        line_nodes_sections[section_idx].reserve(lines_section.size());
        for (const Line &line : lines_section) {
            line_nodes_sections[section_idx].emplace_back(line);
        }
    }

    // Precalculate for each line node which line nodes in the previous and next section this line node overlaps.
    for (auto curr_lines_section_it = line_nodes_sections.begin(); curr_lines_section_it != line_nodes_sections.end(); ++curr_lines_section_it) {
        if (curr_lines_section_it != line_nodes_sections.begin()) {
            const auto prev_lines_section_it = std::prev(curr_lines_section_it);
            for (LineNode &curr_line : *curr_lines_section_it) {
                for (LineNode &prev_line : *prev_lines_section_it) {
                    if (are_lines_overlapping_in_y_axes(curr_line.line, prev_line.line)) {
                        curr_line.prev_section_overlapping_lines.emplace_back(&prev_line);
                    }
                }
            }
        }

        if (std::next(curr_lines_section_it) != line_nodes_sections.end()) {
            const auto next_lines_section_it = std::next(curr_lines_section_it);
            for (LineNode &curr_line : *curr_lines_section_it) {
                for (LineNode &next_line : *next_lines_section_it) {
                    if (are_lines_overlapping_in_y_axes(curr_line.line, next_line.line)) {
                        curr_line.next_section_overlapping_lines.emplace_back(&next_line);
                    }
                }
            }
        }
    }

    const auto MAX_LINE_LENGTH_TO_FILTER = _MAX_LINE_LENGTH_TO_FILTER();
    // Select each section as the initial lines section and propagate line node states from this initial lines section to the last lines section.
    // During this propagation, we remove those lines that meet the conditions for its removal.
    // When some line is removed, we propagate this removal to previous layers.
    for (size_t initial_line_section_idx = 0; initial_line_section_idx < line_nodes_sections.size(); ++initial_line_section_idx) {
        // Stars from non-removed short lines.
        for (LineNode &initial_line : line_nodes_sections[initial_line_section_idx]) {
            if (initial_line.is_removed || initial_line.line.length() >= MAX_LINE_LENGTH_TO_FILTER)
                continue;

            initial_line.state.reset();
            initial_line.state.total_short_lines          = 1;
            initial_line.state.initial_touches_long_lines = initial_line.is_touching_long_lines_in_previous_layer();
            initial_line.state.initialized                = true;
        }

        // Iterate from the initial lines section until the last lines section.
        for (size_t propagation_line_section_idx = initial_line_section_idx; propagation_line_section_idx < line_nodes_sections.size(); ++propagation_line_section_idx) {
            // Before we propagate node states into next lines sections, we reset the state of all line nodes in the next line section.
            if (propagation_line_section_idx + 1 < line_nodes_sections.size()) {
                for (LineNode &propagation_line : line_nodes_sections[propagation_line_section_idx + 1]) {
                    propagation_line.state.reset();
                }
            }

            for (LineNode &propagation_line : line_nodes_sections[propagation_line_section_idx]) {
                if (propagation_line.is_removed || !propagation_line.state.initialized)
                    continue;

                for (LineNode *neighbour_line : propagation_line.next_section_overlapping_lines) {
                    if (neighbour_line->is_removed)
                        continue;

                    const bool is_short_line   = neighbour_line->line.length() < MAX_LINE_LENGTH_TO_FILTER;
                    const bool is_skip_allowed = propagation_line.state.min_skips_taken < int(MAX_SKIPS_ALLOWED);

                    if (!is_short_line && !is_skip_allowed)
                        continue;

                    const int neighbour_total_short_lines = propagation_line.state.total_short_lines + int(is_short_line);
                    const int neighbour_min_skips_taken   = propagation_line.state.min_skips_taken + int(!is_short_line);

                    if (neighbour_line->state.initialized) {
                        // When the state of the node was previously filled, then we need to update data in such a way
                        // that will maximize the possibility of removing this node.
                        neighbour_line->state.min_skips_taken = std::max(neighbour_line->state.min_skips_taken, neighbour_total_short_lines);
                        neighbour_line->state.min_skips_taken = std::min(neighbour_line->state.min_skips_taken, neighbour_min_skips_taken);

                        // We will keep updating neighbor initial_touches_long_lines until it is equal to false.
                        if (neighbour_line->state.initial_touches_long_lines) {
                            neighbour_line->state.initial_touches_long_lines = propagation_line.state.initial_touches_long_lines;
                        }
                    } else {
                        neighbour_line->state.total_short_lines          = neighbour_total_short_lines;
                        neighbour_line->state.min_skips_taken            = neighbour_min_skips_taken;
                        neighbour_line->state.initial_touches_long_lines = propagation_line.state.initial_touches_long_lines;
                        neighbour_line->state.initialized                = true;
                    }
                }

                if (can_line_note_be_removed(propagation_line)) {
                    // Remove the current node and propagate its removal to the previous sections.
                    propagation_line.is_removed = true;
                    propagate_line_node_remove(propagation_line);
                }
            }
        }
    }

    // Create lines sections without filtered-out lines.
    std::vector<Lines> lines_sections_out(line_nodes_sections.size());
    for (const std::vector<LineNode> &line_nodes_section : line_nodes_sections) {
        const size_t section_idx = &line_nodes_section - line_nodes_sections.data();

        for (const LineNode &line_node : line_nodes_section) {
            if (!line_node.is_removed) {
                lines_sections_out[section_idx].emplace_back(line_node.line);
            }
        }
    }

    return lines_sections_out;
}

void split_solid_surface(size_t layer_id, const SurfaceFill &fill, ExPolygons &normal_infill, ExPolygons &narrow_infill)
{
    assert(fill.surface.surface_type == stInternalSolid);

    const bool line_based_pattern =
        fill.params.pattern == ipRectilinear || fill.params.pattern == ipMonotonic ||
        fill.params.pattern == ipMonotonicLine || fill.params.pattern == ipAlignedRectilinear;

    // ORCA: For non-line patterns, split by a geometric "core" so only thin areas get rerouted.
    if (!line_based_pattern) {
        const coord_t scaled_spacing = scaled<coord_t>(fill.params.spacing);

        for (const ExPolygon &expolygon : fill.expolygons) {
            Polygons filled_area = to_polygons(expolygon);

            // "Core" area: open (erode+dilate) to drop thin features, then clamp back to the original polygon.
            Polygons inner_area  = intersection(filled_area, opening(filled_area, scaled_spacing, scaled_spacing));

            if (inner_area.empty()) {
                narrow_infill.emplace_back(expolygon);
                continue;
            }

            ExPolygons inner_ex = union_ex(inner_area);
            ExPolygons expolys{expolygon};
            ExPolygons narrow_ex = diff_ex(expolys, inner_ex);
            ExPolygons normal_ex = intersection_ex(expolys, inner_ex);

            append(normal_infill, normal_ex); // normal infill area
            append(narrow_infill, narrow_ex); // narrow infill area
        }

        return;
    }

    Polygons normal_fill_areas;  // Areas that filled with normal infill

    constexpr double connect_extrusions = true;

    const coord_t scaled_spacing                      = scaled<coord_t>(fill.params.spacing);
    double        distance_limit_reconnection         = 2.0 * double(scaled_spacing);
    double        squared_distance_limit_reconnection = distance_limit_reconnection * distance_limit_reconnection;
    // Calculate infill direction, see Fill::_infill_direction
    double        base_angle                          = fill.params.angle + float(M_PI / 2.);
    // For pattern other than ipAlignedRectilinear, the angle are alternated
    if (fill.params.pattern != ipAlignedRectilinear) {
        size_t idx = layer_id / fill.surface.thickness_layers;
        base_angle += (idx & 1) ? float(M_PI / 2.) : 0;
    }
    const double aligning_angle = -base_angle + PI;

	for (const ExPolygon &expolygon : fill.expolygons) {
        Polygons filled_area = to_polygons(expolygon);
        polygons_rotate(filled_area, aligning_angle);
        BoundingBox bb = get_extents(filled_area);

        Polygons inner_area = intersection(filled_area, opening(filled_area, 2 * scaled_spacing, 3 * scaled_spacing));

        inner_area = shrink(inner_area, scaled_spacing * 0.5 - scaled<double>(fill.params.overlap));

        AABBTreeLines::LinesDistancer<Line> area_walls{to_lines(inner_area)};

        const size_t  n_vlines = (bb.max.x() - bb.min.x() + scaled_spacing - 1) / scaled_spacing;
        const coord_t y_min    = bb.min.y();
        const coord_t y_max    = bb.max.y();
        Lines         vertical_lines(n_vlines);
        for (size_t i = 0; i < n_vlines; i++) {
            coord_t x           = bb.min.x() + i * double(scaled_spacing);
            vertical_lines[i].a = Point{x, y_min};
            vertical_lines[i].b = Point{x, y_max};
        }

        if (!vertical_lines.empty()) {
            vertical_lines.push_back(vertical_lines.back());
            vertical_lines.back().a = Point{coord_t(bb.min.x() + n_vlines * double(scaled_spacing) + scaled_spacing * 0.5), y_min};
            vertical_lines.back().b = Point{vertical_lines.back().a.x(), y_max};
        }

        std::vector<Lines> polygon_sections(n_vlines);

        for (size_t i = 0; i < n_vlines; i++) {
            const auto intersections = area_walls.intersections_with_line<true>(vertical_lines[i]);

            for (int intersection_idx = 0; intersection_idx < int(intersections.size()) - 1; intersection_idx++) {
                const auto &a = intersections[intersection_idx];
                const auto &b = intersections[intersection_idx + 1];
                if (area_walls.outside((a.first + b.first) / 2) < 0) {
                    if (std::abs(a.first.y() - b.first.y()) > scaled_spacing) {
                        polygon_sections[i].emplace_back(a.first, b.first);
                    }
                }
            }
        }

        polygon_sections = filter_vibrating_extrusions(polygon_sections);

        Polygons reconstructed_area{};
        // reconstruct polygon from polygon sections
        {
            struct TracedPoly
            {
                Points lows;
                Points highs;
            };

            std::vector<std::vector<Line>> polygon_sections_w_width = polygon_sections;
            for (auto &slice : polygon_sections_w_width) {
                for (Line &l : slice) {
                    l.a -= Point{0.0, 0.5 * scaled_spacing};
                    l.b += Point{0.0, 0.5 * scaled_spacing};
                }
            }

            std::vector<TracedPoly> current_traced_polys;
            for (const auto &polygon_slice : polygon_sections_w_width) {
                std::unordered_set<const Line *> used_segments;
                for (TracedPoly &traced_poly : current_traced_polys) {
                    auto candidates_begin = std::upper_bound(polygon_slice.begin(), polygon_slice.end(), traced_poly.lows.back(),
                                                             [](const Point &low, const Line &seg) { return seg.b.y() > low.y(); });
                    auto candidates_end   = std::upper_bound(polygon_slice.begin(), polygon_slice.end(), traced_poly.highs.back(),
                                                             [](const Point &high, const Line &seg) { return seg.a.y() > high.y(); });

                    bool segment_added = false;
                    for (auto candidate = candidates_begin; candidate != candidates_end && !segment_added; candidate++) {
                        if (used_segments.find(&(*candidate)) != used_segments.end()) {
                            continue;
                        }
                        if (connect_extrusions && (traced_poly.lows.back() - candidates_begin->a).cast<double>().squaredNorm() <
                                                      squared_distance_limit_reconnection) {
                            traced_poly.lows.push_back(candidates_begin->a);
                        } else {
                            traced_poly.lows.push_back(traced_poly.lows.back() + Point{scaled_spacing / 2, coord_t(0)});
                            traced_poly.lows.push_back(candidates_begin->a - Point{scaled_spacing / 2, 0});
                            traced_poly.lows.push_back(candidates_begin->a);
                        }

                        if (connect_extrusions && (traced_poly.highs.back() - candidates_begin->b).cast<double>().squaredNorm() <
                                                      squared_distance_limit_reconnection) {
                            traced_poly.highs.push_back(candidates_begin->b);
                        } else {
                            traced_poly.highs.push_back(traced_poly.highs.back() + Point{scaled_spacing / 2, 0});
                            traced_poly.highs.push_back(candidates_begin->b - Point{scaled_spacing / 2, 0});
                            traced_poly.highs.push_back(candidates_begin->b);
                        }
                        segment_added = true;
                        used_segments.insert(&(*candidates_begin));
                    }

                    if (!segment_added) {
                        // Zero or multiple overlapping segments. Resolving this is nontrivial,
                        // so we just close this polygon and maybe open several new. This will hopefully happen much less often
                        traced_poly.lows.push_back(traced_poly.lows.back() + Point{scaled_spacing / 2, 0});
                        traced_poly.highs.push_back(traced_poly.highs.back() + Point{scaled_spacing / 2, 0});
                        Polygon &new_poly = reconstructed_area.emplace_back(std::move(traced_poly.lows));
                        new_poly.points.insert(new_poly.points.end(), traced_poly.highs.rbegin(), traced_poly.highs.rend());
                        traced_poly.lows.clear();
                        traced_poly.highs.clear();
                    }
                }

                current_traced_polys.erase(std::remove_if(current_traced_polys.begin(), current_traced_polys.end(),
                                                          [](const TracedPoly &tp) { return tp.lows.empty(); }),
                                           current_traced_polys.end());

                for (const auto &segment : polygon_slice) {
                    if (used_segments.find(&segment) == used_segments.end()) {
                        TracedPoly &new_tp = current_traced_polys.emplace_back();
                        new_tp.lows.push_back(segment.a - Point{scaled_spacing / 2, 0});
                        new_tp.lows.push_back(segment.a);
                        new_tp.highs.push_back(segment.b - Point{scaled_spacing / 2, 0});
                        new_tp.highs.push_back(segment.b);
                    }
                }
            }

            // add not closed polys
            for (TracedPoly &traced_poly : current_traced_polys) {
                Polygon &new_poly = reconstructed_area.emplace_back(std::move(traced_poly.lows));
                new_poly.points.insert(new_poly.points.end(), traced_poly.highs.rbegin(), traced_poly.highs.rend());
            }
        }

        polygons_append(normal_fill_areas, reconstructed_area);
    }

    polygons_rotate(normal_fill_areas, -aligning_angle);

    // Do the split
    ExPolygons normal_fill_areas_ex = union_safety_offset_ex(normal_fill_areas);
    ExPolygons narrow_fill_areas    = diff_ex(fill.expolygons, normal_fill_areas_ex);

    // Merge very small areas that is smaller than a single line width to the normal infill if they touches
    for (auto iter = narrow_fill_areas.begin(); iter != narrow_fill_areas.end();) {
        auto shrinked_expoly = offset_ex(*iter, -scaled_spacing * 0.5);
        if (shrinked_expoly.empty()) {
            // Too small! Check if it touches any normal infills
            auto     expanede_exploy          = offset_ex(*iter, scaled_spacing * 0.3);
            Polygons normal_fill_area_clipped = ClipperUtils::clip_clipper_polygons_with_subject_bbox(normal_fill_areas_ex, get_extents(expanede_exploy));
            auto     touch_check              = intersection_ex(normal_fill_area_clipped, expanede_exploy);
            if (!touch_check.empty()) {
                normal_fill_areas_ex.emplace_back(*iter);
                iter = narrow_fill_areas.erase(iter);
                continue;
            }
        }
        iter++;
    }

    if (narrow_fill_areas.empty()) {
        // No split needed
        return;
    }

    // Expand the normal infills to avoid gaps between normal and narrow infills.
    // The inner_area was shrunk by scaled_spacing * 0.5, so we need to expand
    // by at least that amount to ensure proper coverage and avoid gaps.
    normal_infill = intersection_ex(offset_ex(normal_fill_areas_ex, scaled_spacing * 0.5), fill.expolygons);
    narrow_infill = narrow_fill_areas;

#ifdef DEBUG_SURFACE_SPLIT
    {
        BoundingBox bbox   = get_extents(fill.expolygons);
        bbox.offset(scale_(1.));
        ::Slic3r::SVG svg(debug_out_path("surface_split_%d.svg", layer_id), bbox);
        svg.draw(to_lines(fill.expolygons), "red", scale_(0.1));
        svg.draw(normal_infill, "blue", 0.5);
        svg.draw(narrow_infill, "green", 0.5);
        svg.Close();
    }
#endif
}

std::vector<SurfaceFill> group_fills(const Layer &layer, LockRegionParam &lock_param)
{
	std::vector<SurfaceFill> surface_fills;
	// Fill in a map of a region & surface to SurfaceFillParams.
	std::set<SurfaceFillParams> 						set_surface_params;
	std::vector<std::vector<const SurfaceFillParams*>> 	region_to_surface_params(layer.regions().size(), std::vector<const SurfaceFillParams*>());
    SurfaceFillParams									params;
    bool 												has_internal_voids = false;
	const PrintObjectConfig&							object_config = layer.object()->config();
    const std::vector<TopSurfaceImageRegionPlan>        top_surface_plans = top_surface_image_region_plans(layer);

	auto append_flow_param = [](std::map<Flow, ExPolygons> &flow_params, Flow flow, const ExPolygon &exp) {
        auto it = flow_params.find(flow);
        if (it == flow_params.end())
            flow_params.insert({flow, {exp}});
        else
            it->second.push_back(exp);
    };

	auto append_density_param = [](std::map<float, ExPolygons> &density_params, float density, const ExPolygon &exp) {
        auto it = density_params.find(density);
        if (it == density_params.end())
            density_params.insert({density, {exp}});
        else
            it->second.push_back(exp);
    };

	for (size_t region_id = 0; region_id < layer.regions().size(); ++ region_id) {
		const LayerRegion  &layerm = *layer.regions()[region_id];
		region_to_surface_params[region_id].assign(layerm.fill_surfaces.size(), nullptr);
	    for (const Surface &surface : layerm.fill_surfaces.surfaces)
	        if (surface.surface_type == stInternalVoid)
	        	has_internal_voids = true;
	        else {
		        const PrintRegionConfig &region_config = layerm.region().config();
		        FlowRole extrusion_role = surface.is_top() ? frTopSolidInfill : (surface.is_solid() ? frSolidInfill : frInfill);
		        bool     is_bridge 	    = layer.id() > 0 && surface.is_bridge();
		        params.extruder 	 = layerm.region().extruder(extrusion_role);
		        params.pattern 		 = region_config.sparse_infill_pattern.value;
		        params.density       = float(region_config.sparse_infill_density);
                params.lateral_lattice_angle_1 = region_config.lateral_lattice_angle_1;
                params.lateral_lattice_angle_2 = region_config.lateral_lattice_angle_2;
                params.infill_overhang_angle = region_config.infill_overhang_angle;
                if (params.pattern == ipLockedZag) {
                    params.infill_lock_depth = scale_(region_config.infill_lock_depth);
                    params.skin_infill_depth = scale_(region_config.skin_infill_depth);
                }
                if (params.pattern == ipCrossZag || params.pattern == ipLockedZag) {
                    params.symmetric_infill_y_axis = region_config.symmetric_infill_y_axis;
                } else if (params.pattern == ipZigZag) {
                    params.symmetric_infill_y_axis = region_config.symmetric_infill_y_axis;
                }

                if (surface.is_solid()) {
                    if (surface.is_external() && !is_bridge) {
                        if (surface.is_top()) {
                            params.pattern = region_config.top_surface_pattern.value;
                            params.density = float(region_config.top_surface_density);
                            if (params.density <= 0.0f) continue;
                        } else { // Surface is bottom
                            params.pattern = region_config.bottom_surface_pattern.value;
                            params.density = float(region_config.bottom_surface_density);
                        }
                    } else if (surface.is_solid_infill()) {
                        params.pattern = region_config.internal_solid_infill_pattern.value;
                        params.density = 100.f;
                    } else {
                        if (region_config.top_surface_pattern == ipMonotonic || region_config.top_surface_pattern == ipMonotonicLine)
                            params.pattern = ipMonotonic;
                        else
                            params.pattern = ipRectilinear;
                        params.density = 100.f;
                    }
                } else if (params.density <= 0)
                    continue;

				params.extrusion_role = erInternalInfill;
                if (is_bridge) {
                    if (surface.is_internal_bridge())
                        params.extrusion_role = erInternalBridgeInfill;
                    else
                        params.extrusion_role = erBridgeInfill;
                } else if (surface.is_solid()) {
                    if (surface.is_top()) {
                        params.extrusion_role = erTopSolidInfill;
                    } else if (surface.is_bottom()) {
                        params.extrusion_role = erBottomSurface;
                    } else {
                        params.extrusion_role = erSolidInfill;
                    }
                }
                // Orca: apply fill multiline only for sparse infill
                params.multiline = params.extrusion_role == erInternalInfill ? int(region_config.fill_multiline) : 1;

                if (params.extrusion_role == erInternalInfill) {
                    params.angle = calculate_infill_rotation_angle(layer.object(), layer.id(), region_config.infill_direction.value,
                                                                   region_config.sparse_infill_rotate_template.value);
                    params.fixed_angle = !region_config.sparse_infill_rotate_template.value.empty();
                } else {
                    params.angle = calculate_infill_rotation_angle(layer.object(), layer.id(), region_config.solid_infill_direction.value,
                                                                   region_config.solid_infill_rotate_template.value);
                    params.fixed_angle = !region_config.solid_infill_rotate_template.value.empty();
                }
                params.bridge_angle = float(surface.bridge_angle);
                
                if (region_config.align_infill_direction_to_model) {
                    auto m = layer.object()->trafo().matrix();
                    params.angle += atan2((float) m(1, 0), (float) m(0, 0));
                }

                // Calculate the actual flow we'll be using for this infill.
		        params.bridge = is_bridge || Fill::use_bridge_flow(params.pattern);
                const bool is_thick_bridge = surface.is_bridge() && (surface.is_internal_bridge() ? object_config.thick_internal_bridges : object_config.thick_bridges);
				params.flow   = params.bridge ?
					//Orca: enable thick bridge based on config
					layerm.bridging_flow(extrusion_role, is_thick_bridge) :
					layerm.flow(extrusion_role, (surface.thickness == -1) ? layer.height : surface.thickness);
				// record speed params
                if (!params.bridge) {
                    if (params.extrusion_role == erInternalInfill)
                        params.sparse_infill_speed = region_config.sparse_infill_speed;
                    else if (params.extrusion_role == erTopSolidInfill) {
                        params.top_surface_speed = region_config.top_surface_speed;
                    } else if (params.extrusion_role == erSolidInfill)
                        params.solid_infill_speed = region_config.internal_solid_infill_speed;
                }
				// Calculate flow spacing for infill pattern generation.
		        if (surface.is_solid() || is_bridge) {
		            params.spacing = params.flow.spacing();
		            // Don't limit anchor length for solid or bridging infill.
		            params.anchor_length = 1000.f;
					params.anchor_length_max = 1000.f;
		        } else {
					// Internal infill. Calculating infill line spacing independent of the current layer height and 1st layer status,
					// so that internall infill will be aligned over all layers of the current region.
		            params.spacing = layerm.region().flow(*layer.object(), frInfill, layer.object()->config().layer_height, false).spacing();
		            // Anchor a sparse infill to inner perimeters with the following anchor length:
			        params.anchor_length = float(region_config.infill_anchor);
					if (region_config.infill_anchor.percent)
						params.anchor_length = float(params.anchor_length * 0.01 * params.spacing);
					params.anchor_length_max = float(region_config.infill_anchor_max);
					if (region_config.infill_anchor_max.percent)
						params.anchor_length_max = float(params.anchor_length_max * 0.01 * params.spacing);
					params.anchor_length = std::min(params.anchor_length, params.anchor_length_max);
				}

				//get locked region param
				if (params.pattern == ipLockedZag){
					const PrintObject *object = layerm.layer()->object();
					auto nozzle_diameter = float(object->print()->config().nozzle_diameter.get_at(layerm.region().extruder(extrusion_role) - 1));
					Flow skin_flow = params.bridge ? params.flow : Flow::new_from_config_width(extrusion_role, region_config.skin_infill_line_width, nozzle_diameter, float((surface.thickness == -1) ? layer.height : surface.thickness));
					//add skin flow
					append_flow_param(lock_param.skin_flow_params, skin_flow, surface.expolygon);

					Flow skeleton_flow = params.bridge ? params.flow : Flow::new_from_config_width(extrusion_role, region_config.skeleton_infill_line_width, nozzle_diameter, float((surface.thickness == -1) ? layer.height : surface.thickness)) ;
					// add skeleton flow
					append_flow_param(lock_param.skeleton_flow_params, skeleton_flow, surface.expolygon);

					// add skin density
					append_density_param(lock_param.skin_density_params, float(0.01 * region_config.skin_infill_density), surface.expolygon);

					// add skin density
					append_density_param(lock_param.skeleton_density_params, float(0.01 * region_config.skeleton_infill_density), surface.expolygon);

				}

                auto it_params = set_surface_params.find(params);

		        if (it_params == set_surface_params.end())
		        	it_params = set_surface_params.insert(it_params, params);
		        region_to_surface_params[region_id][&surface - &layerm.fill_surfaces.surfaces.front()] = &(*it_params);
		    }
	}

	surface_fills.reserve(set_surface_params.size());
	for (const SurfaceFillParams &params : set_surface_params) {
		const_cast<SurfaceFillParams&>(params).idx = surface_fills.size();
		surface_fills.emplace_back(params);
	}

	for (size_t region_id = 0; region_id < layer.regions().size(); ++ region_id) {
		const LayerRegion &layerm = *layer.regions()[region_id];
	    for (const Surface &surface : layerm.fill_surfaces.surfaces)
	        if (surface.surface_type != stInternalVoid) {
	        	const SurfaceFillParams *params = region_to_surface_params[region_id][&surface - &layerm.fill_surfaces.surfaces.front()];
				if (params != nullptr) {
                    ExPolygons remaining = { surface.expolygon };
                    if (region_id < top_surface_plans.size() &&
                        top_surface_plans[region_id].zone != nullptr &&
                        (surface.is_top() || surface.surface_type == stInternalSolid) &&
                        !surface.is_bridge()) {
                        const TopSurfaceImageRegionPlan &plan = top_surface_plans[region_id];
                        for (size_t slice_idx = 0; slice_idx < plan.slices.size();) {
                            const TopSurfaceImageStackSlice &slice = plan.slices[slice_idx];
                            if (remaining.empty())
                                break;
                            ExPolygons image_expolygons = intersection_ex(remaining, slice.area, ApplySafetyOffset::Yes);
                            if (image_expolygons.empty()) {
                                ++slice_idx;
                                continue;
                            }
                            ExPolygons image_clip = image_expolygons;
                            if (slice.same_layer_partition) {
                                size_t same_depth_end = slice_idx;
                                while (same_depth_end < plan.slices.size() &&
                                       plan.slices[same_depth_end].same_layer_partition &&
                                       plan.slices[same_depth_end].depth == slice.depth)
                                    ++same_depth_end;
                                for (size_t same_idx = slice_idx; same_idx < same_depth_end; ++same_idx) {
                                    SurfaceFillParams image_params =
                                        top_surface_image_params_for_slice(layer, surface, *params, plan, plan.slices[same_idx]);
                                    ExPolygons component_expolygons = image_expolygons;
                                    SurfaceFill &image_fill = surface_fill_for_params(surface_fills, image_params);
                                    append_surface_fill_expolygons(image_fill, region_id, surface, std::move(component_expolygons), layerm);
                                }
                                slice_idx = same_depth_end;
                            } else {
                                SurfaceFillParams image_params =
                                    top_surface_image_params_for_slice(layer, surface, *params, plan, slice);
                                SurfaceFill &image_fill = surface_fill_for_params(surface_fills, image_params);
                                append_surface_fill_expolygons(image_fill, region_id, surface, std::move(image_expolygons), layerm);
                                ++slice_idx;
                            }
                            remaining = diff_ex(remaining, image_clip, ApplySafetyOffset::Yes);
                        }
                    }
                    if (!remaining.empty()) {
                        SurfaceFill &fill = surface_fills[params->idx];
                        append_surface_fill_expolygons(fill, region_id, surface, std::move(remaining), layerm);
                    }
				}
	        }
	}

	{
		Polygons all_polygons;
		for (SurfaceFill &fill : surface_fills)
			if (! fill.expolygons.empty()) {
                if (fill.params.texture_mapping_top_surface_same_layer_partition)
                    continue;
				if (fill.expolygons.size() > 1 || ! all_polygons.empty()) {
					Polygons polys = to_polygons(std::move(fill.expolygons));
		            // Make a union of polygons, use a safety offset, subtract the preceding polygons.
				    // Bridges are processed first (see SurfaceFill::operator<())
		            fill.expolygons = all_polygons.empty() ? union_safety_offset_ex(polys) : diff_ex(polys, all_polygons, ApplySafetyOffset::Yes);
					append(all_polygons, std::move(polys));
				} else if (&fill != &surface_fills.back())
					append(all_polygons, to_polygons(fill.expolygons));
	        }
	}

    // we need to detect any narrow surfaces that might collapse
    // when adding spacing below
    // such narrow surfaces are often generated in sloping walls
    // by bridge_over_infill() and combine_infill() as a result of the
    // subtraction of the combinable area from the layer infill area,
    // which leaves small areas near the perimeters
    // we are going to grow such regions by overlapping them with the void (if any)
    // TODO: detect and investigate whether there could be narrow regions without
    // any void neighbors
    if (has_internal_voids) {
    	// Internal voids are generated only if "infill_only_where_needed" or "infill_every_layers" are active.
        coord_t  distance_between_surfaces = 0;
        Polygons surfaces_polygons;
        Polygons voids;
		int      region_internal_infill = -1;
		int		 region_solid_infill = -1;
		int		 region_some_infill = -1;
    	for (SurfaceFill &surface_fill : surface_fills)
			if (! surface_fill.expolygons.empty()) {
    			distance_between_surfaces = std::max(distance_between_surfaces, surface_fill.params.flow.scaled_spacing());
				append((surface_fill.surface.surface_type == stInternalVoid) ? voids : surfaces_polygons, to_polygons(surface_fill.expolygons));
				if (surface_fill.surface.surface_type == stInternalSolid)
					region_internal_infill = (int)surface_fill.region_id;
				if (surface_fill.surface.is_solid())
					region_solid_infill = (int)surface_fill.region_id;
				if (surface_fill.surface.surface_type != stInternalVoid)
					region_some_infill = (int)surface_fill.region_id;
			}
    	if (! voids.empty() && ! surfaces_polygons.empty()) {
    		// First clip voids by the printing polygons, as the voids were ignored by the loop above during mutual clipping.
    		voids = diff(voids, surfaces_polygons);
	        // Corners of infill regions, which would not be filled with an extrusion path with a radius of distance_between_surfaces/2
	        Polygons collapsed = diff(
	            surfaces_polygons,
				opening(surfaces_polygons, float(distance_between_surfaces /2), float(distance_between_surfaces / 2 + ClipperSafetyOffset)));
	        //FIXME why the voids are added to collapsed here? First it is expensive, second the result may lead to some unwanted regions being
	        // added if two offsetted void regions merge.
	        // polygons_append(voids, collapsed);
	        ExPolygons extensions = intersection_ex(expand(collapsed, float(distance_between_surfaces)), voids, ApplySafetyOffset::Yes);
	        // Now find an internal infill SurfaceFill to add these extrusions to.
	        SurfaceFill *internal_solid_fill = nullptr;
			unsigned int region_id = 0;
			if (region_internal_infill != -1)
				region_id = region_internal_infill;
			else if (region_solid_infill != -1)
				region_id = region_solid_infill;
			else if (region_some_infill != -1)
				region_id = region_some_infill;
			const LayerRegion& layerm = *layer.regions()[region_id];
	        for (SurfaceFill &surface_fill : surface_fills)
                if (surface_fill.surface.surface_type == stInternalSolid &&
                    !surface_fill.params.texture_mapping_top_surface_image &&
                    std::abs(layer.height - surface_fill.params.flow.height()) < EPSILON) {
	        		internal_solid_fill = &surface_fill;
	        		break;
	        	}
	        if (internal_solid_fill == nullptr) {
	        	// Produce another solid fill.
		        params.extruder 	 = layerm.region().extruder(frSolidInfill);
                const auto top_pattern = layerm.region().config().top_surface_pattern;
                if(top_pattern == ipMonotonic || top_pattern == ipMonotonicLine)
                    params.pattern = top_pattern;
                else
                    params.pattern 		 = ipRectilinear;
	            params.density 		 = 100.f;
		        params.extrusion_role = erSolidInfill;
		        const PrintRegionConfig &region_config = layerm.region().config();
                params.angle = calculate_infill_rotation_angle(layer.object(), layer.id(), region_config.solid_infill_direction.value,
                                                               region_config.solid_infill_rotate_template.value);
                params.fixed_angle = !region_config.solid_infill_rotate_template.value.empty();

                // calculate the actual flow we'll be using for this infill
				params.flow = layerm.flow(frSolidInfill);
		        params.spacing = params.flow.spacing();
				surface_fills.emplace_back(params);
				surface_fills.back().surface.surface_type = stInternalSolid;
				surface_fills.back().surface.thickness = layer.height;
				surface_fills.back().expolygons = std::move(extensions);
	        } else {
	        	append(extensions, std::move(internal_solid_fill->expolygons));
	        	internal_solid_fill->expolygons = union_ex(extensions);
	        }
		}
    }

	// BBS: detect narrow internal solid infill area and use ipConcentricInternal pattern instead
	if (layer.object()->config().detect_narrow_internal_solid_infill) {
		size_t surface_fills_size = surface_fills.size();
		for (size_t i = 0; i < surface_fills_size; i++) {
			if (surface_fills[i].surface.surface_type != stInternalSolid ||
                surface_fills[i].params.texture_mapping_top_surface_image)
				continue;

			ExPolygons normal_infill;
            ExPolygons narrow_infill;
            split_solid_surface(layer.id(), surface_fills[i], normal_infill, narrow_infill);

			if (narrow_infill.empty()) {
				// BBS: has no narrow expolygon
				continue;
			} else if (normal_infill.empty()) {
				// BBS: all expolygons are narrow, directly change the fill pattern
				surface_fills[i].params.pattern = ipConcentricInternal;
			}
			else {
				// BBS: some expolygons are narrow, spilit surface_fills[i] and rearrange the expolygons
				params = surface_fills[i].params;
				params.pattern = ipConcentricInternal;
				surface_fills.emplace_back(params);
				surface_fills.back().region_id = surface_fills[i].region_id;
				surface_fills.back().surface.surface_type = stInternalSolid;
				surface_fills.back().surface.thickness = surface_fills[i].surface.thickness;
                surface_fills.back().region_id_group       = surface_fills[i].region_id_group;
                surface_fills.back().no_overlap_expolygons = surface_fills[i].no_overlap_expolygons;
			    // BBS: move the narrow expolygons to new surface_fills.back();
			    surface_fills.back().expolygons = std::move(narrow_infill);
			    // BBS: delete the narrow expolygons from old surface_fills
                surface_fills[i].expolygons = std::move(normal_infill);
			}
		}
	}

	return surface_fills;
}

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
void export_group_fills_to_svg(const char *path, const std::vector<SurfaceFill> &fills)
{
    BoundingBox bbox;
    for (const auto &fill : fills)
        for (const auto &expoly : fill.expolygons)
            bbox.merge(get_extents(expoly));
    Point legend_size = export_surface_type_legend_to_svg_box_size();
    Point legend_pos(bbox.min(0), bbox.max(1));
    bbox.merge(Point(std::max(bbox.min(0) + legend_size(0), bbox.max(0)), bbox.max(1) + legend_size(1)));

    SVG svg(path, bbox);
    const float transparency = 0.5f;
    for (const auto &fill : fills)
        for (const auto &expoly : fill.expolygons)
            svg.draw(expoly, surface_type_to_color_name(fill.surface.surface_type), transparency);
    export_surface_type_legend_to_svg(svg, legend_pos);
    svg.Close();
}
#endif

// friend to Layer
void Layer::make_fills(FillAdaptive::Octree* adaptive_fill_octree, FillAdaptive::Octree* support_fill_octree, FillLightning::Generator* lightning_generator)
{
	for (LayerRegion *layerm : m_regions)
		layerm->fills.clear();


#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
//	this->export_region_fill_surfaces_to_svg_debug("10_fill-initial");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */
    LockRegionParam lock_param;
    std::vector<SurfaceFill>     surface_fills = group_fills(*this, lock_param);
	const Slic3r::BoundingBox bbox 			= this->object()->bounding_box();
	const auto                resolution 	= this->object()->print()->config().resolution.value;

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
	{
		static int iRun = 0;
		export_group_fills_to_svg(debug_out_path("Layer-fill_surfaces-10_fill-final-%d.svg", iRun ++).c_str(), surface_fills);
	}
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */

    for (SurfaceFill &surface_fill : surface_fills) {
        // Create the filler object.
        std::unique_ptr<Fill> f = std::unique_ptr<Fill>(Fill::new_from_type(surface_fill.params.pattern));
        f->set_bounding_box(bbox);
        f->layer_id = this->id();
        f->z 		= this->print_z;
        f->angle 	= surface_fill.params.angle;
        f->fixed_angle = surface_fill.params.fixed_angle;
        f->adapt_fill_octree   = (surface_fill.params.pattern == ipSupportCubic) ? support_fill_octree : adaptive_fill_octree;
        f->print_config        = &this->object()->print()->config();
        f->print_object_config = &this->object()->config();
		if (surface_fill.params.pattern == ipConcentricInternal) {
            FillConcentricInternal *fill_concentric = dynamic_cast<FillConcentricInternal *>(f.get());
            assert(fill_concentric != nullptr);
            fill_concentric->print_config        = &this->object()->print()->config();
            fill_concentric->print_object_config = &this->object()->config();
        } else if (surface_fill.params.pattern == ipConcentric) {
            FillConcentric *fill_concentric = dynamic_cast<FillConcentric *>(f.get());
            assert(fill_concentric != nullptr);
            fill_concentric->print_config = &this->object()->print()->config();
            fill_concentric->print_object_config = &this->object()->config();
        } else if (surface_fill.params.pattern == ipLightning)
            dynamic_cast<FillLightning::Filler*>(f.get())->generator = lightning_generator;
        // calculate flow spacing for infill pattern generation
        bool using_internal_flow = ! surface_fill.surface.is_solid() && ! surface_fill.params.bridge;
        double link_max_length = 0.;
        if (! surface_fill.params.bridge) {
#if 0
            link_max_length = layerm.region()->config().get_abs_value(surface.is_external() ? "external_fill_link_max_length" : "fill_link_max_length", flow.spacing());
//            printf("flow spacing: %f,  is_external: %d, link_max_length: %lf\n", flow.spacing(), int(surface.is_external()), link_max_length);
#else
            if (surface_fill.params.density > 80.) // 80%
                link_max_length = 3. * f->spacing;
#endif
        }

        LayerRegion* layerm = this->m_regions[surface_fill.region_id];

        // Maximum length of the perimeter segment linking two infill lines.
        f->link_max_length = (coord_t)scale_(link_max_length);
        // Used by the concentric infill pattern to clip the loops to create extrusion paths.
        f->loop_clipping = coord_t(scale_(layerm->region().config().seam_gap.get_abs_value(surface_fill.params.flow.nozzle_diameter())));

        // apply half spacing using this flow's own spacing and generate infill
        FillParams params;
        params.density 		     = float(0.01 * surface_fill.params.density);
        params.multiline         = surface_fill.params.multiline;
		params.dont_adjust		 = false; //  surface_fill.params.dont_adjust;
        params.anchor_length     = surface_fill.params.anchor_length;
		params.anchor_length_max = surface_fill.params.anchor_length_max;
		params.resolution        = resolution;
        params.use_arachne       = surface_fill.params.pattern == ipConcentric || surface_fill.params.pattern == ipConcentricInternal;
        params.layer_height      = layerm->layer()->height;
        params.lateral_lattice_angle_1   = surface_fill.params.lateral_lattice_angle_1;
        params.lateral_lattice_angle_2   = surface_fill.params.lateral_lattice_angle_2;
        params.infill_overhang_angle   = surface_fill.params.infill_overhang_angle;

		// BBS
		params.flow = surface_fill.params.flow;
		params.extrusion_role = surface_fill.params.extrusion_role;
		params.using_internal_flow = using_internal_flow;
		params.no_extrusion_overlap = surface_fill.params.overlap;
        auto &region_config = layerm->region().config();
        params.config               = &region_config;
        params.pattern              = surface_fill.params.pattern;

        if( surface_fill.params.pattern == ipLockedZag ) {
			params.locked_zag = true;
            params.infill_lock_depth = surface_fill.params.infill_lock_depth;
            params.skin_infill_depth = surface_fill.params.skin_infill_depth;
            f->set_lock_region_param(lock_param);
		}
        if (surface_fill.params.pattern == ipCrossZag || surface_fill.params.pattern == ipLockedZag) {
            if (f->layer_id % 2 == 0) {
                params.horiz_move -= scale_(region_config.infill_shift_step) * (f->layer_id / 2);
            } else {
                params.horiz_move += scale_(region_config.infill_shift_step) * (f->layer_id / 2);
            }

            params.symmetric_infill_y_axis = surface_fill.params.symmetric_infill_y_axis;

        } else if (surface_fill.params.pattern == ipZigZag) {
            params.symmetric_infill_y_axis = surface_fill.params.symmetric_infill_y_axis;

        }
		if (surface_fill.params.pattern == ipGrid)
			params.can_reverse = false;

        std::optional<TextureMappingOffsetContext> top_surface_image_context;
        if (surface_fill.params.texture_mapping_top_surface_image) {
            const Print *print = this->object()->print();
            const TextureMappingZone *zone = print != nullptr ?
                print->texture_mapping_manager().zone_from_id(surface_fill.params.texture_mapping_top_surface_zone_id) :
                nullptr;
            if (print != nullptr && zone != nullptr) {
                const PrintConfig &print_config = print->config();
                std::vector<std::string> filament_colours = print_config.filament_colour.values;
                filament_colours.resize(print_config.filament_colour.values.size(), "#FFFFFF");
                std::vector<unsigned int> components =
                    TextureMappingManager::effective_texture_component_ids(*zone,
                                                                           print_config.filament_colour.values.size(),
                                                                           filament_colours);
                const std::optional<std::array<float, 4>> background =
                    top_surface_image_equal_blend_background(print_config, components);
                top_surface_image_context =
                    build_texture_mapping_offset_context_for_layer(*this->object(),
                                                                   *this,
                                                                   *zone,
                                                                   surface_fill.params.texture_mapping_top_surface_zone_id,
                                                                   surface_fill.params.texture_mapping_top_surface_component_id,
                                                                   surface_fill.params.texture_mapping_top_surface_max_width_mm,
                                                                   float(this->height),
                                                                   std::nullopt,
                                                                   surface_fill.params.texture_mapping_top_surface_min_width_mm,
                                                                   background);
            }
        }

		for (ExPolygon& expoly : surface_fill.expolygons) {

      f->no_overlap_expolygons = intersection_ex(surface_fill.no_overlap_expolygons, ExPolygons() = {expoly}, ApplySafetyOffset::Yes);
            if (params.symmetric_infill_y_axis) {
                params.symmetric_y_axis = f->extended_object_bounding_box().center().x();
                expoly.symmetric_y(params.symmetric_y_axis);
            }

			// Spacing is modified by the filler to indicate adjustments. Reset it for each expolygon.
			f->spacing = surface_fill.params.spacing;
			surface_fill.surface.expolygon = std::move(expoly);

			if(surface_fill.params.bridge && surface_fill.surface.is_external() && surface_fill.params.density > 99.0){
				params.density = layerm->region().config().bridge_density.get_abs_value(1.0);
				params.dont_adjust = true;
			}
            if(surface_fill.surface.is_internal_bridge()){
                params.density = f->print_object_config->internal_bridge_density.get_abs_value(1.0);
                params.dont_adjust = true;
            }
            // Orca: Elefant foot compensation for solid layers above bottommost by infill density manipulation.
            float elefant_density = f->print_object_config->elefant_foot_layers_density.get_abs_value(1.0);
            if (!is_approx(elefant_density, 1.0f) && surface_fill.surface.is_solid_infill()) {
                size_t elefant_layers = f->print_object_config->elefant_foot_compensation_layers.value;
                if (f->layer_id > 0 && f->layer_id <= elefant_layers)
                    params.density = elefant_density * (elefant_layers - (f->layer_id - 1)) / elefant_layers;
            }
            // make fill
            ExtrusionEntitiesPtr &fill_entities = m_regions[surface_fill.region_id]->fills.entities;
            if (surface_fill.params.texture_mapping_top_surface_same_layer_partition) {
                ExtrusionEntityCollection *collection = new ExtrusionEntityCollection();
                top_surface_image_same_layer_partition_fill(*collection,
                                                            surface_fill.surface,
                                                            surface_fill.params,
                                                            top_surface_image_context);
                if (!collection->empty()) {
                    apply_top_surface_image_collection_metadata(*collection, surface_fill.params, std::nullopt);
                    fill_entities.push_back(collection);
                } else {
                    delete collection;
                }
            } else {
                const size_t fill_entities_before = fill_entities.size();
			    f->fill_surface_extrusion(&surface_fill.surface,
				    params,
				    fill_entities);
                if (surface_fill.params.texture_mapping_top_surface_image) {
                    for (size_t i = fill_entities_before; i < fill_entities.size(); ++i)
                        if (ExtrusionEntityCollection *collection = dynamic_cast<ExtrusionEntityCollection *>(fill_entities[i]))
                            apply_top_surface_image_collection_metadata(*collection, surface_fill.params, top_surface_image_context);
                }
            }
		}
    }

    // add thin fill regions
    // Unpacks the collection, creates multiple collections per path.
    // The path type could be ExtrusionPath, ExtrusionLoop or ExtrusionEntityCollection.
    // Why the paths are unpacked?
	for (LayerRegion *layerm : m_regions)
	    for (const ExtrusionEntity *thin_fill : layerm->thin_fills.entities) {
	        ExtrusionEntityCollection &collection = *(new ExtrusionEntityCollection());
            if (const auto *thin_fill_collection = dynamic_cast<const ExtrusionEntityCollection *>(thin_fill)) {
                collection.texture_mapping_extruder_override = thin_fill_collection->texture_mapping_extruder_override;
                collection.texture_mapping_top_surface_image = thin_fill_collection->texture_mapping_top_surface_image;
                collection.texture_mapping_top_surface_zone_id = thin_fill_collection->texture_mapping_top_surface_zone_id;
                collection.texture_mapping_top_surface_desired_component_id = thin_fill_collection->texture_mapping_top_surface_desired_component_id;
                collection.texture_mapping_top_surface_stack_depth = thin_fill_collection->texture_mapping_top_surface_stack_depth;
                collection.texture_mapping_top_surface_fixed_coloring = thin_fill_collection->texture_mapping_top_surface_fixed_coloring;
            }
	        layerm->fills.entities.push_back(&collection);
	        collection.entities.push_back(thin_fill->clone());
	    }

#ifndef NDEBUG
	for (LayerRegion *layerm : m_regions)
	    for (size_t i = 0; i < layerm->fills.entities.size(); ++ i)
    	    assert(dynamic_cast<ExtrusionEntityCollection*>(layerm->fills.entities[i]) != nullptr);
#endif
}
/**
 * Generate sparse-infill polylines for anchoring/analysis purposes.
 *
 * This produces the geometric polylines of internal sparse infill for the current
 * layer (using the same infill pattern, angle, rotation template, and spacing that
 * normal slicing would use), but it does not create extrusion entities.
 *
 * The returned polylines are consumed by internal-bridge detection on the next
 * layer to derive anchor lines and compute the bridge direction over sparse infill.
 *
 * Notes:
 * - Only `stInternal` surfaces are considered.
 * - Rotation templates (e.g. `sparse_infill_rotate_template`) are applied so the
 *   anchors reflect the actual infill orientation.
 * - For lightning/adaptive patterns, the respective generators are wired so their
 *   polylines match the final infill layout.
 */
Polylines Layer::generate_sparse_infill_polylines_for_anchoring(FillAdaptive::Octree* adaptive_fill_octree, FillAdaptive::Octree* support_fill_octree,  FillLightning::Generator* lightning_generator) const
{
    LockRegionParam skin_inner_param;
    std::vector<SurfaceFill> surface_fills = group_fills(*this, skin_inner_param);
	const Slic3r::BoundingBox bbox = this->object()->bounding_box();
	const auto                resolution = this->object()->print()->config().resolution.value;

    Polylines sparse_infill_polylines{};

    for (SurfaceFill &surface_fill : surface_fills) {
		if (surface_fill.surface.surface_type != stInternal) {
			continue;
		}

        switch (surface_fill.params.pattern) {
        case ipCount: continue; break;
        case ipSupportBase: continue; break;
        case ipConcentricInternal: continue; break;
        case ipLightning:
		case ipAdaptiveCubic:
        case ipSupportCubic:
        case ipRectilinear:
        case ipMonotonic:
        case ipMonotonicLine:
        case ipAlignedRectilinear:
        case ipGrid:
        case ipLateralLattice:
        case ipTriangles:
        case ipStars:
        case ipCubic:
        case ipLine:
        case ipConcentric:
        case ipHoneycomb:
        case ipLateralHoneycomb:
        case ip3DHoneycomb:
        case ipGyroid:
        case ipTpmsD:
        case ipTpmsFK:
        case ipHilbertCurve:
        case ipArchimedeanChords:
        case ipOctagramSpiral:
        case ipZigZag:
        case ipCrossZag:
		case ipLockedZag: break;
        }

        // Create the filler object.
        std::unique_ptr<Fill> f = std::unique_ptr<Fill>(Fill::new_from_type(surface_fill.params.pattern));
        f->set_bounding_box(bbox);
        f->layer_id = this->id() - this->object()->get_layer(0)->id(); // We need to subtract raft layers.
        f->z        = this->print_z;
        f->angle    = surface_fill.params.angle;
        f->fixed_angle = surface_fill.params.fixed_angle;
        f->adapt_fill_octree   = (surface_fill.params.pattern == ipSupportCubic) ? support_fill_octree : adaptive_fill_octree;
        f->print_config        = &this->object()->print()->config();
        f->print_object_config = &this->object()->config();

        if (surface_fill.params.pattern == ipLightning)
            dynamic_cast<FillLightning::Filler *>(f.get())->generator = lightning_generator;

        // calculate flow spacing for infill pattern generation
        double link_max_length = 0.;
        if (!surface_fill.params.bridge) {
#if 0
            link_max_length = layerm.region()->config().get_abs_value(surface.is_external() ? "external_fill_link_max_length" : "fill_link_max_length", flow.spacing());
//            printf("flow spacing: %f,  is_external: %d, link_max_length: %lf\n", flow.spacing(), int(surface.is_external()), link_max_length);
#else
            if (surface_fill.params.density > 80.) // 80%
                link_max_length = 3. * f->spacing;
#endif
        }

        LayerRegion &layerm = *m_regions[surface_fill.region_id];

        // Maximum length of the perimeter segment linking two infill lines.
        f->link_max_length = (coord_t) scale_(link_max_length);
        // Used by the concentric infill pattern to clip the loops to create extrusion paths.
        f->loop_clipping = coord_t(scale_(layerm.region().config().seam_gap.get_abs_value(surface_fill.params.flow.nozzle_diameter())));

        // apply half spacing using this flow's own spacing and generate infill
        FillParams params;
        params.density           = float(0.01 * surface_fill.params.density);
        params.dont_adjust       = false; //  surface_fill.params.dont_adjust;
        params.anchor_length     = surface_fill.params.anchor_length;
        params.anchor_length_max = surface_fill.params.anchor_length_max;
        params.resolution        = resolution;
        params.use_arachne       = false;
        params.layer_height      = layerm.layer()->height;
        params.lateral_lattice_angle_1   = surface_fill.params.lateral_lattice_angle_1;
        params.lateral_lattice_angle_2   = surface_fill.params.lateral_lattice_angle_2;
        params.infill_overhang_angle   = surface_fill.params.infill_overhang_angle;
        params.multiline         = surface_fill.params.multiline;

        for (ExPolygon &expoly : surface_fill.expolygons) {
            // Spacing is modified by the filler to indicate adjustments. Reset it for each expolygon.
            f->spacing                     = surface_fill.params.spacing;
            surface_fill.surface.expolygon = std::move(expoly);
            try {
                Polylines polylines = f->fill_surface(&surface_fill.surface, params);
                sparse_infill_polylines.insert(sparse_infill_polylines.end(), polylines.begin(), polylines.end());
            } catch (InfillFailedException &) {}
        }
    }

    return sparse_infill_polylines;
}

// Create ironing extrusions over top surfaces.
void Layer::make_ironing()
{
	// LayerRegion::slices contains surfaces marked with SurfaceType.
	// Here we want to collect top surfaces extruded with the same extruder.
	// A surface will be ironed with the same extruder to not contaminate the print with another material leaking from the nozzle.

	// First classify regions based on the extruder used.
	struct IroningParams {
		InfillPattern pattern;
		int 		extruder 	= -1;
		bool 		just_infill = false;
		// Spacing of the ironing lines, also to calculate the extrusion flow from.
		double 		line_spacing;
		// Height of the extrusion, to calculate the extrusion flow from.
		double 		height;
		double 		speed;
		double 		angle;
        bool        fixed_angle;
        double 		inset;

		bool operator<(const IroningParams &rhs) const {
            RETURN_COMPARE_NON_EQUAL(extruder);
            RETURN_COMPARE_NON_EQUAL(just_infill);
            RETURN_COMPARE_NON_EQUAL(line_spacing);
            RETURN_COMPARE_NON_EQUAL(height);
            RETURN_COMPARE_NON_EQUAL(speed);
            RETURN_COMPARE_NON_EQUAL(angle);
            RETURN_COMPARE_NON_EQUAL(fixed_angle);
            RETURN_COMPARE_NON_EQUAL(inset);
			return false;
		}

		bool operator==(const IroningParams &rhs) const {
			return  this->extruder == rhs.extruder  && 
                    this->just_infill == rhs.just_infill &&
				    this->line_spacing == rhs.line_spacing && 
                    this->height == rhs.height && 
                    this->speed == rhs.speed && 
                    this->angle == rhs.angle && 
                    this->fixed_angle == rhs.fixed_angle && 
                    this->pattern == rhs.pattern && 
                    this->inset == rhs.inset;
		}

		LayerRegion *layerm		= nullptr;

		// IdeaMaker: ironing
		// ironing flowrate (5% percent)
		// ironing speed (10 mm/sec)

		// Kisslicer:
		// iron off, Sweep, Group
		// ironing speed: 15 mm/sec

		// Cura:
		// Pattern (zig-zag / concentric)
		// line spacing (0.1mm)
		// flow: from normal layer height. 10%
		// speed: 20 mm/sec
	};

	std::vector<IroningParams> by_extruder;
    double default_layer_height = this->object()->config().layer_height;

	for (LayerRegion *layerm : m_regions)
		if (! layerm->slices.empty()) {
			IroningParams ironing_params;
			const PrintRegionConfig &config = layerm->region().config();
			if (config.ironing_type != IroningType::NoIroning &&
			    (config.ironing_type == IroningType::AllSolid ||
				    ((config.top_shell_layers > 0 || (this->object()->print()->config().spiral_mode && config.bottom_shell_layers > 1)) &&
					    (config.ironing_type == IroningType::TopSurfaces ||
					        (config.ironing_type == IroningType::TopmostOnly && layerm->layer()->upper_layer == nullptr))))) {
				if (config.wall_filament == config.solid_infill_filament || config.wall_loops == 0) {
					// Iron the whole face.
					ironing_params.extruder = config.solid_infill_filament;
				} else {
					// Iron just the infill.
					ironing_params.extruder = config.solid_infill_filament;
				}
			}
			if (ironing_params.extruder != -1) {
				//TODO just_infill is currently not used.
				ironing_params.just_infill 	= false;
				// Get filament-specific overrides if configured, otherwise use default values
				size_t extruder_idx = ironing_params.extruder - 1;
				ironing_params.line_spacing = (!config.filament_ironing_spacing.is_nil(extruder_idx)
					? config.filament_ironing_spacing.get_at(extruder_idx)
					: config.ironing_spacing);
                ironing_params.inset = (!config.filament_ironing_inset.is_nil(extruder_idx)
					? config.filament_ironing_inset.get_at(extruder_idx)
					: config.ironing_inset);
				ironing_params.height = default_layer_height * 0.01 * (!config.filament_ironing_flow.is_nil(extruder_idx)
					? config.filament_ironing_flow.get_at(extruder_idx)
					: config.ironing_flow);
				ironing_params.speed = (!config.filament_ironing_speed.is_nil(extruder_idx)
					? config.filament_ironing_speed.get_at(extruder_idx)
					: config.ironing_speed);
                ironing_params.angle        = (config.ironing_angle_fixed ? 0 : calculate_infill_rotation_angle(this->object(), this->id(), config.solid_infill_direction.value, config.solid_infill_rotate_template.value)) + config.ironing_angle * M_PI / 180.;
                ironing_params.fixed_angle = config.ironing_angle_fixed || !config.solid_infill_rotate_template.value.empty();
				ironing_params.pattern      = config.ironing_pattern;
				ironing_params.layerm 		= layerm;
				by_extruder.emplace_back(ironing_params);
			}
		}
	std::sort(by_extruder.begin(), by_extruder.end());

    FillParams 			fill_params;
    fill_params.density 	 = 1.;
    fill_params.monotonic    = true;
    InfillPattern         f_pattern = ipRectilinear;
    std::unique_ptr<Fill> f         = std::unique_ptr<Fill>(Fill::new_from_type(f_pattern));
    f->set_bounding_box(this->object()->bounding_box());
    f->layer_id = this->id();
    f->z        = this->print_z;
    f->overlap  = 0;
	for (size_t i = 0; i < by_extruder.size();) {
		// Find span of regions equivalent to the ironing operation.
		IroningParams &ironing_params = by_extruder[i];
		// Create the filler object.
		if( f_pattern != ironing_params.pattern )
		{
            f_pattern               = ironing_params.pattern;
            f = std::unique_ptr<Fill>(Fill::new_from_type(f_pattern));
            f->set_bounding_box(this->object()->bounding_box());
            f->layer_id = this->id();
            f->z        = this->print_z;
            f->overlap  = 0;
		}

		size_t j = i;
		for (++ j; j < by_extruder.size() && ironing_params == by_extruder[j]; ++ j) ;

		// Create the ironing extrusions for regions <i, j)
		ExPolygons ironing_areas;
		double nozzle_dmr = this->object()->print()->config().nozzle_diameter.get_at(ironing_params.extruder - 1);
		if (ironing_params.just_infill) {
			//TODO just_infill is currently not used.
			// Just infill.
		} else {
			// Infill and perimeter.
			// Merge top surfaces with the same ironing parameters.
			Polygons polys;
			Polygons infills;
			for (size_t k = i; k < j; ++ k) {
				const IroningParams		 &ironing_params  = by_extruder[k];
				const PrintRegionConfig  &region_config   = ironing_params.layerm->region().config();
				bool					  iron_everything = region_config.ironing_type == IroningType::AllSolid;
				bool					  iron_completely = iron_everything;
				if (iron_everything) {
					// Check whether there is any non-solid hole in the regions.
					bool internal_infill_solid = region_config.sparse_infill_density.value > 95.;
					for (const Surface &surface : ironing_params.layerm->fill_surfaces.surfaces)
						if ((!internal_infill_solid && surface.surface_type == stInternal) || surface.surface_type == stInternalBridge || surface.surface_type == stInternalVoid) {
							// Some fill region is not quite solid. Don't iron over the whole surface.
							iron_completely = false;
							break;
						}
				}
				if (iron_completely) {
					// Iron everything. This is likely only good for solid transparent objects.
					for (const Surface &surface : ironing_params.layerm->slices.surfaces)
						polygons_append(polys, surface.expolygon);
				} else {
					for (const Surface &surface : ironing_params.layerm->slices.surfaces)
						if ((surface.surface_type == stTop && (region_config.top_shell_layers > 0 || this->object()->print()->config().spiral_mode)) || (iron_everything && surface.surface_type == stBottom && region_config.bottom_shell_layers > 0))
							// stBottomBridge is not being ironed on purpose, as it would likely destroy the bridges.
							polygons_append(polys, surface.expolygon);
				}
				if (iron_everything && ! iron_completely) {
					// Add solid fill surfaces. This may not be ideal, as one will not iron perimeters touching these
					// solid fill surfaces, but it is likely better than nothing.
					for (const Surface &surface : ironing_params.layerm->fill_surfaces.surfaces)
						if (surface.surface_type == stInternalSolid)
							polygons_append(infills, surface.expolygon);
				}
			}

			if (! infills.empty() || j > i + 1) {
				// Ironing over more than a single region or over solid internal infill.
				if (! infills.empty())
					// For IroningType::AllSolid only:
					// Add solid infill areas for layers, that contain some non-ironable infil (sparse infill, bridge infill).
					append(polys, std::move(infills));
				polys = union_safety_offset(polys);
			}
			// Trim the top surfaces with half the nozzle diameter.
            // BBS: ironing inset
            double ironing_areas_offset = ironing_params.inset == 0 ? float(scale_(0.5 * nozzle_dmr)) : scale_(ironing_params.inset);
			ironing_areas = intersection_ex(polys, offset(this->lslices, - ironing_areas_offset));
		}

        // Create the filler object.
        f->spacing = ironing_params.line_spacing;
        f->angle = float(ironing_params.angle);
        f->fixed_angle = ironing_params.fixed_angle;
        f->link_max_length = (coord_t) scale_(3. * f->spacing);
		double  extrusion_height = ironing_params.height * f->spacing / nozzle_dmr;
		float  extrusion_width  = Flow::rounded_rectangle_extrusion_width_from_spacing(float(nozzle_dmr), float(extrusion_height));
		double flow_mm3_per_mm = nozzle_dmr * extrusion_height;
        Surface surface_fill(stTop, ExPolygon());
        for (ExPolygon &expoly : ironing_areas) {
			surface_fill.expolygon = std::move(expoly);
			Polylines polylines;
			try {
				polylines = f->fill_surface(&surface_fill, fill_params);
			} catch (InfillFailedException &) {
			}
	        if (! polylines.empty()) {
		        // Save into layer.
				ExtrusionEntityCollection *eec = nullptr;
		        ironing_params.layerm->fills.entities.push_back(eec = new ExtrusionEntityCollection());
		        // Don't sort the ironing infill lines as they are monotonicly ordered.
				eec->no_sort = true;
		        extrusion_entities_append_paths(
		            eec->entities, std::move(polylines),
		            erIroning,
		            flow_mm3_per_mm, extrusion_width, float(extrusion_height));
		    }
		}

		// Regions up to j were processed.
		i = j;
	}
}

} // namespace Slic3r
