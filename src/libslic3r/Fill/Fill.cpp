#include <assert.h>
#include <stdio.h>
#include <array>
#include <condition_variable>
#include <exception>
#include <map>
#include <memory>

#ifndef SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
#define SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2 1
#endif

#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
#include "../Clipper2Utils.hpp"
#endif
#include "../ClipperUtils.hpp"
#include "../Color.hpp"
#include "../ColorSolver.hpp"
#include "../Geometry.hpp"
#include "../Layer.hpp"
#include "../MarchingSquares.hpp"
#include "../Print.hpp"
#include "../PrintConfig.hpp"
#include "../SVG.hpp"
#include "../Surface.hpp"
#include "../TextureMapping.hpp"
#include "../TextureMappingContoning.hpp"
#include "../TextureMappingOffset.hpp"
#include "../VariableWidth.hpp"

#include "AABBTreeLines.hpp"
#include "ExtrusionEntity.hpp"
#include "Arachne/WallToolPaths.hpp"
#include "FillBase.hpp"
#include "FillRectilinear.hpp"
#include "FillLightning.hpp"
#include "FillConcentricInternal.hpp"
#include "FillTpmsD.hpp"
#include "FillTpmsFK.hpp"
#include "FillConcentric.hpp"
#include "libslic3r.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

struct TopSurfaceImageContoningLabelRaster
{
    using ValueType = uint8_t;
    const std::vector<int> *grid { nullptr };
    int cols { 0 };
    int rows { 0 };
    int label { -1 };
};

namespace marchsq {

template<> struct _RasterTraits<TopSurfaceImageContoningLabelRaster>
{
    using ValueType = TopSurfaceImageContoningLabelRaster::ValueType;

    static ValueType get(const TopSurfaceImageContoningLabelRaster &raster, size_t row, size_t col)
    {
        if (raster.grid == nullptr || row >= size_t(raster.rows) || col >= size_t(raster.cols))
            return ValueType(0);
        return (*raster.grid)[row * size_t(raster.cols) + col] == raster.label ? ValueType(255) : ValueType(0);
    }

    static size_t rows(const TopSurfaceImageContoningLabelRaster &raster) { return size_t(std::max(0, raster.rows)); }
    static size_t cols(const TopSurfaceImageContoningLabelRaster &raster) { return size_t(std::max(0, raster.cols)); }
};

}

namespace Slic3r {

using ThrowIfCanceled = std::function<void()>;

static void check_canceled(const ThrowIfCanceled *throw_if_canceled)
{
    if (throw_if_canceled != nullptr)
        (*throw_if_canceled)();
}

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
    bool texture_mapping_top_surface_contoning = false;
    int texture_mapping_top_surface_component_index = 0;
    int texture_mapping_top_surface_component_count = 0;
    int texture_mapping_top_surface_contoning_flat_surface_infill_mode = TextureMappingZone::SlicerDefaultTopSurfaceContoningFlatSurfaceInfillMode;
    bool texture_mapping_top_surface_contoning_no_edge_overlap = false;

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
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_contoning);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_component_index);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_component_count);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_contoning_flat_surface_infill_mode);
        RETURN_COMPARE_NON_EQUAL(texture_mapping_top_surface_contoning_no_edge_overlap);

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
                this->texture_mapping_top_surface_contoning == rhs.texture_mapping_top_surface_contoning &&
                this->texture_mapping_top_surface_component_index == rhs.texture_mapping_top_surface_component_index &&
                this->texture_mapping_top_surface_component_count == rhs.texture_mapping_top_surface_component_count &&
                this->texture_mapping_top_surface_contoning_flat_surface_infill_mode == rhs.texture_mapping_top_surface_contoning_flat_surface_infill_mode &&
                this->texture_mapping_top_surface_contoning_no_edge_overlap == rhs.texture_mapping_top_surface_contoning_no_edge_overlap;
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
    bool contoning = false;
    bool lower_surface = false;
    float angle_rad = float(PI / 4.0);
    ExPolygons area;
    ExPolygons perimeter_area;
};

struct TopSurfaceImageRegionPlan {
    const Layer *target_layer = nullptr;
    const TextureMappingZone *zone = nullptr;
    size_t region_id = size_t(-1);
    unsigned int zone_id = 0;
    std::vector<unsigned int> components_bottom_to_top;
    std::vector<TopSurfaceImageStackSlice> slices;
    float min_width_mm = TextureMappingZone::DefaultTopSurfaceImageMinLineWidthMm;
    float max_width_mm = TextureMappingZone::DefaultTopSurfaceImageMaxLineWidthMm;
    bool fixed_coloring = true;
    bool same_layer_partition = false;
    bool contoning = false;
    bool color_lower_surfaces = true;
    int colored_top_layers = TextureMappingZone::DefaultTopSurfaceImageColoredTopLayers;
    int contoning_stack_layers = TextureMappingZone::DefaultTopSurfaceContoningStackLayers;
    int contoning_pattern_filaments = TextureMappingZone::DefaultTopSurfaceContoningPatternFilaments;
    float contoning_min_feature_mm = 2.f;
    float contoning_external_width_mm = 0.4f;
    bool contoning_only_one_perimeter_around_shell_infill = false;
    bool contoning_replace_top_perimeters_with_infill = false;
    bool contoning_recolor_surrounding_perimeters = false;
    int contoning_perimeter_mode = TextureMappingZone::DefaultTopSurfaceContoningPerimeterMode;
    bool contoning_layer_phase_enabled = false;
    bool contoning_varied_infill_angles_enabled = false;
    bool contoning_blue_noise_error_diffusion_enabled = false;
    bool contoning_supersampled_cells_enabled = false;
    bool contoning_polygonize_color_regions_enabled = false;
    bool contoning_polygonize_high_resolution_enabled = TextureMappingZone::DefaultTopSurfaceContoningPolygonizeHighResolutionEnabled;
    bool contoning_surface_anchored_stacks_enabled = false;
    int contoning_flat_surface_infill_mode = TextureMappingZone::SlicerDefaultTopSurfaceContoningFlatSurfaceInfillMode;
};

enum class TopSurfaceImageSourceSurface {
    Top,
    Bottom
};

static ExPolygons top_surface_clip_union_ex(const Polygons &subject)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return union_ex_2(subject, true);
#else
    return union_ex(subject);
#endif
}

static ExPolygons top_surface_clip_union_ex(const ExPolygons &subject)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return union_ex_2(subject, true);
#else
    return union_ex(subject);
#endif
}

static ExPolygons top_surface_clip_diff_ex(const ExPolygons &subject,
                                           const Polygons &clip,
                                           ApplySafetyOffset do_safety_offset = ApplySafetyOffset::No)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return diff_ex_2(subject, clip, do_safety_offset);
#else
    return diff_ex(subject, clip, do_safety_offset);
#endif
}

static ExPolygons top_surface_clip_diff_ex(const ExPolygons &subject,
                                           const ExPolygons &clip,
                                           ApplySafetyOffset do_safety_offset = ApplySafetyOffset::No)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return diff_ex_2(subject, clip, do_safety_offset);
#else
    return diff_ex(subject, clip, do_safety_offset);
#endif
}

static ExPolygons top_surface_clip_intersection_ex(const ExPolygons &subject,
                                                   const Polygons &clip,
                                                   ApplySafetyOffset do_safety_offset = ApplySafetyOffset::No)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return intersection_ex_2(subject, clip, do_safety_offset);
#else
    return intersection_ex(subject, clip, do_safety_offset);
#endif
}

static ExPolygons top_surface_clip_intersection_ex(const ExPolygons &subject,
                                                   const ExPolygons &clip,
                                                   ApplySafetyOffset do_safety_offset = ApplySafetyOffset::No)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return intersection_ex_2(subject, clip, do_safety_offset);
#else
    return intersection_ex(subject, clip, do_safety_offset);
#endif
}

static ExPolygons top_surface_clip_offset_ex(const ExPolygon &expolygon,
                                             float delta,
                                             ClipperLib::JoinType join_type = DefaultJoinType,
                                             double miter_limit = DefaultMiterLimit)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return offset_ex_2(expolygon, delta, join_type, miter_limit);
#else
    return offset_ex(expolygon, delta, join_type, miter_limit);
#endif
}

static ExPolygons top_surface_clip_offset_ex(const ExPolygons &expolygons,
                                             float delta,
                                             ClipperLib::JoinType join_type = DefaultJoinType,
                                             double miter_limit = DefaultMiterLimit)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return offset_ex_2(expolygons, delta, join_type, miter_limit);
#else
    return offset_ex(expolygons, delta, join_type, miter_limit);
#endif
}

static ExPolygons top_surface_clip_offset2_ex(const ExPolygons &expolygons,
                                              float delta1,
                                              float delta2,
                                              ClipperLib::JoinType join_type = DefaultJoinType,
                                              double miter_limit = DefaultMiterLimit)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return offset2_ex_2(expolygons, delta1, delta2, join_type, miter_limit);
#else
    return offset2_ex(expolygons, delta1, delta2, join_type, miter_limit);
#endif
}

static ExPolygons top_surface_clip_closing_ex(const ExPolygons &expolygons,
                                              float delta,
                                              ClipperLib::JoinType join_type = DefaultJoinType,
                                              double miter_limit = DefaultMiterLimit)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return closing_ex_2(expolygons, delta, join_type, miter_limit);
#else
    return closing_ex(expolygons, delta, join_type, miter_limit);
#endif
}

static Polygons top_surface_clip_union_pt_chained_outside_in(const Polygons &subject)
{
#if SLIC3R_IMAGEMAP_TOP_SURFACE_USE_CLIPPER2
    return union_pt_chained_outside_in_2(subject);
#else
    return union_pt_chained_outside_in(subject);
#endif
}

static bool top_surface_image_debug_enabled()
{
    static const bool enabled = [] {
        const char *value = std::getenv("ORCASLICER_TOP_SURFACE_COLORING_DEBUG");
        if (value == nullptr || *value == '\0')
            return false;
        std::string normalized(value);
        for (char &ch : normalized)
            ch = char(std::tolower(static_cast<unsigned char>(ch)));
        return normalized != "0" && normalized != "false" && normalized != "off" && normalized != "no";
    }();
    return enabled;
}

static std::filesystem::path top_surface_image_debug_output_dir()
{
    static const std::filesystem::path dir = [] {
        std::error_code ec;
        std::filesystem::path cwd = std::filesystem::current_path(ec);
        if (ec)
            cwd = ".";
        std::filesystem::path out = cwd / "top_surface_coloring_debug";
        std::filesystem::create_directories(out, ec);
        return out;
    }();
    return dir;
}

static std::string top_surface_image_debug_z_string(double z)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << z;
    std::string value = out.str();
    for (char &ch : value) {
        if (ch == '-')
            ch = 'm';
        else if (ch == '.')
            ch = 'p';
    }
    return value;
}

static bool top_surface_image_debug_plan_affected(const TopSurfaceImageRegionPlan &plan)
{
    if (plan.zone == nullptr)
        return false;
    for (const TopSurfaceImageStackSlice &slice : plan.slices)
        if (!slice.area.empty() || !slice.perimeter_area.empty())
            return true;
    return false;
}

static void top_surface_image_debug_add_svg_item(
    std::vector<std::pair<ExPolygons, SVG::ExPolygonAttributes>> &items,
    ExPolygons area,
    const std::string &legend,
    const std::string &fill,
    float opacity,
    const std::string &outline)
{
    if (area.empty())
        return;
    area = top_surface_clip_union_ex(area);
    if (area.empty())
        return;
    items.emplace_back(std::move(area),
                       SVG::ExPolygonAttributes(legend, fill, outline, outline, scale_(0.035), opacity));
}

static void top_surface_image_debug_write_layer_svg(const Layer &layer,
                                                    const std::vector<TopSurfaceImageRegionPlan> &plans,
                                                    const std::vector<SurfaceFill> &surface_fills,
                                                    const char *phase,
                                                    const ThrowIfCanceled *throw_if_canceled)
{
    if (!top_surface_image_debug_enabled())
        return;

    bool affected = false;
    for (const TopSurfaceImageRegionPlan &plan : plans) {
        if (top_surface_image_debug_plan_affected(plan)) {
            affected = true;
            break;
        }
    }
    if (!affected)
        return;

    ExPolygons target_top_surfaces;
    ExPolygons target_bottom_surfaces;
    ExPolygons target_internal_surfaces;
    ExPolygons current_wall_footprint;
    ExPolygons planned_top_shell_fill;
    ExPolygons planned_bottom_shell_fill;
    ExPolygons planned_top_perimeter_replacement;
    ExPolygons planned_bottom_perimeter_replacement;
    ExPolygons image_fill_buckets;

    for (size_t region_id = 0; region_id < plans.size() && region_id < layer.regions().size(); ++region_id) {
        check_canceled(throw_if_canceled);
        const TopSurfaceImageRegionPlan &plan = plans[region_id];
        if (!top_surface_image_debug_plan_affected(plan))
            continue;
        const LayerRegion *layerm = layer.regions()[region_id];
        if (layerm == nullptr)
            continue;

        for (const Surface &surface : layerm->fill_surfaces.surfaces) {
            if (surface.is_bridge())
                continue;
            if (surface.is_top())
                target_top_surfaces.emplace_back(surface.expolygon);
            else if (surface.surface_type == stBottom)
                target_bottom_surfaces.emplace_back(surface.expolygon);
            else if (surface.surface_type == stInternalSolid)
                target_internal_surfaces.emplace_back(surface.expolygon);
        }

        Polygons wall_polygons;
        layerm->perimeters.polygons_covered_by_width(wall_polygons, 0.f);
        if (!wall_polygons.empty())
            append(current_wall_footprint, top_surface_clip_union_ex(wall_polygons));

        for (const TopSurfaceImageStackSlice &slice : plan.slices) {
            check_canceled(throw_if_canceled);
            if (slice.lower_surface) {
                append(planned_bottom_shell_fill, slice.area);
                append(planned_bottom_perimeter_replacement, slice.perimeter_area);
            } else {
                append(planned_top_shell_fill, slice.area);
                append(planned_top_perimeter_replacement, slice.perimeter_area);
            }
        }
    }

    for (const SurfaceFill &fill : surface_fills) {
        check_canceled(throw_if_canceled);
        if (fill.params.texture_mapping_top_surface_image && !fill.expolygons.empty())
            append(image_fill_buckets, fill.expolygons);
    }

    ExPolygons planned_replacement = planned_top_perimeter_replacement;
    append(planned_replacement, planned_bottom_perimeter_replacement);
    if (!planned_replacement.empty())
        planned_replacement = top_surface_clip_union_ex(planned_replacement);
    ExPolygons image_fill_area = image_fill_buckets.empty() ? ExPolygons() : top_surface_clip_union_ex(image_fill_buckets);
    ExPolygons replacement_without_image_fill;
    if (!planned_replacement.empty())
        replacement_without_image_fill = image_fill_area.empty() ?
            planned_replacement :
            top_surface_clip_diff_ex(planned_replacement, image_fill_area, ApplySafetyOffset::Yes);
    ExPolygons current_wall_inside_replacement =
        planned_replacement.empty() || current_wall_footprint.empty() ?
        ExPolygons() :
        top_surface_clip_intersection_ex(current_wall_footprint, planned_replacement, ApplySafetyOffset::Yes);

    std::vector<std::pair<ExPolygons, SVG::ExPolygonAttributes>> items;
    top_surface_image_debug_add_svg_item(items, std::move(target_internal_surfaces), "target internal solid", "#d0d0d0", 0.16f, "#888888");
    top_surface_image_debug_add_svg_item(items, std::move(target_top_surfaces), "target top", "#9ecae1", 0.20f, "#3182bd");
    top_surface_image_debug_add_svg_item(items, std::move(target_bottom_surfaces), "target bottom", "#c7b9ff", 0.20f, "#6a51a3");
    top_surface_image_debug_add_svg_item(items, std::move(current_wall_footprint), "current wall footprint", "#ff0000", 0.13f, "#cc0000");
    top_surface_image_debug_add_svg_item(items, image_fill_area, "final image fill buckets", "#ffd92f", 0.24f, "#b38f00");
    top_surface_image_debug_add_svg_item(items, std::move(planned_top_shell_fill), "planned top shell fill", "#1f78b4", 0.34f, "#08519c");
    top_surface_image_debug_add_svg_item(items, std::move(planned_bottom_shell_fill), "planned lower shell fill", "#33a02c", 0.34f, "#006d2c");
    top_surface_image_debug_add_svg_item(items, std::move(planned_top_perimeter_replacement), "planned top perimeter replace", "#ff7f00", 0.45f, "#b15928");
    top_surface_image_debug_add_svg_item(items, std::move(planned_bottom_perimeter_replacement), "planned lower perimeter replace", "#e7298a", 0.45f, "#9e0142");
    top_surface_image_debug_add_svg_item(items, std::move(current_wall_inside_replacement), "current wall inside replacement", "#ffffff", 0.20f, "#000000");
    top_surface_image_debug_add_svg_item(items, std::move(replacement_without_image_fill), "replacement missing image fill", "#000000", 0.70f, "#000000");

    if (items.empty())
        return;

    std::ostringstream filename;
    filename << "layer_"
             << std::setw(5) << std::setfill('0') << layer.id()
             << "_z_" << top_surface_image_debug_z_string(layer.print_z)
             << "_" << phase << ".svg";
    const std::filesystem::path path = top_surface_image_debug_output_dir() / filename.str();
    SVG::export_expolygons(path.string(), items);
}

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

static ExPolygons top_surface_image_visible_surface_mask(const Layer &layer,
                                                         unsigned int zone_id,
                                                         TopSurfaceImageSourceSurface source_surface)
{
    ExPolygons mask;
    for (const LayerRegion *layerm : layer.regions()) {
        if (layerm == nullptr || unsigned(std::max(0, layerm->region().config().solid_infill_filament.value)) != zone_id)
            continue;
        for (const Surface &surface : layerm->fill_surfaces.surfaces)
            if ((source_surface == TopSurfaceImageSourceSurface::Top && surface.is_top()) ||
                (source_surface == TopSurfaceImageSourceSurface::Bottom && surface.surface_type == stBottom))
                mask.emplace_back(surface.expolygon);
    }
    return mask.size() > 1 ? top_surface_clip_union_ex(mask) : mask;
}

static ExPolygons top_surface_image_visible_top_mask(const Layer &layer, unsigned int zone_id)
{
    return top_surface_image_visible_surface_mask(layer, zone_id, TopSurfaceImageSourceSurface::Top);
}

static ExPolygons top_surface_image_colorable_shell_mask(const Layer &layer,
                                                         unsigned int zone_id,
                                                         TopSurfaceImageSourceSurface source_surface)
{
    ExPolygons mask;
    for (const LayerRegion *layerm : layer.regions()) {
        if (layerm == nullptr || unsigned(std::max(0, layerm->region().config().solid_infill_filament.value)) != zone_id)
            continue;
        for (const Surface &surface : layerm->fill_surfaces.surfaces)
            if (surface.surface_type == stInternalSolid ||
                (source_surface == TopSurfaceImageSourceSurface::Top && surface.is_top()) ||
                (source_surface == TopSurfaceImageSourceSurface::Bottom && surface.surface_type == stBottom))
                mask.emplace_back(surface.expolygon);
    }
    return mask.size() > 1 ? top_surface_clip_union_ex(mask) : mask;
}

static ExPolygons top_surface_image_current_layer_surface_mask(const LayerRegion &layerm,
                                                               TopSurfaceImageSourceSurface source_surface)
{
    ExPolygons mask;
    for (const Surface &surface : layerm.fill_surfaces.surfaces)
        if (!surface.is_bridge() &&
            (surface.surface_type == stInternalSolid ||
             (source_surface == TopSurfaceImageSourceSurface::Top && surface.is_top()) ||
             (source_surface == TopSurfaceImageSourceSurface::Bottom && surface.surface_type == stBottom)))
            mask.emplace_back(surface.expolygon);
    return mask.size() > 1 ? top_surface_clip_union_ex(mask) : mask;
}

static bool top_surface_image_contoning_depth_within_shell(const Layer                   &target_layer,
                                                           const Layer                   &source_layer,
                                                           const PrintRegionConfig       &region_config,
                                                           TopSurfaceImageSourceSurface   source_surface,
                                                           int                            depth)
{
    if (depth < 0)
        return false;
    if (source_surface == TopSurfaceImageSourceSurface::Top) {
        const int shell_layers = region_config.top_shell_layers.value;
        if (shell_layers <= 0)
            return false;
        if (depth < shell_layers)
            return true;
        const double shell_thickness = region_config.top_shell_thickness.value;
        return shell_thickness > EPSILON &&
               source_layer.print_z - target_layer.print_z < shell_thickness - EPSILON;
    }

    const int shell_layers = region_config.bottom_shell_layers.value;
    if (shell_layers <= 0)
        return false;
    if (depth < shell_layers)
        return true;
    const double shell_thickness = region_config.bottom_shell_thickness.value;
    return shell_thickness > EPSILON &&
           target_layer.bottom_z() - source_layer.bottom_z() < shell_thickness - EPSILON;
}

static ExPolygon top_surface_image_cell_expolygon(coord_t min_x, coord_t min_y, coord_t max_x, coord_t max_y)
{
    Polygon polygon;
    polygon.points.reserve(4);
    polygon.points.emplace_back(min_x, min_y);
    polygon.points.emplace_back(max_x, min_y);
    polygon.points.emplace_back(max_x, max_y);
    polygon.points.emplace_back(min_x, max_y);
    polygon.make_counter_clockwise();
    return ExPolygon(std::move(polygon));
}

static bool top_surface_image_contoning_sample_eligible(const TextureMappingOffsetContext &context,
                                                        float                              x_mm,
                                                        float                              y_mm,
                                                        float                              threshold_deg,
                                                        TopSurfaceImageSourceSurface       source_surface)
{
    const std::optional<float> normal_z =
        sample_weight_field_normal_z(context.weight_field, x_mm, y_mm, context.high_resolution_texture_sampling);
    const float oriented_normal_z =
        normal_z && source_surface == TopSurfaceImageSourceSurface::Bottom ? -*normal_z : (normal_z ? *normal_z : 1.f);
    return !normal_z || texture_mapping_contoning_normal_eligible(oriented_normal_z, threshold_deg);
}

static std::vector<ExPolygons> top_surface_image_contoning_stack_areas(const Layer &source_layer,
                                                                       unsigned int zone_id,
                                                                       int stack_layers,
                                                                       TopSurfaceImageSourceSurface source_surface,
                                                                       const ThrowIfCanceled *throw_if_canceled)
{
    std::vector<ExPolygons> out;
    out.reserve(size_t(std::max(0, stack_layers)));
    const Layer *stack_layer = &source_layer;
    for (int depth = 0; depth < stack_layers && stack_layer != nullptr; ++depth) {
        check_canceled(throw_if_canceled);
        ExPolygons area = depth == 0 ?
            top_surface_image_visible_surface_mask(*stack_layer, zone_id, source_surface) :
            top_surface_image_colorable_shell_mask(*stack_layer, zone_id, source_surface);
        out.emplace_back(std::move(area));
        stack_layer = source_surface == TopSurfaceImageSourceSurface::Top ?
            stack_layer->lower_layer :
            stack_layer->upper_layer;
    }
    return out;
}

static double top_surface_image_scaled_area_mm2(double scaled_area)
{
    return std::abs(scaled_area) * SCALING_FACTOR * SCALING_FACTOR;
}

static double top_surface_image_abs_area(const ExPolygons &area)
{
    double out = 0.;
    for (const ExPolygon &expolygon : area)
        out += std::abs(expolygon.area());
    return out;
}

static bool top_surface_image_expolygons_contain_point(const ExPolygons &expolygons, const Point &point)
{
    for (const ExPolygon &expolygon : expolygons)
        if (expolygon.contains(point, true))
            return true;
    return false;
}

static int top_surface_image_contoning_local_stack_layers_at_point(const Point &point,
                                                                   const std::vector<ExPolygons> &stack_areas,
                                                                   int max_stack_layers)
{
    int layers = 0;
    for (const ExPolygons &stack_area : stack_areas) {
        if (layers >= max_stack_layers)
            break;
        if (stack_area.empty() || !top_surface_image_expolygons_contain_point(stack_area, point))
            break;
        ++layers;
    }
    return layers;
}

static ExPolygons top_surface_image_contoning_clean_area(ExPolygons &&component_area,
                                                         const ExPolygons &clip_area,
                                                         const ExPolygons &blocked_area,
                                                         float min_feature_mm,
                                                         const ThrowIfCanceled *throw_if_canceled)
{
    if (component_area.empty())
        return {};
    check_canceled(throw_if_canceled);
    ExPolygons out = top_surface_clip_union_ex(component_area);
    const float closing_radius = float(scale_(std::clamp(min_feature_mm * 0.18f, 0.05f, 0.45f)));
    if (closing_radius > float(SCALED_EPSILON))
        out = top_surface_clip_closing_ex(out, closing_radius, ClipperLib::jtRound);
    if (out.empty())
        return {};
    check_canceled(throw_if_canceled);
    out = top_surface_clip_intersection_ex(out, clip_area, ApplySafetyOffset::Yes);
    if (!blocked_area.empty())
        out = top_surface_clip_diff_ex(out, blocked_area, ApplySafetyOffset::Yes);
    if (out.empty())
        return {};
    check_canceled(throw_if_canceled);

    ExPolygons simplified;
    const double tolerance = scale_(std::clamp(min_feature_mm * 0.12f, 0.03f, 0.25f));
    for (const ExPolygon &expolygon : out) {
        check_canceled(throw_if_canceled);
        append(simplified, expolygon.simplify(tolerance));
    }
    if (simplified.empty())
        return {};

    ExPolygons filtered;
    const double min_area_mm2 = std::max(0.05, double(min_feature_mm) * double(min_feature_mm) * 0.08);
    for (ExPolygon &expolygon : simplified)
        if (top_surface_image_scaled_area_mm2(expolygon.area()) >= min_area_mm2)
            filtered.emplace_back(std::move(expolygon));
    return filtered.empty() ? ExPolygons() : top_surface_clip_union_ex(filtered);
}

struct TopSurfaceImageContoningVectorLabel {
    std::vector<unsigned int> bottom_to_top;
    std::array<float, 3> rgb { { 0.f, 0.f, 0.f } };
    std::array<float, 3> oklab { { 0.f, 0.f, 0.f } };
};

struct TopSurfaceImageContoningVectorRegion {
    std::vector<unsigned int> bottom_to_top;
    ExPolygons area;
    int cell_count { 0 };
};

struct TopSurfaceImageContoningCellSample {
    std::array<float, 3> rgb { { 0.f, 0.f, 0.f } };
    int solve_layers { 0 };
    int available_depth { 0 };
};

struct TopSurfaceImageContoningSolvedLabel {
    int label { -1 };
    std::array<float, 3> rgb { { 0.f, 0.f, 0.f } };
};

struct TopSurfaceImageContoningStackPlanCell {
    int label { -1 };
    int available_depth { 0 };
};

struct TopSurfaceImageContoningStackPlan {
    BoundingBox bbox;
    coord_t min_x { 0 };
    coord_t min_y { 0 };
    coord_t step { 1 };
    int cols { 0 };
    int rows { 0 };
    std::vector<TopSurfaceImageContoningVectorLabel> labels;
    std::vector<TopSurfaceImageContoningStackPlanCell> cells;
};

struct TopSurfaceImageContoningDepthRegionPlan {
    std::vector<std::vector<TopSurfaceImageContoningVectorRegion>> fill_regions_by_depth;
    std::vector<std::vector<TopSurfaceImageContoningVectorRegion>> perimeter_regions_by_depth;
};

struct TopSurfaceImageContoningSourceContext {
    TextureMappingOffsetContext offset_context;
    std::vector<ExPolygons> stack_areas;
    ExPolygons normal_filter_bypass_area;
    float threshold_deg { 0.f };
    int stack_layers { 0 };
    int pattern_filaments { 0 };
};

static long long top_surface_image_contoning_float_key(double value)
{
    if (!std::isfinite(value))
        return 0;
    return static_cast<long long>(std::llround(value * 1000000.0));
}

struct TopSurfaceImageContoningStackPlanKey {
    const Layer *source_layer { nullptr };
    size_t source_layer_id { 0 };
    const Layer *target_layer { nullptr };
    size_t target_layer_id { 0 };
    size_t region_id { size_t(-1) };
    int target_depth { -1 };
    int source_surface { 0 };
    long long sample_z { 0 };
    unsigned int zone_id { 0 };
    std::vector<unsigned int> component_ids;
    int stack_layers { 0 };
    int pattern_filaments { 0 };
    long long min_feature_mm { 0 };
    long long min_width_mm { 0 };
    long long max_width_mm { 0 };
    long long external_width_mm { 0 };
    long long angle_threshold_deg { 0 };
    bool layer_phase { false };
    bool replace_top_perimeters { false };
    bool recolor_surrounding_perimeters { false };
    bool supersampled { false };
    bool blue_noise { false };
    bool polygonize { false };
    bool polygonize_high_resolution { false };

    bool operator<(const TopSurfaceImageContoningStackPlanKey &rhs) const
    {
        return std::tie(source_layer,
                        source_layer_id,
                        target_layer,
                        target_layer_id,
                        region_id,
                        target_depth,
                        source_surface,
                        sample_z,
                        zone_id,
                        component_ids,
                        stack_layers,
                        pattern_filaments,
                        min_feature_mm,
                        min_width_mm,
                        max_width_mm,
                        external_width_mm,
                        angle_threshold_deg,
                        layer_phase,
                        replace_top_perimeters,
                        recolor_surrounding_perimeters,
                        supersampled,
                        blue_noise,
                        polygonize,
                        polygonize_high_resolution) <
               std::tie(rhs.source_layer,
                        rhs.source_layer_id,
                        rhs.target_layer,
                        rhs.target_layer_id,
                        rhs.region_id,
                        rhs.target_depth,
                        rhs.source_surface,
                        rhs.sample_z,
                        rhs.zone_id,
                        rhs.component_ids,
                        rhs.stack_layers,
                        rhs.pattern_filaments,
                        rhs.min_feature_mm,
                        rhs.min_width_mm,
                        rhs.max_width_mm,
                        rhs.external_width_mm,
                        rhs.angle_threshold_deg,
                        rhs.layer_phase,
                        rhs.replace_top_perimeters,
                        rhs.recolor_surrounding_perimeters,
                        rhs.supersampled,
                        rhs.blue_noise,
                        rhs.polygonize,
                        rhs.polygonize_high_resolution);
    }
};

class TopSurfaceImageContoningStackPlanCache
{
public:
    template <class Builder>
    std::shared_ptr<const TopSurfaceImageContoningStackPlan> get_or_build(
        const TopSurfaceImageContoningStackPlanKey &key,
        Builder &&builder)
    {
        return get_or_build_impl(m_plans, key, std::forward<Builder>(builder));
    }

    template <class Builder>
    std::shared_ptr<const TopSurfaceImageContoningDepthRegionPlan> get_or_build_depth_regions(
        const TopSurfaceImageContoningStackPlanKey &key,
        Builder &&builder)
    {
        return get_or_build_impl(m_depth_region_plans, key, std::forward<Builder>(builder));
    }

private:
    template <class Plan>
    struct Entry {
        std::mutex mutex;
        std::condition_variable cv;
        std::shared_ptr<const Plan> plan;
        std::exception_ptr error;
        bool ready { false };
    };

    template <class Plan, class Builder>
    std::shared_ptr<const Plan> get_or_build_impl(
        std::map<TopSurfaceImageContoningStackPlanKey, std::shared_ptr<Entry<Plan>>> &plans,
        const TopSurfaceImageContoningStackPlanKey &key,
        Builder &&builder)
    {
        std::shared_ptr<Entry<Plan>> entry;
        bool build = false;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto found = plans.find(key);
            if (found == plans.end()) {
                entry = std::make_shared<Entry<Plan>>();
                plans.emplace(key, entry);
                build = true;
            } else {
                entry = found->second;
            }
        }

        if (build) {
            std::shared_ptr<const Plan> plan;
            std::exception_ptr error;
            try {
                plan = builder();
            } catch (...) {
                error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(entry->mutex);
                entry->plan = plan;
                entry->error = error;
                entry->ready = true;
            }
            entry->cv.notify_all();
            if (error) {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto found = plans.find(key);
                if (found != plans.end() && found->second == entry)
                    plans.erase(found);
                std::rethrow_exception(error);
            }
            return plan;
        }

        std::unique_lock<std::mutex> lock(entry->mutex);
        entry->cv.wait(lock, [&entry]() { return entry->ready; });
        if (entry->error)
            std::rethrow_exception(entry->error);
        return entry->plan;
    }

    std::mutex m_mutex;
    std::map<TopSurfaceImageContoningStackPlanKey, std::shared_ptr<Entry<TopSurfaceImageContoningStackPlan>>> m_plans;
    std::map<TopSurfaceImageContoningStackPlanKey, std::shared_ptr<Entry<TopSurfaceImageContoningDepthRegionPlan>>> m_depth_region_plans;
};

std::shared_ptr<TopSurfaceImageContoningStackPlanCache> make_top_surface_image_contoning_stack_plan_cache()
{
    return std::make_shared<TopSurfaceImageContoningStackPlanCache>();
}

static float top_surface_image_contoning_oklab_error(const std::array<float, 3> &lhs,
                                                     const std::array<float, 3> &rhs)
{
    const float dl = lhs[0] - rhs[0];
    const float da = lhs[1] - rhs[1];
    const float db = lhs[2] - rhs[2];
    return dl * dl + 4.f * da * da + 4.f * db * db;
}

static std::optional<std::array<float, 3>> top_surface_image_contoning_stack_rgb(
    const std::vector<unsigned int> &bottom_to_top,
    const PrintConfig               &print_config)
{
    if (bottom_to_top.empty())
        return std::nullopt;
    std::vector<std::array<float, 3>> colors;
    std::vector<float> weights;
    colors.reserve(bottom_to_top.size());
    weights.reserve(bottom_to_top.size());
    const float weight = 1.f / float(bottom_to_top.size());
    for (unsigned int component_id : bottom_to_top) {
        if (component_id == 0 || component_id > print_config.filament_colour.values.size())
            return std::nullopt;
        ColorRGB color;
        if (!decode_color(print_config.filament_colour.get_at(size_t(component_id - 1)), color))
            return std::nullopt;
        colors.push_back({ color.r(), color.g(), color.b() });
        weights.emplace_back(weight);
    }
    return mix_color_solver_components(colors, weights, ColorSolverMixModel::PigmentPainter);
}

static std::optional<std::array<float, 3>> top_surface_image_contoning_component_rgb(
    unsigned int       component_id,
    const PrintConfig &print_config)
{
    if (component_id == 0 || component_id > print_config.filament_colour.values.size())
        return std::nullopt;
    ColorRGB color;
    if (!decode_color(print_config.filament_colour.get_at(size_t(component_id - 1)), color))
        return std::nullopt;
    return std::array<float, 3>{ color.r(), color.g(), color.b() };
}

static void top_surface_image_contoning_sort_stack_for_top_color(std::vector<unsigned int> &bottom_to_top,
                                                                 const std::array<float, 3> &mix_rgb,
                                                                 const PrintConfig &print_config)
{
    if (bottom_to_top.size() < 2)
        return;
    const std::array<float, 3> mix_oklab = color_solver_oklab_from_srgb(mix_rgb);
    std::stable_sort(bottom_to_top.begin(), bottom_to_top.end(), [&mix_oklab, &print_config](unsigned int lhs, unsigned int rhs) {
        const std::optional<std::array<float, 3>> lhs_rgb =
            top_surface_image_contoning_component_rgb(lhs, print_config);
        const std::optional<std::array<float, 3>> rhs_rgb =
            top_surface_image_contoning_component_rgb(rhs, print_config);
        const float lhs_error = lhs_rgb ?
            top_surface_image_contoning_oklab_error(color_solver_oklab_from_srgb(*lhs_rgb), mix_oklab) :
            std::numeric_limits<float>::max();
        const float rhs_error = rhs_rgb ?
            top_surface_image_contoning_oklab_error(color_solver_oklab_from_srgb(*rhs_rgb), mix_oklab) :
            std::numeric_limits<float>::max();
        return lhs_error > rhs_error;
    });
}

static bool top_surface_image_contoning_can_finish_without_repeat(const std::vector<unsigned int> &ids,
                                                                  const std::vector<int>          &counts,
                                                                  unsigned int                     previous_id,
                                                                  int                              remaining)
{
    for (size_t idx = 0; idx < ids.size() && idx < counts.size(); ++idx) {
        const int limit = ids[idx] == previous_id ? remaining / 2 : (remaining + 1) / 2;
        if (counts[idx] > limit)
            return false;
    }
    return true;
}

static void top_surface_image_contoning_spread_stack_repeats(std::vector<unsigned int> &bottom_to_top)
{
    if (bottom_to_top.size() < 3)
        return;

    std::vector<unsigned int> ids;
    std::vector<int> counts;
    std::vector<int> ranks;
    for (size_t pos = 0; pos < bottom_to_top.size(); ++pos) {
        const unsigned int id = bottom_to_top[pos];
        auto it = std::find(ids.begin(), ids.end(), id);
        if (it == ids.end()) {
            ids.emplace_back(id);
            counts.emplace_back(1);
            ranks.emplace_back(int(pos));
        } else {
            const size_t idx = size_t(it - ids.begin());
            ++counts[idx];
            ranks[idx] = int(pos);
        }
    }
    if (ids.size() < 2)
        return;

    std::vector<unsigned int> top_to_bottom;
    top_to_bottom.reserve(bottom_to_top.size());
    unsigned int previous_id = 0;
    int remaining = int(bottom_to_top.size());

    auto better_candidate = [&counts, &ranks, &ids](int lhs, int rhs) {
        if (rhs < 0)
            return true;
        if (ranks[size_t(lhs)] != ranks[size_t(rhs)])
            return ranks[size_t(lhs)] > ranks[size_t(rhs)];
        if (counts[size_t(lhs)] != counts[size_t(rhs)])
            return counts[size_t(lhs)] > counts[size_t(rhs)];
        return ids[size_t(lhs)] < ids[size_t(rhs)];
    };

    auto select_candidate = [&](bool require_feasible, bool allow_repeat) {
        int best = -1;
        for (size_t idx = 0; idx < ids.size(); ++idx) {
            if (counts[idx] <= 0 || (!allow_repeat && ids[idx] == previous_id))
                continue;
            --counts[idx];
            const bool feasible =
                top_surface_image_contoning_can_finish_without_repeat(ids, counts, ids[idx], remaining - 1);
            ++counts[idx];
            if (require_feasible && !feasible)
                continue;
            if (better_candidate(int(idx), best))
                best = int(idx);
        }
        return best;
    };

    while (remaining > 0) {
        int selected = select_candidate(true, false);
        if (selected < 0)
            selected = select_candidate(false, false);
        if (selected < 0)
            selected = select_candidate(false, true);
        if (selected < 0)
            break;
        top_to_bottom.emplace_back(ids[size_t(selected)]);
        --counts[size_t(selected)];
        previous_id = ids[size_t(selected)];
        --remaining;
    }

    if (top_to_bottom.size() != bottom_to_top.size())
        return;
    std::reverse(top_to_bottom.begin(), top_to_bottom.end());
    bottom_to_top = std::move(top_to_bottom);
}

static float top_surface_image_contoning_sample_pitch_mm(const TopSurfaceImageRegionPlan &plan,
                                                         const BoundingBox               &bbox)
{
    float pitch = std::clamp(plan.contoning_external_width_mm,
                             0.25f,
                             std::max(0.25f, plan.contoning_min_feature_mm * 0.5f));
    const bool high_resolution =
        plan.contoning_polygonize_color_regions_enabled &&
        plan.contoning_polygonize_high_resolution_enabled;
    const float min_pitch = high_resolution ? 0.125f : 0.25f;
    if (high_resolution)
        pitch = std::max(min_pitch, pitch * 0.5f);
    const double width_mm = unscale<double>(bbox.max.x() - bbox.min.x());
    const double height_mm = unscale<double>(bbox.max.y() - bbox.min.y());
    const double max_samples = high_resolution ? 2600000.0 : 650000.0;
    if (width_mm > 0.0 && height_mm > 0.0) {
        const double estimated = std::ceil(width_mm / double(pitch)) * std::ceil(height_mm / double(pitch));
        if (estimated > max_samples)
            pitch = float(std::sqrt(width_mm * height_mm / max_samples));
    }
    return std::clamp(pitch, min_pitch, std::max(min_pitch, plan.contoning_min_feature_mm));
}

static float top_surface_image_contoning_angle_rad(int depth, bool varied_angles)
{
    if (!varied_angles)
        return (depth & 1) ? float(-PI / 4.0) : float(PI / 4.0);
    switch ((depth % 4 + 4) % 4) {
    case 0: return float(PI / 4.0);
    case 1: return float(-PI / 4.0);
    case 2: return 0.f;
    default: return float(PI / 2.0);
    }
}

static coord_t top_surface_image_contoning_grid_min(coord_t value, coord_t step, coord_t phase)
{
    if (step <= 0)
        return value;
    const coord_t shifted = value - phase;
    coord_t quotient = shifted / step;
    if (shifted < 0 && shifted % step != 0)
        --quotient;
    return quotient * step + phase;
}

static std::pair<coord_t, coord_t> top_surface_image_contoning_grid_phase(coord_t step, int depth)
{
    if (step <= 0)
        return { 0, 0 };
    static constexpr int phase_x[8] = { 0, 4, 2, 6, 1, 5, 3, 7 };
    static constexpr int phase_y[8] = { 0, 4, 6, 2, 5, 1, 7, 3 };
    const int idx = (depth % 8 + 8) % 8;
    return {
        coord_t((static_cast<long long>(step) * static_cast<long long>(phase_x[idx])) / 8),
        coord_t((static_cast<long long>(step) * static_cast<long long>(phase_y[idx])) / 8)
    };
}

static uint32_t top_surface_image_contoning_hash_u32(int col, int row, int depth, int channel)
{
    uint32_t h = 2166136261u;
    auto mix = [&h](uint32_t value) {
        h ^= value;
        h *= 16777619u;
    };
    mix(uint32_t(col) * 0x9e3779b9u);
    mix(uint32_t(row) * 0x85ebca6bu);
    mix(uint32_t(depth) * 0xc2b2ae35u);
    mix(uint32_t(channel) * 0x27d4eb2fu);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

static float top_surface_image_contoning_jitter(int col, int row, int depth, int channel)
{
    const uint32_t h = top_surface_image_contoning_hash_u32(col, row, depth, channel);
    const float unit = float(h & 0x00FFFFFFu) / float(0x00FFFFFFu);
    return (unit - 0.5f) * 0.006f;
}

static bool top_surface_image_contoning_grid_component_printable(int cell_count,
                                                                 int min_col,
                                                                 int max_col,
                                                                 int min_row,
                                                                 int max_row,
                                                                 float pitch_mm,
                                                                 float min_feature_mm,
                                                                 float line_width_mm)
{
    if (cell_count <= 0)
        return false;
    const double area_mm2 = double(cell_count) * double(pitch_mm) * double(pitch_mm);
    const double width_mm = double(max_col - min_col + 1) * double(pitch_mm);
    const double height_mm = double(max_row - min_row + 1) * double(pitch_mm);
    const double min_area_mm2 =
        std::max(double(line_width_mm) * double(min_feature_mm),
                 double(min_feature_mm) * double(min_feature_mm) * 0.20);
    const double min_width_mm = std::max(double(line_width_mm), double(pitch_mm));
    if (area_mm2 < min_area_mm2)
        return false;
    if (width_mm < double(min_feature_mm) && height_mm < double(min_feature_mm))
        return false;
    if (std::min(width_mm, height_mm) < min_width_mm &&
        std::max(width_mm, height_mm) < 2.0 * double(min_feature_mm))
        return false;
    return true;
}

static void top_surface_image_contoning_merge_small_grid_regions(
    std::vector<int>                                      &grid,
    int                                                    cols,
    int                                                    rows,
    const std::vector<TopSurfaceImageContoningVectorLabel> &labels,
    float                                                  pitch_mm,
    float                                                  min_feature_mm,
    float                                                  line_width_mm,
    const ThrowIfCanceled                                  *throw_if_canceled)
{
    if (grid.empty() || cols <= 0 || rows <= 0 || labels.empty())
        return;

    for (int pass = 0; pass < 8; ++pass) {
        check_canceled(throw_if_canceled);
        std::vector<unsigned char> visited(grid.size(), 0);
        bool changed = false;
        for (int row = 0; row < rows; ++row) {
            if ((row & 15) == 0)
                check_canceled(throw_if_canceled);
            for (int col = 0; col < cols; ++col) {
                const int start_idx = row * cols + col;
                const int source_label = grid[size_t(start_idx)];
                if (source_label < 0 || visited[size_t(start_idx)])
                    continue;

                std::vector<int> queue;
                std::vector<int> cells;
                std::map<int, int> neighbor_counts;
                queue.push_back(start_idx);
                visited[size_t(start_idx)] = 1;
                int min_col = col;
                int max_col = col;
                int min_row = row;
                int max_row = row;
                for (size_t queue_idx = 0; queue_idx < queue.size(); ++queue_idx) {
                    if ((queue_idx & 255) == 0)
                        check_canceled(throw_if_canceled);
                    const int idx = queue[queue_idx];
                    cells.push_back(idx);
                    const int r = idx / cols;
                    const int c = idx - r * cols;
                    min_col = std::min(min_col, c);
                    max_col = std::max(max_col, c);
                    min_row = std::min(min_row, r);
                    max_row = std::max(max_row, r);
                    const std::array<std::pair<int, int>, 4> neighbors{
                        std::pair<int, int>{ c - 1, r },
                        std::pair<int, int>{ c + 1, r },
                        std::pair<int, int>{ c, r - 1 },
                        std::pair<int, int>{ c, r + 1 }
                    };
                    for (const std::pair<int, int> &neighbor : neighbors) {
                        const int nc = neighbor.first;
                        const int nr = neighbor.second;
                        if (nc < 0 || nc >= cols || nr < 0 || nr >= rows)
                            continue;
                        const int nidx = nr * cols + nc;
                        const int nlabel = grid[size_t(nidx)];
                        if (nlabel == source_label) {
                            if (!visited[size_t(nidx)]) {
                                visited[size_t(nidx)] = 1;
                                queue.push_back(nidx);
                            }
                        } else if (nlabel >= 0) {
                            ++neighbor_counts[nlabel];
                        }
                    }
                }

                if (top_surface_image_contoning_grid_component_printable(int(cells.size()),
                                                                         min_col,
                                                                         max_col,
                                                                         min_row,
                                                                         max_row,
                                                                         pitch_mm,
                                                                         min_feature_mm,
                                                                         line_width_mm))
                    continue;

                int best_label = -1;
                float best_error = std::numeric_limits<float>::max();
                int best_contact = -1;
                for (const auto &entry : neighbor_counts) {
                    const int neighbor_label = entry.first;
                    if (neighbor_label < 0 ||
                        neighbor_label >= int(labels.size()) ||
                        source_label >= int(labels.size()))
                        continue;
                    const float error =
                        top_surface_image_contoning_oklab_error(labels[size_t(source_label)].oklab,
                                                                labels[size_t(neighbor_label)].oklab);
                    if (error < best_error - 1e-6f ||
                        (std::abs(error - best_error) <= 1e-6f && entry.second > best_contact)) {
                        best_label = neighbor_label;
                        best_error = error;
                        best_contact = entry.second;
                    }
                }
                if (best_label < 0)
                    continue;
                check_canceled(throw_if_canceled);
                for (int idx : cells)
                    grid[size_t(idx)] = best_label;
                changed = true;
            }
        }
        if (!changed)
            break;
    }
}

struct TopSurfaceImageContoningGridRect {
    int col_begin { 0 };
    int col_end { 0 };
    int row_begin { 0 };
    int row_end { 0 };
};

static void top_surface_image_contoning_append_grid_rect(ExPolygons &cells,
                                                         const TopSurfaceImageContoningGridRect &rect,
                                                         coord_t min_x,
                                                         coord_t min_y,
                                                         coord_t step,
                                                         const BoundingBox &bbox)
{
    const coord_t x0 = min_x + coord_t(rect.col_begin) * step;
    const coord_t y0 = min_y + coord_t(rect.row_begin) * step;
    const coord_t x1 = std::min<coord_t>(min_x + coord_t(rect.col_end) * step, bbox.max.x());
    const coord_t y1 = std::min<coord_t>(min_y + coord_t(rect.row_end) * step, bbox.max.y());
    if (x1 <= x0 || y1 <= y0)
        return;
    cells.emplace_back(top_surface_image_cell_expolygon(x0, y0, x1, y1));
}

static ExPolygons top_surface_image_contoning_area_from_grid_label(const std::vector<int> &grid,
                                                                   int                     cols,
                                                                   int                     rows,
                                                                   int                     label,
                                                                   coord_t                 min_x,
                                                                   coord_t                 min_y,
                                                                   coord_t                 step,
                                                                   const BoundingBox      &bbox,
                                                                   const ExPolygons       &clip_area,
                                                                   const ExPolygons       &blocked_area,
                                                                   float                   min_feature_mm,
                                                                   const ThrowIfCanceled  *throw_if_canceled)
{
    ExPolygons cells;
    std::vector<TopSurfaceImageContoningGridRect> active;
    for (int row = 0; row < rows; ++row) {
        if ((row & 15) == 0)
            check_canceled(throw_if_canceled);
        std::vector<TopSurfaceImageContoningGridRect> next_active;
        for (int col = 0; col < cols;) {
            const int idx = row * cols + col;
            if (grid[size_t(idx)] != label) {
                ++col;
                continue;
            }
            const int col_begin = col;
            do {
                ++col;
            } while (col < cols && grid[size_t(row * cols + col)] == label);

            auto active_it = std::find_if(active.begin(), active.end(), [col_begin, col](const TopSurfaceImageContoningGridRect &rect) {
                return rect.col_begin == col_begin && rect.col_end == col;
            });
            if (active_it == active.end()) {
                TopSurfaceImageContoningGridRect rect;
                rect.col_begin = col_begin;
                rect.col_end = col;
                rect.row_begin = row;
                rect.row_end = row + 1;
                next_active.emplace_back(rect);
            } else {
                active_it->row_end = row + 1;
                next_active.emplace_back(*active_it);
                active.erase(active_it);
            }
        }
        for (const TopSurfaceImageContoningGridRect &rect : active)
            top_surface_image_contoning_append_grid_rect(cells, rect, min_x, min_y, step, bbox);
        active = std::move(next_active);
    }
    for (const TopSurfaceImageContoningGridRect &rect : active)
        top_surface_image_contoning_append_grid_rect(cells, rect, min_x, min_y, step, bbox);
    return top_surface_image_contoning_clean_area(std::move(cells), clip_area, blocked_area, min_feature_mm, throw_if_canceled);
}

static ExPolygons top_surface_image_contoning_polygonized_area_from_grid_label(const std::vector<int> &grid,
                                                                              int                     cols,
                                                                              int                     rows,
                                                                              int                     label,
                                                                              coord_t                 min_x,
                                                                              coord_t                 min_y,
                                                                              coord_t                 step,
                                                                              const BoundingBox      &bbox,
                                                                              const ExPolygons       &clip_area,
                                                                              const ExPolygons       &blocked_area,
                                                                              float                   min_feature_mm,
                                                                              const ThrowIfCanceled  *throw_if_canceled)
{
    if (grid.empty() || cols <= 0 || rows <= 0 || grid.size() != size_t(cols) * size_t(rows))
        return {};

    const TopSurfaceImageContoningLabelRaster raster{ &grid, cols, rows, label };
    std::vector<marchsq::Ring> rings =
        marchsq::execute(raster, TopSurfaceImageContoningLabelRaster::ValueType(128), marchsq::Coord(1, 1));
    if (rings.empty())
        return {};

    ExPolygons cells;
    cells.reserve(rings.size());
    for (const marchsq::Ring &ring : rings) {
        check_canceled(throw_if_canceled);
        if (ring.size() < 3)
            continue;
        Polygon polygon;
        polygon.points.reserve(ring.size());
        for (const marchsq::Coord &coord : ring) {
            const coord_t x = std::min<coord_t>(min_x + coord_t(coord.c) * step, bbox.max.x());
            const coord_t y = std::min<coord_t>(min_y + coord_t(coord.r) * step, bbox.max.y());
            if (polygon.points.empty() || polygon.points.back() != Point(x, y))
                polygon.points.emplace_back(x, y);
        }
        if (polygon.points.size() >= 3 && polygon.points.front() == polygon.points.back())
            polygon.points.pop_back();
        if (polygon.points.size() >= 3)
            cells.emplace_back(std::move(polygon));
    }

    return top_surface_image_contoning_clean_area(std::move(cells), clip_area, blocked_area, min_feature_mm, throw_if_canceled);
}

static std::vector<TopSurfaceImageContoningVectorRegion> top_surface_image_contoning_component_regions_from_grid(
    const std::vector<int>                                      &label_grid,
    int                                                          cols,
    int                                                          rows,
    const std::vector<TopSurfaceImageContoningVectorLabel>       &labels,
    int                                                          depth,
    coord_t                                                      min_x,
    coord_t                                                      min_y,
    coord_t                                                      step,
    const BoundingBox                                           &bbox,
    const ExPolygons                                            &area,
    float                                                        min_feature_mm,
    bool                                                         polygonize_color_regions,
    const ThrowIfCanceled                                       *throw_if_canceled)
{
    std::vector<TopSurfaceImageContoningVectorRegion> regions;
    if (label_grid.empty() || labels.empty() || cols <= 0 || rows <= 0 ||
        label_grid.size() != size_t(cols) * size_t(rows) || depth < 0)
        return regions;

    unsigned int max_component_id = 0;
    for (const TopSurfaceImageContoningVectorLabel &label : labels)
        for (unsigned int component_id : label.bottom_to_top)
            max_component_id = std::max(max_component_id, component_id);
    if (max_component_id == 0)
        return regions;

    std::vector<int> component_grid(label_grid.size(), -1);
    std::vector<int> cell_counts(size_t(max_component_id) + 1, 0);
    for (size_t idx = 0; idx < label_grid.size(); ++idx) {
        const int label = label_grid[idx];
        if (label < 0 || label >= int(labels.size()))
            continue;
        const std::vector<unsigned int> &bottom_to_top = labels[size_t(label)].bottom_to_top;
        if (bottom_to_top.empty())
            continue;
        const int pattern_depth = depth % int(bottom_to_top.size());
        const unsigned int component_id =
            bottom_to_top[size_t(int(bottom_to_top.size()) - 1 - pattern_depth)];
        if (component_id == 0 || component_id > max_component_id)
            continue;
        component_grid[idx] = int(component_id);
        ++cell_counts[size_t(component_id)];
    }

    std::vector<int> component_order;
    for (size_t idx = 1; idx < cell_counts.size(); ++idx)
        if (cell_counts[idx] > 0)
            component_order.emplace_back(int(idx));
    if (component_order.empty())
        return regions;
    std::sort(component_order.begin(), component_order.end(), [&cell_counts](int lhs, int rhs) {
        if (cell_counts[size_t(lhs)] != cell_counts[size_t(rhs)])
            return cell_counts[size_t(lhs)] > cell_counts[size_t(rhs)];
        return lhs < rhs;
    });

    ExPolygons taken;
    for (int component_id : component_order) {
        check_canceled(throw_if_canceled);
        ExPolygons component_area =
            polygonize_color_regions ?
                top_surface_image_contoning_polygonized_area_from_grid_label(component_grid,
                                                                             cols,
                                                                             rows,
                                                                             component_id,
                                                                             min_x,
                                                                             min_y,
                                                                             step,
                                                                             bbox,
                                                                             area,
                                                                             taken,
                                                                             min_feature_mm,
                                                                             throw_if_canceled) :
                top_surface_image_contoning_area_from_grid_label(component_grid,
                                                                 cols,
                                                                 rows,
                                                                 component_id,
                                                                 min_x,
                                                                 min_y,
                                                                 step,
                                                                 bbox,
                                                                 area,
                                                                 taken,
                                                                 min_feature_mm,
                                                                 throw_if_canceled);
        if (component_area.empty())
            continue;
        append(taken, component_area);
        TopSurfaceImageContoningVectorRegion region;
        region.bottom_to_top.emplace_back(static_cast<unsigned int>(component_id));
        region.cell_count = cell_counts[size_t(component_id)];
        region.area = std::move(component_area);
        regions.emplace_back(std::move(region));
    }

    if (!regions.empty()) {
        check_canceled(throw_if_canceled);
        ExPolygons covered = top_surface_clip_union_ex(taken);
        ExPolygons leftover = top_surface_clip_diff_ex(area, covered, ApplySafetyOffset::Yes);
        if (!leftover.empty()) {
            append(regions.front().area, std::move(leftover));
            regions.front().area = top_surface_clip_union_ex(regions.front().area);
        }
    }

    return regions;
}

static int top_surface_image_contoning_best_completion_component(const std::vector<ExPolygons> &component_areas,
                                                                 const ExPolygons              &leftover,
                                                                 coord_t                        touch_radius)
{
    int best_component = -1;
    double best_contact = 0.;
    double best_area = 0.;
    ExPolygons expanded_leftover = top_surface_clip_offset_ex(leftover, float(touch_radius));
    for (size_t component_id = 1; component_id < component_areas.size(); ++component_id) {
        if (component_areas[component_id].empty())
            continue;
        const ExPolygons contact =
            top_surface_clip_intersection_ex(expanded_leftover, component_areas[component_id], ApplySafetyOffset::Yes);
        const double contact_area = top_surface_image_abs_area(contact);
        const double component_area = top_surface_image_abs_area(component_areas[component_id]);
        if (contact_area > best_contact + EPSILON ||
            (std::abs(contact_area - best_contact) <= EPSILON && component_area > best_area)) {
            best_component = int(component_id);
            best_contact = contact_area;
            best_area = component_area;
        }
    }
    if (best_component >= 0)
        return best_component;
    for (size_t component_id = 1; component_id < component_areas.size(); ++component_id) {
        if (component_areas[component_id].empty())
            continue;
        const double component_area = top_surface_image_abs_area(component_areas[component_id]);
        if (component_area > best_area) {
            best_component = int(component_id);
            best_area = component_area;
        }
    }
    return best_component;
}

static void top_surface_image_contoning_complete_component_area(std::vector<ExPolygons>        &component_areas,
                                                               const ExPolygons               &target_area,
                                                               const std::vector<unsigned int> &fallback_components,
                                                               float                           line_width_mm,
                                                               const ThrowIfCanceled          *throw_if_canceled)
{
    if (component_areas.empty() || target_area.empty())
        return;

    ExPolygons covered;
    for (const ExPolygons &component_area : component_areas) {
        check_canceled(throw_if_canceled);
        if (!component_area.empty())
            append(covered, component_area);
    }

    ExPolygons leftover = covered.empty() ?
        top_surface_clip_union_ex(target_area) :
        top_surface_clip_diff_ex(target_area, top_surface_clip_union_ex(covered), ApplySafetyOffset::Yes);
    if (leftover.empty())
        return;

    const coord_t touch_radius =
        std::max<coord_t>(1, scale_(std::clamp(double(line_width_mm) * 0.25, 0.02, 0.20)));
    int component_id = top_surface_image_contoning_best_completion_component(component_areas, leftover, touch_radius);
    if (component_id < 0) {
        for (unsigned int fallback_component : fallback_components) {
            if (fallback_component > 0 && fallback_component < component_areas.size()) {
                component_id = int(fallback_component);
                break;
            }
        }
    }
    if (component_id < 0)
        return;

    append(component_areas[size_t(component_id)], std::move(leftover));
    component_areas[size_t(component_id)] = top_surface_clip_union_ex(component_areas[size_t(component_id)]);
}

static int top_surface_image_contoning_pattern_filaments(int stack_layers, int configured_pattern_filaments)
{
    const int clamped_stack_layers =
        std::clamp(stack_layers,
                   TextureMappingZone::MinTopSurfaceContoningStackLayers,
                   TextureMappingZone::MaxTopSurfaceContoningStackLayers);
    const int clamped_pattern_filaments =
        std::clamp(configured_pattern_filaments,
                   TextureMappingZone::MinTopSurfaceContoningPatternFilaments,
                   TextureMappingZone::MaxTopSurfaceContoningPatternFilaments);
    return std::max(1, std::min(clamped_stack_layers, clamped_pattern_filaments));
}

static std::optional<TopSurfaceImageContoningCellSample> top_surface_image_contoning_sample_cell(
    const TextureMappingOffsetContext  &context,
    const ExPolygons                   &area,
    const std::vector<ExPolygons>      &source_stack_areas,
    const ExPolygons                   &normal_filter_bypass_area,
    int                                 stack_layers,
    int                                 pattern_filaments,
    int                                 depth,
    coord_t                             x0,
    coord_t                             y0,
    coord_t                             x1,
    coord_t                             y1,
    float                               threshold_deg,
    TopSurfaceImageSourceSurface        source_surface,
    bool                                supersampled)
{
    TopSurfaceImageContoningCellSample out;
    int sample_count = 0;
    auto add_sample = [&](const Point &sample_point) {
        if (!top_surface_image_expolygons_contain_point(area, sample_point))
            return;
        const int local_stack_layers =
            top_surface_image_contoning_local_stack_layers_at_point(sample_point, source_stack_areas, stack_layers);
        if (local_stack_layers <= 0 || (depth >= 0 && depth >= local_stack_layers))
            return;
        const int solve_layers = std::min({ local_stack_layers, stack_layers, pattern_filaments });
        if (solve_layers <= 0)
            return;
        const float sample_x_mm = unscale<float>(sample_point.x());
        const float sample_y_mm = unscale<float>(sample_point.y());
        if (normal_filter_bypass_area.empty() || !top_surface_image_expolygons_contain_point(normal_filter_bypass_area, sample_point)) {
            if (!top_surface_image_contoning_sample_eligible(context, sample_x_mm, sample_y_mm, threshold_deg, source_surface))
                return;
        }
        const std::optional<std::array<float, 3>> rgb =
            texture_mapping_offset_target_rgb_at_point(context,
                                                       sample_x_mm,
                                                       sample_y_mm,
                                                       std::numeric_limits<float>::quiet_NaN());
        if (!rgb)
            return;
        out.rgb[0] += (*rgb)[0];
        out.rgb[1] += (*rgb)[1];
        out.rgb[2] += (*rgb)[2];
        out.solve_layers = sample_count == 0 ? solve_layers : std::min(out.solve_layers, solve_layers);
        out.available_depth = sample_count == 0 ? local_stack_layers : std::min(out.available_depth, local_stack_layers);
        ++sample_count;
    };

    if (supersampled) {
        static constexpr std::array<std::pair<double, double>, 5> offsets {
            std::pair<double, double>{ 0.5, 0.5 },
            std::pair<double, double>{ 0.25, 0.25 },
            std::pair<double, double>{ 0.75, 0.25 },
            std::pair<double, double>{ 0.25, 0.75 },
            std::pair<double, double>{ 0.75, 0.75 }
        };
        for (const std::pair<double, double> &offset : offsets) {
            const coord_t x = coord_t(std::llround(double(x0) + double(x1 - x0) * offset.first));
            const coord_t y = coord_t(std::llround(double(y0) + double(y1 - y0) * offset.second));
            add_sample(Point(x, y));
        }
    } else {
        add_sample(Point((x0 + x1) / 2, (y0 + y1) / 2));
    }

    if (sample_count <= 0)
        return std::nullopt;
    out.rgb[0] = std::clamp(out.rgb[0] / float(sample_count), 0.f, 1.f);
    out.rgb[1] = std::clamp(out.rgb[1] / float(sample_count), 0.f, 1.f);
    out.rgb[2] = std::clamp(out.rgb[2] / float(sample_count), 0.f, 1.f);
    if (out.solve_layers <= 0 || out.available_depth <= 0)
        return std::nullopt;
    return out;
}

static std::optional<TopSurfaceImageContoningSolvedLabel> top_surface_image_contoning_solve_label(
    const std::array<float, 3>                         &rgb,
    int                                                 solve_layers,
    const TextureMappingContoningSolver                &solver,
    const PrintConfig                                  &print_config,
    std::vector<TopSurfaceImageContoningVectorLabel>   &labels,
    std::map<std::vector<unsigned int>, int>           &label_by_stack)
{
    TextureMappingContoningStack stack = solver.solve(rgb, solve_layers);
    if (stack.bottom_to_top.empty())
        return std::nullopt;
    std::optional<std::array<float, 3>> stack_rgb =
        top_surface_image_contoning_stack_rgb(stack.bottom_to_top, print_config);
    if (!stack_rgb)
        return std::nullopt;
    top_surface_image_contoning_sort_stack_for_top_color(stack.bottom_to_top, *stack_rgb, print_config);
    top_surface_image_contoning_spread_stack_repeats(stack.bottom_to_top);
    auto label_it = label_by_stack.find(stack.bottom_to_top);
    int label = -1;
    if (label_it == label_by_stack.end()) {
        TopSurfaceImageContoningVectorLabel label_data;
        label_data.bottom_to_top = stack.bottom_to_top;
        label_data.rgb = *stack_rgb;
        label_data.oklab = color_solver_oklab_from_srgb(*stack_rgb);
        label = int(labels.size());
        labels.emplace_back(std::move(label_data));
        label_by_stack.emplace(labels.back().bottom_to_top, label);
    } else {
        label = label_it->second;
    }
    TopSurfaceImageContoningSolvedLabel out;
    out.label = label;
    out.rgb = *stack_rgb;
    return out;
}

static std::optional<TopSurfaceImageContoningSourceContext> top_surface_image_contoning_source_context(
    const TopSurfaceImageRegionPlan      &plan,
    const Layer                          &source_layer,
    const PrintObject                    &object,
    const TextureMappingZone             &zone,
    const PrintConfig                    &print_config,
    const TextureMappingContoningSolver  &solver,
    TopSurfaceImageSourceSurface          source_surface,
    const std::vector<ExPolygons>        *stack_area_extensions,
    const ThrowIfCanceled                *throw_if_canceled)
{
    if (!solver.valid())
        return std::nullopt;
    check_canceled(throw_if_canceled);

    std::optional<float> sample_z_mm;
    if (source_surface == TopSurfaceImageSourceSurface::Bottom)
        sample_z_mm = float(source_layer.bottom_z());
    std::optional<TextureMappingOffsetContext> offset_context =
        build_texture_mapping_offset_context_for_layer(object,
                                                       source_layer,
                                                       zone,
                                                       plan.zone_id,
                                                       solver.component_ids().front(),
                                                       plan.max_width_mm,
                                                       float(source_layer.height),
                                                       std::nullopt,
                                                       plan.min_width_mm,
                                                       top_surface_image_equal_blend_background(print_config, solver.component_ids()),
                                                       sample_z_mm);
    if (!offset_context)
        return std::nullopt;

    TopSurfaceImageContoningSourceContext out;
    out.offset_context = std::move(*offset_context);
    out.threshold_deg =
        std::clamp(zone.top_surface_contoning_angle_threshold_deg,
                   TextureMappingZone::MinTopSurfaceContoningAngleThresholdDeg,
                   TextureMappingZone::MaxTopSurfaceContoningAngleThresholdDeg);
    out.stack_layers = std::clamp(plan.contoning_stack_layers,
                                  TextureMappingZone::MinTopSurfaceContoningStackLayers,
                                  TextureMappingZone::MaxTopSurfaceContoningStackLayers);
    out.pattern_filaments =
        top_surface_image_contoning_pattern_filaments(out.stack_layers, plan.contoning_pattern_filaments);
    out.stack_areas =
        top_surface_image_contoning_stack_areas(source_layer, plan.zone_id, out.stack_layers, source_surface, throw_if_canceled);
    if (stack_area_extensions != nullptr) {
        const size_t count = std::min(out.stack_areas.size(), stack_area_extensions->size());
        for (size_t idx = 0; idx < count; ++idx) {
            check_canceled(throw_if_canceled);
            if ((*stack_area_extensions)[idx].empty())
                continue;
            append(out.stack_areas[idx], (*stack_area_extensions)[idx]);
            out.stack_areas[idx] = top_surface_clip_union_ex(out.stack_areas[idx]);
            append(out.normal_filter_bypass_area, (*stack_area_extensions)[idx]);
        }
        if (!out.normal_filter_bypass_area.empty())
            out.normal_filter_bypass_area = top_surface_clip_union_ex(out.normal_filter_bypass_area);
    }
    return out;
}

static std::vector<TopSurfaceImageContoningVectorRegion> top_surface_image_contoning_vector_regions(
    const TopSurfaceImageRegionPlan      &plan,
    const Layer                          &source_layer,
    const ExPolygons                     &area,
    const PrintObject                    &object,
    const TextureMappingZone             &zone,
    const PrintConfig                    &print_config,
    int                                   depth,
    const TextureMappingContoningSolver  &solver,
    TopSurfaceImageSourceSurface          source_surface,
    const TopSurfaceImageContoningSourceContext *source_context,
    bool                                  use_blue_noise_error_diffusion,
    const ThrowIfCanceled                *throw_if_canceled)
{
    std::vector<TopSurfaceImageContoningVectorRegion> regions;
    if (area.empty() || !solver.valid())
        return regions;
    check_canceled(throw_if_canceled);

    std::optional<TopSurfaceImageContoningSourceContext> local_source_context;
    const TopSurfaceImageContoningSourceContext *source = source_context;
    if (source == nullptr) {
        local_source_context = top_surface_image_contoning_source_context(plan,
                                                                         source_layer,
                                                                         object,
                                                                         zone,
                                                                         print_config,
                                                                         solver,
                                                                         source_surface,
                                                                         nullptr,
                                                                         throw_if_canceled);
        if (!local_source_context)
            return regions;
        source = &*local_source_context;
    }
    if (source->stack_layers <= 0 || source->pattern_filaments <= 0)
        return regions;

    const BoundingBox bbox = get_extents(area);
    if (!bbox.defined)
        return regions;

    const float pitch_mm = top_surface_image_contoning_sample_pitch_mm(plan, bbox);
    const coord_t step = std::max<coord_t>(1, scale_(double(pitch_mm)));
    const std::pair<coord_t, coord_t> grid_phase =
        plan.contoning_layer_phase_enabled ? top_surface_image_contoning_grid_phase(step, depth) : std::pair<coord_t, coord_t>{ 0, 0 };
    const coord_t min_x = plan.contoning_layer_phase_enabled ?
        top_surface_image_contoning_grid_min(bbox.min.x(), step, grid_phase.first) :
        (bbox.min.x() / step) * step;
    const coord_t min_y = plan.contoning_layer_phase_enabled ?
        top_surface_image_contoning_grid_min(bbox.min.y(), step, grid_phase.second) :
        (bbox.min.y() / step) * step;
    const int cols = std::max(0, int(std::ceil(double(bbox.max.x() - min_x) / double(step))));
    const int rows = std::max(0, int(std::ceil(double(bbox.max.y() - min_y) / double(step))));
    if (cols <= 0 || rows <= 0)
        return regions;

    std::vector<int> grid(size_t(cols) * size_t(rows), -1);
    std::vector<TopSurfaceImageContoningVectorLabel> labels;
    std::map<std::vector<unsigned int>, int> label_by_stack;

    auto solve_cell = [&](int row, int col, const std::array<float, 3> &target_rgb, int solve_layers) {
        std::optional<TopSurfaceImageContoningSolvedLabel> solved =
            top_surface_image_contoning_solve_label(target_rgb, solve_layers, solver, print_config, labels, label_by_stack);
        if (!solved)
            return std::optional<TopSurfaceImageContoningSolvedLabel>();
        grid[size_t(row * cols + col)] = solved->label;
        return solved;
    };

    auto sample_cell = [&](int row, int col) {
        const coord_t x0 = min_x + coord_t(col) * step;
        const coord_t y0 = min_y + coord_t(row) * step;
        const coord_t x1 = std::min<coord_t>(x0 + step, bbox.max.x());
        const coord_t y1 = std::min<coord_t>(y0 + step, bbox.max.y());
        if (x1 <= x0 || y1 <= y0)
            return std::optional<TopSurfaceImageContoningCellSample>();
        return top_surface_image_contoning_sample_cell(source->offset_context,
                                                       area,
                                                       source->stack_areas,
                                                       source->normal_filter_bypass_area,
                                                       source->stack_layers,
                                                       source->pattern_filaments,
                                                       depth,
                                                       x0,
                                                       y0,
                                                       x1,
                                                       y1,
                                                       source->threshold_deg,
                                                       source_surface,
                                                       plan.contoning_supersampled_cells_enabled);
    };

    std::vector<std::optional<TopSurfaceImageContoningCellSample>> cell_samples(grid.size());
    for (int row = 0; row < rows; ++row) {
        if ((row & 15) == 0)
            check_canceled(throw_if_canceled);
        for (int col = 0; col < cols; ++col)
            cell_samples[size_t(row * cols + col)] = sample_cell(row, col);
    }

    if (use_blue_noise_error_diffusion) {
        std::vector<std::array<float, 3>> errors(grid.size(), std::array<float, 3>{ { 0.f, 0.f, 0.f } });
        auto add_error = [&](int row, int col, const std::array<float, 3> &error, float factor) {
            if (row < 0 || row >= rows || col < 0 || col >= cols)
                return;
            std::array<float, 3> &dst = errors[size_t(row * cols + col)];
            dst[0] += error[0] * factor;
            dst[1] += error[1] * factor;
            dst[2] += error[2] * factor;
        };
        for (int row = 0; row < rows; ++row) {
            if ((row & 15) == 0)
                check_canceled(throw_if_canceled);
            const bool left_to_right = (row & 1) == 0;
            for (int step_col = 0; step_col < cols; ++step_col) {
                const int col = left_to_right ? step_col : cols - 1 - step_col;
                const std::optional<TopSurfaceImageContoningCellSample> &sample = cell_samples[size_t(row * cols + col)];
                if (!sample)
                    continue;
                const size_t grid_idx = size_t(row * cols + col);
                std::array<float, 3> target_rgb {
                    std::clamp(sample->rgb[0] + errors[grid_idx][0] + top_surface_image_contoning_jitter(col, row, depth, 0), 0.f, 1.f),
                    std::clamp(sample->rgb[1] + errors[grid_idx][1] + top_surface_image_contoning_jitter(col, row, depth, 1), 0.f, 1.f),
                    std::clamp(sample->rgb[2] + errors[grid_idx][2] + top_surface_image_contoning_jitter(col, row, depth, 2), 0.f, 1.f)
                };
                const std::optional<TopSurfaceImageContoningSolvedLabel> solved =
                    solve_cell(row, col, target_rgb, sample->solve_layers);
                if (!solved)
                    continue;
                const std::array<float, 3> error {
                    target_rgb[0] - solved->rgb[0],
                    target_rgb[1] - solved->rgb[1],
                    target_rgb[2] - solved->rgb[2]
                };
                if (left_to_right) {
                    add_error(row, col + 1, error, 7.f / 16.f);
                    add_error(row + 1, col - 1, error, 3.f / 16.f);
                    add_error(row + 1, col, error, 5.f / 16.f);
                    add_error(row + 1, col + 1, error, 1.f / 16.f);
                } else {
                    add_error(row, col - 1, error, 7.f / 16.f);
                    add_error(row + 1, col + 1, error, 3.f / 16.f);
                    add_error(row + 1, col, error, 5.f / 16.f);
                    add_error(row + 1, col - 1, error, 1.f / 16.f);
                }
            }
        }
    } else {
        for (int row = 0; row < rows; ++row) {
            if ((row & 15) == 0)
                check_canceled(throw_if_canceled);
            for (int col = 0; col < cols; ++col) {
                const std::optional<TopSurfaceImageContoningCellSample> &sample = cell_samples[size_t(row * cols + col)];
                if (!sample)
                    continue;
                solve_cell(row, col, sample->rgb, sample->solve_layers);
            }
        }
    }

    if (labels.empty())
        return regions;

    top_surface_image_contoning_merge_small_grid_regions(grid,
                                                         cols,
                                                         rows,
                                                         labels,
                                                         pitch_mm,
                                                         plan.contoning_min_feature_mm,
                                                         plan.contoning_external_width_mm,
                                                         throw_if_canceled);
    check_canceled(throw_if_canceled);

    return top_surface_image_contoning_component_regions_from_grid(grid,
                                                                  cols,
                                                                  rows,
                                                                  labels,
                                                                  depth,
                                                                  min_x,
                                                                  min_y,
                                                                  step,
                                                                  bbox,
                                                                  area,
                                                                  plan.contoning_min_feature_mm,
                                                                  plan.contoning_polygonize_color_regions_enabled,
                                                                  throw_if_canceled);
}

static std::shared_ptr<const TopSurfaceImageContoningStackPlan> top_surface_image_contoning_build_stack_plan(
    const TopSurfaceImageRegionPlan      &plan,
    const Layer                          &source_layer,
    const ExPolygons                     &source_area,
    const PrintObject                    &object,
    const TextureMappingZone             &zone,
    const PrintConfig                    &print_config,
    const TextureMappingContoningSolver  &solver,
    TopSurfaceImageSourceSurface          source_surface,
    bool                                  use_blue_noise_error_diffusion,
    const std::vector<ExPolygons>        *stack_area_extensions,
    const ThrowIfCanceled                *throw_if_canceled)
{
    std::shared_ptr<TopSurfaceImageContoningStackPlan> out = std::make_shared<TopSurfaceImageContoningStackPlan>();
    if (source_area.empty() || !solver.valid())
        return out;
    check_canceled(throw_if_canceled);

    std::optional<TopSurfaceImageContoningSourceContext> source_context =
        top_surface_image_contoning_source_context(plan,
                                                  source_layer,
                                                  object,
                                                  zone,
                                                  print_config,
                                                  solver,
                                                  source_surface,
                                                  stack_area_extensions,
                                                  throw_if_canceled);
    if (!source_context || source_context->stack_layers <= 0 || source_context->pattern_filaments <= 0)
        return out;

    const BoundingBox bbox = get_extents(source_area);
    if (!bbox.defined)
        return out;

    const float pitch_mm = top_surface_image_contoning_sample_pitch_mm(plan, bbox);
    const coord_t step = std::max<coord_t>(1, scale_(double(pitch_mm)));
    const coord_t min_x = (bbox.min.x() / step) * step;
    const coord_t min_y = (bbox.min.y() / step) * step;
    const int cols = std::max(0, int(std::ceil(double(bbox.max.x() - min_x) / double(step))));
    const int rows = std::max(0, int(std::ceil(double(bbox.max.y() - min_y) / double(step))));
    if (cols <= 0 || rows <= 0)
        return out;

    out->bbox = bbox;
    out->min_x = min_x;
    out->min_y = min_y;
    out->step = step;
    out->cols = cols;
    out->rows = rows;
    out->cells.assign(size_t(cols) * size_t(rows), TopSurfaceImageContoningStackPlanCell());

    std::map<std::vector<unsigned int>, int> label_by_stack;

    auto solve_cell = [&](int row, int col, const std::array<float, 3> &target_rgb, int solve_layers, int available_depth) {
        std::optional<TopSurfaceImageContoningSolvedLabel> solved =
            top_surface_image_contoning_solve_label(target_rgb, solve_layers, solver, print_config, out->labels, label_by_stack);
        if (!solved)
            return std::optional<TopSurfaceImageContoningSolvedLabel>();
        TopSurfaceImageContoningStackPlanCell &cell = out->cells[size_t(row * cols + col)];
        cell.label = solved->label;
        cell.available_depth = available_depth;
        return solved;
    };

    auto sample_cell = [&](int row, int col) {
        const coord_t x0 = min_x + coord_t(col) * step;
        const coord_t y0 = min_y + coord_t(row) * step;
        const coord_t x1 = std::min<coord_t>(x0 + step, bbox.max.x());
        const coord_t y1 = std::min<coord_t>(y0 + step, bbox.max.y());
        if (x1 <= x0 || y1 <= y0)
            return std::optional<TopSurfaceImageContoningCellSample>();
        return top_surface_image_contoning_sample_cell(source_context->offset_context,
                                                       source_area,
                                                       source_context->stack_areas,
                                                       source_context->normal_filter_bypass_area,
                                                       source_context->stack_layers,
                                                       source_context->pattern_filaments,
                                                       -1,
                                                       x0,
                                                       y0,
                                                       x1,
                                                       y1,
                                                       source_context->threshold_deg,
                                                       source_surface,
                                                       plan.contoning_supersampled_cells_enabled);
    };

    std::vector<std::optional<TopSurfaceImageContoningCellSample>> cell_samples(out->cells.size());
    for (int row = 0; row < rows; ++row) {
        if ((row & 15) == 0)
            check_canceled(throw_if_canceled);
        for (int col = 0; col < cols; ++col)
            cell_samples[size_t(row * cols + col)] = sample_cell(row, col);
    }

    if (use_blue_noise_error_diffusion) {
        std::vector<std::array<float, 3>> errors(out->cells.size(), std::array<float, 3>{ { 0.f, 0.f, 0.f } });
        auto add_error = [&](int row, int col, const std::array<float, 3> &error, float factor) {
            if (row < 0 || row >= rows || col < 0 || col >= cols)
                return;
            std::array<float, 3> &dst = errors[size_t(row * cols + col)];
            dst[0] += error[0] * factor;
            dst[1] += error[1] * factor;
            dst[2] += error[2] * factor;
        };
        for (int row = 0; row < rows; ++row) {
            if ((row & 15) == 0)
                check_canceled(throw_if_canceled);
            const bool left_to_right = (row & 1) == 0;
            for (int step_col = 0; step_col < cols; ++step_col) {
                const int col = left_to_right ? step_col : cols - 1 - step_col;
                const std::optional<TopSurfaceImageContoningCellSample> &sample = cell_samples[size_t(row * cols + col)];
                if (!sample)
                    continue;
                const size_t grid_idx = size_t(row * cols + col);
                std::array<float, 3> target_rgb {
                    std::clamp(sample->rgb[0] + errors[grid_idx][0] + top_surface_image_contoning_jitter(col, row, 0, 0), 0.f, 1.f),
                    std::clamp(sample->rgb[1] + errors[grid_idx][1] + top_surface_image_contoning_jitter(col, row, 0, 1), 0.f, 1.f),
                    std::clamp(sample->rgb[2] + errors[grid_idx][2] + top_surface_image_contoning_jitter(col, row, 0, 2), 0.f, 1.f)
                };
                const std::optional<TopSurfaceImageContoningSolvedLabel> solved =
                    solve_cell(row, col, target_rgb, sample->solve_layers, sample->available_depth);
                if (!solved)
                    continue;
                const std::array<float, 3> error {
                    target_rgb[0] - solved->rgb[0],
                    target_rgb[1] - solved->rgb[1],
                    target_rgb[2] - solved->rgb[2]
                };
                if (left_to_right) {
                    add_error(row, col + 1, error, 7.f / 16.f);
                    add_error(row + 1, col - 1, error, 3.f / 16.f);
                    add_error(row + 1, col, error, 5.f / 16.f);
                    add_error(row + 1, col + 1, error, 1.f / 16.f);
                } else {
                    add_error(row, col - 1, error, 7.f / 16.f);
                    add_error(row + 1, col + 1, error, 3.f / 16.f);
                    add_error(row + 1, col, error, 5.f / 16.f);
                    add_error(row + 1, col - 1, error, 1.f / 16.f);
                }
            }
        }
    } else {
        for (int row = 0; row < rows; ++row) {
            if ((row & 15) == 0)
                check_canceled(throw_if_canceled);
            for (int col = 0; col < cols; ++col) {
                const std::optional<TopSurfaceImageContoningCellSample> &sample = cell_samples[size_t(row * cols + col)];
                if (!sample)
                    continue;
                solve_cell(row, col, sample->rgb, sample->solve_layers, sample->available_depth);
            }
        }
    }

    if (out->labels.empty())
        return out;

    std::vector<int> grid(out->cells.size(), -1);
    for (size_t idx = 0; idx < out->cells.size(); ++idx)
        grid[idx] = out->cells[idx].label;
    top_surface_image_contoning_merge_small_grid_regions(grid,
                                                         cols,
                                                         rows,
                                                         out->labels,
                                                         pitch_mm,
                                                         plan.contoning_min_feature_mm,
                                                         plan.contoning_external_width_mm,
                                                         throw_if_canceled);
    for (size_t idx = 0; idx < out->cells.size(); ++idx)
        out->cells[idx].label = grid[idx];
    return out;
}

static TopSurfaceImageContoningStackPlanKey top_surface_image_contoning_stack_plan_key(
    const TopSurfaceImageRegionPlan      &plan,
    const Layer                          &source_layer,
    const TextureMappingZone             &zone,
    const TextureMappingContoningSolver  &solver,
    TopSurfaceImageSourceSurface          source_surface,
    int                                   target_depth,
    bool                                  use_blue_noise_error_diffusion)
{
    const double sample_z = source_surface == TopSurfaceImageSourceSurface::Bottom ?
        source_layer.bottom_z() :
        source_layer.print_z;
    TopSurfaceImageContoningStackPlanKey key;
    key.source_layer = &source_layer;
    key.source_layer_id = source_layer.id();
    const bool target_specific =
        plan.target_layer != nullptr &&
        (plan.contoning_replace_top_perimeters_with_infill ||
         !plan.contoning_recolor_surrounding_perimeters);
    if (target_specific) {
        key.target_layer = plan.target_layer;
        key.target_layer_id = plan.target_layer->id();
        key.region_id = plan.region_id;
        key.target_depth = target_depth;
    }
    key.source_surface = int(source_surface);
    key.sample_z = top_surface_image_contoning_float_key(sample_z);
    key.zone_id = plan.zone_id;
    key.component_ids = solver.component_ids();
    key.stack_layers = plan.contoning_stack_layers;
    key.pattern_filaments = plan.contoning_pattern_filaments;
    key.min_feature_mm = top_surface_image_contoning_float_key(plan.contoning_min_feature_mm);
    key.min_width_mm = top_surface_image_contoning_float_key(plan.min_width_mm);
    key.max_width_mm = top_surface_image_contoning_float_key(plan.max_width_mm);
    key.external_width_mm = top_surface_image_contoning_float_key(plan.contoning_external_width_mm);
    key.angle_threshold_deg = top_surface_image_contoning_float_key(zone.top_surface_contoning_angle_threshold_deg);
    key.layer_phase = plan.contoning_layer_phase_enabled;
    key.replace_top_perimeters = plan.contoning_replace_top_perimeters_with_infill;
    key.recolor_surrounding_perimeters = plan.contoning_recolor_surrounding_perimeters;
    key.supersampled = plan.contoning_supersampled_cells_enabled;
    key.blue_noise = use_blue_noise_error_diffusion;
    key.polygonize = plan.contoning_polygonize_color_regions_enabled;
    key.polygonize_high_resolution =
        plan.contoning_polygonize_color_regions_enabled &&
        plan.contoning_polygonize_high_resolution_enabled;
    return key;
}

static std::shared_ptr<const TopSurfaceImageContoningStackPlan> top_surface_image_contoning_stack_plan(
    TopSurfaceImageContoningStackPlanCache *cache,
    const TopSurfaceImageRegionPlan        &plan,
    const Layer                            &source_layer,
    const ExPolygons                       &source_area,
    const PrintObject                      &object,
    const TextureMappingZone               &zone,
    const PrintConfig                      &print_config,
    const TextureMappingContoningSolver    &solver,
    TopSurfaceImageSourceSurface            source_surface,
    int                                     target_depth,
    bool                                    use_blue_noise_error_diffusion,
    const std::vector<ExPolygons>          *stack_area_extensions,
    const ThrowIfCanceled                  *throw_if_canceled)
{
    const TopSurfaceImageContoningStackPlanKey key =
        top_surface_image_contoning_stack_plan_key(plan, source_layer, zone, solver, source_surface, target_depth, use_blue_noise_error_diffusion);
    auto builder = [&]() {
        return top_surface_image_contoning_build_stack_plan(plan,
                                                           source_layer,
                                                           source_area,
                                                           object,
                                                           zone,
                                                           print_config,
                                                           solver,
                                                           source_surface,
                                                           use_blue_noise_error_diffusion,
                                                           stack_area_extensions,
                                                           throw_if_canceled);
    };
    return cache != nullptr ? cache->get_or_build(key, builder) : builder();
}

static std::vector<TopSurfaceImageContoningVectorRegion> top_surface_image_contoning_vector_regions_from_stack_plan(
    const TopSurfaceImageRegionPlan          &plan,
    const TopSurfaceImageContoningStackPlan  &stack_plan,
    const ExPolygons                         &area,
    int                                       depth,
    const ThrowIfCanceled                    *throw_if_canceled)
{
    std::vector<TopSurfaceImageContoningVectorRegion> regions;
    if (area.empty() || stack_plan.labels.empty() || stack_plan.cells.empty() ||
        stack_plan.cols <= 0 || stack_plan.rows <= 0)
        return regions;
    check_canceled(throw_if_canceled);

    std::vector<int> grid(stack_plan.cells.size(), -1);
    for (size_t idx = 0; idx < stack_plan.cells.size(); ++idx) {
        const TopSurfaceImageContoningStackPlanCell &cell = stack_plan.cells[idx];
        if (cell.label < 0 || cell.label >= int(stack_plan.labels.size()) || depth < 0 || depth >= cell.available_depth)
            continue;
        if (stack_plan.labels[size_t(cell.label)].bottom_to_top.empty())
            continue;
        grid[idx] = cell.label;
    }

    coord_t min_x = stack_plan.min_x;
    coord_t min_y = stack_plan.min_y;
    if (plan.contoning_layer_phase_enabled) {
        const std::pair<coord_t, coord_t> grid_phase =
            top_surface_image_contoning_grid_phase(stack_plan.step, depth);
        min_x = top_surface_image_contoning_grid_min(stack_plan.bbox.min.x(), stack_plan.step, grid_phase.first);
        min_y = top_surface_image_contoning_grid_min(stack_plan.bbox.min.y(), stack_plan.step, grid_phase.second);
    }

    return top_surface_image_contoning_component_regions_from_grid(grid,
                                                                  stack_plan.cols,
                                                                  stack_plan.rows,
                                                                  stack_plan.labels,
                                                                  depth,
                                                                  min_x,
                                                                  min_y,
                                                                  stack_plan.step,
                                                                  stack_plan.bbox,
                                                                  area,
                                                                  plan.contoning_min_feature_mm,
                                                                  plan.contoning_polygonize_color_regions_enabled,
                                                                  throw_if_canceled);
}

static const Layer* top_surface_image_contoning_target_layer_for_depth(const Layer                   &source_layer,
                                                                       TopSurfaceImageSourceSurface  source_surface,
                                                                       int                           depth)
{
    const Layer *target_layer = &source_layer;
    for (int idx = 0; idx < depth && target_layer != nullptr; ++idx)
        target_layer = source_surface == TopSurfaceImageSourceSurface::Top ?
            target_layer->lower_layer :
            target_layer->upper_layer;
    return target_layer;
}

static Polygons top_surface_image_layer_perimeter_polygons(const Layer &layer, const ThrowIfCanceled *throw_if_canceled)
{
    Polygons out;
    for (const LayerRegion *layerm : layer.regions()) {
        check_canceled(throw_if_canceled);
        if (layerm != nullptr)
            layerm->perimeters.polygons_covered_by_width(out, 0.f);
    }
    return out;
}

static ExPolygons top_surface_image_contoning_printable_area(ExPolygons area, float min_feature_mm)
{
    if (area.empty())
        return {};
    area = top_surface_clip_union_ex(area);
    ExPolygons out;
    const double min_area_mm2 = std::max(0.05, double(min_feature_mm) * double(min_feature_mm) * 0.08);
    for (ExPolygon &expolygon : area)
        if (top_surface_image_scaled_area_mm2(expolygon.area()) >= min_area_mm2)
            out.emplace_back(std::move(expolygon));
    return out.empty() ? ExPolygons() : top_surface_clip_union_ex(out);
}

static std::shared_ptr<const TopSurfaceImageContoningDepthRegionPlan> top_surface_image_contoning_build_depth_region_plan(
    const TopSurfaceImageRegionPlan      &plan,
    const Layer                          &source_layer,
    const ExPolygons                     &perimeter_area,
    const PrintObject                    &object,
    const TextureMappingZone             &zone,
    const PrintConfig                    &print_config,
    const TextureMappingContoningSolver  &solver,
    TopSurfaceImageSourceSurface          source_surface,
    const std::vector<ExPolygons>        *stack_area_extensions,
    const ThrowIfCanceled                *throw_if_canceled)
{
    std::shared_ptr<TopSurfaceImageContoningDepthRegionPlan> out =
        std::make_shared<TopSurfaceImageContoningDepthRegionPlan>();
    const int stack_layers =
        std::clamp(plan.contoning_stack_layers,
                   TextureMappingZone::MinTopSurfaceContoningStackLayers,
                   TextureMappingZone::MaxTopSurfaceContoningStackLayers);
    const bool include_perimeter_regions =
        plan.contoning_replace_top_perimeters_with_infill || plan.contoning_recolor_surrounding_perimeters;
    out->fill_regions_by_depth.resize(size_t(stack_layers));
    if (include_perimeter_regions)
        out->perimeter_regions_by_depth.resize(size_t(stack_layers));
    if (perimeter_area.empty() || !solver.valid())
        return out;

    std::optional<TopSurfaceImageContoningSourceContext> source_context =
        top_surface_image_contoning_source_context(plan,
                                                  source_layer,
                                                  object,
                                                  zone,
                                                  print_config,
                                                  solver,
                                                  source_surface,
                                                  stack_area_extensions,
                                                  throw_if_canceled);
    if (!source_context || source_context->stack_layers <= 0 || source_context->pattern_filaments <= 0)
        return out;
    const TopSurfaceImageContoningSourceContext *source = &*source_context;

    auto build_depth = [&](int depth, const TextureMappingContoningSolver &depth_solver) {
        check_canceled(throw_if_canceled);
        const Layer *target_layer =
            top_surface_image_contoning_target_layer_for_depth(source_layer, source_surface, depth);
        if (target_layer == nullptr)
            return;

        ExPolygons area = perimeter_area;
        if (!plan.contoning_replace_top_perimeters_with_infill && include_perimeter_regions) {
            Polygons current_layer_perimeters =
                top_surface_image_layer_perimeter_polygons(*target_layer, throw_if_canceled);
            if (!current_layer_perimeters.empty())
                area = top_surface_clip_diff_ex(area, current_layer_perimeters, ApplySafetyOffset::Yes);
        }
        if (area.empty() && !include_perimeter_regions)
            return;

        if (plan.contoning_replace_top_perimeters_with_infill) {
            std::vector<TopSurfaceImageContoningVectorRegion> stack_regions =
                top_surface_image_contoning_vector_regions(plan,
                                                           source_layer,
                                                           perimeter_area,
                                                           object,
                                                           zone,
                                                           print_config,
                                                           depth,
                                                           depth_solver,
                                                           source_surface,
                                                           source,
                                                           plan.contoning_blue_noise_error_diffusion_enabled,
                                                           throw_if_canceled);
            out->fill_regions_by_depth[size_t(depth)] = stack_regions;
            out->perimeter_regions_by_depth[size_t(depth)] = std::move(stack_regions);
            return;
        }

        if (plan.contoning_blue_noise_error_diffusion_enabled) {
            if (!area.empty()) {
                out->fill_regions_by_depth[size_t(depth)] =
                    top_surface_image_contoning_vector_regions(plan,
                                                               source_layer,
                                                               area,
                                                               object,
                                                               zone,
                                                               print_config,
                                                               depth,
                                                               depth_solver,
                                                               source_surface,
                                                               source,
                                                               true,
                                                               throw_if_canceled);
            }
            if (include_perimeter_regions) {
                out->perimeter_regions_by_depth[size_t(depth)] =
                    top_surface_image_contoning_vector_regions(plan,
                                                               source_layer,
                                                               perimeter_area,
                                                               object,
                                                               zone,
                                                               print_config,
                                                               depth,
                                                               depth_solver,
                                                               source_surface,
                                                               source,
                                                               false,
                                                               throw_if_canceled);
            }
        } else {
            std::vector<TopSurfaceImageContoningVectorRegion> stack_regions =
                top_surface_image_contoning_vector_regions(plan,
                                                           source_layer,
                                                           perimeter_area,
                                                           object,
                                                           zone,
                                                           print_config,
                                                           depth,
                                                           depth_solver,
                                                           source_surface,
                                                           source,
                                                           false,
                                                           throw_if_canceled);
            if (include_perimeter_regions) {
                out->fill_regions_by_depth[size_t(depth)] = stack_regions;
                out->perimeter_regions_by_depth[size_t(depth)] = std::move(stack_regions);
            } else {
                out->fill_regions_by_depth[size_t(depth)] = std::move(stack_regions);
            }
        }
    };

    if (stack_layers <= 1) {
        TextureMappingContoningSolver depth_solver = solver;
        for (int depth = 0; depth < stack_layers; ++depth)
            build_depth(depth, depth_solver);
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, stack_layers, 1),
            [&](const tbb::blocked_range<int> &range) {
                TextureMappingContoningSolver depth_solver = solver;
                for (int depth = range.begin(); depth != range.end(); ++depth)
                    build_depth(depth, depth_solver);
            });
    }

    return out;
}

static std::shared_ptr<const TopSurfaceImageContoningDepthRegionPlan> top_surface_image_contoning_depth_region_plan(
    TopSurfaceImageContoningStackPlanCache *cache,
    const TopSurfaceImageRegionPlan        &plan,
    const Layer                            &source_layer,
    const ExPolygons                       &perimeter_area,
    const PrintObject                      &object,
    const TextureMappingZone               &zone,
    const PrintConfig                      &print_config,
    const TextureMappingContoningSolver    &solver,
    TopSurfaceImageSourceSurface            source_surface,
    int                                     target_depth,
    const std::vector<ExPolygons>          *stack_area_extensions,
    const ThrowIfCanceled                  *throw_if_canceled)
{
    TopSurfaceImageContoningStackPlanKey key =
        top_surface_image_contoning_stack_plan_key(plan,
                                                   source_layer,
                                                   zone,
                                                   solver,
                                                   source_surface,
                                                   target_depth,
                                                   plan.contoning_blue_noise_error_diffusion_enabled);
    auto builder = [&]() {
        return top_surface_image_contoning_build_depth_region_plan(plan,
                                                                  source_layer,
                                                                  perimeter_area,
                                                                  object,
                                                                  zone,
                                                                  print_config,
                                                                  solver,
                                                                  source_surface,
                                                                  stack_area_extensions,
                                                                  throw_if_canceled);
    };
    return cache != nullptr ? cache->get_or_build_depth_regions(key, builder) : builder();
}

static void top_surface_image_append_contoning_slices(TopSurfaceImageRegionPlan          &plan,
                                                      const Layer                        &source_layer,
                                                      const ExPolygons                   &fill_area,
                                                      const ExPolygons                   &vector_area,
                                                      const ExPolygons                   &perimeter_clip_area,
                                                      const ExPolygons                   &stack_extension_area,
                                                      const PrintObject                  &object,
                                                      const TextureMappingZone           &zone,
                                                      const PrintConfig                  &print_config,
                                                      int                                 depth,
                                                      const TextureMappingContoningSolver &solver,
                                                      TopSurfaceImageSourceSurface         source_surface,
                                                      TopSurfaceImageContoningStackPlanCache *stack_plan_cache,
                                                      const ThrowIfCanceled               *throw_if_canceled)
{
    if (vector_area.empty() || !solver.valid())
        return;
    check_canceled(throw_if_canceled);
    const int stack_layers =
        std::clamp(plan.contoning_stack_layers,
                   TextureMappingZone::MinTopSurfaceContoningStackLayers,
                   TextureMappingZone::MaxTopSurfaceContoningStackLayers);
    const int pattern_filaments =
        top_surface_image_contoning_pattern_filaments(stack_layers, plan.contoning_pattern_filaments);
    const bool include_perimeter_regions =
        plan.contoning_replace_top_perimeters_with_infill || plan.contoning_recolor_surrounding_perimeters;
    std::vector<ExPolygons> stack_area_extensions;
    const std::vector<ExPolygons> *stack_area_extensions_ptr = nullptr;
    if (plan.contoning_replace_top_perimeters_with_infill && !stack_extension_area.empty() && depth >= 0) {
        const int extension_layers = std::min(stack_layers, depth + 1);
        stack_area_extensions.resize(size_t(extension_layers));
        for (ExPolygons &extension : stack_area_extensions)
            extension = stack_extension_area;
        stack_area_extensions_ptr = &stack_area_extensions;
    }

    std::vector<ExPolygons> by_component(print_config.filament_colour.values.size() + 1);
    std::vector<ExPolygons> perimeter_by_component(include_perimeter_regions ? print_config.filament_colour.values.size() + 1 : 0);
    auto append_regions = [&](const std::vector<TopSurfaceImageContoningVectorRegion> &regions,
                              bool for_fill,
                              bool for_perimeter) {
        for (const TopSurfaceImageContoningVectorRegion &region : regions) {
            check_canceled(throw_if_canceled);
            if (depth < 0 || region.bottom_to_top.empty() || region.area.empty())
                continue;
            const int pattern_depth = depth % int(region.bottom_to_top.size());
            const unsigned int component_id =
                region.bottom_to_top[size_t(int(region.bottom_to_top.size()) - 1 - pattern_depth)];
            if (component_id == 0 || component_id >= by_component.size())
                continue;
            if (for_perimeter &&
                include_perimeter_regions &&
                !perimeter_clip_area.empty()) {
                ExPolygons component_perimeter_area = top_surface_clip_intersection_ex(region.area, perimeter_clip_area, ApplySafetyOffset::Yes);
                if (!component_perimeter_area.empty())
                    append(perimeter_by_component[component_id], std::move(component_perimeter_area));
            }
            if (for_fill && !fill_area.empty()) {
                ExPolygons component_area = top_surface_clip_intersection_ex(region.area, fill_area, ApplySafetyOffset::Yes);
                if (!component_area.empty())
                    append(by_component[component_id], std::move(component_area));
            }
        }
    };
    std::optional<TopSurfaceImageContoningSourceContext> source_context;
    bool source_context_attempted = false;
    auto get_source_context = [&]() -> const TopSurfaceImageContoningSourceContext* {
        if (!source_context_attempted) {
            source_context_attempted = true;
            source_context = top_surface_image_contoning_source_context(plan,
                                                                        source_layer,
                                                                        object,
                                                                        zone,
                                                                        print_config,
                                                                        solver,
                                                                        source_surface,
                                                                        stack_area_extensions_ptr,
                                                                        throw_if_canceled);
        }
        return source_context ? &*source_context : nullptr;
    };

    if (plan.contoning_surface_anchored_stacks_enabled) {
        if (plan.contoning_blue_noise_error_diffusion_enabled || plan.contoning_layer_phase_enabled) {
            std::shared_ptr<const TopSurfaceImageContoningDepthRegionPlan> depth_region_plan =
                top_surface_image_contoning_depth_region_plan(stack_plan_cache,
                                                              plan,
                                                              source_layer,
                                                              vector_area,
                                                              object,
                                                              zone,
                                                              print_config,
                                                              solver,
                                                              source_surface,
                                                              depth,
                                                              stack_area_extensions_ptr,
                                                              throw_if_canceled);
            if (depth >= 0 && depth < int(depth_region_plan->fill_regions_by_depth.size()))
                append_regions(depth_region_plan->fill_regions_by_depth[size_t(depth)], true, false);
            if (include_perimeter_regions && depth >= 0 && depth < int(depth_region_plan->perimeter_regions_by_depth.size()))
                append_regions(depth_region_plan->perimeter_regions_by_depth[size_t(depth)], false, true);
        } else {
            std::shared_ptr<const TopSurfaceImageContoningStackPlan> stack_plan =
                top_surface_image_contoning_stack_plan(stack_plan_cache,
                                                       plan,
                                                       source_layer,
                                                       vector_area,
                                                       object,
                                                       zone,
                                                       print_config,
                                                       solver,
                                                       source_surface,
                                                       depth,
                                                       false,
                                                       stack_area_extensions_ptr,
                                                       throw_if_canceled);
            const std::vector<TopSurfaceImageContoningVectorRegion> stack_regions =
                top_surface_image_contoning_vector_regions_from_stack_plan(plan, *stack_plan, vector_area, depth, throw_if_canceled);
            append_regions(stack_regions, true, include_perimeter_regions);
        }
    } else if (plan.contoning_blue_noise_error_diffusion_enabled) {
        if (plan.contoning_replace_top_perimeters_with_infill) {
            if (const TopSurfaceImageContoningSourceContext *ctx = get_source_context()) {
                const std::vector<TopSurfaceImageContoningVectorRegion> stack_regions =
                    top_surface_image_contoning_vector_regions(plan,
                                                               source_layer,
                                                               vector_area,
                                                               object,
                                                               zone,
                                                               print_config,
                                                               depth,
                                                               solver,
                                                               source_surface,
                                                               ctx,
                                                               true,
                                                               throw_if_canceled);
                append_regions(stack_regions, true, true);
            }
        } else if (!fill_area.empty()) {
            if (const TopSurfaceImageContoningSourceContext *ctx = get_source_context()) {
                const std::vector<TopSurfaceImageContoningVectorRegion> fill_regions =
                    top_surface_image_contoning_vector_regions(plan,
                                                               source_layer,
                                                               fill_area,
                                                               object,
                                                               zone,
                                                               print_config,
                                                               depth,
                                                               solver,
                                                               source_surface,
                                                               ctx,
                                                               true,
                                                               throw_if_canceled);
                append_regions(fill_regions, true, false);
            }
        }
        if (include_perimeter_regions && !plan.contoning_replace_top_perimeters_with_infill) {
            if (const TopSurfaceImageContoningSourceContext *ctx = get_source_context()) {
                const std::vector<TopSurfaceImageContoningVectorRegion> perimeter_regions =
                    top_surface_image_contoning_vector_regions(plan,
                                                               source_layer,
                                                               vector_area,
                                                               object,
                                                               zone,
                                                               print_config,
                                                               depth,
                                                               solver,
                                                               source_surface,
                                                               ctx,
                                                               false,
                                                               throw_if_canceled);
                append_regions(perimeter_regions, false, true);
            }
        }
    } else {
        if (const TopSurfaceImageContoningSourceContext *ctx = get_source_context()) {
            const std::vector<TopSurfaceImageContoningVectorRegion> stack_regions =
                top_surface_image_contoning_vector_regions(plan,
                                                           source_layer,
                                                           vector_area,
                                                           object,
                                                           zone,
                                                           print_config,
                                                           depth,
                                                           solver,
                                                           source_surface,
                                                           ctx,
                                                           false,
                                                           throw_if_canceled);
            append_regions(stack_regions, true, include_perimeter_regions);
        }
    }

    if (plan.contoning_replace_top_perimeters_with_infill) {
        top_surface_image_contoning_complete_component_area(by_component,
                                                            fill_area,
                                                            plan.components_bottom_to_top,
                                                            plan.contoning_external_width_mm,
                                                            throw_if_canceled);
        top_surface_image_contoning_complete_component_area(perimeter_by_component,
                                                            perimeter_clip_area,
                                                            plan.components_bottom_to_top,
                                                            plan.contoning_external_width_mm,
                                                            throw_if_canceled);
    }

    ExPolygons depth_taken;
    ExPolygons perimeter_depth_taken;
    for (unsigned int component_id = 1; component_id < by_component.size(); ++component_id) {
        check_canceled(throw_if_canceled);
        const bool component_perimeter_empty =
            !include_perimeter_regions || perimeter_by_component[component_id].empty();
        if (by_component[component_id].empty() && component_perimeter_empty)
            continue;
        ExPolygons component_area = by_component[component_id].empty() ? ExPolygons() : top_surface_clip_union_ex(by_component[component_id]);
        ExPolygons component_perimeter_area =
            component_perimeter_empty ? ExPolygons() : top_surface_clip_union_ex(perimeter_by_component[component_id]);
        if (!depth_taken.empty() && !component_area.empty())
            component_area = top_surface_clip_diff_ex(component_area, depth_taken, ApplySafetyOffset::Yes);
        if (!perimeter_depth_taken.empty() && !component_perimeter_area.empty())
            component_perimeter_area = top_surface_clip_diff_ex(component_perimeter_area, perimeter_depth_taken, ApplySafetyOffset::Yes);
        if (component_area.empty() && component_perimeter_area.empty())
            continue;
        if (!component_area.empty())
            append(depth_taken, component_area);
        if (!component_perimeter_area.empty())
            append(perimeter_depth_taken, component_perimeter_area);
        TopSurfaceImageStackSlice slice;
        slice.component_id = component_id;
        slice.depth = depth;
        slice.component_index = size_t(depth % pattern_filaments);
        slice.component_count = size_t(pattern_filaments);
        slice.contoning = true;
        slice.lower_surface = source_surface == TopSurfaceImageSourceSurface::Bottom;
        slice.angle_rad = top_surface_image_contoning_angle_rad(depth, plan.contoning_varied_infill_angles_enabled);
        slice.area = std::move(component_area);
        slice.perimeter_area = std::move(component_perimeter_area);
        plan.slices.emplace_back(std::move(slice));
    }
}

static std::vector<TopSurfaceImageRegionPlan> top_surface_image_region_plans(
    const Layer                                 &layer,
    TopSurfaceImageContoningStackPlanCache      *contoning_stack_plan_cache,
    const ThrowIfCanceled                       *throw_if_canceled)
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
    for (const LayerRegion *layerm : layer.regions()) {
        check_canceled(throw_if_canceled);
        if (layerm != nullptr)
            layerm->perimeters.polygons_covered_by_width(current_layer_perimeters, 0.f);
    }

    for (size_t region_id = 0; region_id < layer.regions().size(); ++region_id) {
        check_canceled(throw_if_canceled);
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
            zone->is_image_texture() ?
                TextureMappingManager::effective_texture_component_ids(*zone, num_physical, filament_colours) :
                TextureMappingManager::selected_component_ids(*zone, num_physical);
        components.erase(std::remove_if(components.begin(), components.end(), [num_physical](unsigned int id) {
            return id == 0 || id > num_physical;
        }), components.end());
        components.erase(std::unique(components.begin(), components.end()), components.end());
        if (components.empty())
            continue;

        TopSurfaceImageRegionPlan plan;
        plan.target_layer = &layer;
        plan.zone = zone;
        plan.region_id = region_id;
        plan.zone_id = zone_id;
        plan.same_layer_partition =
            zone->top_surface_image_printing_method == int(TextureMappingZone::TopSurfaceImageSameLayer45Partition);
        plan.contoning = zone->top_surface_contoning_active();
        plan.components_bottom_to_top = plan.same_layer_partition ?
            components :
            (plan.contoning ?
                 texture_mapping_contoning_components_bottom_to_top(*zone, print_config, components) :
                 top_surface_image_components_bottom_to_top(*zone, print_config, components));
        if (plan.components_bottom_to_top.empty())
            continue;
        plan.max_width_mm = std::clamp(zone->top_surface_image_max_line_width_mm,
                                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
        plan.min_width_mm = std::clamp(zone->top_surface_image_min_line_width_mm,
                                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                                       plan.max_width_mm);
        plan.fixed_coloring = zone->top_surface_image_fixed_coloring_filaments_active();
        plan.color_lower_surfaces = zone->top_surface_contoning_color_lower_surfaces;
        plan.colored_top_layers = std::clamp(zone->top_surface_image_colored_top_layers,
                                             TextureMappingZone::MinTopSurfaceImageColoredTopLayers,
                                             TextureMappingZone::MaxTopSurfaceImageColoredTopLayers);
        plan.contoning_stack_layers = std::clamp(zone->top_surface_contoning_stack_layers,
                                                 TextureMappingZone::MinTopSurfaceContoningStackLayers,
                                                 TextureMappingZone::MaxTopSurfaceContoningStackLayers);
        plan.contoning_pattern_filaments = std::clamp(zone->top_surface_contoning_pattern_filaments,
                                                      TextureMappingZone::MinTopSurfaceContoningPatternFilaments,
                                                      TextureMappingZone::MaxTopSurfaceContoningPatternFilaments);
        plan.contoning_external_width_mm = float(layerm->flow(frExternalPerimeter).width());
        plan.contoning_min_feature_mm =
            texture_mapping_contoning_min_feature_mm(*zone, print_config, components, plan.contoning_external_width_mm);
        plan.contoning_replace_top_perimeters_with_infill =
            zone->effective_top_surface_contoning_replace_top_perimeters_with_infill();
        plan.contoning_only_one_perimeter_around_shell_infill =
            zone->top_surface_contoning_only_one_perimeter_around_shell_infill &&
            !plan.contoning_replace_top_perimeters_with_infill;
        plan.contoning_recolor_surrounding_perimeters =
            zone->effective_top_surface_contoning_recolor_surrounding_perimeters();
        plan.contoning_perimeter_mode =
            std::clamp(zone->stored_top_surface_contoning_perimeter_mode(),
                       int(TextureMappingZone::ContoningPerimeterSegmentBlocks),
                       int(TextureMappingZone::ContoningPerimeterSegmentInfill));
        plan.contoning_flat_surface_infill_mode =
            std::clamp(zone->effective_top_surface_contoning_flat_surface_infill_mode(),
                       int(TextureMappingZone::ContoningFlatSurfaceInfillRectilinear),
                       int(TextureMappingZone::ContoningFlatSurfaceInfillRectilinearWithBoundary));
        plan.contoning_layer_phase_enabled = zone->top_surface_contoning_layer_phase_enabled;
        plan.contoning_varied_infill_angles_enabled = zone->top_surface_contoning_varied_infill_angles_enabled;
        plan.contoning_blue_noise_error_diffusion_enabled = zone->top_surface_contoning_blue_noise_error_diffusion_enabled;
        plan.contoning_supersampled_cells_enabled = zone->top_surface_contoning_supersampled_cells_enabled;
        plan.contoning_polygonize_color_regions_enabled = zone->top_surface_contoning_polygonize_color_regions_enabled;
        plan.contoning_polygonize_high_resolution_enabled =
            zone->top_surface_contoning_polygonize_high_resolution_enabled;
        plan.contoning_surface_anchored_stacks_enabled =
            zone->effective_top_surface_contoning_surface_anchored_stacks_enabled();
        const TextureMappingContoningSolver contoning_solver(*zone, print_config, components);

        const int stack_depth = plan.contoning ?
            plan.contoning_stack_layers :
            (plan.same_layer_partition ?
            plan.colored_top_layers :
            int(plan.components_bottom_to_top.size()));
        ExPolygons current_region_wall_area;
        if (plan.contoning && plan.contoning_replace_top_perimeters_with_infill) {
            Polygons current_region_wall_polygons;
            layerm->perimeters.polygons_covered_by_width(current_region_wall_polygons, 0.f);
            if (!current_region_wall_polygons.empty())
                current_region_wall_area = top_surface_clip_union_ex(current_region_wall_polygons);
        }
        if (plan.contoning) {
            auto append_contoning_surface = [&](TopSurfaceImageSourceSurface source_surface) {
                const bool target_surface_limited =
                    plan.contoning_replace_top_perimeters_with_infill ||
                    !plan.contoning_recolor_surrounding_perimeters;
                const ExPolygons current_target_surface_area =
                    target_surface_limited ?
                    top_surface_image_current_layer_surface_mask(*layerm, source_surface) :
                    ExPolygons();
                std::vector<const Layer *> contoning_depth_layers(size_t(stack_depth), nullptr);
                std::vector<ExPolygons> contoning_fill_areas(static_cast<size_t>(stack_depth));
                std::vector<ExPolygons> contoning_vector_areas(static_cast<size_t>(stack_depth));
                std::vector<ExPolygons> contoning_perimeter_clip_areas(static_cast<size_t>(stack_depth));
                std::vector<ExPolygons> contoning_stack_extension_areas(static_cast<size_t>(stack_depth));
                for (int depth = 0; depth < stack_depth; ++depth) {
                    check_canceled(throw_if_canceled);
                    const Layer *source_layer = &layer;
                    for (int i = 0; i < depth && source_layer != nullptr; ++i)
                        source_layer = source_surface == TopSurfaceImageSourceSurface::Top ?
                            source_layer->upper_layer :
                            source_layer->lower_layer;
                    if (source_layer == nullptr)
                        break;
                    if (target_surface_limited &&
                        !top_surface_image_contoning_depth_within_shell(layer,
                                                                         *source_layer,
                                                                         layerm->region().config(),
                                                                         source_surface,
                                                                         depth))
                        break;

                    ExPolygons source_area = top_surface_image_visible_surface_mask(*source_layer, zone_id, source_surface);
                    if (source_area.empty())
                        continue;
                    ExPolygons fill_area = source_area;
                    ExPolygons vector_area = source_area;
                    ExPolygons perimeter_clip_area = source_area;
                    if (target_surface_limited) {
                        if (current_target_surface_area.empty())
                            continue;
                        fill_area = top_surface_clip_intersection_ex(fill_area, current_target_surface_area, ApplySafetyOffset::Yes);
                        if (fill_area.empty())
                            continue;
                    }
                    if (plan.contoning_replace_top_perimeters_with_infill) {
                        vector_area = fill_area;
                        perimeter_clip_area = current_region_wall_area;
                        if (!perimeter_clip_area.empty()) {
                            append(vector_area, perimeter_clip_area);
                            vector_area = top_surface_clip_union_ex(vector_area);
                        }
                    }
                    if (!plan.contoning_replace_top_perimeters_with_infill) {
                        if (!plan.contoning_recolor_surrounding_perimeters)
                            vector_area = fill_area;
                    }
                    if (fill_area.empty() && perimeter_clip_area.empty() && !plan.contoning_recolor_surrounding_perimeters)
                        continue;
                    contoning_depth_layers[size_t(depth)] = source_layer;
                    contoning_fill_areas[size_t(depth)] = std::move(fill_area);
                    contoning_vector_areas[size_t(depth)] = std::move(vector_area);
                    contoning_perimeter_clip_areas[size_t(depth)] = std::move(perimeter_clip_area);
                    contoning_stack_extension_areas[size_t(depth)] = contoning_perimeter_clip_areas[size_t(depth)];
                }
                for (int depth = 0; depth < stack_depth; ++depth) {
                    check_canceled(throw_if_canceled);
                    if (contoning_depth_layers[size_t(depth)] == nullptr)
                        continue;
                    if (contoning_vector_areas[size_t(depth)].empty())
                        continue;
                    top_surface_image_append_contoning_slices(plan,
                                                              *contoning_depth_layers[size_t(depth)],
                                                              contoning_fill_areas[size_t(depth)],
                                                              contoning_vector_areas[size_t(depth)],
                                                              contoning_perimeter_clip_areas[size_t(depth)],
                                                              contoning_stack_extension_areas[size_t(depth)],
                                                              *object,
                                                              *zone,
                                                              print_config,
                                                              depth,
                                                              contoning_solver,
                                                              source_surface,
                                                              contoning_stack_plan_cache,
                                                              throw_if_canceled);
                }
            };
            append_contoning_surface(TopSurfaceImageSourceSurface::Top);
            if (plan.color_lower_surfaces)
                append_contoning_surface(TopSurfaceImageSourceSurface::Bottom);
        } else {
            for (int depth = 0; depth < stack_depth; ++depth) {
                check_canceled(throw_if_canceled);
                const Layer *top_layer = &layer;
                for (int i = 0; i < depth && top_layer != nullptr; ++i)
                    top_layer = top_layer->upper_layer;
                if (top_layer == nullptr)
                    break;

                ExPolygons area = top_surface_image_visible_top_mask(*top_layer, zone_id);
                if (area.empty())
                    continue;
                if (!current_layer_perimeters.empty())
                    area = top_surface_clip_diff_ex(area, current_layer_perimeters, ApplySafetyOffset::Yes);
                if (area.empty())
                    continue;

                if (plan.same_layer_partition) {
                    const ExPolygons depth_area = area;
                    for (size_t component_idx = 0; component_idx < plan.components_bottom_to_top.size(); ++component_idx) {
                        check_canceled(throw_if_canceled);
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

struct TopSurfaceImagePerimeterMask {
    unsigned int component_id = 0;
    unsigned int zone_id = 0;
    int depth = 0;
    bool lower_surface = false;
    float angle_rad = float(PI / 4.0);
    ExPolygons area;
};

static std::vector<TopSurfaceImagePerimeterMask> top_surface_image_contoning_perimeter_masks(const TopSurfaceImageRegionPlan &plan)
{
    std::vector<TopSurfaceImagePerimeterMask> masks;
    for (const TopSurfaceImageStackSlice &slice : plan.slices) {
        if (!slice.contoning || slice.component_id == 0 || slice.perimeter_area.empty())
            continue;
        TopSurfaceImagePerimeterMask mask;
        mask.component_id = slice.component_id;
        mask.zone_id = plan.zone_id;
        mask.depth = slice.depth;
        mask.lower_surface = slice.lower_surface;
        mask.angle_rad = slice.angle_rad;
        mask.area = slice.perimeter_area;
        masks.emplace_back(std::move(mask));
    }
    return masks;
}

static ExPolygons top_surface_image_contoning_replacement_reservation_area(const TopSurfaceImageRegionPlan &plan)
{
    if (!plan.contoning || !plan.contoning_replace_top_perimeters_with_infill)
        return {};
    ExPolygons area;
    for (const TopSurfaceImageStackSlice &slice : plan.slices)
        if (!slice.perimeter_area.empty())
            append(area, slice.perimeter_area);
    return area.empty() ? ExPolygons() : top_surface_clip_union_ex(area);
}

static ExPolygons top_surface_image_contoning_matching_surface_fill_area(const std::vector<SurfaceFill> &surface_fills,
                                                                         const TopSurfaceImagePerimeterMask &mask,
                                                                         int flat_mode)
{
    ExPolygons area;
    for (const SurfaceFill &fill : surface_fills) {
        const SurfaceFillParams &params = fill.params;
        if (!params.texture_mapping_top_surface_image ||
            !params.texture_mapping_top_surface_contoning ||
            params.texture_mapping_top_surface_contoning_flat_surface_infill_mode != flat_mode ||
            params.texture_mapping_top_surface_zone_id != mask.zone_id ||
            params.texture_mapping_top_surface_component_id != mask.component_id ||
            params.texture_mapping_top_surface_stack_depth != mask.depth ||
            fill.expolygons.empty())
            continue;
        append(area, fill.expolygons);
    }
    return area.empty() ? ExPolygons() : top_surface_clip_union_ex(area);
}

static void top_surface_image_move_collection_entities(ExtrusionEntitiesPtr &out, ExtrusionEntityCollection &collection)
{
    for (ExtrusionEntity *entity : collection.entities)
        out.emplace_back(entity);
    collection.entities.clear();
}

static void top_surface_image_delete_entities(ExtrusionEntitiesPtr &entities)
{
    for (ExtrusionEntity *entity : entities)
        delete entity;
    entities.clear();
}

static void top_surface_image_move_entities(ExtrusionEntitiesPtr &out, ExtrusionEntitiesPtr &entities)
{
    for (ExtrusionEntity *entity : entities)
        out.emplace_back(entity);
    entities.clear();
}

static void top_surface_image_append_entity_minus_mask(ExtrusionEntitiesPtr &out,
                                                       const ExtrusionEntity &entity,
                                                       const ExPolygons &mask)
{
    if (mask.empty()) {
        out.emplace_back(entity.clone());
        return;
    }

    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity)) {
        ExtrusionEntityCollection collection;
        path->subtract_expolygons(mask, &collection);
        top_surface_image_move_collection_entities(out, collection);
        return;
    }
    if (const ExtrusionMultiPath *multipath = dynamic_cast<const ExtrusionMultiPath *>(&entity)) {
        for (const ExtrusionPath &path : multipath->paths)
            top_surface_image_append_entity_minus_mask(out, path, mask);
        return;
    }
    if (const ExtrusionLoop *loop = dynamic_cast<const ExtrusionLoop *>(&entity)) {
        for (const ExtrusionPath &path : loop->paths)
            top_surface_image_append_entity_minus_mask(out, path, mask);
        return;
    }
    if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity)) {
        ExtrusionEntityCollection *out_collection = new ExtrusionEntityCollection(*collection);
        out_collection->clear();
        out_collection->no_sort = collection->no_sort;
        out_collection->texture_mapping_extruder_override = collection->texture_mapping_extruder_override;
        for (const ExtrusionEntity *child : collection->entities)
            if (child != nullptr)
                top_surface_image_append_entity_minus_mask(out_collection->entities, *child, mask);
        if (out_collection->empty())
            delete out_collection;
        else
            out.emplace_back(out_collection);
        return;
    }
    out.emplace_back(entity.clone());
}

static void top_surface_image_trim_perimeters_by_mask(ExtrusionEntityCollection &perimeters, const ExPolygons &mask)
{
    if (perimeters.entities.empty() || mask.empty())
        return;
    ExtrusionEntitiesPtr replacement;
    for (const ExtrusionEntity *entity : perimeters.entities)
        if (entity != nullptr)
            top_surface_image_append_entity_minus_mask(replacement, *entity, mask);
    perimeters.clear();
    perimeters.entities = std::move(replacement);
}

static int top_surface_image_perimeter_override_at_point(const Point &point,
                                                         const std::vector<TopSurfaceImagePerimeterMask> &masks)
{
    for (const TopSurfaceImagePerimeterMask &mask : masks)
        if (mask.component_id > 0 && top_surface_image_expolygons_contain_point(mask.area, point))
            return int(mask.component_id) - 1;
    return -1;
}

static void top_surface_image_append_perimeter_path_piece(ExtrusionEntitiesPtr &out,
                                                          const ExtrusionPath &source,
                                                          Points &&points,
                                                          int extruder_override)
{
    Polyline polyline;
    polyline.points = std::move(points);
    remove_same_neighbor(polyline);
    if (polyline.points.size() < 2 || polyline.length() <= SCALED_EPSILON)
        return;
    ExtrusionPath *path = new ExtrusionPath(std::move(polyline), source);
    if (extruder_override < 0) {
        out.emplace_back(path);
        return;
    }
    ExtrusionEntityCollection *collection = new ExtrusionEntityCollection();
    collection->texture_mapping_extruder_override = extruder_override;
    collection->entities.emplace_back(path);
    out.emplace_back(collection);
}

static bool top_surface_image_append_recolored_perimeter_path(ExtrusionEntitiesPtr &out,
                                                              const ExtrusionPath &path,
                                                              const std::vector<TopSurfaceImagePerimeterMask> &masks,
                                                              float min_run_length_mm)
{
    if (path.polyline.points.size() < 2) {
        out.emplace_back(path.clone());
        return false;
    }

    const double min_run_mm = std::max(0.05, double(min_run_length_mm));
    bool changed = false;
    ExtrusionEntitiesPtr local;
    Points current_points;
    int current_override = -2;

    auto flush = [&]() {
        if (current_points.empty())
            return;
        top_surface_image_append_perimeter_path_piece(local, path, std::move(current_points), current_override);
        current_points.clear();
        current_override = -2;
    };

    for (size_t point_idx = 1; point_idx < path.polyline.points.size(); ++point_idx) {
        const Point &p0 = path.polyline.points[point_idx - 1];
        const Point &p1 = path.polyline.points[point_idx];
        const double len_mm = unscale<double>(p0.distance_to(p1));
        if (!std::isfinite(len_mm) || len_mm <= EPSILON)
            continue;
        const int steps = std::max(1, int(std::floor(len_mm / min_run_mm)));
        for (int step = 0; step < steps; ++step) {
            const double t0 = double(step) / double(steps);
            const double t1 = double(step + 1) / double(steps);
            const Point q0 = lerp(p0, p1, t0);
            const Point q1 = lerp(p0, p1, t1);
            if (q0 == q1)
                continue;
            const Point qm = lerp(p0, p1, 0.5 * (t0 + t1));
            int extruder_override = top_surface_image_perimeter_override_at_point(qm, masks);
            if (extruder_override >= 0 && len_mm / double(steps) < min_run_mm - EPSILON)
                extruder_override = -1;
            changed = changed || extruder_override >= 0;
            if (current_points.empty()) {
                current_override = extruder_override;
                current_points.emplace_back(q0);
                current_points.emplace_back(q1);
            } else if (current_override == extruder_override && current_points.back() == q0) {
                current_points.emplace_back(q1);
            } else {
                flush();
                current_override = extruder_override;
                current_points.emplace_back(q0);
                current_points.emplace_back(q1);
            }
        }
    }
    flush();
    if (!changed) {
        top_surface_image_delete_entities(local);
        out.emplace_back(path.clone());
    } else {
        top_surface_image_move_entities(out, local);
    }
    return changed;
}

static bool top_surface_image_append_recolored_perimeter_entity(ExtrusionEntitiesPtr &out,
                                                                const ExtrusionEntity &entity,
                                                                const std::vector<TopSurfaceImagePerimeterMask> &masks,
                                                                float min_run_length_mm)
{
    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity))
        return top_surface_image_append_recolored_perimeter_path(out, *path, masks, min_run_length_mm);
    if (const ExtrusionMultiPath *multipath = dynamic_cast<const ExtrusionMultiPath *>(&entity)) {
        ExtrusionEntitiesPtr local;
        bool changed = false;
        for (const ExtrusionPath &path : multipath->paths)
            changed = top_surface_image_append_recolored_perimeter_path(local, path, masks, min_run_length_mm) || changed;
        if (changed) {
            top_surface_image_move_entities(out, local);
        } else {
            top_surface_image_delete_entities(local);
            out.emplace_back(entity.clone());
        }
        return changed;
    }
    if (const ExtrusionLoop *loop = dynamic_cast<const ExtrusionLoop *>(&entity)) {
        ExtrusionEntitiesPtr local;
        bool changed = false;
        for (const ExtrusionPath &path : loop->paths)
            changed = top_surface_image_append_recolored_perimeter_path(local, path, masks, min_run_length_mm) || changed;
        if (changed) {
            top_surface_image_move_entities(out, local);
        } else {
            top_surface_image_delete_entities(local);
            out.emplace_back(entity.clone());
        }
        return changed;
    }
    if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity)) {
        if (collection->texture_mapping_extruder_override >= 0) {
            out.emplace_back(entity.clone());
            return false;
        }
        ExtrusionEntityCollection *out_collection = new ExtrusionEntityCollection(*collection);
        out_collection->clear();
        out_collection->no_sort = collection->no_sort;
        bool changed = false;
        for (const ExtrusionEntity *child : collection->entities)
            if (child != nullptr)
                changed = top_surface_image_append_recolored_perimeter_entity(
                    out_collection->entities, *child, masks, min_run_length_mm) || changed;
        if (changed && !out_collection->empty()) {
            out.emplace_back(out_collection);
        } else {
            delete out_collection;
            out.emplace_back(entity.clone());
        }
        return changed;
    }
    out.emplace_back(entity.clone());
    return false;
}

static void top_surface_image_lift_recolored_perimeter_collections(ExtrusionEntityCollection &perimeters);

static void top_surface_image_recolor_perimeters_by_masks(ExtrusionEntityCollection &perimeters,
                                                          const std::vector<TopSurfaceImagePerimeterMask> &masks,
                                                          float min_run_length_mm)
{
    if (perimeters.entities.empty() || masks.empty())
        return;
    ExtrusionEntitiesPtr replacement;
    bool changed = false;
    for (const ExtrusionEntity *entity : perimeters.entities) {
        if (entity == nullptr)
            continue;
        changed = top_surface_image_append_recolored_perimeter_entity(replacement, *entity, masks, min_run_length_mm) || changed;
    }
    if (!changed) {
        top_surface_image_delete_entities(replacement);
        return;
    }
    perimeters.clear();
    perimeters.entities = std::move(replacement);
    top_surface_image_lift_recolored_perimeter_collections(perimeters);
}

static void top_surface_image_append_perimeter_collection(ExtrusionEntitiesPtr &out,
                                                          ExtrusionEntityCollection *collection)
{
    if (collection == nullptr)
        return;
    if (collection->empty()) {
        delete collection;
        return;
    }
    out.emplace_back(collection);
}

static void top_surface_image_lift_recolored_perimeter_collections(ExtrusionEntityCollection &perimeters)
{
    if (perimeters.entities.empty())
        return;

    ExtrusionEntitiesPtr replacement;
    bool changed = false;
    for (ExtrusionEntity *entity : perimeters.entities) {
        ExtrusionEntityCollection *collection = dynamic_cast<ExtrusionEntityCollection *>(entity);
        if (collection == nullptr || collection->texture_mapping_extruder_override >= 0) {
            replacement.emplace_back(entity);
            continue;
        }

        bool has_recolored_child = false;
        for (ExtrusionEntity *child : collection->entities) {
            ExtrusionEntityCollection *child_collection = dynamic_cast<ExtrusionEntityCollection *>(child);
            if (child_collection != nullptr && child_collection->texture_mapping_extruder_override >= 0) {
                has_recolored_child = true;
                break;
            }
        }
        if (!has_recolored_child) {
            replacement.emplace_back(entity);
            continue;
        }

        auto make_plain_collection = [&collection]() {
            ExtrusionEntityCollection *plain = new ExtrusionEntityCollection();
            plain->no_sort = collection->no_sort;
            plain->texture_mapping_top_surface_image = collection->texture_mapping_top_surface_image;
            plain->texture_mapping_top_surface_zone_id = collection->texture_mapping_top_surface_zone_id;
            plain->texture_mapping_top_surface_desired_component_id = collection->texture_mapping_top_surface_desired_component_id;
            plain->texture_mapping_top_surface_stack_depth = collection->texture_mapping_top_surface_stack_depth;
            plain->texture_mapping_top_surface_fixed_coloring = collection->texture_mapping_top_surface_fixed_coloring;
            return plain;
        };

        ExtrusionEntityCollection *plain = make_plain_collection();

        bool split = false;
        for (ExtrusionEntity *child : collection->entities) {
            ExtrusionEntityCollection *child_collection = dynamic_cast<ExtrusionEntityCollection *>(child);
            if (child_collection != nullptr && child_collection->texture_mapping_extruder_override >= 0) {
                top_surface_image_append_perimeter_collection(replacement, plain);
                plain = make_plain_collection();
                replacement.emplace_back(child_collection);
                split = true;
            } else {
                plain->entities.emplace_back(child);
            }
        }

        collection->entities.clear();
        delete collection;
        top_surface_image_append_perimeter_collection(replacement, plain);
        changed = changed || split;
    }

    if (!changed)
        return;
    perimeters.entities.clear();
    perimeters.entities = std::move(replacement);
}

static void top_surface_image_append_colored_block_loops(LayerRegion &layerm,
                                                         const ExPolygons &area,
                                                         unsigned int component_id)
{
    if (area.empty() || component_id == 0)
        return;
    Polygons loops = to_polygons(area);
    if (loops.empty())
        return;
    Flow flow = layerm.flow(frExternalPerimeter);
    ExtrusionEntityCollection *collection = new ExtrusionEntityCollection();
    collection->texture_mapping_extruder_override = int(component_id) - 1;
    extrusion_entities_append_loops(collection->entities,
                                    std::move(loops),
                                    erPerimeter,
                                    flow.mm3_per_mm(),
                                    float(flow.width()),
                                    float(flow.height()));
    if (collection->empty())
        delete collection;
    else
        layerm.perimeters.entities.emplace_back(collection);
}

static bool top_surface_image_contoning_rectilinear_with_boundary_mode(int mode)
{
    return mode == int(TextureMappingZone::ContoningFlatSurfaceInfillRectilinearWithBoundary);
}

struct TopSurfaceImageRectilinearBoundaryKey {
    size_t region_id = size_t(-1);
    unsigned int zone_id = 0;
    int depth = 0;
    SurfaceType surface_type = stInternal;
    ExtrusionRole extrusion_role = erMixed;
    float flow_width = 0.f;
    float flow_height = 0.f;
    float flow_nozzle = 0.f;
    float flow_spacing = 0.f;

    bool operator<(const TopSurfaceImageRectilinearBoundaryKey &rhs) const
    {
        return std::tie(region_id,
                        zone_id,
                        depth,
                        surface_type,
                        extrusion_role,
                        flow_width,
                        flow_height,
                        flow_nozzle,
                        flow_spacing) <
               std::tie(rhs.region_id,
                        rhs.zone_id,
                        rhs.depth,
                        rhs.surface_type,
                        rhs.extrusion_role,
                        rhs.flow_width,
                        rhs.flow_height,
                        rhs.flow_nozzle,
                        rhs.flow_spacing);
    }
};

struct TopSurfaceImageRectilinearBoundaryGroup {
    SurfaceFillParams params;
    ExPolygons area;
    std::map<unsigned int, ExPolygons> component_areas;
};

static void top_surface_image_apply_rectilinear_boundary_metadata(ExtrusionEntityCollection &collection,
                                                                  const SurfaceFillParams &params)
{
    collection.texture_mapping_top_surface_image = true;
    collection.texture_mapping_top_surface_zone_id = params.texture_mapping_top_surface_zone_id;
    collection.texture_mapping_top_surface_stack_depth = params.texture_mapping_top_surface_stack_depth;
    collection.texture_mapping_top_surface_fixed_coloring = params.texture_mapping_top_surface_fixed_coloring;
    collection.texture_mapping_top_surface_desired_component_id =
        collection.texture_mapping_extruder_override >= 0 ?
        unsigned(collection.texture_mapping_extruder_override + 1) :
        params.texture_mapping_top_surface_component_id;
    for (ExtrusionEntity *entity : collection.entities)
        if (ExtrusionEntityCollection *child = dynamic_cast<ExtrusionEntityCollection *>(entity))
            top_surface_image_apply_rectilinear_boundary_metadata(*child, params);
}

static int top_surface_image_rectilinear_boundary_override_at_point(const Point &point,
                                                                    const std::vector<TopSurfaceImagePerimeterMask> &masks,
                                                                    int fallback_override)
{
    const int override = top_surface_image_perimeter_override_at_point(point, masks);
    return override >= 0 ? override : fallback_override;
}

static void top_surface_image_append_rectilinear_boundary_polyline(ExtrusionEntitiesPtr &collections,
                                                                   const ExtrusionPath &source,
                                                                   const Polyline &polyline,
                                                                   const std::vector<TopSurfaceImagePerimeterMask> &masks,
                                                                   float sample_step_mm,
                                                                   int fallback_override)
{
    if (polyline.points.size() < 2 || fallback_override < 0)
        return;
    const double sample_step = std::max(0.02, double(sample_step_mm));
    Points current_points;
    int current_override = -2;
    auto flush = [&]() {
        if (current_points.empty())
            return;
        top_surface_image_append_perimeter_path_piece(collections, source, std::move(current_points), current_override);
        current_points.clear();
        current_override = -2;
    };
    for (size_t point_idx = 1; point_idx < polyline.points.size(); ++point_idx) {
        const Point &p0 = polyline.points[point_idx - 1];
        const Point &p1 = polyline.points[point_idx];
        const double len_mm = unscale<double>(p0.distance_to(p1));
        if (!std::isfinite(len_mm) || len_mm <= EPSILON)
            continue;
        const int steps = std::max(1, int(std::ceil(len_mm / sample_step)));
        for (int step = 0; step < steps; ++step) {
            const double t0 = double(step) / double(steps);
            const double t1 = double(step + 1) / double(steps);
            const Point q0 = lerp(p0, p1, t0);
            const Point q1 = lerp(p0, p1, t1);
            if (q0 == q1)
                continue;
            const Point qm = lerp(p0, p1, 0.5 * (t0 + t1));
            const int override = top_surface_image_rectilinear_boundary_override_at_point(qm,
                                                                                          masks,
                                                                                          current_override >= 0 ?
                                                                                          current_override :
                                                                                          fallback_override);
            if (current_points.empty()) {
                current_override = override;
                current_points.emplace_back(q0);
                current_points.emplace_back(q1);
            } else if (current_override == override && current_points.back() == q0) {
                current_points.emplace_back(q1);
            } else {
                flush();
                current_override = override;
                current_points.emplace_back(q0);
                current_points.emplace_back(q1);
            }
        }
    }
    flush();
}

static ExtrusionEntitiesPtr top_surface_image_rectilinear_boundary_collections(
    const TopSurfaceImageRectilinearBoundaryGroup &group,
    const ThrowIfCanceled *throw_if_canceled)
{
    check_canceled(throw_if_canceled);
    if (group.area.empty() || group.component_areas.empty())
        return {};

    ExPolygons boundary_area = top_surface_clip_union_ex(group.area);
    if (boundary_area.empty())
        return {};

    std::vector<TopSurfaceImagePerimeterMask> masks;
    masks.reserve(group.component_areas.size());
    const float sample_tolerance_mm =
        std::clamp(group.params.flow.width() * 0.20f, 0.02f, 0.08f);
    int fallback_override = -1;
    for (const auto &component_area : group.component_areas) {
        check_canceled(throw_if_canceled);
        if (component_area.first == 0 || component_area.second.empty())
            continue;
        ExPolygons area = top_surface_clip_union_ex(component_area.second);
        ExPolygons sample_area = top_surface_clip_offset_ex(area, float(scale_(sample_tolerance_mm)));
        if (sample_area.empty())
            sample_area = std::move(area);
        TopSurfaceImagePerimeterMask mask;
        mask.component_id = component_area.first;
        mask.zone_id = group.params.texture_mapping_top_surface_zone_id;
        mask.depth = group.params.texture_mapping_top_surface_stack_depth;
        mask.area = std::move(sample_area);
        masks.emplace_back(std::move(mask));
        if (fallback_override < 0)
            fallback_override = int(component_area.first) - 1;
    }
    if (masks.empty() || fallback_override < 0)
        return {};

    Polygons loops = to_polygons(boundary_area);
    if (loops.empty())
        return {};

    ExtrusionEntitiesPtr collections;
    ExtrusionPath source(group.params.extrusion_role,
                         group.params.flow.mm3_per_mm(),
                         group.params.flow.width(),
                         group.params.flow.height());
    const float sample_step_mm =
        std::clamp(group.params.flow.width() * 0.50f, 0.05f, 0.25f);
    for (const Polygon &loop : loops) {
        check_canceled(throw_if_canceled);
        if (loop.points.size() < 3)
            continue;
        Polyline polyline;
        polyline.points.reserve(loop.points.size() + 1);
        polyline.points.insert(polyline.points.end(), loop.points.begin(), loop.points.end());
        polyline.points.emplace_back(loop.points.front());
        remove_same_neighbor(polyline);
        if (!polyline.is_valid())
            continue;
        top_surface_image_append_rectilinear_boundary_polyline(collections,
                                                               source,
                                                               polyline,
                                                               masks,
                                                               sample_step_mm,
                                                               fallback_override);
    }
    for (ExtrusionEntity *entity : collections)
        if (ExtrusionEntityCollection *collection = dynamic_cast<ExtrusionEntityCollection *>(entity))
            top_surface_image_apply_rectilinear_boundary_metadata(*collection, group.params);
    return collections;
}

static std::map<TopSurfaceImageRectilinearBoundaryKey, TopSurfaceImageRectilinearBoundaryGroup>
top_surface_image_rectilinear_boundary_groups(const Layer &layer,
                                              const std::vector<SurfaceFill> &surface_fills,
                                              const ThrowIfCanceled *throw_if_canceled)
{
    std::map<TopSurfaceImageRectilinearBoundaryKey, TopSurfaceImageRectilinearBoundaryGroup> groups;
    for (const SurfaceFill &fill : surface_fills) {
        check_canceled(throw_if_canceled);
        if (fill.expolygons.empty() ||
            fill.region_id >= layer.regions().size() ||
            !fill.params.texture_mapping_top_surface_image ||
            !fill.params.texture_mapping_top_surface_contoning ||
            !top_surface_image_contoning_rectilinear_with_boundary_mode(
                fill.params.texture_mapping_top_surface_contoning_flat_surface_infill_mode) ||
            fill.params.texture_mapping_top_surface_component_id == 0)
            continue;
        TopSurfaceImageRectilinearBoundaryKey key;
        key.region_id = fill.region_id;
        key.zone_id = fill.params.texture_mapping_top_surface_zone_id;
        key.depth = fill.params.texture_mapping_top_surface_stack_depth;
        key.surface_type = fill.surface.surface_type;
        key.extrusion_role = fill.params.extrusion_role;
        key.flow_width = fill.params.flow.width();
        key.flow_height = fill.params.flow.height();
        key.flow_nozzle = fill.params.flow.nozzle_diameter();
        key.flow_spacing = fill.params.spacing;
        TopSurfaceImageRectilinearBoundaryGroup &group = groups[key];
        if (group.component_areas.empty()) {
            group.params = fill.params;
            group.params.texture_mapping_top_surface_component_id = 0;
        }
        append(group.area, fill.expolygons);
        append(group.component_areas[fill.params.texture_mapping_top_surface_component_id], fill.expolygons);
    }
    return groups;
}

static void top_surface_image_append_rectilinear_boundary_collections(Layer &layer,
                                                                      const std::map<TopSurfaceImageRectilinearBoundaryKey,
                                                                                     TopSurfaceImageRectilinearBoundaryGroup> &groups,
                                                                      const ThrowIfCanceled *throw_if_canceled)
{
    for (const auto &entry : groups) {
        check_canceled(throw_if_canceled);
        if (entry.first.region_id >= layer.regions().size() || layer.regions()[entry.first.region_id] == nullptr)
            continue;
        ExtrusionEntitiesPtr collections =
            top_surface_image_rectilinear_boundary_collections(entry.second, throw_if_canceled);
        for (ExtrusionEntity *entity : collections)
            layer.regions()[entry.first.region_id]->fills.entities.emplace_back(entity);
        collections.clear();
    }
}

static void top_surface_image_append_rectilinear_repair_loops(ExtrusionEntityCollection &collection,
                                                              const ExPolygon &expolygon,
                                                              const SurfaceFillParams &params)
{
    Polygons loops = to_polygons(ExPolygons { expolygon });
    if (loops.empty())
        return;
    extrusion_entities_append_loops(collection.entities,
                                    std::move(loops),
                                    params.extrusion_role,
                                    float(params.flow.mm3_per_mm()),
                                    float(params.flow.width()),
                                    float(params.flow.height()));
}

static void top_surface_image_append_rectilinear_repair_lines(ExtrusionEntityCollection &collection,
                                                              const ExPolygon &expolygon,
                                                              const SurfaceFillParams &params,
                                                              const ThrowIfCanceled *throw_if_canceled)
{
    const BoundingBox bbox = get_extents(expolygon);
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
    if (max_u < min_u || max_v < min_v)
        return;
    const double spacing = std::max(0.05, double(params.flow.spacing()) * 0.75);
    const double v_span = std::max(0.0, max_v - min_v);
    const int line_count = std::max(1, int(std::ceil(v_span / spacing)));
    const double margin = std::max(double(params.flow.spacing()), double(params.flow.width())) * 2.0;
    const ExPolygons clip { expolygon };
    for (int line_idx = 0; line_idx < line_count; ++line_idx) {
        check_canceled(throw_if_canceled);
        const double v = line_count == 1 ?
            0.5 * (min_v + max_v) :
            min_v + (double(line_idx) + 0.5) * v_span / double(line_count);
        const double u0 = min_u - margin;
        const double u1 = max_u + margin;
        Polyline polyline;
        polyline.points.emplace_back(Point::new_scale(u0 * cos_t - v * sin_t, u0 * sin_t + v * cos_t));
        polyline.points.emplace_back(Point::new_scale(u1 * cos_t - v * sin_t, u1 * sin_t + v * cos_t));
        if (!polyline.is_valid())
            continue;
        ExtrusionPath path(params.extrusion_role,
                           params.flow.mm3_per_mm(),
                           params.flow.width(),
                           params.flow.height());
        path.polyline = std::move(polyline);
        path.intersect_expolygons(clip, &collection);
    }
}

static ExPolygons top_surface_image_rectilinear_repair_leftover(
    const Layer &layer,
    const TopSurfaceImageRectilinearBoundaryKey &key,
    const TopSurfaceImageRectilinearBoundaryGroup &group,
    const ThrowIfCanceled *throw_if_canceled)
{
    if (key.region_id >= layer.regions().size() || layer.regions()[key.region_id] == nullptr || group.area.empty())
        return {};
    ExPolygons intended = top_surface_clip_union_ex(group.area);
    if (intended.empty())
        return {};
    Polygons covered_polygons;
    const LayerRegion *layerm = layer.regions()[key.region_id];
    for (const ExtrusionEntity *entity : layerm->fills.entities) {
        check_canceled(throw_if_canceled);
        const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(entity);
        if (collection == nullptr ||
            !collection->texture_mapping_top_surface_image ||
            collection->texture_mapping_top_surface_zone_id != key.zone_id ||
            collection->texture_mapping_top_surface_stack_depth != key.depth)
            continue;
        const ExtrusionRole role = collection->role();
        if (key.extrusion_role != erMixed && role != key.extrusion_role)
            continue;
        collection->polygons_covered_by_spacing(covered_polygons, float(scale_(0.02)));
    }
    if (covered_polygons.empty())
        return intended;
    ExPolygons covered =
        top_surface_clip_intersection_ex(top_surface_clip_union_ex(covered_polygons), intended, ApplySafetyOffset::Yes);
    return covered.empty() ? intended : top_surface_clip_diff_ex(intended, covered, ApplySafetyOffset::Yes);
}

static ExtrusionEntitiesPtr top_surface_image_rectilinear_repair_collections(
    const TopSurfaceImageRectilinearBoundaryGroup &group,
    const ExPolygons &leftover,
    const ThrowIfCanceled *throw_if_canceled)
{
    ExtrusionEntitiesPtr out;
    if (leftover.empty() || group.component_areas.empty())
        return out;
    const float sample_tolerance_mm =
        std::clamp(group.params.flow.width() * 0.20f, 0.02f, 0.08f);
    ExPolygons taken;
    for (const auto &component_area : group.component_areas) {
        check_canceled(throw_if_canceled);
        if (component_area.first == 0 || component_area.second.empty())
            continue;
        ExPolygons sample_area = top_surface_clip_union_ex(component_area.second);
        sample_area = top_surface_clip_offset_ex(sample_area, float(scale_(sample_tolerance_mm)));
        if (sample_area.empty())
            sample_area = top_surface_clip_union_ex(component_area.second);
        ExPolygons area = top_surface_clip_intersection_ex(leftover, sample_area, ApplySafetyOffset::Yes);
        if (!taken.empty())
            area = top_surface_clip_diff_ex(area, taken, ApplySafetyOffset::Yes);
        if (!area.empty())
            area = top_surface_clip_union_ex(area);
        if (area.empty())
            continue;
        append(taken, area);
        SurfaceFillParams params = group.params;
        params.texture_mapping_top_surface_component_id = component_area.first;
        ExtrusionEntityCollection *collection = new ExtrusionEntityCollection();
        collection->no_sort = true;
        collection->texture_mapping_extruder_override = int(component_area.first) - 1;
        for (const ExPolygon &expolygon : area) {
            check_canceled(throw_if_canceled);
            top_surface_image_append_rectilinear_repair_lines(*collection, expolygon, params, throw_if_canceled);
        }
        Polygons line_covered_polygons;
        collection->polygons_covered_by_spacing(line_covered_polygons, float(scale_(0.02)));
        ExPolygons loop_area = area;
        if (!line_covered_polygons.empty()) {
            ExPolygons line_covered =
                top_surface_clip_intersection_ex(top_surface_clip_union_ex(line_covered_polygons), area, ApplySafetyOffset::Yes);
            if (!line_covered.empty())
                loop_area = top_surface_clip_diff_ex(area, line_covered, ApplySafetyOffset::Yes);
        }
        for (const ExPolygon &expolygon : loop_area) {
            check_canceled(throw_if_canceled);
            top_surface_image_append_rectilinear_repair_loops(*collection, expolygon, params);
        }
        if (collection->empty()) {
            delete collection;
        } else {
            top_surface_image_apply_rectilinear_boundary_metadata(*collection, params);
            out.emplace_back(collection);
        }
    }
    return out;
}

static void top_surface_image_append_rectilinear_repair_collections(Layer &layer,
                                                                    const std::map<TopSurfaceImageRectilinearBoundaryKey,
                                                                                   TopSurfaceImageRectilinearBoundaryGroup> &groups,
                                                                    const ThrowIfCanceled *throw_if_canceled)
{
    std::vector<std::pair<size_t, ExtrusionEntitiesPtr>> repairs;
    for (const auto &entry : groups) {
        check_canceled(throw_if_canceled);
        ExPolygons leftover = top_surface_image_rectilinear_repair_leftover(layer,
                                                                            entry.first,
                                                                            entry.second,
                                                                            throw_if_canceled);
        if (leftover.empty())
            continue;
        ExtrusionEntitiesPtr collections =
            top_surface_image_rectilinear_repair_collections(entry.second, leftover, throw_if_canceled);
        if (!collections.empty())
            repairs.emplace_back(entry.first.region_id, std::move(collections));
    }
    for (auto &repair : repairs) {
        check_canceled(throw_if_canceled);
        if (repair.first >= layer.regions().size() || layer.regions()[repair.first] == nullptr) {
            top_surface_image_delete_entities(repair.second);
            continue;
        }
        for (ExtrusionEntity *entity : repair.second)
            layer.regions()[repair.first]->fills.entities.emplace_back(entity);
        repair.second.clear();
    }
}

static double top_surface_image_perimeter_infill_adjacency_score(const SurfaceFill &fill,
                                                                 const TopSurfaceImagePerimeterMask &mask,
                                                                 int flat_mode,
                                                                 const ExPolygons &area)
{
    const SurfaceFillParams &existing = fill.params;
    if (!existing.texture_mapping_top_surface_image ||
        !existing.texture_mapping_top_surface_contoning ||
        existing.texture_mapping_top_surface_contoning_flat_surface_infill_mode != flat_mode ||
        existing.texture_mapping_top_surface_zone_id != mask.zone_id ||
        existing.texture_mapping_top_surface_component_id != mask.component_id ||
        existing.texture_mapping_top_surface_stack_depth != mask.depth ||
        fill.expolygons.empty() ||
        area.empty())
        return 0.;

    double spacing = std::isfinite(double(existing.spacing)) && existing.spacing > EPSILON ?
        double(existing.spacing) :
        double(existing.flow.spacing());
    if (!std::isfinite(spacing) || spacing <= EPSILON)
        spacing = 0.4;
    const coord_t touch_radius =
        std::max<coord_t>(1, scale_(std::clamp(spacing * 0.25, 0.02, 0.20)));
    ExPolygons overlap = top_surface_clip_intersection_ex(top_surface_clip_offset_ex(area, float(touch_radius)), fill.expolygons, ApplySafetyOffset::Yes);
    double score = 0.;
    for (const ExPolygon &expolygon : overlap)
        score += std::abs(expolygon.area());
    return score;
}

static void top_surface_image_append_perimeter_infill_surface(std::vector<SurfaceFill> &surface_fills,
                                                              const LayerRegion &layerm,
                                                              size_t region_id,
                                                              const TopSurfaceImageRegionPlan &plan,
                                                              const TopSurfaceImagePerimeterMask &mask,
                                                              int flat_mode,
                                                              ExPolygons &&area)
{
    if (area.empty() || mask.component_id == 0)
        return;
    Surface surface(mask.lower_surface && mask.depth == 0 ? stBottom : (!mask.lower_surface && mask.depth == 0 ? stTop : stInternalSolid));
    surface.thickness = layerm.layer()->height;
    surface.thickness_layers = 1;

    const ExtrusionRole extrusion_role = surface.is_top() ? erTopSolidInfill : (surface.is_bottom() ? erBottomSurface : erSolidInfill);
    if (plan.contoning_replace_top_perimeters_with_infill) {
        SurfaceFill *adjacent_fill = nullptr;
        double best_adjacency_score = 0.;
        for (SurfaceFill &fill : surface_fills) {
            const double score = top_surface_image_perimeter_infill_adjacency_score(fill, mask, flat_mode, area);
            if (score <= best_adjacency_score)
                continue;
            adjacent_fill = &fill;
            best_adjacency_score = score;
        }
        if (adjacent_fill != nullptr) {
            append_surface_fill_expolygons(*adjacent_fill, region_id, adjacent_fill->surface, std::move(area), layerm);
            adjacent_fill->expolygons = top_surface_clip_union_ex(adjacent_fill->expolygons);
            return;
        }
    }

    for (SurfaceFill &fill : surface_fills) {
        const SurfaceFillParams &existing = fill.params;
        if (!existing.texture_mapping_top_surface_image ||
            !existing.texture_mapping_top_surface_contoning ||
            existing.texture_mapping_top_surface_contoning_flat_surface_infill_mode != flat_mode ||
            existing.texture_mapping_top_surface_zone_id != mask.zone_id ||
            existing.texture_mapping_top_surface_component_id != mask.component_id ||
            existing.texture_mapping_top_surface_stack_depth != mask.depth ||
            existing.extrusion_role != extrusion_role)
            continue;
        append_surface_fill_expolygons(fill, region_id, surface, std::move(area), layerm);
        fill.expolygons = top_surface_clip_union_ex(fill.expolygons);
        return;
    }

    SurfaceFillParams params;
    params.extruder = mask.component_id;
    params.pattern = ipRectilinear;
    if (flat_mode == int(TextureMappingZone::ContoningFlatSurfaceInfillConcentric))
        params.pattern = ipConcentric;
    params.density = 100.f;
    params.angle = mask.angle_rad;
    params.fixed_angle = true;
    params.bridge = false;
    params.bridge_angle = 0.f;
    params.multiline = 1;
    params.anchor_length = 1000.f;
    params.anchor_length_max = 1000.f;
    const FlowRole flow_role = surface.is_top() ? frTopSolidInfill : frSolidInfill;
    params.flow = layerm.flow(flow_role, layerm.layer()->height);
    params.spacing = params.flow.spacing();
    params.extrusion_role = extrusion_role;
    params.texture_mapping_top_surface_image = true;
    params.texture_mapping_top_surface_zone_id = mask.zone_id;
    params.texture_mapping_top_surface_component_id = mask.component_id;
    params.texture_mapping_top_surface_stack_depth = mask.depth;
    params.texture_mapping_top_surface_fixed_coloring = true;
    params.texture_mapping_top_surface_contoning = true;
    params.texture_mapping_top_surface_min_width_mm = plan.min_width_mm;
    params.texture_mapping_top_surface_max_width_mm = plan.max_width_mm;
    const int pattern_filaments = top_surface_image_contoning_pattern_filaments(plan.contoning_stack_layers,
                                                                                plan.contoning_pattern_filaments);
    params.texture_mapping_top_surface_component_index = pattern_filaments > 0 ? mask.depth % pattern_filaments : 0;
    params.texture_mapping_top_surface_component_count = std::max(1, pattern_filaments);
    params.texture_mapping_top_surface_contoning_flat_surface_infill_mode = flat_mode;

    SurfaceFill &fill = surface_fill_for_params(surface_fills, params);
    append_surface_fill_expolygons(fill, region_id, surface, std::move(area), layerm);
    fill.expolygons = top_surface_clip_union_ex(fill.expolygons);
}

static void top_surface_image_append_contoning_replacement_surfaces(std::vector<SurfaceFill> &surface_fills,
                                                                    const LayerRegion &layerm,
                                                                    size_t region_id,
                                                                    const TopSurfaceImageRegionPlan &plan,
                                                                    const ThrowIfCanceled *throw_if_canceled)
{
    if (!plan.contoning || !plan.contoning_replace_top_perimeters_with_infill)
        return;
    std::vector<TopSurfaceImagePerimeterMask> masks = top_surface_image_contoning_perimeter_masks(plan);
    if (masks.empty())
        return;
    ExPolygons taken;
    for (const TopSurfaceImagePerimeterMask &mask : masks) {
        check_canceled(throw_if_canceled);
        ExPolygons area = mask.area;
        if (!taken.empty())
            area = top_surface_clip_diff_ex(area, taken, ApplySafetyOffset::Yes);
        if (!area.empty())
            area = top_surface_clip_union_ex(area);
        if (area.empty())
            continue;
        append(taken, area);
        top_surface_image_append_perimeter_infill_surface(surface_fills,
                                                          layerm,
                                                          region_id,
                                                          plan,
                                                          mask,
                                                          plan.contoning_flat_surface_infill_mode,
                                                          std::move(area));
    }
}

static void top_surface_image_apply_contoning_perimeter_options(const Layer &layer,
                                                                std::vector<SurfaceFill> &surface_fills,
                                                                const std::vector<TopSurfaceImageRegionPlan> &plans,
                                                                const ThrowIfCanceled *throw_if_canceled)
{
    for (size_t region_id = 0; region_id < plans.size() && region_id < layer.regions().size(); ++region_id) {
        check_canceled(throw_if_canceled);
        const TopSurfaceImageRegionPlan &plan = plans[region_id];
        if (!plan.contoning ||
            (!plan.contoning_replace_top_perimeters_with_infill && !plan.contoning_recolor_surrounding_perimeters))
            continue;
        LayerRegion *layerm = layer.regions()[region_id];
        if (layerm == nullptr || layerm->perimeters.entities.empty())
            continue;
        std::vector<TopSurfaceImagePerimeterMask> masks = top_surface_image_contoning_perimeter_masks(plan);
        if (masks.empty())
            continue;

        Polygons wall_polygons;
        layerm->perimeters.polygons_covered_by_width(wall_polygons, 0.f);
        ExPolygons wall_area = wall_polygons.empty() ? ExPolygons() : top_surface_clip_union_ex(wall_polygons);
        if (wall_area.empty())
            continue;

        for (TopSurfaceImagePerimeterMask &mask : masks) {
            check_canceled(throw_if_canceled);
            mask.area = top_surface_clip_intersection_ex(mask.area, wall_area, ApplySafetyOffset::Yes);
            if (plan.contoning_replace_top_perimeters_with_infill && !mask.area.empty()) {
                ExPolygons fill_area =
                    top_surface_image_contoning_matching_surface_fill_area(surface_fills,
                                                                           mask,
                                                                           plan.contoning_flat_surface_infill_mode);
                if (fill_area.empty()) {
                    mask.area.clear();
                } else {
                    const Flow perimeter_flow = layerm->flow(frPerimeter);
                    const double trim_radius_mm =
                        std::clamp(double(perimeter_flow.spacing()) * 0.25, 0.02, 0.20);
                    ExPolygons trim_area = top_surface_clip_offset_ex(fill_area, float(scale_(trim_radius_mm)));
                    mask.area = top_surface_clip_intersection_ex(mask.area, trim_area, ApplySafetyOffset::Yes);
                }
            } else {
                mask.area = top_surface_image_contoning_printable_area(std::move(mask.area), plan.contoning_min_feature_mm);
            }
        }
        ExPolygons affected_area;
        for (const TopSurfaceImagePerimeterMask &mask : masks)
            if (!mask.area.empty())
                append(affected_area, mask.area);
        if (affected_area.empty())
            continue;
        affected_area = top_surface_clip_union_ex(affected_area);

        if (plan.contoning_replace_top_perimeters_with_infill) {
            top_surface_image_trim_perimeters_by_mask(layerm->perimeters, affected_area);
            continue;
        }

        if (plan.contoning_perimeter_mode == int(TextureMappingZone::ContoningPerimeterDividedLine)) {
            top_surface_image_recolor_perimeters_by_masks(layerm->perimeters, masks, plan.contoning_min_feature_mm);
            continue;
        }

        top_surface_image_trim_perimeters_by_mask(layerm->perimeters, affected_area);
        ExPolygons taken;
        for (const TopSurfaceImagePerimeterMask &mask : masks) {
            check_canceled(throw_if_canceled);
            ExPolygons area = mask.area;
            if (!taken.empty())
                area = top_surface_clip_diff_ex(area, taken, ApplySafetyOffset::Yes);
            area = top_surface_image_contoning_printable_area(std::move(area), plan.contoning_min_feature_mm);
            if (area.empty())
                continue;
            append(taken, area);
            if (plan.contoning_perimeter_mode == int(TextureMappingZone::ContoningPerimeterSegmentInfill)) {
                top_surface_image_append_perimeter_infill_surface(surface_fills,
                                                                  *layerm,
                                                                  region_id,
                                                                  plan,
                                                                  mask,
                                                                  int(TextureMappingZone::ContoningFlatSurfaceInfillRectilinear),
                                                                  std::move(area));
            } else {
                top_surface_image_append_colored_block_loops(*layerm, area, mask.component_id);
                top_surface_image_append_perimeter_infill_surface(surface_fills,
                                                                  *layerm,
                                                                  region_id,
                                                                  plan,
                                                                  mask,
                                                                  int(TextureMappingZone::ContoningFlatSurfaceInfillRectilinear),
                                                                  std::move(area));
            }
        }
    }
}

static bool top_surface_image_slice_matches_surface(const TopSurfaceImageStackSlice &slice, const Surface &surface)
{
    return slice.lower_surface ?
        (surface.surface_type == stBottom || surface.surface_type == stInternalSolid) :
        (surface.is_top() || surface.surface_type == stInternalSolid);
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
    params.extrusion_role = surface.is_top() ? erTopSolidInfill : (surface.surface_type == stBottom ? erBottomSurface : erSolidInfill);
    const PrintConfig &print_config = layer.object()->print()->config();
    const float nozzle = slice.component_id > 0 && size_t(slice.component_id - 1) < print_config.nozzle_diameter.values.size() ?
        float(print_config.nozzle_diameter.get_at(size_t(slice.component_id - 1))) :
        float(print_config.nozzle_diameter.values.empty() ? 0.4 : print_config.nozzle_diameter.values.front());
    const float height = float((surface.thickness == -1) ? layer.height : surface.thickness);
    if (slice.contoning) {
        const int flat_mode = plan.contoning_flat_surface_infill_mode;
        if (flat_mode == int(TextureMappingZone::ContoningFlatSurfaceInfillConcentric))
            params.pattern = ipConcentric;
        params.flow = base_params.flow;
        params.spacing = base_params.spacing;
    } else {
        params.flow = Flow(plan.max_width_mm, height, nozzle);
        params.spacing = slice.same_layer_partition && slice.component_count > 0 ?
            (plan.min_width_mm * float(slice.component_count) + (plan.max_width_mm - plan.min_width_mm)) :
            params.flow.spacing();
    }
    params.texture_mapping_top_surface_image = true;
    params.texture_mapping_top_surface_zone_id = plan.zone_id;
    params.texture_mapping_top_surface_component_id = slice.component_id;
    params.texture_mapping_top_surface_stack_depth = slice.depth;
    params.texture_mapping_top_surface_fixed_coloring = plan.fixed_coloring;
    params.texture_mapping_top_surface_min_width_mm = plan.min_width_mm;
    params.texture_mapping_top_surface_max_width_mm = plan.max_width_mm;
    params.texture_mapping_top_surface_same_layer_partition = slice.same_layer_partition;
    params.texture_mapping_top_surface_contoning = slice.contoning;
    params.texture_mapping_top_surface_component_index = int(slice.component_index);
    params.texture_mapping_top_surface_component_count = int(slice.component_count);
    params.texture_mapping_top_surface_contoning_flat_surface_infill_mode = plan.contoning_flat_surface_infill_mode;
    params.texture_mapping_top_surface_contoning_no_edge_overlap =
        slice.contoning && plan.contoning_polygonize_color_regions_enabled;
    return params;
}

static ExtrusionPaths top_surface_image_split_path(const ExtrusionPath &path,
                                                   const TextureMappingOffsetContext &context,
                                                   float min_width_mm,
                                                   float max_width_mm,
                                                   float nozzle_diameter,
                                                   const ThrowIfCanceled *throw_if_canceled)
{
    ExtrusionPaths out;
    if (path.polyline.points.size() < 2)
        return out;
    check_canceled(throw_if_canceled);
    const float height = path.height > 0.f ? path.height : context.layer_height_mm;
    const float sample_step_mm = std::clamp(max_width_mm * 0.5f, 0.16f, 0.40f);
    for (size_t point_idx = 1; point_idx < path.polyline.points.size(); ++point_idx) {
        if ((point_idx & 63) == 1)
            check_canceled(throw_if_canceled);
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
                texture_mapping_offset_component_weights_at_point(context,
                                                                  unscale<float>(qm.x()),
                                                                  unscale<float>(qm.y()),
                                                                  std::numeric_limits<float>::quiet_NaN());
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
                                                           float nozzle_diameter,
                                                           const ThrowIfCanceled *throw_if_canceled)
{
    if (const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(&entity)) {
        ExtrusionPaths paths = top_surface_image_split_path(*path, context, min_width_mm, max_width_mm, nozzle_diameter, throw_if_canceled);
        if (paths.empty())
            return entity.clone();
        return new ExtrusionMultiPath(std::move(paths));
    }
    if (const ExtrusionMultiPath *multi_path = dynamic_cast<const ExtrusionMultiPath *>(&entity)) {
        ExtrusionPaths paths;
        for (const ExtrusionPath &path : multi_path->paths) {
            check_canceled(throw_if_canceled);
            ExtrusionPaths split = top_surface_image_split_path(path, context, min_width_mm, max_width_mm, nozzle_diameter, throw_if_canceled);
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
            check_canceled(throw_if_canceled);
            ExtrusionPaths split = top_surface_image_split_path(path, context, min_width_mm, max_width_mm, nozzle_diameter, throw_if_canceled);
            append(paths, std::move(split));
        }
        if (paths.empty())
            return entity.clone();
        return new ExtrusionLoop(std::move(paths));
    }
    if (const ExtrusionEntityCollection *collection = dynamic_cast<const ExtrusionEntityCollection *>(&entity)) {
        ExtrusionEntityCollection *out = new ExtrusionEntityCollection(*collection);
        for (ExtrusionEntity *&child : out->entities) {
            check_canceled(throw_if_canceled);
            ExtrusionEntity *replacement =
                top_surface_image_modulated_entity(*child, context, min_width_mm, max_width_mm, nozzle_diameter, throw_if_canceled);
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
        texture_mapping_offset_component_weights_at_point(*context,
                                                          x_mm,
                                                          y_mm,
                                                          std::numeric_limits<float>::quiet_NaN());
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
                                                        const std::optional<TextureMappingOffsetContext> &context,
                                                        const ThrowIfCanceled                           *throw_if_canceled)
{
    check_canceled(throw_if_canceled);
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
        check_canceled(throw_if_canceled);
        const double band_v = double(band) * double(pitch);
        const double sample_v = band_v + double(pitch) * 0.5;
        for (double u0 = u_start; u0 < u_end - EPSILON; u0 += sample_step) {
            check_canceled(throw_if_canceled);
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

static bool top_surface_image_contoning_boundary_skin_mode(int mode)
{
    return mode == int(TextureMappingZone::ContoningFlatSurfaceInfillBoundarySkinFixed) ||
           mode == int(TextureMappingZone::ContoningFlatSurfaceInfillBoundarySkinVariable) ||
           mode == int(TextureMappingZone::ContoningFlatSurfaceInfillBoundarySkinHybrid);
}

static bool top_surface_image_contoning_spiral_mode(int mode)
{
    return mode == int(TextureMappingZone::ContoningFlatSurfaceInfillSpiral);
}

static Flow top_surface_image_contoning_boundary_skin_flow(const SurfaceFillParams &params, bool variable)
{
    const float min_width = std::clamp(params.texture_mapping_top_surface_min_width_mm,
                                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
    const float max_width = std::clamp(params.texture_mapping_top_surface_max_width_mm,
                                       min_width,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
    const float width = variable ?
        min_width :
        std::clamp(params.flow.width(), min_width, max_width);
    return Flow(width, params.flow.height(), params.flow.nozzle_diameter());
}

static void top_surface_image_filter_short_boundary_skin_entities(ExtrusionEntityCollection &collection, float min_length_mm)
{
    ExtrusionEntitiesPtr kept;
    kept.reserve(collection.entities.size());
    for (ExtrusionEntity *entity : collection.entities) {
        if (entity != nullptr && unscale<double>(entity->length()) >= double(min_length_mm))
            kept.emplace_back(entity);
        else
            delete entity;
    }
    collection.entities = std::move(kept);
}

static ExPolygons top_surface_image_boundary_skin_leftover(const Surface &surface,
                                                           const ExtrusionEntityCollection &collection)
{
    if (collection.empty())
        return ExPolygons { surface.expolygon };
    Polygons covered_polygons;
    collection.polygons_covered_by_width(covered_polygons, float(scale_(0.02)));
    if (covered_polygons.empty())
        return ExPolygons { surface.expolygon };
    ExPolygons covered = top_surface_clip_intersection_ex(top_surface_clip_union_ex(covered_polygons), ExPolygons { surface.expolygon }, ApplySafetyOffset::Yes);
    if (covered.empty())
        return ExPolygons { surface.expolygon };
    return top_surface_clip_diff_ex(ExPolygons { surface.expolygon }, covered, ApplySafetyOffset::Yes);
}

static bool top_surface_image_contoning_connector_printable(const ExPolygon &area,
                                                            const Point &from,
                                                            const Point &to,
                                                            coord_t max_length)
{
    if (from == to)
        return true;
    if (max_length > 0) {
        const double max_length_sq = double(max_length) * double(max_length);
        if ((to - from).cast<double>().squaredNorm() > max_length_sq)
            return false;
    }
    return area.contains(Line(from, to));
}

static coord_t top_surface_image_spiral_spacing(const SurfaceFillParams &params)
{
    const double spacing = params.spacing > EPSILON ? params.spacing : params.flow.spacing();
    return std::max<coord_t>(1, scaled<coord_t>(spacing));
}

static Polygons top_surface_image_spiral_loops(const Surface &surface, const SurfaceFillParams &params)
{
    const coord_t spacing = top_surface_image_spiral_spacing(params);
    const coord_t half_spacing = std::max<coord_t>(1, spacing / 2);
    const coord_t centerline_inset = std::max<coord_t>(1, params.flow.scaled_width() / 2);
    ExPolygons last = top_surface_clip_offset_ex(surface.expolygon, -float(centerline_inset));
    Polygons loops = to_polygons(last);
    while (!last.empty()) {
        last = top_surface_clip_offset2_ex(last, -(spacing + half_spacing), +half_spacing);
        append(loops, to_polygons(last));
    }
    loops = top_surface_clip_union_pt_chained_outside_in(loops);
    for (Polygon &loop : loops)
        if (loop.points.size() >= 3)
            loop.densify(float(spacing));
    return loops;
}

static int top_surface_image_spiral_loop_start_index(const Polygon &loop,
                                                     const Point &from,
                                                     const ExPolygon *clip,
                                                     coord_t max_connector)
{
    int best = -1;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < loop.points.size(); ++i) {
        const Point &point = loop.points[i];
        if (clip != nullptr &&
            !top_surface_image_contoning_connector_printable(*clip, from, point, max_connector))
            continue;
        const double dist = (point - from).cast<double>().squaredNorm();
        if (dist < best_dist) {
            best_dist = dist;
            best = int(i);
        }
    }
    return best;
}

static Polyline top_surface_image_spiral_loop_arc(const Polygon &loop, int start_idx)
{
    Polyline out;
    const int point_count = int(loop.points.size());
    if (point_count < 3)
        return out;
    int idx = ((start_idx % point_count) + point_count) % point_count;
    const int end_idx = (idx + point_count - 1) % point_count;
    out.points.reserve(size_t(point_count));
    for (;;) {
        out.points.emplace_back(loop.points[size_t(idx)]);
        if (idx == end_idx)
            break;
        idx = (idx + 1) % point_count;
    }
    remove_same_neighbor(out);
    return out;
}

static Polylines top_surface_image_spiral_polylines(const Surface &surface,
                                                    const Polygons &loops,
                                                    coord_t max_connector,
                                                    const ThrowIfCanceled *throw_if_canceled)
{
    Polylines out;
    Polyline current;
    Point last_pos(0, 0);
    for (const Polygon &loop : loops) {
        check_canceled(throw_if_canceled);
        if (loop.points.size() < 3)
            continue;
        int start_idx = current.empty() ?
            top_surface_image_spiral_loop_start_index(loop, last_pos, nullptr, 0) :
            top_surface_image_spiral_loop_start_index(loop, current.last_point(), &surface.expolygon, max_connector);
        if (start_idx < 0) {
            if (current.is_valid())
                out.emplace_back(std::move(current));
            current = Polyline();
            start_idx = top_surface_image_spiral_loop_start_index(loop, last_pos, nullptr, 0);
        }
        if (start_idx < 0)
            continue;
        Polyline loop_polyline = top_surface_image_spiral_loop_arc(loop, start_idx);
        remove_same_neighbor(loop_polyline);
        if (!loop_polyline.is_valid())
            continue;
        if (!current.empty() &&
            !top_surface_image_contoning_connector_printable(surface.expolygon,
                                                             current.last_point(),
                                                             loop_polyline.first_point(),
                                                             max_connector)) {
            if (current.is_valid())
                out.emplace_back(std::move(current));
            current = Polyline();
        }
        current.append(std::move(loop_polyline));
        last_pos = current.last_point();
    }
    if (current.is_valid())
        out.emplace_back(std::move(current));
    return out;
}

static std::unique_ptr<ExtrusionEntityCollection> top_surface_image_spiral_collection(
    const Surface           &surface,
    const SurfaceFillParams &params,
    ExPolygons              *leftover,
    const ThrowIfCanceled   *throw_if_canceled)
{
    check_canceled(throw_if_canceled);
    std::unique_ptr<ExtrusionEntityCollection> collection(new ExtrusionEntityCollection());
    collection->no_sort = true;
    const coord_t spacing = top_surface_image_spiral_spacing(params);
    const coord_t max_connector = std::max<coord_t>(1, 2 * spacing);
    Polygons loops = top_surface_image_spiral_loops(surface, params);
    Polylines polylines = top_surface_image_spiral_polylines(surface, loops, max_connector, throw_if_canceled);
    const float min_length_mm =
        std::max(0.18f, std::min(0.75f, params.flow.width() * 0.55f));
    for (Polyline &polyline : polylines) {
        check_canceled(throw_if_canceled);
        remove_same_neighbor(polyline);
        if (!polyline.is_valid() || unscale<double>(polyline.length()) < double(min_length_mm))
            continue;
        ExtrusionPath *path = new ExtrusionPath(params.extrusion_role,
                                                params.flow.mm3_per_mm(),
                                                params.flow.width(),
                                                params.flow.height());
        path->polyline = std::move(polyline);
        collection->entities.emplace_back(path);
    }
    if (leftover != nullptr)
        *leftover = top_surface_image_boundary_skin_leftover(surface, *collection);
    return collection;
}

static Polyline top_surface_image_boundary_skin_polyline_from_thick(const ThickPolyline &source)
{
    Polyline out;
    if (source.points.size() < 2)
        return out;
    out.points = source.points;
    remove_same_neighbor(out);
    return out;
}

static std::unique_ptr<ExtrusionEntityCollection> top_surface_image_boundary_skin_collection(
    const Surface             &surface,
    const SurfaceFillParams   &params,
    const PrintConfig         &print_config,
    const PrintObjectConfig   &object_config,
    int                        layer_id,
    bool                       variable,
    bool                       hybrid,
    ExPolygons                *leftover,
    const ThrowIfCanceled     *throw_if_canceled)
{
    check_canceled(throw_if_canceled);
    std::unique_ptr<ExtrusionEntityCollection> collection(new ExtrusionEntityCollection());
    collection->no_sort = true;
    const float min_width = std::clamp(params.texture_mapping_top_surface_min_width_mm,
                                       TextureMappingZone::MinTopSurfaceImageLineWidthMm,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
    const float max_width = std::clamp(params.texture_mapping_top_surface_max_width_mm,
                                       min_width,
                                       TextureMappingZone::MaxTopSurfaceImageLineWidthMm);
    const Flow max_flow = Flow(max_width, params.flow.height(), params.flow.nozzle_diameter());
    const Flow min_flow = Flow(min_width, params.flow.height(), params.flow.nozzle_diameter());
    const Flow output_flow = top_surface_image_contoning_boundary_skin_flow(params, variable);
    const coord_t preferred_spacing = std::max<coord_t>(1, variable ? max_flow.scaled_spacing() : output_flow.scaled_spacing());
    const coordf_t min_spacing = std::max<coordf_t>(1.0, min_flow.scaled_spacing());
    const coordf_t max_spacing = std::max<coordf_t>(min_spacing, max_flow.scaled_spacing());
    const coordf_t output_spacing = std::max<coordf_t>(1.0, output_flow.scaled_spacing());

    Polygons outline = to_polygons(surface.expolygon);
    ExPolygons inner_leftover;
    if (!outline.empty()) {
        Arachne::WallToolPathsParams input_params =
            Arachne::make_paths_params(layer_id, object_config, print_config);
        input_params.min_bead_width = variable ? min_width : output_flow.width();
        if (!std::isfinite(input_params.min_feature_size) || input_params.min_feature_size <= 0.f)
            input_params.min_feature_size = min_width * 0.5f;
        else
            input_params.min_feature_size = std::min(input_params.min_feature_size, min_width * 0.5f);
        input_params.min_length_factor =
            (!std::isfinite(input_params.min_length_factor) || input_params.min_length_factor <= 0.f) ?
                0.5f :
                std::min(input_params.min_length_factor, 0.5f);
        input_params.wall_distribution_count = std::max(1, input_params.wall_distribution_count);
        input_params.is_top_or_bottom_layer = true;

        Arachne::WallToolPaths wall_tool_paths(outline,
                                               preferred_spacing,
                                               preferred_spacing,
                                               1,
                                               0,
                                               params.flow.height(),
                                               input_params);
        std::vector<Arachne::VariableWidthLines> loops = wall_tool_paths.getToolPaths();
        inner_leftover = top_surface_clip_intersection_ex(top_surface_clip_union_ex(wall_tool_paths.getInnerContour()),
                                                          ExPolygons { surface.expolygon },
                                                          ApplySafetyOffset::Yes);
        ThickPolylines thick_polylines;
        Point last_pos(0, 0);
        for (Arachne::VariableWidthLines &loop : loops) {
            check_canceled(throw_if_canceled);
            for (const Arachne::ExtrusionLine &wall : loop) {
                if (wall.size() < 2)
                    continue;
                ThickPolyline thick_polyline = Arachne::to_thick_polyline(wall);
                if (thick_polyline.points.size() < 2 || thick_polyline.width.empty())
                    continue;
                if (wall.is_closed &&
                    thick_polyline.points.front() == thick_polyline.points.back() &&
                    thick_polyline.width.front() == thick_polyline.width.back()) {
                    thick_polyline.points.pop_back();
                    if (!thick_polyline.points.empty()) {
                        const int nearest_idx = last_pos.nearest_point_index(thick_polyline.points);
                        std::rotate(thick_polyline.points.begin(), thick_polyline.points.begin() + nearest_idx, thick_polyline.points.end());
                        std::rotate(thick_polyline.width.begin(), thick_polyline.width.begin() + 2 * nearest_idx, thick_polyline.width.end());
                        thick_polyline.points.emplace_back(thick_polyline.points.front());
                    }
                }
                for (coordf_t &width : thick_polyline.width)
                    width = variable ? std::clamp(width, min_spacing, max_spacing) : output_spacing;
                if (thick_polyline.is_valid()) {
                    last_pos = thick_polyline.last_point();
                    thick_polylines.emplace_back(std::move(thick_polyline));
                }
            }
        }
        const float min_length_mm =
            std::max(0.18f, std::min(0.75f, (variable ? min_width : output_flow.width()) * 0.55f));
        if (variable || !hybrid) {
            variable_width(thick_polylines,
                           params.extrusion_role,
                           variable ? max_flow : output_flow,
                           collection->entities);
            top_surface_image_filter_short_boundary_skin_entities(*collection, min_length_mm);
        } else {
            for (const ThickPolyline &thick_polyline : thick_polylines) {
                check_canceled(throw_if_canceled);
                Polyline polyline = top_surface_image_boundary_skin_polyline_from_thick(thick_polyline);
                remove_same_neighbor(polyline);
                if (!polyline.is_valid() || unscale<double>(polyline.length()) < double(min_length_mm))
                    continue;
                ExtrusionPath *path = new ExtrusionPath(params.extrusion_role,
                                                        output_flow.mm3_per_mm(),
                                                        output_flow.width(),
                                                        output_flow.height());
                path->polyline = std::move(polyline);
                collection->entities.emplace_back(path);
            }
        }
    }

    if (leftover != nullptr)
        *leftover = collection->empty() ? ExPolygons { surface.expolygon } : std::move(inner_leftover);
    return collection;
}

static void apply_top_surface_image_collection_metadata(ExtrusionEntityCollection &collection,
                                                        const SurfaceFillParams &params,
                                                        const std::optional<TextureMappingOffsetContext> &context,
                                                        const ThrowIfCanceled *throw_if_canceled)
{
    check_canceled(throw_if_canceled);
    collection.texture_mapping_top_surface_image = true;
    collection.texture_mapping_top_surface_zone_id = params.texture_mapping_top_surface_zone_id;
    collection.texture_mapping_top_surface_desired_component_id = params.texture_mapping_top_surface_component_id;
    collection.texture_mapping_top_surface_stack_depth = params.texture_mapping_top_surface_stack_depth;
    collection.texture_mapping_top_surface_fixed_coloring = params.texture_mapping_top_surface_fixed_coloring;
    if (params.texture_mapping_top_surface_fixed_coloring &&
        params.texture_mapping_top_surface_component_id > 0)
        collection.texture_mapping_extruder_override = int(params.texture_mapping_top_surface_component_id - 1);
    if (!context || params.texture_mapping_top_surface_contoning)
        return;
    ExtrusionEntitiesPtr replacement;
    replacement.reserve(collection.entities.size());
    for (const ExtrusionEntity *entity : collection.entities) {
        check_canceled(throw_if_canceled);
        replacement.emplace_back(top_surface_image_modulated_entity(*entity,
                                                                    *context,
                                                                    params.texture_mapping_top_surface_min_width_mm,
                                                                    params.texture_mapping_top_surface_max_width_mm,
                                                                    params.flow.nozzle_diameter(),
                                                                    throw_if_canceled));
    }
    collection.clear();
    collection.entities = std::move(replacement);
}

static void apply_top_surface_image_entities_metadata(ExtrusionEntitiesPtr &entities,
                                                      const SurfaceFillParams &params,
                                                      const std::optional<TextureMappingOffsetContext> &context,
                                                      const ThrowIfCanceled *throw_if_canceled)
{
    for (ExtrusionEntity *entity : entities)
        if (ExtrusionEntityCollection *collection = dynamic_cast<ExtrusionEntityCollection *>(entity))
            apply_top_surface_image_collection_metadata(*collection, params, context, throw_if_canceled);
}

static ExtrusionPath *top_surface_image_last_extrusion_path(ExtrusionEntitiesPtr &entities)
{
    for (auto it = entities.rbegin(); it != entities.rend(); ++it) {
        if (ExtrusionPath *path = dynamic_cast<ExtrusionPath *>(*it))
            return path;
        if (ExtrusionMultiPath *multipath = dynamic_cast<ExtrusionMultiPath *>(*it))
            if (!multipath->paths.empty())
                return &multipath->paths.back();
        if (ExtrusionEntityCollection *collection = dynamic_cast<ExtrusionEntityCollection *>(*it)) {
            if (collection->entities.empty())
                continue;
            return top_surface_image_last_extrusion_path(collection->entities);
        }
        if (*it != nullptr)
            return nullptr;
    }
    return nullptr;
}

static double top_surface_image_boundary_skin_start_distance_sq(const Polyline &polyline, const Point &anchor)
{
    if (!polyline.is_valid())
        return std::numeric_limits<double>::max();
    if (polyline.is_closed()) {
        double best = std::numeric_limits<double>::max();
        const size_t point_count = polyline.points.size() > 1 ? polyline.points.size() - 1 : polyline.points.size();
        for (size_t i = 0; i < point_count; ++i)
            best = std::min(best, (polyline.points[i] - anchor).cast<double>().squaredNorm());
        return best;
    }
    return std::min((polyline.first_point() - anchor).cast<double>().squaredNorm(),
                    (polyline.last_point() - anchor).cast<double>().squaredNorm());
}

static void top_surface_image_boundary_skin_orient_near(Polyline &polyline, const Point &anchor)
{
    if (!polyline.is_valid())
        return;
    if (polyline.is_closed()) {
        Points points = polyline.points;
        points.pop_back();
        if (points.empty())
            return;
        const int nearest_idx = anchor.nearest_point_index(points);
        std::rotate(points.begin(), points.begin() + nearest_idx, points.end());
        points.emplace_back(points.front());
        polyline.points = std::move(points);
    } else if ((polyline.last_point() - anchor).cast<double>().squaredNorm() <
               (polyline.first_point() - anchor).cast<double>().squaredNorm()) {
        polyline.reverse();
    }
}

static std::unique_ptr<ExtrusionPath> top_surface_image_take_nearest_boundary_skin_path(
    ExtrusionEntityCollection &collection,
    const Point &anchor)
{
    size_t best_idx = collection.entities.size();
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < collection.entities.size(); ++i) {
        const ExtrusionPath *path = dynamic_cast<const ExtrusionPath *>(collection.entities[i]);
        if (path == nullptr || !path->polyline.is_valid())
            continue;
        const double dist = top_surface_image_boundary_skin_start_distance_sq(path->polyline, anchor);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    if (best_idx == collection.entities.size())
        return nullptr;
    ExtrusionPath *path = static_cast<ExtrusionPath *>(collection.entities[best_idx]);
    collection.entities.erase(collection.entities.begin() + best_idx);
    top_surface_image_boundary_skin_orient_near(path->polyline, anchor);
    return std::unique_ptr<ExtrusionPath>(path);
}

static bool top_surface_image_try_join_boundary_skin_hybrid(ExtrusionEntitiesPtr &interior_entities,
                                                            ExtrusionEntityCollection &boundary_collection,
                                                            const ExPolygon &area,
                                                            coord_t max_connector)
{
    ExtrusionPath *interior_path = top_surface_image_last_extrusion_path(interior_entities);
    if (interior_path == nullptr || !interior_path->polyline.is_valid())
        return false;
    std::unique_ptr<ExtrusionPath> boundary_path =
        top_surface_image_take_nearest_boundary_skin_path(boundary_collection, interior_path->last_point());
    if (!boundary_path)
        return false;
    if (std::abs(interior_path->width - boundary_path->width) > EPSILON ||
        std::abs(interior_path->height - boundary_path->height) > EPSILON ||
        std::abs(interior_path->mm3_per_mm - boundary_path->mm3_per_mm) > EPSILON ||
        !top_surface_image_contoning_connector_printable(area,
                                                         interior_path->last_point(),
                                                         boundary_path->first_point(),
                                                         max_connector)) {
        boundary_collection.entities.insert(boundary_collection.entities.begin(), boundary_path.release());
        return false;
    }
    interior_path->polyline.append(std::move(boundary_path->polyline));
    return true;
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

enum class GroupFillsPurpose
{
    InfillToolpath,
    SparseInfillAnchoring,
};

struct GroupFillsOptions
{
    GroupFillsPurpose purpose { GroupFillsPurpose::InfillToolpath };
    const ThrowIfCanceled *throw_if_canceled { nullptr };
    bool apply_contoning_perimeter_options { false };
    TopSurfaceImageContoningStackPlanCache *contoning_stack_plan_cache { nullptr };
};

std::vector<SurfaceFill> group_fills(const Layer &layer,
                                     LockRegionParam &lock_param,
                                     const GroupFillsOptions &options)
{
    const ThrowIfCanceled *throw_if_canceled = options.throw_if_canceled;
    check_canceled(throw_if_canceled);
	std::vector<SurfaceFill> surface_fills;
	// Fill in a map of a region & surface to SurfaceFillParams.
	std::set<SurfaceFillParams> 						set_surface_params;
	std::vector<std::vector<const SurfaceFillParams*>> 	region_to_surface_params(layer.regions().size(), std::vector<const SurfaceFillParams*>());
    SurfaceFillParams									params;
    bool 												has_internal_voids = false;
	const PrintObjectConfig&							object_config = layer.object()->config();
    const bool                                          build_top_surface_plans =
        options.purpose == GroupFillsPurpose::InfillToolpath;
    const std::vector<TopSurfaceImageRegionPlan>        top_surface_plans =
        build_top_surface_plans ?
            top_surface_image_region_plans(layer, options.contoning_stack_plan_cache, throw_if_canceled) :
            std::vector<TopSurfaceImageRegionPlan>();
    std::vector<ExPolygons>                             top_surface_replacement_reservations(top_surface_plans.size());
    for (size_t region_id = 0; region_id < top_surface_plans.size(); ++region_id) {
        check_canceled(throw_if_canceled);
        top_surface_replacement_reservations[region_id] =
            top_surface_image_contoning_replacement_reservation_area(top_surface_plans[region_id]);
    }

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
        check_canceled(throw_if_canceled);
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
        check_canceled(throw_if_canceled);
		const_cast<SurfaceFillParams&>(params).idx = surface_fills.size();
		surface_fills.emplace_back(params);
	}

	for (size_t region_id = 0; region_id < layer.regions().size(); ++ region_id) {
        check_canceled(throw_if_canceled);
		const LayerRegion &layerm = *layer.regions()[region_id];
	    for (const Surface &surface : layerm.fill_surfaces.surfaces)
	        if (surface.surface_type != stInternalVoid) {
	        	const SurfaceFillParams *params = region_to_surface_params[region_id][&surface - &layerm.fill_surfaces.surfaces.front()];
				if (params != nullptr) {
                    ExPolygons remaining = { surface.expolygon };
                    if (region_id < top_surface_plans.size() &&
                        top_surface_plans[region_id].zone != nullptr &&
                        (surface.is_top() || surface.surface_type == stInternalSolid || surface.surface_type == stBottom) &&
                        !surface.is_bridge()) {
                        const TopSurfaceImageRegionPlan &plan = top_surface_plans[region_id];
                        for (size_t slice_idx = 0; slice_idx < plan.slices.size();) {
                            check_canceled(throw_if_canceled);
                            const TopSurfaceImageStackSlice &slice = plan.slices[slice_idx];
                            if (!top_surface_image_slice_matches_surface(slice, surface)) {
                                ++slice_idx;
                                continue;
                            }
                            if (remaining.empty())
                                break;
                            ExPolygons image_expolygons = top_surface_clip_intersection_ex(remaining, slice.area, ApplySafetyOffset::Yes);
                            if (image_expolygons.empty() && !slice.contoning) {
                                ++slice_idx;
                                continue;
                            }
                            ExPolygons image_clip = image_expolygons;
                            if (slice.same_layer_partition || slice.contoning) {
                                size_t same_depth_end = slice_idx;
                                while (same_depth_end < plan.slices.size() &&
                                       plan.slices[same_depth_end].same_layer_partition == slice.same_layer_partition &&
                                       plan.slices[same_depth_end].contoning == slice.contoning &&
                                       plan.slices[same_depth_end].lower_surface == slice.lower_surface &&
                                       plan.slices[same_depth_end].depth == slice.depth)
                                    ++same_depth_end;
                                ExPolygons depth_clip;
                                for (size_t same_idx = slice_idx; same_idx < same_depth_end; ++same_idx) {
                                    check_canceled(throw_if_canceled);
                                    SurfaceFillParams image_params =
                                        top_surface_image_params_for_slice(layer, surface, *params, plan, plan.slices[same_idx]);
                                    ExPolygons component_expolygons = slice.same_layer_partition ?
                                        image_expolygons :
                                        top_surface_clip_intersection_ex(remaining, plan.slices[same_idx].area, ApplySafetyOffset::Yes);
                                    if (!component_expolygons.empty()) {
                                        append(depth_clip, component_expolygons);
                                        if (plan.slices[same_idx].contoning &&
                                            plan.contoning_only_one_perimeter_around_shell_infill &&
                                            !layerm.fill_no_overlap_expolygons.empty())
                                            component_expolygons = top_surface_clip_intersection_ex(component_expolygons,
                                                                                                    layerm.fill_no_overlap_expolygons,
                                                                                                    ApplySafetyOffset::Yes);
                                        if (!component_expolygons.empty()) {
                                            SurfaceFill &image_fill = surface_fill_for_params(surface_fills, image_params);
                                            append_surface_fill_expolygons(image_fill, region_id, surface, std::move(component_expolygons), layerm);
                                        }
                                    }
                                }
                                if (!depth_clip.empty())
                                    image_clip = top_surface_clip_union_ex(depth_clip);
                                slice_idx = same_depth_end;
                            } else {
                                SurfaceFillParams image_params =
                                    top_surface_image_params_for_slice(layer, surface, *params, plan, slice);
                                SurfaceFill &image_fill = surface_fill_for_params(surface_fills, image_params);
                                append_surface_fill_expolygons(image_fill, region_id, surface, std::move(image_expolygons), layerm);
                                ++slice_idx;
                            }
                            remaining = top_surface_clip_diff_ex(remaining, image_clip, ApplySafetyOffset::Yes);
                        }
                    }
                    if (!remaining.empty()) {
                        if (region_id < top_surface_replacement_reservations.size() &&
                            !top_surface_replacement_reservations[region_id].empty() &&
                            (surface.is_top() || surface.surface_type == stInternalSolid || surface.surface_type == stBottom) &&
                            !surface.is_bridge()) {
                            remaining = top_surface_clip_diff_ex(remaining,
                                                                  top_surface_replacement_reservations[region_id],
                                                                  ApplySafetyOffset::Yes);
                        }
                    }
                    if (!remaining.empty()) {
                        SurfaceFill &fill = surface_fills[params->idx];
                        append_surface_fill_expolygons(fill, region_id, surface, std::move(remaining), layerm);
                    }
				}
	        }
        if (region_id < top_surface_plans.size() &&
            top_surface_plans[region_id].zone != nullptr)
            top_surface_image_append_contoning_replacement_surfaces(surface_fills,
                                                                    layerm,
                                                                    region_id,
                                                                    top_surface_plans[region_id],
                                                                    throw_if_canceled);
	}

	{
		Polygons all_polygons;
		for (SurfaceFill &fill : surface_fills)
			if (! fill.expolygons.empty()) {
                check_canceled(throw_if_canceled);
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
        check_canceled(throw_if_canceled);
    	// Internal voids are generated only if "infill_only_where_needed" or "infill_every_layers" are active.
        coord_t  distance_between_surfaces = 0;
        Polygons surfaces_polygons;
        Polygons voids;
		int      region_internal_infill = -1;
		int		 region_solid_infill = -1;
		int		 region_some_infill = -1;
    	for (SurfaceFill &surface_fill : surface_fills)
			if (! surface_fill.expolygons.empty()) {
                check_canceled(throw_if_canceled);
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
            check_canceled(throw_if_canceled);
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
                surface_fills.back().region_id = region_id;
				surface_fills.back().surface.surface_type = stInternalSolid;
				surface_fills.back().surface.thickness = layer.height;
                surface_fills.back().region_id_group.push_back(region_id);
                surface_fills.back().no_overlap_expolygons = layerm.fill_no_overlap_expolygons;
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
            check_canceled(throw_if_canceled);
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

    top_surface_image_debug_write_layer_svg(layer,
                                            top_surface_plans,
                                            surface_fills,
                                            options.apply_contoning_perimeter_options ? "before_perimeter_trim" : "final",
                                            throw_if_canceled);

    if (options.apply_contoning_perimeter_options) {
        top_surface_image_apply_contoning_perimeter_options(layer, surface_fills, top_surface_plans, throw_if_canceled);
        top_surface_image_debug_write_layer_svg(layer,
                                                top_surface_plans,
                                                surface_fills,
                                                "after_perimeter_trim",
                                                throw_if_canceled);
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
void Layer::make_fills(FillAdaptive::Octree* adaptive_fill_octree,
                       FillAdaptive::Octree* support_fill_octree,
                       FillLightning::Generator* lightning_generator,
                       std::function<void()> throw_if_canceled,
                       TopSurfaceImageContoningStackPlanCache *contoning_stack_plan_cache)
{
    const ThrowIfCanceled *throw_if_canceled_ptr = throw_if_canceled ? &throw_if_canceled : nullptr;
    check_canceled(throw_if_canceled_ptr);
	for (LayerRegion *layerm : m_regions)
		layerm->fills.clear();


#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
//	this->export_region_fill_surfaces_to_svg_debug("10_fill-initial");
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */
    LockRegionParam lock_param;
    GroupFillsOptions group_fills_options;
    group_fills_options.purpose = GroupFillsPurpose::InfillToolpath;
    group_fills_options.throw_if_canceled = throw_if_canceled_ptr;
    group_fills_options.apply_contoning_perimeter_options = true;
    group_fills_options.contoning_stack_plan_cache = contoning_stack_plan_cache;
    std::vector<SurfaceFill>     surface_fills =
        group_fills(*this, lock_param, group_fills_options);
	const Slic3r::BoundingBox bbox 			= this->object()->bounding_box();
	const auto                resolution 	= this->object()->print()->config().resolution.value;
    check_canceled(throw_if_canceled_ptr);

#ifdef SLIC3R_DEBUG_SLICE_PROCESSING
	{
		static int iRun = 0;
		export_group_fills_to_svg(debug_out_path("Layer-fill_surfaces-10_fill-final-%d.svg", iRun ++).c_str(), surface_fills);
	}
#endif /* SLIC3R_DEBUG_SLICE_PROCESSING */

    const std::map<TopSurfaceImageRectilinearBoundaryKey, TopSurfaceImageRectilinearBoundaryGroup> rectilinear_boundary_groups =
        top_surface_image_rectilinear_boundary_groups(*this, surface_fills, throw_if_canceled_ptr);
    top_surface_image_append_rectilinear_boundary_collections(*this, rectilinear_boundary_groups, throw_if_canceled_ptr);

    for (SurfaceFill &surface_fill : surface_fills) {
        check_canceled(throw_if_canceled_ptr);
        if (surface_fill.expolygons.empty() ||
            surface_fill.region_id >= this->m_regions.size() ||
            this->m_regions[surface_fill.region_id] == nullptr)
            continue;
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
        if (throw_if_canceled_ptr != nullptr) {
            params.throw_if_canceled = [](void *context) {
                (*static_cast<const ThrowIfCanceled *>(context))();
            };
            params.throw_if_canceled_context = const_cast<ThrowIfCanceled *>(throw_if_canceled_ptr);
        }
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
        params.no_edge_overlap = surface_fill.params.texture_mapping_top_surface_contoning_no_edge_overlap;
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
        if (surface_fill.params.texture_mapping_top_surface_image &&
            !surface_fill.params.texture_mapping_top_surface_contoning) {
            check_canceled(throw_if_canceled_ptr);
            const Print *print = this->object()->print();
            const TextureMappingZone *zone = print != nullptr ?
                print->texture_mapping_manager().zone_from_id(surface_fill.params.texture_mapping_top_surface_zone_id) :
                nullptr;
            if (print != nullptr && zone != nullptr) {
                const PrintConfig &print_config = print->config();
                std::vector<std::string> filament_colours = print_config.filament_colour.values;
                filament_colours.resize(print_config.filament_colour.values.size(), "#FFFFFF");
                std::vector<unsigned int> components =
                    zone->is_image_texture() ?
                        TextureMappingManager::effective_texture_component_ids(*zone,
                                                                               print_config.filament_colour.values.size(),
                                                                               filament_colours) :
                        TextureMappingManager::selected_component_ids(*zone, print_config.filament_colour.values.size());
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
            check_canceled(throw_if_canceled_ptr);

      f->no_overlap_expolygons = intersection_ex(surface_fill.no_overlap_expolygons, ExPolygons() = {expoly}, ApplySafetyOffset::Yes);
            check_canceled(throw_if_canceled_ptr);
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
                std::unique_ptr<ExtrusionEntityCollection> collection(new ExtrusionEntityCollection());
                top_surface_image_same_layer_partition_fill(*collection,
                                                            surface_fill.surface,
                                                            surface_fill.params,
                                                            top_surface_image_context,
                                                            throw_if_canceled_ptr);
                if (!collection->empty()) {
                    apply_top_surface_image_collection_metadata(*collection, surface_fill.params, std::nullopt, throw_if_canceled_ptr);
                    fill_entities.push_back(collection.release());
                }
            } else if (surface_fill.params.texture_mapping_top_surface_contoning &&
                       top_surface_image_contoning_spiral_mode(
                           surface_fill.params.texture_mapping_top_surface_contoning_flat_surface_infill_mode)) {
                ExPolygons leftover;
                std::unique_ptr<ExtrusionEntityCollection> collection =
                    top_surface_image_spiral_collection(surface_fill.surface,
                                                        surface_fill.params,
                                                        &leftover,
                                                        throw_if_canceled_ptr);
                if (collection && !collection->empty()) {
                    apply_top_surface_image_collection_metadata(*collection, surface_fill.params, std::nullopt, throw_if_canceled_ptr);
                    fill_entities.push_back(collection.release());
                }
                if (!leftover.empty()) {
                    SurfaceFillParams fallback_surface_params = surface_fill.params;
                    fallback_surface_params.pattern = ipRectilinear;
                    fallback_surface_params.texture_mapping_top_surface_contoning_flat_surface_infill_mode =
                        int(TextureMappingZone::ContoningFlatSurfaceInfillRectilinear);
                    FillParams fallback_params = params;
                    fallback_params.pattern = ipRectilinear;
                    for (ExPolygon &leftover_expoly : leftover) {
                        check_canceled(throw_if_canceled_ptr);
                        Surface leftover_surface = surface_fill.surface;
                        leftover_surface.expolygon = std::move(leftover_expoly);
                        f->spacing = fallback_surface_params.spacing;
                        const size_t fill_entities_before = fill_entities.size();
                        f->fill_surface_extrusion(&leftover_surface, fallback_params, fill_entities);
                        for (size_t i = fill_entities_before; i < fill_entities.size(); ++i)
                            if (ExtrusionEntityCollection *fallback_collection = dynamic_cast<ExtrusionEntityCollection *>(fill_entities[i]))
                                apply_top_surface_image_collection_metadata(*fallback_collection,
                                                                            fallback_surface_params,
                                                                            std::nullopt,
                                                                            throw_if_canceled_ptr);
                    }
                }
            } else if (surface_fill.params.texture_mapping_top_surface_contoning &&
                       top_surface_image_contoning_boundary_skin_mode(
                           surface_fill.params.texture_mapping_top_surface_contoning_flat_surface_infill_mode)) {
                const bool variable_width_boundary_skin =
                    surface_fill.params.texture_mapping_top_surface_contoning_flat_surface_infill_mode ==
                    int(TextureMappingZone::ContoningFlatSurfaceInfillBoundarySkinVariable);
                const bool hybrid_boundary_skin =
                    surface_fill.params.texture_mapping_top_surface_contoning_flat_surface_infill_mode ==
                    int(TextureMappingZone::ContoningFlatSurfaceInfillBoundarySkinHybrid);
                ExPolygons leftover;
                std::unique_ptr<ExtrusionEntityCollection> collection =
                    top_surface_image_boundary_skin_collection(surface_fill.surface,
                                                               surface_fill.params,
                                                               this->object()->print()->config(),
                                                               this->object()->config(),
                                                               int(this->id()),
                                                               variable_width_boundary_skin,
                                                               hybrid_boundary_skin,
                                                               &leftover,
                                                               throw_if_canceled_ptr);
                SurfaceFillParams fallback_surface_params = surface_fill.params;
                fallback_surface_params.pattern = ipRectilinear;
                fallback_surface_params.texture_mapping_top_surface_contoning_flat_surface_infill_mode =
                    int(TextureMappingZone::ContoningFlatSurfaceInfillRectilinear);
                const Flow fallback_flow = top_surface_image_contoning_boundary_skin_flow(surface_fill.params,
                                                                                          variable_width_boundary_skin);
                fallback_surface_params.flow = fallback_flow;
                fallback_surface_params.spacing = fallback_flow.spacing();
                FillParams fallback_params = params;
                fallback_params.flow = fallback_flow;
                fallback_params.pattern = ipRectilinear;
                ExtrusionEntitiesPtr hybrid_interior_entities;
                if (!hybrid_boundary_skin && collection && !collection->empty()) {
                    apply_top_surface_image_collection_metadata(*collection, surface_fill.params, std::nullopt, throw_if_canceled_ptr);
                    fill_entities.push_back(collection.release());
                }
                if (!leftover.empty()) {
                    for (ExPolygon &leftover_expoly : leftover) {
                        check_canceled(throw_if_canceled_ptr);
                        Surface leftover_surface = surface_fill.surface;
                        leftover_surface.expolygon = std::move(leftover_expoly);
                        f->spacing = fallback_surface_params.spacing;
                        ExtrusionEntitiesPtr *fallback_entities = hybrid_boundary_skin ? &hybrid_interior_entities : &fill_entities;
                        const size_t fill_entities_before = fallback_entities->size();
                        f->fill_surface_extrusion(&leftover_surface, fallback_params, *fallback_entities);
                        if (!hybrid_boundary_skin) {
                            for (size_t i = fill_entities_before; i < fallback_entities->size(); ++i)
                                if (ExtrusionEntityCollection *fallback_collection = dynamic_cast<ExtrusionEntityCollection *>((*fallback_entities)[i]))
                                    apply_top_surface_image_collection_metadata(*fallback_collection,
                                                                                fallback_surface_params,
                                                                                std::nullopt,
                                                                                throw_if_canceled_ptr);
                        }
                    }
                }
                if (hybrid_boundary_skin) {
                    if (collection && !collection->empty() && !hybrid_interior_entities.empty()) {
                        const coord_t max_connector = std::max<coord_t>(1, 2 * fallback_flow.scaled_spacing());
                        top_surface_image_try_join_boundary_skin_hybrid(hybrid_interior_entities,
                                                                        *collection,
                                                                        surface_fill.surface.expolygon,
                                                                        max_connector);
                    }
                    apply_top_surface_image_entities_metadata(hybrid_interior_entities,
                                                              fallback_surface_params,
                                                              std::nullopt,
                                                              throw_if_canceled_ptr);
                    fill_entities.insert(fill_entities.end(), hybrid_interior_entities.begin(), hybrid_interior_entities.end());
                    hybrid_interior_entities.clear();
                }
                if (hybrid_boundary_skin && collection && !collection->empty()) {
                    apply_top_surface_image_collection_metadata(*collection, surface_fill.params, std::nullopt, throw_if_canceled_ptr);
                    fill_entities.push_back(collection.release());
                }
            } else {
                const size_t fill_entities_before = fill_entities.size();
			    f->fill_surface_extrusion(&surface_fill.surface,
				    params,
				    fill_entities);
                if (surface_fill.params.texture_mapping_top_surface_image) {
                    for (size_t i = fill_entities_before; i < fill_entities.size(); ++i)
                        if (ExtrusionEntityCollection *collection = dynamic_cast<ExtrusionEntityCollection *>(fill_entities[i]))
                            apply_top_surface_image_collection_metadata(*collection, surface_fill.params, top_surface_image_context, throw_if_canceled_ptr);
                }
            }
			}
	    }

    top_surface_image_append_rectilinear_repair_collections(*this, rectilinear_boundary_groups, throw_if_canceled_ptr);

    // add thin fill regions
    // Unpacks the collection, creates multiple collections per path.
    // The path type could be ExtrusionPath, ExtrusionLoop or ExtrusionEntityCollection.
    // Why the paths are unpacked?
	for (LayerRegion *layerm : m_regions)
	    for (const ExtrusionEntity *thin_fill : layerm->thin_fills.entities) {
            check_canceled(throw_if_canceled_ptr);
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
    GroupFillsOptions group_fills_options;
    group_fills_options.purpose = GroupFillsPurpose::SparseInfillAnchoring;
    std::vector<SurfaceFill> surface_fills = group_fills(*this, skin_inner_param, group_fills_options);
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
