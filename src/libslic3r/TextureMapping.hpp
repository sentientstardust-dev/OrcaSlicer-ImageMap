// original author: sentientstardust

#ifndef slic3r_TextureMapping_hpp_
#define slic3r_TextureMapping_hpp_

#ifndef SLIC3R_SHOW_EXPERIMENTAL_CONTONING_OPTIONS
#define SLIC3R_SHOW_EXPERIMENTAL_CONTONING_OPTIONS 0
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Slic3r {

struct TextureMappingZone
{
    static constexpr float DefaultPreviewOpacityPct = 100.f;

    enum SurfacePattern : uint8_t {
        ImageTexture = 0,
        Gradient2D   = 1,
        LinearGradient = 2
    };

    enum LinearGradientMode : uint8_t {
        LinearGradientLinear = 0,
        LinearGradientRadial = 1
    };

    enum OffsetControlMode : uint8_t {
        OffsetBasic = 0,
        OffsetAdvanced = 1
    };

    enum OffsetFadeMode : uint8_t {
        OffsetFadeNone = 0,
        OffsetFadeInUp,
        OffsetFadeOutUp,
        OffsetFadeInOut,
        OffsetFadeOutIn,
        OffsetFadeOutInReversed
    };

    enum OffsetAngleMode : uint8_t {
        OffsetAngleConfigured = 0,
        OffsetAngleSurfaceNormal = 1,
        OffsetAngleObjectCenter = 2
    };

    enum TextureMappingMode : uint8_t {
        TextureMappingFilamentBlending = 0,
        TextureMappingRawValues = 1
    };

    enum ModulationMode : uint8_t {
        ModulationLineWidth = 0,
        ModulationPerimeterPath = 1,
        ModulationPerimeterPathV2 = 2
    };

    enum TopVisiblePerimeterRecolorAggressiveness : uint8_t {
        TopVisibleRecolorConservative = 0,
        TopVisibleRecolorBalanced = 1,
        TopVisibleRecolorAggressive = 2
    };

    enum TopSurfaceImagePrintingMethod : uint8_t {
        TopSurfaceImageSameAngle45Width = 0,
        TopSurfaceImageSameLayer45Partition = 1,
        TopSurfaceImageContoning = 2
    };

    enum TopSurfaceContoningPerimeterMode : uint8_t {
        ContoningPerimeterSegmentBlocks = 0,
        ContoningPerimeterDividedLine = 1,
        ContoningPerimeterSegmentInfill = 2
    };

    enum TopSurfaceContoningFlatSurfaceInfillMode : uint8_t {
        ContoningFlatSurfaceInfillDefault = 0,
        ContoningFlatSurfaceInfillRectilinear = 1,
        ContoningFlatSurfaceInfillConcentric = 2,
        ContoningFlatSurfaceInfillBoundarySkinFixed = 3,
        ContoningFlatSurfaceInfillBoundarySkinVariable = 4,
        ContoningFlatSurfaceInfillSpiral = 5,
        ContoningFlatSurfaceInfillBoundarySkinHybrid = 6,
        ContoningFlatSurfaceInfillRectilinearWithBoundary = 7
    };

    enum FilamentColorMode : uint8_t {
        FilamentColorAny = 0,
        FilamentColorRGB = 1,
        FilamentColorCMY = 2,
        FilamentColorCMYK = 3,
        FilamentColorCMYW = 4,
        FilamentColorRGBK = 5,
        FilamentColorRGBW = 6,
        FilamentColorBW = 7,
        FilamentColorCMYKW = 8,
        FilamentColorRGBKW = 9
    };

    enum GenericSolverLookupMode : uint8_t {
        GenericSolverClosestMix = 0,
        GenericSolverBlendClosestTwo = 1
    };

    enum GenericSolverMode : uint8_t {
        GenericSolverLegacy = 0,
        GenericSolverV2 = 1
    };

    enum GenericSolverMixModel : uint8_t {
        GenericSolverPigmentPainter = 0
    };

    enum DitheringMethod : uint8_t {
        DitheringClosest = 0,
        DitheringFloydSteinberg = 1,
        DitheringOrderedBayer = 2,
        DitheringHalftone = 3,
        DitheringHalftoneIncreasedDetail = 4,
        DitheringHalftoneV2 = 5
    };

    enum TransmissionDistanceCalibrationMode : uint8_t {
        TDCalibrationNone = 0,
        TDCalibrationAbsolute = 1,
        TDCalibrationNeighbor = 2
    };

    static constexpr int   DefaultSurfacePattern = int(ImageTexture);
    static constexpr int   DefaultLinearGradientMode = int(LinearGradientLinear);
    static constexpr float DefaultLinearGradientRadiusMm = 0.f;
    static constexpr bool  DefaultLinearGradientRadiusPercent = true;
    static constexpr float DefaultLinearGradientRadiusPct = 100.f;
    static constexpr int   DefaultOffsetMode = int(OffsetBasic);
    static constexpr bool  DefaultOffsetRotationEnabled = true;
    static constexpr float DefaultOffsetRotations = 1.f;
    static constexpr float DefaultOffsetRepeats = 1.f;
    static constexpr bool  DefaultOffsetReverseRepeats = true;
    static constexpr bool  DefaultOffsetClockwise = true;
    static constexpr int   DefaultOffsetFadeMode = int(OffsetFadeNone);
    static constexpr int   DefaultOffsetAngleMode = int(OffsetAngleObjectCenter);
    static constexpr int   DefaultTextureMappingMode = int(TextureMappingFilamentBlending);
    static constexpr int   DefaultFilamentColorMode = int(FilamentColorAny);
    static constexpr bool  DefaultForceSequentialFilaments = false;
    static constexpr bool  DefaultReduceOuterSurfaceTexture = false;
    static constexpr bool  DefaultSeamHiding = false;
    static constexpr bool  DefaultNonlinearOffsetAdjustment = false;
    static constexpr int   DefaultImageTextureModulationMode = int(ModulationLineWidth);
    static constexpr int   Default2DGradientModulationMode = int(ModulationPerimeterPath);
    static constexpr int   DefaultLinearGradientModulationMode = int(ModulationPerimeterPath);
    static constexpr int   DefaultModulationMode = DefaultImageTextureModulationMode;
    static constexpr bool  DefaultModulationModeManuallyChanged = false;
    static constexpr bool  DefaultUseModulatedOverhangGeometryForSupport = false;
    static constexpr bool  DefaultRecolorSmallPerimeterLoops = false;
    static constexpr bool  DefaultRecolorTopVisiblePerimeterSections = false;
    static constexpr int   DefaultTopVisiblePerimeterRecolorAggressiveness = int(TopVisibleRecolorAggressive);
    static constexpr int   MinTopVisiblePerimeterRecolorAboveLayers = 1;
    static constexpr int   MaxTopVisiblePerimeterRecolorAboveLayers = 5;
    static constexpr int   DefaultTopVisiblePerimeterRecolorAboveLayers = 2;
    static constexpr bool  DefaultTopVisiblePerimeterRecolorPointSampling = true;
    static constexpr bool  DefaultTopSurfaceImagePrintingEnabled = false;
    static constexpr int   DefaultTopSurfaceImagePrintingMethod = int(TopSurfaceImageContoning);
    static constexpr float MinTopSurfaceImageLineWidthMm = 0.32f;
    static constexpr float MaxTopSurfaceImageLineWidthMm = 0.80f;
    static constexpr float DefaultTopSurfaceImageMinLineWidthMm = 0.32f;
    static constexpr float DefaultTopSurfaceImageMaxLineWidthMm = 0.80f;
    static constexpr int   MinTopSurfaceImageColoredTopLayers = 1;
    static constexpr int   MaxTopSurfaceImageColoredTopLayers = 20;
    static constexpr int   DefaultTopSurfaceImageColoredTopLayers = 3;
    static constexpr bool  DefaultTopSurfaceImageFixedColoringFilaments = true;
    static constexpr float MinTopSurfaceContoningAngleThresholdDeg = 0.f;
    static constexpr float MaxTopSurfaceContoningAngleThresholdDeg = 180.f;
    static constexpr float DefaultTopSurfaceContoningAngleThresholdDeg = 20.f;
    static constexpr int   MinTopSurfaceContoningStackLayers = 1;
    static constexpr int   MaxTopSurfaceContoningStackLayers = 20;
    static constexpr int   DefaultTopSurfaceContoningStackLayers = 15;
    static constexpr int   MinTopSurfaceContoningPatternFilaments = 1;
    static constexpr int   MaxTopSurfaceContoningPatternFilaments = 20;
    static constexpr int   DefaultTopSurfaceContoningPatternFilaments = 4;
    static constexpr float MinTopSurfaceContoningMinFeatureMm = 0.f;
    static constexpr float MaxTopSurfaceContoningMinFeatureMm = 20.f;
    static constexpr float DefaultTopSurfaceContoningMinFeatureMm = 0.f;
    static constexpr bool  DefaultTopSurfaceContoningColorLowerSurfaces = true;
    static constexpr bool  DefaultTopSurfaceContoningOnlyColorSurfaceInfill = true;
    static constexpr bool  DefaultTopSurfaceContoningOnlyOnePerimeterAroundShellInfill = false;
    static constexpr bool  DefaultTopSurfaceContoningReplaceTopPerimetersWithInfill = false;
    static constexpr bool  DefaultTopSurfaceContoningRecolorSurroundingPerimeters = false;
    static constexpr int   DefaultTopSurfaceContoningPerimeterMode = int(ContoningPerimeterDividedLine);
    static constexpr int   DefaultTopSurfaceContoningFlatSurfaceInfillMode = int(ContoningFlatSurfaceInfillDefault);
    static constexpr int   SlicerDefaultTopSurfaceContoningFlatSurfaceInfillMode = int(ContoningFlatSurfaceInfillRectilinear);
    static constexpr bool  DefaultTopSurfaceContoningLayerPhaseEnabled = false;
    static constexpr bool  DefaultTopSurfaceContoningVariedInfillAnglesEnabled = false;
    static constexpr bool  DefaultTopSurfaceContoningBlueNoiseErrorDiffusionEnabled = false;
    static constexpr bool  DefaultTopSurfaceContoningSupersampledCellsEnabled = false;
    static constexpr bool  DefaultTopSurfaceContoningPolygonizeColorRegionsEnabled = false;
    static constexpr bool  DefaultTopSurfaceContoningSurfaceAnchoredStacksEnabled = false;
    static constexpr bool  DefaultCompactOffsetMode = true;
    static constexpr bool  DefaultUseLegacyFixedColorMode = false;
    static constexpr bool  DefaultHighSpeedImageTextureSampling = true;
    static constexpr bool  DefaultMinimumVisibilityOffsetEnabled = true;
    static constexpr float DefaultMinimumVisibilityOffsetPct = 30.f;
    static constexpr int   DefaultGenericSolverLookupMode = int(GenericSolverClosestMix);
    static constexpr int   DefaultGenericSolverMode = int(GenericSolverV2);
    static constexpr int   DefaultGenericSolverMixModel = int(GenericSolverPigmentPainter);
    static constexpr bool  DefaultDitheringEnabled = false;
    static constexpr int   DefaultDitheringMethod = int(DitheringHalftoneV2);
    static constexpr float MinDitheringResolutionMm = 0.04f;
    static constexpr float MaxDitheringResolutionMm = 10.f;
    static constexpr float DefaultDitheringResolutionMm = 0.4f;
    static constexpr float MinHalftoneDotSizeMm = 0.08f;
    static constexpr float MaxHalftoneDotSizeMm = 50.f;
    static constexpr float DefaultHalftoneDotSizeMm = 2.f;
    static constexpr float DefaultContrastPct = 100.f;
    static constexpr bool  DefaultHighResolutionSampling = true;
    static constexpr float DefaultToneGamma = 1.f;
    static constexpr int   DefaultTransmissionDistanceCalibrationMode = int(TDCalibrationAbsolute);
    static constexpr bool  DefaultPreviewSimulateColors = false;
    static constexpr bool  DefaultPreviewLimitResolution = true;
    static constexpr bool  DefaultPreviewSimulateTopSurfaceLod = true;
    static constexpr bool  DefaultAutoAdjustFilamentSelection = true;
    static constexpr bool  ShowExperimentalTopSurfaceContoningOptions = SLIC3R_SHOW_EXPERIMENTAL_CONTONING_OPTIONS != 0;

    static constexpr int default_modulation_mode_for_surface_pattern(int surface_pattern)
    {
        switch (surface_pattern) {
        case int(Gradient2D):      return Default2DGradientModulationMode;
        case int(LinearGradient):  return DefaultLinearGradientModulationMode;
        default:                   return DefaultImageTextureModulationMode;
        }
    }

    static constexpr int effective_top_surface_contoning_flat_surface_infill_mode(int mode)
    {
        if (!ShowExperimentalTopSurfaceContoningOptions)
            return SlicerDefaultTopSurfaceContoningFlatSurfaceInfillMode;
        return mode == int(ContoningFlatSurfaceInfillDefault) ?
            SlicerDefaultTopSurfaceContoningFlatSurfaceInfillMode :
            mode;
    }

    static constexpr int stored_top_surface_contoning_flat_surface_infill_mode(int mode)
    {
        return ShowExperimentalTopSurfaceContoningOptions ? mode : DefaultTopSurfaceContoningFlatSurfaceInfillMode;
    }

    static constexpr int stored_top_surface_contoning_perimeter_mode(int mode)
    {
        return ShowExperimentalTopSurfaceContoningOptions ? mode : DefaultTopSurfaceContoningPerimeterMode;
    }

    static constexpr bool effective_top_surface_contoning_only_color_surface_infill(bool value)
    {
        return ShowExperimentalTopSurfaceContoningOptions ? value : true;
    }

    static constexpr bool effective_top_surface_contoning_replace_top_perimeters_with_infill(bool value)
    {
        return ShowExperimentalTopSurfaceContoningOptions && value;
    }

    static constexpr bool effective_top_surface_contoning_recolor_surrounding_perimeters(bool value, bool replace)
    {
        return ShowExperimentalTopSurfaceContoningOptions && value && !replace;
    }

    static constexpr bool effective_top_surface_contoning_surface_anchored_stacks_enabled(bool value)
    {
        return ShowExperimentalTopSurfaceContoningOptions && value;
    }

    struct LinearGradientAnchor {
        bool valid = false;
        size_t object_id = 0;
        size_t instance_id = 0;
        int object_backup_id = -1;
        bool object_index_valid = false;
        size_t object_index = 0;
        bool instance_index_valid = false;
        size_t instance_index = 0;
        size_t instance_loaded_id = 0;
        std::array<float, 3> local_point { { 0.f, 0.f, 0.f } };
        std::array<float, 3> global_point { { 0.f, 0.f, 0.f } };
    };

    struct LinearGradientStop {
        float position = 0.f;
        unsigned int filament_id = 1;
    };

    uint64_t     stable_id = 0;
    unsigned int zone_id = 0;
    bool         enabled = true;
    bool         deleted = false;
    int          surface_pattern = DefaultSurfacePattern;
    unsigned int component_a = 1;
    unsigned int component_b = 2;
    std::string  component_ids;
    std::string  component_weights;
    std::string  display_color;

    std::string offset_distances;
    std::string offset_angles;
    int         offset_mode = DefaultOffsetMode;
    bool        offset_rotation_enabled = DefaultOffsetRotationEnabled;
    float       offset_rotations = DefaultOffsetRotations;
    float       offset_repeats = DefaultOffsetRepeats;
    bool        offset_reverse_repeats = DefaultOffsetReverseRepeats;
    bool        offset_clockwise = DefaultOffsetClockwise;
    int         offset_fade_mode = DefaultOffsetFadeMode;
    int         offset_angle_mode = DefaultOffsetAngleMode;

    int         texture_mapping_mode = DefaultTextureMappingMode;
    int         filament_color_mode = DefaultFilamentColorMode;
    bool        force_sequential_filaments = DefaultForceSequentialFilaments;
    bool        reduce_outer_surface_texture = DefaultReduceOuterSurfaceTexture;
    bool        seam_hiding = DefaultSeamHiding;
    bool        nonlinear_offset_adjustment = DefaultNonlinearOffsetAdjustment;
    int         modulation_mode = DefaultModulationMode;
    bool        use_modulated_overhang_geometry_for_support = DefaultUseModulatedOverhangGeometryForSupport;
    bool        modulation_mode_manually_changed = DefaultModulationModeManuallyChanged;
    bool        recolor_small_perimeter_loops = DefaultRecolorSmallPerimeterLoops;
    bool        recolor_top_visible_perimeter_sections = DefaultRecolorTopVisiblePerimeterSections;
    int         top_visible_perimeter_recolor_aggressiveness = DefaultTopVisiblePerimeterRecolorAggressiveness;
    int         top_visible_perimeter_recolor_above_layers = DefaultTopVisiblePerimeterRecolorAboveLayers;
    bool        top_visible_perimeter_recolor_point_sampling = DefaultTopVisiblePerimeterRecolorPointSampling;
    bool        top_surface_image_printing_enabled = DefaultTopSurfaceImagePrintingEnabled;
    int         top_surface_image_printing_method = DefaultTopSurfaceImagePrintingMethod;
    float       top_surface_image_min_line_width_mm = DefaultTopSurfaceImageMinLineWidthMm;
    float       top_surface_image_max_line_width_mm = DefaultTopSurfaceImageMaxLineWidthMm;
    int         top_surface_image_colored_top_layers = DefaultTopSurfaceImageColoredTopLayers;
    bool        top_surface_image_fixed_coloring_filaments = DefaultTopSurfaceImageFixedColoringFilaments;
    float       top_surface_contoning_angle_threshold_deg = DefaultTopSurfaceContoningAngleThresholdDeg;
    int         top_surface_contoning_stack_layers = DefaultTopSurfaceContoningStackLayers;
    int         top_surface_contoning_pattern_filaments = DefaultTopSurfaceContoningPatternFilaments;
    float       top_surface_contoning_min_feature_mm = DefaultTopSurfaceContoningMinFeatureMm;
    bool        top_surface_contoning_color_lower_surfaces = DefaultTopSurfaceContoningColorLowerSurfaces;
    bool        top_surface_contoning_only_color_surface_infill = DefaultTopSurfaceContoningOnlyColorSurfaceInfill;
    bool        top_surface_contoning_only_one_perimeter_around_shell_infill = DefaultTopSurfaceContoningOnlyOnePerimeterAroundShellInfill;
    bool        top_surface_contoning_replace_top_perimeters_with_infill = DefaultTopSurfaceContoningReplaceTopPerimetersWithInfill;
    bool        top_surface_contoning_recolor_surrounding_perimeters = DefaultTopSurfaceContoningRecolorSurroundingPerimeters;
    int         top_surface_contoning_perimeter_mode = DefaultTopSurfaceContoningPerimeterMode;
    int         top_surface_contoning_flat_surface_infill_mode = DefaultTopSurfaceContoningFlatSurfaceInfillMode;
    bool        top_surface_contoning_layer_phase_enabled = DefaultTopSurfaceContoningLayerPhaseEnabled;
    bool        top_surface_contoning_varied_infill_angles_enabled = DefaultTopSurfaceContoningVariedInfillAnglesEnabled;
    bool        top_surface_contoning_blue_noise_error_diffusion_enabled = DefaultTopSurfaceContoningBlueNoiseErrorDiffusionEnabled;
    bool        top_surface_contoning_supersampled_cells_enabled = DefaultTopSurfaceContoningSupersampledCellsEnabled;
    bool        top_surface_contoning_polygonize_color_regions_enabled = DefaultTopSurfaceContoningPolygonizeColorRegionsEnabled;
    bool        top_surface_contoning_surface_anchored_stacks_enabled = DefaultTopSurfaceContoningSurfaceAnchoredStacksEnabled;
    bool        compact_offset_mode = DefaultCompactOffsetMode;
    bool        use_legacy_fixed_color_mode = DefaultUseLegacyFixedColorMode;
    bool        high_speed_image_texture_sampling = DefaultHighSpeedImageTextureSampling;
    bool        minimum_visibility_offset_enabled = DefaultMinimumVisibilityOffsetEnabled;
    float       minimum_visibility_offset_pct = DefaultMinimumVisibilityOffsetPct;
    int         generic_solver_lookup_mode = DefaultGenericSolverLookupMode;
    int         generic_solver_mode = DefaultGenericSolverMode;
    int         generic_solver_mix_model = DefaultGenericSolverMixModel;
    bool        dithering_enabled = DefaultDitheringEnabled;
    int         dithering_method = DefaultDitheringMethod;
    float       dithering_resolution_mm = DefaultDitheringResolutionMm;
    float       halftone_dot_size_mm = DefaultHalftoneDotSizeMm;
    float       contrast_pct = DefaultContrastPct;
    bool        high_resolution_sampling = DefaultHighResolutionSampling;
    float       tone_gamma = DefaultToneGamma;
    int         transmission_distance_calibration_mode = DefaultTransmissionDistanceCalibrationMode;
    bool        auto_adjust_filament_selection = DefaultAutoAdjustFilamentSelection;
    std::vector<float> filament_strengths_pct;
    std::vector<float> filament_minimum_offsets_pct;
    std::vector<float> filament_transmission_distances_mm;
    LinearGradientAnchor linear_gradient_start;
    LinearGradientAnchor linear_gradient_end;
    int                  linear_gradient_mode = DefaultLinearGradientMode;
    float                linear_gradient_radius_mm = DefaultLinearGradientRadiusMm;
    bool                 linear_gradient_radius_percent = DefaultLinearGradientRadiusPercent;
    float                linear_gradient_radius_pct = DefaultLinearGradientRadiusPct;
    bool                 show_linear_gradient_direction_arrow = true;
    std::vector<LinearGradientStop> linear_gradient_stops;

    bool is_image_texture() const { return surface_pattern == int(ImageTexture); }
    bool is_2d_gradient() const { return surface_pattern == int(Gradient2D); }
    bool uses_perimeter_path_modulation() const
    {
        return modulation_mode == int(ModulationPerimeterPath) ||
               modulation_mode == int(ModulationPerimeterPathV2);
    }
    bool uses_legacy_perimeter_path_modulation() const { return modulation_mode == int(ModulationPerimeterPath); }
    bool uses_perimeter_path_modulation_v2() const { return modulation_mode == int(ModulationPerimeterPathV2); }
    bool is_linear_gradient() const { return surface_pattern == int(LinearGradient); }
    bool is_radial_linear_gradient() const { return is_linear_gradient() && linear_gradient_mode == int(LinearGradientRadial); }
    bool is_surface_gradient() const { return is_2d_gradient() || is_linear_gradient(); }
    bool top_surface_image_printing_active() const
    {
        return top_surface_image_printing_enabled &&
               (is_image_texture() || is_surface_gradient()) &&
               (top_surface_image_printing_method == int(TopSurfaceImageSameAngle45Width) ||
                top_surface_image_printing_method == int(TopSurfaceImageSameLayer45Partition) ||
                (top_surface_image_printing_method == int(TopSurfaceImageContoning) &&
                 uses_perimeter_path_modulation_v2()));
    }
    bool top_surface_contoning_active() const
    {
        return top_surface_image_printing_enabled &&
               (is_image_texture() || is_surface_gradient()) &&
               top_surface_image_printing_method == int(TopSurfaceImageContoning) &&
               uses_perimeter_path_modulation_v2();
    }

    bool top_surface_image_fixed_coloring_filaments_active() const
    {
        return top_surface_image_printing_method == int(TopSurfaceImageContoning) ||
               top_surface_image_fixed_coloring_filaments;
    }
    bool top_surface_contoning_perimeters_active() const
    {
        return is_image_texture() &&
               top_surface_contoning_active() &&
               !effective_top_surface_contoning_only_color_surface_infill() &&
               !effective_top_surface_contoning_replace_top_perimeters_with_infill() &&
               !effective_top_surface_contoning_recolor_surrounding_perimeters();
    }

    bool effective_top_surface_contoning_only_color_surface_infill() const
    {
        return effective_top_surface_contoning_only_color_surface_infill(top_surface_contoning_only_color_surface_infill);
    }

    bool effective_top_surface_contoning_replace_top_perimeters_with_infill() const
    {
        return effective_top_surface_contoning_replace_top_perimeters_with_infill(
            top_surface_contoning_replace_top_perimeters_with_infill);
    }

    bool effective_top_surface_contoning_recolor_surrounding_perimeters() const
    {
        return effective_top_surface_contoning_recolor_surrounding_perimeters(
            top_surface_contoning_recolor_surrounding_perimeters,
            effective_top_surface_contoning_replace_top_perimeters_with_infill());
    }

    int stored_top_surface_contoning_perimeter_mode() const
    {
        return stored_top_surface_contoning_perimeter_mode(top_surface_contoning_perimeter_mode);
    }

    int stored_top_surface_contoning_flat_surface_infill_mode() const
    {
        return stored_top_surface_contoning_flat_surface_infill_mode(top_surface_contoning_flat_surface_infill_mode);
    }

    int effective_top_surface_contoning_flat_surface_infill_mode() const
    {
        return effective_top_surface_contoning_flat_surface_infill_mode(stored_top_surface_contoning_flat_surface_infill_mode());
    }

    bool effective_top_surface_contoning_surface_anchored_stacks_enabled() const
    {
        return effective_top_surface_contoning_surface_anchored_stacks_enabled(
            top_surface_contoning_surface_anchored_stacks_enabled);
    }

    void apply_top_surface_contoning_experimental_defaults()
    {
        if (top_surface_image_printing_enabled &&
            top_surface_image_printing_method == int(TopSurfaceImageContoning)) {
            modulation_mode = int(ModulationPerimeterPathV2);
            modulation_mode_manually_changed = true;
            top_surface_image_fixed_coloring_filaments = true;
        }
        if (!ShowExperimentalTopSurfaceContoningOptions) {
            top_surface_contoning_only_color_surface_infill = true;
            top_surface_contoning_replace_top_perimeters_with_infill = false;
            top_surface_contoning_recolor_surrounding_perimeters = false;
            top_surface_contoning_perimeter_mode = DefaultTopSurfaceContoningPerimeterMode;
            top_surface_contoning_flat_surface_infill_mode = DefaultTopSurfaceContoningFlatSurfaceInfillMode;
            top_surface_contoning_surface_anchored_stacks_enabled = false;
        } else if (top_surface_contoning_replace_top_perimeters_with_infill) {
            top_surface_contoning_recolor_surrounding_perimeters = false;
        }
    }

    void apply_default_modulation_mode()
    {
        if (!modulation_mode_manually_changed)
            modulation_mode = default_modulation_mode_for_surface_pattern(surface_pattern);
    }

    void reset_offset_settings()
    {
        offset_distances.clear();
        offset_angles.clear();
        offset_mode = DefaultOffsetMode;
        offset_rotation_enabled = DefaultOffsetRotationEnabled;
        offset_rotations = DefaultOffsetRotations;
        offset_repeats = DefaultOffsetRepeats;
        offset_reverse_repeats = DefaultOffsetReverseRepeats;
        offset_clockwise = DefaultOffsetClockwise;
        offset_fade_mode = DefaultOffsetFadeMode;
        offset_angle_mode = DefaultOffsetAngleMode;
    }

    void reset_texture_options()
    {
        texture_mapping_mode = DefaultTextureMappingMode;
        filament_color_mode = DefaultFilamentColorMode;
        force_sequential_filaments = DefaultForceSequentialFilaments;
        reduce_outer_surface_texture = DefaultReduceOuterSurfaceTexture;
        seam_hiding = DefaultSeamHiding;
        nonlinear_offset_adjustment = DefaultNonlinearOffsetAdjustment;
        use_modulated_overhang_geometry_for_support = DefaultUseModulatedOverhangGeometryForSupport;
        modulation_mode = default_modulation_mode_for_surface_pattern(surface_pattern);
        modulation_mode_manually_changed = DefaultModulationModeManuallyChanged;
        recolor_small_perimeter_loops = DefaultRecolorSmallPerimeterLoops;
        recolor_top_visible_perimeter_sections = DefaultRecolorTopVisiblePerimeterSections;
        top_visible_perimeter_recolor_aggressiveness = DefaultTopVisiblePerimeterRecolorAggressiveness;
        top_visible_perimeter_recolor_above_layers = DefaultTopVisiblePerimeterRecolorAboveLayers;
        top_visible_perimeter_recolor_point_sampling = DefaultTopVisiblePerimeterRecolorPointSampling;
        top_surface_image_printing_enabled = DefaultTopSurfaceImagePrintingEnabled;
        top_surface_image_printing_method = DefaultTopSurfaceImagePrintingMethod;
        top_surface_image_min_line_width_mm = DefaultTopSurfaceImageMinLineWidthMm;
        top_surface_image_max_line_width_mm = DefaultTopSurfaceImageMaxLineWidthMm;
        top_surface_image_colored_top_layers = DefaultTopSurfaceImageColoredTopLayers;
        top_surface_image_fixed_coloring_filaments = DefaultTopSurfaceImageFixedColoringFilaments;
        top_surface_contoning_angle_threshold_deg = DefaultTopSurfaceContoningAngleThresholdDeg;
        top_surface_contoning_stack_layers = DefaultTopSurfaceContoningStackLayers;
        top_surface_contoning_pattern_filaments = DefaultTopSurfaceContoningPatternFilaments;
        top_surface_contoning_min_feature_mm = DefaultTopSurfaceContoningMinFeatureMm;
        top_surface_contoning_color_lower_surfaces = DefaultTopSurfaceContoningColorLowerSurfaces;
        top_surface_contoning_only_color_surface_infill = DefaultTopSurfaceContoningOnlyColorSurfaceInfill;
        top_surface_contoning_only_one_perimeter_around_shell_infill = DefaultTopSurfaceContoningOnlyOnePerimeterAroundShellInfill;
        top_surface_contoning_replace_top_perimeters_with_infill = DefaultTopSurfaceContoningReplaceTopPerimetersWithInfill;
        top_surface_contoning_recolor_surrounding_perimeters = DefaultTopSurfaceContoningRecolorSurroundingPerimeters;
        top_surface_contoning_perimeter_mode = DefaultTopSurfaceContoningPerimeterMode;
        top_surface_contoning_flat_surface_infill_mode = DefaultTopSurfaceContoningFlatSurfaceInfillMode;
        top_surface_contoning_layer_phase_enabled = DefaultTopSurfaceContoningLayerPhaseEnabled;
        top_surface_contoning_varied_infill_angles_enabled = DefaultTopSurfaceContoningVariedInfillAnglesEnabled;
        top_surface_contoning_blue_noise_error_diffusion_enabled = DefaultTopSurfaceContoningBlueNoiseErrorDiffusionEnabled;
        top_surface_contoning_supersampled_cells_enabled = DefaultTopSurfaceContoningSupersampledCellsEnabled;
        top_surface_contoning_polygonize_color_regions_enabled = DefaultTopSurfaceContoningPolygonizeColorRegionsEnabled;
        top_surface_contoning_surface_anchored_stacks_enabled = DefaultTopSurfaceContoningSurfaceAnchoredStacksEnabled;
        compact_offset_mode = DefaultCompactOffsetMode;
        use_legacy_fixed_color_mode = DefaultUseLegacyFixedColorMode;
        high_speed_image_texture_sampling = DefaultHighSpeedImageTextureSampling;
        minimum_visibility_offset_enabled = DefaultMinimumVisibilityOffsetEnabled;
        minimum_visibility_offset_pct = DefaultMinimumVisibilityOffsetPct;
        generic_solver_lookup_mode = DefaultGenericSolverLookupMode;
        generic_solver_mode = DefaultGenericSolverMode;
        generic_solver_mix_model = DefaultGenericSolverMixModel;
        dithering_enabled = DefaultDitheringEnabled;
        dithering_method = DefaultDitheringMethod;
        dithering_resolution_mm = DefaultDitheringResolutionMm;
        halftone_dot_size_mm = DefaultHalftoneDotSizeMm;
        contrast_pct = DefaultContrastPct;
        high_resolution_sampling = DefaultHighResolutionSampling;
        tone_gamma = DefaultToneGamma;
        transmission_distance_calibration_mode = DefaultTransmissionDistanceCalibrationMode;
        auto_adjust_filament_selection = DefaultAutoAdjustFilamentSelection;
        linear_gradient_mode = DefaultLinearGradientMode;
        linear_gradient_radius_mm = DefaultLinearGradientRadiusMm;
        linear_gradient_radius_percent = DefaultLinearGradientRadiusPercent;
        linear_gradient_radius_pct = DefaultLinearGradientRadiusPct;
        linear_gradient_stops.clear();
        filament_strengths_pct.clear();
        filament_minimum_offsets_pct.clear();
        filament_transmission_distances_mm.clear();
    }

    void clear_linear_gradient_points()
    {
        linear_gradient_start = {};
        linear_gradient_end = {};
    }

    bool has_custom_offset_settings() const
    {
        constexpr float eps = 1e-6f;
        return !offset_distances.empty() ||
               !offset_angles.empty() ||
               offset_mode != DefaultOffsetMode ||
               offset_rotation_enabled != DefaultOffsetRotationEnabled ||
               std::abs(offset_rotations - DefaultOffsetRotations) > eps ||
               std::abs(offset_repeats - DefaultOffsetRepeats) > eps ||
               offset_reverse_repeats != DefaultOffsetReverseRepeats ||
               offset_clockwise != DefaultOffsetClockwise ||
               offset_fade_mode != DefaultOffsetFadeMode ||
               offset_angle_mode != DefaultOffsetAngleMode;
    }

    bool operator==(const TextureMappingZone &rhs) const;
    bool operator!=(const TextureMappingZone &rhs) const { return !(*this == rhs); }
};

struct TextureMappingPrimeTowerImage
{
    std::vector<uint8_t> rgba;
    unsigned int width = 0;
    unsigned int height = 0;
    std::string image_name;

    bool valid() const
    {
        return width > 0 && height > 0 && rgba.size() >= size_t(width) * size_t(height) * 4;
    }

    void clear()
    {
        rgba.clear();
        width = 0;
        height = 0;
        image_name.clear();
    }
};

struct TextureMappingGlobalSettings
{
    bool enabled = false;
    float angle_offset_deg = 0.f;
    bool preserve_aspect_ratio = false;
    float preview_opacity_pct = TextureMappingZone::DefaultPreviewOpacityPct;
    bool preview_simulate_colors = TextureMappingZone::DefaultPreviewSimulateColors;
    bool preview_limit_resolution = TextureMappingZone::DefaultPreviewLimitResolution;
    bool preview_simulate_top_surface_lod = TextureMappingZone::DefaultPreviewSimulateTopSurfaceLod;
    std::string prime_tower_color_mode = "auto";
    uint64_t prime_tower_settings_zone_uid = 0;
    std::string image_file;
    std::string image_name;
    unsigned int image_width = 0;
    unsigned int image_height = 0;
    std::string image_file_back;
    std::string image_name_back;
    unsigned int image_width_back = 0;
    unsigned int image_height_back = 0;

    bool has_image_reference() const
    {
        return !image_file.empty() || image_width > 0 || image_height > 0 ||
               !image_file_back.empty() || image_width_back > 0 || image_height_back > 0;
    }
    bool effective_enabled(const TextureMappingPrimeTowerImage &image) const { return enabled && image.valid(); }
    bool effective_enabled(const TextureMappingPrimeTowerImage &image, const TextureMappingPrimeTowerImage &image_back) const
    {
        return enabled && (image.valid() || image_back.valid());
    }

    std::string serialize() const;
    void load(const std::string &serialized);
    void clear_image_reference();
    static std::string normalize_color_mode_name(const std::string &mode);
    static bool is_generic_solver_color_mode(const std::string &mode);
};

struct TextureMappingColorMatch
{
    unsigned int filament_id = 0;
    std::string  expected_color_name;
    float        perceptual_distance = 0.f;
};

class TextureMappingManager
{
public:
    TextureMappingManager() = default;

    void clear();
    void refresh(const std::vector<std::string> &filament_colours);
    void remove_physical_filament(unsigned int deleted_filament_id);

    TextureMappingZone *add_zone(size_t num_physical,
                                 const std::vector<std::string> &filament_colours,
                                 int surface_pattern = int(TextureMappingZone::ImageTexture));
    bool duplicate_zone(size_t zone_index,
                        size_t num_physical,
                        const std::vector<std::string> &filament_colours);

    unsigned int find_image_texture_zone_id(size_t num_physical,
                                            bool allow_raw_values = false,
                                            bool prefer_raw_values = false) const;
    unsigned int ensure_image_texture_zone(size_t num_physical,
                                           const std::vector<std::string> &filament_colours,
                                           bool allow_raw_values = false,
                                           bool prefer_raw_values = false);

    std::string serialize_entries();
    void load_entries(const std::string &serialized, const std::vector<std::string> &filament_colours);

    int zone_index_from_id(unsigned int zone_id) const;
    unsigned int zone_id_for_index(size_t zone_index) const;
    std::vector<unsigned int> zone_ids_by_index() const;
    unsigned int allocate_zone_id(size_t num_physical) const;
    void normalize_zone_ids(size_t num_physical);
    const TextureMappingZone *zone_from_stable_id(uint64_t stable_id) const;
    TextureMappingZone       *zone_from_stable_id(uint64_t stable_id);
    const TextureMappingZone *zone_from_id(unsigned int zone_id) const;
    TextureMappingZone       *zone_from_id(unsigned int zone_id);
    bool is_texture_mapping_zone_id(unsigned int zone_id) const { return zone_from_id(zone_id) != nullptr; }
    unsigned int resolve_zone_component(unsigned int zone_id, size_t num_physical, int layer_index) const;

    size_t total_filaments(size_t num_physical) const;
    std::vector<std::string> display_colors(size_t num_physical) const;
    std::vector<std::string> display_colors() const { return display_colors(m_filament_colours.size()); }

    static std::string filament_color_mode_name(int filament_color_mode);
    static size_t expected_component_count(int mapping_mode, int filament_color_mode);
    static bool component_count_mismatch(const TextureMappingZone &zone, size_t num_physical);
    static std::vector<unsigned int> effective_texture_component_ids(const TextureMappingZone      &zone,
                                                                     size_t                         num_physical,
                                                                     const std::vector<std::string> &filament_colours);
    static std::vector<unsigned int> selected_component_ids(const TextureMappingZone &zone, size_t num_physical);
    static bool auto_adjust_texture_component_ids(TextureMappingZone            &zone,
                                                  size_t                         num_physical,
                                                  const std::vector<std::string> &filament_colours);
    static std::vector<TextureMappingZone::LinearGradientStop> normalized_linear_gradient_stops(const TextureMappingZone &zone,
                                                                                                size_t                    num_physical);
    static std::vector<unsigned int> linear_gradient_component_ids_from_stops(const TextureMappingZone &zone,
                                                                              size_t                    num_physical);
    static std::vector<float> linear_gradient_compact_weights(float t,
                                                              const std::vector<TextureMappingZone::LinearGradientStop> &stops,
                                                              const std::vector<unsigned int> &component_ids);
    static std::vector<TextureMappingColorMatch> texture_component_color_matches(const TextureMappingZone      &zone,
                                                                                 size_t                         num_physical,
                                                                                 const std::vector<std::string> &filament_colours);
    static float poor_color_match_distance();

    static std::vector<float> default_offset_distances(size_t component_count, float reference_width_mm = 0.4f);
    static std::vector<float> default_offset_angles(size_t component_count);
    static std::vector<float> effective_offset_distances(const TextureMappingZone &zone,
                                                         size_t                    component_count,
                                                         float                     reference_width_mm = 0.4f);
    static std::vector<float> effective_offset_angles(const TextureMappingZone &zone, size_t component_count);
    static float max_component_surface_offset_mm(float reference_width_mm = 0.4f);
    static unsigned int resolve_zone_component(const TextureMappingZone      &zone,
                                               size_t                         num_physical,
                                               const std::vector<std::string> &filament_colours,
                                               int                            layer_index);

    static std::string blend_color_multi(const std::vector<std::pair<std::string, int>> &color_percents);

    const std::vector<TextureMappingZone> &zones() const { return m_zones; }
    std::vector<TextureMappingZone>       &zones()       { return m_zones; }

private:
    uint64_t allocate_stable_id();
    uint64_t normalize_stable_id(uint64_t stable_id);

    std::vector<TextureMappingZone> m_zones;
    uint64_t                        m_next_stable_id = 1;
    std::vector<std::string>        m_filament_colours;
};

} // namespace Slic3r

#endif /* slic3r_TextureMapping_hpp_ */
