// original author: sentientstardust

#ifndef slic3r_TextureMapping_hpp_
#define slic3r_TextureMapping_hpp_

#include <algorithm>
#include <cmath>
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
        Gradient2D   = 1
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
        DitheringHalftone = 3
    };

    enum TransmissionDistanceCalibrationMode : uint8_t {
        TDCalibrationNone = 0,
        TDCalibrationAbsolute = 1,
        TDCalibrationNeighbor = 2
    };

    static constexpr int   DefaultSurfacePattern = int(ImageTexture);
    static constexpr int   DefaultOffsetMode = int(OffsetBasic);
    static constexpr bool  DefaultOffsetRotationEnabled = true;
    static constexpr float DefaultOffsetRotations = 1.f;
    static constexpr float DefaultOffsetRepeats = 1.f;
    static constexpr bool  DefaultOffsetReverseRepeats = true;
    static constexpr bool  DefaultOffsetClockwise = true;
    static constexpr int   DefaultOffsetFadeMode = int(OffsetFadeNone);
    static constexpr int   DefaultOffsetAngleMode = int(OffsetAngleObjectCenter);
    static constexpr int   DefaultTextureMappingMode = int(TextureMappingFilamentBlending);
    static constexpr int   DefaultFilamentColorMode = int(FilamentColorCMYK);
    static constexpr bool  DefaultForceSequentialFilaments = false;
    static constexpr bool  DefaultReduceOuterSurfaceTexture = false;
    static constexpr bool  DefaultSeamHiding = false;
    static constexpr bool  DefaultNonlinearOffsetAdjustment = false;
    static constexpr bool  DefaultCompactOffsetMode = true;
    static constexpr bool  DefaultUseLegacyFixedColorMode = false;
    static constexpr bool  DefaultHighSpeedImageTextureSampling = true;
    static constexpr bool  DefaultMinimumVisibilityOffsetEnabled = false;
    static constexpr float DefaultMinimumVisibilityOffsetPct = 30.f;
    static constexpr int   DefaultGenericSolverLookupMode = int(GenericSolverClosestMix);
    static constexpr int   DefaultGenericSolverMode = int(GenericSolverV2);
    static constexpr int   DefaultGenericSolverMixModel = int(GenericSolverPigmentPainter);
    static constexpr bool  DefaultDitheringEnabled = false;
    static constexpr int   DefaultDitheringMethod = int(DitheringFloydSteinberg);
    static constexpr float DefaultDitheringResolutionMm = 0.08f;
    static constexpr float DefaultHalftoneDotSizeMm = 0.32f;
    static constexpr float DefaultContrastPct = 100.f;
    static constexpr bool  DefaultHighResolutionSampling = true;
    static constexpr float DefaultToneGamma = 1.f;
    static constexpr int   DefaultTransmissionDistanceCalibrationMode = int(TDCalibrationAbsolute);
    static constexpr bool  DefaultPreviewSimulateColors = false;
    static constexpr bool  DefaultPreviewLimitResolution = true;
    static constexpr bool  DefaultAutoAdjustFilamentSelection = true;

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
    float       preview_opacity_pct = DefaultPreviewOpacityPct;
    bool        preview_simulate_colors = DefaultPreviewSimulateColors;
    bool        preview_limit_resolution = DefaultPreviewLimitResolution;
    bool        auto_adjust_filament_selection = DefaultAutoAdjustFilamentSelection;
    std::vector<float> filament_strengths_pct;
    std::vector<float> filament_minimum_offsets_pct;
    std::vector<float> filament_transmission_distances_mm;

    bool is_image_texture() const { return surface_pattern == int(ImageTexture); }
    bool is_2d_gradient() const { return surface_pattern == int(Gradient2D); }

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
        preview_opacity_pct = DefaultPreviewOpacityPct;
        preview_simulate_colors = DefaultPreviewSimulateColors;
        preview_limit_resolution = DefaultPreviewLimitResolution;
        auto_adjust_filament_selection = DefaultAutoAdjustFilamentSelection;
        filament_strengths_pct.clear();
        filament_minimum_offsets_pct.clear();
        filament_transmission_distances_mm.clear();
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
