#version 110

const vec3 ZERO = vec3(0.0, 0.0, 0.0);
const float INVALID_TEXTURE_CHECKER_SCALE = 0.2;
const int MAX_GRADIENT_COMPONENTS = 10;
const float EPSILON = 0.000001;

struct PrintVolumeDetection
{
    int type;
    vec4 xy_data;
    vec2 z_data;
};

uniform vec4 uniform_color;
uniform float texture_preview_mix;
uniform bool invalid_texture_mapping;
uniform PrintVolumeDetection print_volume;

uniform int gradient_component_count;
uniform vec3 gradient_base_color;
uniform vec3 gradient_component_colors[MAX_GRADIENT_COMPONENTS];
uniform float gradient_distances_mm[MAX_GRADIENT_COMPONENTS];
uniform float gradient_angles_deg[MAX_GRADIENT_COMPONENTS];
uniform float gradient_strength_factors[MAX_GRADIENT_COMPONENTS];
uniform float gradient_minimum_offset_factors[MAX_GRADIENT_COMPONENTS];
uniform float gradient_max_component_distance_mm;
uniform float gradient_max_width_delta_limit_mm;
uniform int gradient_angle_mode;
uniform bool gradient_rotation_enabled;
uniform float gradient_rotations;
uniform float gradient_repeats;
uniform bool gradient_reverse_repeats;
uniform bool gradient_clockwise;
uniform int gradient_fade_mode;
uniform vec3 gradient_center;
uniform float gradient_z_min;
uniform float gradient_z_max;

varying vec2 intensity;
varying vec3 clipping_planes_dots;
varying vec4 world_pos;
varying vec3 world_normal;

float normalize_angle(float angle)
{
    float out_angle = mod(angle, 360.0);
    if (out_angle < 0.0)
        out_angle += 360.0;
    return out_angle;
}

float angular_distance_deg(float a, float b)
{
    float d = abs(normalize_angle(a) - normalize_angle(b));
    return min(d, 360.0 - d);
}

float angular_distance_cw(float from_deg, float to_deg)
{
    float d = normalize_angle(to_deg) - normalize_angle(from_deg);
    if (d < 0.0)
        d += 360.0;
    return d;
}

float repeated_rotation_progress(float progress01, float repeats, bool reverse_repeats)
{
    float p = clamp(progress01, 0.0, 1.0);
    float r = max(1.0, repeats);
    if (r <= 1.0 + EPSILON)
        return p;

    float repeated_pos = p * r;
    float segment_idx = floor(repeated_pos);
    float local = repeated_pos - segment_idx;

    if (p >= 1.0 - EPSILON) {
        segment_idx = max(0.0, ceil(r) - 1.0);
        local = 1.0;
    }

    if (reverse_repeats && mod(segment_idx, 2.0) >= 1.0)
        local = 1.0 - local;
    return clamp(local, 0.0, 1.0);
}

float offset_fade_factor(int fade_mode, float progress01)
{
    float p = clamp(progress01, 0.0, 1.0);
    if (fade_mode == 1)
        return p;
    if (fade_mode == 2)
        return 1.0 - p;
    if (fade_mode == 3)
        return 1.0 - abs(2.0 * p - 1.0);
    if (fade_mode == 4)
        return abs(2.0 * p - 1.0);
    if (fade_mode == 5)
        return 1.0 - 2.0 * p;
    return 1.0;
}

float component_angular_influence(int component_idx, float theta_deg)
{
    int count = min(gradient_component_count, MAX_GRADIENT_COMPONENTS);
    if (count <= 0)
        return 0.0;
    if (count == 1)
        return 1.0;

    float self_angle = normalize_angle(gradient_angles_deg[component_idx]);
    float prev_angle = self_angle;
    float next_angle = self_angle;
    float prev_to_self_deg = 360.0;
    float self_to_next_deg = 360.0;

    for (int i = 0; i < MAX_GRADIENT_COMPONENTS; ++i) {
        if (i >= count || i == component_idx)
            continue;
        float other_angle = normalize_angle(gradient_angles_deg[i]);
        float prev_distance = angular_distance_cw(other_angle, self_angle);
        float next_distance = angular_distance_cw(self_angle, other_angle);
        if (prev_distance < prev_to_self_deg) {
            prev_to_self_deg = prev_distance;
            prev_angle = other_angle;
        }
        if (next_distance < self_to_next_deg) {
            self_to_next_deg = next_distance;
            next_angle = other_angle;
        }
    }

    if (prev_to_self_deg <= 0.001 || self_to_next_deg <= 0.001) {
        float total_weight = 0.0;
        float active_weight = 0.0;
        for (int i = 0; i < MAX_GRADIENT_COMPONENTS; ++i) {
            if (i >= count)
                continue;
            float weight = max(0.0, 1.0 - angular_distance_deg(theta_deg, gradient_angles_deg[i]) / 180.0);
            total_weight += weight;
            if (i == component_idx)
                active_weight += weight;
        }
        if (total_weight <= EPSILON)
            return 0.0;
        return clamp(active_weight / total_weight, 0.0, 1.0);
    }

    float theta_norm = normalize_angle(theta_deg);
    float prev_to_theta_deg = angular_distance_cw(prev_angle, theta_norm);
    if (prev_to_theta_deg <= prev_to_self_deg + 0.0001)
        return clamp(prev_to_theta_deg / prev_to_self_deg, 0.0, 1.0);

    float self_to_theta_deg = angular_distance_cw(self_angle, theta_norm);
    if (self_to_theta_deg <= self_to_next_deg + 0.0001)
        return clamp(1.0 - self_to_theta_deg / self_to_next_deg, 0.0, 1.0);

    return 0.0;
}

float variable_width_delta(float inset_strength, float max_width_delta_limit_mm, float minimum_offset_factor, float strength_factor)
{
    if (max_width_delta_limit_mm <= 0.0)
        return 0.0;

    float desired_width_factor = 1.0 - clamp(inset_strength, 0.0, 1.0);
    float min_width_factor = clamp(minimum_offset_factor, 0.0, 1.0);
    float adjusted_width_factor = min_width_factor + desired_width_factor * clamp(strength_factor, 0.0, 1.0) * (1.0 - min_width_factor);
    return clamp(max_width_delta_limit_mm * (1.0 - adjusted_width_factor), 0.0, max_width_delta_limit_mm);
}

vec3 surface_gradient_color()
{
    int count = min(gradient_component_count, MAX_GRADIENT_COMPONENTS);
    if (count <= 0)
        return uniform_color.rgb;

    float z_span = gradient_z_max - gradient_z_min;
    float z_progress = z_span > EPSILON ? clamp((world_pos.z - gradient_z_min) / z_span, 0.0, 1.0) : 0.0;

    float rotation_deg = 0.0;
    if (gradient_rotation_enabled) {
        float repeated = repeated_rotation_progress(z_progress, max(1.0, gradient_repeats), gradient_reverse_repeats);
        float direction = gradient_clockwise ? -1.0 : 1.0;
        rotation_deg = direction * 360.0 * gradient_rotations * repeated;
    }

    vec2 direction_vec = vec2(0.0);
    if (gradient_angle_mode == 1)
        direction_vec = world_normal.xy;
    if (dot(direction_vec, direction_vec) <= EPSILON) {
        vec3 radial = world_pos.xyz - gradient_center;
        direction_vec = radial.xy;
    }
    if (dot(direction_vec, direction_vec) <= EPSILON)
        direction_vec = vec2(1.0, 0.0);

    float theta_deg = normalize_angle(degrees(atan(direction_vec.y, direction_vec.x)) - rotation_deg);
    float signed_fade_factor = offset_fade_factor(gradient_fade_mode, z_progress);
    float fade_factor = abs(signed_fade_factor);
    float sample_theta_deg = signed_fade_factor < 0.0 ? normalize_angle(theta_deg + 180.0) : theta_deg;
    float influences[MAX_GRADIENT_COMPONENTS];
    float visibility_weights[MAX_GRADIENT_COMPONENTS];
    float min_visibility = 1.0;
    float max_visibility = 0.0;

    for (int i = 0; i < MAX_GRADIENT_COMPONENTS; ++i) {
        influences[i] = 0.0;
        visibility_weights[i] = 0.0;
        if (i < count)
            influences[i] = component_angular_influence(i, sample_theta_deg);
    }

    for (int i = 0; i < MAX_GRADIENT_COMPONENTS; ++i) {
        if (i >= count)
            continue;
        float raw_inset_mm = 0.0;
        for (int j = 0; j < MAX_GRADIENT_COMPONENTS; ++j) {
            if (j >= count || i == j)
                continue;
            raw_inset_mm += gradient_distances_mm[j] * influences[j];
        }
        float inset_strength = clamp(raw_inset_mm / max(gradient_max_component_distance_mm, EPSILON), 0.0, 1.0);
        float width_delta_mm = variable_width_delta(inset_strength * fade_factor,
                                                    gradient_max_width_delta_limit_mm,
                                                    gradient_minimum_offset_factors[i],
                                                    gradient_strength_factors[i]);
        visibility_weights[i] = 1.0 - clamp(width_delta_mm / max(gradient_max_width_delta_limit_mm, EPSILON), 0.0, 1.0);
        min_visibility = min(min_visibility, visibility_weights[i]);
        max_visibility = max(max_visibility, visibility_weights[i]);
    }

    vec3 target_color = vec3(0.0);
    float total_excess_weight = 0.0;
    for (int i = 0; i < MAX_GRADIENT_COMPONENTS; ++i) {
        if (i >= count)
            continue;
        float weight = max(0.0, visibility_weights[i] - min_visibility);
        target_color += gradient_component_colors[i] * weight;
        total_excess_weight += weight;
    }
    if (total_excess_weight <= EPSILON)
        return clamp(gradient_base_color, 0.0, 1.0);
    float contrast = clamp(max_visibility - min_visibility, 0.0, 1.0);
    target_color = clamp(target_color / total_excess_weight, 0.0, 1.0);
    return mix(clamp(gradient_base_color, 0.0, 1.0), target_color, contrast);
}

float invalid_texture_mapping_checker()
{
    vec3 normal_axes = abs(world_normal);
    vec2 checker_pos = world_pos.xy;
    if (normal_axes.x > normal_axes.y && normal_axes.x > normal_axes.z)
        checker_pos = world_pos.yz;
    else if (normal_axes.y > normal_axes.z)
        checker_pos = world_pos.xz;
    return mod(floor(checker_pos.x * INVALID_TEXTURE_CHECKER_SCALE) + floor(checker_pos.y * INVALID_TEXTURE_CHECKER_SCALE), 2.0);
}

void main()
{
    if (any(lessThan(clipping_planes_dots, ZERO)))
        discard;

    vec4 color = uniform_color;
    float mix_factor = clamp(texture_preview_mix, 0.0, 1.0);
    color.rgb = mix(color.rgb, surface_gradient_color(), mix_factor);
    if (invalid_texture_mapping) {
        float checker = invalid_texture_mapping_checker();
        vec3 checker_color = mix(vec3(0.0), vec3(1.0), checker);
        color.rgb = mix(color.rgb, checker_color, 0.62);
    }

    vec3 pv_check_min = ZERO;
    vec3 pv_check_max = ZERO;
    if (print_volume.type == 0) {
        pv_check_min = world_pos.xyz - vec3(print_volume.xy_data.x, print_volume.xy_data.y, print_volume.z_data.x);
        pv_check_max = world_pos.xyz - vec3(print_volume.xy_data.z, print_volume.xy_data.w, print_volume.z_data.y);
    }
    else if (print_volume.type == 1) {
        float delta_radius = print_volume.xy_data.z - distance(world_pos.xy, print_volume.xy_data.xy);
        pv_check_min = vec3(delta_radius, 0.0, world_pos.z - print_volume.z_data.x);
        pv_check_max = vec3(0.0, 0.0, world_pos.z - print_volume.z_data.y);
    }

    color.rgb = (any(lessThan(pv_check_min, ZERO)) || any(greaterThan(pv_check_max, ZERO))) ? mix(color.rgb, ZERO, 0.3333) : color.rgb;
    gl_FragColor = vec4(vec3(intensity.y) + color.rgb * intensity.x, color.a);
}
