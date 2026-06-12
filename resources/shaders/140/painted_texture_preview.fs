#version 140

const vec3 ZERO = vec3(0.0, 0.0, 0.0);
const float UV_EDGE_EPSILON = 0.000001;
const float INVALID_TEXTURE_CHECKER_SCALE = 0.2;
const float CONTONING_FLAT_SURFACE_NORMAL_Z = 0.999;

struct PrintVolumeDetection
{
    int type;
    vec4 xy_data;
    vec2 z_data;
};

uniform vec4 uniform_color;
uniform sampler2D uniform_texture;
uniform sampler2D contoning_flat_surface_texture;
uniform sampler2D contoning_flat_surface_bottom_texture;
uniform float texture_preview_mix;
uniform bool invalid_texture_mapping;
uniform bool contoning_flat_surface_texture_enabled;
uniform bool contoning_flat_surface_bottom_texture_enabled;
uniform bool raw_atlas_surface_filter_enabled;
uniform bool raw_atlas_side_texture_enabled;
uniform bool raw_atlas_flat_texture_enabled;
uniform bool color_match_preview_active;
uniform vec3 color_match_target_oklab;
uniform float color_match_tolerance_sq;
uniform vec4 color_match_highlight_color;
uniform vec4 color_match_background_color;
uniform PrintVolumeDetection print_volume;

in vec2 intensity;
in vec3 clipping_planes_dots;
in vec4 world_pos;
in vec3 world_normal;
in vec2 tex_coord;

out vec4 out_color;

float texture_preview_coord(float uv)
{
    if (uv >= -UV_EDGE_EPSILON && uv <= 1.0 + UV_EDGE_EPSILON)
        return clamp(uv, 0.0, 1.0);

    return fract(uv);
}

vec2 texture_preview_coord(vec2 uv)
{
    return vec2(texture_preview_coord(uv.x), texture_preview_coord(uv.y));
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

float srgb_channel_to_linear(float c)
{
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

vec3 oklab_from_srgb(vec3 c)
{
    vec3 linear = vec3(srgb_channel_to_linear(c.r), srgb_channel_to_linear(c.g), srgb_channel_to_linear(c.b));
    float l = 0.4122214708 * linear.r + 0.5363325363 * linear.g + 0.0514459929 * linear.b;
    float m = 0.2119034982 * linear.r + 0.6806995451 * linear.g + 0.1073969566 * linear.b;
    float s = 0.0883024619 * linear.r + 0.2817188376 * linear.g + 0.6299787005 * linear.b;
    float l_ = pow(max(l, 0.0), 1.0 / 3.0);
    float m_ = pow(max(m, 0.0), 1.0 / 3.0);
    float s_ = pow(max(s, 0.0), 1.0 / 3.0);
    return vec3(0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_);
}

void main()
{
    if (any(lessThan(clipping_planes_dots, ZERO)))
        discard;

    vec4 color = uniform_color;
    vec4 texture_color = texture(uniform_texture, texture_preview_coord(tex_coord));
    float normal_z = normalize(world_normal).z;
    bool flat_surface = abs(normal_z) >= CONTONING_FLAT_SURFACE_NORMAL_Z;
    if (raw_atlas_surface_filter_enabled) {
        bool raw_atlas_surface_allowed = flat_surface ? raw_atlas_flat_texture_enabled : raw_atlas_side_texture_enabled;
        if (!raw_atlas_surface_allowed) {
            if (color_match_preview_active)
                discard;
            texture_color = color;
        }
    }
    if (normal_z >= CONTONING_FLAT_SURFACE_NORMAL_Z) {
        if (contoning_flat_surface_texture_enabled)
            texture_color = texture(contoning_flat_surface_texture, texture_preview_coord(tex_coord));
    } else if (normal_z <= -CONTONING_FLAT_SURFACE_NORMAL_Z) {
        if (contoning_flat_surface_bottom_texture_enabled)
            texture_color = texture(contoning_flat_surface_bottom_texture, texture_preview_coord(tex_coord));
    }
    float mix_factor = clamp(texture_preview_mix, 0.0, 1.0);
    if (color_match_preview_active) {
        float source_alpha = clamp(texture_color.a, 0.0, 1.0);
        vec3 source_rgb = texture_color.rgb * source_alpha + color_match_background_color.rgb * (1.0 - source_alpha);
        vec3 delta = oklab_from_srgb(source_rgb) - color_match_target_oklab;
        if (dot(delta, delta) > color_match_tolerance_sq)
            discard;
        color = color_match_highlight_color;
    } else {
        color.rgb = mix(color.rgb, texture_color.rgb, mix_factor);
        if (invalid_texture_mapping) {
            float checker = invalid_texture_mapping_checker();
            vec3 checker_color = mix(vec3(0.0), vec3(1.0), checker);
            color.rgb = mix(color.rgb, checker_color, 0.62);
        }
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
    out_color = vec4(vec3(intensity.y) + color.rgb * intensity.x, color.a);
}
