#version 110

const vec3 ZERO = vec3(0.0, 0.0, 0.0);
const float INVALID_TEXTURE_CHECKER_SCALE = 0.2;

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

varying vec2 intensity;
varying vec3 clipping_planes_dots;
varying vec4 world_pos;
varying vec3 world_normal;
varying vec4 vertex_color;

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
    color.rgb = mix(color.rgb, vertex_color.rgb, mix_factor);
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
