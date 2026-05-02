#version 110

const vec3 ZERO = vec3(0.0, 0.0, 0.0);
const float UV_EDGE_EPSILON = 0.000001;

struct PrintVolumeDetection
{
    int type;
    vec4 xy_data;
    vec2 z_data;
};

uniform vec4 uniform_color;
uniform sampler2D uniform_texture;
uniform float texture_preview_mix;
uniform bool invalid_texture_mapping;
uniform PrintVolumeDetection print_volume;

varying vec2 intensity;
varying vec3 clipping_planes_dots;
varying vec4 world_pos;
varying vec2 tex_coord;

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

void main()
{
    if (any(lessThan(clipping_planes_dots, ZERO)))
        discard;

    vec4 color = uniform_color;
    vec4 texture_color = texture2D(uniform_texture, texture_preview_coord(tex_coord));
    float mix_factor = clamp(texture_preview_mix, 0.0, 1.0);
    color.rgb = mix(color.rgb, texture_color.rgb, mix_factor);
    if (invalid_texture_mapping) {
        float checker = mod(floor(tex_coord.x * 24.0) + floor(tex_coord.y * 24.0), 2.0);
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
