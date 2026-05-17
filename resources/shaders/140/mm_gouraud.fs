#version 140

#define INTENSITY_CORRECTION 0.6

// normalized values for (-0.6/1.31, 0.6/1.31, 1./1.31)
const vec3 LIGHT_TOP_DIR = vec3(-0.4574957, 0.4574957, 0.7624929);
#define LIGHT_TOP_DIFFUSE    (0.8 * INTENSITY_CORRECTION)
#define LIGHT_TOP_SPECULAR   (0.125 * INTENSITY_CORRECTION)
#define LIGHT_TOP_SHININESS  20.0

// normalized values for (1./1.43, 0.2/1.43, 1./1.43)
const vec3 LIGHT_FRONT_DIR = vec3(0.6985074, 0.1397015, 0.6985074);
#define LIGHT_FRONT_DIFFUSE  (0.3 * INTENSITY_CORRECTION)

#define INTENSITY_AMBIENT    0.3

const vec3  ZERO    = vec3(0.0, 0.0, 0.0);
const float EPSILON = 0.0001;
//BBS: add grey and orange
//const vec3 GREY = vec3(0.9, 0.9, 0.9);
const vec3 ORANGE = vec3(0.8, 0.4, 0.0);
const vec3 LightRed = vec3(0.78, 0.0, 0.0);
const vec3 LightBlue = vec3(0.73, 1.0, 1.0);
uniform vec4 uniform_color;

uniform bool volume_mirrored;

uniform mat4 view_model_matrix;
uniform mat3 view_normal_matrix;

in vec3 clipping_planes_dots;
in vec4 model_pos;
in vec4 world_pos;
in float smooth_world_normal_z;

struct SlopeDetection
{
    bool actived;
	 float normal_z;
    mat3 volume_world_normal_matrix;
    int preview_mode;
    float top_z;
    float bottom_z;
    vec4 highlight_color;
    bool override_all;
    vec4 override_mask0;
    vec4 override_mask1;
    vec4 override_mask2;
    vec4 override_mask3;
    int current_state;
    int base_state;
};
uniform SlopeDetection slope;

out vec4 out_color;

//BBS: add wireframe logic
in vec3 barycentric_coordinates;
float edgeFactor(float lineWidth) {
    vec3 d = fwidth(barycentric_coordinates);
    vec3 a3 = smoothstep(vec3(0.0), d * lineWidth, barycentric_coordinates);
    return min(min(a3.x, a3.y), a3.z);
}

vec3 wireframe(vec3 fill, vec3 stroke, float lineWidth) {
    return mix(stroke, fill, edgeFactor(lineWidth));
	//if (any(lessThan(barycentric_coordinates, vec3(0.005, 0.005, 0.005))))
	//	return vec3(1.0, 0.0, 0.0);
	//else
	//	return fill;
}

vec3 getWireframeColor(vec3 fill) {
    float brightness = 0.2126 * fill.r + 0.7152 * fill.g + 0.0722 * fill.b;
    return (brightness > 0.75) ? vec3(0.11, 0.165, 0.208) : vec3(0.988, 0.988, 0.988);
}
uniform bool show_wireframe;

float slopeOverrideMaskSlot(int slot) {
    if (slot == 0)
        return slope.override_mask0.x;
    if (slot == 1)
        return slope.override_mask0.y;
    if (slot == 2)
        return slope.override_mask0.z;
    if (slot == 3)
        return slope.override_mask0.w;
    if (slot == 4)
        return slope.override_mask1.x;
    if (slot == 5)
        return slope.override_mask1.y;
    if (slot == 6)
        return slope.override_mask1.z;
    if (slot == 7)
        return slope.override_mask1.w;
    if (slot == 8)
        return slope.override_mask2.x;
    if (slot == 9)
        return slope.override_mask2.y;
    if (slot == 10)
        return slope.override_mask2.z;
    if (slot == 11)
        return slope.override_mask2.w;
    if (slot == 12)
        return slope.override_mask3.x;
    if (slot == 13)
        return slope.override_mask3.y;
    if (slot == 14)
        return slope.override_mask3.z;
    if (slot == 15)
        return slope.override_mask3.w;
    return 0.0;
}

bool slopeOverrideMatches(int state) {
    if (slope.override_all)
        return true;
    if (state < 0 || state >= 256)
        return false;
    int slot = state / 16;
    int bit = state - slot * 16;
    float mask = slopeOverrideMaskSlot(slot);
    float divisor = pow(2.0, float(bit));
    return floor(mod(floor(mask / divisor), 2.0)) > 0.5;
}

bool slopePreviewMatches(float world_normal_z) {
    if (slope.preview_mode == 1)
        return world_normal_z >= slope.top_z - EPSILON;
    if (slope.preview_mode == 2)
        return world_normal_z <= slope.bottom_z + EPSILON;
    if (slope.preview_mode == 3)
        return world_normal_z <= slope.top_z + EPSILON && world_normal_z >= slope.bottom_z - EPSILON;
    return false;
}

float slopePreviewChecker(vec3 position, vec3 normal) {
    vec2 uv;
    vec3 abs_normal = abs(normal);
    if (abs_normal.z >= abs_normal.x && abs_normal.z >= abs_normal.y)
        uv = position.xy;
    else if (abs_normal.x >= abs_normal.y)
        uv = position.yz;
    else
        uv = position.xz;
    vec2 cells = floor(uv / 3.0);
    return mod(cells.x + cells.y, 2.0);
}

void main()
{
    if (any(lessThan(clipping_planes_dots, ZERO)))
        discard;
    vec3  color = uniform_color.rgb;
    float alpha = uniform_color.a;

    vec3 triangle_normal = normalize(cross(dFdx(model_pos.xyz), dFdy(model_pos.xyz)));
#ifdef FLIP_TRIANGLE_NORMALS
    triangle_normal = -triangle_normal;
#endif

    if (volume_mirrored)
        triangle_normal = -triangle_normal;

    vec3 transformed_normal = normalize(slope.volume_world_normal_matrix * triangle_normal);
     
    if (slope.actived) {
        if (slope.preview_mode == 0) {
            if(world_pos.z<0.1&&world_pos.z>-0.1)
             {
                  color = LightBlue;
                  alpha = 1.0;
             }
             else if( transformed_normal.z < slope.normal_z - EPSILON)
            {
                color = color * 0.5 + LightRed * 0.5;
                alpha = 1.0;
            }
        } else if (slope.preview_mode == 4) {
            float preview_luma = dot(slope.highlight_color.rgb, vec3(0.2126, 0.7152, 0.0722));
            vec3 preview_accent = preview_luma > 0.75 ? vec3(0.02, 0.08, 0.24) : vec3(1.0, 1.0, 1.0);
            float preview_check = slopePreviewChecker(world_pos.xyz, transformed_normal);
            vec3 preview_color_a = slope.highlight_color.rgb * 0.78 + preview_accent * 0.22;
            vec3 preview_color_b = slope.highlight_color.rgb * 0.25 + preview_accent * 0.75;
            color = mix(preview_color_a, preview_color_b, preview_check);
            alpha = max(alpha, slope.highlight_color.a);
        } else {
            int effective_state = slope.current_state == 0 ? slope.base_state : slope.current_state;
            if (slopeOverrideMatches(effective_state) && slopePreviewMatches(smooth_world_normal_z)) {
                float preview_luma = dot(slope.highlight_color.rgb, vec3(0.2126, 0.7152, 0.0722));
                vec3 preview_accent = preview_luma > 0.75 ? vec3(0.02, 0.08, 0.24) : vec3(1.0, 1.0, 1.0);
                float preview_check = slopePreviewChecker(world_pos.xyz, transformed_normal);
                vec3 preview_color_a = slope.highlight_color.rgb * 0.78 + preview_accent * 0.22;
                vec3 preview_color_b = slope.highlight_color.rgb * 0.25 + preview_accent * 0.75;
                color = mix(preview_color_a, preview_color_b, preview_check);
                alpha = max(alpha, slope.highlight_color.a);
            }
        }
    }
    // First transform the normal into camera space and normalize the result.
    vec3 eye_normal = normalize(view_normal_matrix * triangle_normal);

    // Compute the cos of the angle between the normal and lights direction. The light is directional so the direction is constant for every vertex.
    // Since these two are normalized the cosine is the dot product. We also need to clamp the result to the [0,1] range.
    float NdotL = max(dot(eye_normal, LIGHT_TOP_DIR), 0.0);

    // x = diffuse, y = specular;
    vec2 intensity = vec2(0.0);
    intensity.x = INTENSITY_AMBIENT + NdotL * LIGHT_TOP_DIFFUSE;
    vec3 position = (view_model_matrix * model_pos).xyz;
    intensity.y = LIGHT_TOP_SPECULAR * pow(max(dot(-normalize(position), reflect(-LIGHT_TOP_DIR, eye_normal)), 0.0), LIGHT_TOP_SHININESS);

    // Perform the same lighting calculation for the 2nd light source (no specular applied).
    NdotL = max(dot(eye_normal, LIGHT_FRONT_DIR), 0.0);
    intensity.x += NdotL * LIGHT_FRONT_DIFFUSE;

    if (show_wireframe) {
        vec3 wireframeColor = show_wireframe ? getWireframeColor(color) : color;
        vec3 triangleColor = wireframe(color, wireframeColor, 1.0);
        out_color = vec4(vec3(intensity.y) + triangleColor * intensity.x, alpha);
    }
    else {
        out_color = vec4(vec3(intensity.y) + color * intensity.x, alpha);
    }
}
