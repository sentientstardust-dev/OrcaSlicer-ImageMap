#version 140

const vec3 ZERO = vec3(0.0, 0.0, 0.0);

uniform mat4 view_model_matrix;
uniform mat4 projection_matrix;

uniform mat4 volume_world_matrix;
// Clipping plane, x = min z, y = max z. Used by the FFF and SLA previews to clip with a top / bottom plane.
uniform vec2 z_range;
// Clipping plane - general orientation. Used by the SLA gizmo.
uniform vec4 clipping_plane;

in vec3 v_position;
in vec3 v_slope_normal;
in vec3 v_barycentric;

out vec3 clipping_planes_dots;
out vec4 model_pos;
out vec4 world_pos;
out float smooth_world_normal_z;
out vec3 barycentric_coordinates;

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
void main()
{
    model_pos = vec4(v_position, 1.0);
    // Point in homogenous coordinates.
    world_pos = volume_world_matrix * model_pos;
    smooth_world_normal_z = normalize(slope.volume_world_normal_matrix * v_slope_normal).z;

    gl_Position = projection_matrix * view_model_matrix * model_pos;
    // Fill in the scalars for fragment shader clipping. Fragments with any of these components lower than zero are discarded.
    clipping_planes_dots = vec3(dot(world_pos, clipping_plane), world_pos.z - z_range.x, z_range.y - world_pos.z);

    //compute the Barycentric Coordinates
    barycentric_coordinates = v_barycentric;
}
