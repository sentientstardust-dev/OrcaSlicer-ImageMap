#version 140

uniform mat4 view_model_matrix;
uniform mat4 projection_matrix;

in vec3 v_position;
in vec4 v_color;

out vec4 vertex_color;

void main()
{
    vertex_color = v_color;
    gl_Position = projection_matrix * view_model_matrix * vec4(v_position, 1.0);
}
