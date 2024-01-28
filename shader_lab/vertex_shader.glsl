#version 330 core
layout(location = 0) in vec2 in_position;
uniform vec2 translations[10];
uniform float rotations[10];

void main()
{
    vec2 translated_position = in_position + translations[gl_InstanceID];
    mat2 rotation_matrix = mat2(cos(rotations[gl_InstanceID]), -sin(rotations[gl_InstanceID]),
                                 sin(rotations[gl_InstanceID]), cos(rotations[gl_InstanceID]));
    gl_Position = vec4(rotation_matrix * translated_position, 0.0, 1.0);
}
"""
