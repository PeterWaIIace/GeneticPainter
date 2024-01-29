import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders
from PIL import Image
import time

width, height = 512, 512
image_array = np.zeros((width, height))
num_brushes = 1

# Vertex Shader for Background
background_vertex_shader_source = """
#version 330 core
layout(location = 0) in vec2 in_position;
out vec2 tex_coords;

void main()
{
    gl_Position = vec4(in_position, 0.0, 1.0);
    tex_coords = in_position * 0.5 + 0.5;
}
"""

# Fragment Shader for Background
background_fragment_shader_source = """
#version 330 core
layout(location = 0) out vec4 frag_color;
in vec2 tex_coords;
uniform sampler2D background_texture;

void main()
{
    frag_color = texture(background_texture, tex_coords);
}
"""

# Vertex Shader for Triangles
triangle_vertex_shader_source = f"""
#version 330 core
layout(location = 0) in vec2 in_position;
uniform vec2  translations[{num_brushes}];
uniform float rotations[{num_brushes}];
uniform vec3  triangle_colors[{num_brushes} * 3];

out vec3 aColor; // Varying variable to pass color to fragment shader

void main()
{{
    int index = gl_VertexID % ({num_brushes} * 3);
    vec2 translated_position = in_position + translations[index];
    mat2 rotation_matrix = mat2(cos(rotations[index]), -sin(rotations[index]),
                                 sin(rotations[index]), cos(rotations[index]));
    gl_Position = vec4(rotation_matrix[index] * translated_position[index], 0.0, 1.0);
    aColor = triangle_colors[index];
}}
"""

# Fragment Shader for Triangles
triangle_fragment_shader_source = f"""
#version 330 core
in vec3 aColor;
layout(location = 0) out vec4 frag_color;

void main()
{{
    frag_color = vec4(aColor, 1.0);
}}
"""

def initialize():
    glEnable(GL_TEXTURE_2D)

    # Compile and link shaders for background
    background_vertex_shader = shaders.compileShader(background_vertex_shader_source, GL_VERTEX_SHADER)
    background_fragment_shader = shaders.compileShader(background_fragment_shader_source, GL_FRAGMENT_SHADER)
    background_shader_program = shaders.compileProgram(background_vertex_shader, background_fragment_shader)
    glUseProgram(background_shader_program)

    # Set up uniform locations for background
    background_texture_loc = glGetUniformLocation(background_shader_program, "background_texture")
    glUniform1i(background_texture_loc, 0)  # Use texture unit 0 for the background texture

    # Compile and link shaders for triangles
    triangle_vertex_shader = shaders.compileShader(triangle_vertex_shader_source, GL_VERTEX_SHADER)
    triangle_fragment_shader = shaders.compileShader(triangle_fragment_shader_source, GL_FRAGMENT_SHADER)
    triangle_shader_program = shaders.compileProgram(triangle_vertex_shader, triangle_fragment_shader)
    glUseProgram(triangle_shader_program)

    # Create and bind VBOs for translations, rotations, and scales
    translations_loc = glGetUniformLocation(triangle_shader_program, "translations")
    rotations_loc = glGetUniformLocation(triangle_shader_program, "rotations")
    colors_loc = glGetUniformLocation(triangle_shader_program, "triangle_colors")
    
    return (background_shader_program,triangle_shader_program,translations_loc,rotations_loc,colors_loc)

def update_location_1fv(update_location, length, buffor):
    glUniform1fv(update_location,length,buffor)

def update_location_2fv(update_location, length, buffor):
    glUniform2fv(update_location,length,buffor)

def update_location_3fv(update_location, length, buffor):
    glUniform3fv(update_location,length,buffor)


def draw_background():
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(-1.0, -1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(1.0, -1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(1.0, 1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(-1.0, 1.0)
    glEnd()

def draw_triangles():
    glDrawArrays(GL_TRIANGLES, 0, 3 * num_brushes)

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Draw background
    # glUseProgram(background_shader_program)
    # draw_background()

    # Draw triangles on top of the background
    glUseProgram(triangle_shader_program)
    draw_triangles()

    glutSwapBuffers()
    time.sleep(1)

def save_texture():
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # Convert the pixel data to a NumPy array
    image_data = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))

    # Now you can use the 'image_data' NumPy array as needed
    # Save the NumPy array as an image using Pillow
    # image = Image.fromarray(image_data)
    # image.save("output_texture.png")
    return image_data

def load_texture_from_array(image_array):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_array.shape[1], image_array.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image_array)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture_id

def main_loop():
    global image_array

    background_texture = load_texture_from_array(image_array)    
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, background_texture)

    points_array = np.array([[-1.0,-1.0],[1.0,-1.0],[0.0,1.0]])
    translations = np.random.rand(num_brushes, 2) * 2.0 - 1.0
    rotations    = np.random.rand(num_brushes) * 2 * np.pi
    colors       = np.random.rand(num_brushes, 3)

    points_array = np.tile(points_array, (num_brushes, 1))
    translations = np.repeat(translations,3,axis=0) + points_array
    print(translations)
    rotations = np.repeat(rotations,3,axis=0)
    colors = np.repeat(colors,3,axis=0)
    
    update_location_2fv(translations_loc, num_brushes,translations)
    update_location_1fv(rotation_loc, num_brushes,rotations)
    update_location_3fv(color_loc, num_brushes,colors)

    glutPostRedisplay()
    glutMainLoopEvent()
    image_array = save_texture()
    # Use texture unit 0 for the background texture
    # time.sleep(1)

if __name__ == "__main__":
    # Load your image as a NumPy array (replace this line with your image loading code)
    
    # Generate random data for 100 triangles

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutCreateWindow("Textured Background with Triangles")
    glutReshapeWindow(width, height)

    background_shader_program, triangle_shader_program,translations_loc,rotation_loc,color_loc = initialize()
    image_array = save_texture()
    
    glutDisplayFunc(draw)

    # Use glutIdleFunc to continuously redraw in the main loop
    glutIdleFunc(main_loop)

    glutMainLoop()
