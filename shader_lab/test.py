import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders
from PIL import Image
import time

width, height = 512, 512

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
triangle_vertex_shader_source = """
#version 330 core
layout(location = 0) in vec2 in_position;
uniform vec2 translations;
uniform float rotation;

void main()
{
    vec2 translated_position = in_position + translations;
    mat2 rotation_matrix = mat2(cos(rotation), -sin(rotation),
                                 sin(rotation), cos(rotation));
    gl_Position = vec4(rotation_matrix * translated_position, 0.0, 0.5);
}
"""

# Fragment Shader for Triangles
triangle_fragment_shader_source = """
#version 330 core
layout(location = 0) out vec4 frag_color;
uniform vec3 triangle_color;

void main()
{
    frag_color = vec4(triangle_color, 1.0);
}
"""

image_array = np.zeros((width, height))

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

    # Set up uniform locations for triangles
    translations_loc = glGetUniformLocation(triangle_shader_program, "translations")
    rotation_loc = glGetUniformLocation(triangle_shader_program, "rotation")
    color_loc = glGetUniformLocation(triangle_shader_program, "triangle_color")

    return (background_shader_program,triangle_shader_program,translations_loc,rotation_loc,color_loc)

def load_texture_from_array(image_array):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_array.shape[1], image_array.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image_array)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture_id

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
    vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]], dtype=np.float32)

    for i in range(1):
        translations = np.random.rand(2) * 2.0 - 1.0
        rotation = np.random.rand() * 2 * np.pi
        color = np.random.rand(3).astype(np.float32)

        glUniform2fv(translations_loc, 1, translations)
        glUniform1f(rotation_loc, rotation)
        glUniform3fv(color_loc, 1, color)

        glBegin(GL_TRIANGLES)
        for vertex in vertices:
            glVertex2f(*vertex)
        glEnd()

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Draw background
    glUseProgram(background_shader_program)
    draw_background()

    # Draw triangles on top of the background
    glUseProgram(triangle_shader_program)
    draw_triangles()

    glutSwapBuffers()

def save_texture():
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # Convert the pixel data to a NumPy array
    image_data = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))

    # Now you can use the 'image_data' NumPy array as needed
    # Save the NumPy array as an image using Pillow
    image = Image.fromarray(image_data)
    image.save("output_texture.png")
    return image_data

def main_loop():
    global image_array

    background_texture = load_texture_from_array(image_array)    
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, background_texture)

    glutPostRedisplay()
    glutMainLoopEvent()
    image_array = save_texture()
    # Use texture unit 0 for the background texture
    time.sleep(1)


if __name__ == "__main__":
    # Load your image as a NumPy array (replace this line with your image loading code)
    
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutCreateWindow("Textured Background with Triangles")
    glutReshapeWindow(width, height)

    background_shader_program,triangle_shader_program,translations_loc,rotation_loc,color_loc = initialize()
    image_array = save_texture()
    
    glutDisplayFunc(draw)

    # Use glutIdleFunc to continuously redraw in the main loop
    glutIdleFunc(main_loop)

    glutMainLoop()
