import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders
from PIL import Image
import time


class ShaderPainter:


    def __init__(self):

        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutCreateWindow("Textured Background with Triangles")
        glutReshapeWindow(self.width, self.height)

        self.initialize()
        pass

    def 
    width, height = 512, 512
    image_array = np.zeros((width, height))
    num_brushes = 120

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
    uniform vec2  translations[{3 * num_brushes}];
    uniform float rotations[{3 * num_brushes}];
    uniform vec3  triangle_colors[{3 * num_brushes}];

    out vec3 aColor; // Varying variable to pass color to fragment shader

    void main()
    {{
        int index = gl_VertexID % {3 * num_brushes};
        vec2 translated_position = translations[index];
        mat2 rotation_matrix = mat2(cos(rotations[index]), -sin(rotations[index]),
                                    sin(rotations[index]), cos(rotations[index]));
        gl_Position = vec4(rotation_matrix * translated_position, 0.0, 1.0);
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


    def initialize(self):
        glEnable(GL_TEXTURE_2D)

        # Compile and link shaders for background
        self.background_vertex_shader = shaders.compileShader(self.background_vertex_shader_source, GL_VERTEX_SHADER)
        self.background_fragment_shader = shaders.compileShader(self.background_fragment_shader_source, GL_FRAGMENT_SHADER)
        self.background_shader_program = shaders.compileProgram(self.background_vertex_shader, self.background_fragment_shader)
        glUseProgram(self.background_shader_program)

        # Set up uniform locations for background
        self.background_texture_loc = glGetUniformLocation(self.background_shader_program, "background_texture")
        glUniform1i(self.background_texture_loc, 0)  # Use texture unit 0 for the background texture

        # Compile and link shaders for triangles
        self.triangle_vertex_shader = shaders.compileShader(self.triangle_vertex_shader_source, GL_VERTEX_SHADER)
        self.triangle_fragment_shader = shaders.compileShader(self.triangle_fragment_shader_source, GL_FRAGMENT_SHADER)
        self.triangle_shader_program = shaders.compileProgram(self.triangle_vertex_shader, self.triangle_fragment_shader)
        glUseProgram(self.triangle_shader_program)

        # Create and bind VBOs for translations, rotations, and scales
        self.translations_loc = glGetUniformLocation(self.triangle_shader_program, "translations")
        self.rotations_loc = glGetUniformLocation(self.triangle_shader_program, "rotations")
        self.colors_loc = glGetUniformLocation(self.triangle_shader_program, "triangle_colors")
        
    def update_location_1fv(self,update_location, length, buffor):
        glUniform1fv(update_location,length,buffor)

    def update_location_2fv(self,update_location, length, buffor):
        glUniform2fv(update_location,length,buffor)

    def update_location_3fv(self,update_location, length, buffor):
        glUniform3fv(update_location,length,buffor)


    def draw_background(self):
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

    def draw_triangles(self):
        glDrawArrays(GL_TRIANGLES, 0, 3 * self.num_brushes)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw background
        # glUseProgram(background_shader_program)
        # self.draw_background()

        # Draw triangles on top of the background
        glUseProgram(self.triangle_shader_program)
        self.draw_triangles()

        glutSwapBuffers()
        glutLeaveMainLoop()
    
    def save_texture(self):
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)

        # Convert the pixel data to a NumPy array
        image_data = np.frombuffer(pixels, dtype=np.uint8).reshape((self.height, self.width, 3))

        # Now you can use the 'image_data' NumPy array as needed
        # Save the NumPy array as an image using Pillow
        image = Image.fromarray(image_data)
        image.save("output_texture.png")
        return image_data

    def load_texture_from_array(self,image_array):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image_array.shape[1], self.image_array.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, self.image_array)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture_id

    def main_loop(self):
        global image_array

        background_texture = self.load_texture_from_array(self.image_array)    
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, background_texture)

        points_array = np.array([
            [-0.5, -0.5],  # Vertex 1
            [0.5, -0.5],   # Vertex 2
            [0.0, 0.5]     # Vertex 3
        ])

        translations = np.random.rand(self.num_brushes, 2) * 2.0 - 1.0
        rotations    = np.random.rand(self.num_brushes) * 2 * np.pi
        colors       = np.random.rand(self.num_brushes, 3)

        points_array = np.tile(points_array, (self.num_brushes, 1))
        translations = np.repeat(translations,3,axis=0) + points_array
        rotations = np.repeat(rotations,3,axis=0)
        colors = np.repeat(colors,3,axis=0)

        self.update_location_2fv(self.translations_loc, 3 * self.num_brushes, translations)
        self.update_location_1fv(self.rotations_loc, 3 * self.num_brushes, rotations)
        self.update_location_3fv(self.colors_loc, 3 * self.num_brushes, colors)

        glutPostRedisplay()
        glutMainLoopEvent()
        self.image_array = self.save_texture()

    def run(self):
    
        # Generate random data for 100 triangles
        self.image_array = self.save_texture()
        
        glutDisplayFunc(self.draw)
        # Use glutIdleFunc to continuously redraw in the main loop
        glutIdleFunc(self.main_loop)
        glutMainLoop()
        
        # background_shader_program, triangle_shader_program,translations_loc,rotations_loc,colors_loc = initialize()
        # image_array = self.save_texture()
    

if __name__ == "__main__":

    painter = ShaderPainter()
    painter.paint()