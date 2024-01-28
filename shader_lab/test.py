import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image

# Create a NumPy array for your texture data
width, height = 512, 512
texture_data = np.random.rand(width, height, 3) * 255  # Example: Random RGB data

def initialize():
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    
    # Generate a texture ID
    texture_id = glGenTextures(1)
    
    # Bind the texture
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Set the texture parameters (you may need to adjust these based on your needs)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw a quad with the texture
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
    glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0)
    glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
    glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0)
    glEnd()
    
    glutSwapBuffers()

def update(texture):
    # Upload the NumPy array as the texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture.astype(np.uint8))

def save_texture():
    # glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # Convert the pixel data to a NumPy array
    image_data = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))

    # Now you can use the 'image_data' NumPy array as needed
    # Save the NumPy array as an image using Pillow
    image = Image.fromarray(image_data)
    image.save("output_texture.png")

def main_loop():
    texture_data = np.random.rand(width, height, 3) * 255  # Example: Random RGB data
    update(texture_data)
    glutPostRedisplay()
    glutMainLoopEvent()
    save_texture

if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutCreateWindow("NumPy Array as OpenGL Texture")
    glutReshapeWindow(width, height)
    
    initialize()
    update(texture_data)
    glutDisplayFunc(draw)
    glutIdleFunc(main_loop)  # Save the texture when the application is idle
    glutMainLoop()
