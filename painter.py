import GeneticAlgorithm as GA
import moviepy.editor as mp
from tqdm import tqdm
import numpy as np
import argparse
import imageio
import time
import cv2
import os
from sys import platform
if platform == "linux" or platform == "linux2":
    import shader_painter as sp
else:
    import dummy_painter as sp


def timeit(function):
    def timing(*args, **kwargs):
        start = time.perf_counter()

        retVal = function(*args, **kwargs)
        print(f"[{function.__name__}] performance: {time.perf_counter() - start}s")
        return retVal
    return timing

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if img2.ndim<3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
        
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

class Painter:

    def __init__(self,reference,greyScale=True, shader = False, use_cirlces = False, background = ""):
        self.shader = shader
        self.use_circles = use_cirlces
        self.frames = []
        self.fps = 2000
        self.reference = reference
        self.greyScale = greyScale
        if self.greyScale:
            self.refImg = cv2.imread(reference, cv2.IMREAD_GRAYSCALE)
        else:
            self.refImg = cv2.imread(reference)

        # resize if is too big
        size = 400
        if self.refImg.shape[0] > size  or self.refImg.shape[1] > size :
            ratio = self.refImg.shape[0]/self.refImg.shape[1]
            self.refImg = cv2.resize(self.refImg, (int(size /ratio),size ))

        self.height = self.refImg.shape[0]
        self.width  = self.refImg.shape[1]
        self.img = np.ones(self.refImg.shape, np.uint8) * 255

        # self.blurKernelSize = 1201
        # self.size_of_unblur = 30
        # self.update_blur(self.refImg)

        self.lowestScore = 1000000000
        if len(background) <= 0:
            self.img = np.ones(self.refImg.shape, np.uint8) * 255
        else:
            self.img = cv2.imread(background)
            size = 400
            if self.img.shape[0] > size  or self.img.shape[1] > size :
                ratio = self.img.shape[0]/self.img.shape[1]
                self.img = cv2.resize(self.img, (int(size /ratio),size ))

        self.brush_background = np.zeros(self.img.shape, dtype="uint8")
        self.brushes = []
        f_brushes = os.listdir("brushes/")
        for f_brush in f_brushes:
            brush = cv2.imread(f'brushes/{f_brush}', cv2.IMREAD_GRAYSCALE)
            brush = cv2.bitwise_not(brush,brush)
            self.brushes.append(brush)
        
        # # here change brushes
        # self.brush = np.zeros((20,20,1))
        # cv2.circle(self.brush,(10,10), 10, 255,-1)
    
    def paint_shader(self,genomes):
        improved = False
        theBestCopy = []
        error_results = []

        # for genome in genomes:
        self.shader_painter.load_texture_from_array(self.img)
        for genome in genomes:
            translations,rotations,colors,brush_size = self.decode_for_shader(genome)
            copyImg = self.shader_painter.paint(translations,rotations,colors,brush_size).copy()[:,:,[2,1,0]]
            
            #convert to brg for opencv
            
            errorScores = self.compare(self.refImg,copyImg)
            error_results.append(errorScores)

            if self.lowestScore > errorScores:
                theBestCopy = copyImg
                self.lowestScore = errorScores
                improved = True

        if len(theBestCopy):
            self.img = theBestCopy

        return error_results,improved


    def decode_for_shader(self,genome):

        translations_x =  ((genome[0::self.n_params]%480)/480.0) * 2.0 - 1.0
        translations_y =  ((genome[1::self.n_params]%480)/480.0) * 2.0 - 1.0
        translations = np.column_stack((translations_x, translations_y))

        rotations      =  ((genome[2::self.n_params]%360)) * np.pi/180
        red            =  (genome[3::self.n_params]%1024/512) #/(np.uint32(-1)/2)
        blue           =  (genome[4::self.n_params]%1024/512) #/(np.uint32(-1)/2)
        green          =  (genome[5::self.n_params]%1024/512) #/(np.uint32(-1)/2)
        colors = np.column_stack((red, green, blue))
        brush_size     =  (genome[6::self.n_params]%480)/480.0
        
        return translations,rotations,colors,brush_size

    # @timeit
    def paint(self,genomes):
        improved = False
        theBestCopy = []
        error_results = []

        # for genome in genomes:
        for genome in genomes:
            if not self.use_circles:
                errorScores ,copyImg = self.decode(genome)
            else:
                errorScores ,copyImg = self.decode_circles(genome)
            error_results.append(errorScores)

            # OH TO MANY MAGIC NUMBERS - THIS 100 IS SETTING PRECISION FOR FLOAT NUMBERS

            if self.lowestScore > errorScores:
                theBestCopy = copyImg
                self.lowestScore = errorScores
                improved = True

        if len(theBestCopy):
            self.img = theBestCopy

        return error_results,improved

    # @timeit
    def decode(self,genome):

        pos_x =  genome[0::self.n_params]%480
        pos_y =  genome[1::self.n_params]%480
        rotation = genome[2::self.n_params]%360
        brushes  = np.zeros(len(rotation))
        red      = genome[3::self.n_params]%255  #/(np.uint32(-1)/2)
        blue     = genome[4::self.n_params]%255  #/(np.uint32(-1)/2)
        green    = genome[5::self.n_params]%255  #/(np.uint32(-1)/2)
        colors   = np.column_stack((blue, green, red))
        radius   =  genome[6::self.n_params]%480

        # overlay = np.copy(self.img)
        copyImg = np.copy(self.img)
        
        for x,y,r,rot,brush,color in zip(pos_x,pos_y,radius,rotation, brushes, colors):
            r = r % 200
            if r == 0:
                r = 1
            mask = np.zeros(copyImg.shape[:2], dtype="uint8")    
        
            brush_rotated = rotate_image(self.brushes[int(brush)],rot) 
            brush_resized = cv2.resize(brush_rotated,(r,r))
            brush_w, brush_h = mask[x:x+r,y:y+r].shape[:2]

            mask[x:x+r,y:y+r] = brush_resized[:brush_w,:brush_h]
            (_, mask) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            copyImg = self.paintBrush(copyImg,mask,tuple(map(int, color)))


        return (self.compare(self.refImg,copyImg),copyImg)

    def decode_circles(self,genome):

        overlay = np.copy(self.img)
        copyImg = np.copy(self.img)

        pos_x =  genome[0::self.n_params]%480
        pos_y =  genome[1::self.n_params]%480
        rotation = genome[2::self.n_params]%360
        red      = genome[3::self.n_params]%255  #/(np.uint32(-1)/2)
        blue     = genome[4::self.n_params]%255  #/(np.uint32(-1)/2)
        green    = genome[5::self.n_params]%255  #/(np.uint32(-1)/2)
        colors   = np.column_stack((blue, green, red))
        radius   =  genome[6::self.n_params]%480

        # pos_x,pos_y,radius,colors = [0]*self.genLen,[0]*self.genLen,[0]*self.genLen,[(0,0,0)]*self.genLen
        
        for x,y,r,rot,color in zip(pos_x,pos_y,radius,rotation, colors):
            # repurpose rot to radius
            r = rot
            c = np.ndarray.tolist(color)
            cv2.circle(copyImg,(x,y), int(r%60), c,-1)
            copyImg = cv2.addWeighted(overlay,0.5,copyImg,0.5,0)

        return (self.compare(self.refImg,copyImg),copyImg)


    def paintBrush(self, img, mask, color):

        brush_background = self.brush_background
        cv2.rectangle(brush_background,(0,0), (mask.shape[1],mask.shape[0]) , color, cv2.FILLED)
        brush_background = cv2.addWeighted(img.copy(), 0.5, brush_background, 0.5, 0)
        masked_brush = cv2.bitwise_and(brush_background,brush_background,mask=mask)
        masked_img   = cv2.bitwise_and(img,img,mask=cv2.bitwise_not(mask,mask))
        newImg       = cv2.bitwise_or(masked_img,masked_brush)
        
        return newImg


    def compare(self,refImg,newImg):

        diff1       = cv2.subtract(refImg, newImg) #values are too low
        diff2       = cv2.subtract(newImg,refImg) #values are too high
        totalDiff   = cv2.add(diff1, diff2)
        cv2.imshow("totalDiff",totalDiff)
        totalDiff   = np.sum(totalDiff)/(self.width*self.height)
        return totalDiff

    def update_blur(self,refImg):
        mask = np.zeros(refImg.shape[:2], np.uint8)
        H,W = refImg.shape[:2]
        cv2.circle(mask, (int(W/2),int(H/2)-20), self.size_of_unblur, (255,255,255), -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (self.blurKernelSize,self.blurKernelSize),11)
        blured = cv2.GaussianBlur(refImg, (self.blurKernelSize,self.blurKernelSize), self.blurKernelSize)
        if(self.size_of_unblur < 100):
            cv2.rectangle(blured,(0,0), (mask.shape[1],mask.shape[0]) , (255,255,255), cv2.FILLED)

        self.blurredImg = alphaBlend(refImg, blured, 255 - mask)

        # we need to update lowestScore so algorithm can reset and paint other details
        self.lowestScore = self.compare(self.blurredImg,self.img)
        cv2.imshow("blurredImg",self.blurredImg)

    # @timeit
    def epoch(self,epoch,genomes,population):
        improved = False
        # errorScores = self.paint(genomes)
        while not improved:
            if self.shader:
                errorScores,improved = self.paint_shader(genomes)
            else:
                errorScores,improved = self.paint(genomes)
        
            if not improved:
                genomes = GA.mixAndMutate(genomes,errorScores,mr=0.9,ms=int(self.n_params*0.9),maxPopulation=population,genomePurifying=False)

        genomes = GA.mixAndMutate(genomes,errorScores,mr=0.5,ms=int(self.n_params*0.5),maxPopulation=population,genomePurifying=False)
        self.paintTheBest()

        return genomes

    def run(self,genLen = 20,population = 10, epochs = 1000):
 
        self.shader_painter = sp.ShaderPainter(genLen,self.width,self.height)
        n_params = 7
        
        self.n_params = n_params
        self.genLen = genLen * self.n_params
        genomes = [np.random.randint(2**31,size=(self.genLen)) for _ in range(population)]

        for epoch in tqdm(range(epochs)):
            genomes = self.epoch(epoch,genomes,population)

            if epoch%int(self.fps/30):
                if self.greyScale:
                    self.frames.append(self.img)
                else:
                    self.frames.append(self.img.copy()[:,:,[2,1,0]])

        imageio.mimsave(f'{self.reference[:-4]}.gif', self.frames, fps=30)
        clip = mp.VideoFileClip(f'{self.reference[:-4]}.gif')
        clip.write_videofile(f'{self.reference[:-4]}.mp4')

    # @timeit
    def paintTheBest(self):

        cv2.imshow('painted image',self.img)
        cv2.imshow('reference image',self.refImg)
        if self.lowestScore == 0.0:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
        # cv2.destroyAllWindows()



if __name__=="__main__":
    parser = argparse.ArgumentParser("Genetic Painter")
    parser.add_argument("-f", '--file', dest="file", help="Reference picture", type=str, default="Lena.png")
    parser.add_argument("-cb", '--concurent_brushes', dest="concurent_brushes", help="Number of concurent brushes to run", type=int, default=4)
    parser.add_argument("-g", '--gray', dest="gray_scale", help="Decides if you want image in greyscale or not", type=bool, default=False)
    parser.add_argument("-gen", '--genomes', dest="genomes", help="Decides on number of used concurent genomes", type=int, default=40)
    parser.add_argument("-epochs", '--epochs', dest="epochs", help="Decides on number of epochs", type=int, default=2000)
    parser.add_argument("-shader", '--shader', dest="shader", help="Decides on using shader or not", type=bool, default=False)
    parser.add_argument("-use_circles", '--use_circles', dest="use_circles", help="Decides on using shader or not", type=bool, default=False)
    parser.add_argument("-use_background", '--use_background', dest="use_background", help="Reference picture for background", type=str, default="")
    args = parser.parse_args()

    painter = Painter(args.file, args.gray_scale, args.shader, args.use_circles, args.use_background)
    painter.run(genLen = args.concurent_brushes,population = args.genomes, epochs= args.epochs)


# Load image using PIL
# img = Image.open('480px-Lenna_(test_image).jpg')
# Convert PIL image to NumPy array
# img_array = np.array(img)