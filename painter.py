import GeneticAlgorithm as GA
import moviepy.editor as mp
from tqdm import tqdm
import concurrent.futures
import numpy as np
import argparse
import imageio
import time
import sys
import cv2
import os

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

    def __init__(self,reference,greyScale=True):
        self.frames = []
        self.fps = 2000
        self.reference = reference
        self.greyScale = greyScale
        if self.greyScale:
            self.refImg = cv2.imread(reference, cv2.IMREAD_GRAYSCALE)
        else:
            self.refImg = cv2.imread(reference)

        # resize if is too big
        if self.refImg.shape[0] > 400 or self.refImg.shape[1] > 400:
            ratio = self.refImg.shape[0]/self.refImg.shape[1]
            self.refImg = cv2.resize(self.refImg, (int(400/ratio),400))

        self.height = self.refImg.shape[0]
        self.width  = self.refImg.shape[1]
        self.img = np.ones(self.refImg.shape, np.uint8) * 255

        # self.blurKernelSize = 1201
        # self.size_of_unblur = 30
        # self.update_blur(self.refImg)

        self.lowestScore = 1000000000
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

    # @timeit
    def paint(self,genomes):

        theBestCopy = []
        error_results = []

        # for genome in genomes:
        for genome in genomes:
            errorScores ,copyImg = self.decode(genome)
            error_results.append(errorScores)

            # OH TO MANY MAGIC NUMBERS - THIS 100 IS SETTING PRECISION FOR FLOAT NUMBERS

            if self.lowestScore > errorScores:
                theBestCopy = copyImg
                self.lowestScore = errorScores

        if len(theBestCopy):
            self.img = theBestCopy

        return error_results

    # @timeit
    def decode(self,genome):

        length = int(self.genLen/self.n_params)
        pos_x,pos_y,radius, rotation, brushes, colors = [0]*length,[0]*length,[0]*length,[0]*length,[None]*length,[(0,0,0)]*length

        for n in range(0,len(genome),self.n_params):
            pos_x[int(n/self.n_params)]  = int(genome[n]%480)
            pos_y[int(n/self.n_params)]  = int(genome[n+1]%480)
            radius[int(n/self.n_params)] = int(genome[n+2]%480)
            brushes[int(n/self.n_params)] = self.brushes[int(genome[n+3])%len(self.brushes)]
            rotation[int(n/self.n_params)] = int(genome[n+4]%360)
            if self.greyScale:
                colors[int(n/self.n_params)] = int(genome[n+5]%255)
            else:
                colors[int(n/self.n_params)] = [int(genome[n+5]%255),int((genome[n+6])%255),int((genome[n+7])%255)]

        # print(colors)
        
        # overlay = np.copy(self.img)
        copyImg = np.copy(self.img)
        for x,y,r,rot, brush, color in zip(pos_x,pos_y,radius,rotation, brushes, colors):
            r = r % 200
            if r == 0:
                r = 1
            mask = np.zeros(copyImg.shape[:2], dtype="uint8")    
        
            brush_rotated = rotate_image(brush,rot) 
            brush_resized = cv2.resize(brush_rotated,(r,r))
            brush_w, brush_h = mask[x:x+r,y:y+r].shape[:2]

            mask[x:x+r,y:y+r] = brush_resized[:brush_w,:brush_h]
            (_, mask) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            copyImg = self.paintBrush(copyImg,mask,color)

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
        errorScores = self.paint(genomes)
        genomes = GA.mixAndMutate(genomes,errorScores,mr=0.5,ms=2,maxPopulation=population,genomePurifying=True)

        self.paintTheBest()

        # if self.blurKernelSize > 1 and epoch % 10 == 0:
        #     self.blurKernelSize -= 8
        #     self.size_of_unblur = (self.size_of_unblur + 1) % 200
        #     self.blurKernelSize = max(self.blurKernelSize,1)
        #     self.update_blur(self.refImg)

        return genomes

    def run(self,genLen = 20,population = 10, epochs = 1000):
        if not self.greyScale:
            n_params = 8 # if RGB then we need two more genes for colors
        else:
            n_params = 6
        
        self.n_params = n_params
        self.genLen = genLen * self.n_params
        genomes = [np.random.randint(2**31,size=(self.genLen)) for _ in range(population)]

        for epoch in tqdm(range(epochs)):
            genomes = self.epoch(epoch,genomes,population)

            if epoch%int(self.fps/30):
                if self.greyScale:
                    self.frames.append(self.img)
                else:
                    self.frames.append(self.img)

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
    parser.add_argument("-cb", '--concurent_brushes', dest="concurent_brushes", help="Number of concurent brushes to run", type=int, default=20)
    parser.add_argument("-g", '--gray', dest="gray_scale", help="Decides if you want image in greyscale or not", type=bool, default=False)
    parser.add_argument("-gen", '--genomes', dest="genomes", help="Decides on number of used concurent genomes", type=int, default=20)
    parser.add_argument("-epochs", '--epochs', dest="epochs", help="Decides on number of epochs", type=int, default=1000)
    args = parser.parse_args()

    painter = Painter(args.file,args.gray_scale)
    painter.run(genLen = args.concurent_brushes,population = args.genomes, epochs= args.epochs)


# Load image using PIL
# img = Image.open('480px-Lenna_(test_image).jpg')
# Convert PIL image to NumPy array
# img_array = np.array(img)