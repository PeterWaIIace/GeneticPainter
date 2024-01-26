import GeneticAlgorithm as GA
import moviepy.editor as mp
from tqdm import tqdm
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
        self.lowestScore = 1000000000
        self.blurKernelSize = 1
        
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
        results = []

        for genome in genomes:
            errorScores ,copyImg = self.decode(genome)
            results.append(errorScores)

            if self.lowestScore > errorScores:
                theBestCopy = copyImg
                self.lowestScore = errorScores

        if len(theBestCopy):
            self.img = theBestCopy

        return results

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
                colors[int(n/self.n_params)] = (int(genome[n+5]%255),int((genome[n+5]/1000)%255) ,int((genome[n+5]/1000000)%255))

        # print(colors)
        
        overlay = np.copy(self.img)
        copyImg = np.copy(self.img)
        for x,y,r,rot, brush, color in zip(pos_x,pos_y,radius,rotation, brushes, colors):
            brush_background = np.zeros(copyImg.shape, dtype="uint8")
            mask = np.zeros(copyImg.shape[:2], dtype="uint8")
            r = r%200
            if r == 0:
                r = 1

            brush_rotated = rotate_image(brush,rot) 
            brush_resized = cv2.resize(brush_rotated,(r,r))
            brush_w, brush_h = mask[x:x+r,y:y+r].shape[:2]

            mask[x:x+r,y:y+r] = brush_resized[:brush_w,:brush_h]
            (thresh, mask) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            cv2.rectangle(brush_background,(0,0), (copyImg.shape[1],copyImg.shape[0]) , color,-1)
            masked_brush = cv2.bitwise_and(brush_background,brush_background,mask=mask)
            masked_img = cv2.bitwise_and(copyImg,copyImg,mask=cv2.bitwise_not(mask,mask))
            copyImg = cv2.bitwise_or(masked_img.copy(),masked_brush.copy())
        copyImg = cv2.addWeighted(overlay,0.5,copyImg,0.5,0)

        # calculate score
        self.blurredImg = cv2.blur(self.refImg, (self.blurKernelSize, self.blurKernelSize))

        diff1 = cv2.subtract(self.blurredImg, copyImg) #values are too low
        diff2 = cv2.subtract(copyImg,self.blurredImg) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        totalDiff = np.sum(totalDiff)/(self.width*self.height)

        return (totalDiff,copyImg)

    # @timeit
    def epoch(self,epoch,genomes,population):
        errorScores = self.paint(genomes)
        genomes = GA.mixAndMutate(genomes,errorScores,mr=0.5,ms=int(10),maxPopulation=population,genomePurifying=True)

        self.paintTheBest()

        if self.blurKernelSize > 1 and epoch % 20 == 0:
            self.blurKernelSize -= 5
            self.blurKernelSize = max(self.blurKernelSize,1)

        return genomes

    def run(self,genLen = 20, n_params = 6,population = 200, epochs = 1000):
        self.n_params = n_params
        self.genLen = genLen * self.n_params
        genomes = [np.random.randint(2**31,size=(self.genLen)) for _ in range(population)]

        for epoch in tqdm(range(epochs)):
            genomes = self.epoch(epoch,genomes,population)

            if epoch%int(self.fps/30):
                if self.greyScale:
                    self.frames.append(self.img)
                else:
                    self.frames.append(cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))

        imageio.mimsave(f'{self.reference[:-4]}.gif', self.frames, fps=30)
        clip = mp.VideoFileClip(f'{self.reference[:-4]}.gif')
        clip.write_videofile(f'{self.reference[:-4]}.mp4')

    # @timeit
    def paintTheBest(self):

        cv2.imshow('painted image',self.img)
        cv2.imshow('reference image',self.blurredImg)
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
    args = parser.parse_args()

    painter = Painter(args.file,args.gray_scale)
    painter.run(args.concurent_brushes)


# Load image using PIL
# img = Image.open('480px-Lenna_(test_image).jpg')
# Convert PIL image to NumPy array
# img_array = np.array(img)