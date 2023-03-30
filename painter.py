import GeneticAlgorithm as GA
from tqdm import tqdm
import numpy as np
import imageio
import time
import sys
import cv2

def timeit(function):
    def timing(*args, **kwargs):
        start = time.perf_counter()

        retVal = function(*args, **kwargs)
        print(f"[{function.__name__}] performance: {time.perf_counter() - start}s")
        return retVal
    return timing

class Painter:

    def __init__(self,reference):
        self.frames = []
        self.fps = 500
        self.reference = reference
        self.refImg = cv2.imread(reference, cv2.IMREAD_GRAYSCALE)
        self.height = self.refImg.shape[0]
        self.width  = self.refImg.shape[1]
        self.img = np.zeros((self.height,self.width,1), np.uint8)
        self.lowestScore = 1000000000
        self.blurKernelSize = 1

        self.brush = np.zeros((20,20,1))
        cv2.circle(self.brush,(10,10), 10, 255,-1)

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

        overlay = np.copy(self.img)
        copyImg = np.copy(self.img)
        cpBrush = np.copy(self.brush)

        pos_x,pos_y,radius,colors = [0]*self.genLen,[0]*self.genLen,[0]*self.genLen,[0]*self.genLen
        for n in range(0,len(genome),4):
            pos_x[int(n/4)]  = int(genome[n]%480)
            pos_y[int(n/4)]  = int(genome[n+1]%480)
            radius[int(n/4)] = int(genome[n+2]%480)
            colors[int(n/4)] = int(genome[n+3]%255)

        # print(colors)
        for x,y,r,color in zip(pos_x,pos_y,radius,colors):
            cv2.circle(copyImg,(x,y), int(r%60), color,-1)

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

        if self.blurKernelSize > 1 and epoch % 10 == 0:
            self.blurKernelSize -= 5
            self.blurKernelSize = max(self.blurKernelSize,1)

        return genomes

    def run(self,genLen = 20*4,population = 200, epochs = 6000):
        self.genLen = genLen
        genomes = [np.random.randint(2**31,size=(self.genLen)) for _ in range(population)]

        for epoch in tqdm(range(epochs)):
            genomes = self.epoch(epoch,genomes,population)

            if epoch%int(self.fps/30):
                self.frames.append(self.img)

        imageio.mimsave(f'{self.reference}.gif', self.frames, fps=30)

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
    painter = Painter(sys.argv[1])
    painter.run()


# Load image using PIL
# img = Image.open('480px-Lenna_(test_image).jpg')
# Convert PIL image to NumPy array
# img_array = np.array(img)