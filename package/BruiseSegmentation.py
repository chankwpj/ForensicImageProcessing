import numpy as np
import cv2
from Worker import *
from random import randint

class BruiseSegmentation(Worker):
    def __init__(self):
        self.initRadius = 50
        self.radiusStep = 50;
        self.seedPoolSize = 1000
        self.timesGrow = 10
        self.regionSizeThreshold = 1000
        
    def mask(self, im, imgName = None):
        mask = self.segmentation(im, imgName)
        return mask
    
    def segmentation(self, im, imgName=None):
        resKMeans, resLabels = self.kMeansGeneration(im, imgName)
        return self.regionGrowingInKMeans(resKMeans, resLabels)

    def kMeansGeneration(self, in_im, imgName=None):
        resKMeans = cv2.imread(str(imgName) +'kmeansRGB.jpg') 
        resLabels = cv2.imread(str(imgName) +'LabelRGB.jpg') 
        if ( (resKMeans is not None) & (resLabels is not None)):
            #rgb to hsv
            resKMeans = cv2.cvtColor(resKMeans, cv2.COLOR_BGR2HSV)
            #label to gray?
            resLabels = cv2.cvtColor(resLabels , cv2.COLOR_BGR2GRAY)
            return [resKMeans, resLabels]

        im = np.copy(in_im)
        Z = im.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS, bestLabels=None, centers=None)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((im.shape))
        #res2 = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        #resKMeans = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
        resKMeans = res2
        resLabels = np.array(label).reshape(im.shape[0:2])
        if (imgName is not None):
            cv2.imwrite(str(imgName) +'kmeansRGB.jpg',resKMeans)
            cv2.imwrite(str(imgName) +'LabelRGB.jpg', resLabels)
        else:
            cv2.imwrite("temp" +'kmeansRGB.jpg', resKMeans)
            cv2.imwrite("temp" +'LabelRGB.jpg', resLabels)
        return [resKMeans, resLabels]

    def regionGrowingInKMeans(self, restKmeans, map):
        restKmeans = cv2.cvtColor(restKmeans, cv2.COLOR_BGR2GRAY)
        in0 = np.mean(restKmeans[map == 0])
        in1 = np.mean(restKmeans[map == 1])
        in2 = np.mean(restKmeans[map == 2])
        flags = [in0, in1, in2]
        #bflag = flags.index(np.median(flags))
        bflag = flags.index(np.max(flags))
        bmask = cv2.inRange(map, bflag , bflag)

        radius = self.initRadius
        while (True):
            circleMask = np.zeros(map.shape, dtype=np.uint8)
            cv2.circle(circleMask, (map.shape[1]/2, map.shape[0]/2), radius, 255, -1)
            flagMap = cv2.inRange(map, bflag, bflag)
            #cv2.imwrite("Test1.jpg", flagMap)
            flagMap = cv2.bitwise_and(flagMap, circleMask)
            #candidateImage = np.copy(img)
            #candidateImage[np.where(flagMap > 0)] = [255,0,0]
            #cv2.imwrite(str(index)+"Candidates.jpg", candidateImage)
            pointsCandidate = np.where(flagMap > 0)
            if len(pointsCandidate[0]) >= self.seedPoolSize:
                break
            else:
                radius = radius + self.radiusStep

        out = []
        for ind in range(0,self.timesGrow):
            maskGrow = cv2.bitwise_not(cv2.inRange(map, bflag, bflag))
            maskGrow = cv2.copyMakeBorder(maskGrow, 1, 1, 1, 1, cv2.BORDER_REPLICATE,value=255)
            randomInd = randint(0,len(pointsCandidate[0]) - 1)
            seed = tuple([pointsCandidate[1][randomInd],pointsCandidate[0][randomInd]])
            #cv2.circle(maskGrow, seed, 15, 255, -1)
            #cv2.imwrite(str(ind)+"Test.jpg", maskGrow)
            sp = np.zeros(map.shape, dtype=np.uint8)
            cv2.floodFill(sp, maskGrow, seed, 255)
            if (np.sum(sp) > self.regionSizeThreshold*255): 
                out.append(sp)
            #cv2.imwrite(str(ind)+"Test.jpg", sp)


        bruiseMask = np.uint8(np.zeros(sp.shape))
        for ind, sp in enumerate(out):
            bruiseMask = cv2.bitwise_or(bruiseMask, sp)

        return bruiseMask;