import numpy as np
import cv2
from Worker import *
from random import randint

class BruiseSegmentation(Worker):
    def __init__(self, preProcessor):
        self.preProcessor = preProcessor;
        self.initRadius = 50
        self.radiusStep = 50;
        self.seedPoolSize = 1000
        self.timesGrow = 10
        self.regionSizeThreshold = 1000
        
    def setPreProcessor(self, preProcessor):
        self.preProcessor = preProcessor

    def getPrePorcessor(self):
        return self.preProcessor

    def getRadiusParameters(self):
        return [self.initRadius, self.radiusStep]
    
    def setRadiusParameters(self, initRadius, radiusStep):
        self.initRadius = initRadius;
        self.radiusStep = radiusStep;

    def getRegionGrowingParameters(self):
        return [self.seedPoolSize, self.timesGrow, self.regionSizeThreshold]

    def setRegionGrowingParameters(self, seedPoolSize, timesGrow,  regionSizeThreshold):
        self.seedPoolSize = seedPoolSize
        self.timesGrow = timesGrow
        self.regionSizeThreshold = regionSizeThreshold

    def mask(self, im):
        mask = self.preProcessor.mask(im)
        mask = self.segmentation(cv2.bitwise_and(im,im,mask))
        return mask

    def segmentation(self, im):
        resKMeans, resLabels = self.kMeansGeneration(im)
        return self.regionGrowingInKMeans(resKMeans, resLabels)

    def kMeansGeneration(self, im):
        img = np.copy(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        Z = im.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS, bestLabels=None, centers=None)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((im.shape))
        #res2 = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        resKMeans = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        resLabels = np.array(label).reshape(img.shape[0:2])
        #cv2.imwrite("k" + str(index) +'kmeansHSV.jpg',res2)
        #cv2.imwrite("k" + str(index) +'LabelHSV.jpg', np.array(label).reshape((img.shape[0:2])))
        return [resKMeans, resLabels]

    def regionGrowingInKMeans(self, restKmeans, map):
        in0 = np.mean(restKmeans[map == 0])
        in1 = np.mean(restKmeans[map == 1])
        in2 = np.mean(restKmeans[map == 2])
        flags = [in0, in1, in2]
        bflag = flags.index(np.median(flags))
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
            cv2.floodFill(sp, maskGrow, seed, 1)
            if (np.sum(sp) > self.regionSizeThreshold): 
                out.append(sp)
            #cv2.imwrite(str(ind)+"Test.jpg", sp)


        bruiseMask = np.uint8(np.zeros(sp.shape))
        for ind, sp in enumerate(out):
            bruiseMask = cv2.bitwise_or(bruiseMask, sp)

        return bruiseMask;