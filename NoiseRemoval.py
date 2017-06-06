import numpy as np
import cv2

class NoiseRemoval:
    
    def __init__(self):
        self.purpleDotLower = np.array([100, 70, 40])
        self.purpleDotUpper = np.array([180, 255, 150])
        self.purpleDotKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        self.birthMarkLower = np.array([0, 140, 40])
        self.birthMarkUpper = np.array([40, 255, 255])
    
    def setPurpleDotRemovalParameter(self, purpleDotLower, purpleDotUpper, purpleDotKernel):
        self.purpleDotLower = purpleDotLower
        self.purpleDotUpper = purpleDotUpper
        self.purpleDotKernel = purpleDotKernel

    def getPurpleDotRemovalParameter(self):
        return [self.purpleDotLower, self.purpleDotUpper, self.purpleDotKernel]

    def setBirthMarkRemovalParameter(self, birthMarkLower, birthMarkUpper):
        self.birthMarkLower = birthMarkLower
        self.birthMarkUpper = birthMarkUpper

    def getBirthMarkRemovalParameter(self):
        return [self.birthMarkLower, self.birthMarkUpper]

    def RemovePurpleDot(self, imbgr):
        imhsv = cv2.cvtColor(imbgr, cv2.COLOR_BGR2HSV)
        dotMask = cv2.inRange(imhsv, self.purpleDotLower , self.purpleDotUpper )
        if self.purpleDotKernel is not None:
            dotMask = cv2.bitwise_not(cv2.morphologyEx(dotMask, cv2.MORPH_OPEN, self.purpleDotKernel))
        return dotMask

    def RemoveBirthMark(self, imbgr):
        im = cv2.cvtColor(imbgr, cv2.COLOR_BGR2HSV)
        birthMask = cv2.bitwise_not(cv2.inRange(im, self.birthMarkLower, self.birthMarkUpper))
        return birthMask

    def mask(self, im):
        return cv2.bitwise_and(self.RemovePurpleDot(im), self.RemoveBirthMark(im));