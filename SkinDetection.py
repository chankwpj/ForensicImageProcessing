import cv2
import numpy as np

class SkinDetection:
    def __init__(self, lower_boundHSV=None, upper_boundHSV=None, lower_boundYCC=None, upper_boundYCC=None):
        if lower_boundHSV!=None: 
            self.lower_boundHSV = lower_boundHSV
        else:
            self.lower_boundHSV = np.array([0, 10, 60])

        if upper_boundHSV!=None:
            self.upper_boundHSV = upper_boundHSV
        else:
            self.upper_boundHSV = np.array([20, 150, 255])
        
        if lower_boundYCC!=None:
            self.lower_boundYCC = lower_boundYCC
        else:
            self.lower_boundYCC = np.array([0,133,77])
             
        if upper_boundYCC!=None:
            self.upper_boundYCC = upper_boundYCC 
        else:
            self.upper_boundYCC = np.array([255,173,127])

        self.kSize = 22
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.kSize,self.kSize))

        self.erodeKernel = np.ones((13,13),np.uint8)
        self.iterations = 2
   
    def setHSVBounds(self, lower_boundHSV, upper_boundHSV):
        self.lower_boundHSV = lower_boundHSV
        self.upper_boundHSV = upper_boundHSV

    def setYCCBounds(self, lower_boundYCC, upper_boundYCC):
        self.lower_boundYCC = lower_boundYCC
        self.upper_boundYCC = upper_boundYCC 

    def setMorphoClosingKernel(self, kernel):
        self.kernel = kernel

    def setErode(self, kernel, iteration):
        self.erodeKernel = kernel
        self.iterations = iteration

    def FillHoles(self, mask):
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        for h,cnt in enumerate(contours):
            cv2.drawContours(mask,[cnt],0,255,-1)
        return mask
    

    def MorphoClosing(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask


    def SkinThresholding(self, im, closing, fill):
        #thresholding HSV space
        maskHSV = cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), self.lower_boundHSV, self.upper_boundHSV)
        #thresholding YCC space
        maskYCC = cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB), self.lower_boundYCC, self.upper_boundYCC)
        
        if (closing == True):
            maskHSV = self.MorphoClosing(maskHSV)
            maskYCC = self.MorphoClosing(maskYCC)

        if (fill == True):
            maskHSV = self.FillHoles(maskHSV)
            maskYCC = self.FillHoles(maskYCC)
        
        #IO.ImageWrite(maskHSV, "maskHSV", ".jpg")
        #IO.ImageWrite(maskYCC, "maskYCC", ".jpg")

        #combine result
        mask = cv2.bitwise_and(maskHSV, maskYCC)
        if (self.iterations > 0):
            mask  = cv2.erode(mask ,self.erodeKernel,iterations = self.iterations)

        return mask, maskHSV, maskYCC