import cv2
import numpy as np
from Worker import *

class SkinDetection(Worker):

    #default parameters
    def __init__(self):
        self.lower_boundHSV = np.array([0, 10, 60])
        self.upper_boundHSV = np.array([20, 150, 255])
        self.lower_boundYCC = np.array([0,133,77])
        self.upper_boundYCC = np.array([255,173,127])
        self.kSizeClosing = 22 #kernel size for closing 
        self.kSizeErode = 13 
        self.iterations = 2
        self.closing = True;
        self.fill = True;

    #standard opencv fill hole method
    #input: binary image
    #output: binary image with holes filled
    def FillHoles(self, mask):
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        for h,cnt in enumerate(contours):
            cv2.drawContours(mask,[cnt],0,255,-1)
        return mask

    #opencv closing
    #input: binary image
    #output: binary image with clsoing 
    def MorphoClosing(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.kSizeClosing,self.kSizeClosing)))
        return mask

    #implmentation of abstract methods
    def mask(self, im):
        mask, _, _ = self.SkinThresholding(im);
        return mask;

    #input rgb image
    #output combined result, hsv result, ycc result. They are all binary images
    def SkinThresholding(self, im):
        #thresholding HSV space
        maskHSV = cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), self.lower_boundHSV, self.upper_boundHSV)
        #thresholding YCC space
        maskYCC = cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB), self.lower_boundYCC, self.upper_boundYCC)
        
        #apply morph closing for the masks
        if (self.closing == True):
            maskHSV = self.MorphoClosing(maskHSV)
            maskYCC = self.MorphoClosing(maskYCC)

        #apply contour filling 
        if (self.fill == True):
            maskHSV = self.FillHoles(maskHSV)
            maskYCC = self.FillHoles(maskYCC)
        
        #combine result
        mask = cv2.bitwise_and(maskHSV, maskYCC)
        if (self.iterations > 0):
            mask  = cv2.erode(mask , np.ones((self.kSizeErode,self.kSizeErode),np.uint8),iterations = self.iterations)

        return mask, maskHSV, maskYCC