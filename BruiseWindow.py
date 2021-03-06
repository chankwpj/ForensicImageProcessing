import numpy as np
import cv2
import cv2.cv as cv
from sympy import Point
import math
from Worker import *
from NoiseRemoval import *

class BruiseWindow:
    #default parameters
    def __init__(self, noiseRemoval=None):
        self.lower = np.array([180, 180, 180]) # okay
        self.upper = np.array([255, 255, 255])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        self.iteration = 5
        self.edgeRemovalKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.edgeRemovalIteration = 5

        self.houghCirclesParameters = [cv.CV_HOUGH_GRADIENT,1,250, 90,20,120,440];
        if (noiseRemoval is None):
            self.noiseRemoval = None
        else:
            self.noiseRemoval = noiseRemoval
    
    #input: image
    #output: rulerMask, binary image (ruler)
    def extractRuler(self, im):
        rulerMask = cv2.inRange(im, self.lower, self.upper)
        return rulerMask 

    #input: binary image
    #output1: circles, opencv circle list 
    #output2: rulerMaskE, the eroded mask
    def getAllDetectedCircles(self, rulerMask):
        rulerMaskE = cv2.erode(rulerMask, self.kernel, iterations=self.iteration)
        circles = cv2.HoughCircles(rulerMaskE,self.houghCirclesParameters[0],self.houghCirclesParameters[1],self.houghCirclesParameters[2], param1=self.houghCirclesParameters[3],param2=self.houghCirclesParameters[4],minRadius=self.houghCirclesParameters[5],maxRadius=self.houghCirclesParameters[6]) #120 #20 #480 
        return circles, rulerMaskE 
    
    
    #input1: binary ruler image 
    #input2: opencv circle list 
    #output1-4: pt0 ... pt3 the points that are valid circles. pt0 is grid 0 vaild point. ptx[0] is x direction [1] is y direction
    #grid
    #0 1
    #2 3
    #output5: gridWithNoPoint, indicates the grid id without targeted point
    def selectTheCornerCircles(self, rulerMask, input_circles):
        circles = np.copy(input_circles)
        #draw bounding box for the circle
        circleSpace = np.zeros(rulerMask.shape)
        circles = np.uint16(np.around(circles))
        cindex = 1;
        for i in circles[0,:]:
            cv2.circle(circleSpace,(i[0],i[1]),i[2],255,2)
            #visuliazation 
            '''
            if (cindex <= 3):
                # draw the outer circle
                #cv2.circle(circleSpace,(i[0],i[1]),2,255,3)
                cv2.circle(im,(i[0],i[1]),i[2],(0,0,255),10)
                cv2.circle(im,(i[0],i[1]),2,(0,0,255),10)
            else:
                # draw the outer circle
                #cv2.circle(circleSpace,(i[0],i[1]),2,255,3)
                cv2.circle(im,(i[0],i[1]),i[2],(255,0,0),10)
                cv2.circle(im,(i[0],i[1]),2,(255,0,0),10)
            '''
            cindex = cindex + 1;
        
        points = cv2.findNonZero(cv2.convertScaleAbs(circleSpace))
    
        #4 sub boxes in the bounding box
        x,y,w,h = cv2.boundingRect(np.int32(points))
        LeftUpBox = [x,y,x + w/2, y + h/2]
        RightUpBox = [x+ w/2,y, x+w-1, y + h/2 -1]
        LeftDownBox = [x,y+h/2,x + w/2,y+h]
        RightDownBox = [x+w/2,y+h/2,x+w-1,y+h-1]
    
        tempSpace = np.zeros(rulerMask.shape)
        cv2.rectangle(tempSpace, (LeftUpBox[0], LeftUpBox[1]), (LeftUpBox[2], LeftUpBox[3] ), (255))
        cnt0 = cv2.findNonZero(cv2.convertScaleAbs(tempSpace))

        tempSpace = np.zeros(rulerMask.shape)
        cv2.rectangle(tempSpace, (RightUpBox[0], RightUpBox[1]), (RightUpBox[2], RightUpBox[3] ), (255))
        cnt1 = cv2.findNonZero(cv2.convertScaleAbs(tempSpace))

        tempSpace = np.zeros(rulerMask.shape)
        cv2.rectangle(tempSpace, (LeftDownBox[0], LeftDownBox[1]), (LeftDownBox[2], LeftDownBox[3] ), (255))
        cnt2 = cv2.findNonZero(cv2.convertScaleAbs(tempSpace))

        tempSpace = np.zeros(rulerMask.shape)
        cv2.rectangle(tempSpace, (RightDownBox[0], RightDownBox[1]), (RightDownBox[2], RightDownBox[3] ), (255))
        cnt3 = cv2.findNonZero(cv2.convertScaleAbs(tempSpace))
    
        contours = [cnt0, cnt1, cnt2, cnt3]
        #cv2.drawContours(im, contours, 1, (0,255,0), 3)

        '''
        Vis
        cv2.drawContours(im, cnt0, -1, (255,0,0), 8)
        cv2.drawContours(im, cnt1, -1, (0,255,0), 8)
        cv2.drawContours(im, cnt2, -1, (0,0,255), 8)
        cv2.drawContours(im, cnt3, -1, (175,175,175), 8)
        '''
        #count box
        #grid
        #0 1
        #2 3
        grid = [[],[],[],[]]
        gridWithNoPoint = 0;
        minNum = 999
        for index, cnt in enumerate(contours):
            for i in circles[0,:]:
                if (cv2.pointPolygonTest(np.array(cnt), (i[0], i[1]), False) >= 0 ):
                    grid[index].append((i[0], i[1]))
                    #cv2.circle(im,(i[0], i[1]),50,(0,255,0),-1)
            if ( len(grid[index]) < minNum):
                minNum = len(grid[index])
                gridWithNoPoint = index
    
                midPt = [x+w/2, y+h/2]
        #cv2.circle(im,(midPt[0], midPt[1]),50,(255,0,0),-1)
    
        '''
            x,y,w,h = cv2.boundingRect(np.int32(points))
        LeftUpBox = [x,y,x + w/2, y + h/2]
        RightUpBox = [x+ w/2,y, x+w-1, y + h/2 -1]
        LeftDownBox = [x,y+h/2,x + w/2,y+h]
        RightDownBox = [x+w/2,y+h/2,x+w-1,y+h-1]
        '''
        pt0 = self.minDistanceCornerToPoint(grid[0], [x,y])
        pt1 = self.minDistanceCornerToPoint(grid[1], [x+w,y])
        pt2 = self.minDistanceCornerToPoint(grid[2], [x,y+h])
        pt3 = self.minDistanceCornerToPoint(grid[3], [x+w,y+h])
        return [pt0, pt1, pt2, pt3, gridWithNoPoint];


    #input1: empty mask for drawing window
    #input 2 - 5: points 
    #input6: the grid id that is invalid
    #output1: the window mask
    #output2: the points used to draw the window
    def threePointsToWindow(self, emptyMask, pt0, pt1, pt2, pt3, gridWithNoPoint):
        Tpoints = []
        if (gridWithNoPoint == 0):
            #print "com0"
            #get pt2x pt1y 
            cv2.rectangle(emptyMask, (pt2[0], pt1[1]), pt3, 255, -1)
            Tpoints = [pt1, pt2, pt3]
            #cv2.circle(im,(pt2[0], pt1[1]),20,(0,255,0),-1)
            #cv2.circle(im,pt3,20,(0,255,0),-1)
        elif (gridWithNoPoint == 3):
            #print "com3"
            #get pt1x pt2y 
            cv2.rectangle(emptyMask, pt0, (pt1[0], pt2[1]), 255, -1)
            Tpoints = [pt0, pt1, pt2]
            #cv2.circle(im,pt0,20,(0,255,0),-1)
            #cv2.circle(im,(pt1[0], pt2[1]),20,(0,255,0),-1)
        else:
            #print "norm"
            cv2.rectangle(emptyMask, pt0, pt3, 255, -1)
            if (gridWithNoPoint == 1 ):
                Tpoints = [pt0, pt2, pt3]
            else:
                Tpoints = [pt0, pt1, pt3]
            #cv2.circle(im,pt0,20,(0,255,0),-1)
            #cv2.circle(im,pt3,20,(0,255,0),-1)
            #straight forward
        return [cv2.convertScaleAbs(emptyMask), Tpoints]

    #input1: points, a list containts the points
    #input2: corner, a point indicates a corner
    def minDistanceCornerToPoint(self, points, corner):
        minDist = 9000000
        cloestPt = [];
        for pt in enumerate(points):
            pt = pt[1]
            dist = math.sqrt( (corner[0] - pt[0])**2 + (corner[1] - pt[1])**2 )       
            if dist < minDist: 
                cloestPt = pt
                minDist = dist

        return cloestPt

    #input1: orginal image
    #output1: mask, the mask of the window
    #output2: ratio, number of pixel * rato = millimeter on the image
    def mask(self, im):
        maskWindow, _, _, _, _, ratio, _ = self.getWindowMask(im);
        maskClearRuler = self.removeRulerInWindow(cv2.bitwise_and(im,im, mask=maskWindow))
        mask = cv2.bitwise_and(maskWindow, maskClearRuler)
        if ( self.noiseRemoval is not None):
            maskNoiseRemoval = self.noiseRemoval.mask(cv2.bitwise_and(im,im, mask=mask))
            mask = cv2.bitwise_and(maskNoiseRemoval, mask)
        return mask, ratio;

    #input1: orginal image
    #output1: mask, the mask of the window
    #output2: rulerMask, binary image (ruler)
    #output3: erodedMask, the eroded mask
    #output4: Tpoints, list of valid points
    #output5: circles, opencv circle list 
    #output6: unit, num of pixel * unit = millmeters
    #output7: pt, the point that is used to convert to unit
    def getWindowMask(self, im):
        rulerMask = self.extractRuler(im)
        circles, erodedMask = self.getAllDetectedCircles(rulerMask)  
        pt0, pt1, pt2, pt3, gridWithNoPoint = self.selectTheCornerCircles(rulerMask, circles)
        unit, pt = self.getRatio(circles, [pt0, pt1, pt2, pt3])
        windowMask, Tpoints = self.threePointsToWindow(np.zeros(rulerMask.shape),pt0,pt1,pt2,pt3,gridWithNoPoint)
        return [windowMask, rulerMask, erodedMask, Tpoints, circles, unit, pt];

    #input1: circles, opencv circle list
    #input2: points, list of valid points
    #output1: unit, num of pixel * unit = millmeters
    #output2: pt, the point that is used to convert to unit
    def getRatio(self, circles, points):
        index = -1
        for indx, circle in enumerate(circles):
            i = np.uint16(np.around(circle[indx,:]))
            for pt in points:
                if (np.array_equal(pt, np.array(i[0:2]))):
                    index = indx
                    unit = 10/circles[0,index][2]
                    return [unit, pt]
        return [0, (0,0)];

    #input1: windowIm, image that is applied window
    #output1: mask, binary image that is removed the ruler and applied window
    def removeRulerInWindow(self, windowIm):
        rulerMask = cv2.bitwise_not(cv2.inRange(windowIm, self.lower, self.upper)) #white == non ruler area
        windowMask = cv2.inRange(windowIm, np.array([1,1,1]), np.array([255,255,255])) #white == inside window area
        mask = cv2.bitwise_and(rulerMask, windowMask)
        
        #closing
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        #erode mask
        mask = cv2.erode(mask, self.kernel, iterations=self.iteration)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaArray = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)
        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        if (len(sorteddata) != 0):
            largestcontour = sorteddata[0][1]
            mask = np.zeros(mask.shape).astype(np.uint8)
            hull = cv2.convexHull(largestcontour)
            cv2.drawContours(mask, [hull], 0, 255, -1)
            #mask = cv2.erode(mask,self.edgeRemovalKernel,iterations = self.edgeRemovalIteration) no erode
            return mask
        else:
            return windowMask

        '''
        org = np.copy(mask)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaArray = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)
        #first sort the array by area
        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
        if (len(sorteddata) != 0):
            largestcontour = sorteddata[0][1]
            mask = np.zeros(mask.shape).astype(np.uint8)
            cv2.drawContours(mask, [largestcontour], 0, 255, -1)
            return mask
        else:
            return org

        '''