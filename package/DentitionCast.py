import numpy as np
import cv2
import collections


#************************************#
#Start - Part 1#
#Input: orginal image and thresholding upper and lower bound
#Output: masks that remove ruler from image
#main method:
#RemoveRuler:
#Colour thresholding the ruler
#Find the largest areas indicates left and right parts of the ruler
#Draw the largest areas on the mask and return it.
#************************************#

def ColourThresholdingRuler(im, lower, upper): #return mask
    mask = cv2.inRange(im, lower, upper)
    #masks.append(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask, kernel, iterations=5)
    #fill the holes
    #fill white
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for h,cnt in enumerate(contours):
        cv2.drawContours(mask,[cnt],0,255,-1)
    #fill black
    mask = cv2.bitwise_not(mask);
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    for h,cnt in enumerate(contours):
        cv2.drawContours(mask,[cnt],0,255,-1)
    mask = np.copy(cv2.bitwise_not(mask)); #convert
    return mask

def FindTwoLargestContour(mask):
    #find the largest contours (left and right)
    contoursSort = {}
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for h,cnt in enumerate(contours):
        contoursSort[cv2.contourArea(cnt)] = cnt
    contoursSort = collections.OrderedDict(sorted(contoursSort.items()))
    #assume left and right are the biggest contours 
    largestContour1 = [contoursSort.get(contoursSort.keys()[len(contoursSort)-1])]
    largestContour2 = [contoursSort.get(contoursSort.keys()[len(contoursSort)-2])]

    #check if left and right are merged as a contour then seperate it
    area1 = largestContour1[0].shape[0]
    area2 = largestContour2[0].shape[0]
    w = mask.shape[1]
    if (area1/3 > area2): #area 1 include left and right
        temp = np.copy(largestContour1[0])
        largestContour1 = [np.array(temp[np.where(temp[:,:,0] <= w/2)])]
        largestContour2 = [np.array(temp[np.where(temp[:,:,0] >= w/2)])]

    return [largestContour1, largestContour2]

def DrawNewMask(im, largestContour1, largestContour2, hShift, vShift):
    
    mask = np.zeros([im.shape[0],im.shape[1]]).astype(np.uint8)
    h,w,d = im.shape
    pt1 = (w/2, 0) #mid point when y == 0
    l1 = largestContour1[0].reshape(-1,2) #reshape
    l2 = largestContour2[0].reshape(-1,2) #reshape
    hy1 = l1[np.where(l1[:,1] == np.amax(l1[:,1]))][-1] #get the far most point of the ruler
    hy2 = l2[np.where(l2[:,1] == np.amax(l2[:,1]))][-1] #get the far most point of the ruler
    hShift = - 20; #shift the mid point
    vShift = 90; #shift down another the mid point

    #draw polygon, and check if the points are left or right to draw the correct polygon
    if (hy1[0] > w/2):
        cv2.fillConvexPoly(mask, np.array([[w,0], pt1, (w/2 + hShift, vShift), [hy1[0], hy1[1]], [w, hy1[1]]], dtype=np.int32).reshape((-1,1,2)), 255)
    else:
        cv2.fillConvexPoly(mask, np.array([[0,0], pt1, (w/2 + hShift, vShift), [hy1[0], hy1[1]], [0, hy1[1]]], dtype=np.int32).reshape((-1,1,2)), 255)
        
    if (hy2[0] > w/2):
        cv2.fillConvexPoly(mask, np.array([[w,0], pt1, (w/2 + hShift, vShift), [hy2[0], hy2[1]], [w, hy2[1]]], dtype=np.int32).reshape((-1,1,2)), 255)
    else:
        cv2.fillConvexPoly(mask, np.array([[0,0], pt1, (w/2 + hShift, vShift), [hy2[0], hy2[1]], [0, hy2[1]]], dtype=np.int32).reshape((-1,1,2)), 255)
    
    return cv2.bitwise_not(mask)

def RemoveRuler(im, lower, upper, hShift, vShift):
    rulerMask = ColourThresholdingRuler(im, lower, upper)
    largestContour1, largestContour2 = FindTwoLargestContour(rulerMask)
    return DrawNewMask(im, largestContour1, largestContour2, hShift, vShift)

#************************************#
#End - Part 1#
#************************************#


#************************************#
#Start - Part 2#
#im - colour image that has been removed upper ruler
#dilationKsize - dilation kernel size
#dilationIteration - iteration times
#output a mask that remove background from image
#main method: 
#Intensity Thresholding
#ConvexHull for the intensity result
#************************************#

def IntensityThresholding(im, lower, upper):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(im, lower, upper)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    for h,cnt in enumerate(contours):
        cv2.drawContours(mask,[cnt],0,255,-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31)) #25
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def GenerateConvexHullForCast(mask, dilationKsize, dilationIteration):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    #print contours
    items = len(contours) + 1 # + 1 is correct
    pts = np.array([])
    if items > 2:
        for h,cnt in enumerate(contours):
            res = cnt.reshape(len(cnt),2)
            if len(pts) == 0:
                pts = np.copy(res)
            else:
                pts = np.concatenate((pts, res))
    
        hull = cv2.convexHull(pts)
        points = np.int32(np.around(hull))  
        cv2.drawContours(mask,[points.reshape((-1,1,2))],0,255,-1)
    else:
        for h,cnt in enumerate(contours):
            cv2.drawContours(mask,[cnt],0,255,-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilationKsize,dilationKsize)) #25
    mask = cv2.dilate(mask, kernel, iterations=dilationIteration)
    return mask

def LocalizeCast(im, lower, upper, dilationKsize, dilationIteration):
    mask = IntensityThresholding(im, lower, upper)
    mask = GenerateConvexHullForCast(mask, dilationKsize, dilationIteration)
    return mask

#************************************#
#End - Part 2#
#************************************#

def MSER(im):
    vis = np.copy(im)
    mser = cv2.MSER(5, 100000, 5000000, 0.25, 0.2, 100, 1.01, 0.0005, 5)
    regions = mser.detect(im, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (255, 0, 0), 10)
    return vis
