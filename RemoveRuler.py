import cv2
import numpy as np

class RulerRemover:
    def __init__(self, lower=None, upper=None):
        if lower!=None: 
            self.lower = lower
        else:
            self.lower = np.array([180, 180, 180])

        if upper!=None:
            self.upper = upper
        else:
            self.upper = np.array([255, 255, 255])
        
    def RemoveNoise(self, im):
        mask = cv2.bitwise_not(cv2.inRange(im, self.lower, self.upper))
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

