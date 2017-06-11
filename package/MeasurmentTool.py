
import cv2
import numpy as np

class MeasurmentTool():
    
    def measure(self, mask):
        try:
            coo = np.where(mask >= 1)
            cnt = np.transpose( [coo[1], coo[0]])
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int32(np.around(box)) 
            return [box, rect]
        except:
            return [np.array([]), ((0,0), (0,0), 0)] 