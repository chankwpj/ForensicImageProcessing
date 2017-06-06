import numpy as np
import cv2

class BruiseSegmentation:
    
    def kMeansGeneration(self, im):
        img = np.copy(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        Z = im.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS, bestLabels=None, centers=None)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        #res2 = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        cv2.imwrite("k" + str(index) +'kmeansHSV.jpg',res2)
        cv2.imwrite("k" + str(index) +'LabelHSV.jpg', np.array(label).reshape((img.shape[0:2])))

