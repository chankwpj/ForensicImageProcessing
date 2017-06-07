from Worker import *
import cv2
import numpy as np

class PreProcessor(Worker):
    """description of class"""
    def __init__(self, workers):
        self.workers = workers

    def getWorkers(self):
        return self.workers;

    def setWorkers(self, workers):
        self.workers = workers

    def mask(self, im):
        res = np.ones(im.shape[0:2], dtype=np.uint8)
        for index, worker in enumerate(self.workers):
            mask = worker.mask(im);
            res = cv2.bitwise_and(res, mask);

        return cv2.inRange(res, 1, 255);
