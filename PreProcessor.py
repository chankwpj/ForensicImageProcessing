from Worker import *
import cv2
import numpy as np

class PreProcessor():

    def __init__(self, skinDetector, bruiseWindow):
        self.skinDetector = skinDetector
        self.bruiseWindow = bruiseWindow

    def mask(self, im):
        skinMask = self.skinDetector.mask(im)
        windowMask, unit = self.bruiseWindow.mask(im)
        return cv2.bitwise_and(skinMask, windowMask), unit;

    def process(self, im):
        skinMask = self.skinDetector.mask(im)
        windowMask, unit = self.bruiseWindow.mask(im)
        mask = cv2.bitwise_and(skinMask, windowMask)
        return cv2.bitwise_and(im, im, mask=mask), unit;
