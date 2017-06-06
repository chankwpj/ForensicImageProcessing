import glob
import numpy as np
import cv2
import MyIO as IO
from sympy import Point
from BruiseWindow import *
from SkinDetection import *
from NoiseRemoval import *

imageFolderPath = './dataset2'
imagePath = glob.glob(imageFolderPath +'/*.jpg') 
imagePath = imagePath + glob.glob(imageFolderPath +'/*.png') 
imageFolderPath = './general-bruise'
imagePath = imagePath + glob.glob(imageFolderPath +'/*.jpg') 

workers = []
workers.append(SkinDetection())
workers.append(BruiseWindow())


for index, path in enumerate(imagePath):
    imgName = "Img" + str(index) 
    print "Working on Img " + imgName 
    im = cv2.imread(path)
    out = np.copy(im)
    for index, worker in enumerate(workers):
        mask = worker.mask(im);
        out = cv2.bitwise_and(out, out, mask=mask);

    cv2.imwrite(imgName + "out.jpg",out)