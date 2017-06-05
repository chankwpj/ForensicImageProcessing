import glob
import numpy as np
import cv2
import MyIO as IO
import RemoveRuler as rr
from sympy import Point
import BruiseWindow
from BruiseWindow import BruiseWindow

imageFolderPath = './dataset2'
imagePath = glob.glob(imageFolderPath +'/*.jpg') 
imagePath = imagePath + glob.glob(imageFolderPath +'/*.png') 
imageFolderPath = './general-bruise'
imagePath = imagePath + glob.glob(imageFolderPath +'/*.jpg') 

bruiseWindow = BruiseWindow()

for index, path in enumerate(imagePath):

    imgName = "Img" + str(index) 
    print "Working on Img " + imgName 
    im = cv2.imread(path)
    mask, _ , _ , _ = bruiseWindow.getWindowMask(im);
    windowIm = cv2.bitwise_and(im,im, mask=mask)
    mask = bruiseWindow.removeRulerInWindow(windowIm)
    cv2.imwrite(imgName + "Post.jpg",im)
    IO.ImageWrite(im, imgName, ".jpg", mask=mask)
